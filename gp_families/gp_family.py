#############################################################
# Copyright (C) 2015 Audrey Durand, Julien-Charles Levesque
#
# Distributed under the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
##############################################################

import copy
import numpy as np
import pygp
from pygp.learning import optimization

import ei
import grid
import sobol_lib
import util


class GPFamily:
    '''GP Family class for Bayesian optimization of hyperparameters and model choice.

    Creates one GP per subspace defined in the search space. Each GP is tuned
     independently of others. For shared GP hyperparameters across GPs, see
     GPFamilyShared.
    '''
    def __init__(self, search_space, grid_size, gp_priors, optimist):
        '''GPFamily constructor

        Parameters:
        -----------
        search_space: list of subspaces representing conditional or independent
            hyperparameters. For now only one level of subspaces is supported.

        grid_size: total number of points to allow for the discretization
            of the search space. Budget is spread evenly across subspaces
            to the ratio of their dimensionalities.

        gp_priors: priors to apply for the optimization of each GP's
            hyperparameters.

        optimist: if True, the GP will fix the prior mean to the maximum
            possible value (e.g., 1 for 100% accuracy)
        '''
        self.__name__ = "gpfamily"

        self.n_members = len(search_space)
        dims = [len(ss) for ss in search_space]
        dims_total = np.sum(dims)
        self.dims = dims
        self.dims_total = dims_total

        if optimist:
            # optimisim in face of uncertainty
            # set mean (not optimized) of GPs
            self.init_mu = 1
            gp_priors["mu"] = None
        else:
            # set initial optimization mean of GPs
            self.init_mu = 0

        # build gridsss
        self.maps = []
        self.grids = []
        self.members = []
        for d, subspace in zip(dims, search_space):
            # Create grids and gridmaps for subspaces
            if d > 0:
                subspace_map = grid.GridMap(subspace)
                subspace_grid = np.transpose(sobol_lib.i4_sobol_generate(d,
                    int(grid_size * (d / dims_total)), 9001))
            else:
                # This fake parameter will be used to get a value for
                # the acquisition function in this subspace with no
                # hyperparameters.
                fake_params = [{'name': 'placeholder', 'min': 0, 'max': 0,
                    'size': 1, 'type': 'int'}]
                subspace_map = grid.GridMap(fake_params)
                subspace_grid = np.array([[0.]])

            member = pygp.BasicGP(sn=1, sf=1, ell=np.ones(d), mu=self.init_mu,
                                                            kernel="matern5")

            self.maps.append(subspace_map)
            self.grids.append(subspace_grid)
            self.members.append(member)

        self.gp_priors = gp_priors

        # Scores of trained classifiers, all spaces mixed
        self.scores = []

    def next(self):
        '''Return the next point to evaluate according to acquisition
         function. Returns a subspace index, the index in the given subspace
         and the corresponding learner parameters in a dict which is
         directly unpackable. '''
        for i, (mmap, grid, membah) in \
                enumerate(zip(self.maps, self.grids, self.members)):
            # don't do anything fancy if we haven't sampled each classifier
            # at least once or twice, depending
            if len(grid) == 1:
                # placeholder params, just send nothing
                if membah.ndata < 1:
                    return 0, i, {}
            else:
                if membah.ndata < 2:
                    grid_i = np.random.choice(len(grid))
                    return grid_i, i, self.raw_to_learner_params(i, grid_i)

        # get everybody's posteriors
        posteriors = [m.posterior(g) for m, g in zip(self.members, self.grids)]
        mus = np.concatenate([p[0] for p in posteriors])
        s2s = np.concatenate([p[1] for p in posteriors])

        # pick the parameters maximizing the acquisition function
        acq = ei.expected_improvement(mus, s2s, self.scores)
        # handle cases where all acq are the same
        global_i = util.eq_rand_idx(acq, np.max(acq))
        member_i, grid_i = self.which_member(global_i)

        learner_params = self.raw_to_learner_params(member_i, grid_i)
        return grid_i, member_i, learner_params

    def which_member(self, i):
        '''Which family member does raw index `i` belong to'''
        count = 0
        for member_i in range(self.n_members):
            grid_size = len(self.grids[member_i])
            if i < (count + grid_size):
                break
            count += grid_size
        grid_i = i - count
        return member_i, grid_i

    def raw_to_learner_params(self, member_i, grid_i):
        '''Return learning ID and its params from raw points (in search space
        scaled to 0-1).
        '''
        raw = self.grids[member_i][grid_i]
        params = self.maps[member_i].get_params(raw)
        learner_params = {p["name"]: p["val"] for p in params
                          if p['name'] is not 'placeholder'}
        return learner_params

    def update(self, member_i, grid_i, score):
        ''' Update the model with the `score` obtained for `member_i`
         at index `grid_i` in the member's subspace.'''
        self.scores.append(score)

        self.members[member_i].add_data(self.grids[member_i][grid_i], score)

        # Optimize the model
        optimization.optimize_random_start(self.members[member_i], self.gp_priors)


class GPFamilyShared(GPFamily):
    '''GP Family Shared class for Bayesian optimization of hyperparameters and model choice.

    Creates one GP per subspace defined in the search space. All GPs are tuned
        jointly to select GP amplitude, noise and mean (in the non-optimistic
        case). Lengthscales are still independent.
    '''
    def __init__(self, search_space, grid_size, gp_priors, optimist):
        '''GPFamilyShared constructor

        Parameters:
        -----------
        search_space: list of subspaces representing conditional or independent
            hyperparameters. For now only one level of subspaces is supported.

        grid_size: total number of points to allow for the discretization
            of the search space. Budget is spread evenly across subspaces
            to the ratio of their dimensionalities.

        gp_priors: priors to apply for the optimization of each GP's
            hyperparameters.

        optimist: if True, the GP will fix the prior mean to the maximum
            possible value (e.g., 1 for 100% accuracy)
        '''
        super().__init__(search_space, grid_size, gp_priors, optimist)

        self.__name__ = "gpfamilyshared"

        # interface pyGP
        self.nhyper = 3 + self.dims_total
        self.hypers = np.log(np.ones(3 + self.dims_total))
        self.hypers[-1] = self.init_mu

        self.prior_structs, self.active_idx, _ =\
                optimization.get_priors_active_bounds(self, self.gp_priors)

        # flag to know when we have enough data (at least 1 per member)
        # to optimize
        self.enough_data = False

    def _params(self):
        # HARD CODED KERNEL STRUCTURE
        params = [('sn', 1, True)]
        params += [('sf', 1, True), ('ell', self.dims_total, True)]
        params += [('mu', 1, False)]
        return params

    def update(self, member_i, grid_i, score):
        ''' Update the model with the `score` obtained for `member_i`
         at index `grid_i` in the member's subspace'''
        self.scores.append(score)
        self.members[member_i].add_data(self.grids[member_i][grid_i], score)

        if not self.enough_data:
            enough_data = True
            for gp in self.members:
                if gp.ndata < 1:
                    enough_data = False
                    break
            self.enough_data = enough_data

        if self.enough_data:
            # optimize the model
            optimization.optimize_random_start_custom(self,
                                        self.negloglikelihood, self.gp_priors)

    def get_hyper(self):
        return self.hypers

    def set_hyper(self, hypers):
        self.hypers = hypers

        count_dims = 0
        for gp, dims in zip(self.members, self.dims):
            hypers_i = np.zeros(3 + dims)
            hypers_i[0:2] = hypers[0:2] # sn + sf
            hypers_i[2:2+dims] = hypers[2+count_dims:2+count_dims+dims] # ells
            hypers_i[-1] = hypers[-1] # mu
            gp.set_hyper(hypers_i)

            count_dims += dims

    def negloglikelihood(self, x):
        '''
        Returns the negative loglikelihood of the current configuration.
        '''
        hyper = self.hypers.copy()
        hyper[self.active_idx] = x
        # propagate hypers to members
        self.set_hyper(hyper)

        lZ = 0
        # cumulate likelihood per members
        for gp in self.members:
            lZ += gp.loglikelihood(False)
        # add likelihood based on priors
        for block, log, prior in self.prior_structs:
            if log:
                ltheta = prior.logprob(np.exp(hyper[block]), False)
            else:
                ltheta = prior.logprob(hyper[block], False)
            lZ += ltheta

        if np.isnan(lZ):
            raise Exception("NaN likelihood of prior")
        if np.isinf(lZ):
            print("Infinite likelihood of prior")

        return -lZ
