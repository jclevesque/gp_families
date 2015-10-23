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


class GPSingle:
    '''
    GPSingle class for the optimization of a conditional hyperparameter space,
     which models the whole joint space with a single GP, effectively ignoring
     the information contained in the conditional space.

    Appends an extra learner_choice parameter to the search_space given at
     initialization time, and uses this parameter to optimize the learner
     choice.
    '''
    def __init__(self, search_space, grid_size, gp_priors, optimist):
        '''GPSingle constructor

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
        self.__name__ = "gpsingle"
        self.flat_search_space = [{"name": "learner_choice",
                                   "min": 0,
                                   "max": len(search_space)-1,
                                   "type": "int"}]
        self.learner_param_indexes = []
        self.search_space = search_space

        if optimist:
            mu = 1
            gp_priors["mu"] = None
        else:
            mu = 0

        n_params = 1
        for i in range(len(search_space)):
            ss_i = search_space[i] # list of dicts

            self.flat_search_space.extend(ss_i)

            cur_indexes = []
            for j in range(len(ss_i)):
                cur_indexes.append(n_params)
                n_params += 1
            self.learner_param_indexes.append(cur_indexes)

        # build grid
        dims = len(self.flat_search_space)
        self.map = grid.GridMap(self.flat_search_space)
        self.space = np.transpose(sobol_lib.i4_sobol_generate(dims,
            grid_size, 9001))

        self.gp = pygp.BasicGP(sn=1, sf=1, ell=np.ones(dims), mu=mu,
                                                            kernel="matern5")
        self.gp_priors = gp_priors

    def next(self):
        '''Return the next point to evaluate according to acquisition
         function. Returns a subspace index, the index in the given subspace
         and the corresponding learner parameters in a dict which is
         directly unpackable. '''
        if self.gp.ndata < 2:
            # not enough points pick random parameters
            grid_i = np.random.choice(len(self.space))
        else:
            # optimize the model
            optimization.optimize_random_start(self.gp, self.gp_priors)
            # pick the parameters maximizing the acquisition function
            mu, var = self.gp.posterior(self.space)
            acq = ei.expected_improvement(mu, var, self.gp.data[1])
            grid_i = util.eq_rand_idx(acq, np.max(acq))

        learner_i, learner_params = self.raw_to_learner_params(
            self.space[grid_i])
        return grid_i, learner_i, learner_params

    def raw_to_learner_params(self, raw):
        '''Return learning ID and its params from raw points (in search space
        scaled to 0-1).
        '''
        params = self.map.get_params(raw)
        learner_i = params[0]["val"]
        l_param_i = self.learner_param_indexes[learner_i]

        learner_params = {params[i]["name"]:params[i]["val"] for i in l_param_i}
        return learner_i, learner_params

    def update(self, learner_i, grid_i, score):
        ''' Update the model with the `score` obtained for `learner_i`
         at index `grid_i` in the member's subspace.'''
        self.gp.add_data(self.space[grid_i], score)
