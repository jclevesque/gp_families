#############################################################
# Copyright (C) 2015 Audrey Durand, Julien-Charles Levesque
#
# Distributed under the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
##############################################################

import functools

import numpy as np
import pygp
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split

import gp_families.gp_single
import gp_families.gp_family


def split(data, labels):
    '''Simple util function generating train, validation and testing splits.'''
    rng = np.random.RandomState()

    remain_data, test_data, remain_labels, test_labels = train_test_split(
        data, labels, train_size=0.5, test_size=0.5, random_state=rng)

    train_data, val_data, train_labels, val_labels = train_test_split(
        remain_data, remain_labels, train_size=0.8, test_size=0.2, random_state=rng)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def main(learner_classes, optimizer):
    # Load data
    all_data = np.loadtxt('data/winequality-white.csv', delimiter=';')
    data, labels = all_data[:, :-1], all_data[:, -1]
    train, val, test = split(data, labels)

    # Number of iterations for the optimization
    T = 20

    # Bookkeeping structures
    val_scores = []
    test_scores = []
    hist = []

    for i in range(T):
        print(i)
        # Optimizer returns the following:
        #  grid_i: internal grid index
        #  learner_i: index of the learner to evaluate at this iteration
        #  learner_params: parameters to give to the learner
        grid_i, learner_i, learner_params = optimizer.next()

        # Train and evaluate a learner of the given class and params
        learner = learner_classes[learner_i](**learner_params)
        learner.fit(train[0], train[1])
        score = learner.score(val[0], val[1])

        # Add the new observed point to the optimizer's data
        optimizer.update(learner_i, grid_i, score)
        print("Chosen learner:%s, grid_i:%i, val_score:%f" %
            (learner_classes[learner_i], grid_i, score))
        # Keep track of selected classifiers
        hist.append([learner_i, learner_params])
        val_scores.append(score)
        test_scores.append(learner.score(test[0], test[1]))

    # Find the best model trained so far.
    best_i = np.argmax(val_scores)

    print("Done optimizing! Best model: %s (%s), val acc: %f, test acc: %f" %
         (hist[best_i], learner_classes[hist[best_i][0]],
         val_scores[best_i], test_scores[best_i]))

if __name__ == '__main__':
    # Define the search space
    # Learning algorithms which we will optimize
    LinSVR = functools.partial(SVR, kernel='linear')
    learner_classes = [Lasso, RandomForestRegressor, LinSVR, SVR,
        KNeighborsRegressor]

    # Definition of hyperparameters for each learning algorithm
    search_space = [
    # Lasso
    [{'name':'alpha', 'min':1e-4, 'max':1, 'type':'float', 'scale':'log'}],
    # RF
    [{'name':'n_estimators', 'min':1 , 'max':1000, 'type':'int'},
     {'name':'min_samples_split', 'min':1, 'max':7, 'type':'int'},
     {'name':'min_samples_leaf', 'min':2, 'max':14, 'type':'int'}],
    # LinSVR
    [{'name':'C', 'min':1e-3, 'max':1, 'type':'float', 'scale':'log'},
     {'name':'epsilon', 'min':1e-4, 'max':1e-1, 'type':'float', 'scale':'log'}],
    # SVR
    [{'name':'C', 'min':1e-3, 'max':1, 'type':'float', 'scale':'log'},
     {'name':'epsilon', 'min':1e-4, 'max':1e-1, 'type':'float', 'scale':'log'},
     {'name':'gamma', 'min':1e-2, 'max':1, 'type':'float', 'scale':'log'}],
    # KNN
    [{'name':'n_neighbors', 'min':1, 'max':15, 'type':'int'}] ]

    # Priors given to pygp to optimize the GP hyperparameters
    # There are also bound parameters for each prior (first two parameters),
    #  these ensure that the lbfgs procedure does not return invalid values.
    gp_priors = {"sf":  pygp.priors.LogNormal(1e-9, 1e2, 1, 1),
                 "ell": pygp.priors.LogNormal(1e-6, 2, 0., 1),
                 "sn":  pygp.priors.Horseshoe(1e-9, 1, 1)}

    # Create an optimizer
    optimizer = gp_family.GPFamily(search_space, grid_size=1000,
        gp_priors=gp_priors, optimist=True)

    main(learner_classes, optimizer)
