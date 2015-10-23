#############################################################
# Copyright (C) 2015 Audrey Durand, Julien-Charles Levesque
#
# Distributed under the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
##############################################################

import numpy as np


def eq_rand_idx(X, target):
    equal = np.where(X == target)[0]
    idx = np.random.choice(equal)
    return idx
