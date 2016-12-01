#!/usr/bin/python

"""
This sub-module contains scoring functions for multi-label indicator arrays
and a class for housing statistics which wraps around a pandas dataframe.
"""

import numpy as np
from sklearn.metrics import *


def multilabel_decorator(y, y_pred):
    """
    Decorator to turn any scoring function within sklearn into a multi-label
    function that can deal with multi-label indicator arrays.
    """
    def wrapper_func(score_func):
        scores = []
        n_classes = np.unique(y)
        for i in range(n_classes):
            scores.append(score_func(y[:, i], y_pred[:, i]))
        return scores
    return wrapper_func
