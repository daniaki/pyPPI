#!/usr/bin/python

"""
This sub-module contains scoring functions for multi-label indicator arrays
and a class for housing statistics which wraps around a pandas dataframe.
"""

import numpy as np

from sklearn.metrics import confusion_matrix


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def fdr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fp) == 0:
        return 0.0
    return fp / (tp + fp)


class MultilabelScorer(object):
    """
    Simple helper class to wrap over a binary metric in sci-kit to
    enable support for simple multi-label scoring.
    """

    def __init__(self, sklearn_binary_metric):
        self.scorer = sklearn_binary_metric

    def __call__(self, y, y_pred, **kwargs):
        scores = []
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        n_classes = y.shape[1]
        for i in range(n_classes):
            score = self.scorer(y[:, i], y_pred[:, i], **kwargs)
            scores.append(score)
        return np.asarray(scores)
