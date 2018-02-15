#!/usr/bin/python

"""
This sub-module contains scoring functions for multi-label indicator arrays
and a class for housing statistics which wraps around a pandas dataframe.
"""

import numpy as np

from sklearn.metrics import confusion_matrix, hamming_loss


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def fdr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tp + fp) == 0:
        return 0.0
    return fp / (tp + fp)


def positive_label_hammming_loss(y_true, y_score):
    """
    This is a relaxed version of the Hamming Loss function.
    It computes the hamming loss over the positive labels appearing in
    `y_true` only. This may be desirable when you are most interested in
    penalising false negatives, or where false positives may represent
    latent labels not seen during training and are not necessarily incorrect.
    """
    selectors = [np.where(row == 1)[0] for row in y_true]
    y_true = [
        row[selector]
        for row, selector in zip(y_true, selectors)
        if len(selector)
    ]
    y_score = [
        row[selector]
        for row, selector in zip(y_score, selectors)
        if len(selector)
    ]
    losses = [
        hamming_loss(row_true, row_score)
        for row_true, row_score in zip(y_true, y_score)
    ]
    return sum(losses) / len(selectors)


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
