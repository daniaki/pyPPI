#!/usr/bin/env python

"""
This module contains functions to construct a binary relevance classifier
using the OVR estimaor in sklearn
"""
import logging
from operator import itemgetter

import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from ..data import get_term_description


logger = logging.getLogger("pyppi")


def get_coefs(clf):
    """
    Return the feature weightings for each estimator. If estimator is a
    pipeline, then it assumes the last step is the estimator.

    :return: array-like, shape (n_classes_, n_features)
    """
    def feature_imp(estimator):
        if hasattr(estimator, 'steps'):
            estimator = estimator.steps[-1][-1]

        if hasattr(estimator, "coef_"):
            return estimator.coef_
        elif hasattr(estimator, "coefs_"):
            return estimator.coefs_
        elif hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        elif hasattr(estimator, "feature_log_prob_"):
            return estimator.feature_log_prob_[1]
        else:
            raise AttributeError(
                "Estimator {} doesn't support "
                "feature coefficients.".format(type(estimator)))

    if hasattr(clf, "best_estimator_"):
        return feature_imp(clf.best_estimator_)
    else:
        return feature_imp(clf)


def rename(term):
    """
    Re-format feature terms after they've been formated by the vectorizer.

    Parameters:
    ----------
    term : str
        Mutilated term in string format.
    Returns
    -------
    str
        The normalised term.
    """
    term = term.upper()
    if 'IPR' in term:
        return term
    elif 'PF' in term:
        return term
    else:
        term = "GO:" + term
        return term


def top_n_features(n, clf, go_dag, ipr_map, pfam_map,
                   absolute=False, vectorizer=None):
    """
    Return the top N features. If clf is a pipeline, then it assumes
    the first step is the vectoriser holding the feature names.

    :return: array like, shape (n_estimators, n).
        Each element in a list is a tuple (feature_idx, weight).
    """
    if isinstance(clf, KNeighborsClassifier):
        logger.warning("Top features not supported for KNeighborsClassifier.")
        return None
    if isinstance(clf, GaussianNB):
        logger.warning("Top features not supported for GaussianNB.")
        return None

    top_features = []
    coefs = get_coefs(clf)
    try:
        n_features = max(coefs.shape[0], coefs.shape[1])
    except IndexError:
        n_features = coefs.shape[0]
    coefs = coefs.reshape((n_features,))

    if absolute:
        coefs = abs(coefs)
    if hasattr(clf, 'steps') and vectorizer is None:
        vectorizer = clf.steps[0][-1]
    idx_coefs = sorted(
        enumerate(coefs), key=itemgetter(1), reverse=True
    )[:n]
    if vectorizer:
        idx = [idx for (idx, w) in idx_coefs]
        ws = [w for (idx, w) in idx_coefs]
        names = np.asarray(rename(vectorizer.get_feature_names()))[idx]
        descriptions = np.asarray(
            [
                get_term_description(
                    term=x, go_dag=go_dag,
                    ipr_map=ipr_map, pfam_map=pfam_map,
                )
                for x in names
            ]
        )
        return list(zip(names, descriptions, ws))
    else:
        return [(idx, idx, coef) for (idx, coef) in idx_coefs]
