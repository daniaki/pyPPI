#!/usr/bin/env python

"""
This module contains functions to construct a binary relevance classifier
using the OVR estimaor in sklearn
"""

from operator import itemgetter

import numpy as np

from ..data import get_term_description


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
        else:
            raise AttributeError(
                "Estimator {} doesn't support "
                "feature coefficients.".format(type(estimator)))

    if hasattr(clf, "best_estimator_"):
        return feature_imp(clf.best_estimator_)
    else:
        return feature_imp(clf)


def top_n_features(n, clf, go_dag, ipr_map, pfam_map,
                   absolute=False, vectorizer=None):
    """
    Return the top N features. If clf is a pipeline, then it assumes
    the first step is the vectoriser holding the feature names.

    :return: array like, shape (n_estimators, n).
        Each element in a list is a tuple (feature_idx, weight).
    """
    top_features = []
    coefs = get_coefs(clf)[0]

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
        names = np.asarray(vectorizer.get_feature_names())[idx]
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
