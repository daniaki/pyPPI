#!/usr/bin/env python

"""
Top level module to quick instantiting various classifiers
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.base import clone


def supported_estimators():
    allowed = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'MultinomialNB': MultinomialNB,
        'GaussianNB': GaussianNB,
        'BernoulliNB': BernoulliNB
    }
    return allowed


def get_parameter_distribution_for_model(model):
    if model not in supported_estimators():
        raise ValueError("{} is not a supported model".format(model))

    params = {}
    if model == 'LogisticRegression':
        params['C'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1)) + \
            list(np.arange(1.0, 10.5, step=0.5))
        params['penalty'] = ['l1', 'l2']

    elif model == 'RandomForestClassifier':
        params["n_estimators"] = np.arange(32, 528, step=16)
        params["criterion"] = ["gini", "entropy"]
        params['max_features'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.05, step=0.05))
        params["min_samples_leaf"] = np.arange(2, 21, step=1)
        params["class_weight"] = ['balanced', 'balanced_subsample']
        params["bootstrap"] = [False, True]

    elif model == "KNeighborsClassifier":
        params["n_neighbors"] = np.arange(1, 50, step=1)
        params["weights"] = ["uniform", "distance"]
        params["algorithm"] = ['auto', 'ball_tree', 'kd_tree', 'brute']
        params["leaf_size"] = np.arange(2, 100, step=2)
        params['p'] = np.arange(1, 10, step=1)

    elif model == "MultinomialNB":
        params['alpha'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1)) + \
            list(np.arange(1.0, 10.5, step=0.5))

    elif model == "BernoulliNB":
        params['alpha'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1)) + \
            list(np.arange(1.0, 10.5, step=0.5))

    elif model == "GaussianNB":
        params = {}

    return params


def make_classifier(algorithm, class_weight='balanced', random_state=None, n_jobs=1):
    supported = supported_estimators()
    estimator = supported.get(algorithm, LogisticRegression)()
    if hasattr(estimator, 'n_jobs'):
        estimator.set_params(**{'n_jobs': n_jobs})
    if hasattr(estimator, 'class_weight'):
        estimator.set_params(**{'class_weight': class_weight})
    if hasattr(estimator, 'random_state'):
        estimator.set_params(**{'random_state': random_state})
    if hasattr(estimator, 'probability'):
        estimator.set_params(**{'probability': True})
    return estimator
