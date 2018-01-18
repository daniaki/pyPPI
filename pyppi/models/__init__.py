#!/usr/bin/env python

"""
Top level module to quick instantiting various classifiers
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.svm import OneClassSVM, SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.base import clone


def supported_estimators():
    allowed = {
        'SVC': SVC,
        'ElasticNet': ElasticNet,
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'OneClassSVM': OneClassSVM,
        'AdaBoostClassifier': AdaBoostClassifier,
        'IsolationForest': IsolationForest,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'BayesianGaussianMixture': BayesianGaussianMixture,
        'GaussianMixture': GaussianMixture,
        'MultinomialNB': MultinomialNB,
        'GaussianNB': GaussianNB,
        'BernoulliNB': BernoulliNB
    }
    return allowed


def get_parameter_distribution_for_model(model):
    if model not in supported_estimators():
        raise ValueError("{} is not a supported model".format(model))

    params = {}
    if model == 'SVC':
        params['C'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1)) + \
            list(np.arange(1.0, 10.5, step=0.5))
        params['shrinking'] = [False, True]
        params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        params['degree'] = np.arange(1, 10, step=1)
        params['gamma'] = list(np.arange(0.01, 1, step=0.01)) + ['auto']
        params['coef0'] = np.arange(0.01, 1, step=0.01)

    elif model == 'LogisticRegression':
        params['C'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1)) + \
            list(np.arange(1.0, 10.5, step=0.5))
        params['penalty'] = ['l1', 'l2']

    elif model == 'ElasticNet':
        params['alpha'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1)) + \
            list(np.arange(1.0, 10.5, step=0.5))
        params['l1_ratio'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.0, step=0.1))
        params['normalize'] = [False, True]

    elif model == 'RandomForestClassifier':
        params["n_estimators"] = np.arange(32, 528, step=16)
        params["criterion"] = ["gini", "entropy"]
        params['max_features'] = list(np.arange(0.0001, 0.001, step=0.0001)) + \
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01)) + \
            list(np.arange(0.1, 1.05, step=0.05))
        params["min_samples_leaf"] = np.arange(2, 21, step=1)
        params["class_weight"] = ['balanced', 'balanced_subsample']

    elif model == 'OneClassSVM':
        params['nu'] = np.arange(0.01, 1, step=0.01)
        params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        params['shrinking'] = [False, True]
        params['degree'] = np.arange(1, 10, step=1)
        params['gamma'] = list(np.arange(0, 1, step=0.01)) + ['auto']
        params['coef0'] = np.arange(0, 1, step=0.01)

    elif model == 'AdaBoostClassifier':
        params["base_estimator"] = [LogisticRegression]
        base_params = get_parameter_distribution_form_model(
            "LogisticRegression"
        )
        for key, value in base_params.items():
            params["base_estimator__" + key] = value
        params["n_estimators"] = np.arange(32, 272, step=16)
        params["learning_rate"] = np.arange(0.01, 3, step=0.01)
        params["algorithm"] = ['SAMME', 'SAMME.R']

    elif model == 'IsolationForest':
        params["max_features"] = np.arange(0.01, 1.01, step=0.01)
        params["max_samples"] = np.arange(0.01, 1.01, step=0.01)
        params["n_estimators"] = np.arange(32, 257, step=1)
        params["bootstrap"] = [True, False]
        params["contamination"] = np.arange(0.01, 0.5, step=0.01)

    elif model == 'GradientBoostingClassifier':
        params["loss"] = ['deviance', 'exponential']
        params["max_features"] = np.arange(0.01, 1.01, step=0.01)
        params["max_depth"] = np.arange(1, 15, step=1)
        params["n_estimators"] = np.arange(32, 257, step=1)
        params["subsample"] = np.arange(0.01, 1.01, step=0.01)
        params["learning_rate"] = np.arange(0.01, 3, step=0.01)
        params["min_samples_split"] = np.arange(2, 11, step=1)
        params["min_samples_leaf"] = np.arange(2, 11, step=1)
        params["min_weight_fraction_leaf"] = np.arange(0, 0.5, step=0.01)

    elif model == "KNeighborsClassifier":
        params["n_neighbors"] = np.arange(1, 50, step=1)
        params["weights"] = ["uniform", "distance"]
        params["algorithm"] = ['auto', 'ball_tree', 'kd_tree', 'brute']
        params["leaf_size"] = np.arange(5, 100, step=1)
        params['p'] = np.arange(1, 10, step=1)

    elif model == "BayesianGaussianMixture":
        params["covariance_type"] = ['full', 'tied', 'diag', 'spherical']
        params["n_components"] = np.arange(1, 4, step=1)
        params["max_iter"] = [100, 150, 200, 250, 300]
        params["n_init"] = np.arange(1, 4, step=1)
        params["init_params"] = ['kmeans']
        params["weight_concentration_prior"] = np.arange(0.01, 1, step=0.01)
        params["mean_precision_prior"] = np.arange(0.01, 2, step=0.01)

    elif model == "GaussianMixture":
        params["covariance_type"] = ['full', 'tied', 'diag', 'spherical']
        params["n_components"] = np.arange(1, 4, step=1)
        params["max_iter"] = [100, 150, 200, 250, 300]
        params["n_init"] = np.arange(1, 4, step=1)
        params["init_params"] = ['kmeans']

    elif model == "MultinomialNB":
        params["alpha"] = np.arange(0, 5, step=0.01)

    elif model == "BernoulliNB":
        params["alpha"] = np.arange(0, 5, step=0.01)

    elif model == "GaussianNB":
        params = {}

    return params


def make_classifier(algorithm, class_weight='balanced', random_state=None, n_jobs=1):
    supported = supported_estimators()
    estimator = supported.get(algorithm, LogisticRegression)()
    if isinstance(estimator, LogisticRegression):
        estimator.set_params(**{'solver': "saga"})
    if hasattr(estimator, 'n_jobs'):
        estimator.set_params(**{'n_jobs': n_jobs})
    if hasattr(estimator, 'max_iter'):
        estimator.set_params(**{'max_iter': 2500})
    if hasattr(estimator, 'class_weight'):
        estimator.set_params(**{'class_weight': class_weight})
    if hasattr(estimator, 'bootstrap'):
        estimator.set_params(**{'bootstrap': True})
    if hasattr(estimator, 'random_state'):
        estimator.set_params(**{'random_state': random_state})
    if hasattr(estimator, 'probability'):
        estimator.set_params(**{'probability': True})
    return estimator
