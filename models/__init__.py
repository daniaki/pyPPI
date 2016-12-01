#!/usr/bin/env python

"""
Top level module to quick instantiting various classifiers
"""

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.svm import OneClassSVM, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


def supported_estimators():
    return {
        'SVC': SVC,
        'ElasticNet': ElasticNet,
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'OneClassSVM': OneClassSVM,
        'AdaBoostClassifier': AdaBoostClassifier,
        'IsolationForest': IsolationForest,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'BayesianGaussianMixture': BayesianGaussianMixture,
        'GaussianMixture': GaussianMixture,
        'MLPClassifier': MLPClassifier,
        'MultinomialNB': MultinomialNB,
        'GaussianNB': GaussianNB,
        'BernoulliNB': BernoulliNB
    }


def make_classifier(algorithm, class_weight='balanced', random_state=None):
    supported = supported_estimators()
    estimator = supported.get(algorithm, default=LogisticRegression)
    if hasattr(estimator, 'n_jobs'):
        estimator.set_params(**{'n_jobs': 1})
    if hasattr(estimator, 'class_weight'):
        estimator.set_params(**{'class_weight': class_weight})
    if hasattr(estimator, 'random_state'):
        estimator.set_params(**{'random_state': random_state})

    return estimator
