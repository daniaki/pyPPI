#!/usr/bin/env python

"""
This module contains functions to construct classifiers, get their param
distributions for a grid search, get top features and other
smaller utility functions.
"""

__all__ = [
    "supported_estimators",
    "publication_ensemble",
    "get_parameter_distribution_for_model",
    "make_classifier",
    "make_gridsearch_clf"
]


import logging
import numpy as np
from numpy.random import RandomState
from operator import itemgetter

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from ..base.utilities import get_term_description, rename
from ..base.constants import MAX_SEED

from .classifier_chain import KRandomClassifierChains
from .binary_relevance import MixedBinaryRelevanceClassifier

logger = logging.getLogger("pyppi")


def publication_ensemble():
    """Returns a `dict` mapping labels to their the classifier used 
    in the publication experiments."""
    label_model_map = {
        'Acetylation': 'RandomForestClassifier',
        'Activation': 'RandomForestClassifier',
        'Binding/association': 'RandomForestClassifier',
        'Carboxylation': 'LogisticRegression',
        'Deacetylation': 'RandomForestClassifier',
        'Dephosphorylation': 'RandomForestClassifier',
        'Dissociation': 'RandomForestClassifier',
        'Glycosylation': 'LogisticRegression',
        'Inhibition': 'RandomForestClassifier',
        'Methylation': 'LogisticRegression',
        'Myristoylation': 'LogisticRegression',
        'Phosphorylation': 'RandomForestClassifier',
        'Prenylation': 'LogisticRegression',
        'Proteolytic-cleavage': 'LogisticRegression',
        'State-change': 'LogisticRegression',
        'Sulfation': 'RandomForestClassifier',
        'Sumoylation': 'RandomForestClassifier',
        'Ubiquitination': 'LogisticRegression'
    }
    return label_model_map


def supported_estimators():
    """Return a `dict` of supported estimators."""
    allowed = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'MultinomialNB': MultinomialNB,
        'GaussianNB': GaussianNB,
        'BernoulliNB': BernoulliNB
    }
    return allowed


def get_parameter_distribution_for_model(model, step=None):
    """Returns the parameter distribution for a given `Scikit-learn` estimator

    Parameters
    ----------
    model: str
        String class name of the `SciKit-Learn` model

    step: str, optional, default: None
        If the model is placed inside a parent classifier such as a `Pipeline`
        then supply the step suffix to prepend to the parameter keys.

    Returns
    -------
    `dict`
        Dictionary of parameters for the model that can be used in a
        grid search estimator.

    See Also
    --------
    link to grid searches
    """
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

    elif model == 'DecisionTreeClassifier':
        params["criterion"] = ["gini", "entropy"]
        params['max_features'] = ['auto', 'log2'] +\
            list(np.arange(0, 1.0, step=0.02))
        params["min_samples_leaf"] = list(np.arange(2, 21, step=1))

    elif model == 'RandomForestClassifier':
        params["n_estimators"] = list(np.arange(10, 250, step=10))
        params["criterion"] = ["gini", "entropy"]
        params['max_features'] = ['auto', 'log2'] +\
            list(np.arange(0.001, 0.01, step=0.001)) + \
            list(np.arange(0.01, 0.1, step=0.01))
        params["min_samples_leaf"] = list(np.arange(2, 21, step=1))
        params["class_weight"] = ['balanced', 'balanced_subsample']
        params["bootstrap"] = [False, True]

    elif model == "KNeighborsClassifier":
        params["n_neighbors"] = list(np.arange(1, 50, step=1))
        params["weights"] = ["uniform", "distance"]
        params["algorithm"] = ['auto', 'ball_tree', 'kd_tree', 'brute']
        params["leaf_size"] = list(np.arange(2, 100, step=2))
        params['p'] = list(np.arange(1, 10, step=1))

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

    if step:
        keys = list(params.keys())
        for key in keys:
            params['{}__{}'.format(step, key)] = params[key]
            params.pop(key)

    return params


def make_classifier(algorithm, class_weight='balanced', random_state=None,
                    n_jobs=1):
    """Wrapper function for building a default classifier with the correct
    parameters.

    Parameters:
    ----------
    algorithm: str
        String class name of the `SciKit-Learn` model

    class_weight : str, optional, default: 'balanced'
        Sets the `class_weight` parameter if supported by the classifier.

    random_state : int or :class:`RandomState` or None, optional, default: None
        Sets the `random_state` parameter if supported by the classifier.

    n_jobs : int, optional, default: 1
        Sets the `n_jobs` parameter if supported by the classifier.

    Returns
    -------
    `estimator`
        Classifier with the supplied parameters set if applicable.   
    """
    supported = supported_estimators()
    if algorithm not in supported:
        raise ValueError(
            "'{}' is not a support classifier. Choose from: {}".format(
                algorithm, ', '.join(list(supported.keys()))
            )
        )
    estimator = supported[algorithm]()
    if hasattr(estimator, 'n_jobs'):
        estimator.set_params(**{'n_jobs': n_jobs})
    if hasattr(estimator, 'class_weight'):
        estimator.set_params(**{'class_weight': class_weight})
    if hasattr(estimator, 'random_state'):
        estimator.set_params(**{'random_state': random_state})
    if hasattr(estimator, 'probability'):
        estimator.set_params(**{'probability': True})
    return estimator


def make_gridsearch_clf(model, rcv_splits=3, rcv_iter=30, scoring='f1',
                        binary=True, n_jobs_model=1, random_state=None,
                        search_vectorizer=True, n_jobs_gs=1):
    """Wrapper function to automate the mundane setup of a `Pipeline` classifier
    within a `RandomGridSearchCV` estimator. See the links below for more
    details on the parameters.

    Parameters:
    ----------
    model: str
        String class name of the `SciKit-Learn` model which will be the
        `estimator` within the `Pipeline`.

    rcv_splits : int, optional, default: 3
        The number of splits to use during hyper-parameter cross-validation.

    rcv_iter : int, optional, default: 30
        The number of grid search iterations to perform.

    scoring : str, optional default: f1
        Scoring method used during hyperparameter search.

    binary : bool, optional, default: True
        If True sets the `binary` attribute of the `CountVectorizer` to True.

    n_jobs_model : int, optional, default: 1
        Sets the `n_jobs` parameter of the Pipeline's estimator step.

    random_state : int or None, optional, default: None
        This is a seed used to generate random_states for all estimator
        objects such as the base model and the grid search.

    search_vectorizer : bool, optional, default: False
        If True, adds the `binary` attribute of the `CountVectorizer` to the 
        grid search parameter distribution dictionary.

    n_jobs_gs : int, optional, default: 1
        Sets the `n_jobs` parameter of the `RandomizedGridSearch` classifier.

    Returns
    -------
    `estimator`
        A `RandomisedGridSearchCV` or :class:`KRandomClassifierChains` 
        classifier.

    See Also
    --------
    `Classifier Chain <http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain>`_
    `Randomized Grid Search <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_
    `Pipeline <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_
    """

    max_int = MAX_SEED
    rng = RandomState(random_state)

    model_random_state = rng.randint(max_int)
    cv_random_state = rng.randint(max_int)
    rcv_random_state = rng.randint(max_int)
    chain_random_state = rng.randint(max_int)

    base_estimator = make_classifier(
        model, random_state=model_random_state, n_jobs=n_jobs_model)
    params = get_parameter_distribution_for_model(model, step="estimator")

    vectorizer = CountVectorizer(lowercase=False, binary=binary)
    pipeline = Pipeline(
        steps=[('vectorizer', vectorizer), ('estimator', base_estimator)])
    if search_vectorizer:
        params['vectorizer__binary'] = [False, True]

    clf = RandomizedSearchCV(
        estimator=pipeline,
        cv=StratifiedKFold(
            n_splits=rcv_splits,
            shuffle=True,
            random_state=cv_random_state
        ),
        n_iter=rcv_iter,
        n_jobs=n_jobs_gs,
        refit=True,
        random_state=rcv_random_state,
        scoring=scoring,
        error_score=0.0,
        param_distributions=params
    )

    return clf
