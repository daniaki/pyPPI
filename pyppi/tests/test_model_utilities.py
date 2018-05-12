import numpy as np
from numpy.random import RandomState

from unittest import TestCase

from ..models.classifier_chain import KRandomClassifierChains

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from ..base.constants import MAX_SEED
from ..models.utilities import (
    get_parameter_distribution_for_model,
    make_classifier,
    make_gridsearch_clf
)


class TestGetParams(TestCase):

    def test_valuerror_unsupported_model(self):
        with self.assertRaises(ValueError):
            get_parameter_distribution_for_model("not a model")

    def test_step_used_as_suffix_to_each_key(self):
        params = get_parameter_distribution_for_model(
            "LogisticRegression", 'estimator'
        )
        for key in params:
            self.assertIn('estimator__', key)


class TestMakeClassifier(TestCase):

    def test_propagates_keyword_args(self):
        clf = make_classifier(
            'RandomForestClassifier',
            class_weight='balanced', random_state=0, n_jobs=10
        )
        self.assertIsInstance(clf, RandomForestClassifier)
        self.assertEqual(clf.n_jobs, 10)
        self.assertEqual(clf.class_weight, 'balanced')
        self.assertEqual(clf.random_state, 0)

    def test_unrecognised_classifier_raises_keyerror(self):
        with self.assertRaises(ValueError):
            clf = make_classifier(None)


class TestMakeGridSearchClf(TestCase):

    def test_each_stage_gets_own_random_number(self):
        clf = make_gridsearch_clf(
            'LogisticRegression', rcv_splits=5, rcv_iter=50, scoring='accuracy',
            n_jobs_gs=2, n_jobs_model=4, random_state=0,
            binary=False, search_vectorizer=False, multilabel=False
        )
        params = get_parameter_distribution_for_model(
            "LogisticRegression", step='estimator'
        )

        rng = RandomState(0)
        max_int = MAX_SEED
        cv_random_state = rng.randint(max_int)
        model_random_state = rng.randint(max_int)
        rcv_random_state = rng.randint(max_int)
        # chain_random_state = rng.randint(max_int)

        rgs_est = clf  # grid search estimator
        self.assertIsInstance(clf, RandomizedSearchCV)
        self.assertEqual(rgs_est.param_distributions, params)
        self.assertEqual(rgs_est.random_state, rcv_random_state)
        self.assertEqual(rgs_est.n_iter, 50)
        self.assertEqual(rgs_est.error_score, 0.0)
        self.assertEqual(rgs_est.n_jobs, 2)
        self.assertEqual(rgs_est.refit, True)
        self.assertEqual(rgs_est.scoring, 'accuracy')

        pipe = clf.estimator
        self.assertIsInstance(pipe, Pipeline)
        self.assertEqual(pipe.steps[0][0], 'vectorizer')
        self.assertEqual(pipe.steps[1][0], 'estimator')

        vec = pipe.steps[0][1]
        self.assertIsInstance(vec, CountVectorizer)
        self.assertEqual(vec.binary, False)
        self.assertEqual(vec.stop_words, None)
        self.assertEqual(vec.lowercase, False)

        model = pipe.steps[1][1]
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.random_state, model_random_state)
        self.assertEqual(model.n_jobs, 4)

        cv = rgs_est.cv
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 5)
        self.assertEqual(cv.random_state, cv_random_state)

    def test_make_pipeline_false_estimator_is_not_pipeline(self):
        clf = make_gridsearch_clf(
            'LogisticRegression', rcv_splits=5, rcv_iter=50, scoring='accuracy',
            n_jobs_gs=2, n_jobs_model=4, random_state=0,
            binary=False, search_vectorizer=False, make_pipeline=False
        )
        params = get_parameter_distribution_for_model("LogisticRegression")

        rgs_est = clf  # grid search estimator
        self.assertIsInstance(clf, RandomizedSearchCV)
        self.assertEqual(rgs_est.param_distributions, params)
        self.assertIsInstance(clf.estimator, LogisticRegression)
