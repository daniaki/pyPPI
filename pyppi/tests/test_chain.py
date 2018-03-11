import numpy as np
from numpy.random import RandomState
from unittest import TestCase

from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import hamming_loss, label_ranking_loss
from sklearn.metrics import f1_score, log_loss


from ..models.classifier_chain import KRandomClassifierChains
from ..base.constants import MAX_SEED

MAX_INT = MAX_SEED


class TestKRandomClassifierChains(TestCase):

    def setUp(self):
        self.X, self.y = make_multilabel_classification(
            100, 20, n_labels=5, random_state=0
        )
        self.base = LogisticRegression()
        self.rng = RandomState(seed=0)

    def test_can_get_params_returns_initial_rng_seed_when_int_passed(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        params = clf.get_params()
        self.assertEqual(params['random_state'], 0)

    def test_can_get_params_returns_initial_rng_seed_when_None_passed(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=None)
        params = clf.get_params()
        self.assertEqual(params['random_state'], None)

    def test_can_get_params_returns_initial_rng_seed_when_RS_passed(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=self.rng)
        params = clf.get_params()
        self.assertEqual(params['random_state'], self.rng)

    def test_rng_is_a_RS_object_when_None_is_not_passed(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=self.rng)
        self.assertIsInstance(clf.rng_, RandomState)

        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        self.assertIsInstance(clf.rng_, RandomState)

        clf = KRandomClassifierChains(self.base, k=3, random_state=None)
        self.assertIsNone(clf.rng_, None)

    def test_get_params_base_estimator_is_a_clone_with_same_params(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        params = clf.get_params()
        self.assertIs(clf.base_estimator, params["base_estimator"])
        self.assertEqual(
            clf.base_estimator.get_params(),
            params["base_estimator"].get_params()
        )

    def test_estimators_each_get_different_random_seed_when_not_None(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        seeds = set(c.random_state for c in clf.estimators_)
        self.assertEqual(len(seeds), len(clf.estimators_))

        # rng = RandomState(0)
        # for e in clf.estimators_:
        #     state = self.rng.randint(MAX_INT)
        #     self.assertEqual(e.base_estimator.random_state, state)
        #     self.assertEqual(e.random_state, state)

        clf = KRandomClassifierChains(self.base, k=3, random_state=None)
        seeds = set(c.random_state for c in clf.estimators_)
        self.assertEqual(seeds, set([None]))

    def test_estimators_has_length_k(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        self.assertEqual(len(clf.estimators_), clf.k)

    def test_estimators_are_all_clones(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        count = 0
        for e1 in clf.estimators_:
            for e2 in clf.estimators_:
                if e1 is not e2:
                    count += 1
        self.assertEqual(count, clf.k * 2)

    def test_can_set_params_of_each_estimator(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=0)
        params = {
            "base_estimator__C": 50,
            "base_estimator__random_state": None
        }
        clf.set_params(**params)
        for e in clf.estimators_:
            self.assertEqual(e.base_estimator.C, 50)
            self.assertIsNone(e.base_estimator.random_state)

    def test_can_set_params_of_each_pipeline_estimator(self):
        params = {
            "base_estimator__vectorizer__binary": True,
            "base_estimator__estimator__C": 50,
            "base_estimator__estimator__random_state": None
        }
        base = Pipeline([
            ("vectorizer", CountVectorizer(binary=False)),
            ("estimator", LogisticRegression(C=1))
        ])
        clf = KRandomClassifierChains(base, k=3, random_state=0)
        clf.set_params(**params)
        for e in clf.estimators_:
            self.assertEqual(e.base_estimator.steps[1][1].C, 50)
            self.assertIsNone(e.base_estimator.steps[1][1].random_state)
            self.assertTrue(e.base_estimator.steps[0][1].binary)

    def test_set_random_state_also_updates_rng(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=None)
        clf.set_params(**{"random_state": 0})
        self.assertEqual(clf.random_state, 0)
        # setting the state also changes estimators so spin through these
        # values first.
        for _ in range(clf.k):
            self.rng.randint(MAX_INT)
        # arbitrary number to check same ints are being generated
        for _ in range(5):
            self.assertEqual(
                clf.rng_.randint(MAX_INT), self.rng.randint(MAX_INT))

    def test_set_random_propagates_to_base_estimators(self):
        clf = KRandomClassifierChains(self.base, k=3, random_state=None)
        clf.set_params(**{"random_state": 0})
        for e in clf.estimators_:
            self.assertEqual(e.random_state, self.rng.randint(MAX_INT))
        clf.set_params(**{"random_state": None})
        for e in clf.estimators_:
            self.assertIsNone(e.random_state)

    def test_setting_k_regenerates_esimtators(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        self.assertEqual(len(clf.estimators_), 10)
        for e in clf.estimators_:
            self.assertEqual(e.random_state, self.rng.randint(MAX_INT))

        clf.set_params(**{"k": 3})
        self.assertEqual(len(clf.estimators_), 3)
        for e in clf.estimators_:
            self.assertEqual(e.random_state, self.rng.randint(MAX_INT))

    def test_setting_k_resets_fitted_status(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        self.assertTrue(clf.fitted_)
        clf.set_params(**{"k": 3})
        self.assertFalse(clf.fitted_)

    def test_setting_base_estimator_regenerates_esimtators(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        base = LogisticRegression(C=220)
        clf.set_params(**{"base_estimator": base})
        for e in clf.estimators_:
            self.assertEqual(e.base_estimator.C, 220)

    def test_setting_base_estimator_resets_fitted(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        self.assertTrue(clf.fitted_)

        base = LogisticRegression(C=220)
        clf.set_params(**{"base_estimator": base})
        self.assertFalse(clf.fitted_)

    def test_each_estimator_has_different_order(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        for e1 in clf.estimators_:
            for e2 in clf.estimators_:
                if (np.all(e1.order_ == e2.order_)) and (e1 is not e2):
                    self.fail("Orders are not unique: {} == {}".format(
                        e1.order_, e2.order_
                    ))

    def test_predictions_have_shape_n_instance_n_labels(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        pred = clf.predict(self.X)
        self.assertEqual(pred.shape, (self.y.shape))

    def test_predict_proba_correctly_stacks_arrays_from_each_estimator(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        self.assertEqual(probas.shape, self.y.shape)

    def test_scores_returns_per_class_score_when_average_is_binary(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        probas = clf.score(
            self.X, self.y, scorer=f1_score, **{'average': 'binary'}
        )
        self.assertEqual(probas.shape, (5,))

    def test_scores_returns_single_score_when_not_average_is_binary(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        predictions = clf.predict(self.X)
        result = clf.score(self.X, self.y, scorer=f1_score, average='weighted')
        exptected = f1_score(self.y, predictions, average='weighted')
        self.assertAlmostEqual(exptected, result)

    def test_score_propagates_kw_args(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        predictions = clf.predict(self.X)

        kwargs = {'eps': 0.01, 'normalize': False}
        result = clf.score(self.X, self.y, scorer=log_loss, **kwargs)
        exptected = log_loss(self.y, predictions, **kwargs)
        self.assertAlmostEqual(exptected, result)

    def test_score_will_use_proba(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        expected_score = log_loss(self.y, probas)
        result = clf.score(self.X, self.y, scorer=log_loss, use_proba=True)
        self.assertAlmostEqual(expected_score, result)

    def test_error_y_shape_not_multilabel(self):
        clf = KRandomClassifierChains(self.base, random_state=0)
        with self.assertRaises(ValueError):
            clf._check_y_shape(self.y[:, 0])
