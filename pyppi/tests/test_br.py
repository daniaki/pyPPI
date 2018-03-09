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


from ..models.binary_relevance import MixedBinaryRelevanceClassifier


class TestMixedBinaryRelevanceClassifier(TestCase):

    def setUp(self):
        self.n_labels = 5
        self.X, self.y = make_multilabel_classification(
            100, 20, n_labels=self.n_labels, random_state=0
        )
        self.base = [
            LogisticRegression(random_state=0) for _ in range(self.n_labels)
        ]

    def test_get_params_returns_clones(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        params = clf.get_params()
        for (e1, e2) in zip(clf.estimators, params["estimators"]):
            self.assertIsNot(e1, e2)
            self.assertEqual(e1.get_params(), e2.get_params())

    def test_clone_method_clones_internal_classifiers(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        new_clf = clf.clone()
        for (e1, e2) in zip(clf.estimators, new_clf.estimators):
            self.assertIsNot(e1, e2)
            self.assertEqual(e1.get_params(), e2.get_params())

    def test_error_estimators_not_list(self):
        with self.assertRaises(TypeError):
            clf = MixedBinaryRelevanceClassifier(self.base[0])
        with self.assertRaises(TypeError):
            clf = MixedBinaryRelevanceClassifier(self.base)
            clf.set_params(**{"estimators": self.base[0]})

    def test_set_params_estimators_deletes_fitted_attrs(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'estimators_'))
        self.assertTrue(hasattr(clf, 'n_labels_'))

        clf.set_params(**{'estimators': self.base})
        self.assertFalse(hasattr(clf, 'estimators_'))
        self.assertFalse(hasattr(clf, 'n_labels_'))

    def test_set_params_estimators_are_clones(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.set_params(**{'estimators': self.base})
        for (e1, e2) in zip(self.base, clf.estimators):
            self.assertIsNot(e1, e2)
            self.assertEqual(e1.get_params(), e2.get_params())

    def test_fitted_estimators_are_clones(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        for (e1, e2) in zip(clf.estimators_, clf.estimators):
            self.assertIsNot(e1, e2)
            self.assertEqual(e1.get_params(), e2.get_params())

    def test_check_y_raises_error_not_same_shape_len_estimators(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        with self.assertRaises(ValueError):
            clf.fit(self.X, self.y[:, [1, 2]])

    def test_check_y_raises_error_y_not_multi_label(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        with self.assertRaises(ValueError):
            clf.fit(self.X, self.y[:, 0])

    def test_predict_returns_correct_shape(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        predict = clf.predict(self.X)
        self.assertEqual(self.y.shape, predict.shape)

    def test_predict_proba_returns_shape_n_labels(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        proba = clf.predict_proba(self.X)
        self.assertEqual(self.y.shape, proba.shape)

    def test_predict_proba_correctly_stacks(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        proba_idx_0 = clf.predict_proba(self.X)[:, 0]
        expected = clf.estimators_[0].predict_proba(self.X)[:, 1]
        self.assertTrue(np.all(np.isclose(proba_idx_0, expected)))

    def test_scores_returns_per_class_score_when_average_is_binary(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        probas = clf.score(
            self.X, self.y, scorer=f1_score, **{'average': 'binary'}
        )
        self.assertEqual(probas.shape, (self.n_labels,))

    def test_scores_returns_single_score_when_average_is_not_binary(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        predictions = clf.predict(self.X)
        result = clf.score(self.X, self.y, scorer=f1_score, average='weighted')
        exptected = f1_score(self.y, predictions, average='weighted')
        self.assertAlmostEqual(exptected, result)

    def test_score_propagates_kw_args(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        predictions = clf.predict(self.X)
        kwargs = {'eps': 0.01, 'normalize': False}
        result = clf.score(self.X, self.y, scorer=log_loss, **kwargs)
        exptected = log_loss(self.y, predictions, **kwargs)
        self.assertAlmostEqual(exptected, result)

    def test_score_will_use_proba(self):
        clf = MixedBinaryRelevanceClassifier(self.base)
        clf.fit(self.X, self.y)
        probas = clf.predict_proba(self.X)
        expected_score = log_loss(self.y, probas)
        result = clf.score(self.X, self.y, scorer=log_loss, use_proba=True)
        self.assertAlmostEqual(expected_score, result)
