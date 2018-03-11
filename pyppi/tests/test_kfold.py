import numpy as np
from numpy.random import RandomState
from unittest import TestCase

from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import hamming_loss, label_ranking_loss
from sklearn.metrics import f1_score, log_loss

from ..base.constants import MAX_SEED
from ..model_selection.k_fold import StratifiedKFoldCrossValidation
from ..models.binary_relevance import MixedBinaryRelevanceClassifier
from ..models.classifier_chain import KRandomClassifierChains

MAX_INT = MAX_SEED


class TestStratifiedKFoldCrossValidation(TestCase):

    def setUp(self):
        self.n_labels = 5
        self.X, self.y = make_classification(random_state=0)
        self.Xml, self.yml = make_multilabel_classification(random_state=0)
        self.base = LogisticRegression(random_state=0)
        self.rng = RandomState(seed=0)

    def test_can_fit_pipeline(self):
        base = Pipeline([
            ("vectorizer", CountVectorizer(binary=False)),
            ("estimator", LogisticRegression(C=1))
        ])
        X = np.asarray(
            [
                'dog,cat', 'cat,elephant', 'salmon,tuna', 'board,game',
                'dog,salmon', 'cat,game', 'cat,tuna', 'elephant,game'
            ]
        )
        y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        clf = StratifiedKFoldCrossValidation(
            base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(X, y)
        self.assertTrue(hasattr(clf, 'fold_estimators_'))
        self.assertTrue(hasattr(clf, 'cv_'))
        self.assertTrue(hasattr(clf, 'multilabel_'))

    def test_input_can_detect_multilabel_input(self):
        clf = StratifiedKFoldCrossValidation(
            self.base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        self.assertTrue(clf._input_is_multilabel(self.yml))
        self.assertFalse(clf._input_is_multilabel(self.y))

    def test_can_fit_regular_clf(self):
        clf = StratifiedKFoldCrossValidation(
            self.base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'fold_estimators_'))
        self.assertTrue(hasattr(clf, 'cv_'))
        self.assertTrue(hasattr(clf, 'multilabel_'))

    def test_can_fit_gridsearch_clf(self):
        base = RandomizedSearchCV(
            self.base, {'C': range(1, 100)}
        )
        clf = StratifiedKFoldCrossValidation(
            base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'fold_estimators_'))
        self.assertTrue(hasattr(clf, 'cv_'))
        self.assertTrue(hasattr(clf, 'multilabel_'))

    def test_fitted_estimators_are_clones(self):
        clf = StratifiedKFoldCrossValidation(
            self.base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.X, self.y)
        for estimator in clf.fold_estimators_:
            self.assertIsNot(estimator, clf.estimator)

    def test_score_averages_by_default(self):
        clf = StratifiedKFoldCrossValidation(
            self.base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.X, self.y)

        result = clf.score(self.X, self.y)
        e_std = np.std([e.score(self.X, self.y) for e in clf.fold_estimators_])
        e_mean = np.mean(
            [e.score(self.X, self.y) for e in clf.fold_estimators_]
        )
        e_err = e_std / np.sqrt(len(clf.fold_estimators_))
        self.assertAlmostEqual(e_mean, result[0])
        self.assertAlmostEqual(e_err, result[1])

    def test_score_returns_array_of_fold_scores_if_not_averaging(self):
        clf = StratifiedKFoldCrossValidation(
            self.base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.X, self.y)
        result = clf.score(self.X, self.y, avg_folds=False)
        self.assertEqual(result.shape, (3,))

    def test_score_uses_validation_folds_if_validation_is_true(self):
        clf = StratifiedKFoldCrossValidation(
            self.base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.X, self.y)

        means = []
        for e, (_, v_idx) in zip(clf.fold_estimators_, clf.cv_):
            means.append(e.score(self.X[v_idx], self.y[v_idx]))
        e_mean = np.mean(means)
        e_std = np.std(means)
        e_err = e_std / np.sqrt(len(clf.fold_estimators_))
        result = clf.score(self.X, self.y, validation=True)

        self.assertAlmostEqual(e_mean, result[0])
        self.assertAlmostEqual(e_err, result[1])

    def test_score_propagates_score_params(self):
        base = MixedBinaryRelevanceClassifier(
            [LogisticRegression(random_state=0) for i in range(5)]
        )
        clf = StratifiedKFoldCrossValidation(
            base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.Xml, self.yml)

        params = {"use_proba": False, 'scorer': f1_score, 'average': 'macro'}
        mean, err = clf.score(self.Xml, self.yml, **params)

        expected = []
        for e in clf.fold_estimators_:
            expected.append(
                e.score(
                    self.Xml, self.yml, use_proba=False, scorer=f1_score,
                    **{'average': 'macro'}
                )
            )
        e_mean = np.mean(expected)
        e_std = np.std(expected)
        e_err = e_std / np.sqrt(clf.n_folds)

        self.assertAlmostEqual(e_mean, mean)
        self.assertAlmostEqual(e_err, err)

    def test_score_averages_multi_label_binary_correctly(self):
        base = MixedBinaryRelevanceClassifier(
            [LogisticRegression(random_state=0) for i in range(5)]
        )
        clf = StratifiedKFoldCrossValidation(
            base, n_folds=3, shuffle=False, random_state=0, verbose=False
        )
        clf.fit(self.Xml, self.yml)

        params = {"use_proba": False, 'scorer': f1_score, 'average': 'binary'}
        mean, err = clf.score(self.Xml, self.yml, **params)

        expected = []
        for e in clf.fold_estimators_:
            y_pred = e.predict(self.Xml)
            means_f = []
            for i in range(5):
                means_f.append(
                    f1_score(self.yml[:, i], y_pred[:, i], average='binary')
                )
            expected.append(means_f)
        e_mean = np.mean(expected, axis=0)
        e_std = np.std(expected, axis=0)
        e_err = e_std / np.sqrt(clf.n_folds)

        self.assertTrue(np.all(np.isclose(mean, e_mean)))
        self.assertTrue(np.all(np.isclose(err, e_err)))
