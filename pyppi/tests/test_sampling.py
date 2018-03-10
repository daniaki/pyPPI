
import numpy as np
from unittest import TestCase

from ..model_selection.sampling import IterativeStratifiedKFold

from sklearn.datasets import make_multilabel_classification


class TestIterativeStratifiedKFold(TestCase):

    def test_min_label_appears_at_least_once_in_each_fold(self):
        cv = IterativeStratifiedKFold(
            n_splits=3, shuffle=False, random_state=0)
        X, y = make_multilabel_classification(
            100, 20, n_labels=5, random_state=0,
            allow_unlabeled=True
        )

        # Make label 0 have only 3 positive instances.
        y[:, 0] = 0
        y[[1, 2, 3], 0] = 1

        folds = list(cv.split(X, y))
        for train_idx, valid_idx in folds:
            self.assertEqual(np.sum(y[train_idx, 0]), 2)
            self.assertEqual(np.sum(y[valid_idx, 0]), 1)

    def test_generates_same_splits_same_random_state(self):
        cv1 = IterativeStratifiedKFold(
            n_splits=3, shuffle=False, random_state=0)
        cv2 = IterativeStratifiedKFold(
            n_splits=3, shuffle=False, random_state=0)
        X, y = make_multilabel_classification(
            100, 20, n_labels=5, random_state=0,
            allow_unlabeled=True
        )
        folds1 = list(cv1.split(X, y))
        folds2 = list(cv2.split(X, y))
        for (a, c), (b, d) in zip(folds1, folds2):
            self.assertEqual(list(a), list(b))
            self.assertEqual(list(c), list(d))

    def test_evenly_distributes_unlabelled(self):
        cv = IterativeStratifiedKFold(
            n_splits=5, shuffle=False, random_state=0)
        X, y = make_multilabel_classification(
            100, 20, n_labels=5, random_state=0, allow_unlabeled=False
        )

        y[[0, 1, 2, 3, 4], :] = 0

        # Make label 0 have only 3 positive instances.

        folds = list(cv.split(X, y))
        for train_idx, valid_idx in folds:
            unlabelled_in_train = np.where(
                np.sum(y[train_idx, :], axis=1) == 0
            )[0].shape[0]
            unlabelled_in_valid = np.where(
                np.sum(y[valid_idx, :], axis=1) == 0
            )[0].shape[0]
            self.assertEqual(unlabelled_in_train, 4)
            self.assertEqual(unlabelled_in_valid, 1)

    def test_valueerror_all_labels_counts_less_than_n_splits(self):
        cv = IterativeStratifiedKFold(
            n_splits=5, shuffle=False, random_state=0)
        X, y = make_multilabel_classification(
            5, 20, n_labels=5, random_state=0, allow_unlabeled=False
        )
        y[0, :] = 0

        with self.assertRaises(ValueError):
            list(cv.split(X, y))

    def test_warns_label_count_less_than_n_split(self):
        cv = IterativeStratifiedKFold(
            n_splits=5, shuffle=False, random_state=0)
        X, y = make_multilabel_classification(
            5, 20, n_labels=5, random_state=0, allow_unlabeled=False
        )
        y[:, :] = 1
        y[0, 0] = 0

        with self.assertWarns(Warning):
            list(cv.split(X, y))

    def test_evenly_distributes_label_with_smallest_count(self):
        X, y = make_multilabel_classification(
            100, 20, n_labels=18, random_state=0, allow_unlabeled=False,
            n_classes=18
        )
        d_idx = 0
        c_idx = 1
        m_idx = 2
        y[:, d_idx] = 0
        y[:, c_idx] = 0
        y[:, m_idx] = 0

        y[[0, 10, 20, 12, 4], d_idx] = 1
        y[[1, 11, 21, 13, 5], c_idx] = 1
        y[[2, 12, 22, 14, 6, 7], m_idx] = 1

        # With Shuffle
        iskf = IterativeStratifiedKFold(
            n_splits=3, shuffle=True, random_state=42)
        cv = list(iskf.split(X, y))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[d_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[d_idx], (1, 2))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[c_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[c_idx], (1, 2))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[m_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[m_idx], (1, 2))

        # Without shuffle
        iskf = IterativeStratifiedKFold(
            n_splits=3, shuffle=False, random_state=42)
        cv = list(iskf.split(X, y))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[d_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[d_idx], (1, 2))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[c_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[c_idx], (1, 2))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[m_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[m_idx], (1, 2))

    def test_shuffle_shuffles_splits(self):
        X, y = make_multilabel_classification(
            100, 20, n_labels=5, random_state=0, allow_unlabeled=False
        )

        # With Shuffle
        iskf = IterativeStratifiedKFold(
            n_splits=3, shuffle=True, random_state=42)
        cv1 = list(iskf.split(X, y))

        # Without shuffle
        iskf = IterativeStratifiedKFold(
            n_splits=3, shuffle=False, random_state=42)
        cv2 = list(iskf.split(X, y))

        for train_shuff, valid_shuff in cv1:
            for train_no_shuff, valid_no_shuff in cv2:
                self.assertNotEqual(list(train_shuff), list(train_no_shuff))
                self.assertNotEqual(list(valid_shuff), list(valid_no_shuff))

    def test_evenly_distributes_label_with_multilabel(self):
        X, y = make_multilabel_classification(
            100, 20, n_labels=18, random_state=0, allow_unlabeled=False,
            n_classes=18
        )
        d_idx = 0
        c_idx = 1
        y[:, d_idx] = 0
        y[:, c_idx] = 0

        y[[0, 10, 20, 12, 4], d_idx] = 1
        y[[1, 11, 21, 12, 4], c_idx] = 1

        # With Shuffle
        iskf = IterativeStratifiedKFold(
            n_splits=3, shuffle=True, random_state=42)
        cv = list(iskf.split(X, y))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[d_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[d_idx], (1, 2))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[c_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[c_idx], (1, 2))

        # With Shuffle
        iskf = IterativeStratifiedKFold(
            n_splits=3, shuffle=False, random_state=42)
        cv = list(iskf.split(X, y))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[d_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[d_idx], (1, 2))
        for train, valid in cv:
            self.assertIn(y[train].sum(axis=0)[c_idx], (4, 3))
            self.assertIn(y[valid].sum(axis=0)[c_idx], (1, 2))
