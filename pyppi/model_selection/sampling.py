#!/usr/bin/python

"""
This is where data sampling functions and classes can be found
"""

import warnings
import numpy as np

from ..base import chunk_list

from sklearn.utils import check_random_state
from sklearn.utils.fixes import bincount
from sklearn.model_selection._split import _BaseKFold, check_array


class IterativeStratifiedKFold(_BaseKFold):
    """
    Implementation of the Iterative Stratification algorithm from
    Sechidis et al. 2011 for mult-label outputs.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.

    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Examples
    --------
    >> from sklearn.model_selection import StratifiedKFold
    >> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >> y = np.array([[0,0,1], [1,0,0], [1,0,1], [1,1,0]])
    >> iskf = IterativeStratifiedKFold(n_splits=2)
    >> iskf.get_n_splits(X, y)
    2
    >> print(iskf)  # doctest: +NORMALIZE_WHITESPACE
    IterativeStratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >> for train_index, test_index in iskf.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [1 3] TEST: [0 2]

    Notes
    -----
    All the folds have size ``trunc(n_samples / n_splits)``, the last one has
    the complementary.

    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(IterativeStratifiedKFold, self).__init__(
            n_splits, shuffle, random_state
        )

    def clone(self):
        return IterativeStratifiedKFold(self.n_splits,
                                        self.shuffle, self.random_state)

    def _iterative_stratification(self, y):
        """
        Implementation of the Iterative Stratification algorithm from
        Sechidis et al. 2011.
        """
        random = check_random_state(self.random_state)

        y = np.asarray(y)
        if len(y.shape) != 2 and y.shape[1] < 2:
            raise ValueError(
                "Requires y to be of shape (n_samples, n_labels)")

        n_labels = y.shape[1]
        n_instances = y.shape[0]
        desired_proportions = 1 / self.n_splits
        desired_counts_per_fold = desired_proportions * n_instances

        folds = [[] for _ in range(self.n_splits)]
        fold_instance_counts = np.zeros(shape=(self.n_splits,))
        fold_labels_counts = np.zeros(shape=(self.n_splits, n_labels))

        label_idxs = np.asarray(range(n_labels))
        y_counts = np.sum(y, axis=0)
        placed_instances = np.zeros(shape=(n_instances,))
        desired_label_counts_per_fold = y_counts * desired_proportions

        while sum(fold_instance_counts) < n_instances:
            # Find the label with the fewest (but at least one) remaining
            # examples breaking ties randomly
            min_remaining = np.min(y_counts[y_counts > 0])
            l_fewest_remaining_instances = random.choice(
                np.where(y_counts == min_remaining)[0], size=1
            )[0]
            label = l_fewest_remaining_instances

            # Get the instances with this label that have not been placed
            # already.
            idxs = np.where(y[:, label] == 1)[0]
            idxs = idxs[placed_instances[idxs] == 0]
            for idx, instance in zip(idxs, y[idxs, :]):
                # Find the fold with the largest number of desired examples
                # for this label
                fold_counts_for_label = fold_labels_counts[:, label]
                min_label_count = np.min(fold_counts_for_label)
                fs_with_least_label_count = np.where(
                    fold_counts_for_label == min_label_count)[0]

                # Breaking ties by considering the largest number of desired
                # examples in these folds, breaking further ties randomly.
                instance_counts_for_fs = fold_instance_counts[
                    fs_with_least_label_count]
                min_instance_count = np.min(instance_counts_for_fs)
                fs_with_least_instance_count = np.where(
                    instance_counts_for_fs == min_instance_count)

                fold_pool = fs_with_least_label_count[
                    fs_with_least_instance_count]
                chosen_fold = random.choice(fold_pool, size=1)[0]

                # Append the index and update fold information
                folds[chosen_fold].append(idx)
                fold_instance_counts[chosen_fold] += 1

                instance_labels = np.where(instance == 1)
                fold_labels_counts[chosen_fold, instance_labels] += 1
                placed_instances[idx] = 1
                y_counts[instance_labels] -= 1

        # Check that all folds are disjoint.
        assert(sum(fold_instance_counts) == n_instances)
        for f1 in folds:
            for f2 in folds:
                if f1 is f2:
                    continue
                is_disjoint = len(set(f1) & set(f2)) == 0
                if not is_disjoint:
                    raise ValueError("Folds are not disjoint.")

        test_folds = np.zeros(shape=(n_instances,))
        for i, fold in enumerate(folds):
            test_folds[fold] = i
        return test_folds

    def _make_test_folds(self, X, y=None, groups=None):
        rng = check_random_state(self.random_state)
        # shuffle X and y here.
        if self.shuffle:
            Xy = list(zip(X, y))
            rng.shuffle(Xy)
            X = np.array([xy[0] for xy in Xy])
            y = np.array([xy[1] for xy in Xy])

        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = bincount(y_inversed)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("All the n_groups for individual classes"
                             " are less than n_splits=%d."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of groups for any class cannot"
                           " be less than n_splits=%d."
                           % (min_groups, self.n_splits)), Warning)

        # pre-assign each sample to a test fold index using
        test_folds = self._iterative_stratification(y)
        return test_folds

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples, n_outputs)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(IterativeStratifiedKFold, self).split(X, y, groups)


def iterative_stratified_k_fold(y, n_splits, random_state=None, shuffle=True):
    """
    Function implementation of the Iterative Stratification algorithm from
    Sechidis et al. 2011 for mult-label outputs.

    Parameters
    ----------
    y : array-like, shape (n_samples, n_outputs)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

    n_splits : int, default=3
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.

    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Returns
    -------
    `generator`
        A generator of tuples of with train indices and test indices.
    """
    if random_state is None:
        random = np.random
    else:
        random = np.random.RandomState(seed=random_state)

    y = np.asarray(y)
    if len(y.shape) != 2 and y.shape[1] < 2:
        raise ValueError("Requires y to be of shape (n_samples, n_labels)")

    if shuffle:
        random.shuffle(y)

    n_labels = y.shape[1]
    n_instances = y.shape[0]
    desired_proportions = 1 / n_splits
    desired_counts_per_fold = desired_proportions * n_instances

    folds = [[] for _ in range(n_splits)]
    fold_instance_counts = np.zeros(shape=(n_splits,))
    fold_labels_counts = np.zeros(shape=(n_splits, n_labels))

    label_idxs = np.asarray(range(n_labels))
    y_counts = np.sum(y, axis=0)
    placed_instances = np.zeros(shape=(n_instances,))
    desired_label_counts_per_fold = y_counts * desired_proportions

    while sum(fold_instance_counts) < n_instances:
        # Find the label with the fewest (but at least one) remaining examples
        # breaking ties randomly
        min_remaining = np.min(y_counts[y_counts > 0])
        l_fewest_remaining_instances = random.choice(
            np.where(y_counts == min_remaining)[0], size=1
        )[0]
        label = l_fewest_remaining_instances

        # Get the instances with this label that have not been placed already.
        idxs = np.where(y[:, label] == 1)[0]
        idxs = idxs[placed_instances[idxs] == 0]
        for idx, instance in zip(idxs, y[idxs, :]):
            # Find the fold with the largest number of desired examples for
            # this label
            fold_counts_for_label = fold_labels_counts[:, label]
            min_label_count = np.min(fold_counts_for_label)
            fs_with_least_label_count = np.where(
                fold_counts_for_label == min_label_count)[0]

            # Breaking ties by considering the largest number of desired
            # examples in these folds, breaking further ties randomly.
            instance_counts_for_fs = fold_instance_counts[
                fs_with_least_label_count]
            min_instance_count = np.min(instance_counts_for_fs)
            fs_with_least_instance_count = np.where(
                instance_counts_for_fs == min_instance_count)

            fold_pool = fs_with_least_label_count[
                fs_with_least_instance_count]
            chosen_fold = random.choice(fold_pool, size=1)[0]

            # Append the index and update fold information
            folds[chosen_fold].append(idx)
            fold_instance_counts[chosen_fold] += 1

            instance_labels = np.where(instance == 1)
            fold_labels_counts[chosen_fold, instance_labels] += 1
            placed_instances[idx] = 1
            y_counts[instance_labels] -= 1

    # Check that all folds are disjoint.
    assert(sum(fold_instance_counts) == n_instances)
    for f1 in folds:
        for f2 in folds:
            if f1 is f2:
                continue
            is_disjoint = len(set(f1) & set(f2)) == 0
            if not is_disjoint:
                raise ValueError("Folds are not disjoint.")

    test_folds = np.zeros(shape=(n_instances,))
    for i, fold in enumerate(folds):
        test_folds[fold] = i

    for i in range(n_splits):
        train_idx, test_idx = np.where(test_folds != i)[
            0], np.where(test_folds == i)[0]
        is_disjoint = len(set(train_idx) & set(test_idx)) == 0
        if not is_disjoint:
            raise ValueError("Folds are not disjoint.")
        if not len(train_idx) + len(test_idx) == n_instances:
            raise ValueError("Missing instances for fold {}".format(i))
        yield train_idx, test_idx
