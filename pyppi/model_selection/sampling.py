#!/usr/bin/python

"""
This is where data sampling functions and classes can be found
"""

__all__ = ["IterativeStratifiedKFold"]

import warnings
import numpy as np

from ..base.utilities import chunk_list

from sklearn.utils import check_random_state
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

    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(IterativeStratifiedKFold, self).__init__(
            n_splits, shuffle, random_state
        )

    def clone(self):
        return IterativeStratifiedKFold(
            self.n_splits, self.shuffle, self.random_state)

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
        elif y.shape[1] < 2:
            raise ValueError(
                "Requires y to be of shape (n_samples, n_labels)")

        n_labels = y.shape[1]
        n_instances = y.shape[0]
        unlabelled = np.where(np.sum(y, axis=1) == 0)[0]
        n_unlabelled = unlabelled.shape[0]

        folds = [[] for _ in range(self.n_splits)]
        fold_instance_counts = np.zeros(shape=(self.n_splits,))
        fold_labels_counts = np.zeros(shape=(self.n_splits, n_labels))

        y_counts = np.sum(y, axis=0)
        placed_instances = np.zeros(shape=(n_instances,))

        while sum(fold_instance_counts) < (n_instances - n_unlabelled):
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
                # for this label (fold with fewest number of this label)
                fold_counts_for_label = fold_labels_counts[:, label]
                min_label_count = np.min(fold_counts_for_label)

                fs_with_least_label_count = np.where(
                    fold_counts_for_label == min_label_count)[0]

                # Breaking ties by considering the largest number of desired
                # examples in these folds, breaking further ties randomly (
                # folds with fewest assigned instances so far)
                instance_counts_for_fs = fold_instance_counts[
                    fs_with_least_label_count
                ]

                min_instance_count = np.min(instance_counts_for_fs)
                fs_with_least_instance_count = np.where(
                    instance_counts_for_fs == min_instance_count)[0]

                fold_pool = fs_with_least_label_count[
                    fs_with_least_instance_count]
                chosen_fold = random.choice(fold_pool, size=1)[0]

                # Append the index and update fold information
                folds[chosen_fold].append(idx)
                fold_instance_counts[chosen_fold] += 1

                instance_labels = np.where(instance == 1)[0]
                fold_labels_counts[chosen_fold, instance_labels] += 1

                placed_instances[idx] = 1
                y_counts[instance_labels] -= 1

        # Distribute unlabelled samples evenly.
        chosen_fold = 0
        for instance in unlabelled:
            folds[chosen_fold].append(instance)
            fold_instance_counts[chosen_fold] += 1
            chosen_fold += 1
            if chosen_fold >= self.n_splits:
                chosen_fold = 0

        assert(sum(fold_instance_counts) == n_instances)
        return folds

    def _make_test_folds(self, X, y=None, groups=None):
        rng = check_random_state(self.random_state)
        n_samples = y.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            # Shuffle the original indices and keep a map so later we can
            # map the new indices back to the original positions in the
            # original y input. If we don't map back then we will have
            # chosen our test indices based on the shuffled variant of y
            # and the resulting splits will be wrong if these indicies are
            # then used to index the original y.
            indices_shuff = rng.permutation(indices)
            mapping = {i: o_idx for i, o_idx in enumerate(indices_shuff)}
            y = y[indices_shuff]
            assert y.shape[0] == n_samples

        y_counts = np.sum(y, axis=0)
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
        if self.shuffle:
            # Map the shuffled indices back to the their values relative
            # to the original y.
            test_folds = [
                [mapping[s_idx] for s_idx in fold]
                for fold in test_folds
            ]

        # Check that all folds are disjoint.
        for f1 in test_folds:
            for f2 in test_folds:
                if f1 is not f2:
                    is_disjoint = len(set(f1) & set(f2)) == 0
                    if not is_disjoint:
                        raise ValueError("Folds are not disjoint.")

        # Assign each instance an indicator value indicating which test
        # fold it belongs to.
        count = 0
        test_fold_indicators = np.zeros(shape=(n_samples,))
        for i, fold in enumerate(test_folds):
            test_fold_indicators[fold] = i
            count += sum(test_fold_indicators[fold] == i)

        assert count == n_samples
        return test_fold_indicators

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_fold_indicators = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_fold_indicators == i

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

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples, n_labels)
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
