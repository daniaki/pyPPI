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

    def _arg_with_ties(self, array, arg_func=min, break_ties=False):
        """
        Breaks a tie by random choice using supplied if one occurs.
        """
        # get min/max element
        array = list(array)
        elem = arg_func(array)
        argxs = [i for i in range(len(array)) if array[i] == elem]
        rng = check_random_state(self.random_state)
        if break_ties:
            return rng.choice(argxs, size=1, replace=False)[0]
        else:
            return sorted(argxs)

    def _iterative_stratification(self, y):
        """
        Implementation of the Iterative Stratification algorithm from
        Sechidis et al. 2011.
        """
        r = 1 / self.n_splits
        range_folds = range(self.n_splits)
        folds_idx = [[] for _ in range_folds]
        rng = check_random_state(self.random_state)
        test_idx = np.zeros(y.shape[0], dtype=int)

        # Unlabelled samples removed and split up last.
        no_label_y = [idx for idx, s in enumerate(y) if sum(s) == 0]
        labelled_y = [idx for idx, s in enumerate(y) if sum(s) != 0]

        n_samples = len(labelled_y)
        unique_y = np.asarray(range(y.shape[1]))
        y_counts = np.zeros((len(unique_y),), dtype=int)
        for c in unique_y:
            y_counts[c] = np.sum(y[:, c])
        unique_y = np.asarray([s for s, c in zip(unique_y, y_counts) if c > 0])
        y_counts = y_counts[y_counts != 0]

        # Calculate the desired number of samples for each subset
        subset_c_j = np.asarray([n_samples*r for _ in range_folds])

        # Calculate the desired number of samples from each label for
        # each subset
        n_samples_for_label = {l: c*r for (c, l) in zip(y_counts, unique_y)}
        subset_c_ij = np.asarray([n_samples_for_label.copy()
                                  for _ in range_folds], dtype='object')
        completed_labels = []
        sampled_indices = set()

        while len(sampled_indices) < n_samples:
            # Find the label with the fewest (but at least one) remaining
            # samples, breaking ties randomly.
            n_samples_for_label = {l: c*r for (c, l) in zip(y_counts, unique_y)
                                   if l not in completed_labels}
            index = self._arg_with_ties(
                n_samples_for_label.values(), arg_func=min, break_ties=True
            )
            label = list(n_samples_for_label.keys())[index]
            y_l = [idx for idx in labelled_y if y[idx][label] == 1]

            for idx in y_l:
                # Ignore indices we have already sampled.
                if idx in sampled_indices:
                    continue

                # Find the subset(s) with the largest number of desired
                # samples for this label.
                fold_indices = self._arg_with_ties(
                    [subset_c_ij[i][label] for i in range_folds], arg_func=max,
                    break_ties=False
                )

                # Breaking ties by considering the largest number of desired
                # samples, breaking further ties randomly.
                index = self._arg_with_ties(
                    subset_c_j[fold_indices], arg_func=max, break_ties=True
                )
                fold_index = fold_indices[index]

                # Enter the instance to the proper fold
                row = y[idx]
                folds_idx[fold_index].append(idx)
                sampled_indices.add(idx)

                # Update the sampled labeled statistics of this fold
                for row_l in unique_y:
                    if row[row_l] == 1:
                        subset_c_ij[fold_index][row_l] -= 1

            # Update the sampled proportion statistics of this fold
            subset_c_j[fold_index] -= 1
            completed_labels.append(label)

        # # Split up the unlablled samples
        rng.shuffle(no_label_y)
        for i, c in enumerate(chunk_list(no_label_y, self.n_splits)):
            folds_idx[i] += c

        # Test for disjoint-ness:
        for a in folds_idx:
            for b in folds_idx:
                if id(a) == id(b):
                    continue
                assert len(set(a) & set(b)) == 0

        assert sum([len(f) for f in folds_idx]) == y.shape[0]
        for i in range(self.n_splits):
            idx = np.asarray(folds_idx[i], dtype=int)
            test_idx[idx] = i
        return test_idx

    def _make_test_folds(self, X, y=None, groups=None):
        rng = check_random_state(self.random_state)
        # shuffle X and y here.
        if self.shuffle:
            Xy = list(zip(X,y))
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

        # pre-assign each sample to a test fold index using individual KFold
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

