#!/usr/bin/python

"""
This sub-module contains scoring functions for multi-label indicator arrays
and a class for housing statistics which wraps around a pandas dataframe.
"""

import numpy as np
import pandas as pd


class MultilabelScorer(object):
    """
    Simple helper class to wrap over a binary metric in sci-kit to
    enable support for simple multi-label scoring.
    """
    def __init__(self, sklearn_binary_metric):
        self.scorer = sklearn_binary_metric

    def __call__(self, y, y_pred, **kwargs):
        scores = []
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        n_classes = y.shape[1]
        for i in range(n_classes):
            score = self.scorer(y[:, i], y_pred[:, i], **kwargs)
            scores.append(score)
        return np.asarray(scores)


class Statistics(object):
    """
    Simple class to house statistics from learning. Essentially a
    wrapper around a dataframe  with three columns:
    ['Label', 'Data', 'Statistic'].
    """

    def __init__(self):
        self._label_col = 'Label'
        self._data_col = 'Data'
        self._statistic_col = 'Statistic'
        self._columns = [self._label_col, self._data_col, self._statistic_col]
        self._statistics = pd.DataFrame(columns=self._columns)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return self.__repr__()

    def frame(self):
        return self._statistics

    def labels(self):
        return np.unique(self._statistics[self._label_col].values)

    def columns(self):
        return self._columns

    def statistic_types(self):
        return set(self._statistics[self._statistic_col].values)

    def update_statistics(self, label, s_type, data):
        _df = pd.DataFrame(columns=self._columns)
        if isinstance(data, list) or isinstance(data, np.ndarray):
            n = len(data)
            _df[self._label_col] = [label for label in np.repeat(label, n)]
            _df[self._data_col] = [d for d in data]
            _df[self._statistic_col] = [s_type for s_type
                                        in np.repeat(s_type, n)]
        else:
            _df[self._label_col] = [label]
            _df[self._data_col] = [data]
            _df[self._statistic_col] = [s_type]
        self._statistics = pd.concat(
            [self._statistics, _df], ignore_index=True
        )
        return self

    def get_statistic(self, label, s_type):
        assert label in self.labels()
        assert s_type in self.statistic_types()
        _df = self._statistics[self._statistics[self._label_col] == label]
        statistic = (_df[_df[self._statistic_col] == s_type])[self._data_col]
        mu = np.mean(statistic.values)
        std = np.std(statistic.values)
        return mu, std

    def print_statistics(self, label):
        print("{}:".format(label))
        for statistic in self.statistic_types():
            mu, std = self.get_statistic(label, statistic)
            print("\t{}:\t{:.6}\t{:.6}".format(statistic, mu, std))

    def merge(self, other):
        assert self.columns() == other.columns()
        _df = pd.concat([self.frame(), other.frame()], ignore_index=True)
        assert _df.size == self.frame().size + other.frame().size
        self._statistics = _df
        return self

    def write(self, fp, mode='a'):
        labels = self.labels()
        if len(self._statistics) == 0:
            raise ValueError("Nothing to write.")
        if isinstance(fp, str):
            fp = open(fp, mode)
        max_word_len = max([len(l) for l in labels])
        for l in sorted(labels):
            fp.write('{}'.format(l) + ' ' * (max_word_len-len(l))
                     + '\tmean' + ' '*6 + '\tstd\n')
            for s in sorted(self.statistic_types()):
                mu, std = self.get_statistic(l, s)
                fp.write('\t{}:'.format(s) + ' ' *
                         (max_word_len - len(s) - 5) +
                         '\t{:.6f}\t{:.6f}\n'.format(mu, std))
            fp.write('\n')
        return

    @staticmethod
    def statistics_from_data(data, df_statistics, classes, return_df=False):
        """
        Utility function to put the statistics generated from either
        KFolExperiment or Bootstrap into a dataframe with the columns
        'Label' for the class labels, 'Statistic' for the scoring function
        types and `Data` for the numerical results.

        :param data: array-like, shape (a, b, c, d)
            a: `self.n_iter` or 1 if mean_bs is True
            b: `n_splits` in `kfold_experiemnt` or 1 if mean_kf is True
            c: number of score functions passed in `score_funcs` during eval
            d: number of classes for multi-label, multiclass `y`
        :param df_statistics: array-like, shape (n_score_funcs, )
            The names you would like each scoring function used during
            evaluation to have.
        :param classes: array-like, shape (n_classes, )
            The classes in the order determined by a label binaraizer or as
            order used by the estimator, typically found in the `classes_`
            attribute or similar for a sklearn estimator object. Consult the
            sklearn documentation for specfic details.
        :param return_df: boolean, optional
            Return the pd.DataFrame wrapped by Statistics

        :return: Statistics Object or pd.DataFrame
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        stats = Statistics()
        axis_0 = data.shape[0]
        axis_1 = data.shape[1]
        axis_2 = data.shape[2]
        axis_3 = data.shape[3]

        if axis_2 != len(df_statistics):
            raise ValueError("The length of `df_statistics` doesn't match the "
                             "length of axis 2 in `data`.")
        if axis_3 != len(classes):
            raise ValueError("The length of `classes` doesn't match the "
                             "length of axis 3 in `data`.")

        for i, s in enumerate(df_statistics):
            for j, c in enumerate(classes):
                data_sc = data[:, :, i, j].ravel()
                stats.update_statistics(c, s, data_sc)

        assert len(stats.frame()) == axis_0*axis_1*axis_2*axis_3
        if return_df:
            return stats.frame()
        return stats
