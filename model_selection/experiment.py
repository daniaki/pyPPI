#!/usr/bin/env python

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.externals.joblib import Parallel
from sklearn.utils.validation import check_is_fitted

from base import create_seeds
from model_selection.sampling import IterativeStratifiedKFold


class Bootstrap(object):
    """
    This is a class that performs KFold cross-validation to
    perform a single KFoldExperiment a number of times.

    Parameters
    ----------
    kfold_experiemnt : KFoldExperiment object
        A KFoldExperiment object instantiated with the desired parameters
        that will be used as a template for all iterations.

    n_iter : int, optional
        The number of iterations to compute.

    n_jobs : int, optional
        The number of parallel jobs to spawn to run iterations in parallel.

    verbose : boolean, optional.
        Set to true to print out intermediate messages from parallel and
        joblib.

    Attributes
    ----------
    experiments : List[kfold_experiemnt]
        The fitted KFoldExperiment objects.
    """

    def __init__(self, kfold_experiemnt, n_iter, n_jobs=1, verbose=False):
        seeds = create_seeds(n_iter)
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.experiments = [kfold_experiemnt.clone(s) for s in seeds]

    def _fit_kfold(self, X, y, i):
        return self.experiments[i].fit(X, y)

    def _scores(self, X, y, i, dispatch_func, score_funcs, thresholds, mean):
        results = []
        for (f, t), idx in enumerate(zip(score_funcs, thresholds)):
            scores = getattr(self.experiments[i], dispatch_func)(
                X, y, f, t, mean
            )
            results.append(scores)
        return np.asarray(results)

    def fit(self, X, y):
        self.experiments = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                    backend='multiprocessing')(
            self._fit_kfold(X, y, i) for i in range(self.n_iter)
        )
        self.fitted_ = True
        return self

    def validation_scores(self, X, y, score_funcs, thresholds, mean=False):
        check_is_fitted(self, '_fitted')
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           backend='multiprocessing')(
            self._scores(X, y, i, 'validation_score', score_funcs,
                         thresholds, mean)
            for i in range(self.n_iter)
        )
        return np.asarray(results)

    def held_out_scores(self, X, y,  score_funcs, thresholds, mean=False):
        check_is_fitted(self, '_fitted')
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           backend='multiprocessing')(
            self._scores(X, y, i, 'held_out_score', score_funcs,
                         thresholds, mean)
            for i in range(self.n_iter)
        )
        return np.asarray(results)


class KFoldExperiment(object):
    """
    This is a class that performs KFold cross-validation to
    collect numerous statistics over K folds.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.

    random_state : None, int or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions on each fold.
    """

    def clone(self, random_state):
        cv = clone(self.cv_)
        cv.random_state = random_state
        return KFoldExperiment(
            estimator=clone(self.base_estimator_),
            cv=cv,
            shuffle=self.shuffle_,
            random_state=random_state
        )

    def __init__(self, estimator, cv=3, shuffle=True, random_state=None):
        self.base_estimator_ = clone(estimator)
        self.estimators_ = []
        self.shuffle_ = shuffle
        self.random_state_ = random_state

        if isinstance(cv, int):
            self.cv_ = StratifiedKFold(cv, shuffle, random_state)
        elif isinstance(cv, KFold):
            self.cv_ = cv
        elif isinstance(cv, StratifiedKFold):
            self.cv_ = cv
        elif isinstance(cv, IterativeStratifiedKFold):
            self.cv_ = cv
        else:
            raise ValueError("Argument `cv` must be an integer or an "
                             "instance of StratifiedKFold, KFold or "
                             "IterativeStratifiedKFold.")
        self.n_splits = self.cv_.n_splits

    def _fit(self, X, y):
        estimator = clone(self.base_estimator_)
        estimator.fit(X, y)
        return estimator

    def fit(self, X, y):
        """
         Top-level function to train the estimators over each fold, without
         performing validation.

         :param X: numpy array X, shape (n_samples, n_features)
         :param y: numpy array y, shape (n_samples, n_outputs)
         :return: Self
         """
        if hasattr(self.cv_, 'split'):
            self.cv_ = list(self.cv_.split(X, y))

        for (train_idx, _) in self.cv_:
            X_train = X[train_idx, ]
            y_train = y[train_idx, ]

            estimator = clone(self.base_estimator_)
            estimator.fit(X_train, y_train)
            self.estimators_.append(estimator)

        self.fitted_ = True
        return self

    def validation_score(self, X, y, score_func, threshold=None, mean=False):
        """
        Prodivde the cross-validation scores on a the validation dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_func: function
            Method with signature score_func(y, y_pred)
        :param threshold: float, optional
            If using probability predictons, the threshod represents the
            positive prediction cut-off.
        :param mean: booleam, optional
            Return the averaged result accross the folds.

        :return: List of scores as returned by score_func for each fold.
        """
        scores = []
        check_is_fitted(self, '_fitted')
        for (_, test_idx), estimator in zip(self.cv_, self.estimators_):
            X_test = X[test_idx, ]
            y_test = y[test_idx, ]

            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba(X_test)
                if threshold:
                    y_pred = (y_pred >= threshold).astype(int)
            else:
                print("Warning: {} does not have a `predict_proba` "
                      "implementation. Using `predict` "
                      "instead.".format(type(estimator)))
                y_pred = estimator.predict(X_test)

            score = score_func(y_test, y_pred)
            scores.append(score)

        if mean:
            return np.mean(scores)
        return np.asarray(scores)

    def held_out_score(self, X, y, score_func, threshold=None, mean=False):
        """
        Prodivde the cross-validation scores on a held out dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_func: function
            Method with signature score_func(y, y_pred)
        :param threshold: float, optional
            If using probability predictons, the threshod represents the
            positive prediction cut-off.
        :param mean: booleam, optional
            Return the averaged result accross the folds.

        :return: List of scores as returned by score_func for each fold.
        """
        scores = []
        check_is_fitted(self, '_fitted')
        for estimator in self.estimators_:
            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba(X)
                if threshold:
                    y_pred = (y_pred >= threshold).astype(int)
            else:
                print("Warning: {} does not have a `predict_proba` "
                      "implementation. Using `predict` "
                      "instead.".format(type(self.base_estimator_)))
                y_pred = estimator.predict(X)

            score = score_func(y, y_pred)
            scores.append(score)

        if mean:
            return np.mean(scores)
        return np.asarray(scores)


# -------------------------------- TESTS ------------------------------------ #
def test_kfold():
    from sklearn.datasets import make_multilabel_classification
    from model_selection.sampling import IterativeStratifiedKFold
    from models.binary_relevance import BinaryRelevance
    from sklearn.linear_model import LogisticRegression

    X, y = make_multilabel_classification()
    est = [LogisticRegression() for _ in range(5)]
    est = BinaryRelevance(est)
    kf = KFoldExperiment(est, cv=IterativeStratifiedKFold())
    kf.fit(X, y)
    return kf


def test_bs():
    from sklearn.datasets import make_multilabel_classification
    from model_selection.sampling import IterativeStratifiedKFold
    from models.binary_relevance import BinaryRelevance
    from sklearn.linear_model import LogisticRegression

    X, y = make_multilabel_classification()
    est = [LogisticRegression() for _ in range(5)]
    est = BinaryRelevance(est)
    kf = KFoldExperiment(est, cv=IterativeStratifiedKFold())
    bs = Bootstrap(kf, n_iter=3, verbose=True)
    bs.fit(X, y)
    return bs