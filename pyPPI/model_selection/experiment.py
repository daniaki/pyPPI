#!/usr/bin/env python

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

from ..base import create_seeds
from .sampling import IterativeStratifiedKFold


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

    def _check_fitted(self):
        if not hasattr(self, 'fitted_'):
            raise AttributeError("Please use the `fit` function before"
                                 "attempting to score performance.")
        return

    def __init__(self, kfold_experiemnt, n_iter, n_jobs=1, verbose=False):
        seeds = create_seeds(n_iter)
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.experiments = [kfold_experiemnt.clone(s) for s in seeds]

    def _fit(self, X, y, i):
        return self.experiments[i].fit(X, y)

    def _scores(self, X, y, i, dispatch_func, score_funcs, thresholds, mean):
        scores = getattr(self.experiments[i], dispatch_func)(
            X, y, score_funcs, thresholds, mean
        )
        return np.asarray(scores)

    def fit(self, X, y):
        fit_func = delayed(self._fit)
        self.experiments = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                    backend='multiprocessing')(
            fit_func(X, y, i) for i in range(self.n_iter)
        )
        self.fitted_ = True
        return self

    def validation_scores(self, X, y, score_funcs, thresholds,
                          mean_kf=False, mean_bs=False):
        """
        Prodivde the cross-validation scores on a the validation dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_funcs: function
            Method with signature score_func(y, y_pred)
        :param thresholds: float, optional
            If using probability predictons, the threshod represents the
            positive prediction cut-off.
        :param mean_kf: Average over the iterations.
        :param mean_bs: Average over the folds.

        :return: array-like, shape (a, b, c, d)
            a: `self.n_iter` or 1 if mean_bs is True
            b: `n_splits` in `kfold_experiemnt` or 1 if mean_kf is True
            c: number of score functions passed in `score_funcs`
            d: number of classes for multi-label, multiclass `y`
        """
        self._check_fitted()
        score = delayed(self._scores)
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                          backend='multiprocessing')(
            score(X, y, i, 'validation_score', score_funcs,
                  thresholds, mean_kf)
            for i in range(self.n_iter)
        )
        scores = np.asarray(scores)[:, 0, :, :, :]
        if mean_bs:
            scores = [np.mean(scores, axis=0)]
        return np.asarray(scores)

    def held_out_scores(self, X, y,  score_funcs, thresholds,
                        mean_kf=False, mean_bs=False):
        """
        Prodivde the cross-validation scores on a the held-out dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_funcs: function
            Method with signature score_func(y, y_pred)
        :param thresholds: float, optional
            If using probability predictons, the threshod represents the
            positive prediction cut-off.
        :param mean_kf: Average over the iterations.
        :param mean_bs: Average over the folds.

        :return: array-like, shape (a, b, c, d)
            a: `self.n_iter` or 1 if mean_bs is True
            b: `n_splits` in `kfold_experiemnt` or 1 if mean_kf is True
            c: number of score functions passed in `score_funcs`
            d: number of classes for multi-label, multiclass `y`
        """
        self._check_fitted()
        score = delayed(self._scores)
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                          backend='multiprocessing')(
            score(X, y, i, 'held_out_score', score_funcs,
                  thresholds, mean_kf)
            for i in range(self.n_iter)
        )
        scores = np.asarray(scores)[:, 0, :, :, :]
        if mean_bs:
            scores = [np.mean(scores, axis=0)]
        return np.asarray(scores)


class KFoldExperiment(object):
    """
    This is a class that performs KFold cross-validation to
    collect numerous statistics over K folds.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`. If using a Pipeline estimator, classification step
        must be named 'clf'.

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

    def __init__(self, estimator, cv=3, shuffle=True, random_state=None,
                 n_jobs=1, verbose=False):
        self.base_estimator_ = clone(estimator)
        self.estimators_ = []
        self.shuffle_ = shuffle
        self.random_state_ = random_state
        self.n_jobs_ = n_jobs
        self.verbose_ = verbose

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
        self.cv_.random_state = random_state

    def clone(self, random_state):
        klass = self.cv_.__class__
        cv = klass(**self.cv_.__dict__)
        return KFoldExperiment(
            estimator=clone(self.base_estimator_),
            cv=cv,
            shuffle=self.shuffle_,
            random_state=random_state,
            n_jobs=self.n_jobs_,
            verbose=self.verbose_
        )

    def _fit_single(self, X, y, train_idx):
        estimator = clone(self.base_estimator_)
        if hasattr(estimator, 'random_state'):
            estimator.set_params(**{'random_state': self.random_state_})
        if hasattr(estimator, 'clf__random_state'):
            estimator.set_params(**{'clf__random_state': self.random_state_})
        estimator.fit(X[train_idx, ], y[train_idx, ])
        return estimator

    def _score_single(self, X, y, test_idx, estimator,
                      score_funcs, thresholds):
        """
         Wrapper over the scoring function to use in parallel map.
        """
        if not isinstance(score_funcs, list):
            score_funcs = [score_funcs]
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        if len(score_funcs) != len(thresholds):
            raise ValueError("`score_funcs` must be the same length as"
                             "`thresholds` if using multiple values.")
        X_test = X
        y_test = y
        if test_idx is not None:
            X_test = X[test_idx, ]
            y_test = y[test_idx, ]

        scores = []
        for idx, (f, t) in enumerate(zip(score_funcs, thresholds)):
            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba(X_test)
                y_pred = (y_pred >= t).astype(int)
            else:
                print("Warning: {} does not have a `predict_proba` "
                      "implementation. Using `predict` "
                      "instead.".format(type(estimator)))
                y_pred = estimator.predict(X_test)
            scores.append(f(y_test, y_pred))
        return np.asarray(scores)

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

        fit = delayed(self._fit_single)
        self.estimators_ = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose_,
                                    backend='multiprocessing')(
            fit(X, y, train_idx) for (train_idx, _) in self.cv_
        )
        self.fitted_ = True
        return self

    def validation_score(self, X, y, score_funcs=None, thresholds=None,
                         mean=False):
        """
        Prodivde the cross-validation scores on a the validation dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_funcs: function
            Method with signature score_func(y, y_pred)
        :param thresholds: float, optional
            If using probability predictons, the threshod represents the
            positive prediction cut-off.
        :param mean: booleam, optional
            Return the averaged result accross the folds.

        :return: array-like, shape (a, b, c, d)
            a: n_iterations=1
            b: `self.n_splits` or 1 if mean is True
            c: number of score functions passed in `score_funcs`
            d: number of classes for multi-label, multiclass `y`
        """
        check_is_fitted(self, 'fitted_')
        if not score_funcs:
            score_funcs = [accuracy_score]
        if not thresholds:
            thresholds = [0.5]
        score = delayed(self._score_single)
        scores = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose_,
                          backend='multiprocessing')(
            score(X, y, test_idx, estimator, score_funcs, thresholds)
            for (_, test_idx), estimator in zip(self.cv_, self.estimators_)
        )
        if mean:
            scores = [np.mean(scores, axis=0)]
        return np.asarray([scores])

    def held_out_score(self, X, y, score_funcs=None, thresholds=None,
                       mean=False):
        """
        Prodivde the cross-validation scores on a held out dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_funcs: function
            Method with signature score_func(y, y_pred)
        :param thresholds: float, optional
            If using probability predictons, the threshod represents the
            positive prediction cut-off.
        :param mean: booleam, optional
            Return the averaged result accross the folds.

        :return: array-like, shape (a, b, c, d)
            a: n_iterations=1
            b: `self.n_splits` or 1 if mean is True
            c: number of score functions passed in `score_funcs`
            d: number of classes for multi-label, multiclass `y`
        """
        check_is_fitted(self, 'fitted_')
        if not score_funcs:
            score_funcs = [accuracy_score]
        if not thresholds:
            thresholds = [0.5]
        score = delayed(self._score_single)
        scores = Parallel(n_jobs=self.n_jobs_, verbose=self.verbose_,
                          backend='multiprocessing')(
            score(X, y, None, estimator, score_funcs, thresholds)
            for estimator in self.estimators_
        )
        if mean:
            scores = [np.mean(scores, axis=0)]
        return np.asarray([scores])


# -------------------------------- TESTS ------------------------------------ #
def test_kfold():
    from sklearn.datasets import make_multilabel_classification
    from .sampling import IterativeStratifiedKFold
    from ..models.binary_relevance import BinaryRelevance
    from sklearn.linear_model import LogisticRegression
    from .scoring import MultilabelScorer
    from sklearn.metrics import f1_score

    scorer = MultilabelScorer(f1_score)
    X, y = make_multilabel_classification()
    est = [LogisticRegression() for _ in range(5)]
    est = BinaryRelevance(est)
    kf = KFoldExperiment(est, cv=IterativeStratifiedKFold(), n_jobs=3,
                         verbose=True)
    kf.fit(X, y)
    val = kf.validation_score(X, y, [scorer, scorer], [0.5, 0.5], True)
    hol = kf.held_out_score(X, y, [scorer, scorer], [0.5, 0.5], False)
    return kf, X, y, scorer, val, hol


def test_boots():
    from sklearn.datasets import make_multilabel_classification
    from .sampling import IterativeStratifiedKFold
    from ..models.binary_relevance import BinaryRelevance
    from sklearn.linear_model import LogisticRegression
    from .scoring import MultilabelScorer
    from sklearn.metrics import f1_score

    scorer = MultilabelScorer(f1_score)
    X, y = make_multilabel_classification()
    est = [LogisticRegression() for _ in range(5)]
    est = BinaryRelevance(est)
    kf = KFoldExperiment(est, cv=IterativeStratifiedKFold())
    bs = Bootstrap(kf, n_jobs=3, n_iter=3, verbose=True)
    bs.fit(X, y)
    val = bs.validation_scores(X, y, [scorer, scorer], [0.5, 0.8])
    hol = bs.held_out_scores(X, y, [scorer, scorer], [0.5, 0.8], True, False)
    return bs, X, y, scorer, val, hol
