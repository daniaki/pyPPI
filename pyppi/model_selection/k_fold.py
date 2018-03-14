import logging
import numpy as np

from joblib import Parallel, delayed
from numpy.random import RandomState

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from ..models.binary_relevance import MixedBinaryRelevanceClassifier

from .sampling import IterativeStratifiedKFold

__all__ = [
    'StratifiedKFoldCrossValidation'
]

logger = logging.getLogger("pyppi")


def _clone(estimator):
    if isinstance(estimator, MixedBinaryRelevanceClassifier):
        return estimator.clone()
    else:
        return clone(estimator)


def _fit_fold(estimator, X, y, train_idx, fold_idx, verbose, vectorizer,
              **fit_params):
    if verbose:
        logger.info("Fitting fold %d." % (fold_idx + 1))
    X = X[train_idx]
    y = y[train_idx]
    if vectorizer is not None:
        X = vectorizer.fit_transform(X)
    return estimator.fit(X, y, **fit_params)


def _score_fold(estimator, X, y, validation_idx, fold_idx, verbose,
                sample_weight, vectorizer, **score_params):
    if validation_idx is not None:
        X = X[validation_idx]
        y = y[validation_idx]
    if vectorizer is not None:
        X = vectorizer.transform(X)
    return estimator.score(X, y, sample_weight, **score_params)


class StratifiedKFoldCrossValidation(object):
    """StratifiedKFoldCrossValidation classifier.

    This is a 'meta-classifier' object which fits an estimator for
    each fold parition as generated from `n_folds`.

    Parameters
    ----------
    estimator : object
        An esimtator object supporting `fit` and `score` methods.

    n_folds : int, default: 5
        The number of folds to split the training data into.

    shuffle : boolean, default: False
        Shuffle data prior to fold

    multilabel : boolean, default: False
        Used to indicate if the classifier accepts multi-label y input.
        If `True`, the splits will be generated using `IterativeStratifiedKFold`.

    n_jobs : int, default: 1
        Number of jobs used to train folds in parallel.

    random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling
        the data and generating folds. If int, random_state is the seed used
        by the random number generator; If RandomState instance, random_state
        is the random number generator; If None, the random number generator
        is the RandomState instance used by `np.random`.

    Attributes
    ----------
    fold_estimators_ : array, shape (n_folds,)
        The estimators fit using each fold parititoning.

    multilabel_ : boolean
        Set to True if estimator is fit on a multi-label indicator format 
        array.

    cv_ : array-like, shape (n_folds, )
        The fold indicies generated.

    vectorizer_ : VectorizerMixin object
        If a vectorizer was passed into the `fit` call, then this attribute
        will hold that vectorizer to be used on for future calls
        to `score`. Otherwise it is None.

    """

    def __init__(self, estimator, n_folds=5, shuffle=False, n_jobs=1,
                 random_state=None, verbose=False):
        self.estimator = estimator
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def __repr__(self):
        return (
            "StratifiedKFoldCrossValidation(estimator={}, n_folds={}, "
            "shuffle={}, n_jobs={}, random_state={} verbose={})".format(
                self.estimator, self.n_folds, self.shuffle,
                self.n_jobs, self.random_state, self.verbose
            )
        )

    def _check_fitted(self):
        if not hasattr(self, 'fold_estimators_'):
            raise ValueError("This classifier has not yet been fit.")
        if not hasattr(self, 'cv_'):
            raise ValueError("This classifier has not yet been fit.")
        if not hasattr(self, 'multilabel_'):
            raise ValueError("This classifier has not yet been fit.")
        if not hasattr(self, 'vectorizers_'):
            raise ValueError("This classifier has not yet been fit.")

    def _check_n_folds(self, n_folds):
        if not isinstance(n_folds, int):
            raise TypeError("n_folds must be an integer.")
        if n_folds <= 0:
            raise ValueError("n_folds must be > 1.")
        return int(n_folds)

    def _input_is_multilabel(self, y):
        try:
            return y.shape[1] >= 1
        except IndexError:
            return False

    def _generate_splits(self, X, y):
        y = np.asarray(y)
        self.multilabel_ = self._input_is_multilabel(y)
        if self.multilabel_:
            self.cv_ = list(IterativeStratifiedKFold(
                n_splits=self.n_folds, shuffle=self.shuffle,
                random_state=self.random_state
            ).split(X, y))
        else:
            self.cv_ = list(StratifiedKFold(
                n_splits=self.n_folds, shuffle=self.shuffle,
                random_state=self.random_state
            ).split(X, y))
        return self

    def fit(self, X, y, vectorizer=None, **fit_params):
        """
        Fit an estimator for each fold.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples, n_labels)
            Target vector relative to X.

        fit_params :  dict
            Fit function parameters which will be passed directly to the 
            the cloned base_estimator for each fold.

        Returns
        -------
        self : object
            Returns self.
        """

        self._generate_splits(X, y)
        if vectorizer is not None:
            self.vectorizers_ = [clone(vectorizer)
                                 for _ in range(self.n_folds)]
        else:
            self.vectorizers_ = [None for _ in range(self.n_folds)]

        self.fold_estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_fold)(
                estimator=_clone(self.estimator), X=X, y=y,
                train_idx=train_idx, fold_idx=i, verbose=self.verbose,
                vectorizer=self.vectorizers_[i], **fit_params
            )
            for i, (train_idx, _) in enumerate(self.cv_)
        )
        return self

    def score(self, X, y, validation=False, avg_folds=True,
              sample_weight=None, **score_params):
        """
        Runs the `score` method of each fold estimator on X and y.
        If X and y are the training inputs and `validation` is True then
        only the validation subsets are scored for each fold.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples, n_labels)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        avg_folds : boolean, default: False
            If True, all folds are averaged to give a single score and
            standard error for each label.

        validation : boolean, default: False
            If True, assumes the X and y are the same as the arrays used
            during training and applies the scoring function over the
            validation fold subset of X and y. Otherwise scoring is
            performed on all samples in X and y.

        score_params : dict, optional
            Keyword arguments for `score` function in the supplied
            `base_estimator`.

        Returns
        -------
        C : tuple or array-like
            If `avg_folds` is True then two arrays (n_labels,) for the mean
            score and standard error will be returned. Otherwise a single
            array (n_labels, n_folds) will be returned.
        """

        self._check_fitted()
        scores = np.asarray(Parallel(n_jobs=self.n_jobs)(
            delayed(_score_fold)(
                estimator=fold_estimator, X=X, y=y,
                sample_weight=sample_weight,
                validation_idx=valid_idx if validation else None,
                fold_idx=i,
                verbose=self.verbose,
                vectorizer=self.vectorizers_[i],
                ** score_params
            )
            for i, (fold_estimator, (_, valid_idx)) in
            enumerate(zip(self.fold_estimators_, self.cv_))
        ))
        if avg_folds:
            if self._input_is_multilabel(scores):
                score = np.mean(scores, axis=0)
                stderr = np.std(scores, axis=0) / np.sqrt(self.n_folds)
            else:
                score = np.mean(scores)
                stderr = np.std(scores) / np.sqrt(self.n_folds)
            return score, stderr
        else:
            return scores.T
