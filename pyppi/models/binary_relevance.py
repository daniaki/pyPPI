"""
A classifier implementing the Binary Relevance approach to multi-label
learning.
"""

__all__ = [
    "MixedBinaryRelevanceClassifier"
]

import logging
import numpy as np
from joblib import Parallel, delayed

from sklearn.base import clone, BaseEstimator
from sklearn.metrics import (
    hamming_loss, label_ranking_loss, f1_score,
    precision_score, recall_score
)

from pyppi.model_selection.scoring import fdr_score, specificity


logger = logging.getLogger("pyppi")


def _fit_label(estimator, X, y, label_idx, n_labels, verbose, **fit_params):
    if verbose:
        logger.info("Fitting label {}/{}.".format(label_idx+1, n_labels))
    return estimator.fit(X, y, **fit_params)


def _predict_proba_label(estimator, X):
    return estimator.predict_proba(X)


def _predict_label(estimator, X):
    return estimator.predict(X)


class MixedBinaryRelevanceClassifier(object):
    """Mimics the `OneVsRest` classifier from Sklearn allowing
    a different type of classifier for each label as opposed to one classifier
    for all labels.

    Parameters:
    ----------
    estimators : `list`
        List of `Scikit-Learn` estimators supporting `fit`, `predict` and
        `predict_proba`.

    n_jobs : int, optional, default: 1
        Number of processes to use when fitting each label.

    verbose : bool, optional, default: False
        Logs messages regarding fitting progress.
    """

    def __init__(self, estimators, n_jobs=1, verbose=False):
        if not isinstance(estimators, list):
            raise TypeError("estimators must be a list.")
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __repr__(self):
        return (
            "MixedBinaryRelevanceClassifier(estimators={}, n_jobs={})".format(
                self.estimators, self.n_jobs
            )
        )

    def _check_y_shape(self, y):
        try:
            if y.shape[1] <= 1:
                raise ValueError(
                    "y must be in multi-label indicator matrix format. "
                    "For binary or multi-class classification use scikit-learn."
                )
            if y.shape[1] != len(self.estimators):
                raise ValueError(
                    "Shape of y {} along dim 1 does not match {}.".format(
                        y.shape, len(self.estimators)
                    )
                )
        except IndexError:
            raise ValueError(
                "y must be in multi-label indicator matrix format. "
                "For binary or multi-class classification use scikit-learn."
            )

    def clone(self, deep=True):
        params = self.get_params(deep)
        return self.__class__(**params)

    def _check_fitted(self):
        if not hasattr(self, 'estimators_'):
            raise ValueError("This estimator has not yet been fit.")
        if not hasattr(self, 'n_labels_'):
            raise ValueError("This estimator has not yet been fit.")

    def get_params(self, deep=True):
        return {
            "estimators": [clone(e) for e in self.estimators],
            "n_jobs": self.n_jobs,
            "verbose": self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self.get_params().keys():
                raise ValueError(
                    "'{}' is not a valid param for {}.".format(
                        key, self.__class__.__name__
                    )
                )
            elif key == 'estimators':
                if not isinstance(value, list):
                    raise TypeError("'estimators' must be a list.")

                self.estimators = [clone(e) for e in value]
                if hasattr(self, 'n_labels_'):
                    delattr(self, 'n_labels_')
                if hasattr(self, 'estimators_'):
                    delattr(self, 'estimators_')
            else:
                setattr(self, key, value)
        return self

    def fit(self, X, y, **fit_params):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples, n_labels)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        self._check_y_shape(y)
        n_labels = len(self.estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_label)(
                estimator=clone(estimator),
                X=X, y=y[:, i], label_idx=i,
                n_labels=n_labels, verbose=self.verbose,
                **fit_params
            )
            for i, estimator in enumerate(self.estimators)
        )
        self.n_labels_ = len(self.estimators_)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples, n_labels)
            Predicted class labels per sample.
        """
        self._check_fitted()
        predictions = np.vstack(Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_label)(
                estimator=estimator, X=X
            )
            for estimator in self.estimators_
        )).T
        return predictions

    def predict_proba(self, X):
        """
        Probability estimates for each label.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        T : array-like, shape = (n_samples, n_labels)
            Returns the probability of the sample for each label in the model,
            where labels are ordered as the indices of 'y' used during fit.
        """
        self._check_fitted()
        probas = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_label)(
                estimator=estimator, X=X
            )
            for estimator in self.estimators_
        )
        probas = np.vstack([x[:, 1] for x in probas]).T
        return probas

    def score(self, X, y, sample_weight=None, use_proba=False,
              scorer=hamming_loss, **score_params):
        """
        Returns the score as determined by `scoring` on the given
        test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples, n_labels)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        use_proba : boolean, default: False
            If True, apply scoring function to probability estimates.

        scorer : function, optional
            The scoring method to apply to predictions.

        score_params : dict, optional
            Keyword arguments for scorer.

        Returns
        -------
        `float` or array-like (n_labels, ) if scoring uses binary.
            Mean score of self.predict(X) wrt. y.
        """
        self._check_y_shape(y)
        self._check_fitted()

        if use_proba:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)

        average = score_params.get("average", None)
        if average == "binary":
            return np.asarray([
                scorer(
                    y[:, i], y_pred[:, i],
                    sample_weight=sample_weight,
                    **score_params
                )
                for i in range(self.n_labels_)
            ])
        else:
            return scorer(
                y, y_pred, sample_weight=sample_weight,
                **score_params
            )
