"""
A classifier implementing that performs chained learning averaged over
K distinct orderings of label chains.
"""

__all__ = [
    "KRandomClassifierChains"
]

from joblib import Parallel, delayed
import numpy as np
from numpy.random import RandomState

from ..base.constants import MAX_SEED

from sklearn.base import clone
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import hamming_loss

MAX_INT = MAX_SEED


def _fit_chain(estimator, X, y, **fit_params):
    return estimator.fit(X, y, **fit_params)


def _predict_proba_chain(estimator, X):
    return estimator.predict_proba(X)


class KRandomClassifierChains(object):
    """A meta-estimator which fits `k` random ordered Classifier
    chains and soft-averages prediction probabilities over all chains
    using an un-weighted mean.

    Parameters
    ----------
    base_esimtaors : estimator
        A `Scikit-Learn` estimator supporting `fit`, `predict` and
        `predict_proba`.

    k : int, optional, default: 10
        Number of random chains to train.

    cv : int, cross-validation generator or an iterable, optional, default: None
        Determines whether to use cross validated predictions or true labels 
        for the results of previous estimators in the chain. See link below for 
        further details.

    n_jobs : int, optional, default: 1
        Number of processes to use when fitting each label.

    random_state :  int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used 
        by np.random.

        The random number generator is used to generate random chain orders.
        See link below for further details.

    See Also
    --------
    `Classifier Chain <http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain>`_
    """

    def __init__(self, base_estimator, k=10, cv=None, n_jobs=1,
                 random_state=None):
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.cv = cv
        self.n_jobs = n_jobs

        self.rng_ = self._check_random_state(random_state)
        self.k_ = self._check_k(k)
        self.fitted_ = False
        self._set_estimators_reset_fitted()

    def __repr__(self):
        return (
            "KRandomClassifierChains(base_estimator={}, k={}, cv={}, "
            "random_state={})".format(
                self.base_estimator, self.k, self.cv, self.random_state
            )
        )

    def get_params(self, deep=True):
        return {
            "base_estimator": self.base_estimator,
            "k": self.k,
            "cv": self.cv,
            'n_jobs': self.n_jobs,
            "random_state": self.random_state,
        }

    def _set_random_state_of_estimators(self):
        for e in self.estimators_:
            if self.rng_ is None:
                state = None
            else:
                state = self.rng_.randint(MAX_INT)
            e.set_params(**{'random_state': state})

    def set_params(self, **params):
        for key, value in params.items():
            if "__" in key:
                for estimator in self.estimators_:
                    estimator.set_params(**{key: value})
            else:
                if key in self.get_params().keys():
                    if key == 'random_state':
                        setattr(self, 'rng_', self._check_random_state(value))
                        setattr(self, 'random_state', value)
                        self._set_random_state_of_estimators()

                    elif key == "k":
                        self.k = value  # this will also create new estimators

                    elif key == 'base_estimator':
                        self.base_estimator = clone(value)
                        self._set_estimators_reset_fitted()
                    else:
                        setattr(self, key, value)
                else:
                    raise AttributeError(
                        "Invalid param '{}' for class {}.".format(
                            key, self.__class__.__name__
                        )
                    )
        return self

    @property
    def k(self):
        return self.k_

    @k.setter
    def k(self, value):
        self.k_ = self._check_k(value)
        self._set_estimators_reset_fitted()

    def _set_estimators_reset_fitted(self):
        self.estimators_ = [
            ClassifierChain(
                clone(self.base_estimator), order="random",
                cv=self.cv, random_state=None
            )
            for _ in range(self.k_)
        ]
        self._set_random_state_of_estimators()
        self.fitted_ = False

    def _check_fitted(self):
        if not self.fitted_:
            raise ValueError("Please run fit method first.")

    def _check_k(self, k):
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        return int(k)

    def _check_random_state(self, random_state):
        if random_state is None:
            return None
        elif isinstance(random_state, int):
            return RandomState(seed=random_state)
        elif isinstance(random_state, RandomState):
            return random_state
        else:
            raise TypeError("random_state must be None, int or RandomState")

    def _check_y_shape(self, y):
        try:
            if y.shape[1] <= 1:
                raise ValueError(
                    "y must be in multi-label indicator matrix format. "
                    "For binary or multi-class classification use scikit-learn."
                )
        except IndexError:
            raise ValueError(
                "y must be in multi-label indicator matrix format. "
                "For binary or multi-class classification use scikit-learn."
            )

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
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_chain)(estimator, X, y, **fit_params)
            for estimator in self.estimators_
        )
        self.fitted_ = True
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
        predictions = (self.predict_proba(X) >= 0.5).astype(int)
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
        probas_ls = np.asarray(Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_proba_chain)(estimator, X)
            for estimator in self.estimators_
        ))
        probas = probas_ls.mean(axis=0)
        return probas

    def score(self, X, y, sample_weight=None, use_proba=False,
              scorer=hamming_loss, **scorer_kwargs):
        """
        Returns the score as determined by `scoring` on the given
        test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

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

        scorer_kwargs : dict, optional
            Keyword arguments for scorer.

        Returns
        -------
        `float`
            Mean score of self.predict(X) wrt. y.
        """
        self._check_fitted()
        self._check_y_shape(y)

        if use_proba:
            y_pred = self.predict_proba(X)
        else:
            y_pred = self.predict(X)

        average = scorer_kwargs.get("average", None)
        if average == "binary":
            return np.asarray([
                scorer(
                    y[:, i], y_pred[:, i],
                    sample_weight=sample_weight,
                    **scorer_kwargs
                )
                for i in range(y.shape[1])
            ])
        else:
            return scorer(
                y, y_pred, sample_weight=sample_weight,
                **scorer_kwargs
            )
