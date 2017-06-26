#!/usr/bin/env python

"""
This module contains functions to construct a binary relevance classifier
using the OVR estimaor in sklearn
"""

import array
import numpy as np
import scipy.sparse as sp
from operator import itemgetter

from sklearn.multiclass import _partial_fit_binary,_predict_binary, _fit_binary
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MetaEstimatorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call

from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


class BinaryRelevance(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """BinaryRelevance multilabel strategy

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.

    This strategy can also be used for multilabel learning, where a classifier
    is used to predict multiple labels for instance, by fitting on a 2-d matrix
    in which cell [i, j] is 1 if sample i has label j and 0 otherwise.

    In the multilabel learning literature, OvR is also known as the binary
    relevance method.

    Read more in the :ref:`User Guide <ovr_classification>`.

    Parameters
    ----------
    estimators : array-like, shape = (n_classes, )
        Estimator objects implementing `fit` and one of `decision_function`
        or `predict_proba`.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    Attributes
    ----------
    estimators : array-like, shape = (n_classes, )
        Estimators used for prediction, satisfying above definition.

    classes_ : array, shape = (n_classes, )
        Class labels.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.

    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.
    """

    def __init__(self, estimators, n_jobs=1, backend='threading'):
        self.estimators = estimators
        self.n_jobs = n_jobs
        self._backend = backend

    def __str__(self):
        return "BinaryRelevance(estimators={}, n_jobs={})".format(
            self.estimators, self.n_jobs)

    def set_params(self, **params):
        return super(BinaryRelevance, self).set_params(**params)

    def get_params(self, deep=True):
        params = super(BinaryRelevance, self).get_params(deep=deep)
        if deep:
            params['estimators'] = [clone(e) for e in self.estimators]
        return params

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        if len(self.estimators) != len(self.classes_):
            raise ValueError("Number of estimators does not match the number"
                             "of classes present in `y`.")
        columns = (col.toarray().ravel() for col in Y.T)

        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.

        self.estimators = Parallel(n_jobs=self.n_jobs, backend=self._backend)(
            delayed(_fit_binary)(
                e, X, column, classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i]
                ]
            )
            for (i, column), e in zip(enumerate(columns), self.estimators)
        )
        self.fitted_ = True
        return self

    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iteration.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self
        """
        if _check_partial_fit_first_call(self, classes):
            for e in self.estimators:
                if not hasattr(e, "partial_fit"):
                    raise ValueError(("Base estimator {0}, doesn't have "
                                      "partial_fit method").format(e))

            assert len(self.estimators) == self.n_classes_
            self.estimators = [clone(e) for e in self.estimators]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory
            # consumption in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        if np.setdiff1d(y, self.classes_):
            raise ValueError(("Mini-batch contains {0} while classes " +
                             "must be subset of {1}").format(np.unique(y),
                                                             self.classes_))
        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators = Parallel(n_jobs=self.n_jobs, backend=self._backend)(
            delayed(_partial_fit_binary)(self.estimators[i], X, next(columns))
            for i in range(self.n_classes_))

        self.fitted_ = True
        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.
        """
        check_is_fitted(self, 'fitted_')
        if (hasattr(self.estimators[0], "decision_function") and
                is_classifier(self.estimators[0])):
            thresh = 0
        else:
            thresh = .5

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators)))
            return self.label_binarizer_.inverse_transform(indicator)

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.

        In the single label multiclass case, the rows of the returned matrix
        sum to 1.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : (sparse) array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, 'fitted_')
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators]).T

        if len(self.estimators) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # Then, probabilities should be normalized to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
        return Y

    def decision_function(self, X):
        """Returns the distance of each sample from the decision boundary for
        each class. This can only be used with estimators which implement the
        decision_function method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'fitted_')
        if not hasattr(self.estimators[0], "decision_function"):
            raise AttributeError(
                "Base estimator doesn't have a decision_function attribute.")
        return np.array([est.decision_function(X).ravel()
                         for est in self.estimators]).T

    def score(self, X, y, sample_weight=None):
        raise AttributeError("BinaryRelevance has no support for multi-label "
                             "scoring yet. Use the metrics module.")

    def get_classes(self):
        return self.classes_

    def is_multilabel(self):
        return self.multilabel_

    def get_label_binarizer(self):
        return self.label_binarizer_

    def get_estimators(self):
        if isinstance(self.estimators[-1], RandomizedSearchCV):
            return [e.best_estimator_ for e in self.estimators]
        if isinstance(self.estimators[-1], GridSearchCV):
            return [e.best_estimator_ for e in self.estimators]
        elif isinstance(self.estimators[-1], CalibratedClassifierCV):
            return [e.calibrated_classifiers_ for e in self.estimators]
        else:
            return self.estimators

    def zip_classes(self):
        estimators = self.get_estimators()
        classes = self.get_classes()
        return list(zip(classes, estimators))

    def top_n_features(self, n, absolute=False):
        """
        Return the top N features. If estimator is a pipeline, then it assumes
        the first step is the vectoriser holding the feature names.

        :return: array like, shape (n_estimators, n).
            Each element in a list is a tuple (feature_idx, weight).
        """
        check_is_fitted(self, 'fitted_')
        top_features = []
        for e, coef in zip(self.get_estimators(), self.coef_):
            vectorizer = None
            if absolute:
                coef = abs(coef)
            if hasattr(e, 'steps'):
                vectorizer = e.steps[0][-1]
            idx_coefs = sorted(enumerate(coef),
                               key=itemgetter(1), reverse=True)[:n]
            if vectorizer:
                idx = [idx for (idx, w) in idx_coefs]
                ws = [w for (idx, w) in idx_coefs]
                print(vectorizer.get_feature_names())
                features = np.asarray(vectorizer.get_feature_names())[idx]
                top_features.append(list(zip(features, ws)))
            else:
                top_features.append(idx_coefs)
        return top_features

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier"""
        return self.label_binarizer_.y_type_.startswith('multilabel')

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def coef_(self):
        """
        Return the feature weightings for each estimator. If estimator is a
        pipeline, then it assumes the last step is the estimator.
 
        :return: array-like, shape (n_classes_, n_features)
        """
        check_is_fitted(self, 'fitted_')

        def feature_imp(estimator):
            if hasattr(estimator, 'steps'):
                estimator = estimator.steps[-1][-1]
            if hasattr(estimator, "coef_"):
                return estimator.coef_
            elif hasattr(estimator, "coefs_"):
                return estimator.coefs_
            elif hasattr(estimator, "feature_importances_"):
                return estimator.feature_importances_
            else:
                raise AttributeError(
                    "Estimator {} doesn't support "
                    "feature coefficients.".format(type(estimator)))

        coefs = [feature_imp(e) for e in self.get_estimators()]
        if sp.issparse(coefs[0]):
            return sp.vstack(coefs)
        return np.vstack(coefs)

    @property
    def intercept_(self):
        check_is_fitted(self, 'fitted_')
        if not hasattr(self.estimators[0], "intercept_"):
            raise AttributeError(
                "Base estimator doesn't have an intercept_ attribute.")
        return np.array([e.intercept_.ravel() for e in self.estimators])


# -------------------------- Skmulti-learn Utilities ----------------------- #
def get_coefs(clf):
    """
    Return the feature weightings for each estimator. If estimator is a
    pipeline, then it assumes the last step is the estimator.

    :return: array-like, shape (n_classes_, n_features)
    """
    check_is_fitted(clf, 'fitted_')

    def feature_imp(estimator):
        if hasattr(estimator, 'steps'):
            estimator = estimator.steps[-1][-1]
        if hasattr(estimator, "coef_"):
            return estimator.coef_
        elif hasattr(estimator, "coefs_"):
            return estimator.coefs_
        elif hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        else:
            raise AttributeError(
                "Estimator {} doesn't support "
                "feature coefficients.".format(type(estimator)))

    coefs = [feature_imp(e) for e in get_br_estimators(clf)]
    if sp.issparse(coefs[0]):
        return sp.vstack(coefs)
    return np.vstack(coefs)


def zip_classes(clf):
    """
    Zip the class indices with the classifiers
    """
    if not isinstance(clf, BinaryRelevance):
        raise TypeError("Need a Skmultilearn BR classifier.")
    estimators = clf.estimators
    classes = clf.partition
    return list(zip(classes, estimators))


def get_br_estimators(clf):
    """
    Get the classifiers from a multi-label Binary Relevance classifier.
    """
    from skmultilearn.problem_transform.br import BinaryRelevance
    if not isinstance(clf, BinaryRelevance):
        raise TypeError("Need a Skmultilearn BR classifier.")
    br_classifers = clf.classifiers
    if isinstance(br_classifers[-1], RandomizedSearchCV):
        return [e.best_estimator_ for e in br_classifers]
    if isinstance(br_classifers[-1], GridSearchCV):
        return [e.best_estimator_ for e in br_classifers]
    elif isinstance(br_classifers[-1], CalibratedClassifierCV):
        return [e.calibrated_classifiers_ for e in br_classifers]
    else:
        return br_classifers


def top_n_features(clf, n, absolute=False, vectorizer=None):
    """
    Return the top N features. If estimator is a pipeline, then it assumes
    the first step is the vectoriser holding the feature names.

    :return: array like, shape (n_estimators, n).
        Each element in a list is a tuple (feature_idx, weight).
    """
    check_is_fitted(clf, 'fitted_')
    top_features = []
    estimators = get_br_estimators(clf)
    coefs = get_coefs(clf)
    for e, coef in zip(estimators, coefs):
        if absolute:
            coef = abs(coef)
        if hasattr(e, 'steps') and vectorizer is None:
            vectorizer = e.steps[0][-1]
        idx_coefs = sorted(
            enumerate(coef), key=itemgetter(1), reverse=True)[:n]
        if vectorizer:
            idx = [idx for (idx, w) in idx_coefs]
            ws = [w for (idx, w) in idx_coefs]
            print(vectorizer.get_feature_names())
            features = np.asarray(vectorizer.get_feature_names())[idx]
            top_features.append(list(zip(features, ws)))
        else:
            top_features.append(idx_coefs)
    return top_features

