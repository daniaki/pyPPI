#!/usr/bin/env python

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from model_selection.sampling import IterativeStratifiedKFold


class KFoldExperiment(object):
    """
    This is a class that performs KFold cross-validation to
    collect numer

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    vectorizer : vectorizer object
        A vectorizer object implementing `fit`, `fit_transform` and
        `transform`.

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

    vectorizers_ : list of `n_classes` vectorisers
        Vectorisers used for predictions on each fold.
    """

    def __init__(self, estimator, vectorizer=None, cv=None,
                 shuffle=False, random_state=None):
        self.base_estimator_ = clone(estimator)
        if vectorizer:
            self.base_vectorizer_ = clone(vectorizer)
        else:
            self.base_vectorizer_ = None

        self.estimators_ = []
        self.vectorizers_ = []
        self.fitted_ = False

        if not cv:
            self.cv_ = StratifiedKFold(3, shuffle, random_state)
        elif isinstance(cv, int):
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

    def fit(self, X, y):
        """
        Fit the estimators for each fold using the supplied data.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :return: self.
        """
        if hasattr(self.cv_, 'split'):
            self.cv_ = list(self.cv_.split(X, y))

        for train_idx, _ in self.cv_:
            print(train_idx)
            print(X.shape)
            X_f = X[train_idx, ]
            y_f = y[train_idx, ]
            if self.base_vectorizer_ is not None:
                vectorizer = clone(self.base_vectorizer_)
                vectorizer.fit(X_f)
                X_f = vectorizer.transform(X_f)
                self.vectorizers_.append(vectorizer)
            else:
                self.vectorizers_.append(self.base_vectorizer_)

            estimator = clone(self.base_estimator_)
            estimator.fit(X_f, y_f)
            self.estimators_.append(estimator)

        self.fitted_ = True
        return self

    def validation_score(self, X, y, score_func, threshold=None):
        """
        Prodivde the cross-validation scores on a the validation dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_func: Method with signature score_func(y, y_pred)
        :param threshold: If using probability predictons, the threshod
               represents the positive prediction cut-off.
        :return: List of scores as returned by score_func for each fold.
        """
        if not self.fitted_:
            raise ValueError("Estimators have not been fit yet.")

        scores = []
        for (_, test_idx), estimator, vectorizer in \
                zip(self.cv_, self.estimators_, self.vectorizers_):
            X_f = X[test_idx, ]
            y_f = y[test_idx, ]
            if vectorizer:
                X_f = vectorizer.transform(X_f)

            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba(X_f)
                if threshold:
                    y_pred = (y_pred >= threshold).astype(int)
            else:
                print("Warning: {} does not have a `predict_proba` "
                      "implementation. Using `predict` "
                      "instead.".format(type(self.base_estimator_)))
                y_pred = estimator.predict(X_f)

            score = score_func(y_f, y_pred)
            scores.append(score)

        return scores

    def held_out_score(self, X, y, score_func, threshold=None):
        """
        Prodivde the cross-validation scores on a held out dataset.

        :param X: numpy array X, shape (n_samples, n_features)
        :param y: numpy array y, shape (n_samples, n_outputs)
        :param score_func: Method with signature score_func(y, y_pred)
        :param threshold: If using probability predictons, the threshod
               represents the positive prediction cut-off.
        :return: List of scores as returned by score_func for each fold.
        """
        if not self.fitted_:
            raise ValueError("Estimators have not been fit yet.")

        scores = []
        for estimator, vectorizer in zip(self.estimators_, self.vectorizers_):
            if vectorizer:
                X = vectorizer.transform(X)

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

        return scores





