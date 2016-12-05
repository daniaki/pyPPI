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

    def __init__(self, estimator, cv=3, shuffle=True, random_state=None):
        self.base_estimator_ = clone(estimator)
        self.estimators_ = []
        self.fitted_ = False

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
        if hasattr(self.cv_, 'split'):
            self.cv_ = list(self.cv_.split(X, y))

        scores = []
        for (train_idx, test_idx), estimator in \
                zip(self.cv_, self.estimators_):
            X_train = X[train_idx, ]
            y_train = y[train_idx, ]
            X_test = X[test_idx, ]
            y_test = y[test_idx, ]

            estimator = clone(self.base_estimator_)
            estimator.fit(X_train, y_train)
            self.estimators_.append(estimator)

            if hasattr(estimator, 'predict_proba'):
                y_pred = estimator.predict_proba(X_test)
                if threshold:
                    y_pred = (y_pred >= threshold).astype(int)
            else:
                print("Warning: {} does not have a `predict_proba` "
                      "implementation. Using `predict` "
                      "instead.".format(type(self.base_estimator_)))
                y_pred = estimator.predict(X_test)

            score = score_func(y_test, y_pred)
            scores.append(score)

        self.fitted_ = True
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

        return scores
