#!/usr/bin/env python

"""
This module contains functions to construct a binary relevance classifier
using the OVR estimaor in sklearn
"""

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV


class BinaryRelevance(OneVsRestClassifier):
    """
    Wrapper over the OVR classifier with some helper functions
    """

    def __init__(self, *args, **kwargs):
        super(OneVsRestClassifier, self).__init__(*args, **kwargs)

    def classes(self):
        return self.classes_

    def multilabel(self):
        return self.multilabel_

    def label_binarizer(self):
        return self.label_binarizer_

    def estimators(self):
        if isinstance(self.estimators_[-1], RandomizedSearchCV):
            return [e.best_estimator_ for e in self.estimators_]
        if isinstance(self.estimators_[-1], GridSearchCV):
            return [e.best_estimator_ for e in self.estimators_]
        elif isinstance(self.estimators_[-1], CalibratedClassifierCV):
            return [e.calibrated_classifiers_ for e in self.estimators_]
        else:
            return self.estimators_

    def zip_classes(self):
        estimators = self.estimators()
        classes = self.classes()
        return zip(classes, estimators)
