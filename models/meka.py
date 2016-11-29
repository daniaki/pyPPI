#!/usr/bin/env python

"""
This module contains functions to construct an interface to the MEKA
wrapper provided by sklearn-multilearn
"""

from skmultilearn.ext import Meka
from sklearn.calibration import CalibratedClassifierCV


class MekaClassifier(Meka):
    """
    Wrapper over Meka Object with some additional helper functions
    """

    def __init__(self, *args, **kwargs):
        super(Meka, self).__init__(*args, **kwargs)

    def predict_proba(self, X, method='sigmoid', cv=3):
        calibratedCV = CalibratedClassifierCV(self, method, cv)
        calibratedCV.fit(X)
        return calibratedCV.predict_proba(X)
