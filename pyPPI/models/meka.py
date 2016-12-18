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

    def predict_proba(self, X, y=None, method='sigmoid', cv=3):
        calibrated = CalibratedClassifierCV(self, method, cv)
        calibrated.fit(X, y)
        return calibrated.predict_proba(X)
