#!/usr/bin/env python

"""
This module contains class and method definitions related to extracting
features from PPIs, including feature induction as per Maestechze et al., 2011
"""

import goatools
import numpy as np
import pandas as pd

from data import PPI
from data import load_go_dag

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class AnnotationExtractor(BaseEstimator, TransformerMixin):
    """
    Convert a collection of leaf Gene Ontology annotations to a set of
    Gene Ontology annotations including additional terms up to the lowest
    common ancestor.

    Parameters
    ----------
    annotation_func : function
        Function with the signature func(PPI, selection) that takes a PPI and
         returns a dictionary for the feature databases in selection for the
         PPI

    Attributes
    ----------
    vocabulary_ : dict
        A pd.DataFrame of PPIs to to textual features.

    Notes
    -----
    """

    def __init__(self, annotation_func):
        self._annotation_func = annotation_func

    def __update(self, item):
        pass

    def fit(self, raw_documents, y=None):
        """
        Given a list of protein UniProt accessions, builds a dictionary of
        induced GO annotations for each.

        Parameters
        ----------
        :param raw_documents : iterable
            An iterable which yields PPI objects.

        :return:
            self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Given a list of protein UniProt accessions, builds a dataframe of
        features

        Parameters
        ----------
        :param X: raw_documents: iterable
            An iterable which yields PPI objects.

        :param y:
            None

        :param fit_params: parameters to pass to the fit_transform method

        :return: X :  array-like, shape (n_samples, )
            List of induced GO terms for each PPI sample.
        """
        default = ['go_cc', 'go_bp', 'go_mf', 'pfam', 'ipr']
        selection = fit_params.get('selection', default=default)

        self._vocabulary = pd.DataFrame()
        return self


    def transform(self, raw_documents):
        """
        Transform a list of PPIs to a list of GO annotations
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        check_is_fitted(self, '_vocabulary')
        return self
