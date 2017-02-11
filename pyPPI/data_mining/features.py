#!/usr/bin/env python

"""
This module contains class and method definitions related to extracting
features from PPIs, including feature induction as per Maestechze et al., 2011
"""

import pandas as pd
import numpy as np
from itertools import chain

from ..base import PPI, chunk_list
from ..data import load_accession_features, load_ppi_features
from .uniprot import UniProt
from .ontology import get_up_to_lca, group_go_by_ontology

from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Parallel, delayed


class AnnotationExtractor(object):
    """
    Convert a collection of leaf Gene Ontology annotations to a set of
    Gene Ontology annotations including additional terms up to the lowest
    common ancestor.

    Parameters
    ----------

    Attributes
    ----------
    vocabulary_ : dict
        A pd.DataFrame of PPIs to to textual features.

    Notes
    -----
    """

    def __init__(self, uniprot, godag, induce, selection, n_jobs,
                 verbose=False, cache=True):
        self._uniprot = uniprot
        self._godag = godag
        self._n_jobs = n_jobs
        self._selection = selection
        self._induce = induce
        self._cache = cache
        self._verbose = verbose
        if cache:
            try:
                self._accession_df = load_accession_features()
                self._ppi_df = load_ppi_features()
            except IOError:
                print('Warning: No cache files found.')

    def fit(self, X, y=None):
        """
        Given a list of protein UniProt accessions, builds a dictionary of
        induced GO annotations for each.

        Parameters
        ----------
        :param X : iterable
            An iterable which yields PPI objects.

        :param y:
            None

        :return:
            self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """
        Given a list of protein UniProt accessions, builds a dataframe of
        features

        Parameters
        ----------
        :param X: raw_documents: iterable
            An iterable which yields PPI objects.

        :return: X :  array-like, shape (n_samples, )
            List of induced GO terms for each PPI sample.
        """
        data = []
        ppis = [PPI(a, b) for (a, b) in X]
        accessions = set(acc for ppi in ppis for acc in ppi)

        _df = self._uniprot.features_to_dataframe(accessions)
        if hasattr(self, '_accession_df'):
            _df = pd.concat([self._accession_df, _df], ignore_index=True)
            _df = _df.drop_duplicates(subset=UniProt.accession_column())
        self._accession_df = _df

        # Run the intensive computation in parallel and append the local cache
        compute_features = delayed(self._compute_features)
        dfs = Parallel(n_jobs=self._n_jobs, backend='multiprocessing',
                        verbose=self._verbose)(
            compute_features(ppi) for ppi in ppis
        )
        for df in dfs:
            self._update(df)

        for ppi in ppis:
            terms = {}
            entry = self._ppi_df[self._ppi_df.accession.values == ppi]
            for s in self._selection:
                if UniProt.data_types().GO.value in s and self._induce:
                    s = 'i' + s
                terms[s] = entry[s].values
            features = [str(x) for k, v in terms.items() for x in v[0]]
            features = ','.join(features)
            data.append(features)
        return np.asarray(data)

    def transform(self, X):
        """
        Transform a list of PPIs to a list of GO annotations
        Parameters
        ----------
        :param X: raw_documents: iterable
            An iterable which yields PPI objects.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        check_is_fitted(self, '_accession_df')
        check_is_fitted(self, '_ppi_df')
        data = []
        ppis = [PPI(a, b) for (a, b) in X]
        new = [ppi for ppi in ppis if ppi not in
               set(self._ppi_df[UniProt.accession_column()].values)]

        # Get the new accessions, update the _accession_df and then compute
        # features for these only. This will update _ppi_df
        accessions = set(acc for ppi in new for acc in ppi)
        for acc in accessions:
            self._update(acc)

        # Run the intensive computation in parallel and append the local cache
        compute_features = delayed(self._compute_features)
        dfs = Parallel(n_jobs=self._n_jobs, backend='multiprocessing',
                        verbose=self._verbose)(
            compute_features(ppi) for ppi in new
        )
        for df in dfs:
            self._update(df)

        for ppi in ppis:
            terms = {}
            entry = self._ppi_df[self._ppi_df.accession.values == ppi]
            for s in self._selection:
                if UniProt.data_types().GO.value in s and self._induce:
                    s = 'i' + s
                terms[s] = entry[s].values
            features = [str(x) for k, v in terms.items() for x in v[0]]
            features = ','.join(features)
            data.append(features)
        return np.asarray(data)

    def _compute_features(self, ppi):
        terms = {}
        if self._induce:
            print('Inducing GO annoations...')
            induced_cc, induced_bp, induced_mf = \
                self._compute_induced_terms(ppi)
            terms['igo'] = [induced_cc + induced_bp + induced_mf]
            terms['igo_cc'] = [induced_cc]
            terms['igo_bp'] = [induced_bp]
            terms['igo_mf'] = [induced_mf]

        # Concatenate the other terms into strings
        for s in self._selection:
            data = [self._get_for_accession(p, s) for p in ppi]
            if isinstance(data[-1], list):
                data = list(chain.from_iterable(data))
            terms[s] = [data]

        grouped = group_go_by_ontology(terms['go'][0], self._godag)
        terms['go_cc'] = [grouped['cc']]
        terms['go_bp'] = [grouped['bp']]
        terms['go_mf'] = [grouped['mf']]

        terms['accession'] = [ppi]
        _df = pd.DataFrame(data=terms, columns=list(terms.keys()))
        if not hasattr(self, '_ppi_df'):
            self._ppi_df = pd.DataFrame(columns=list(terms.keys()))
        return _df

    @property
    def ppi_vocabulary(self):
        check_is_fitted(self, '_ppi_df')
        return self._ppi_df

    @property
    def accession_vocabulary(self):
        check_is_fitted(self, '_accession_df')
        return self._accession_df

    def _compute_induced_terms(self, ppi):
        check_is_fitted(self, '_accession_df')
        go_col = UniProt.data_types().GO.value
        if not go_col in self._selection:
            raise ValueError("Cannot induce without `{}` in selection.".format(
                go_col
            ))

        go_terms_sets = [self._get_for_accession(p, go_col) for p in ppi]
        ont_dicts = [group_go_by_ontology(ts, self._godag, UniProt.sep())
                     for ts in go_terms_sets]
        term_sets_cc = [d['cc'] for d in ont_dicts]
        term_sets_bp = [d['bp'] for d in ont_dicts]
        term_sets_mf = [d['mf'] for d in ont_dicts]
        induced_cc = get_up_to_lca(term_sets_cc, self._godag)
        induced_bp = get_up_to_lca(term_sets_bp, self._godag)
        induced_mf = get_up_to_lca(term_sets_mf, self._godag)
        return induced_cc, induced_bp, induced_mf

    def _get_for_accession(self, accession, column):
        check_is_fitted(self, '_accession_df')
        try:
            df = self._accession_df
            entry = df[df[UniProt.accession_column()] == accession]
            if entry.empty:
                raise KeyError(accession)

            # If nan or singular then return empty list/single item
            values = entry[column].values[0]
            if len(values) == 0:
                values = []
            return list(values)

        except KeyError:
            print("Warning: Couldn't find entry for {}. Try transforming new "
                  "ppis first.")
            return []

    def _update(self, item):
        if isinstance(item, pd.DataFrame):
            assert hasattr(self, '_ppi_df')
            _df = pd.concat([self._ppi_df, item], ignore_index=True)
            _df = _df.drop_duplicates(subset=UniProt.accession_column())
            self._ppi_df = _df
        else:
            assert hasattr(self, '_accession_df')
            _df = self._uniprot.features_to_dataframe([item])
            _df = pd.concat([self._accession_df, _df], ignore_index=True)
            _df = _df.drop_duplicates(subset=UniProt.accession_column())
            self._accession_df = _df
        return self

    def cache(self, accession_path, ppi_path):
        if hasattr(self, '_accession_df'):
            self._accession_df.to_pickle(accession_path)
        if hasattr(self, '_ppi_df'):
            self._ppi_df.to_pickle(ppi_path)
        return self


# -------------------------------------------------------------------------- #
def test(up, dag):
    ppis = [PPI('P00813', 'P40855'), PPI('Q6ICG6', 'Q8NFS9')]
    a = AnnotationExtractor(
        up, dag,
        True,
        ['go', 'interpro', 'pfam'], 1,
        True
    )
    a.fit(ppis)
    return a, ppis


def do_it():
    from ..data import load_go_dag
    from ..data_mining import uniprot as up
    return up.UniProt(sprot_cache='', trembl_cache=''), load_go_dag()
