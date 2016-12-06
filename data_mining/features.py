#!/usr/bin/env python

"""
This module contains class and method definitions related to extracting
features from PPIs, including feature induction as per Maestechze et al., 2011
"""

import pandas as pd
from itertools import chain

from data import PPI, accession_features, ppi_features
from data import accession_features_path, ppi_features_path
from data_mining.uniprot import UniProt
from data_mining.ontology import get_up_to_lca, group_go_by_ontology

from sklearn.utils.validation import check_is_fitted


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

    def __init__(self, uniprot, godag, induce, selection, n_jobs, cache=True):
        self._uniprot = uniprot
        self._godag = godag
        self._n_jobs = n_jobs
        self._selection = selection
        self._induce = induce
        self._cache = cache
        if cache:
            try:
                self._accession_df = accession_features()
                self._ppi_df = ppi_features()
            except IOError:
                print('Warning: No cache files found.')

    def fit(self, X, y=None):
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

        :param y:
            None

        :param fit_params: parameters to pass to the fit_transform method

        :return: X :  array-like, shape (n_samples, )
            List of induced GO terms for each PPI sample.
        """
        ppis = [PPI(a, b) for (a, b) in X]
        data = [self._compute_features(ppi) for ppi in ppis]
        self.cache()
        return data

    def transform(self, X):
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
        check_is_fitted(self, '_ppi_df')
        data = []
        ppis = [PPI(a, b) for (a, b) in X]
        new = [ppi for ppi in ppis if ppi not in
               self._ppi_df[UniProt.accession_column()].values]
        self.fit([tuple(ppi) for ppi in new])

        for ppi in ppis:
            terms = {}
            entry = self._ppi_df[self._ppi_df.accession.values == ppi]
            for s in self._selection:
                terms[s] = entry[s].values
            features = [str(x) for k, v in terms.items() for x in v[0]]
            data.append(','.join(features))
        return data

    def _compute_features(self, ppi):
        terms = {}
        accessions = set([p for p in ppi])
        print(accessions)

        _df = self._uniprot.features_to_dataframe(accessions)
        if hasattr(self, '_accession_df'):
            self._accession_df = pd.concat([self._accession_df, _df],
                                           ignore_index=True)
        else:
            self._accession_df = _df

        if self._induce:
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

        features = [str(x) for k, v in terms.items() for x in v[0]]
        features = ','.join(features)

        terms['accession'] = [ppi]
        if not hasattr(self, '_ppi_df'):
            self._ppi_df = pd.DataFrame(columns=list(terms.keys()))
            for k, v in terms.items():
                self._ppi_df[k] = v
        else:
            self._update(terms)
        return features

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
        go_col = UniProt.data_types().GO
        if go_col.value not in self._selection:
            raise ValueError("Cannot induce without `{}` in selection.".format(
                go_col
            ))

        go_terms_sets = [self._get_for_accession(p, go_col.value) for p in ppi]
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
        assert hasattr(self, '_accession_df')
        try:
            df = self._accession_df
            entry = df[df[UniProt.accession_column()] == accession]
            if entry.empty:
                raise KeyError()

            # If nan or singular then return empty list/single item
            values = entry[column].values[0]
            if len(values) == 0:
                values = []
            return list(values)

        except KeyError:
            print("Warning: Couldn't find entry for {}".format(accession))
            self._update(accession)
            self._get_for_accession(accession, column)

    def _update(self, item):
        if isinstance(item, dict):
            assert hasattr(self, '_ppi_df')
            _df = pd.DataFrame(item, columns=list(item.keys()))
            self._ppi_df = pd.concat([self._ppi_df, _df], ignore_index=True)
        else:
            assert hasattr(self, '_accession_df')
            _df = self._uniprot.features_to_dataframe([item])
            self._accession_df = pd.concat(
                [self._accession_df, _df], ignore_index=True
            )
        return self

    def cache(self):
        if hasattr(self, '_accession_df'):
            self._accession_df.to_pickle(accession_features_path())
        if hasattr(self, '_ppi_df'):
            self._ppi_df.to_pickle(ppi_features_path())
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
    from data import load_go_dag
    from data_mining import uniprot as up
    return up.UniProt(sprot_cache='', trembl_cache=''), load_go_dag()
