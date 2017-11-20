#!/usr/bin/env python

"""
This module contains class and method definitions related to extracting
features from PPIs, including feature induction as per Maestechze et al., 2011
"""

import pandas as pd
from itertools import chain

from ..base import PPI, chunk_list, concat_dataframes
from ..data import pickle_pd_object, read_pd_pickle
from ..data import accession_features_path, ppi_features_path
from .uniprot import UniProt, get_active_instance
from .ontology import get_up_to_lca, group_go_by_ontology
from .ontology import get_active_instance as get_active_godag

from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Parallel, delayed

_ACCESSION_COLUMN = UniProt.accession_column()
_DATA_TYPES = UniProt.data_types()
_SEP = UniProt.sep()


class AnnotationExtractor(object):
    """
    Convert a collection of leaf Gene Ontology annotations to a set of
    Gene Ontology annotations including additional terms up to the lowest
    common ancestor.

    Parameters
    ----------

    Attributes
    ----------
    accession_vocabulary : dict
        A pd.DataFrame of single accessions and their textual features.

    ppi_vocabulary : dict
        A pd.DataFrame of PPIs and their textual features.

    Examples
    -----
    """

    def __init__(self, induce, selection, n_jobs, verbose=False, cache=True,
                 backend='threading'):
        self._n_jobs = n_jobs
        self._selection = selection
        self._induce = induce
        self._cache = cache
        self._verbose = verbose
        self._backend = backend
        self.ppi_idx_map = {}
        if cache:
            self._accession_df = read_pd_pickle(accession_features_path)
            self._ppi_df = read_pd_pickle(ppi_features_path)
            self._build_ppi_idx_map()

    @property
    def ppi_vocabulary(self):
        check_is_fitted(self, '_ppi_df')
        return self._ppi_df

    @property
    def accession_vocabulary(self):
        check_is_fitted(self, '_accession_df')
        return self._accession_df

    @property
    def selection(self):
        r_selection = []
        go_cols = [_DATA_TYPES.GO.value, _DATA_TYPES.GO_MF.value,
                   _DATA_TYPES.GO_BP.value, _DATA_TYPES.GO_CC.value]
        for s in self._selection:
            if s in go_cols and self._induce:
                s = 'i' + s
            r_selection.append(s)
        return r_selection

    def invalid_ppis(self, X, indices=False):
        """
        Return a list of PPIs with accessions that have been marked invalid
        due to missing data.

        Parameters
        ----------
        :param indices: list
            Return a list indices that are invalid in the fitted data,
            otherwise return the PPIs themselves.

        Returns
        ----------
        :return: List of indices or PPIs
        """
        check_is_fitted(self, '_accession_df')
        check_is_fitted(self, '_ppi_df')
        ppis = self._validate_input(X, use_set=False)
        fitted_ppis = set(self._ppi_df[_ACCESSION_COLUMN].values)

        new_ppis = set(ppis) - fitted_ppis
        if len(new_ppis) > 0:
            raise ValueError("New PPIs detected, please use transform first.")

        fitted_accessions = set(self._accession_df[_ACCESSION_COLUMN].values)
        validity = self._accession_df['valid'].values
        v_map = {a: v for (a, v) in zip(fitted_accessions, validity)}
        v_indicator = [sum((v_map[a], v_map[b])) for (a, b) in ppis]
        invalid_ppis = [ppi for (ppi, i) in zip(ppis, v_indicator) if i < 2]
        invalid_indices = [ind for (ind, i) in enumerate(v_indicator) if i < 2]
        if indices:
            return invalid_indices
        return invalid_ppis

    def valid_ppis(self, X, indices=False):
        """
        Return a list of PPIs with accessions that have been marked as valid.

        Parameters
        ----------
        :param indices: list
            Return a list indices that are valid in the fitted data,
            otherwise return the PPIs themselves.

        Returns
        ----------
        :return: List of indices or PPIs
        """
        check_is_fitted(self, '_accession_df')
        check_is_fitted(self, '_ppi_df')
        ppis = self._validate_input(X, use_set=False)
        fitted_ppis = set(self._ppi_df[_ACCESSION_COLUMN].values)

        new_ppis = set(ppis) - fitted_ppis
        if len(new_ppis) > 0:
            raise ValueError("New PPIs detected, please use transform first.")

        fitted_accessions = set(self._accession_df[_ACCESSION_COLUMN].values)
        validity = self._accession_df['valid'].values
        v_map = {a: v for (a, v) in zip(fitted_accessions, validity)}
        v_indicator = [sum((v_map[a], v_map[b])) for (a, b) in ppis]
        valid_ppis = [ppi for (ppi, i) in zip(ppis, v_indicator) if i > 1]
        valid_indices = [ind for (ind, i) in enumerate(v_indicator) if i > 1]
        if indices:
            return valid_indices
        return valid_ppis

    def fit(self, X, y=None):
        """
        Given a list of protein UniProt accessions, builds a dictionary of
        induced GO annotations for each. Fitting on new data will erase
        any previous data.

        Parameters
        ----------
        :param X : iterable
            An iterable which yields PPI objects or tuples of accessions.

        :param y:
            None

        :return:
            self
        """
        ppis = self._validate_input(X, use_set=True)
        accessions = set(acc for ppi in ppis for acc in ppi)
        self._accession_df = pd.DataFrame()
        self._ppi_df = pd.DataFrame()

        # first run sets up the accession_df attribute dataframe
        if self._verbose:
            print('Acquiring features for PPIs...')
        _df = get_active_instance().features_to_dataframe(accessions)
        if hasattr(self, '_accession_df'):
            _df = pd.concat([self._accession_df, _df], ignore_index=True)
            _df = _df.drop_duplicates(subset=_ACCESSION_COLUMN)
        self._accession_df = _df

        # Run the intensive computation in parallel and append the local cache
        if self._verbose:
            print('Computing selected features for PPIs...')
        chunks = chunk_list(list(ppis), n=self._n_jobs)
        compute_features = delayed(self._compute_features)
        dfs = Parallel(n_jobs=self._n_jobs, verbose=self._verbose,
                       backend=self._backend)(
            compute_features(ppi_ls) for ppi_ls in chunks
        )

        # Running this in parallel speeds up the concatenation process a lot.
        if self._verbose:
            print('Updating instance attributes...')

        combine_dfs = delayed(concat_dataframes)
        dfs = Parallel(n_jobs=self._n_jobs, verbose=self._verbose,
                       backend=self._backend)(
            combine_dfs(df_list) for df_list in dfs
        )
        for df in dfs:
            self._update(df)
        self._build_ppi_idx_map()
        return self

    def fit_transform(self, X):
        """
        Given a list of protein UniProt accessions, builds a dataframe of
        features

        Parameters
        ----------
        :param X : iterable
            An iterable which yields PPI objects or tuples of accessions.

        :return: X :  array-like, shape (n_samples, )
            List of induced GO terms for each PPI sample.
        """
        self.fit(X)
        check_is_fitted(self, '_accession_df')
        check_is_fitted(self, '_ppi_df')

        # Convert the features for each ppi in the dataframe into a string
        features = pd.DataFrame()
        ppis = self._validate_input(X, use_set=False)
        selector = [self.ppi_idx_map[ppi] for ppi in ppis]
        selected_ppis = self._ppi_df.loc[selector, ]
        features['X'] = selected_ppis.apply(self._string_features, axis=1)
        return features['X'].values

    def transform(self, X):
        """
        Transform a list of PPIs to a list of GO annotations

        Parameters
        ----------
        :param X : iterable
            An iterable which yields PPI objects or tuples of accessions.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        check_is_fitted(self, '_accession_df')
        check_is_fitted(self, '_ppi_df')
        ppis_set = self._validate_input(X, use_set=True)
        fitted_ppis = set(self._ppi_df[_ACCESSION_COLUMN].values)
        new_ppis = ppis_set - fitted_ppis

        if len(new_ppis) > 0:
            raise ValueError(
                "Encountered new ppis {}. Please re-fit.".format(new_ppis)
            )

        # Convert the features for each ppi in the dataframe into a string
        features = pd.DataFrame()
        ppis = self._validate_input(X, use_set=False)
        selector = [self.ppi_idx_map[ppi] for ppi in ppis]
        selected_ppis = self._ppi_df.loc[selector, ]
        features['X'] = selected_ppis.apply(self._string_features, axis=1)
        return features['X'].values

    def _validate_input(self, X, use_set=False):
        """
        Validate the input checking for NoneTypes, zero length. Converts a
        list of tuples into a list/set of PPIs.
        """
        if len(X) == 0 or X is None:
            raise ValueError("Cannot fit/transform an empty input.")

        if isinstance(X[-1], PPI):
            ppis = X
        elif isinstance(X[-1], tuple):
            ppis = [PPI(a, b) for (a, b) in X]
        else:
            raise ValueError("Inputs must be either tuples or PPIs")

        if use_set:
            return set(ppis)
        else:
            return ppis

    def _compute_features(self, ppis):
        """
        Builds a dataframe with columns `accession` containing the PPIs
        and columns containing the features requested by the `selection`
        attribute.

        Parameters
        ----------
        :param ppi: PPI object
            A single PPI object.

        Returns
        -------
        :return: pd.DataFrame
            Singular dataframe for PPI object.
        """
        dfs = []
        for ppi in ppis:
            terms = {}
            if self._induce:
                induced_cc, induced_bp, induced_mf = \
                    self._compute_induced_terms(ppi)
                if _DATA_TYPES.GO.value in self._selection:
                    terms['igo'] = [induced_cc + induced_bp + induced_mf]
                if _DATA_TYPES.GO_MF.value in self._selection:
                    terms['igo_mf'] = [induced_mf]
                if _DATA_TYPES.GO_BP.value in self._selection:
                    terms['igo_bp'] = [induced_bp]
                if _DATA_TYPES.GO_CC.value in self._selection:
                    terms['igo_cc'] = [induced_cc]

            # Concatenate the other terms into strings
            for s in self._selection:
                data = [self._get_data_for_accession(p, s) for p in ppi]
                if isinstance(data[-1], list):
                    data = list(chain.from_iterable(data))
                terms[s] = [data]

            terms[_ACCESSION_COLUMN] = [ppi]
            df = pd.DataFrame(data=terms, columns=list(terms.keys()))
            dfs.append(df)
        return dfs

    def _compute_induced_terms(self, ppi):
        """
        Compute the ULCA inducer for a PPI

        Parameters
        ----------
        :param ppi: PPI object
            Compute the ULCA inducer for a PPI

        Returns
        ----------
        :return: string tuple
            set of string GO annotations for each ontology
        """
        check_is_fitted(self, '_accession_df')
        go_cols = [_DATA_TYPES.GO.value, _DATA_TYPES.GO_MF.value,
                   _DATA_TYPES.GO_BP.value, _DATA_TYPES.GO_CC.value]
        valid = sum([s in go_cols for s in self._selection])
        if not valid:
            raise ValueError("Cannot induce without any of `{}` "
                             "in selection.".format(go_cols))
        go_terms_sets = []
        for p in ppi:
            term_set = []
            for s in self._selection:
                if s in go_cols:
                    term_set += self._get_data_for_accession(p, s)
            go_terms_sets += [term_set]

        ont_dicts = [group_go_by_ontology(ts, get_active_godag(), _SEP)
                     for ts in go_terms_sets]
        term_sets_cc = [d['cc'] for d in ont_dicts]
        term_sets_bp = [d['bp'] for d in ont_dicts]
        term_sets_mf = [d['mf'] for d in ont_dicts]
        induced_cc = get_up_to_lca(term_sets_cc, get_active_godag())
        induced_bp = get_up_to_lca(term_sets_bp, get_active_godag())
        induced_mf = get_up_to_lca(term_sets_mf, get_active_godag())
        return induced_cc, induced_bp, induced_mf

    def _get_data_for_accession(self, accession, column):
        """
        Get the features from column `column` for accession `accession`
        from the `_accession_df` instance attribute.

        Parameters
        ----------
        :param accession: string
            accession code

        :param column: string
            column to get features from

        Returns
        ----------
        :return: List of features
        """
        check_is_fitted(self, '_accession_df')
        try:
            df = self.accession_vocabulary
            entry = df[df[_ACCESSION_COLUMN] == accession]
            if entry.empty:
                raise KeyError(accession)

            # If nan or singular then return empty list/single item
            values = entry[column].values[0]
            if len(values) == 0:
                values = []
            return list(values)

        except KeyError:
            raise ValueError(
                "Warning: Couldn't find entry for {}. Use the fit_transform "
                "method if PPI list contains new entries.".format(accession)
            )

    def _string_features(self, row):
        """
        For a given PPI takes the features from _ppi_df and combines those
        specificed in `selection` and combines them into a string that
        is comma delimited.

        Parameters
        ----------
        :param row: pd.DataFrame
            Single row from the `_ppi_df` dataframe

        Returns
        ----------
        :return: string
        """
        terms = []
        for col in self.selection:
            try:
                terms += row[col]
            except KeyError:
                raise KeyError("Selection contains a non-existent "
                               "column: {}. Make sure selection matches"
                               "the loaded cache file.".format(col))
        terms = [
            t.replace(':', '')
            for t in ','.join(terms).split(',')
            if t.strip() != ''
        ]
        return ','.join(terms)

    def _update(self, item):
        """
        Helper function to update instance attributes. Avoids duplicate
        entries in the attribute dataframes.

        Parameters
        ----------
        :param item: pd.DataFrame or list/set of PPI objects

        Returns
        ----------
        :return: self
        """
        if isinstance(item, pd.DataFrame):
            if not hasattr(self, '_ppi_df'):
                _df = item.reset_index(drop=True)
                self._ppi_df = _df
            else:
                _df = pd.concat([self._ppi_df, item], ignore_index=True)
                _df = _df.drop_duplicates(subset=_ACCESSION_COLUMN)
                _df = _df.reset_index(drop=True)
                self._ppi_df = _df
            return self
        elif isinstance(item, list) or isinstance(item, set):
            check_is_fitted(self, '_accession_df')
            if len(item) == 0:
                return self
            _df = get_active_instance().features_to_dataframe(item)
            _df = pd.concat([self._accession_df, _df], ignore_index=True)
            _df = _df.drop_duplicates(subset=_ACCESSION_COLUMN)
            _df = _df.reset_index(drop=True)
            self._accession_df = _df
        else:
            raise TypeError("Expected DataFrame or List of PPI objects")

    def _build_ppi_idx_map(self):
        ppi_acc = self._ppi_df[_ACCESSION_COLUMN].values
        ppi_index = self._ppi_df.index
        self.ppi_idx_map = {ppi: idx for (ppi, idx) in zip(ppi_acc, ppi_index)}

    def cache(self):
        """
        Save instance attributes to a pickle file.

        Parameters
        ----------
        None

        Returns
        ----------
        :return: self
        """
        check_is_fitted(self, '_accession_df')
        check_is_fitted(self, '_ppi_df')
        pickle_pd_object(self.accession_vocabulary, accession_features_path)
        pickle_pd_object(self.ppi_vocabulary, ppi_features_path)
        return self
