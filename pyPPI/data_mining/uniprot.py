#!/usr/bin/env python

"""
Author: Daniel Esposito
Date: 27/12/2015

Purpose: Wrapper class for accessing uniprot records using biopython. See
http://biopython.org/DIST/docs/api/Bio.SwissProt.Record-class.html for more
information about how biopython stores records.
"""

import time
import pandas as pd

from Bio import SwissProt
from Bio import ExPASy
from bioservices import UniProt as UniProtMapper
from urllib.error import HTTPError
from enum import Enum, EnumMeta

from ..data import uniprot_sprot, uniprot_trembl
from sklearn.externals.joblib import delayed, Parallel


__UNIPROT__ = None
UNIPROT_ORD_KEY = dict(P=0, Q=1, O=2)


def get_active_instance(**kwargs):
    global __UNIPROT__
    if __UNIPROT__ is None:
        print("First time loading on UniProt instance. "
              "Make take a few moments")
        __UNIPROT__ = UniProt(**kwargs)
    return __UNIPROT__


class UniProt(object):
    """
    A simple interface to to Uniprot to download data for accessions.

    Public Methods
    --------------
    data_types:             Returns a dictionary of accepted datatypes and
                            their associated function.
    entry:					Returns a databse handle for a given accession.
    download_entry:         Download a record if not found in cache.
    single_accession_data: 	Batch download several types for an accession.
    batch_accession_data: 	Batch download several types for a
                            list of accessions.
    features_to_dataframe:  Wraps the downloaded accesion features
                            into a DataFrame.

    """

    def __init__(self, sprot_cache=uniprot_sprot, trembl_cache=uniprot_trembl,
                 taxonid='9606', verbose=False, retries=10, wait=10, n_jobs=1,
                 allow_download=False):
        """
        Class constructor

        :param sprot_cache: Path to SwissProt dat file.
        :param trembl_cache: Path to TrEMBL dat file.
        :param taxonid: taxonomy id supported by UniProt.
        :param verbose: Set to True to print warnings and errors.
        :param retries: Number of record download retries upon a HTTP error.
        :param wait: Seconds to wait before a retry.
        :param n_jobs: Number of proccesses to spawn.
        :param allow_download: Download missing records from uniprot.

        :return: UniProt object.
        """
        self.records = {}
        self.tax = taxonid.strip().capitalize()
        self.verbose = verbose
        self.retries = retries
        self.wait = wait
        self.n_jobs = n_jobs
        self.allow_download = allow_download

        print('Warning: Loading dat files, may take a few minutes.')
        if sprot_cache:
            # Load the swissprot records if file can be found
            for record in SwissProt.parse(sprot_cache()):
                for accession in record.accessions:
                    self.records[accession] = record

        if trembl_cache:
            # Load the trembl records if file can be found
            for record in SwissProt.parse(uniprot_trembl()):
                for accession in record.accessions:
                    self.records[accession] = record

    @staticmethod
    def __xrefs(db_name, record):
        result = []
        for xref in record.cross_references:
            extdb = xref[0]
            if extdb == db_name:
                result.append(xref[1:])
        return result

    @staticmethod
    def accession_column():
        return 'accession'

    @staticmethod
    def sep():
        return ','

    @staticmethod
    def data_types():
        class Allowed(Enum):
            GO = 'go'
            GO_MF = 'go_mf'
            GO_BP = 'go_bp'
            GO_CC = 'go_cc'
            GO_NAME = 'go_name'
            GO_MF_NAME = 'go_name_mf'
            GO_BP_NAME = 'go_name_bp'
            GO_CC_NAME = 'go_name_cc'
            GO_EVD = 'goe'
            GO_MF_EVD = 'goe_mf'
            GO_BP_EVD = 'goe_bp'
            GO_CC_EVD = 'goe_cc'
            PFAM = 'pfam'
            INTERPRO = 'interpro'
            EMBL = 'embl'
            SEQ = 'sequence'
            CLASS = 'class'
            ALT = 'alt'
            ORG = 'org'
            GENE = 'gene_name'
            KW = 'keyword'
            TAX = 'taxonid'
            NAME = 'entry_name'
            ORG_CODE = 'org_code'
            REF_REC = 'ref_records'
            REC = 'record'
            GENE_SYN = 'gene_name_syn'
            XREF = 'xrefs'
            FEATURE = 'features'
        return Allowed

    def _data_func(self):
        functions = {
            'go': 		    self.go_term_ids,
            'go_name': 	    self.go_term_names,
            'goe':		    self.go_term_evidence,
            'go_mf':        self.go_term_ids,
            'go_name_mf':   self.go_term_names,
            'goe_mf':       self.go_term_evidence,
            'go_bp':        self.go_term_ids,
            'go_name_bp':   self.go_term_names,
            'goe_bp':       self.go_term_evidence,
            'go_cc':        self.go_term_ids,
            'go_name_cc':   self.go_term_names,
            'goe_cc':       self.go_term_evidence,
            'pfam': 	    self.pfam_terms,
            'interpro':	    self.interpro_terms,
            'embl':		    self.embl_terms,
            'sequence':	    self.sequence,
            'class':	    self.review_status,
            'alt':		    self.alt_accesions,
            'org':		    self.organism,
            'gene_name':    self.gene_name,
            'keyword':      self.keywords,
            'taxonid':      self.taxonid,
            'entry_name':   self.entry_name,
            'org_code':     self.organism_code,
            'ref_records':  self.references,
            'record':       self.entry,
            'gene_name_syn': self.synonyms,
            'xrefs':         self.crossrefs,
            'features':      self.features
        }
        return functions

    def entry(self, accession):
        try:
            return self.records[accession]
        except KeyError:
            if self.verbose:
                print('Record for {} not found.'.format(accession))
            if self.allow_download:
                if self.verbose:
                    print('Downloading record for {}...'.format(accession))
                return self.download_entry(accession)
            else:
                self.records[accession] = None
                return None

    def download_entry(self, accession):
        record = None
        success = False
        try:
            handle = ExPASy.get_sprot_raw(accession)
            record = SwissProt.read(handle)
            success = True
        except HTTPError:
            if self.verbose:
                print("HTTPError downloading record for {}. "
                      "Retrying...".format(accession))
            for i in range(self.retries):
                if self.verbose:
                    print('\tWaiting {}s...'.format(self.wait))
                time.sleep(self.wait)
                try:
                    if self.verbose:
                        print('Retry {}/{}...'.format(i+1, self.retries))
                    handle = ExPASy.get_sprot_raw(accession)
                    record = SwissProt.read(handle)
                    success = True
                except HTTPError:
                    continue
        except ValueError:
            if self.verbose:
                print("Warning: No SwissProt record found for {}".format(
                    accession))
            record = None
            success = False

        if not success:
            if self.verbose:
                print("Warning: Failed to download "
                      "record for {}".format(accession))
            record = None

        if record and record.taxonomy_id != [self.tax]:
            if self.verbose:
                print("Warning: Taxonomy IDs do not match: {}, {}".format(
                    record.taxonomy_id, [self.tax]))
            record = None

        self.records[accession] = record
        return record

    def gene_name(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        try:
            data = record.gene_name.split(';')[0].split('=')[-1].split(' ')[0]
        except (KeyError, AssertionError, Exception):
            data = None
        return data

    def keywords(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        data = record.keywords
        return data

    def go_term_ids(self, accession, ont):
        record = self.entry(accession)
        if not record:
            return []
        data = self.__xrefs("GO", record)
        ids = list(map(lambda x: x[0], data))
        names = list(map(lambda x: x[1], data))
        if ont == 'mf':
            ids = [i for (i, n) in zip(ids, names) if n[0] == 'F']
        elif ont == 'bp':
            ids = [i for (i, n) in zip(ids, names) if n[0] == 'P']
        elif ont == 'cc':
            ids = [i for (i, n) in zip(ids, names) if n[0] == 'C']
        else:
            pass
        return ids

    def references(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        data = record.references
        return data

    def taxonid(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        data = record.taxonomy_id[0]
        return data

    def organism_code(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        data = record.entry_name
        data = data.split('_')[1]
        return data

    def go_term_names(self, accession, ont):
        record = self.entry(accession)
        if not record:
            return []
        data = self.__xrefs("GO", record)
        names = list(map(lambda x: x[1], data))
        if ont == 'mf':
            names = [n for n in names if n[0] == 'F']
        elif ont == 'bp':
            names = [n for n in names if n[0] == 'P']
        elif ont == 'cc':
            names = [n for n in names if n[0] == 'C']
        else:
            pass
        return names

    def go_term_evidence(self, accession, ont):
        record = self.entry(accession)
        if not record:
            return []
        data = self.__xrefs("GO", record)
        evds = list(map(lambda x: x[2][0:3], data))
        names = list(map(lambda x: x[1], data))
        if ont == 'mf':
            evds = [e for (e, n) in zip(evds, names) if n[0] == 'F']
        elif ont == 'bp':
            evds = [e for (e, n) in zip(evds, names) if n[0] == 'P']
        elif ont == 'cc':
            evds = [e for (e, n) in zip(evds, names) if n[0] == 'C']
        else:
            pass
        return evds

    def entry_name(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        data = record.entry_name
        return data

    def pfam_terms(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        data = self.__xrefs("Pfam", record)
        return list(map(lambda x: x[0], data))

    def interpro_terms(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        data = self.__xrefs("InterPro", record)
        return list(map(lambda x: x[0], data))

    def embl_terms(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        data = self.__xrefs("EMBL", record)
        return list(map(lambda x: x[0], data))

    def sequence(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        return record.sequence

    def recent_accession(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        return record.accessions[0]

    def alt_accesions(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        return record.accessions

    def review_status(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        return record.data_class

    def organism(self, accession):
        record = self.entry(accession)
        if not record:
            return ''
        return record.organism

    def synonyms(self, accession):
        record = self.entry(accession)
        try:
            data = record.gene_name.split(';')[1].split('=')[1].split(
                ', ').split(' ')[0]
        except (KeyError, AssertionError, Exception):
            data = '-'
        return data

    def crossrefs(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        return record.cross_references

    def features(self, accession):
        record = self.entry(accession)
        if not record:
            return []
        return record.features

    def single_accession_data(self, i, n, accession, data_types):
        """
        Queries uniprot servers the data types given by data_types.

        :param accession: String uniprot accession.
        :param data_types: Types from the enum to extract.
        :return: A dictionary indexed by accession information type.
        """
        data = {}
        if self.verbose:
            print("Acquiring data for accession {}/{}..".format(i+1, n))

        for d in data_types:
            if isinstance(d, Enum) or isinstance(d, EnumMeta):
                d = d.value
            if self.entry(accession) is None:
                data['valid'] = False
            else:
                data['valid'] = True
            try:
                if 'go' in d:
                    ont = d.split('_')[-1]
                    annots = self._data_func()[d](accession, ont)
                else:
                    annots = self._data_func()[d](accession)
                if not isinstance(annots, list):
                    annots = [annots]
                data[d] = annots
            except KeyError as e:
                if self.verbose:
                    print('Warning: Invalid data type: {}. '
                          'Skipping.'.format(e))
                continue
        return data

    def batch_map(self, accessions, fr='ACC+ID', keep_unreviewed=True,):
        """
        Map a list of accessions using the UniProt batch mapping service.

        :param accessions: List of accessions.
        :param fr: Database to map from.
        :param keep_unreviewed: Also keep the unreviewed accession in mapping.
        :return: Dictionary of accessions to list of accessions.
        """
        uniprot_mapper = UniProtMapper()
        filtered_mapping = {}
        mapping = uniprot_mapper.mapping(fr=fr, to='ACC', query=accessions)
        for fr, to in mapping.items():
            status = list(zip(to, [self.review_status(a) for a in to]))
            reviewed = [a for (a, s) in status if s.lower() == 'reviewed']
            unreviewed = [a for (a, s) in status if s.lower() == 'unreviewed']
            targets = reviewed
            if keep_unreviewed:
                targets += unreviewed
            targets = list(set(targets))
            targets = [t for t in targets if self.taxonid(t) == self.tax]
            filtered_mapping[fr] = targets
        return filtered_mapping

    def batch_accession_data(self, accessions, data_types):
        """
        Takes an iterable object of uniprot string accessions and downloads
        the specified datatypes.

        :param accessions: List of string uniprot accessions.
        :param data_types: Types from the enum to extract.
        :return: Returns a dictionary of accession and data,
                 which is dictionary indexed by accession information type.
        """
        single_accession_data = delayed(self.single_accession_data)
        data = Parallel(n_jobs=self.n_jobs, backend='multiprocessing',
                        verbose=self.verbose)(
            single_accession_data(i, len(accessions), p, data_types)
            for i, p in enumerate(accessions)
        )
        acc_data_map = {p: d for p, d in zip(accessions, data)}
        return acc_data_map

    def features_to_dataframe(self, accessions, data_types=None, columns=None):
        """
        Wrapper function that will download specified
        datatypes into a dataframe.

        :param accessions: List of uniprotkb accessions.
        :param data_types: List of datatypes accepted by UniProt.
        :param columns: Dictionary mapping datatypes to column
                        names for the dataframe.
        :return: DataFrame of accessions and acquired data.
        """
        unique = set(list(accessions))
        if data_types is None:
            data_types = list(self._data_func().keys())
        features = self.batch_accession_data(unique, data_types)
        column_keys = data_types if not columns else list(columns.values())
        column_keys += ['valid']
        dataframe_data = {k: [] for k in column_keys}

        if columns:
            dataframe_data = {columns[d]: [] for d in data_types}

        for _, data in features.items():
            for k, v in data.items():
                column_key = k if not columns else columns[k]
                dataframe_data[column_key].append(v)

        dataframe_data[self.accession_column()] = list(features.keys())
        df = pd.DataFrame(data=dataframe_data)
        return df
