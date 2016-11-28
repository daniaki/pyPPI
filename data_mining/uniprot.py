#!/usr/bin/env python

import time
import pandas as pd
from Bio import SwissProt
from Bio import ExPASy
from urllib.error import HTTPError

"""
Author: Daniel Esposito
Date: 27/12/2015

Purpose: Wrapper class for accessing uniprot records using biopython. See 
http://biopython.org/DIST/docs/api/Bio.SwissProt.Record-class.html for more 
information about how biopython stores records.
"""

SWISSPROT_DAT = 'data/uniprot_sprot_human.dat'
TREMBL_DAT = 'data/uniprot_trembl_human.dat'
UNIPROT_ORD_KEY = dict(P=0, Q=1, O=2)


class UniProt(object):
    """
    A simple interface to to Uniprot to download data for accessions.

    Public Methods
    --------------
    data_types:                 Returns a dictionary of accepted datatypes and their associated function.
    get_entry:					Returns a databse handle for a given accession.
    download_entry:             Download a record if not found in cache.
    get_single_accession_data: 	Batch download several types for an accession.
    get_batch_accession_data: 	Batch download several types for a list of accessions.
    features_to_dataframe:      Wraps the downloaded accesion features into a DataFrame.

    """

    def __init__(self, sprot_cache=SWISSPROT_DAT, trembl_cache=TREMBL_DAT,
                 taxonid='9606', verbose=False, retries=10, wait=5):
        """
        Class constructor

        :param sprot_cache: Path to SwissProt dat file.
        :param trembl_cache: Path to TrEMBL dat file.
        :param taxonid: taxonomy id supported by UniProt.
        :param verbose: Set to True to print warnings and errors.
        :param retries: Number of record download retries upon a HTTP error.
        :param wait: Seconds to wait before a retry.

        :return: UniProt object.
        """
        self.records = {}
        self.taxonid = taxonid.strip().capitalize()
        self.verbose = verbose
        self.retries = retries
        self.wait = wait
        print('Warning: Loading cache files may take a few minutes.')
        if sprot_cache:
            # Load the swissprot records if file can be found
            try:
                with open(sprot_cache) as fp:
                    for record in SwissProt.parse(fp):
                        for accession in record.accessions:
                            self.records[accession] = record
            except IOError as e:
                print("Warning: SwissProt cache not loaded:\n\n{}".format(e))

        if trembl_cache:
            # Load the trembl records if file can be found
            try:
                with open(trembl_cache) as fp:
                    for record in SwissProt.parse(fp):
                        for accession in record.accessions:
                            self.records[accession] = record
            except IOError as e:
                print("Warning: Tremble in fear because the Trembl cache was not loaded:\n\n{}".format(e))

    @staticmethod
    def __get_xrefs(db_name, record):
        result = []
        for xref in record.cross_references:
            extdb = xref[0]
            if extdb == db_name:
                result.append(xref[1:])
        return result

    def data_types(self):
        data_types = {
            'GO': 		    self.get_go_term_ids,
            'GO_NAME': 	    self.get_go_term_names,
            'GOE':		    self.get_go_term_evidence,
            'PFAM': 	    self.get_pfam_terms,
            'INTERPRO':	    self.get_interpro_terms,
            'EMBL':		    self.get_embl_terms,
            'SEQUENCE':	    self.get_sequence,
            'CLASS':	    self.get_review_status,
            'ALT':		    self.get_alt_accesions,
            'ORG':		    self.get_organism,
            'GENE_NAME':    self.get_gene_name,
            'KEYWORD':      self.get_keywords,
            'TAXONID':      self.get_taxonid,
            'ENTRY_NAME':   self.get_entry_name,
            'ORG_CODE':     self.get_organism_code,
            'REF_RECORDS':  self.get_references,
            'RECORD':       self.get_entry,
            'GENE_NAME_SYN': self.get_synonyms,
            'XREFS':         self.get_crossrefs,
            'FEATURES':      self.get_features
        }
        return data_types

    def get_entry(self, accession, verbose=False):
        try:
            return self.records[accession]
        except KeyError:
            if verbose:
                print('Record for {} now downloading...'.format(accession))
            return self.download_entry(accession)

    def download_entry(self, accession):
        record = None
        success = False
        try:
            handle = ExPASy.get_sprot_raw(accession)
            record = SwissProt.read(handle)
            success = True
        except HTTPError:
            if self.verbose:
                print("HTTPError downloading record for {}. Retrying...".format(accession))
            for i in range(self.retries):
                if self.verbose:
                    print('\tWaiting {}s...'.format(self.wait))
                time.sleep(self.wait)
                try:
                    if self.verbose:
                        print('Retry {}/{}...'.format(i, self.retries))
                    handle = ExPASy.get_sprot_raw(accession)
                    record = SwissProt.read(handle)
                    success = True
                except HTTPError:
                    continue
        finally:
            if not success:
                if self.verbose:
                    print("Warning: Failed to download record for {}".format(accession))

        if record and record.taxonomy_id != [self.taxonid]:
            if self.verbose:
                print("Warning: Taxonomy IDs do not match: {}, {}".format(
                    record.taxonomy_id, [self.taxonid]))
            record = None

        self.records[accession] = record
        return record

    def get_gene_name(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        try:
            data = record.gene_name.split(';')[0].split('=')[-1]
        except (KeyError, AssertionError, Exception):
            data = None
        return data

    def get_keywords(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = record.keywords
        return data

    def get_go_term_ids(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = self.__get_xrefs("GO", record)
        data = list(map(lambda x: x[0], data))
        return data

    def get_references(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = record.references
        return data

    def get_taxonid(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = record.taxonomy_id[0]
        return data

    def get_organism_code(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = record.entry_name
        data = data.split('_')[1]
        return data

    def get_go_term_names(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = self.__get_xrefs("GO", record)
        data = list(map(lambda x: x[1], data))
        return data

    def get_go_term_evidence(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = self.__get_xrefs("GO", record)
        data = list(map(lambda x: x[2][0:3], data))
        return data

    def get_entry_name(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = record.entry_name
        return data

    def get_pfam_terms(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = self.__get_xrefs("Pfam", record)
        return list(map(lambda x: x[0], data))

    def get_interpro_terms(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = self.__get_xrefs("InterPro", record)
        return list(map(lambda x: x[0], data))

    def get_embl_terms(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        data = self.__get_xrefs("EMBL", record)
        return list(map(lambda x: x[0], data))

    def get_sequence(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        return record.sequence

    def get_alt_accesions(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        return record.accessions

    def get_review_status(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        return record.data_class

    def get_organism(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        return record.organism

    def get_synonyms(self, accession):
        record = self.get_entry(accession)
        try:
            data = record.gene_name.split(';')[1].split('=')[1].split(', ')
        except (KeyError, AssertionError, Exception):
            data = '-'
        return data

    def get_crossrefs(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        print(record.cross_references)
        print(len(record.cross_references))
        return record.cross_references

    def get_features(self, accession):
        record = self.get_entry(accession)
        if not record:
            return None
        return record.features

    def get_single_accession_data(self, accession, data_types):
        """
        Queries uniprot servers the data types given by data_types.

        :param accession: String uniprot accession.
        :param data_types: Types from the enum to extract.
        :return: A dictionary indexed by accession information type.
        """
        data = {}
        for d in data_types:
            try:
                data[d] = self.data_types()[d](accession)
            except KeyError as e:
                print('Invalid data type: {}'.format(e))
                continue
        return data

    def get_batch_accession_data(self, accessions, data_types):
        """
        Takes an iterable object of uniprot string accessions and downloads
        the specified datatypes.

        :param accessions: List of string uniprot accessions.
        :param data_types: Types from the enum to extract.
        :return: Returns a dictionary of accession and data, which is dictionary
                 indexed by accession information type.
        """
        acc_data_map = {}
        for p in accessions:
            acc_data_map[p] = self.get_single_accession_data(p, data_types)
        return acc_data_map

    def features_to_dataframe(self, accessions, data_types, columns=None):
        """
        Wrapper function that will download specified datatypes into a dataframe.

        :param accessions: List of uniprotkb accessions.
        :param data_types: List of datatypes accepted by UniProt.
        :param columns: Dictionary mapping datatypes to column names for the dataframe.
        :return: DataFrame of accessions and acquired data.
        """
        unique = set(accessions)
        features = self.get_batch_accession_data(unique, data_types)
        column_keys = data_types if not columns else list(columns.values())
        dataframe_data = {k: [] for k in column_keys}

        if columns:
            dataframe_data = {columns[d]: [] for d in data_types}

        for _, data in features.items():
            for k, v in data.items():
                column_key = k if not columns else columns[k]
                dataframe_data[column_key].append(v)

        dataframe_data['ID'] = list(features.keys())
        df = pd.DataFrame(data=dataframe_data)
        return df
