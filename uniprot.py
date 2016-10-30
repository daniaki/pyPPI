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


def features_to_dataframe(uniprot_db, accessions, data_types, columns=None):
    """
    Wrapper function that will download specified datatypes into a dataframe.

    :param uniprot_db: UniProt DB interface object.
    :param accessions: List of uniprotkb accessions.
    :param data_types: Datatypes accepted by UniProt.
    :param columns: Dictionary mapping datatypes to column names for the dataframe.
    :return: DataFrame of accessions and acquired data.
    """
    features = uniprot_db.get_batch_accession_data(accessions, data_types)
    column_keys = data_types if not columns else list(columns.values())
    dataframe_data = {k: [] for k in column_keys}

    if columns:
        dataframe_data = {columns[d]: [] for d in data_types}

    for _, data in features.items():
        for k, v in data.items():
            column_key = k if not columns else columns[k]
            if isinstance(v, list) or isinstance(v, set):
                v = ','.join(str(x) for x in sorted(v))
            dataframe_data[column_key].append(v)

    df = pd.DataFrame(data=dataframe_data)
    return df


class UniProt(object):
    """
    A simple interface to to Uniprot to download data for accessions.

    Private Methods
    ---------------
    __get_xrefs: Method to get the data from a cross-referenced database. Internal use only.
    __get_data:  Wrapper for accessing public methods from __DATA_TYPES.

    Public Methods
    --------------
    get_entry:					Returns a databse handle for a given accession.
    get_go_terms: 				Returns string of comma delimited go ids.
    get_pfam_terms: 			Returns string of comma delimited pfam ids
    get_interpro_terms: 		Returns string of comma delimited interpro ids.
    get_sequence: 				Returns string of amino acid sequence.
    get_embl_terms: 			Returns string of comma delimited EMBL terms.
    get_all_accesions: 			Return a list of alternative uniprot identifiers.
    get_get_review_status: 		Return review status of protein.
    get_organism: 				Return string organism code for accession.
    get_get_keywords:           Return a list of keywords.
    get_single_accession_data: 	Batch download several types for an accession.
    get_batch_accession_data: 	Batch download several types for a list of accessions.
    download_entry:             Download a record if not found in cache.

    """

    def __init__(self, sprot_cache=SWISSPROT_DAT, trembl_cache=TREMBL_DAT, organism='human'):
        self.records = {}
        self.organism = organism.strip().upper()
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

    def __get_data(self, accession, data_type):
        data_types = {
            'GO': 		self.get_go_term_ids,
            'GO_NAME': 	self.get_go_term_names,
            'GOE':		self.get_go_term_evidence,
            'PFAM': 	self.get_pfam_terms,
            'INTERPRO':	self.get_interpro_terms,
            'EMBL':		self.get_embl_terms,
            'SEQUENCE':	self.get_sequence,
            'CLASS':	self.get_review_status,
            'ALT':		self.get_alt_accesions,
            'ORG':		self.get_organism,
            'NAME':     self.get_gene_name,
            'KEYWORD':  self.get_keywords
        }
        try:
            func = data_types[data_type.strip().upper()]
            return func(accession=accession)
        except KeyError as e:
            print(e)

    def get_entry(self, accession, verbose=False):
        try:
            return self.records[accession]
        except KeyError:
            if verbose:
                print('Record for {} now downloading...'.format(accession))
            return self.download_entry(accession)

    def download_entry(self, accession, retries=10, wait=5):
        record = SwissProt.Record()
        try:
            handle = ExPASy.get_sprot_raw('{}_{}'.format(accession, self.organism.upper()))
            record = SwissProt.read(handle)
        except HTTPError:
            for i in range(retries):
                time.sleep(wait)
                try:
                    handle = ExPASy.get_sprot_raw('{}_{}'.format(accession, self.organism.upper()))
                    record = SwissProt.read(handle)
                except HTTPError:
                    continue
        finally:
            # TODO: Return some form of null datatype?
            Exception("Failed to download {}".format(accession))

        self.records[accession] = record
        return record

    def get_gene_name(self, accession):
        record = self.get_entry(accession)
        try:
            data = record.gene_name.split(';')[0].split('=')[-1]
        except (KeyError, AssertionError, Exception):
            data = '-'
        return data

    def get_keywords(self, accession):
        record = self.get_entry(accession)
        data = record.keywords
        return data

    def get_go_term_ids(self, accession):
        record = self.get_entry(accession)
        data = self.__get_xrefs("GO", record)
        data = list(map(lambda x: x[0], data))
        return data

    def get_go_term_names(self, accession):
        record = self.get_entry(accession)
        data = self.__get_xrefs("GO", record)
        data = list(map(lambda x: x[1], data))
        return data

    def get_go_term_evidence(self, accession):
        record = self.get_entry(accession)
        data = self.__get_xrefs("GO", record)
        data = list(map(lambda x: x[2][0:3], data))
        return data

    def get_pfam_terms(self, accession):
        record = self.get_entry(accession)
        data = self.__get_xrefs("Pfam", record)
        return list(map(lambda x: x[0], data))

    def get_interpro_terms(self, accession):
        record = self.get_entry(accession)
        data = self.__get_xrefs("InterPro", record)
        return list(map(lambda x: x[0], data))

    def get_embl_terms(self, accession):
        record = self.get_entry(accession)
        data = self.__get_xrefs("EMBL", record)
        return list(map(lambda x: x[0], data))

    def get_sequence(self, accession):
        record = self.get_entry(accession)
        return record.sequence

    def get_alt_accesions(self, accession):
        record = self.get_entry(accession)
        return record.accessions

    def get_review_status(self, accession):
        record = self.get_entry(accession)
        return record.data_class

    def get_organism(self, accession):
        record = self.get_entry(accession)
        return record.organism

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
                data[d] = self.__get_data(accession, d)
            except KeyError as e:
                print(e)
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
