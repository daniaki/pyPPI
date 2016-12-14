#!/usr/bin/env python

"""
This module contains functions to open and parse the files found in this
directory (data) and generally act as datatypes. This is so other parts of the
program do not become coupled to the data parsing process.
"""

import pandas as pd
from goatools import obo_parser


def line_generator(io_func):
    """
    Decorator to turn a io dealing function into an iterator of file lines.
    :param io_func: function that opens a file with error handling
    """
    def wrapper_func():
        fp = io_func()
        for line in fp:
            yield line
        fp.close()
    return wrapper_func


@line_generator
def generic_io(file):
    try:
        return open(file, 'r')
    except IOError as e:
        print(e)


@line_generator
def uniprot_hsa_list():
    try:
        return open("data/uniprot_hsa.list", 'r')
    except IOError as e:
        print(e)@line_generator


@line_generator
def swissprot_hsa_list():
    try:
        return open("data/swiss_hsa.list", 'r')
    except IOError as e:
        print(e)


@line_generator
def uniprot_sprot():
    try:
        return open("data/uniprot_sprot_human.dat", 'r')
    except IOError as e:
        print(e)


@line_generator
def uniprot_trembl():
    try:
        return open("data/uniprot_trembl_human.dat", 'r')
    except IOError as e:
        print(e)


@line_generator
def hprd_ptms():
    try:
        return open("data/hprd/POST_TRANSLATIONAL_MODIFICATIONS.txt", 'r')
    except IOError as e:
        print(e)


@line_generator
def hprd_id_map():
    try:
        return open("data/hprd/HPRD_ID_MAPPINGS.txt", 'r')
    except IOError as e:
        print(e)


@line_generator
def bioplex_v2():
    try:
        return open("data/networks/BioPlex_interactionList_v2.tsv", 'r')
    except IOError as e:
        print(e)


@line_generator
def bioplex_v4():
    try:
        return open("data/networks/BioPlex_interactionList_v4.tsv", 'r')
    except IOError as e:
        print(e)


@line_generator
def innate_curated():
    try:
        return open("data/networks/innatedb_curated.mitab", 'r')
    except IOError as e:
        print(e)


@line_generator
def innate_imported():
    try:
        return open("data/networks/innatedb_imported.mitab", 'r')
    except IOError as e:
        print(e)


@line_generator
def pina2():
    try:
        return open("data/networks/PINA2_Homo_sapiens-20140521.sif", 'r')
    except IOError as e:
        print(e)


def hsa_swissprot_map():
    hsa_sp = {}
    for line in swissprot_hsa_list():
        xs = line.strip().split('\t')
        uprot = xs[0].split(':')[1]
        hsa = xs[1].split(':')[1]
        hsa_sp[hsa] = uprot
    return hsa_sp


def hsa_uniprot_map():
    hsa_sp = {}
    for line in uniprot_hsa_list():
        xs = line.strip().split('\t')
        uprot = xs[0].split(':')[1]
        hsa = xs[1].split(':')[1]
        hsa_sp[hsa] = uprot
    return hsa_sp


def load_go_dag(optional_attrs=None):
    """Load an obo file into a goatools GODag object"""
    default = optional_attrs or ['defn', 'is_a', 'relationship', 'part_of']
    return obo_parser.GODag("data/gene_ontology.1_2.obo",
                            optional_attrs=default)


def ipr_shortname_map(lowercase_keys=False):
    """
    Parse the interpro list into a dictionary.
    """
    fp = open("data/ipr_short_names.dat", 'r')
    ipr_map = {}
    for line in fp:
        if lowercase_keys:
            xs = line.strip().lower().split("\t")
        else:
            xs = line.strip().upper().split("\t")
        term = xs[0].upper()
        descrip = xs[1]
        ipr_map[term] = descrip
    fp.close()
    return ipr_map


def ipr_longname_map(lowercase_keys=False):
    """
    Parse the interpro list into a dictionary.
    """
    fp = open("data/ipr_names.dat", 'r')
    ipr_map = {}
    for line in fp:
        if lowercase_keys:
            xs = line.strip().lower().split("\t")
        else:
            xs = line.strip().upper().split("\t")
        term = xs[0].upper()
        descrip = xs[1]
        ipr_map[term] = descrip
    fp.close()
    return ipr_map


def pfam_name_map(lowercase_keys=False):
    """
    Parse the pfam list into a dictionary.
    """
    fp = open("data/data/pfam_names.tsv", 'r')
    pf_map = {}
    for line in fp:
        if lowercase_keys:
            xs = line.strip().lower().split("\t")
        else:
            xs = line.strip().upper().split("\t")
        term = xs[0].upper()
        descrip = xs[-1]
        pf_map[term] = descrip
    fp.close()
    return pf_map


def ptm_labels():
    """
    Load the labels in the tsv file into a list.
    """
    labels = set()
    with open("data/labels.tsv", 'r') as fp:
        for line in fp:
            l = line.strip().replace(' ', '-').lower()
            labels.add(l)
    return list(labels)


def load_training_network():
    return pd.read_csv(training_network_path(), sep='\t')


def load_testing_network():
    return pd.read_csv(testing_network_path(), sep='\t')


def load_interactome_network():
    return pd.read_csv(testing_network_path(), sep='\t')


def load_accession_features():
    return pd.read_pickle(accession_features_path())


def load_ppi_features():
    return pd.read_pickle(ppi_features_path())


def accession_features_path():
    return 'data/accession_features.pkl'


def ppi_features_path():
    return 'data/ppi_features.pkl'


def training_network_path():
    return 'data/networks/training_network.tsv'


def testing_network_path():
    return 'data/networks/testing_network.tsv'


def interactome_network_path():
    return 'data/networks/interactome_network.tsv'

