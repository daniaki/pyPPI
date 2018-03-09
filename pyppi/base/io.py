"""
This module contains functions to open and parse the files found in 
:module:`.constants`. This is so other parts of the program do not become 
coupled to the data parsing processes.
"""

__all__ = [
    "line_generator",
    "generic_io",
    "uniprot_sprot",
    "uniprot_trembl",
    "hprd_ptms",
    "hprd_id_map",
    "bioplex_v4",
    "innate_curated",
    "innate_imported",
    "pina2_mitab",
    'pina2_sif',
    "ipr_name_map",
    "pfam_name_map",
    "load_uniprot_accession_map",
    "save_uniprot_accession_map",
    "load_network_from_path",
    "save_network_to_path",
    "save_classifier",
    "load_classifier",
    "download_from_url",
    "download_program_data"
]


import json
import os
import sys
import gzip
import pandas as pd
import joblib
import logging
from urllib.request import urlretrieve

from .file_paths import (
    pina2_mitab_path, bioplex_v4_path, uniprot_sprot_dat,
    uniprot_trembl_dat, hprd_ptms_txt, hprd_mappings_txt,
    innate_c_mitab_path, ipr_names_path, pfam_names_path,
    innate_i_mitab_path, PATH, obo_file, psimi_obo_file
)
from .file_paths import (
    interpro_names_url, pfam_clans_url, uniprot_sp_human_url,
    uniprot_tr_human_url, mi_obo_url, go_obo_url, bioplex_url, pina2_mitab_url,
    innate_imported_url, innate_curated_url, innate_i_mitab_path,
    pina2_sif_path, uniprot_map_path, classifier_path
)


def line_generator(io_func):
    """
    Decorator to turn a io dealing function into an iterator of file lines, 
    performing additional steps wrapping each line, such as byte decoding.
    """
    def wrapper_func(*args, **kwargs):
        fp = io_func(*args, **kwargs)
        for line in fp:
            if isinstance(line, bytes):
                yield line.decode('utf-8')
            yield line
        fp.close()
    return wrapper_func


@line_generator
def generic_io(file):
    """Opens a generic file such as an edgelist file."""
    return open(file, 'r')


@line_generator
def uniprot_sprot():
    """Opens the gzipped UniProt SwissProt dat file."""
    return gzip.open(uniprot_sprot_dat, 'rt')


@line_generator
def uniprot_trembl():
    """Opens the gzipped UniProt TrEMBL dat file."""
    return gzip.open(uniprot_trembl_dat, 'rt')


@line_generator
def hprd_ptms():
    """Opens HPRD post-translational modifications flat file."""
    return open(hprd_ptms_txt, 'r')


@line_generator
def hprd_id_map():
    """Opens HPRD enzyme to UniProt mapping file."""
    return open(hprd_mappings_txt, 'r')


@line_generator
def bioplex_v4():
    """Opens the BioPlex network edgelist."""
    return gzip.open(bioplex_v4_path, 'rt')


@line_generator
def innate_curated():
    """Opens the InnateDB curated psi-mi file."""
    return gzip.open(innate_c_mitab_path, 'rt')


@line_generator
def innate_imported():
    """Opens the InnateDB imported psi-mi file."""
    return gzip.open(innate_i_mitab_path, 'rt')


@line_generator
def pina2_sif():
    """Opens the Pina2 network edgelist."""
    return gzip.open(pina2_sif_path, 'rt')


@line_generator
def pina2_mitab():
    """Opens the Pina2 network edgelist."""
    return gzip.open(pina2_mitab_path, 'rt')


def ipr_name_map():
    """
    Parse the interpro list into a dictionary. Expects uppercase accessions.

    Returns
    -------
    `dict`
        Dictionary mapping from InterPro accession to description.
    """
    fp = open(ipr_names_path, 'r')
    ipr_map = {}
    for line in fp:
        xs = line.strip().split("\t")
        term = xs[0].upper()
        descrip = xs[-1].strip()
        ipr_map[term] = descrip
    fp.close()
    return ipr_map


def pfam_name_map():
    """
    Parse the pfam list into a dictionary. Expects uppercase accessions.

    Returns
    -------
    `dict`
        Dictionary mapping from Pfam accession to description.
    """
    fp = gzip.open(pfam_names_path, 'rt')
    pf_map = {}
    for line in fp:
        xs = line.strip().split("\t")
        term = xs[0].upper()
        descrip = xs[-1].strip()
        pf_map[term] = descrip
    fp.close()
    return pf_map


def load_uniprot_accession_map():
    """
    Loads a saved JSON UniProt accession mapping, mapping UniProt accessions 
    to their most recent UniProt accessions, which may be multiple values.

    Returns
    -------
    `dict[str, list]`
        Dictionary mapping UniProt accession to UniProt accession.
    """
    if not os.path.isfile(uniprot_map_path):
        raise IOError("No mapping file could be found.")
    with open(uniprot_map_path, 'r') as fp:
        return json.load(fp)


def save_uniprot_accession_map(mapping):
    """Save an accession map to the default application home directory."""
    with open(uniprot_map_path, 'w') as fp:
        return json.dump(mapping, fp)


def load_network_from_path(path):
    """Load a tab separated-file into a dataframe."""
    from .constants import NULL_VALUES
    return pd.read_csv(path, sep='\t', na_values=NULL_VALUES)


def save_network_to_path(interactions, path):
    """Save dataframe to a tab-separated file at path."""
    return interactions.to_csv(path, sep='\t', index=False, na_rep=str(None))


def save_classifier(clf, selection, path=None):
    """Save a tuple of :module:`sklearn` classifier object and `list` to 
    path (defaults to home directory),
    """
    if path is None:
        path = classifier_path
    return joblib.dump((clf, selection), path)


def load_classifier(path=None):
    """Load a tuple of :module:`sklearn` classifier object and `list` from  
    path (defaults to home directory). The list contains the databases
    used to train the classifier, which can be one of `go_mf`, `go_cc`,
    `go_bp`, `interpro` or `pfam`.

    Returns
    -------
    tuple[:class:`sklearn.base.BaseEstimator`, list]
        Tuple of :class:`sklearn.base.BaseEstimator` and a list of
        feature sets the classifier was trained on.
    """
    if path is None:
        path = classifier_path
    clf, sel = joblib.load(path)
    return clf, sel


def download_from_url(url, save_path, compress=False):
    print("Downloading file from %s" % url)
    if compress:
        tmp, info = urlretrieve(url)
        bytes_ = info['Content-Length']
        print("\tCompresing file with size %s bytes" % bytes_)
        with open(tmp, 'rb') as f_in, gzip.open(save_path, 'wb') as f_out:
            f_out.writelines(f_in)
    else:
        urlretrieve(url, save_path)


def download_program_data():
    # TODO: Add HPRD links to this
    os.makedirs(os.path.normpath(PATH), exist_ok=True)
    os.makedirs(os.path.normpath(PATH + '/networks/'), exist_ok=True)
    download_from_url(interpro_names_url, ipr_names_path, compress=False)
    download_from_url(pfam_clans_url, pfam_names_path, compress=False)
    download_from_url(mi_obo_url, psimi_obo_file, compress=True)
    download_from_url(go_obo_url, obo_file, compress=True)

    download_from_url(pina2_mitab_url, pina2_mitab_path, compress=True)
    download_from_url(bioplex_url, bioplex_v4_path, compress=True)
    download_from_url(innate_curated_url, innate_c_mitab_path, compress=False)
    download_from_url(innate_imported_url, innate_i_mitab_path, compress=False)
    # download_from_url(hprd_map_url, hprd_mappings_txt, compress=False)
    # download_from_url(hprd_ptm_url, hprd_ptms_txt, compress=False)

    download_from_url(uniprot_sp_human_url, uniprot_sprot_dat, compress=False)
    download_from_url(uniprot_tr_human_url, uniprot_trembl_dat, compress=False)
