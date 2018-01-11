#!/usr/bin/env python

"""
Purpose: Wrapper class for accessing uniprot records using biopython. See
http://biopython.org/DIST/docs/api/Bio.SwissProt.Record-class.html for more
information about how biopython stores records.
"""

import time
import logging
import pandas as pd

from Bio import SwissProt
from Bio import ExPASy
from bioservices import UniProt as UniProtMapper
from urllib.error import HTTPError
from enum import Enum
from joblib import delayed, Parallel

from ..base import chunk_list
from ..data import uniprot_sprot, uniprot_trembl
from ..database.models import Protein
from ..database.managers import ProteinManager


UNIPROT_ORD_KEY = dict(P=0, Q=1, O=2)
logger = logging.getLogger("pyppi")
http_error_msg = "HTTPError downloading record for {}."


# --------------------------------------------------------------------------- #
#
#                    Biopython/SwissProt Download Utilities
#
# --------------------------------------------------------------------------- #


def download_record(accession, verbose=False, wait=5,
                    retries=3, taxon_id=9606):
    """
    Stub
    """
    record = None
    success = False

    try:
        handle = ExPASy.get_sprot_raw(accession)
        record = SwissProt.read(handle)
        success = True

    except HTTPError:
        if verbose:
            logger.exception(http_error_msg.format(accession))
            logger.info("Re-attempting to download.")

        for _ in range(retries):
            time.sleep(wait)
            try:
                handle = ExPASy.get_sprot_raw(accession)
                record = SwissProt.read(handle)
                success = True
            except HTTPError:
                if verbose:
                    logger.exception(http_error_msg.format(accession))

    except ValueError:
        if verbose:
            logger.exception("No record found for '{}'.".format(accession))
        record = None
        success = False

    if not success:
        if verbose:
            logger.error(
                "Failed to download record for '{}'".format(accession)
            )
        record = None

    if record and int(record.taxonomy_id[0]) != taxon_id:
        if verbose:
            logger.error(
                "Taxonomy IDs do not match. "
                "Expected '{}' but found '{}'.".format(
                    taxon_id, int(record.taxonomy_id[0])
                )
            )
        record = None

    return record


def download_records(accessions, verbose=False, wait=5,
                     retries=3, taxon_id=9606):
    """
    Stub
    """
    return [
        download_record(a, verbose, wait, retries, taxon_id)
        for a in accessions
    ]


def parallel_download(accessions, backend="multiprocessing",
                      verbose=False, n_jobs=1, wait=5,
                      retries=3, taxon_id=9606):
    """
    Stub
    """
    # Warning: Setting backend to multiprocessing may cause strange errors.
    # This is most likely due to this function not being run in a
    # protected main loop.
    accession_chunks = chunk_list(accessions, n=n_jobs)
    records = Parallel(backend=backend, verbose=verbose, n_jobs=n_jobs)(
        delayed(download_records)(chunk, verbose, wait, retries, taxon_id)
        for chunk in list(accession_chunks)
    )
    return [r for sublist in records for r in sublist]


def parse_record_into_protein(record, verbose=False):
    if record is None:
        return None
    try:
        uniprot_id = recent_accession(record)
        taxon_id = taxonid(record)
        gene_id = gene_name(record)
        go_mf = go_terms(record, ont="mf")
        go_bp = go_terms(record, ont="bp")
        go_cc = go_terms(record, ont="cc")
        interpro = interpro_terms(record)
        pfam = pfam_terms(record)
        reviewed = True if review_status(record) == 'Reviewed' else False
        keywords_ = keywords(record)

        entry = Protein(
            uniprot_id=uniprot_id, taxon_id=taxon_id, reviewed=reviewed,
            gene_id=gene_id, go_mf=go_mf, go_bp=go_bp, go_cc=go_cc,
            interpro=interpro, pfam=pfam, keywords=keywords_
        )
        return entry
    except:
        if verbose:
            logger.exception(
                "An error occured when trying to parse "
                "record into a dictionary."
            )
        raise

# --------------------------------------------------------------------------- #
#
#                    Biopython/SwissProt Record Parsing
#
# --------------------------------------------------------------------------- #


def batch_map(session, accessions, fr='ACC+ID', allow_download=False,
              keep_unreviewed=True, match_taxon_id=9606
              ):
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

    # No data was downloaded, try again a few times.
    if mapping == {}:
        for i in range(0, 5):
            time.sleep(3)
            mapping = uniprot_mapper.mapping(
                fr=fr, to='ACC', query=accessions
            )
            if mapping:
                break
    if mapping == {}:
        raise ValueError("Could not download map from uniprot server.")

    pm = ProteinManager(verbose=False, match_taxon_id=None)
    for fr, to in mapping.items():
        # Make sure any new accessions are in the database
        invalid_to = []
        for accession in to:
            if pm.get_by_uniprot_id(session, accession) is None:
                if allow_download:
                    logger.info(
                        "Mapping to {}, but entry not found in database. "
                        "Attempting download.".format(accession)
                    )
                    record = download_record(
                        accession, verbose=True, taxon_id=match_taxon_id
                    )
                    protein = parse_record_into_protein(record)
                    if protein is not None:
                        protein.save(session, commit=True)
                    else:
                        logger.info(
                            "No valid record for {} was found".format(
                                accession))
                        invalid_to.append(accession)
                else:
                    invalid_to.append(accession)

        to = [a for a in to if a not in invalid_to]
        status = [pm.get_by_uniprot_id(session, a).reviewed for a in to]
        reviewed = [a for (a, s) in zip(to, status) if s is True]
        unreviewed = [a for (a, s) in zip(to, status) if s is False]
        targets = reviewed
        if keep_unreviewed:
            targets += unreviewed

        targets = list(set(targets))
        if match_taxon_id is not None:
            taxon_ids = [
                pm.get_by_uniprot_id(session, a).taxon_id for a in targets
            ]
            targets = [
                t for (t, taxon_id) in zip(targets, taxon_ids)
                if match_taxon_id == taxon_id
            ]
        filtered_mapping[fr] = list(sorted(targets))
    return filtered_mapping


def __xrefs(db_name, record):
    """
    Stub
    """
    result = []
    for xref in record.cross_references:
        extdb = xref[0]
        if extdb == db_name:
            result.append(xref[1:])
    return result


def recent_accession(record):
    """
    Stub
    """
    if not record:
        return None
    return record.accessions[0]


def taxonid(record):
    """
    Stub
    """
    if not record:
        return None
    data = record.taxonomy_id[0]
    return int(data)


def review_status(record):
    """
    Stub
    """
    if not record:
        return None
    return record.data_class


def gene_name(record):
    """
    Stub
    """
    if not record:
        return None
    try:
        data = record.gene_name.split(';')[0].split('=')[-1].split(' ')[0]
    except (KeyError, AssertionError, Exception):
        data = None
    if not data:
        return None
    return data


def go_terms(record, ont):
    """
    Stub
    """
    if not record:
        return None
    data = __xrefs("GO", record)
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


def pfam_terms(record):
    """
    Stub
    """
    if not record:
        return None
    data = __xrefs("Pfam", record)
    return list(map(lambda x: x[0], data))


def interpro_terms(record):
    """
    Stub
    """
    if not record:
        return None
    data = __xrefs("InterPro", record)
    return list(map(lambda x: x[0], data))


def keywords(record):
    """
    Stub
    """
    if not record:
        return None
    data = record.keywords
    return data


def organism_code(record):
    """
    Stub
    """
    if not record:
        return None
    data = record.entry_name
    data = data.split('_')[1]
    return data


def entry_name(record):
    """
    Stub
    """
    if not record:
        return None
    return record.entry_name


def synonyms(record):
    """
    Stub
    """
    if not record:
        return None
    try:
        data = record.gene_name.split(';')[1].split('=')[1].split(
            ', ').split(' ')[0]
    except (KeyError, AssertionError, Exception):
        data = None
    return data
