#!/usr/bin/env python

"""
Author: Daniel Esposito
Date: 27/12/2015

Purpose: Generic parsing functions for various file formats and the
functionality to create data frames from the parsing results.
"""

import itertools
import numpy as np
from ..data import generic_io
from ..base import PUBMED, EXPERIMENT_TYPE
from .tools import process_interactions, make_interaction_frame

INVALID_ACCESSIONS = ['', ' ', '-', 'unknown']


def validate_accession(accession):
    if accession.strip().lower() in INVALID_ACCESSIONS:
        return None
    else:
        return accession.strip().upper()


def edgelist_func(fp):
    """
    Parsing function a generic edgelist file.

    :param fp: Open file handle containing the file to parse.
    :return: Tuple source, target and label lists.
    """
    source_idx = 0
    target_idx = 1
    sources = []
    targets = []
    labels = []

    # Remove header
    next(fp)

    for line in fp:
        xs = line.strip().split('\t')
        source = validate_accession(xs[source_idx].strip().upper())
        target = validate_accession(xs[target_idx].strip().upper())
        sources.append(source)
        targets.append(target)
        labels.append(None)
    return sources, targets, labels


def bioplex_func(fp):
    """
    Parsing function for bioplex tsv format.

    :param fp: Open file handle containing the file to parse.
    :return: Tuple source, target and label lists.
    """
    source_idx = 2
    target_idx = 3
    sources = []
    targets = []
    labels = []

    # Remove header
    next(fp)

    for line in fp:
        xs = line.strip().split('\t')
        source = validate_accession(xs[source_idx].strip().upper())
        target = validate_accession(xs[target_idx].strip().upper())
        sources.append(source)
        targets.append(target)
        labels.append(None)
    return sources, targets, labels


def pina_func(fp):
    """
    Parsing function for bioplex tsv format.

    :param fp: Open file handle containing the file to parse.
    :return: Tuple source, target and label lists.
    """
    source_idx = 0
    target_idx = 2
    sources = []
    targets = []
    labels = []
    for line in fp:
        xs = line.strip().split(' ')
        source = validate_accession(xs[source_idx].strip().upper())
        target = validate_accession(xs[target_idx].strip().upper())
        sources.append(source)
        targets.append(target)
        labels.append(None)
    return sources, targets, labels


def mitab_func(fp):
    """
    Parsing function for psimitab format.

    :param fp: Open file handle containing the file to parse.
    :return: Tuple source, target and label lists.
    """
    uniprot_source_idx = 4
    uniprot_target_idx = 5
    source_idx = 2
    target_idx = 3
    d_method_idx = 6  # detection method
    pmid_idx = 8
    i_type_idx = 11  # interaction type

    sources = []
    targets = []
    labels = []
    pmids = []
    experiment_types = []

    # Remove header
    next(fp)

    for line in fp:
        xs = line.strip().split('\t')
        ensembl_source = xs[source_idx].strip()
        ensembl_target = xs[target_idx].strip()
        if ('ENSG' not in ensembl_source) or ('ENSG' not in ensembl_target):
            continue

        # These formats might contain multiple uniprot interactors in a
        # single line, or none. Continue parsing if the latter.
        source_ls = [
            elem.split(':')[1] for elem in xs[uniprot_source_idx].split('|')
            if ('uniprotkb' in elem and not '_' in elem)
        ]
        target_ls = [
            elem.split(':')[1] for elem in xs[uniprot_target_idx].split('|')
            if ('uniprotkb' in elem) and (not '_' in elem)
        ]
        if len(source_ls) < 1 or len(target_ls) < 1:
            continue

        d_method_line = xs[d_method_idx].strip()
        d_psimi = None
        d_description = None
        if d_method_line not in ('', '-'):
            _, d_method_text = d_method_line.strip().split("psi-mi:")
            _, d_psimi, d_description = d_method_text.split('"')
            d_description = d_description.replace('(', '').replace(')', '')

        pmid_line = xs[pmid_idx].strip()
        pmid = None
        if pmid_line not in ('', '-'):
            pmid = ','.join(
                sorted(set(
                    [pmid_line.split(':')[-1] for t in pmid_line.split('|')]
                ))
            )

        i_type_line = xs[i_type_idx].strip()
        i_psimi = None
        i_description = None
        if i_type_line not in ('', '-'):
            _, i_method_text = i_type_line.strip().split("psi-mi:")
            _, i_psimi, i_description = i_method_text.split('"')
            i_description = i_description.replace('(', '').replace(')', '')

        # Iterate through the list of tuples, each tuple being a
        # list of accessions found within a line for each of the two proteins.
        for source, target in itertools.product(source_ls, target_ls):
            source = validate_accession(source)
            target = validate_accession(target)
            if source is None or target is None:
                continue
            else:
                label = None
                sources.append(source)
                targets.append(target)
                labels.append(label)
                pmids.append(pmid)
                experiment_types.append(d_psimi)

    return sources, targets, labels, pmids, experiment_types


def generic_to_dataframe(f_input, parsing_func, drop_nan=False,
                         allow_self_edges=False, allow_duplicates=False,
                         min_label_count=None, merge=False,
                         exclude_labels=None):
    """
    Generic function to parse an interaction file using the supplied parsing
    function into a dataframe object.

    :param f_input: Path to file or generator of file lines
    :param parsing_func: function that accepts a file pointer object.
    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param min_label_count: Remove labels with less than the specified count.
    :param merge: Merge entries with identical source and target columns
                  during filter.
    :param exclude_labels: List of labels to remove from the dataframe.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    lines = f_input
    if isinstance(f_input, str):
        lines = generic_io(f_input)

    if parsing_func == mitab_func:
        sources, targets, labels, pmids, e_types = parsing_func(lines)
        interactions = make_interaction_frame(
            sources, targets, labels,
            **{PUBMED: pmids, EXPERIMENT_TYPE: e_types}
        )
    else:
        sources, targets, labels = parsing_func(lines)
        interactions = make_interaction_frame(sources, targets, labels)

    interactions = process_interactions(
        interactions=interactions,
        drop_nan=drop_nan,
        allow_self_edges=allow_self_edges,
        allow_duplicates=allow_duplicates,
        exclude_labels=exclude_labels,
        min_counts=min_label_count,
        merge=merge
    )
    return interactions
