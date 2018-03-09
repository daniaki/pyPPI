#!/usr/bin/env python

"""
Author: Daniel Esposito
Date: 27/12/2015

Purpose: Generic parsing functions for various file formats and the
functionality to create data frames from the parsing results.
"""

import itertools
import numpy as np
from collections import OrderedDict

from ..base.io import generic_io
from ..base.utilities import remove_duplicates, is_null
from ..base.constants import PUBMED, EXPERIMENT_TYPE
from .tools import process_interactions, make_interaction_frame

INVALID_ACCESSIONS = ['', ' ', '-', 'unknown']


def validate_accession(accession):
    """Return None if an accession is invalid, else strip and uppercase it."""
    if accession.strip().lower() in INVALID_ACCESSIONS:
        return None
    else:
        return accession.strip().upper()


def edgelist_func(fp):
    """
    Parsing function a generic edgelist file.

    fp : :class:`io.TextIOWrapper`
        Open file handle containing the file to parse.

    Returns
    -------
    `tuple[str, str, None]`
        Source, target and label lists. Label is always a list of `None`
        values.
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

    fp : :class:`io.TextIOWrapper`
        Open file handle containing the file to parse.

    Returns
    -------
    `tuple[str, str, None]`
        Source, target and label lists. Label is always a list of `None`
        values.
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


def pina_sif_func(fp):
    """
    Parsing function for bioplex tsv format.

    fp : :class:`io.TextIOWrapper`
        Open file handle containing the file to parse.

    Returns
    -------
    `tuple[str, str, None]`
        Source, target and label lists. Label is always a list of `None`
        values.
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


def innate_mitab_func(fp):
    """
    Parsing function for psimitab format files issued by `InnateDB`.

    fp : :class:`io.TextIOWrapper`
        Open file handle containing the file to parse.

    Returns
    -------
    `tuple[str, str, None, str, str]`
        Source, target, label, pubmed and psimi lists. Label is always a list 
        of `None` values. The other entries may be `None` if invalid values
        are enountered.
    """
    uniprot_source_idx = 4
    uniprot_target_idx = 5
    source_idx = 2
    target_idx = 3
    d_method_idx = 6  # detection method
    pmid_idx = 8

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
        if not is_null(d_method_line):
            _, d_method_text = d_method_line.strip().split("psi-mi:")
            _, d_psimi, _ = d_method_text.split('"')
            if is_null(d_psimi):
                d_psimi = None
            else:
                d_psimi.strip().upper()

        pmid_line = xs[pmid_idx].strip()
        pmid = None
        if not is_null(pmid_line):
            pmid = pmid_line.split(':')[-1]
            if is_null(pmid):
                pmid = None
            else:
                pmid.strip().upper()

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


def pina_mitab_func(fp):
    """
    Parsing function for psimitab format files from `PINA2`.

    fp : :class:`io.TextIOWrapper`
        Open file handle containing the file to parse.

    Returns
    -------
    `tuple[str, str, None, str, str]`
        Source, target, label, pubmed and psimi lists. Label is always a list 
        of `None` values. The other entries may be `None` if invalid values
        are enountered.
    """
    uniprot_source_idx = 0
    uniprot_target_idx = 1
    d_method_idx = 6  # detection method
    pmid_idx = 8

    sources = []
    targets = []
    labels = []
    pubmed_ids = []
    experiment_types = []

    # Remove header
    next(fp)

    for line in fp:
        xs = line.strip().split('\t')
        source = xs[uniprot_source_idx].split(':')[-1].strip().upper()
        target = xs[uniprot_target_idx].split(':')[-1].strip().upper()
        if is_null(target) or is_null(source):
            continue

        pmids = [x.split(':')[-1] for x in xs[pmid_idx].strip().split('|')]
        psimis = [x.split('(')[0] for x in xs[d_method_idx].strip().split('|')]
        assert len(psimis) == len(pmids)

        annotations = OrderedDict()
        for pmid, psimi in zip(pmids, psimis):
            if is_null(pmid):
                continue
            pmid = pmid.strip().upper()
            if not pmid in annotations:
                annotations[pmid] = set()
            if not is_null(psimi):
                psimi = psimi.strip().upper()
                annotations[pmid].add(psimi)

        pmid_group = ','.join(annotations.keys()) or None
        if pmid_group is not None:
            for pmid, psimi_group in annotations.items():
                annotations[pmid] = '|'.join(sorted(psimi_group)) or str(None)
            psimi_groups = ','.join(annotations.values())
        else:
            psimi_groups = None

        label = None
        sources.append(source)
        targets.append(target)
        labels.append(label)
        pubmed_ids.append(pmid_group)
        experiment_types.append(psimi_groups)

    return sources, targets, labels, pubmed_ids, experiment_types


def generic_to_dataframe(f_input, parsing_func, drop_nan=None,
                         allow_self_edges=False, allow_duplicates=False,
                         min_label_count=None, merge=False,
                         exclude_labels=None):
    """
    Generic function to parse an interaction file using the supplied parsing
    function into a dataframe object.

    Parameters
    ----------
    f_input : str
        Path to file or generator of file lines

    parsing_func : callable 
        function that accepts a file pointer object.

    drop_nan : bool, str or list, default: None
        Drop entries containing null values in any column. If 'default'
        rows are dropped if null values occur in the `source`, `target` or 
        `label` columns. If a list of column names are supplied, then 
        rows are dropped if null values occur in either of those columns. If
        False or None then no rows will be dropped. If True, rows with 
        a null value in any column are dropped.

    allow_self_edges : bool, default: False
        If True, removes rows for which `source` is equal to `target`.

    allow_duplicates : bool, default: False
        If True, removes rows for which `source`, `target` and `label` are the
        same. If different annotations are seen in the `pubmed` and `experiment_type`
        columns, then these are merged so they are not lost.

    min_label_count : int, optional
        First computes the counts of labels over all rows, then removes those
        rows with labels that have less than the threshold count.

    merge : bool, default: False
        If True, merges entries with identical source and target columns. If 
        different annotations are seen in the `pubmed` and `experiment_type`
        columns, then these are also merged so they are not lost.

    exclude_labels : list, optional
        List of labels to remove from the dataframe. All rows with label equal
        to any in the supplied labels are removed.

    Returns
    -------
    :class:`pandas.DataFrame`
        With 'source', 'target' and 'label', 'pubmed' and 'experiment_type'
        columns.
    """
    lines = f_input
    if isinstance(f_input, str):
        lines = generic_io(f_input)

    if parsing_func in (pina_mitab_func, innate_mitab_func):
        sources, targets, labels, pmids, e_types = parsing_func(lines)
        interactions = make_interaction_frame(
            sources, targets, labels, pmids, e_types
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
