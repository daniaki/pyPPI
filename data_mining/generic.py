#!/usr/bin/env python

"""
Author: Daniel Esposito
Date: 27/12/2015

Purpose: Generic parsing functions for various file formats and the
functionality to create data frames from the parsing results.
"""

import itertools
import numpy as np

from data_mining.tools import make_interaction_frame, process_interactions
from data_mining.tools import write_to_edgelist
from data import generic_io

INVALID_ACCESSIONS = ['', ' ', '-', 'unknown']


def validate_accession(accession):
    if accession.strip().lower() in INVALID_ACCESSIONS:
        return np.NaN
    else:
        return accession.strip().upper()


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
        labels.append('-')
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
        labels.append('-')
    return sources, targets, labels


def reactome_func(fp):
    """
    Parsing function for reactome interaction file format.

    :param fp: Open file handle containing the file to parse.
    :return: Tuple source, target and label lists.
    """
    source_idx = 0
    target_idx = 3
    label_idx = 6
    sources = []
    targets = []
    labels = []
    for line in fp:
        xs = line.strip().split('\t')
        source = validate_accession(xs[source_idx].strip().upper())
        target = validate_accession(xs[target_idx].strip().upper())
        label = xs[label_idx].strip().lower().replace(' ', '-')
        sources.append(source)
        targets.append(target)
        labels.append(label)
    return sources, targets, labels


def mitab_func(fp):
    """
    Parsing function for psimitab format.

    :param fp: Open file handle containing the file to parse.
    :return: Tuple source, target and label lists.
    """
    source_idx = 0
    target_idx = 1
    sources = []
    targets = []
    labels = []
    ppis = []

    # Remove header
    next(fp)

    for line in fp:
        accessions = []
        xs = [l for l in line.strip().split('\t') if 'uniprotkb' in l]
        if len(xs) < 2:
            accessions = [[], []]
        else:
            for index in [source_idx, target_idx]:
                ps = [e for e in xs[index].split('|')
                      if ('uniprotkb' in e) and ('_' not in e)]
                if len(ps) == 0:
                    accessions.append([])
                else:
                    p = [e for e in xs[index].split('|')
                         if ('uniprotkb' in e) and ('_' not in e)]
                    p = [x.split(':')[1] for x in p]
                    accessions.append(p)
        ppis.append(accessions)

    # Iterate through the list of tuples, each tuple being a
    # list of accessions found within a line for each of the two proteins.
    for source_xs, target_xs in ppis:
        for source, target in itertools.product(source_xs, target_xs):
            source = validate_accession(source)
            target = validate_accession(target)
            label = '-'
            sources.append(source)
            targets.append(target)
            labels.append(label)

    return sources, targets, labels


def generic_to_dataframe(f_input, parsing_func, drop_nan=True,
                         allow_self_edges=False, allow_duplicates=False,
                         min_label_count=None, merge=False,
                         exclude_labels=None, output=None):
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
    :param output: File to write dataframe to.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    lines = f_input
    if isinstance(f_input, str):
        lines = generic_io(f_input)

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
    if output:
        write_to_edgelist(interactions, output)
    return interactions
