#!/usr/bin/env python

"""
Author: Daniel Esposito
Contact: danielce90@gmail.com

This module provides functionality to perform filtering and processing on
interaction dataframes.
"""

import pandas as pd

from collections import Counter
from itertools import product
from ..base import PPI

SOURCE = 'source'
TARGET = 'target'
LABEL = 'label'


def make_interaction_frame(sources, targets, labels):
    """
    Wrapper to construct a PPI dataframe.

    :param sources: Interactor ID that is the source node.
    :param targets: Interactor ID that is the target node.
    :param labels: Edge label for the interaction.
    :return: DataFrame with SOURCE, TARGET and LABEL columns.
    """
    ppis = [tuple(PPI(a, b)) for a, b in zip(sources, targets)]
    sources = [a for a, _ in ppis]
    targets = [b for _, b in ppis]
    interactions = dict(
        source=sources,
        target=targets,
        label=labels
    )
    return pd.DataFrame(data=interactions, columns=[SOURCE, TARGET, LABEL])


def map_network_accessions(interactions, accession_map, drop_nan,
                           allow_self_edges, allow_duplicates,
                           min_counts, merge):
    """
    Map the accession in the `source` and `target` columns to some other
    accessions mapped to y the `accession_map`.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param accession_map: Dictionary from old accession to new.
    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param min_counts: Remove labels below this count.
    :param merge: Merge PPIs with the same source and target but
                  different labels into the same entry.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    ppis = [tuple(PPI(accession_map.get(a, None), accession_map.get(b, None)))
            for (a, b) in zip(interactions.sources, interactions.targets)]
    sources = [a for a, _ in ppis]
    targets = [b for _, b in ppis]
    new_interactions = make_interaction_frame(sources, targets,
                                              interactions.label)
    new_interactions = process_interactions(
        interactions=new_interactions,
        drop_nan=drop_nan,
        allow_self_edges=allow_self_edges,
        allow_duplicates=allow_duplicates,
        exclude_labels=None,
        min_counts=min_counts,
        merge=merge
    )
    return new_interactions


def remove_nan(interactions):
    """
    Drop interactions with missing source, target or labels as indicated
    by np.NaN/None.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    from numpy import NaN
    df = interactions.replace(to_replace=str(None), value=NaN, inplace=False)
    df.replace(to_replace=str(NaN), value=NaN, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df = df.reset_index(drop=True)
    return df


def remove_intersection(interactions, other):
    """

    :param interactions:
    :param other:
    :return:
    """
    selector = set()
    hash_map = {}
    other_ppis = zip(other[SOURCE], other[TARGET], other[LABEL])
    for (source, target, label) in other_ppis:
        a, b = sorted([source, target])
        for s, t, l in product([a], [b], label.split(',')):
            hash_map[(s, t, l)] = True

    df = interactions
    interactions_ppis = zip(df[SOURCE], df[TARGET], df[LABEL])
    for i, (source, target, label) in enumerate(interactions_ppis):
        a, b = sorted([source, target])
        for s, t, l in product([a], [b], label.split(',')):
            if hash_map.get((s, t, l)) is None:
                selector.add(i)

    return interactions.loc[selector, ], hash_map, selector


def remove_labels(interactions, subtypes):
    """
    Remove PPIs with a subtype in exclusions list.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param subtypes: Character list of subtypes to exclude.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    print('Warning: Removing labels should be done before merging labels '
          'as the merge can result in new concatenated labels.')
    labels = interactions.label.values
    selector = [(l not in subtypes) for l in labels]
    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True)
    return df


def remove_min_counts(interactions, min_count):
    """
    Remove all PPIs with labels that have overall count less
    than some threshold.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param min_count: Minimum count threshold to keep label.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    print('Warning: Removing low count labels '
          'should be done before merging labels '
          'as the merge can result in many new low count labels.')
    counts = Counter(interactions.label.values)
    labels_to_exclude = [k for k, v in counts.items() if v < min_count]
    df = remove_labels(interactions, labels_to_exclude)
    return df


def remove_self_edges(interactions):
    """
    Remove PPIs in which 'target' is equal to 'source'.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions
    selector = [str(s) != str(t) for (s, t) in zip(df.source, df.target)]
    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True)
    assert sum([str(a) == str(b) for (a, b) in zip(df.source, df.target)]) == 0
    return df


def merge_labels(interactions):
    """
    Merge PPIs with the same source and target but different labels into the
    same entry with labels being separated by a comma.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions
    merged_ppis = {}
    ppis = [tuple(PPI(s, t)) for (s, t) in zip(df.source, df.target)]

    for (s, t), label in zip(ppis, interactions.label):
        if (s, t) in merged_ppis:
            labels = sorted(set(merged_ppis[(s, t)].split(',') + [label]))
            merged_ppis[(s, t)] = ','.join(labels)
        else:
            merged_ppis[(s, t)] = label.lower().replace(" ", '-')

    sources = [ppi[0] for ppi in merged_ppis.keys()]
    targets = [ppi[1] for ppi in merged_ppis.keys()]
    labels = [label for label in merged_ppis.values()]

    interactions = make_interaction_frame(sources, targets, labels)
    return interactions


def remove_duplicates(interactions):
    """
    Remove rows with identical source, target and label column entries.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions
    merged_ppis = {}
    ppis = [tuple(PPI(s, t)) for (s, t) in zip(df.source, df.target)]
    merged = any([len(l.split(',')) > 1 for l in df[LABEL]])

    if merged:
        for (s, t), label in zip(ppis, df.label):
            for l in label.split(','):
                merged_ppis[(s, t, l)] = 1.0
    else:
        for (s, t), label in zip(ppis, df.label):
            merged_ppis[(s, t, label)] = 1

    sources = [ppi[0] for ppi in merged_ppis.keys()]
    targets = [ppi[1] for ppi in merged_ppis.keys()]
    labels = [ppi[2] for ppi in merged_ppis.keys()]
    interactions = make_interaction_frame(sources, targets, labels)

    if merged:
        interactions = merge_labels(interactions)

    assert sum(interactions.duplicated()) == 0
    return interactions


def write_to_edgelist(interactions, file):
    """
    Write interactions dataframe to a tab-sep file.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param file: File to write to.
    :return: None
    """
    interactions.to_csv(file, sep='\t', index=False)


def process_interactions(interactions, drop_nan, allow_self_edges,
                         allow_duplicates, exclude_labels, min_counts, merge):
    """
    Wrapper to filter an interaction dataframe.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param exclude_labels: List of labels to remove.
    :param min_counts: Remove labels below this count.
    :param merge: Merge PPIs with the same source and target but
                  different labels into the same entry.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    if drop_nan:
        interactions = remove_nan(interactions)
    if not allow_self_edges:
        interactions = remove_self_edges(interactions)
    if not allow_duplicates:
        interactions = remove_duplicates(interactions)
    if exclude_labels:
        interactions = remove_labels(interactions, exclude_labels)
    if min_counts:
        interactions = remove_min_counts(interactions, min_count=min_counts)
    if merge:
        interactions = merge_labels(interactions)
    return interactions
