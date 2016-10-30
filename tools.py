#!/usr/bin/env python
from collections import Counter
import pandas as pd

"""
Author: Daniel Esposito
Contact: danielce90@gmail.com

This module provides functionality to perform filtering and processing on interaction dataframes.
"""

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
    interactions = dict(
        source=sources,
        target=targets,
        label=labels
    )
    return pd.DataFrame(data=interactions, columns=[SOURCE, TARGET, LABEL])


def remove_nan(interactions):
    """
    Drop interactions with missing source, target or labels as indicated by np.NaN.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions.dropna(axis=0, how='any', inplace=False)
    df = df.reset_index(drop=True)
    return df


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
    Remove all PPIs with labels that have overall count less than some threshold.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param min_count: Minimum count threshold to keep label.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    print('Warning: Removing low count labels should be done before merging labels '
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
    selector = [str(s) != str(t) for (s, t) in zip(interactions.source, interactions.target)]
    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True)
    assert sum([str(a) == str(b) for (a, b) in zip(df.source, df.target)]) == 0
    return df


def merge_labels(interactions):
    """
    Merge PPIs with the same source and target but different labels into the same
    entry with labels being separated by a comma.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    merged_ppis = {}
    ppis = [tuple(sorted([s, t], key=lambda x: str(x)))
            for (s, t) in zip(interactions.source, interactions.target)]

    for (s, t), label in zip(ppis, interactions.label):
        if (s, t) in merged_ppis:
            labels = sorted(set(merged_ppis[(s, t)].split(',') + [label]))
            merged_ppis[(s, t)] = ','.join(labels)
        else:
            merged_ppis[(s, t)] = label.lower().replace(" ", '-')
    labels = [label for label in merged_ppis.values()]

    sources = [ppi[0] for ppi in merged_ppis.keys()]
    targets = [ppi[1] for ppi in merged_ppis.keys()]
    interactions = make_interaction_frame(sources, targets, labels)
    return interactions


def remove_duplicates(interactions):
    """
    Remove rows with identical source, target and label column entries.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    merged_ppis = {}
    ppis = [tuple(sorted([s, t], key=lambda x: str(x)))
            for (s, t) in zip(interactions.source, interactions.target)]

    for (s, t), label in zip(ppis, interactions.label):
        merged_ppis[(s, t, label)] = 1.0
    labels = [ppi[2] for ppi in merged_ppis.keys()]

    sources = [ppi[0] for ppi in merged_ppis.keys()]
    targets = [ppi[1] for ppi in merged_ppis.keys()]
    interactions = make_interaction_frame(sources, targets, labels)
    assert sum(interactions.duplicated()) == 0
    return interactions


def write_to_edgelist(interactions, file):
    """
    Write interactions dataframe to a tab-sep file.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param file: File to write to.
    :return: None
    """
    interactions.to_csv(filename=file, sep='\t')


def process_interactions(interactions, drop_nan, allow_self_edges, allow_duplicates,
                         exclude_labels, min_counts, merge):
    """
    Wrapper to filter an interaction dataframe.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param exclude_labels: List of labels to remove.
    :param min_counts: Remove labels below this count.
    :param merge: Merge PPIs with the same source and target but different labels into the same entry.
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




