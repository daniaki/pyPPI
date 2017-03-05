#!/usr/bin/env python

"""
Author: Daniel Esposito
Contact: danielce90@gmail.com

This module provides functionality to perform filtering and processing on
interaction dataframes.
"""

from collections import Counter
from itertools import product

import pandas as pd
from numpy import NaN

from pyPPI.base import SOURCE, TARGET, LABEL
from ..base import PPI


def xy_from_interaction_frame(interactions):
    """
    Utility function to convert an interaction dataframe into seperate
    X and y numpy arrays.

    :param interactions: pd.DataFrame
        DataFrame with 'source', 'target' and 'label' columns.

    :return: array-like, shape (n_samples, )
    """
    X = ppis_from_interaction_frame(interactions, use_set=False)
    y = labels_from_interaction_frame(interactions, use_set=False)
    return X, y


def labels_from_interaction_frame(interactions, use_set=False):
    """
    Utility function to create an iterable of PPI objects from an interaction
    dataframe.

    :param interactions: pd.DataFrame
        DataFrame with 'source', 'target' and 'label' columns.
    :param use_set: boolean
        Use True to return a set of strings.

    :return: List or Set
        List or Set of string objects.
    """
    df = interactions
    labels = [l.lower().replace(" ", '-').split(',') for l in df[LABEL]]
    if use_set:
        return set(labels)
    return labels


def ppis_from_interaction_frame(interactions, use_set=False):
    """
    Utility function to create an iterable of PPI objects from an interaction
    dataframe.

    :param interactions: pd.DataFrame
        DataFrame with 'source', 'target' and 'label' columns.
    :param use_set: boolean
        Use True to return a set of PPIs

    :return: List or Set
        List or Set of PPI objects.
    """
    df = interactions
    ppis = [tuple(PPI(a, b)) for a, b in zip(df[SOURCE], df[TARGET])]
    if use_set:
        return set(ppis)
    return ppis


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
    labels = [l.lower().replace(" ", '-') for l in labels]
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
    :param accession_map: Dictionary from old accession to new. Dictionary
                          values must be a list to handle one to many mappings.
    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param min_counts: Remove labels below this count.
    :param merge: Merge PPIs with the same source and target but
                  different labels into the same entry.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    ppis = []
    labels = []
    for (a, b, l) in zip(interactions[SOURCE], interactions[TARGET],
                         interactions[LABEL]):
        sources = accession_map.get(a, [])
        targets = accession_map.get(b, [])
        for (s, t) in product(sources, targets):
            ppis.append(tuple(PPI(s, t)))
            labels.append(l)
    sources = [a for a, _ in ppis]
    targets = [b for _, b in ppis]
    new_interactions = make_interaction_frame(sources, targets, labels)
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
    df = interactions.replace(to_replace=str(None), value=NaN, inplace=False)
    df.replace(to_replace=str(NaN), value=NaN, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df = df.reset_index(drop=True, inplace=False)
    return df


def remove_intersection(interactions, other):
    """
    Remove any interaction from `interactions` appearing in `other` from
    `interactions`.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param other: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    selector = set()
    ppis_in_other = {}
    other_ppis = list(zip(other[SOURCE], other[TARGET], other[LABEL]))
    for (source, target, label) in other_ppis:
        a, b = sorted([source, target])
        for (s, t, l) in product([a], [b], label.split(',')):
            ppis_in_other[(s, t, l)] = True

    df = interactions.reset_index(drop=True, inplace=False)
    interactions_ppis = list(zip(df[SOURCE], df[TARGET], df[LABEL]))
    for i, (source, target, label) in enumerate(interactions_ppis):
        a, b = sorted([source, target])
        for (s, t, l) in product([a], [b], label.split(',')):
            if ppis_in_other.get((s, t, l)) is None:
                selector.add(i)

    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True, inplace=False)
    return df


def remove_labels(interactions, labels_to_exclude):
    """
    Remove PPIs with a subtype in exclusions list.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param labels_to_exclude: string list of subtypes to exclude.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    print('Warning: Removing labels should be done before merging labels '
          'as the merge can result in new concatenated labels.')
    labels = interactions.label.values
    selector = [(l not in labels_to_exclude) for l in labels]
    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True, inplace=False)
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
    labels_to_exclude = set([k for k, v in counts.items() if v < min_count])
    df = remove_labels(interactions, labels_to_exclude)
    df = df.reset_index(drop=True, inplace=False)
    return df


def remove_self_edges(interactions):
    """
    Remove PPIs in which 'target' is equal to 'source'.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions.reset_index(drop=True, inplace=False)
    selector = [str(s) != str(t) for (s, t) in zip(df.source, df.target)]
    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True, inplace=False)
    assert sum([str(a) == str(b) for (a, b) in zip(df.source, df.target)]) == 0
    return df


def merge_labels(interactions):
    """
    Merge PPIs with the same source and target but different labels into the
    same entry with labels being separated by a comma.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions.reset_index(drop=True, inplace=False)
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
    labels = [l.lower().replace(" ", '-') for l in merged_ppis.values()]

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
        assert len(ppis) == len(df.label.values)
        for (s, t), label in zip(ppis, df.label):
            for l in set(label.split(',')):
                merged_ppis[(s, t, l)] = 1
    else:
        assert len(ppis) == len(df.label.values)
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
    interactions = interactions.reset_index(drop=True, inplace=False)

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

    interactions = interactions.reset_index(drop=True, inplace=False)
    return interactions
