#!/usr/bin/env python

"""
Author: Daniel Esposito
Contact: danielce90@gmail.com

This module provides functionality to perform filtering and processing on
interaction dataframes.
"""
import logging
from collections import Counter
from collections import OrderedDict
from itertools import product

import pandas as pd
import numpy as np
from numpy import NaN

from ..base import SOURCE, TARGET, LABEL, NULL_VALUES
from ..base import PPI

logger = logging.getLogger("pyppi")


def _null_to_none(value):
    if str(value) in NULL_VALUES:
        return None
    elif value in NULL_VALUES:
        return None
    else:
        return value


def _make_ppi_tuples(sources, targets):
    sources = [_null_to_none(x) for x in sources]
    targets = [_null_to_none(x) for x in targets]
    ppis = [
        tuple(sorted((str(a), str(b))))
        for a, b in zip(sources, targets)
    ]
    return ppis


def _split_label(label, sep=','):
    return [l.strip() for l in str(_null_to_none(label)).strip().split(sep)]


def _format_label(label):
    if _null_to_none(label) is None:
        return str(None).lower()
    return label.strip().lower().replace(" ", "-")


def _format_labels(label_ls, sort_after_split=True, rejoin_after_split=False,
                   remove_duplicates_after_split=True, sep=','):
    labels = []
    for ls in label_ls:
        labels.append([_format_label(l) for l in _split_label(ls, sep)])

    if remove_duplicates_after_split:
        labels_ = []
        for ls in labels:
            ls_ = []
            for l in ls:
                if l not in ls_:
                    ls_.append(l)
            labels_.append(ls_)
        labels = labels_

    if sort_after_split:
        labels = [list(sorted(ls)) for ls in labels]

    if rejoin_after_split:
        labels = [sep.join(ls) for ls in labels]

    return labels


def xy_from_interaction_frame(interactions):
    """
    Utility function to convert an interaction dataframe into seperate
    X and y numpy arrays.

    :param interactions: pd.DataFrame
        DataFrame with 'source', 'target' and 'label' columns.

    :return: array-like, shape (n_samples, )
    """
    X = ppis_from_interaction_frame(interactions)
    y = labels_from_interaction_frame(interactions)
    return X, y


def labels_from_interaction_frame(interactions):
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
    labels = _format_labels(
        df[LABEL], sort_after_split=True, remove_duplicates_after_split=True
    )
    labels = [[_null_to_none(l) for l in ls] for ls in labels]
    return labels


def ppis_from_interaction_frame(interactions):
    """
    Utility function to create an iterable of tuples from an interaction
    dataframe.

    :param interactions: pd.DataFrame
        DataFrame with 'source', 'target' and 'label' columns.
    :param use_set: boolean
        Use True to return a set of PPIs

    :return: List or Set
        List or Set of PPI objects.
    """
    df = interactions
    ppis = [
        (_null_to_none(a), _null_to_none(b))
        for a, b in zip(df[SOURCE], df[TARGET])
    ]
    return ppis


def make_interaction_frame(sources, targets, labels, **additional_columns):
    """
    Wrapper to construct a non-directional PPI dataframe.

    :param sources: Interactor ID that is the source node.
    :param targets: Interactor ID that is the target node.
    :param labels: Edge label for the interaction.
    :return: DataFrame with SOURCE, TARGET and LABEL columns.
    """
    ppis = _make_ppi_tuples(sources, targets)
    sources = [a for a, _ in ppis]
    targets = [b for _, b in ppis]
    labels = [_format_label(l) for l in labels]
    interactions = {
        SOURCE: sources,
        TARGET: targets,
        LABEL: labels
    }

    df_columns = [SOURCE, TARGET, LABEL]
    for column_key in sorted(additional_columns.keys()):
        interactions[column_key] = additional_columns[column_key]
        df_columns.append(column_key)

    return normalise_nan(
        pd.DataFrame(data=interactions, columns=df_columns)
    )


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
    df = interactions
    base_columns = [SOURCE, TARGET, LABEL]
    meta_columns = [col for col in df.columns if col not in base_columns]
    metadata = {c: [] for c in meta_columns}

    zipped = zip(
        df[SOURCE], df[TARGET], df[LABEL], *[df[c] for c in meta_columns]
    )
    for a, b, l, *extra in zipped:
        sources = accession_map.get(a, [])
        targets = accession_map.get(b, [])
        for (s, t) in product(sources, targets):
            ppis.append(tuple(sorted((str(s), str(t)))))
            labels.append(l)
            for i, col in enumerate(meta_columns):
                metadata[col].append(extra[i])

    sources = [a for a, _ in ppis]
    targets = [b for _, b in ppis]
    new_interactions = make_interaction_frame(
        sources, targets, labels, **metadata
    )
    new_interactions = process_interactions(
        interactions=new_interactions,
        drop_nan=drop_nan,
        allow_self_edges=allow_self_edges,
        allow_duplicates=allow_duplicates,
        exclude_labels=None,
        min_counts=min_counts,
        merge=merge
    )
    return normalise_nan(new_interactions)


def normalise_nan(interactions, replace=NULL_VALUES, replace_with=None):
    """
    Replace values appearing in interactions defined in replace with None.

    Arguments
    ---------
    interactions: DataFrame with 'source', 'target' and 'label' columns.
    replace: Tuple of values to replace.

    Returns
    -------
    DataFrame with `null` values in replace, replaced with `None`
    """
    df = interactions
    for value in replace:
        df = df.replace(
            to_replace=[value], value=[replace_with], inplace=False
        )
    return df


def remove_nan(interactions, subset=[SOURCE, TARGET, LABEL]):
    """
    Drop interactions with missing source, target or labels as indicated
    by None/None.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = normalise_nan(interactions, replace_with=np.NaN)
    df.dropna(axis=0, how='any', inplace=True, subset=subset)
    selector = df.index.values
    new_df = interactions.loc[selector, ].reset_index(drop=True, inplace=False)
    return new_df


def remove_intersection(interactions, other, use_label=True):
    """
    Remove any interaction from `interactions` appearing in `other`.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param other: DataFrame with 'source', 'target' and 'label' columns.
    :param use_label: By default only look for (source, target, label) in the 
                      `other` dataframe. Otherwise look for (source, target).
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    selector = []
    ppis_in_other = OrderedDict()
    other_ppis = list(zip(other[SOURCE], other[TARGET], other[LABEL]))
    for (source, target, label) in other_ppis:
        a, b = sorted([str(source), str(target)])
        if use_label:
            for (s, t, l) in product([a], [b], _split_label(label)):
                ppis_in_other[(s, t, l)] = True
        else:
            ppis_in_other[(a, b)] = True

    df = interactions.reset_index(drop=True, inplace=False)
    interactions_ppis = list(zip(df[SOURCE], df[TARGET], df[LABEL]))
    for i, (source, target, label) in enumerate(interactions_ppis):
        a, b = sorted([str(source), str(target)])
        if use_label:
            for (s, t, l) in product([a], [b], _split_label(label)):
                if ppis_in_other.get((s, t, l)) is None:
                    selector.append(True)
                else:
                    selector.append(False)
        else:
            if ppis_in_other.get((a, b)) is None:
                selector.append(True)
            else:
                selector.append(False)

    selector = np.asarray(selector)
    df = interactions.loc[selector, ].reset_index(drop=True, inplace=False)
    removed = interactions.loc[~selector, ].reset_index(
        drop=True, inplace=False)
    return df, removed


def remove_labels(interactions, labels_to_exclude):
    """
    Remove PPIs with a subtype in exclusions list.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param labels_to_exclude: string list of subtypes to exclude.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    def exclude(l):
        if str(l) == str(np.NaN):
            return str(np.NaN) in labels_to_exclude
        else:
            return l in labels_to_exclude
    selector = [
        not exclude(l)
        for l in interactions[LABEL].values
    ]
    df = interactions.loc[selector, ]
    df = df.reset_index(drop=True, inplace=False)
    return df


def remove_min_counts(interactions, min_count):
    """
    Remove all PPIs with labels that have overall count less
    than some threshold. Ignores labels that are NaN/null valued.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :param min_count: Minimum count threshold to keep label.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    counts = Counter(interactions.label.values)
    labels_to_exclude = set([k for k, v in counts.items() if v < min_count])
    df = remove_labels(interactions, labels_to_exclude)
    df.reset_index(drop=True, inplace=True)
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
    base_columns = [SOURCE, TARGET, LABEL]
    extra_columns = [col for col in df.columns if col not in base_columns]

    merged_ppis = OrderedDict()
    ppis = [
        tuple(sorted((str(s), str(t))))
        for (s, t) in zip(df[SOURCE], df[TARGET])
    ]
    zipped = zip(
        ppis, df[LABEL], *[df[c] for c in extra_columns]
    )

    for (s, t), label, *extra in zipped:
        if merged_ppis.get((s, t)) is None:
            merged_ppis[(s, t)] = {k: [] for k in extra_columns + [LABEL]}
        # Some labels might be merged already, so split them first before
        # formatting.
        merged_ppis[(s, t)][LABEL].extend(
            [
                _format_label(l) for l in _split_label(label)
                if _format_label(l) not in NULL_VALUES
            ]
        )
        # Same as above, some additional data may have already been merged.
        for i, column in enumerate(extra_columns):
            merged_ppis[(s, t)][column].extend(
                [
                    e.strip() for e in _split_label(extra[i])
                    if e.strip() not in NULL_VALUES
                ]
            )

    # Format the labels by set, sorted and then comma delimiting.
    sources = [ppi[0] for ppi in merged_ppis.keys()]
    targets = [ppi[1] for ppi in merged_ppis.keys()]
    labels = [
        ','.join(list(sorted(set(_format_label(l) for l in ls)))) or None
        for ls in [row[LABEL] for row in merged_ppis.values()]
    ]

    # Format the additonal columns by set and sorting.
    additional = {c: [] for c in extra_columns}
    for column in extra_columns:
        data = [
            ','.join(list(sorted(set(d.strip() for d in ls)))) or None
            for ls in [row[column] for row in merged_ppis.values()]
        ]
        additional[column] = data

    interactions = make_interaction_frame(
        sources, targets, labels, **additional
    )
    return interactions


def remove_common_ppis(df_1, df_2):
    """
    Collects all ppis which are common to df_1 and df_2 by looking at the
    SOURCE and TARGET columns. Removes these common ppis from df_1 and df_2
    and collects them into a new dataframe.

    Note: Expected the SOURCE and TARGET columns to be pre-sorted, otherwise
    this method will not detect permuted ppis (A, B)/(B, A).

    :param df_1: 
        DataFrame with 'source', 'target' and 'label' columns.
    :param df_2: 
        DataFrame with 'source', 'target' and 'label' columns.
    :return: 
        tuple of DataFrames (df_1_unique, df_2_unique, common)
    """
    ppis_df_1 = list(zip(df_1[SOURCE], df_1[TARGET]))
    ppis_df_2 = list(zip(df_2[SOURCE], df_2[TARGET]))
    common_ppis = set(ppis_df_1) & set(ppis_df_2)

    unique_idx = set()
    common_idx = set()
    for idx, ppi in enumerate(ppis_df_1):
        if ppi in common_ppis:
            common_idx.add(idx)
        else:
            unique_idx.add(idx)
    df_1_unique = df_1.loc[unique_idx, :]
    df_1_common = df_1.loc[common_idx, :]

    unique_idx = set()
    common_idx = set()
    for idx, ppi in enumerate(ppis_df_2):
        if ppi in common_ppis:
            common_idx.add(idx)
        else:
            unique_idx.add(idx)
    df_2_unique = df_2.loc[unique_idx, :]
    df_2_common = df_2.loc[common_idx, :]

    df_1_unique.reset_index(drop=True, inplace=True)
    df_2_unique.reset_index(drop=True, inplace=True)
    common = pd.concat([df_1_common, df_2_common], ignore_index=True)
    common.reset_index(drop=True, inplace=True)

    return df_1_unique, df_2_unique, common


def remove_duplicates(interactions):
    """
    Remove rows with identical source, target and label column entries.

    :param interactions: DataFrame with 'source', 'target' and 'label' columns.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    df = interactions
    merged_ppis = OrderedDict()
    ppis = [
        tuple(sorted((str(s), str(t))))
        for (s, t) in zip(df[SOURCE], df[TARGET])
    ]
    merged = any([len(_split_label(l)) > 1 for l in df[LABEL]])
    base_columns = [SOURCE, TARGET, LABEL]
    extra_columns = [col for col in df.columns if col not in base_columns]
    zipped = zip(
        ppis, df[LABEL], *[df[c] for c in extra_columns]
    )

    if merged:
        assert len(ppis) == len(df[LABEL].values)
        for (s, t), label, *extra in zipped:
            for l in set(_split_label(label)):
                if merged_ppis.get((s, t, l)) is None:
                    merged_ppis[(s, t, l)] = {c: [] for c in extra_columns}
                for i, column in enumerate(extra_columns):
                    merged_ppis[(s, t, l)][column].extend(
                        [
                            e.strip() for e in _split_label(extra[i])
                            if e.strip() not in NULL_VALUES
                        ]
                    )
    else:
        assert len(ppis) == len(df.label.values)
        for (s, t), label, *extra in zipped:
            if label in NULL_VALUES or str(label) in NULL_VALUES:
                label = str(None)
            if merged_ppis.get((s, t, label)) is None:
                merged_ppis[(s, t, label)] = {c: [] for c in extra_columns}
            for i, column in enumerate(extra_columns):
                merged_ppis[(s, t, label)][column].extend(
                    [
                        e.strip() for e in _split_label(extra[i])
                        if e.strip() not in NULL_VALUES
                    ]
                )

    sources = [ppi[0] for ppi in merged_ppis.keys()]
    targets = [ppi[1] for ppi in merged_ppis.keys()]
    labels = [ppi[2] for ppi in merged_ppis.keys()]

    # Format the additonal columns by set and sorting.
    additional = {c: [] for c in extra_columns}
    for column in extra_columns:
        data = [
            ','.join(list(sorted(set(d.strip() for d in ls)))) or None
            for ls in [row[column] for row in merged_ppis.values()]
        ]
        additional[column] = data

    interactions = make_interaction_frame(
        sources, targets, labels, **additional
    )

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
    interactions = normalise_nan(
        interactions.reset_index(drop=True, inplace=False)
    )
    if drop_nan == 'default':
        drop_nan = [SOURCE, TARGET, LABEL]
    elif (not isinstance(drop_nan, list)) and (not drop_nan is None):
        raise TypeError(
            "`drop_nan` must be either a list of columns to search for None "
            "values over, None to bypass this process or 'default' to specify "
            "the default columns."
        )

    if drop_nan:
        interactions = remove_nan(interactions, subset=drop_nan)
    if not allow_self_edges:
        interactions = remove_self_edges(interactions)
    if not allow_duplicates:
        interactions = remove_duplicates(interactions)
    if exclude_labels:
        interactions = remove_labels(interactions, exclude_labels)
    if min_counts:  # This must come before merge as merge will concat labels.
        interactions = remove_min_counts(interactions, min_count=min_counts)
    if merge:
        interactions = merge_labels(interactions)

    interactions = interactions.reset_index(drop=True, inplace=False)
    return interactions
