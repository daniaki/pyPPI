#!/usr/bin/env python

"""
Author: Daniel Esposito
Contact: danielce90@gmail.com

This module provides functionality to mine interactions with labels from
HPRD flat files
"""

from collections import OrderedDict as Od

from .uniprot import UNIPROT_ORD_KEY
from ..data_mining.tools import make_interaction_frame, process_interactions
from ..data import hprd_id_map, hprd_ptms

SUBTYPES_TO_EXCLUDE = []

__PTM_FIELDS = Od()
__PTM_FIELDS['substrate_hprd_id'] = 'na'
__PTM_FIELDS['substrate_gene_symbol'] = 'na'
__PTM_FIELDS['substrate_isoform_id'] = 'na'
__PTM_FIELDS['substrate_refseq_id'] = 'na'
__PTM_FIELDS['site'] = 'na'
__PTM_FIELDS['residue'] = 'na'
__PTM_FIELDS['enzyme_name'] = 'na'
__PTM_FIELDS['enzyme_hprd_id'] = 'na'
__PTM_FIELDS['modification_type'] = 'na'
__PTM_FIELDS['experiment_type'] = 'na'
__PTM_FIELDS['reference_id'] = 'na'
__PTM_INDEX = {k: i for (i, k) in enumerate(__PTM_FIELDS.keys())}

__HPRD_XREF_FIELDS = Od()
__HPRD_XREF_FIELDS['hprd_id'] = 'na'
__HPRD_XREF_FIELDS['gene_symbol'] = 'na'
__HPRD_XREF_FIELDS['nucleotide_accession'] = 'na'
__HPRD_XREF_FIELDS['protein_accession'] = 'na'
__HPRD_XREF_FIELDS['entrezgene_id'] = 'na'
__HPRD_XREF_FIELDS['omim_id'] = 'na'
__HPRD_XREF_FIELDS['swissprot_id'] = 'na'
__HPRD_XREF_FIELDS['main_name'] = 'na'
__HPRD_XREF_INDEX = {k: i for (i, k) in enumerate(__HPRD_XREF_FIELDS.keys())}


class PTMEntry(object):
    """
    Class to row data in the Post translational mod text file.
    """

    def __init__(self, dictionary):
        for k, v in dictionary.items():
            self.__dict__[k] = v

    def __repr__(self):
        line = 'PTMEntry():\n'
        for k, v in self.__dict__.items():
            line += '\t{0}:\t{1}\n'.format(k, v)
        return line

    def __str__(self):
        return self.__repr__()


class HPRDXrefEntry(object):
    """
    Class to row data in the HPRD text file.
    """

    def __init__(self, dictionary):
        for k, v in dictionary.items():
            self.__dict__[k] = v

    def __repr__(self):
        line = 'HPRDXrefEntry():\n'
        for k, v in self.__dict__.items():
            line += '\t{0}:\t{1}\n'.format(k, v)
        return line

    def __str__(self):
        return self.__repr__()


def parse_ptm(header=False, col_sep='\t'):
    """
    Parse HPRD post_translational_modifications file.

    :param header: If file has header. Default is False.
    :param col_sep: Column separator.
    :return: List of PTMEntry objects.
    """
    ptms = []
    lines = hprd_ptms()
    if header:
        next(lines)

    for line in lines:
        class_fields = __PTM_FIELDS.copy()
        xs = line.strip().split(col_sep)
        for k in __PTM_FIELDS.keys():
            data = xs[__PTM_INDEX[k]]
            if k == 'reference_id':
                data = data.split(',')
            class_fields[k] = data
        ptms.append(PTMEntry(class_fields))
    return ptms


def parse_hprd_mapping(header=False, col_sep='\t'):
    """
    Parse a hprd mapping file into HPRDXref Objects.

    :param header: If file has header. Default is False.
    :param col_sep: Column separator.
    :return: Dict of HPRDXrefEntry objects indexed by hprd accession.
    """
    xrefs = {}
    lines = hprd_id_map()
    if header:
        next(lines)

    for line in lines:
        class_fields = __HPRD_XREF_FIELDS.copy()
        xs = line.strip().split(col_sep)
        for k in __HPRD_XREF_FIELDS.keys():
            data = xs[__HPRD_XREF_INDEX[k]]
            if k == 'swissprot_id':
                data = data.split(',')
            class_fields[k] = data
        xrefs[xs[0]] = HPRDXrefEntry(class_fields)
    return xrefs


def hprd_to_dataframe(drop_nan=False, allow_self_edges=False,
                      allow_duplicates=False, exclude_labels=None,
                      min_label_count=None, merge=False):
    """
    Parse the FLAT_FILES from HPRD into a dataframe.

    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param exclude_labels: List of labels to remove.
    :param min_label_count: Remove labels below this count.
    :param merge: Merge PPIs with the same source and target but different
                  labels into the same entry.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    try:
        ptms = parse_ptm()
        xrefs = parse_hprd_mapping()
    except IOError as e:
        print(e)
        return make_interaction_frame([], [], [])

    sources = []
    targets = []
    labels = []
    for ptm in ptms:
        label = ptm.modification_type.lower().replace(' ', '-')
        if ptm.enzyme_hprd_id == '-':
            ptm.enzyme_hprd_id = None
        if ptm.substrate_hprd_id == '-':
            ptm.substrate_hprd_id = None
        if label == '-':
            ptm.modification_type = None
        sources.append(ptm.enzyme_hprd_id)
        targets.append(ptm.substrate_hprd_id)
        labels.append(label)

    # Since there's a many swissprot to hprd_id mapping, priortise P, Q and O.
    for i, source in enumerate(sources):
        if source is not None:
            sources[i] = sorted(
                xrefs[source].swissprot_id,
                key=lambda x: UNIPROT_ORD_KEY.get(x, 4))[0]

    for i, target in enumerate(targets):
        if target is not None:
            targets[i] = sorted(
                xrefs[target].swissprot_id,
                key=lambda x: UNIPROT_ORD_KEY.get(x, 4))[0]

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
