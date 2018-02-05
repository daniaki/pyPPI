#!/usr/bin/env python

"""
Author: Daniel Esposito
Contact: danielce90@gmail.com

This module provides functionality to mine interactions with labels from
HPRD flat files
"""

from itertools import product
from collections import OrderedDict

from ..base import SOURCE, TARGET, LABEL
from .uniprot import UNIPROT_ORD_KEY
from ..data_mining.tools import make_interaction_frame, process_interactions
from ..data import hprd_id_map, hprd_ptms
from ..database import begin_transaction
from ..database.managers import ProteinManager

SUBTYPES_TO_EXCLUDE = []

__PTM_FIELDS = OrderedDict()
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

__HPRD_XREF_FIELDS = OrderedDict()
__HPRD_XREF_FIELDS['hprd_id'] = 'na'
__HPRD_XREF_FIELDS['gene_symbol'] = 'na'
__HPRD_XREF_FIELDS['nucleotide_accession'] = 'na'
__HPRD_XREF_FIELDS['protein_accession'] = 'na'
__HPRD_XREF_FIELDS['entrezgene_id'] = 'na'
__HPRD_XREF_FIELDS['omim_id'] = 'na'
__HPRD_XREF_FIELDS['swissprot_id'] = 'na'
__HPRD_XREF_FIELDS['main_name'] = 'na'
__HPRD_XREF_INDEX = {k: i for (i, k) in enumerate(__HPRD_XREF_FIELDS.keys())}


psimi_mapping = {
    "in vitro": 'MI:0492',
    "invitro": 'MI:0492',
    "in vivo": 'MI:0493',
    "invivo": 'MI:0493',
    'yeast 2-hybrid': 'MI:0018'
}


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


def parse_ptm(file_input=None, header=False, col_sep='\t'):
    """
    Parse HPRD post_translational_modifications file.

    :param header: If file has header. Default is False.
    :param col_sep: Column separator.
    :return: List of PTMEntry objects.
    """
    ptms = []
    if file_input is None:
        lines = hprd_ptms()
    else:
        lines = file_input

    if header:
        next(lines)

    for line in lines:
        class_fields = __PTM_FIELDS.copy()
        xs = line.strip().split(col_sep)
        for k in __PTM_FIELDS.keys():
            data = xs[__PTM_INDEX[k]]
            if k == 'reference_id':
                data = [x.strip() for x in data.split(',')]
                # od = OrderedDict()
                # for pmid in data.split(','):
                #     od[pmid.strip()] = True
                # data = ','.join(od.keys())
            if k == "experiment_type":
                data = [x.strip() for x in data.split(';')]
            class_fields[k] = data
        ptms.append(PTMEntry(class_fields))
    return ptms


def parse_hprd_mapping(file_input=None, header=False, col_sep='\t'):
    """
    Parse a hprd mapping file into HPRDXref Objects.

    :param header: If file has header. Default is False.
    :param col_sep: Column separator.
    :return: Dict of HPRDXrefEntry objects indexed by hprd accession.
    """
    xrefs = {}
    if file_input is None:
        lines = hprd_id_map()
    else:
        lines = file_input
    if header:
        next(lines)

    for line in lines:
        class_fields = __HPRD_XREF_FIELDS.copy()
        xs = line.strip().split(col_sep)
        for k in __HPRD_XREF_FIELDS.keys():
            data = xs[__HPRD_XREF_INDEX[k]]
            if k == 'swissprot_id':
                data = [x.strip() for x in data.split(',')]
            class_fields[k] = data
        xrefs[xs[0]] = HPRDXrefEntry(class_fields)
    return xrefs


def hprd_to_dataframe(session, allow_self_edges=False, drop_nan='default',
                      allow_duplicates=False, exclude_labels=None,
                      min_label_count=None, merge=False, ptm_input=None,
                      mapping_input=None):
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
    ptms = parse_ptm(file_input=ptm_input)
    xrefs = parse_hprd_mapping(file_input=mapping_input)

    sources = []
    targets = []
    labels = []
    pmids = []
    experiment_types = []

    for ptm in ptms:
        label = ptm.modification_type.lower().replace(' ', '-')
        if ptm.enzyme_hprd_id == '-':
            ptm.enzyme_hprd_id = None
        if ptm.substrate_hprd_id == '-':
            ptm.substrate_hprd_id = None
        if label == '-':
            ptm.modification_type = None
            label = None

        has_nan = (ptm.enzyme_hprd_id is None) or \
            (ptm.substrate_hprd_id is None) or \
            (label is None)
        if has_nan and drop_nan:
            continue

        invalid = ('-', '', 'na', 'None', None)
        e_types = [x for x in ptm.experiment_type if x not in invalid]
        unique = OrderedDict()
        for e in e_types:
            unique[psimi_mapping[e]] = True
        e_types = ','.join(unique.keys())
        if not e_types:
            e_types = None

        reference_ids = [x for x in ptm.reference_id if x not in invalid]
        unique = OrderedDict()
        for r in reference_ids:
            unique[r] = True
        reference_ids = ','.join(unique.keys())
        if not reference_ids:
            reference_ids = None

        # Comment to break up colour monotony. Mmmmmm feel the Feng Shui...
        if ptm.enzyme_hprd_id is None:
            uniprot_sources = [None]
        else:
            uniprot_sources = xrefs[ptm.enzyme_hprd_id].swissprot_id
        if ptm.substrate_hprd_id is None:
            uniprot_targets = [None]
        else:
            uniprot_targets = xrefs[ptm.substrate_hprd_id].swissprot_id

        pm = ProteinManager(verbose=False, match_taxon_id=9606)
        for (s, t) in product(uniprot_sources, uniprot_targets):
            s_entry = pm.get_by_uniprot_id(session, s)
            if s_entry is None:
                s = None
            elif not s_entry.reviewed:
                s = None

            t_entry = pm.get_by_uniprot_id(session, t)
            if t_entry is None:
                t = None
            elif not t_entry.reviewed:
                t = None

            sources.append(s)
            targets.append(t)
            labels.append(label)
            pmids.append(reference_ids)
            experiment_types.append(e_types)

    meta_columns = OrderedDict()
    meta_columns['pubmed'] = pmids
    meta_columns['experiment_type'] = experiment_types
    interactions = make_interaction_frame(
        sources, targets, labels, **meta_columns
    )
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
