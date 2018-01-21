"""
This module provides functionality to query pathways by keyword, extract
pathways and parse pathways into a :pd.DataFrame: of interactions
with reaction labels.
"""
import logging
import pandas as pd
from collections import defaultdict

from itertools import product
from bioservices import KEGG
from bioservices import UniProt as UniProtMapper

from ..database.managers import ProteinManager
from ..data import load_kegg_to_up
from .tools import make_interaction_frame, process_interactions

kegg = KEGG(cache=True)
uniprot_mapper = UniProtMapper(cache=True)
kegg.organism = 'hsa'
links_to_include = ['PCrel', 'PPrel', 'ECrel', 'GGrel']
types_to_include = ['group', 'gene', 'enzyme']
logger = logging.getLogger("pyppi")

motif_pathway_ids = [
    'path:hsa04010',
    'path:hsa04151',
    'path:hsa01521'
]

subtypes_to_exclude = [
    'missing-interaction',
    'indirect-effect',
    'expression',
    'repression',
    'compound',
    'hidden-compound'
]


def reset_kegg():
    """
    Clear the KEGG connection Cache

    :return: None
    """
    kegg.clear_cache()


def reset_uniprot():
    """
    Clear the UniProt connection Cache

    :return: None
    """
    uniprot_mapper.clear_cache()


def download_pathway_ids(organism):
    """
    Query KEGG for a recent list of pathways for an organism.

    :param organism: A KEGG organism code. For example 'hsa'.
    :return: List of pathway ids.
    """
    kegg.organism = organism
    pathways = kegg.pathwayIds
    return pathways


def pathways_to_dataframe(session, pathway_ids, drop_nan=False,
                          allow_self_edges=False, allow_duplicates=False,
                          min_label_count=None, map_to_uniprot=False,
                          trembl=False, merge=False, verbose=False):
    """
    Download and parse a list of pathway ids into a dataframe of interactions.

    :param pathway_ids: List KEGG pathway accessions. Example ['path:hsa00010']
    :param drop_nan: Drop entries containing NaN in any column.
    :param allow_self_edges: Remove rows for which source is target.
    :param allow_duplicates: Remove exact copies accross columns.
    :param min_label_count: Remove labels with less than the specified count.
    :param map_to_uniprot: Map KEGG_IDs to uniprot.
    :param trembl: Use trembl acc when swissprot in unavailable.
                   Otherwise, kegg_id is considered unmappable.
    :param merge: Merge entries with identical source and target
                  columns during filter.
    :param verbose: True to logger.info progress.
    :return: DataFrame with 'source', 'target' and 'label' columns.
    """
    interaction_frames = [pathway_to_dataframe(p_id, verbose)
                          for p_id in pathway_ids]
    interactions = pd.concat(interaction_frames, ignore_index=True)
    if map_to_uniprot:
        interactions = keggid_to_uniprot(session, interactions, trembl=trembl)
    interactions = process_interactions(
        interactions=interactions,
        drop_nan=drop_nan,
        allow_self_edges=allow_self_edges,
        allow_duplicates=allow_duplicates,
        exclude_labels=subtypes_to_exclude,
        min_counts=min_label_count,
        merge=merge
    )
    return interactions


def pathway_to_dataframe(pathway_id, verbose=False):
    """
    Extract protein-protein interaction from KEGG pathway to
    a pandas DataFrame. NOTE: Interactions will be directionless.

    :param: str pathwayId: a valid pathway Id
    :return: DataFrame with columns source, target and label
    """
    res = kegg.parse_kgml_pathway(pathway_id)
    sources = []
    targets = []
    labels = []
    kegg_to_up = load_kegg_to_up()

    if verbose:
        logger.info("# --- Parsing pathway {} --- #".format(pathway_id))

    for rel in res['relations']:
        id1 = rel['entry1']
        id2 = rel['entry2']
        name1 = res['entries'][[x['id']
                                for x in res['entries']].index(id1)]['name']
        name2 = res['entries'][[x['id']
                                for x in res['entries']].index(id2)]['name']
        type1 = res['entries'][[x['id']
                                for x in res['entries']].index(id1)]['type']
        type2 = res['entries'][[x['id']
                                for x in res['entries']].index(id2)]['type']
        reaction_type = rel['name'].replace(' ', '-')
        link_type = rel['link']

        if link_type not in links_to_include:
            continue

        if type1 not in types_to_include or type2 not in types_to_include:
            continue

        for a in name1.strip().split(' '):
            for b in name2.strip().split(' '):
                valid_db_a = \
                    (kegg.organism in a or 'ec' in a) and (a in kegg_to_up)
                valid_db_b = \
                    (kegg.organism in b or 'ec' in b) and (b in kegg_to_up)

                if valid_db_a and valid_db_b:
                    sources.append(a)
                    targets.append(b)
                    labels.append(reaction_type)

    interactions = make_interaction_frame(sources, targets, labels)
    return interactions


def keggid_to_uniprot(session, interactions, trembl=False):
    """
    Map KEGG_ID accessions into uniprot. Takes the first if multiple accesssion
    are found, favoring SwissProt over TrEmbl

    :param interactions: DataFrame with columns source, target and label.
    :param trembl: Use Trembl if SwissProt is unavailable.
    :return: DataFrame with columns source, target and label.
    """
    filtered_map = {}
    sources = [a for a in interactions.source.values]
    targets = [b for b in interactions.target.values]
    unique_ids = list(set(sources) | set(targets))

    kegg_to_up = load_kegg_to_up()
    mapping = {k: kegg_to_up[k] for k in unique_ids}
    pm = ProteinManager(verbose=False, match_taxon_id=None)

    for kegg_id, uniprot_ls in mapping.items():
        status_ls = [
            pm.get_by_uniprot_id(session, a).reviewed for a in uniprot_ls
        ]
        status_ls = list(zip(uniprot_ls, status_ls))
        reviewed = [a for (a, s) in status_ls if s is True]
        unreviewed = [a for (a, s) in status_ls if s is False]
        if len(reviewed) > 0:
            if len(reviewed) > 1:
                logger.info(
                    'Warning: More that one reviewed '
                    'acc found for {}: {}'.format(kegg_id, reviewed)
                )
            filtered_map[kegg_id] = reviewed
        else:
            logger.info(
                'Warning: No reviewed acc found for {}.'.format(kegg_id)
            )
            if trembl and len(unreviewed) > 0:
                if len(reviewed) > 1:
                    logger.info(
                        'Warning: More that one unreviewed '
                        'acc found for {}: {}'.format(kegg_id, unreviewed)
                    )
                filtered_map[kegg_id] = unreviewed
            else:
                logger.info('Warning: Could not map {}.'.format(kegg_id))

    # Remaining kegg_ids that have not mapped to anything go to None
    zipped = list(zip(interactions.source.values,
                      interactions.target.values, interactions.label.values))
    sources = []
    targets = []
    labels = []
    for source, target, label in zipped:
        source_acc = filtered_map.get(source, [])
        target_acc = filtered_map.get(target, [])

        # Some Kegg_Ids genuinely map to more than 1 distinct uniprot
        # accession, so we use a list product to account for this.
        ppis = product(source_acc, target_acc)
        for (s, t) in ppis:
            sources.append(s)
            targets.append(t)
            labels.append(label)

    interactions = make_interaction_frame(sources, targets, labels)
    return interactions
