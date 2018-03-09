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
from bioservices import UniProt

from ..database.models import Protein
from ..base.constants import SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE

from .tools import make_interaction_frame, process_interactions

links_to_include = ['PCrel', 'PPrel', 'ECrel', 'GGrel']
types_to_include = ['group', 'gene', 'enzyme']
logger = logging.getLogger("pyppi")

motif_pathway_ids = [
    'path:hsa04010',
    'path:hsa04151',
    'path:hsa01521'
]

subtypes_to_exclude = [
    'Missing-interaction',
    'Indirect-effect',
    'Expression',
    'Repression',
    'Compound',
    'Hidden-compound'
]


def reset_kegg():
    """
    Clear the KEGG connection Cache.
    """
    kegg = KEGG(cache=True)
    kegg.clear_cache()


def reset_uniprot():
    """
    Clear the UniProt connection Cache.
    """
    uniprot_mapper = UniProt(cache=True)
    uniprot_mapper.clear_cache()


def kegg_to_uniprot(fr='hsa', cache=False):
    """Downloads a mapping from a `KEGG` database to `UniProt`, including
    both `TrEMBL` and `SwissProt`.

    Parameters:
    ----------
    fr : str, optional, default: 'hsa'
        KEGG database identifier to convert. Defaults to 'hsa'.

    cache : bool, optional, default: False
        If True, results are cached by `bioservices`. This can save
        time but you will eventually miss out on new database releases if
        your cache is old.

    Returns
    -------
    `dict`
        Mapping from `KEGG` identifiers to a list of `UniProt` accessions.
    """

    kegg = KEGG(cache=cache)
    mapping = kegg.conv(fr, 'uniprot')

    parsed_mapping = {}
    for upid, org in mapping.items():
        upid = upid.split(':')[1]  # remove the 'up:' prefix
        if org in parsed_mapping:
            parsed_mapping[org] += [upid]
        else:
            parsed_mapping[org] = [upid]
    return parsed_mapping


def download_pathway_ids(organism, cache=False):
    """
    Query KEGG for a recent list of pathways for an organism.

    Parameters
    ----------
    organism: str
        A KEGG organism code. For example 'hsa'.

    cache : bool, optional, default: False
        If True, results are cached by `bioservices`. This can save
        time but you will eventually miss out on new database releases if
        your cache is old.

    Returns
    -------
    `list`
        List of str pathway identifiers.
    """
    kegg = KEGG(cache=cache)
    kegg.organism = organism
    pathways = kegg.pathwayIds
    return pathways


def pathways_to_dataframe(pathway_ids=None, org='hsa', drop_nan=False,
                          allow_self_edges=False, allow_duplicates=False,
                          min_label_count=None, map_to_uniprot=False,
                          trembl=False, merge=False, verbose=False,
                          cache=False):
    """Download and parse a list of pathway ids into a dataframe of 
    interactions.

    Interaction dataframe will have the columns 'source', 'target', 'label', 
    'pubmed', and 'experiment_type'. The latter are the psimi accessions
    associated with each pmid.

    Parameters
    ----------
    pathway_ids : list, optional, default: None
        List KEGG pathway accessions. Example ['path:hsa00010']. If None,
        then all pathways are downloaded for `org`.

    org : str or None, optioanl, default: 'hsa'
        A KEGG organism code to download pathways for. Used to download 
        pathways for the specified org if `pathway_ids` is None. 

    drop_nan : list, str or bool, default: None
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

    map_to_uniprot : bool, optional, default: False
        Map KEGG_IDs in `source` and `target` to `UniProt` accessions. If 
        a source or target maps to multiple accessions then a product is taken
        and multiple new rows may result. Example if A in (A, B) maps to 
        [C, D] then the mapping process will create two rows: (C, B) and 
        (D, B). Label, pubmed and experiment type information is copied into 
        both rows.

    trembl : bool, optional, default: False
        If True, during the mapping process, keeps mapped rows containing 
        TrEMBL accessions in either `source` or `target`. Otherwise, these
        rows are deleted.

    merge : bool, optional, default: False
        If True, merges entries with identical source and target columns. If 
        different annotations are seen in the `pubmed` and `experiment_type`
        columns, then these are also merged so they are not lost.

    verbose : bool, optional, default: False
        If True, logs messages to stdout to inform of current progress.

    cache : bool, optional, default: False
        If True, HTTP responses are cached by `bioservices`. This can save
        time but you will eventually miss out on new database releases if
        your cache is old.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with 'source', 'target', 'label', 'pubmed', and 
        'experiment_type' columns.
    """
    if pathway_ids is None:
        pathway_ids = download_pathway_ids(org, cache)

    interaction_frames = [
        pathway_to_dataframe(p_id, org, verbose, cache)
        for p_id in pathway_ids
    ]
    interactions = pd.concat(interaction_frames, ignore_index=True)
    if map_to_uniprot:
        interactions = keggid_to_uniprot(
            interactions, trembl=trembl, cache=cache, verbose=verbose
        )

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


def pathway_to_dataframe(pathway_id, org='hsa', verbose=False, cache=False):
    """
    Extract protein-protein interaction from KEGG pathway to
    a pandas DataFrame. NOTE: Interactions will be directionless.

    Parameters
    ----------
    pathway_id : str
        Pathway identifier to parse into a dataframe. Example: 'path:hsa00010'

    org : str or None, optioanl, default: 'hsa'
        If supplied, filters out all interactions with identifiers that
        are not in the dictionary created from :func:`kegg_to_uniprot`. If None,
        all interactions are parsed regardless of mappability to UniProt.

    verbose : bool, optional, default: False
        If True, logs messages to stdout to inform of current progress.

    cache : bool, optional, default: False
        If True, HTTP responses are cached by `bioservices`. This can save
        time but you will eventually miss out on new database releases if
        your cache is old.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with 'source', 'target', 'label', 'pubmed', and 
        'experiment_type' columns.

    """
    kegg = KEGG(cache=cache)
    kegg.organism = org
    kegg_to_up = kegg_to_uniprot(org, cache)
    res = kegg.parse_kgml_pathway(pathway_id)
    sources = []
    targets = []
    labels = []

    if verbose:
        logger.info("Parsing pathway {}".format(pathway_id))

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
                valid_db_a = (kegg.organism in a or 'ec' in a)
                valid_db_b = (kegg.organism in b or 'ec' in b)

                valid_db_a &= (a in kegg_to_up)
                valid_db_b &= (b in kegg_to_up)

                if valid_db_a and valid_db_b:
                    sources.append(a)
                    targets.append(b)
                    labels.append(reaction_type)

    interactions = make_interaction_frame(sources, targets, labels)
    return interactions


def keggid_to_uniprot(interactions, verbose=False, trembl=False, cache=False):
    """
    Map KEGG_ID accessions into uniprot. Performs a product operation
    to product multiple new interactions from a single interaction if
    multiple possible mappings are found.

    Parameters
    ----------
    interactions : :class:`pd.DataFrame`
        DataFrame with 'source', 'target', 'label', 'pubmed', and 
        'experiment_type' columns.

    trembl : bool, optional, default: False
        If True, during the mapping process, keeps mapped rows containing 
        TrEMBL accessions in either `source` or `target`. Otherwise, these
        rows are deleted.

    verbose : bool, optional, default: False
        If True, logs messages regarding mapping warnings and other information.

    cache : bool, optional, default: False
        If True, HTTP responses are cached by `bioservices`. This can save
        time but you will eventually miss out on new database releases if
        your cache is old.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with 'source', 'target', 'label', 'pubmed', and 
        'experiment_type' columns.
    """
    filtered_map = {}
    sources = [a for a in interactions.source.values]
    targets = [b for b in interactions.target.values]
    unique_ids = list(set(sources) | set(targets))

    mapper = UniProt(cache=cache)
    mapping = mapper.mapping(fr='KEGG_ID', to='ACC', query=unique_ids)

    for kegg_id, uniprot_ls in mapping.items():
        # Check that the accessions are actually in the database.
        # If not, ignore them and warn the user.
        proteins_all = [Protein.get_by_uniprot_id(a) for a in uniprot_ls]
        proteins_valid = []
        zipped = list(zip(proteins_all, uniprot_ls))
        for p, accession in zipped:
            if p is None:
                uniprot_ls.remove(accession)
                if verbose:
                    logger.warning(
                        "No protein for '{}' found in the database. Consider "
                        "downloading the latest UniProt dat files and "
                        "updating the database.".format(accession)
                    )
            else:
                proteins_valid.append(p)

        # Only process the proteins in the database.
        status_ls = [p.reviewed for p in proteins_valid]
        status_ls = list(zip(uniprot_ls, status_ls))
        reviewed = [a for (a, s) in status_ls if s is True]
        unreviewed = [a for (a, s) in status_ls if s is False]
        if len(reviewed) > 0:
            if len(reviewed) > 1:
                if verbose:
                    logger.warning(
                        'More that one reviewed '
                        'acc found for {}: {}'.format(kegg_id, reviewed)
                    )
            filtered_map[kegg_id] = reviewed
        else:
            if verbose:
                logger.warning(
                    'No reviewed acc found for {}.'.format(kegg_id)
                )
            if trembl and len(unreviewed) > 0:
                if len(reviewed) > 1:
                    if verbose:
                        logger.warning(
                            'More that one unreviewed '
                            'acc found for {}: {}'.format(kegg_id, unreviewed)
                        )
                filtered_map[kegg_id] = unreviewed
            else:
                if verbose:
                    logger.warning('Could not map {}.'.format(kegg_id))

    # Remaining kegg_ids that have not mapped to anything go to None
    zipped = list(
        zip(
            interactions[SOURCE].values,
            interactions[TARGET].values,
            interactions[LABEL].values,
            interactions[PUBMED].values,
            interactions[EXPERIMENT_TYPE].values
        )
    )
    sources = []
    targets = []
    labels = []
    pmids = []
    psimis = []
    for source, target, label, pmid, psimi in zipped:
        source_acc = filtered_map.get(source, [])
        target_acc = filtered_map.get(target, [])

        # Some Kegg_Ids genuinely map to more than 1 distinct uniprot
        # accession, so we use a list product to account for this.
        ppis = product(source_acc, target_acc)
        for (s, t) in ppis:
            sources.append(s)
            targets.append(t)
            labels.append(label)
            pmids.append(pmid)
            psimis.append(psimi)

    interactions = make_interaction_frame(
        sources, targets, labels, pmids, psimis
    )
    return interactions
