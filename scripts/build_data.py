"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  build_data.py [--clear_cache] [--n_jobs=J] [--verbose]
  build_data.py -h | --help

Options:
  -h --help  Show this screen.
  --n_jobs=J  Number of processes to run in parallel [default: 1]
  --clear_cache  Delete previous bioservices KEGG/UniProt cache
  --verbose  Log information and warning output to console.
"""

import os
import pandas as pd
import logging
from Bio import SwissProt
from joblib import Parallel, delayed
from docopt import docopt

from pyppi.base.utilities import delete_cache, is_null
from pyppi.base.utilities import generate_interaction_tuples
from pyppi.base.arg_parsing import parse_args
from pyppi.base.constants import SOURCE, TARGET, LABEL
from pyppi.base.constants import PUBMED, EXPERIMENT_TYPE
from pyppi.base.log import create_logger

from pyppi.base.file_paths import bioplex_network_path, pina2_network_path
from pyppi.base.file_paths import innate_i_network_path, innate_c_network_path
from pyppi.base.file_paths import interactome_network_path, full_training_network_path
from pyppi.base.file_paths import kegg_network_path, hprd_network_path
from pyppi.base.file_paths import testing_network_path, training_network_path
from pyppi.base.file_paths import default_db_path

from pyppi.base.io import save_uniprot_accession_map, save_network_to_path
from pyppi.base.io import bioplex_v4, pina2_mitab, innate_curated, innate_imported
from pyppi.base.io import uniprot_sprot, uniprot_trembl

from pyppi.database import delete_database, db_session, cleanup_module
from pyppi.database.models import Protein, Interaction
from pyppi.database.models import Pubmed, Psimi, Reference
from pyppi.database.utilities import create_interaction, uniprotid_entry_map

from pyppi.data_mining.uniprot import parse_record_into_protein
from pyppi.data_mining.uniprot import batch_map
from pyppi.data_mining.generic import bioplex_func
from pyppi.data_mining.generic import pina_mitab_func, innate_mitab_func
from pyppi.data_mining.generic import generic_to_dataframe
from pyppi.data_mining.hprd import hprd_to_dataframe
from pyppi.data_mining.tools import process_interactions, make_interaction_frame
from pyppi.data_mining.tools import remove_common_ppis, remove_labels
from pyppi.data_mining.tools import map_network_accessions
from pyppi.data_mining.kegg import download_pathway_ids, pathways_to_dataframe
from pyppi.data_mining.ontology import get_active_instance
from pyppi.data_mining.psimi import get_active_instance as load_mi_ontology
from pyppi.data_mining.features import compute_interaction_features


logger = create_logger("scripts", logging.INFO)


if __name__ == "__main__":
    args = docopt(__doc__)
    args = parse_args(args)
    n_jobs = args['n_jobs']
    clear_cache = args['clear_cache']
    verbose = args['verbose']

    # Setup the protein table in the database
    # ----------------------------------------------------------------------- #
    if clear_cache:
        logger.info("Clearing Biopython/Bioservices cache.")
        delete_cache()

    logger.info("Clearing existing database tables.")
    delete_database(db_session)

    logger.info("Parsing UniProt and PSI-MI into database.")
    records = list(SwissProt.parse(uniprot_sprot())) + \
        list(SwissProt.parse(uniprot_trembl()))
    proteins = [parse_record_into_protein(r) for r in records]

    psimi_objects = []
    mi_ont = load_mi_ontology()
    for key, term in mi_ont.items():
        obj = Psimi(accession=key, description=term.name)
        psimi_objects.append(obj)

    try:
        db_session.add_all(proteins + psimi_objects)
        db_session.commit()
    except:
        db_session.rollback()
        raise

    # Construct all the networks
    # ----------------------------------------------------------------------- #
    logger.info("Building KEGG interactions.")
    pathways = download_pathway_ids('hsa')
    kegg = pathways_to_dataframe(
        pathway_ids=pathways,
        map_to_uniprot=True,
        drop_nan='default',
        allow_self_edges=True,
        allow_duplicates=False,
        org='hsa',
        cache=True,
        verbose=verbose
    )

    logger.info("Building HPRD interactions.")
    hprd = hprd_to_dataframe(
        drop_nan='default',
        allow_self_edges=True,
        allow_duplicates=False
    )

    logger.info("Building Interactome interactions.")
    bioplex = generic_to_dataframe(
        f_input=bioplex_v4(),
        parsing_func=bioplex_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    pina2_mitab = generic_to_dataframe(
        f_input=pina2_mitab(),
        parsing_func=pina_mitab_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    innate_c = generic_to_dataframe(
        f_input=innate_curated(),
        parsing_func=innate_mitab_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    innate_i = generic_to_dataframe(
        f_input=innate_imported(),
        parsing_func=innate_mitab_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    logger.info("Mapping to most recent uniprot accessions.")
    # Get a set of all the unique uniprot accessions
    networks = [kegg, hprd, bioplex, pina2_mitab, innate_i, innate_c]
    sources = set(p for df in networks for p in df.source.values)
    targets = set(p for df in networks for p in df.target.values)
    accessions = list(sources | targets)
    accession_mapping = batch_map(
        cache=True,
        verbose=verbose,
        allow_download=False,
        accessions=accessions,
        keep_unreviewed=True,
        match_taxon_id=9606
    )
    save_uniprot_accession_map(accession_mapping)

    logger.info("Mapping each network to the most recent uniprot accessions.")
    kegg = map_network_accessions(
        interactions=kegg, accession_map=accession_mapping,
        drop_nan='default', allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    hprd = map_network_accessions(
        interactions=hprd, accession_map=accession_mapping,
        drop_nan='default', allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    pina2_mitab = map_network_accessions(
        interactions=pina2_mitab, accession_map=accession_mapping,
        drop_nan=[SOURCE, TARGET], allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    bioplex = map_network_accessions(
        interactions=bioplex, accession_map=accession_mapping,
        drop_nan=[SOURCE, TARGET], allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    innate_c = map_network_accessions(
        interactions=innate_c, accession_map=accession_mapping,
        drop_nan=[SOURCE, TARGET], allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    innate_i = map_network_accessions(
        interactions=innate_i, accession_map=accession_mapping,
        drop_nan=[SOURCE, TARGET], allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    networks = [hprd, kegg, bioplex, pina2_mitab, innate_i, innate_c]

    logger.info("Saving raw networks.")
    save_network_to_path(kegg, kegg_network_path)
    save_network_to_path(hprd, hprd_network_path)
    save_network_to_path(pina2_mitab, pina2_network_path)
    save_network_to_path(bioplex, bioplex_network_path)
    save_network_to_path(innate_i, innate_i_network_path)
    save_network_to_path(innate_c, innate_c_network_path)

    logger.info("Building and saving processed networks.")
    hprd_test_labels = ['Dephosphorylation', 'Phosphorylation']
    hprd_train_labels = set(
        [l for l in hprd[LABEL] if l not in hprd_test_labels]
    )
    train_hprd = remove_labels(hprd, hprd_test_labels)
    testing = remove_labels(hprd, hprd_train_labels)
    training = pd.concat([kegg, train_hprd], ignore_index=True).reset_index(
        drop=True, inplace=False)

    # Some ppis will be the same between training/testing sets but
    # with different labels. Put all the ppis appearing in testing
    # with a different label compared to the same instance in training
    # into the training set. This way we can keep the testing and
    # training sets completely disjoint.
    training, testing, common = remove_common_ppis(
        df_1=training,
        df_2=testing
    )
    full_training = pd.concat(
        [training, testing, common],
        ignore_index=True
    ).reset_index(
        drop=True, inplace=False
    )

    testing = process_interactions(
        interactions=testing,
        drop_nan='default', allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    training = process_interactions(
        interactions=training,
        drop_nan='default', allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    full_training = process_interactions(
        interactions=full_training,
        drop_nan='default', allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    common = process_interactions(
        interactions=common,
        drop_nan='default', allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=None, merge=True
    )

    interactome_networks = [bioplex, pina2_mitab, innate_i, innate_c]
    interactome = pd.concat(interactome_networks, ignore_index=True)
    interactome = process_interactions(
        interactions=interactome, drop_nan=[SOURCE, TARGET],
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=None, merge=True
    )
    save_network_to_path(interactome, interactome_network_path)
    save_network_to_path(training, training_network_path)
    save_network_to_path(testing, testing_network_path)
    save_network_to_path(full_training, full_training_network_path)

    logger.info("Saving Interaction records to database.")
    protein_map = uniprotid_entry_map()
    ppis = [
        (protein_map[a], protein_map[b])
        for network in [full_training, interactome]
        for (a, b) in zip(network[SOURCE], network[TARGET])
    ]

    feature_map = {}
    logger.info("Computing features.")
    features_ls = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(compute_interaction_features)(source, target)
        for (source, target) in ppis
    )
    for (source, target), features in zip(ppis, features_ls):
        feature_map[(source.uniprot_id, target.uniprot_id)] = features

    # Create and save all the psimi and pubmed objects if they don't already
    # exist in the database. If the psi-mi obo parsed in the first step
    # is up-to-date then the following code should not need to create any
    # new entries.
    logger.info("Updating Pubmed/PSI-MI database entries.")
    psimi_map = {p.accession: p for p in Psimi.query.all()}
    objects = []
    networks = [full_training, interactome]

    # Get all pmids from the parsed networks.
    pmids = set([
        p.upper()
        for ls in pd.concat(networks, ignore_index=True)[PUBMED]
        for p in str(ls).split(',') if not is_null(p)
    ])
    for pmid in pmids:
        if is_null(pmid):
            continue
        else:
            objects.append(Pubmed(accession=pmid))

    # Get all the psimi-groups from the parsed networks, then parse each
    # group individually.
    psimis = set([
        p.upper()
        for ls in pd.concat(networks, ignore_index=True)[EXPERIMENT_TYPE]
        for p in str(ls).split(',') if not is_null(p)
    ])
    for psimi_group in psimis:
        psimis = psimi_group.split('|')
        for p in psimis:
            if is_null(p):
                continue
            elif p not in psimi_map:
                if verbose:
                    logger.info("Creating new PSI-MI entry '{}'.".format(p))
                desc = None if p not in mi_ont else mi_ont[p].name
                if verbose and desc is None:
                    logger.info(
                        "No data found for PSI-MI entry '{}'. Try downloading "
                        "the most recent releases using the setup "
                        "script.".format(p)
                    )
                objects.append(Psimi(accession=p, description=desc))

    try:
        db_session.add_all(objects)
        db_session.commit()
    except:
        db_session.rollback()
        raise

    logger.info("Creating Interaction database entries.")
    interactions = {}
    for interaction in Interaction.query.all():
        a = Protein.query.get(interaction.source)
        b = Protein.query.get(interaction.target)
        uniprot_a, uniprot_b = sorted([a.uniprot_id, b.uniprot_id])
        interactions[(uniprot_a, uniprot_b)] = interaction

    # Training should only update the is_training to true and leave other
    # boolean fields alone.
    logger.info("Creating training interaction entries.")
    generator = generate_interaction_tuples(training)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        uniprot_a, uniprot_b = sorted([uniprot_a, uniprot_b])
        source = protein_map[uniprot_a]
        target = protein_map[uniprot_b]
        class_kwargs = feature_map[(uniprot_a, uniprot_b)]
        class_kwargs["is_training"] = True
        entry = create_interaction(
            source, target, label, session=db_session, save=False,
            commit=False, verbose=verbose, **class_kwargs
        )
        interactions[(uniprot_a, uniprot_b)] = (entry, pmids, psimis)

    # Testing should only update the is_holdout to true and leave other
    # boolean fields alone.
    logger.info("Creating holdout interaction entries.")
    generator = generate_interaction_tuples(testing)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        uniprot_a, uniprot_b = sorted([uniprot_a, uniprot_b])
        entry = interactions.get((uniprot_a, uniprot_b), None)
        if entry is None:
            source = protein_map[uniprot_a]
            target = protein_map[uniprot_b]
            class_kwargs = feature_map[(uniprot_a, uniprot_b)]
            class_kwargs["is_holdout"] = True
            entry = create_interaction(
                source, target, label, session=db_session, save=False,
                commit=False, verbose=verbose, **class_kwargs
            )
            interactions[(uniprot_a, uniprot_b)] = (entry, pmids, psimis)
        else:
            entry[0].is_holdout = True
            entry[0].add_label(label)
            pmids = entry[1] + pmids
            psimis = entry[2] + psimis
            interactions[(uniprot_a, uniprot_b)] = (entry[0], pmids, psimis)

    # Common are in both kegg and hprd so should only update the is_training
    # and is_holdout to true and leave other boolean fields alone.
    logger.info("Creating training/holdout interaction entries.")
    generator = generate_interaction_tuples(common)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        uniprot_a, uniprot_b = sorted([uniprot_a, uniprot_b])
        entry = interactions.get((uniprot_a, uniprot_b), None)
        if entry is None:
            source = protein_map[uniprot_a]
            target = protein_map[uniprot_b]
            class_kwargs = feature_map[(uniprot_a, uniprot_b)]
            class_kwargs["is_holdout"] = True
            class_kwargs["is_training"] = True
            entry = create_interaction(
                source, target, label, session=db_session, save=False,
                commit=False, verbose=verbose, **class_kwargs
            )
            interactions[(uniprot_a, uniprot_b)] = (entry, pmids, psimis)
        else:
            entry[0].is_training = True
            entry[0].is_holdout = True
            entry[0].add_label(label)
            pmids = entry[1] + pmids
            psimis = entry[2] + psimis
            interactions[(uniprot_a, uniprot_b)] = (entry[0], pmids, psimis)

    # Training should only update the is_interactome to true and leave other
    # boolean fields alone.
    logger.info("Creating interactome interaction entries.")
    generator = generate_interaction_tuples(interactome)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        uniprot_a, uniprot_b = sorted([uniprot_a, uniprot_b])
        entry = interactions.get((uniprot_a, uniprot_b), None)
        if entry is None:
            source = protein_map[uniprot_a]
            target = protein_map[uniprot_b]
            class_kwargs = feature_map[(uniprot_a, uniprot_b)]
            class_kwargs["is_interactome"] = True
            entry = create_interaction(
                source, target, label, session=db_session, save=False,
                commit=False, verbose=verbose, **class_kwargs
            )
            interactions[(uniprot_a, uniprot_b)] = (entry, pmids, psimis)
        else:
            entry[0].is_interactome = True
            pmids = entry[1] + pmids
            psimis = entry[2] + psimis
            interactions[(uniprot_a, uniprot_b)] = (entry[0], pmids, psimis)

    # Batch commit might be quicker than calling save on each interaction.
    logger.info("Commiting interactions to database.")
    try:
        entries = [tup[0] for tup in interactions.values()]
        db_session.add_all(entries)
        db_session.commit()
    except:
        db_session.rollback()
        raise

    logger.info("Linking Pubmed/Psimi references.")
    pubmed_map = {p.accession: p for p in Pubmed.query.all()}
    psimi_map = {p.accession: p for p in Psimi.query.all()}
    references = []
    for entry, pmid_ls, psimi_ls in interactions.values():
        for pmid, psimis in zip(pmid_ls, psimi_ls):
            if pmid is None:
                continue
            if psimis is None:
                ref = Reference(entry, pubmed_map[pmid], None)
                references.append(ref)
            else:
                for psimi in psimis:
                    if psimi is None:
                        ref = Reference(entry, pubmed_map[pmid], None)
                        continue
                    ref = Reference(entry, pubmed_map[pmid], psimi_map[psimi])
                    references.append(ref)

    try:
        db_session.add_all(references)
        db_session.commit()
        cleanup_module()
    except:
        db_session.rollback()
        raise
