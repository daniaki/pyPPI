"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  build_data.py [--clear_cache] [--init_database] [--verbose] [--n_jobs=J]
  build_data.py -h | --help

Options:
  -h --help     Show this screen.
  --verbose     Print intermediate output for debugging.
  --n_jobs=J            Number of processes to run in parallel [default: 1]
  --clear_cache      Delete previous bioservices KEGG/UniProt cache
  --init_database    Save uniprot SwissProt/Trembl dumps into local database.
"""

import os
import pandas as pd
import logging
from Bio import SwissProt
from joblib import Parallel, delayed
from docopt import docopt

from pyppi.base import delete_cache, delete_database
from pyppi.base import parse_args, SOURCE, TARGET, LABEL

from pyppi.data import bioplex_network_path, pina2_network_path
from pyppi.data import bioplex_v4, pina2, innate_curated, innate_imported
from pyppi.data import innate_i_network_path, innate_c_network_path
from pyppi.data import interactome_network_path, full_training_network_path
from pyppi.data import kegg_network_path, hprd_network_path
from pyppi.data import save_uniprot_accession_map
from pyppi.data import testing_network_path, training_network_path
from pyppi.data import save_network_to_path
from pyppi.data import save_ptm_labels
from pyppi.data import default_db_path
from pyppi.data import uniprot_sprot, uniprot_trembl

from pyppi.database import begin_transaction
from pyppi.database.models import Protein, Interaction
from pyppi.database.managers import InteractionManager, ProteinManager

from pyppi.data_mining.uniprot import parse_record_into_protein
from pyppi.data_mining.uniprot import batch_map
from pyppi.data_mining.generic import bioplex_func, mitab_func, pina_func
from pyppi.data_mining.generic import generic_to_dataframe
from pyppi.data_mining.hprd import hprd_to_dataframe
from pyppi.data_mining.tools import process_interactions, make_interaction_frame
from pyppi.data_mining.tools import remove_intersection, remove_labels
from pyppi.data_mining.tools import map_network_accessions
from pyppi.data_mining.kegg import download_pathway_ids, pathways_to_dataframe
from pyppi.data_mining.ontology import get_active_instance
from pyppi.data_mining.features import compute_interaction_features


if __name__ == "__main__":
    logger = logging.getLogger("scripts")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # ---------------------------------------------------------------------- #
    #                     MODIFY THESE TO SUIT YOUR NEEDS
    # ---------------------------------------------------------------------- #
    args = docopt(__doc__)
    args = parse_args(args)
    n_jobs = args['n_jobs']
    verbose = args['verbose']
    clear_cache = args['clear_cache']
    init_database = args['init_database']

    i_manager = InteractionManager(verbose=verbose, match_taxon_id=9606)
    p_manager = ProteinManager(verbose=verbose, match_taxon_id=9606)


    # Setup the protein table in the database
    # -------------------------------------------------------------------------- #
    if clear_cache:
        logger.info("Clearing Biopython/Bioservices cache.")
        delete_cache()

    if init_database:
        logger.info("Clearing database.")
        delete_database()
        records = list(SwissProt.parse(uniprot_sprot())) + list(SwissProt.parse(uniprot_trembl()))
        proteins = [parse_record_into_protein(r) for r in records]
        with begin_transaction() as session:
            logger.info("Saving proteins to database.")
            for protein in proteins:
                protein.save(session, commit=False)
            try:
                session.commit()
            except:
                session.rollback()
                raise

    # Construct all the networks
    # -------------------------------------------------------------------------- #
    logger.info("Building KEGG interactions.")
    with begin_transaction() as session:
        pathways = download_pathway_ids('hsa')
        kegg = pathways_to_dataframe(
            session=session,
            pathway_ids=pathways,
            map_to_uniprot=True,
            drop_nan=True,
            allow_self_edges=True,
            allow_duplicates=False
        )

    logger.info("Building HPRD interactions.")
    hprd = hprd_to_dataframe(
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    logger.info("Building Interactome interactions.")
    bioplex = generic_to_dataframe(
        f_input=bioplex_v4(),
        parsing_func=bioplex_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    pina2 = generic_to_dataframe(
        f_input=pina2(),
        parsing_func=pina_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
)

    innate_c = generic_to_dataframe(
        f_input=innate_curated(),
        parsing_func=mitab_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    innate_i = generic_to_dataframe(
        f_input=innate_imported(),
        parsing_func=mitab_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    logger.info("Mapping to most recent uniprot accessions.")
    # Get a set of all the unique uniprot accessions
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]
    sources = set(p for df in networks for p in df.source.values)
    targets = set(p for df in networks for p in df.target.values)
    accessions = list(sources | targets)
    with begin_transaction() as session:
        accession_mapping = batch_map(
            session=session,
            allow_download=False,
            accessions=accessions,
            keep_unreviewed=True,
            match_taxon_id=9606
        )
    save_uniprot_accession_map(accession_mapping)

    logger.info("Mapping each network to the most recent uniprot accessions.")
    kegg = map_network_accessions(
        interactions=kegg, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    hprd = map_network_accessions(
        interactions=hprd, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    pina2 = map_network_accessions(
        interactions=pina2, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    bioplex = map_network_accessions(
        interactions=bioplex, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    innate_c = map_network_accessions(
        interactions=innate_c, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    innate_i = map_network_accessions(
        interactions=innate_i, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    networks = [hprd, kegg, bioplex, pina2, innate_i, innate_c]

    logger.info("Saving raw networks.")
    save_network_to_path(kegg, kegg_network_path)
    save_network_to_path(hprd, hprd_network_path)
    save_network_to_path(pina2, pina2_network_path)
    save_network_to_path(bioplex, bioplex_network_path)
    save_network_to_path(innate_i, innate_i_network_path)
    save_network_to_path(innate_c, innate_c_network_path)

    logger.info("Building and saving processed networks.")
    hprd_test_labels = ['dephosphorylation', 'phosphorylation']
    hprd_train_labels = set([l for l in hprd[LABEL] if l not in hprd_test_labels])
    train_hprd = remove_labels(hprd, hprd_test_labels)
    training = pd.concat([kegg, train_hprd], ignore_index=True)
    testing = remove_intersection(remove_labels(hprd, hprd_train_labels), kegg)

    # Some ppis will be the same between training/testing sets but
    # with different labels. Put all the ppis appearing in testing
    # with a different label compared to the same instance in training
    # into the training set. This way we can keep the testing and
    # training sets completely disjoint.
    labels = []
    sources= []
    targets = []
    for (a, b, l) in zip(testing[SOURCE], testing[TARGET], testing[LABEL]):
        if (a, b) in zip(training[SOURCE], training[TARGET]):
            sources.append(a)
            targets.append(b)
            labels.append(l)
    common_ppis = make_interaction_frame(sources, targets, labels)
    training = pd.concat([training, common_ppis], ignore_index=True)
    testing = remove_intersection(testing, training)
    full_training = pd.concat([training, testing], ignore_index=True)

    testing = process_interactions(
        interactions=testing, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    training = process_interactions(
        interactions=training,
        drop_nan=True, allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    full_training = process_interactions(
        interactions=full_training,
        drop_nan=True, allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )

    labels = list(training[LABEL]) + list(testing[LABEL])
    ptm_labels = set(l for merged in labels for l in merged.split(','))
    save_ptm_labels(ptm_labels)

    interactome_networks = [bioplex, pina2, innate_i, innate_c]
    interactome = pd.concat(interactome_networks, ignore_index=True)
    interactome = process_interactions(
        interactions=interactome, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=None, merge=True
    )
    save_network_to_path(interactome, interactome_network_path)
    save_network_to_path(training, training_network_path)
    save_network_to_path(testing, testing_network_path)
    save_network_to_path(full_training, full_training_network_path)

    logger.info("Saving Interaction records to database.")
    with begin_transaction() as session:
        protein_map = p_manager.uniprotid_entry_map(session)
        ppis = [
            (protein_map[a], protein_map[b])
            for network in [training, testing, interactome]
            for (a, b) in zip(network[SOURCE], network[TARGET])
        ]

    feature_map = {}
    logger.info("Computing features.")
    features = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=verbose)(
        delayed(compute_interaction_features)(source, target)
        for (source, target) in ppis
    )
    for (source, target), features in zip(ppis, features):    
        feature_map[(source.uniprot_id, target.uniprot_id)] = features

    # Training
    interactions = {}
    for (uniprot_a, uniprot_b, label) in zip(training[SOURCE], training[TARGET], training[LABEL]):
        entry = interactions.get((uniprot_a, uniprot_b), None)
        if entry is None:
            interaction = Interaction(
                source=protein_map[uniprot_a].id,
                target=protein_map[uniprot_b].id,
                is_training=True,
                is_holdout=False,
                is_interactome=False,
                label=label,
                **feature_map[(uniprot_a, uniprot_b)]
            )
            interactions[(uniprot_a, uniprot_b)] = interaction
        else:
            # If this raises, then the training/testing are not disjoint as expected.
            raise ValueError("Interaction already exists.")

    # Testing/Holdout
    for (uniprot_a, uniprot_b, label) in zip(testing[SOURCE], testing[TARGET], testing.label):
        entry = interactions.get((uniprot_a, uniprot_b), None)
        if entry is None:
            interaction = Interaction(
                source=protein_map[uniprot_a].id,
                target=protein_map[uniprot_b].id,
                is_training=False,
                is_holdout=True,
                is_interactome=False,
                label=label,
                **feature_map[(uniprot_a, uniprot_b)]
            )
            interactions[(uniprot_a, uniprot_b)] = interaction
        else:
            # If this raises, then the training/testing are not disjoint as expected.
            raise ValueError("Interaction already exists.")

    # Interactome
    for (uniprot_a, uniprot_b) in zip(interactome[SOURCE], interactome[TARGET]):
        entry = interactions.get((uniprot_a, uniprot_b), None)
        if entry is None:
            interaction = Interaction(
                source=protein_map[uniprot_a].id,
                target=protein_map[uniprot_b].id,
                is_training=False,
                is_holdout=False,
                is_interactome=True,
                label=None,
                **feature_map[(uniprot_a, uniprot_b)]
            )
            interactions[(uniprot_a, uniprot_b)] = interaction
        else:
            entry.is_interactome = True
            interactions[(uniprot_a, uniprot_b)] = entry

    with begin_transaction() as session:
        logger.info("Writing to database.")
        for interaction in interactions.values():
            interaction.save(session, commit=False)
        try:
            session.commit()
        except:
            session.rollback()
            raise
