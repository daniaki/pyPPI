"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  build_data.py [--clear_cache] [--n_jobs=J]
  build_data.py -h | --help

Options:
  -h --help  Show this screen.
  --n_jobs=J  Number of processes to run in parallel [default: 1]
  --clear_cache  Delete previous bioservices KEGG/UniProt cache
"""

import os
import pandas as pd
import logging
from Bio import SwissProt
from joblib import Parallel, delayed
from docopt import docopt

from pyppi.base import delete_cache
from pyppi.base import parse_args, SOURCE, TARGET, LABEL
from pyppi.base import PUBMED, EXPERIMENT_TYPE, NULL_VALUES

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

from pyppi.database import make_session, begin_transaction, delete_database
from pyppi.database.models import Protein, Interaction, Pubmed, Psimi
from pyppi.database.utilities import generate_interaction_tuples
from pyppi.database.utilities import update_interaction
from pyppi.database.utilities import psimi_string_to_list, pmid_string_to_list
from pyppi.database.managers import InteractionManager, ProteinManager

from pyppi.data_mining.uniprot import parse_record_into_protein
from pyppi.data_mining.uniprot import batch_map
from pyppi.data_mining.generic import bioplex_func, mitab_func, pina_func
from pyppi.data_mining.generic import generic_to_dataframe
from pyppi.data_mining.hprd import hprd_to_dataframe
from pyppi.data_mining.tools import process_interactions, make_interaction_frame
from pyppi.data_mining.tools import remove_common_ppis, remove_labels
from pyppi.data_mining.tools import map_network_accessions
from pyppi.data_mining.kegg import download_pathway_ids, pathways_to_dataframe
from pyppi.data_mining.ontology import get_active_instance
from pyppi.data_mining.psimi import get_active_instance as load_mi_ontology
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

    args = docopt(__doc__)
    args = parse_args(args)
    n_jobs = args['n_jobs']
    clear_cache = args['clear_cache']

    i_manager = InteractionManager(verbose=True, match_taxon_id=9606)
    p_manager = ProteinManager(verbose=True, match_taxon_id=9606)

    # Setup the protein table in the database
    # ----------------------------------------------------------------------- #
    if clear_cache:
        logger.info("Clearing Biopython/Bioservices cache.")
        delete_cache()

    logger.info("Clearing existing database tables.")
    with begin_transaction(db_path=default_db_path) as session:
        delete_database(session=session)

    logger.info("Parsing UniProt and PSI-MI into database.")
    records = list(SwissProt.parse(uniprot_sprot())) + \
        list(SwissProt.parse(uniprot_trembl()))
    with begin_transaction(db_path=default_db_path) as session:
        proteins = [parse_record_into_protein(r) for r in records]
        psimi_objects = []
        mi_ont = load_mi_ontology()
        for key, term in mi_ont.items():
            obj = Psimi(accession=key, description=term.name)
            psimi_objects.append(obj)

        try:
            session.add_all(proteins + psimi_objects)
            session.commit()
        except:
            session.rollback()
            raise

    logger.info("Starting new database session.")
    session = make_session(db_path=default_db_path)

    # Construct all the networks
    # ----------------------------------------------------------------------- #
    logger.info("Building KEGG interactions.")
    pathways = download_pathway_ids('hsa')
    kegg = pathways_to_dataframe(
        session=session,
        pathway_ids=pathways,
        map_to_uniprot=True,
        drop_nan='default',
        allow_self_edges=True,
        allow_duplicates=False
    )

    logger.info("Building HPRD interactions.")
    hprd = hprd_to_dataframe(
        session=session,
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

    pina2 = generic_to_dataframe(
        f_input=pina2(),
        parsing_func=pina_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    innate_c = generic_to_dataframe(
        f_input=innate_curated(),
        parsing_func=mitab_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    innate_i = generic_to_dataframe(
        f_input=innate_imported(),
        parsing_func=mitab_func,
        drop_nan=[SOURCE, TARGET],
        allow_self_edges=True,
        allow_duplicates=False
    )

    logger.info("Mapping to most recent uniprot accessions.")
    # Get a set of all the unique uniprot accessions
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]
    sources = set(p for df in networks for p in df.source.values)
    targets = set(p for df in networks for p in df.target.values)
    accessions = list(sources | targets)
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
        drop_nan='default', allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    hprd = map_network_accessions(
        interactions=hprd, accession_map=accession_mapping,
        drop_nan='default', allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    pina2 = map_network_accessions(
        interactions=pina2, accession_map=accession_mapping,
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

    labels = list(full_training[LABEL])
    ptm_labels = set(l for merged in labels for l in merged.split(','))
    save_ptm_labels(ptm_labels)

    interactome_networks = [bioplex, pina2, innate_i, innate_c]
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
    protein_map = p_manager.uniprotid_entry_map(session)
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
    # exist in the database.
    logger.info("Updating Pubmed/PSI-MI database entries.")
    objects = []
    mi_ont = load_mi_ontology()
    networks = [full_training, interactome]
    pmids = set([
        p for ls in pd.concat(networks, ignore_index=True)[PUBMED]
        for p in str(ls).split(',')
        if str(p) not in NULL_VALUES
    ])
    psimis = set([
        p for ls in pd.concat(networks, ignore_index=True)[EXPERIMENT_TYPE]
        for p in str(ls).split(',')
        if str(p) not in NULL_VALUES
    ])
    for pmid in pmids:
        if not session.query(Pubmed).filter_by(accession=pmid).count():
            objects.append(Pubmed(accession=pmid))
    for psimi in psimis:
        if not session.query(Psimi).filter_by(accession=psimi).count():
            objects.append(
                Psimi(accession=psimi, description=mi_ont[psimi].name)
            )
    try:
        session.add_all(objects)
        session.commit()
    except:
        session.rollback()
        raise

    logger.info("Creating Interaction database entries.")
    interactions = {}
    for interaction in session.query(Interaction).all():
        a = p_manager.get_by_id(session, id=interaction.source)
        b = p_manager.get_by_id(session, id=interaction.target)
        interactions[(a.uniprot_id, b.uniprot_id)] = interaction

    # Training should only update the is_training to true and leave other
    # boolean fields alone.
    logger.info("Creating training interaction entries.")
    generator = generate_interaction_tuples(training)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        class_kwargs = feature_map[(uniprot_a, uniprot_b)]
        class_kwargs["source"] = protein_map[uniprot_a]
        class_kwargs["target"] = protein_map[uniprot_b]
        class_kwargs["label"] = label
        class_kwargs["is_training"] = True
        class_kwargs["is_holdout"] = False
        class_kwargs["is_interactome"] = False
        entry = update_interaction(
            session=session,
            commit=False,
            psimi_ls=psimi_string_to_list(session, psimi),
            pmid_ls=pmid_string_to_list(session, pmids),
            replace_fields=False,
            override_boolean=False,
            create_if_not_found=True,
            match_taxon_id=9606,
            verbose=False,
            update_features=False,
            **class_kwargs
        )
        interactions[(uniprot_a, uniprot_b,)] = entry

    # Testing should only update the is_holdout to true and leave other
    # boolean fields alone.
    logger.info("Creating holdout interaction entries.")
    generator = generate_interaction_tuples(testing)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        class_kwargs = feature_map[(uniprot_a, uniprot_b)]
        class_kwargs["source"] = protein_map[uniprot_a]
        class_kwargs["target"] = protein_map[uniprot_b]
        class_kwargs["label"] = label
        class_kwargs["is_training"] = False
        class_kwargs["is_holdout"] = True
        class_kwargs["is_interactome"] = False
        entry = update_interaction(
            session=session,
            commit=False,
            psimi_ls=psimi_string_to_list(session, psimi),
            pmid_ls=pmid_string_to_list(session, pmids),
            replace_fields=False,
            override_boolean=False,
            create_if_not_found=True,
            match_taxon_id=9606,
            verbose=False,
            update_features=False,
            **class_kwargs
        )
        interactions[(uniprot_a, uniprot_b,)] = entry

    # Common are in both kegg and hprd so should only update the is_training
    # and is_holdout to true and leave other boolean fields alone.
    logger.info("Creating training/holdout interaction entries.")
    generator = generate_interaction_tuples(common)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        class_kwargs = feature_map[(uniprot_a, uniprot_b)]
        class_kwargs["source"] = protein_map[uniprot_a]
        class_kwargs["target"] = protein_map[uniprot_b]
        class_kwargs["label"] = label
        class_kwargs["is_training"] = True
        class_kwargs["is_holdout"] = True
        class_kwargs["is_interactome"] = False
        entry = update_interaction(
            session=session,
            commit=False,
            psimi_ls=psimi_string_to_list(session, psimi),
            pmid_ls=pmid_string_to_list(session, pmids),
            replace_fields=False,
            override_boolean=False,
            create_if_not_found=True,
            match_taxon_id=9606,
            verbose=False,
            update_features=False,
            **class_kwargs
        )
        interactions[(uniprot_a, uniprot_b,)] = entry

    # Training should only update the is_interactome to true and leave other
    # boolean fields alone.
    logger.info("Creating interactome interaction entries.")
    generator = generate_interaction_tuples(interactome)
    for (uniprot_a, uniprot_b, label, pmids, psimis) in generator:
        class_kwargs = feature_map[(uniprot_a, uniprot_b)]
        class_kwargs["source"] = protein_map[uniprot_a]
        class_kwargs["target"] = protein_map[uniprot_b]
        class_kwargs["label"] = label
        class_kwargs["is_training"] = False
        class_kwargs["is_holdout"] = False
        class_kwargs["is_interactome"] = True
        entry = update_interaction(
            session=session,
            commit=False,
            psimi_ls=psimi_string_to_list(session, psimi),
            pmid_ls=pmid_string_to_list(session, pmids),
            replace_fields=False,
            override_boolean=False,
            create_if_not_found=True,
            match_taxon_id=9606,
            verbose=False,
            update_features=False,
            **class_kwargs
        )
        interactions[(uniprot_a, uniprot_b,)] = entry

    # Batch commit might be quicker than calling save on each interaction.
    logger.info("Writing to database.")
    try:
        session.commit()
        session.close()
    except:
        session.rollback()
        raise
