#!/usr/bin/env python -W ignore::UndefinedMetricWarning

"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  predict_ppis.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
                  [--retrain] [--chain] [--induce] [--verbose]
                  [--model=M] [--n_jobs=J] [--n_splits=S] [--n_iterations=I]
                  [--input=FILE] [--output=FILE] [--directory=DIR]
  predict_ppis.py -h | --help

Options:
  -h --help     Show this screen.
  --interpro    Use interpro domains in features.
  --pfam        Use Pfam domains in features.
  --mf          Use Molecular Function Gene Ontology in features.
  --cc          Use Cellular Compartment Gene Ontology in features.
  --bp          Use Biological Process Gene Ontology in features.
  --induce      Use ULCA inducer over Gene Ontology.
  --verbose     Print intermediate output for debugging.
  --chain       Use Classifier chains to learn label dependencies.
  --retrain     Re-train classifier instead of loading previous version. If
                using a previous version, you must use the same selection of
                features along with the same induce setting.
  --model=M         A binary classifier from Scikit-Learn implementing fit,
                    predict and predict_proba [default: LogisticRegression].
                    Ignored if using 'retrain'.
  --n_jobs=J        Number of processes to run in parallel [default: 1]
  --n_splits=S      Number of cross-validation splits used during randomized
                    grid search [default: 5]
  --n_iterations=I  Number of randomized grid search iterations [default: 60]
  --input=FILE      Uniprot edge-list, with a path directory that absolute or
                    relative to this script. Entries must be tab separated with
                    header columns 'source' and 'target'. [default: None]
  --output=FILE     Output file name [default: predictions.tsv]
  --directory=DIR   Absolute or relative output directory [default: ./results/]
"""
import sys
import os
import json
import csv
import logging
import numpy as np
import pandas as pd
import joblib
from collections import Counter, OrderedDict
from numpy.random import RandomState
from joblib import Parallel, delayed
from datetime import datetime
from docopt import docopt
import warnings

from pyppi.base.utilities import su_make_dir, chunk_list, is_null
from pyppi.base.arg_parsing import parse_args
from pyppi.base.constants import (
    P1, P2, G1, G2, SOURCE, TARGET, PUBMED, EXPERIMENT_TYPE
)
from pyppi.base.log import create_logger
from pyppi.base.io import generic_io, save_classifier, load_classifier
from pyppi.base.file_paths import classifier_path, default_db_path

from pyppi.models.utilities import (
    make_classifier, get_parameter_distribution_for_model,
    publication_ensemble
)

from pyppi.database import db_session
from pyppi.database.models import Interaction, Protein
from pyppi.database.models import Pubmed, Psimi, Reference
from pyppi.database.utilities import create_interaction, uniprotid_entry_map
from pyppi.database.utilities import full_training_network
from pyppi.database.utilities import interactome_interactions
from pyppi.database.utilities import labels_from_interactions

from pyppi.data_mining.tools import xy_from_interaction_frame
from pyppi.data_mining.generic import edgelist_func, generic_to_dataframe
from pyppi.data_mining.tools import map_network_accessions
from pyppi.data_mining.uniprot import batch_map
from pyppi.data_mining.features import compute_interaction_features

from pyppi.predict.utilities import interactions_to_Xy_format
from pyppi.predict import parse_interactions
from pyppi.predict.plotting import plot_threshold_curve
from pyppi.models.utilities import make_gridsearch_clf
from pyppi.predict.utilities import paper_model, validation_model
from pyppi.models.binary_relevance import MixedBinaryRelevanceClassifier

from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_STATE = 42
logger = create_logger("scripts", logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


if __name__ == "__main__":
    args = parse_args(docopt(__doc__))
    n_jobs = args['n_jobs']
    n_splits = args['n_splits']
    rcv_iter = args['n_iterations']
    induce = args['induce']
    verbose = args['verbose']
    selection = args['selection']
    model = args['model']
    out_file = args['output']
    input_file = args['input']
    direc = args['directory']
    retrain = args['retrain']
    chain = args['chain']

    # Set up the folder for each experiment run named after the current time
    # -------------------------------------------------------------------- #
    folder = datetime.now().strftime("pred_%y-%m-%d_%H-%M-%S")
    direc = "{}/{}/".format(direc, folder)
    su_make_dir(direc)
    json.dump(
        args, fp=open("{}/settings.json".format(direc), 'w'),
        indent=4, sort_keys=True
    )

    logger.info("Starting new database session.")
    protein_map = uniprotid_entry_map()
    testing = []
    accession_mapping = {}

    # Get the input edge-list ready
    # -------------------------------------------------------------------- #
    training = full_training_network(taxon_id=9606)
    labels = labels_from_interactions(training)

    if input_file == None:
        logger.info("Loading interactome data.")
        testing = interactome_interactions(taxon_id=9606)
    else:
        logger.info("Loading custom ppi data.")
        testing = generic_to_dataframe(
            f_input=generic_io(input_file),
            parsing_func=edgelist_func,
            drop_nan=True,
            allow_self_edges=True,
            allow_duplicates=True
        )

        sources = set(p for p in testing.source.values)
        targets = set(p for p in testing.target.values)
        accessions = list(sources | targets)
        accession_mapping = batch_map(
            session=db_session,
            accessions=accessions,
            keep_unreviewed=True,
            match_taxon_id=9606,
            allow_download=True
        )
        testing_network = map_network_accessions(
            interactions=testing, accession_map=accession_mapping,
            drop_nan=True, allow_self_edges=True,
            allow_duplicates=False, min_counts=None, merge=False
        )

        testing, invalid, new_upids = parse_interactions(
            testing_network, session=db_session, taxon_id=9606,
            verbose=verbose, n_jobs=n_jobs
        )
        # Writing some additional data returned during the parsing process.
        # Namely any invalid interactions and if old uniprot identifiers
        # have been supplied, a mapping to the recent UniProt record used
        # to calculate features from and to build an interaction instance with.
        with open('{}/accession_map.json'.format(direc), 'wt') as fp:
            for k, v in new_upids.items():
                existing = accession_mapping.get(k, [v])
                new = set(existing) | set([v])
                accession_mapping[k] = list(new)
            json.dump(accession_mapping, fp)
        if invalid:
            if verbose:
                logger.info("Encountered invalid PPIs: {}".format(
                    ', '.join(invalid)
                ))
            with open('{}/invalid.tsv'.format(direc), 'wt') as fp:
                fp.write("{}\t{}".format(SOURCE, TARGET))
                for (a, b) in invalid:
                    fp.write("{}\t{}".format(a, b))

        # Save and close session
        logger.info("Commiting changes to database.")
        try:
            db_session.commit()
            db_session.close()
        except:
            db_session.rollback()
            raise

    # Get the features into X, and multilabel y indicator format
    # -------------------------------------------------------------------- #
    logger.info("Preparing training and input interactions.")
    X_train, y_train = interactions_to_Xy_format(training, selection)
    X_test, _ = interactions_to_Xy_format(testing, selection)

    logger.info("Computing usable feature proportions in testing samples.")

    def separate_features(row):
        features = row[0].upper().split(',')
        interpro = set(term for term in features if 'IPR' in term)
        go = set(term for term in features if 'GO:' in term)
        pfam = set(term for term in features if 'PF' in term)
        return (go, interpro, pfam)

    def compute_proportions_shared(row):
        go, ipr, pf = row
        try:
            go_prop = len(go & go_training) / len(go)
        except ZeroDivisionError:
            go_prop = 0
        try:
            ipr_prop = len(ipr & ipr_training) / len(ipr)
        except ZeroDivisionError:
            ipr_prop = 0
        try:
            pf_prop = len(pf & pf_training) / len(pf)
        except ZeroDivisionError:
            pf_prop = 0
        return go_prop, ipr_prop, pf_prop

    X_train_split_features = np.apply_along_axis(
        separate_features, axis=1, arr=X_train.reshape((X_train.shape[0], 1))
    )
    go_training = set()
    ipr_training = set()
    pf_training = set()
    for (go, ipr, pf) in X_train_split_features:
        go_training |= go
        ipr_training |= ipr
        pf_training |= pf

    X_test_split_features = np.apply_along_axis(
        separate_features, axis=1, arr=X_test.reshape((X_test.shape[0], 1))
    )
    X_test_useable_props = np.apply_along_axis(
        compute_proportions_shared, axis=1, arr=X_test_split_features
    )

    # Make the estimators and BR classifier
    # -------------------------------------------------------------------- #
    if retrain or not os.path.isfile(classifier_path):
        mlb = MultiLabelBinarizer(classes=sorted(labels))
        mlb.fit(y_train)
        y_train = mlb.transform(y_train)

        if model == 'paper':
            clf = paper_model(
                labels=mlb.classes,
                rcv_splits=n_splits,
                rcv_iter=rcv_iter,
                scoring="f1",
                n_jobs_model=n_jobs,
                n_jobs_gs=n_jobs,
                n_jobs_br=1,
                random_state=RANDOM_STATE
            )
        else:
            pipeline = make_gridsearch_clf(
                model=model,
                rcv_splits=n_splits,
                rcv_iter=rcv_iter,
                scoring="f1",
                n_jobs_model=n_jobs,
                n_jobs_gs=n_jobs,
                random_state=RANDOM_STATE,
                search_vectorizer=True
            )
            estimators = [clone(pipeline) for _ in mlb.classes]
            clf = MixedBinaryRelevanceClassifier(
                estimators, n_jobs=1, verbose=verbose
            )

        # Saved to both the home directory and the output directory.
        logger.info("Fitting data.")
        clf.fit(X_train, y_train)
        save_classifier(clf, selection, mlb, classifier_path)
        save_classifier(clf, selection, mlb, "{}/classifier.pkl".format(direc))

    # Loads a previously (or recently trained) classifier from disk
    # and then performs the predictions on the new dataset.
    # -------------------------------------------------------------------- #
    logger.info("Making predictions.")
    clf, selection, mlb = load_classifier(classifier_path)
    predictions = clf.predict_proba(X_test)

    # Write the predictions to a tsv file
    # -------------------------------------------------------------------- #
    logger.info("Writing results to file.")
    usable_go_term_props = [go for (go, _, _) in X_test_useable_props]
    usable_ipr_term_props = [ipr for (_, ipr, _) in X_test_useable_props]
    usable_pf_term_props = [pf for (_, _, pf) in X_test_useable_props]
    predicted_labels = [
        ','.join(np.asarray(labels)[selector]) or None for selector in
        [np.where(row >= 0.5) for row in predictions]
    ]

    predicted_label_at_max = [
        labels[idx] for idx in
        [np.argmax(row) for row in predictions]
    ]
    entryid_uniprotid_map = {
        entry.id: uniprot_id for (uniprot_id, entry) in protein_map.items()
    }

    # Make a map so we can map back to the original input UniProt ids.
    reverse_acc_map = {}
    for k, vs in accession_mapping.items():
        for v in vs:
            reverse_acc_map[v] = k

    # This block of code takes the references for each entry and turns
    # then into pubmed and psimi accessions
    pmids = []
    psimis = []
    pubmed_map = {p.id: p for p in Pubmed.query.all()}
    psimi_map = {p.id: p for p in Psimi.query.all()}
    for entry in testing:
        refs = entry.references()
        pmid_psimis = OrderedDict()
        for ref in refs:
            pmid = pubmed_map[ref.pubmed_id].accession
            if ref.psimi_id is not None:
                psimi = psimi_map[ref.psimi_id].accession
            else:
                psimi = None
            if pmid not in pmid_psimis:
                pmid_psimis[pmid] = set()
            if psimi is not None:
                pmid_psimis[pmid].add(psimi)

        # Join all associated psimi annotations with a pmid with '|'
        # to indicate a grouped collection.
        for pmid, group in pmid_psimis.items():
            pmid_psimis[pmid] = '|'.join(group) or str(None)
        # Join all pmids and their assoicated groups with a comma.
        joined_pmids = ','.join(pmid_psimis.keys()) or None
        joined_psimis = ','.join(pmid_psimis.values()) or None
        pmids.append(joined_pmids)
        psimis.append(joined_psimis)

    p1 = [entryid_uniprotid_map[entry.source] for entry in testing]
    p2 = [entryid_uniprotid_map[entry.target] for entry in testing]
    data_dict = {
        P1: p1,
        P2: p2,
        PUBMED: pmids,
        EXPERIMENT_TYPE: psimis,
        'input_%s' % SOURCE: [reverse_acc_map.get(upid, upid) for upid in p1],
        'input_%s' % TARGET: [reverse_acc_map.get(upid, upid) for upid in p2],
        "sum-pr": np.sum(predictions, axis=1),
        "max-pr": np.max(predictions, axis=1),
        "classification": predicted_labels,
        "classification_at_max": predicted_label_at_max,
        "proportion_go_used": usable_go_term_props,
        "proportion_interpro_used": usable_ipr_term_props,
        "proportion_pfam_used": usable_pf_term_props
    }

    for idx, label in enumerate(mlb.classes):
        data_dict['{}-pr'.format(label)] = predictions[:, idx]

    columns = [P1, P2, G1, G2, 'input_%s' % SOURCE, 'input_%s' % TARGET] + \
        ['{}-pr'.format(l) for l in mlb.classes] + \
        ['sum-pr', 'max-pr', 'classification', 'classification_at_max'] + \
        ['proportion_go_used', 'proportion_interpro_used'] + \
        ["proportion_pfam_used", PUBMED, EXPERIMENT_TYPE]
    df = pd.DataFrame(data=data_dict, columns=columns)

    accession_gene_map = {
        p.uniprot_id: p.gene_id for p in protein_map.values()
    }
    df['{}'.format(G1)] = df.apply(
        func=lambda row: accession_gene_map.get(row[P1], None) or 'None',
        axis=1
    )
    df['{}'.format(G2)] = df.apply(
        func=lambda row: accession_gene_map.get(row[P2], None) or 'None',
        axis=1
    )
    df.to_csv(
        "{}/{}".format(direc, out_file),
        sep='\t', index=False, na_rep=str(None)
    )

    # Calculate the proportion of the interactome classified at a threshold
    # value, t.
    logger.info("Computing threshold curve.")
    thresholds = np.arange(0.0, 1.05, 0.05)
    proportions = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        classified = sum(map(lambda p: np.max(p) >= t, predictions))
        proportion = classified / predictions.shape[0]
        proportions[i] = proportion

    # ------------- Rise and shine, it's plotting time! ------------------- #
    plot_threshold_curve(
        '{}/threshold.jpg'.format(direc),
        thresholds=thresholds,
        proportions=proportions,
        dpi=350
    )

    with open("{}/thresholds.csv".format(direc), 'wt') as fp:
        for (t, p) in zip(thresholds, proportions):
            fp.write("{},{}\n".format(t, p))

    # Compute some basic statistics and numbers and save as a json object
    logger.info("Computing dataset statistics.")
    num_in_training = sum(
        1 for entry in testing if (entry.is_training or entry.is_holdout)
    )
    prop_in_training = num_in_training / X_test.shape[0]
    num_classified = sum(1 for label in predicted_labels if label is not None)
    prop_classified = num_classified / X_test.shape[0]
    data = {
        "number of samples": X_test.shape[0],

        "number of samples seen in training": num_in_training,
        "proportion of samples seen in training": prop_in_training,

        "number of samples classified at 0.5": num_classified,
        "proportion of samples classified at 0.5": prop_classified,

        "number of samples not classified at 0.5": (
            X_test.shape[0] - num_classified),
        "proportion of samples not classified at 0.5": 1.0 - prop_classified,

        "samples with go usability >= 0.5": sum(
            1 for prop in usable_go_term_props if prop >= 0.5
        ) / X_test.shape[0],
        "samples with interpro usability >= 0.5": sum(
            1 for prop in usable_ipr_term_props if prop >= 0.5
        ) / X_test.shape[0],
        "samples with pfam usability >= 0.5": sum(
            1 for prop in usable_pf_term_props if prop >= 0.5
        ) / X_test.shape[0]
    }
    with open("{}/dataset_statistics.json".format(direc), 'wt') as fp:
        json.dump(data, fp, indent=4, sort_keys=True)

    # Count how many labels prediction distribution
    # -------------------------------------------------------------------- #
    label_dist = Counter(
        l for ls in predicted_labels for l in str(ls).split(',')
        if ls is not None
    )
    with open("{}/prediction_distribution.json".format(direc), 'wt') as fp:
        json.dump(label_dist, fp, indent=4, sort_keys=True)
