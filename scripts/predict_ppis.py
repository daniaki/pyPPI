"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  predict_ppis.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
                  [--induce] [--verbose] [--classifier=FILE]
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
  --model=M         A binary classifier from Scikit-Learn implementing fit,
                    predict and predict_proba [default: LogisticRegression].
                    Ignored if using 'retrain'.
  --n_jobs=J        Number of processes to run in parallel [default: 1]
  --n_splits=S      Number of cross-validation splits used during randomized
                    grid search [default: 5]
  --n_iterations=I  Number of randomized grid search iterations [default: 60]
  --classifier=FILE  Path to a pre-trained classifier [default: None]
  --input=FILE      Uniprot edge-list, with a path directory that absolute or
                    relative to this script. Entries must be tab separated with
                    header columns 'source' and 'target'. [default: None]
  --output=FILE     Output file name [default: predictions.tsv]
  --directory=DIR   Absolute or relative output directory [default: ./results/]
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
import warnings
from collections import Counter
from numpy.random import RandomState
from joblib import Parallel, delayed
from datetime import datetime
from docopt import docopt

from pyppi.base import parse_args, su_make_dir, chunk_list
from pyppi.base import P1, P2, G1, G2, SOURCE, TARGET, PUBMED, EXPERIMENT_TYPE
from pyppi.base.logging import create_logger
from pyppi.data import load_network_from_path, load_ptm_labels
from pyppi.data import full_training_network_path, generic_io
from pyppi.data import interactome_network_path, classifier_path
from pyppi.data import default_db_path

from pyppi.models import make_classifier, get_parameter_distribution_for_model

from pyppi.database import make_session
from pyppi.database.models import Interaction
from pyppi.database.managers import InteractionManager, ProteinManager
from pyppi.database.managers import format_interactions_for_sklearn
from pyppi.database.utilities import update_interaction

from pyppi.data_mining.tools import xy_from_interaction_frame
from pyppi.data_mining.generic import edgelist_func, generic_to_dataframe
from pyppi.data_mining.tools import map_network_accessions
from pyppi.data_mining.uniprot import batch_map
from pyppi.data_mining.features import compute_interaction_features

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


MAX_SEED = 1000000
RANDOM_STATE = 42
logger = create_logger("scripts", logging.INFO)


def get_model_for_label(label):
    label_model_map = {
        'Acetylation': 'RandomForestClassifier',
        'Activation': 'RandomForestClassifier',
        'Binding/association': 'RandomForestClassifier',
        'Carboxylation': 'LogisticRegression',
        'Deacetylation': 'RandomForestClassifier',
        'Dephosphorylation': 'RandomForestClassifier',
        'Dissociation': 'RandomForestClassifier',
        'Glycosylation': 'LogisticRegression',
        'Inhibition': 'RandomForestClassifier',
        'Methylation': 'LogisticRegression',
        'Myristoylation': 'LogisticRegression',
        'Phosphorylation': 'RandomForestClassifier',
        'Prenylation': 'LogisticRegression',
        'Proteolytic-cleavage': 'LogisticRegression',
        'State-change': 'LogisticRegression',
        'Sulfation': 'RandomForestClassifier',
        'Sumoylation': 'RandomForestClassifier',
        'Ubiquitination': 'LogisticRegression'
    }
    return 'LogisticRegression'  # label_model_map[label]


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
    saved_classifier = args['classifier']

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
    session = make_session(db_path=default_db_path)
    i_manager = InteractionManager(verbose=verbose, match_taxon_id=9606)
    p_manager = ProteinManager(verbose=verbose, match_taxon_id=9606)
    protein_map = p_manager.uniprotid_entry_map(session)

    # Get the input edge-list ready
    # -------------------------------------------------------------------- #
    labels = i_manager.training_labels(session, include_holdout=True)
    training = i_manager.training_interactions(
        session, keep_holdout=True
    )

    if input_file == None:
        logger.info("Loading interactome data.")
        testing = i_manager.interactome_interactions(
            session=session,
            keep_holdout=True,
            keep_training=True
        )
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
            session=session,
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

        # Compute features for new ppis
        testing = []
        feature_map = {}
        ppis = [
            (protein_map[a], protein_map[b])
            for (a, b) in zip(testing_network[SOURCE], testing_network[TARGET])
            if i_manager.get_by_source_target(session, a, b) is None
        ]

        logger.info("Computing features.")
        features = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(compute_interaction_features)(source, target)
            for (source, target) in ppis
        )
        for (source, target), features in zip(ppis, features):
            feature_map[(source.uniprot_id, target.uniprot_id)] = features

        existing_interactions = {}
        for interaction in session.query(Interaction).all():
            a = p_manager.get_by_id(session, id=interaction.source)
            b = p_manager.get_by_id(session, id=interaction.target)
            existing_interactions[(a.uniprot_id, b.uniprot_id)] = interaction

        for (a, b) in zip(testing_network[SOURCE], testing_network[TARGET]):
            class_kwargs = feature_map[(a, b)]
            class_kwargs["source"] = protein_map[a]
            class_kwargs["target"] = protein_map[b]
            class_kwargs["label"] = None
            class_kwargs["is_training"] = False
            class_kwargs["is_holdout"] = False
            class_kwargs["is_interactome"] = False
            entry = update_interaction(
                session=session,
                commit=False,
                psimi_ls=[],
                pmid_ls=[],
                replace_fields=False,
                override_boolean=False,
                create_if_not_found=True,
                match_taxon_id=9606,
                verbose=False,
                update_features=False,
                existing_interactions=existing_interactions,
                **class_kwargs
            )
            existing_interactions[(a, b)] = entry
            testing.append(entry)
        session.commit()

    # Get the features into X, and multilabel y indicator format
    # -------------------------------------------------------------------- #
    logger.info("Preparing training and testing data.")
    X_train, y_train = format_interactions_for_sklearn(training, selection)
    X_test, _ = format_interactions_for_sklearn(testing, selection)

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

    mlb = MultiLabelBinarizer(classes=sorted(labels))
    mlb.fit(y_train)
    y_train = mlb.transform(y_train)

    # Make the estimators and BR classifier
    # -------------------------------------------------------------------- #
    logger.info("Fitting data.")
    rng = RandomState(seed=RANDOM_STATE)

    if not saved_classifier:
        clfs = []
        model_random_state = rng.randint(MAX_SEED)
        cv_random_state = rng.randint(MAX_SEED)
        rgs_random_state = rng.randint(MAX_SEED)

        for i, label in enumerate(mlb.classes):
            model = get_model_for_label(label)
            params = get_parameter_distribution_for_model(model)
            keys = list(params.keys())
            for key in keys:
                value = params.pop(key)
                params['estimator__{}'.format(key)] = value
            params['vectorizer__binary'] = [False, True]

            vectorizer = CountVectorizer(
                lowercase=False, stop_words=[':', 'GO'])
            base_estimator = make_classifier(
                model, random_state=model_random_state, n_jobs=n_jobs
            )
            pipeline = Pipeline(
                steps=[('vectorizer', vectorizer),
                       ('estimator', base_estimator)]
            )

            logger.info("Training {} for {}.".format(model, label))
            random_cv = RandomizedSearchCV(
                cv=StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=cv_random_state
                ),
                n_iter=rcv_iter,
                n_jobs=n_jobs,
                refit=True,
                random_state=rgs_random_state,
                scoring='f1',
                error_score=0.0,
                param_distributions=params,
                estimator=pipeline
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                random_cv.fit(X_train, y_train[:, i])
            clfs.append(random_cv)
        joblib.dump(clfs, '{}/classifier.pkl'.format(direc))
    else:
        clfs = joblib.load(saved_classifier)

    # Loads a previously (or recently trained) classifier from disk
    # and then performs the predictions on the new dataset.
    # -------------------------------------------------------------------- #
    logger.info("Making predictions.")
    predictions = np.vstack([e.predict_proba(X_test)[:, 1] for e in clfs]).T

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
    data_dict = {
        P1: [entryid_uniprotid_map[entry.source] for entry in testing],
        P2: [entryid_uniprotid_map[entry.target] for entry in testing],
        PUBMED: [
            ','.join(pmid.accession for pmid in entry.pmid) or None
            for entry in testing
        ],
        EXPERIMENT_TYPE: [
            ','.join(psimi.accession for psimi in entry.psimi) or None
            for entry in testing
        ],
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

    columns = [P1, P2, G1, G2] + \
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

        "number of samples not classified at 0.5": X_test.shape[0] - num_classified,
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

    # Save and close session
    logger.info("Commiting changes to database.")
    try:
        session.commit()
        session.close()
    except:
        session.rollback()
        raise
