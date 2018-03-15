#!/usr/bin/env python -W ignore::UndefinedMetricWarning

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.

Usage:
  validation.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
             [--induce] [--chain] [--verbose] [--binary] [--top=T]
             [--model=M] [--n_jobs=J] [--n_splits=S] [--n_iterations=I]
             [--h_iterations=H] [--directory=DIR] [--output_folder=OUT]
  validation.py -h | --help

Options:
  -h --help     Show this screen.
  --interpro    Use interpro domains in features.
  --pfam        Use Pfam domains in features.
  --mf          Use Molecular Function Gene Ontology in features.
  --cc          Use Cellular Compartment Gene Ontology in features.
  --bp          Use Biological Process Gene Ontology in features.
  --binary      Use binary feature encoding instead of ternary.
  --induce      Use ULCA inducer over Gene Ontology.
  --chain       Use Classifier chains to learn label dependencies.
  --verbose     Print intermediate output for debugging.
  --model=M         A binary classifier from Scikit-Learn implementing fit,
                    predict and predict_proba [default: LogisticRegression]
  --n_jobs=J        Number of processes to run in parallel [default: 1]
  --n_splits=S      Number of cross-validation splits [default: 5]
  --h_iterations=H  Number of hyperparameter tuning
                    iterations per fold [default: 30]
  --n_iterations=I  Number of bootstrap iterations [default: 5]
  --directory=DIR   Output directory [default: ./results/]
  --output_folder=OUT  Output directory [default: None]
"""

import json
import logging
import pandas as pd
import numpy as np
import scipy as sp
import warnings
import joblib

from itertools import product
from operator import itemgetter
from collections import Counter
from datetime import datetime
from docopt import docopt
from numpy.random import RandomState

from pyppi.base.constants import MAX_SEED
from pyppi.base.utilities import su_make_dir
from pyppi.base.arg_parsing import parse_args
from pyppi.base.log import create_logger
from pyppi.base.io import ipr_name_map, pfam_name_map, save_classifier

from pyppi.models.utilities import (
    make_classifier, get_parameter_distribution_for_model,
)

from pyppi.model_selection.k_fold import StratifiedKFoldCrossValidation
from pyppi.model_selection.scoring import (
    fdr_score, specificity, positive_label_hammming_loss
)

from pyppi.models.binary_relevance import MixedBinaryRelevanceClassifier
from pyppi.models.classifier_chain import KRandomClassifierChains
from pyppi.models.utilities import make_gridsearch_clf

from pyppi.data_mining.ontology import get_active_instance

from pyppi.predict.utilities import load_validation_dataset
from pyppi.predict.utilities import interactions_to_Xy_format
from pyppi.predict.plotting import plot_heatmaps


from sklearn.exceptions import UndefinedMetricWarning, FitFailedWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import clone
from sklearn.metrics import (
    label_ranking_loss, hamming_loss,
    f1_score, precision_score, recall_score
)

RANDOM_STATE = 42
MAX_RAND = MAX_SEED
logger = create_logger("scripts", logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

if __name__ == "__main__":
    args = docopt(__doc__)
    args = parse_args(args)
    n_jobs = args['n_jobs']
    n_splits = args['n_splits']
    n_iter = args['n_iterations']
    induce = args['induce']
    verbose = args['verbose']
    selection = args['selection']
    model = args['model']
    direc = args['directory']
    hyperparam_iter = args['h_iterations']
    use_binary = args['binary']
    chain = args['chain']
    folder = args['output_folder']

    # Set up the folder for each experiment run named after the current time
    rng = RandomState(seed=RANDOM_STATE)
    if folder is None:
        folder = datetime.now().strftime("val_%y-%m-%d_%H-%M")
    direc = "{}/{}/".format(direc, folder)
    su_make_dir(direc)
    seeds_clf = [int(rng.randint(1, MAX_RAND)) for i in range(n_iter)]
    seeds_kfold = [int(rng.randint(1, MAX_RAND)) for i in range(n_iter)]
    args["RandomState"] = {}
    args["RandomState"]["seeds_clf"] = seeds_clf
    args["RandomState"]["seeds_kfold"] = seeds_kfold
    args["RandomState"]["random_state"] = RANDOM_STATE
    args["RandomState"]["max_randint"] = MAX_RAND
    json.dump(
        args,
        fp=open("{}/settings.json".format(direc), 'w'),
        indent=4,
        sort_keys=True
    )

    # Load all the training data, features etc.
    # ------------------------------------------------------------------- #
    logger.info("Loading training and testing data.")
    ipr_map = ipr_name_map()
    pfam_map = pfam_name_map()
    go_dag = get_active_instance()

    # Get the features into X, and multilabel y indicator format
    # -------------------------------------------------------------------- #
    logger.info("Preparing training and testing data.")
    data = load_validation_dataset(selection=selection, taxon_id=9606)
    labels = data['labels']
    X_train, y_train = data["training"]
    X_test, y_test = data["testing"]
    mlb = data["binarizer"]

    logging.info("Computing class distributions.")
    counter = {l: int(c) for l, c in zip(mlb.classes, y_train.sum(axis=0))}
    counter["n_samples"] = int(y_train.shape[0])
    json.dump(
        counter,
        fp=open("{}/training_distribution.json".format(direc), 'w'),
        indent=4, sort_keys=True
    )

    counter = Counter([l for ls in y_test for l in ls])
    counter = {l: int(c) for l, c in zip(mlb.classes, y_test.sum(axis=0))}
    counter["n_samples"] = int(y_test.shape[0])
    json.dump(
        counter,
        fp=open("{}/testing_distribution.json".format(direc), 'w'),
        indent=4, sort_keys=True
    )

    # Set up the numpy arrays and dictionarys for statistics etc
    # -------------------------------------------------------------------- #
    logger.info("Setting up preliminaries and the statistics arrays")
    logger.info("Found classes {}".format(', '.join(mlb.classes_)))
    n_labels = len(mlb.classes_)
    binary_scoring_funcs = [
        ('Binary F1', f1_score),
        ('Precision', precision_score),
        ('Recall', recall_score),
        ('Specificity', specificity),
        ('FDR', fdr_score)
    ]
    multilabel_scoring_funcs = [
        ('Label Ranking Loss', label_ranking_loss),
        ('Macro (weighted) F1', f1_score),
        ('Macro (unweighted) F1', f1_score),
        ('Samples F1', f1_score),
        ('Hamming Loss', hamming_loss),
        ('Positive Label Hamming Loss', positive_label_hammming_loss)
    ]
    n_scorers = len(binary_scoring_funcs)
    n_ml_scorers = len(multilabel_scoring_funcs)

    # 2: position 0 is for validation, position 1 is for testing
    binary_statistics = np.zeros((n_labels, 2, n_scorers, n_iter, n_splits))
    multilabel_statistics = np.zeros((2, n_ml_scorers, n_iter, n_splits))
    rng = RandomState(seed=RANDOM_STATE)

    fp = open("{}/stability.txt".format(direc), 'wt')

    # Begin the main show!
    # ------------------------------------------------------------------- #
    for bs_iter in range(n_iter):
        logger.info("Fitting bootstrap iteration {}.".format(bs_iter + 1))
        pipeline = make_gridsearch_clf(
            model=model,
            rcv_splits=3,
            rcv_iter=hyperparam_iter,
            scoring="f1",
            binary=use_binary,
            n_jobs_model=n_jobs,
            n_jobs_gs=n_jobs,
            random_state=seeds_clf[bs_iter],
            make_pipeline=False,
            search_vectorizer=False
        )

        estimators = [clone(pipeline) for _ in mlb.classes]
        br_clf = MixedBinaryRelevanceClassifier(
            estimators, n_jobs=1, verbose=verbose
        )
        clf = StratifiedKFoldCrossValidation(
            estimator=br_clf,
            n_folds=n_splits,
            shuffle=True,
            n_jobs=1,
            random_state=seeds_kfold[bs_iter],
            verbose=verbose
        )
        vec = CountVectorizer(binary=use_binary, lowercase=False)
        clf.fit(X_train, y_train, vectorizer=vec)

        fp.write(f"\nBS {bs_iter}\n")
        for i, br in enumerate(clf.fold_estimators_):
            fp.write(f"\tfold {i+1}\n")
            for l, label in enumerate(mlb.classes):
                fp.write(f"\t\tlabel {label}\n")
                fp.write(f"\t\t\t{br.estimators_[l].best_score_}\n")
                fp.write(f"\t\t\t{br.estimators_[l].best_params_}\n")

        logger.info("\tComputing cross-validation scores.")
        for func_idx, (func_name, func) in enumerate(binary_scoring_funcs):
            score_params = {
                "scorer": func,
                "average": "binary",
                "use_proba": False
            }
            scores_v = clf.score(
                X_train, y_train, avg_folds=False, validation=True,
                **score_params
            )
            scores_t = clf.score(
                X_test, y_test, avg_folds=False, validation=False,
                **score_params
            )
            binary_statistics[:, 0, func_idx, bs_iter, :] = scores_v
            binary_statistics[:, 1, func_idx, bs_iter, :] = scores_t

        for func_idx, (func_name, func) in enumerate(multilabel_scoring_funcs):
            score_params = {
                "scorer": func,
                "use_proba": False
            }
            if func_name == 'Macro (weighted) F1':
                score_params["average"] = "weighted"
            if func_name == 'Macro (unweighted) F1':
                score_params["average"] = "macro"
            elif func_name == 'Samples F1':
                score_params["average"] = "samples"

            scores_v = clf.score(
                X_train, y_train, avg_folds=False, validation=True,
                **score_params
            )
            scores_t = clf.score(
                X_test, y_test, avg_folds=False, validation=False,
                **score_params
            )

            multilabel_statistics[0, func_idx, bs_iter, :] = scores_v
            multilabel_statistics[1, func_idx, bs_iter, :] = scores_t

    # Write out all the statistics to a multi-indexed dataframe
    # -------------------------------------------------------------------- #
    logger.info("Writing statistics to file.")

    # Binary Statistics
    # -------------------------------------------------------------------- #
    dim_a_size = len(mlb.classes_) * 2 * len(binary_scoring_funcs)
    dim_b_size = n_iter * n_splits

    func_names = [n for n, _ in binary_scoring_funcs]
    iterables = [mlb.classes_, ["validation", "holdout"], func_names]
    names = ['Labels', 'Condition', 'Metric']
    tuples = list(product(*iterables))
    index = pd.MultiIndex.from_tuples(tuples, names=names)

    names = ['Bootstrap Iteration', 'Fold Iteration']
    arrays = [
        ['B{}'.format(i + 1) for i in range(n_iter)],
        ['F{}'.format(i + 1) for i in range(n_splits)]
    ]
    tuples = list(product(*arrays))
    columns = pd.MultiIndex.from_tuples(tuples, names=names)

    binary_df = pd.DataFrame(
        binary_statistics.reshape((dim_a_size, dim_b_size)),
        index=index, columns=columns
    ).sort_index()
    binary_df.to_csv('{}/{}'.format(direc, 'binary_stats.csv'), sep=',')

    # Multi-label Statistics
    # -------------------------------------------------------------------- #
    dim_a_size = 2 * len(multilabel_scoring_funcs)
    dim_b_size = n_iter * n_splits

    func_names = [n for n, _ in multilabel_scoring_funcs]
    iterables = [["validation", "holdout"], func_names]
    names = ['Condition', 'Metric']
    tuples = list(product(*iterables))
    index = pd.MultiIndex.from_tuples(tuples, names=names)

    names = ['Bootstrap Iteration', 'Fold Iteration']
    arrays = [
        ['B{}'.format(i + 1) for i in range(n_iter)],
        ['F{}'.format(i + 1) for i in range(n_splits)]
    ]
    tuples = list(product(*arrays))
    columns = pd.MultiIndex.from_tuples(tuples, names=names)

    multilabel_df = pd.DataFrame(
        multilabel_statistics.reshape((dim_a_size, dim_b_size)),
        index=index, columns=columns
    ).sort_index()
    multilabel_df.to_csv(
        '{}/{}'.format(direc, 'multilabel_stats.csv'), sep=','
    )

    # Top N Features, train/y-array index order
    # -------------------------------------------------------------------- #
    logger.info("Writing label training order.")
    with open("{}/{}".format(direc, "label_order.csv"), 'wt') as fp:
        fp.write(",".join(mlb.classes))

    # Compute label similarity heatmaps and label correlation heatmap
    # -------------------------------------------------------------------- #
    label_features = {l: set() for l in mlb.classes_}
    for idx, label in enumerate(mlb.classes_):
        selector = y_train[:, idx] == 1
        positive_cases = X_train[selector]
        for feature_string in positive_cases:
            unique = set(feature_string.split(','))
            label_features[label] |= unique

    j_v_similarity_matrix = np.zeros((len(mlb.classes_), len(mlb.classes_)))
    d_v_similarity_matrix = np.zeros((len(mlb.classes_), len(mlb.classes_)))
    for i, class_1 in enumerate(sorted(mlb.classes_)):
        for j, class_2 in enumerate(sorted(mlb.classes_)):
            set_1 = label_features[class_1]
            set_2 = label_features[class_2]
            jaccard = len(set_1 & set_2) / len(set_1 | set_2)
            dice = 2 * len(set_1 & set_2) / (len(set_1) + len(set_2))
            j_v_similarity_matrix[i, j] = jaccard
            d_v_similarity_matrix[i, j] = dice

    # Create label correlation matrix and then create a new one
    # Where the columns and rows are in alphabetical order.
    label_correlation, _ = sp.stats.spearmanr(y_train)
    s_label_correlation = np.zeros_like(label_correlation)
    for i, class_1 in enumerate(sorted(mlb.classes_)):
        for j, class_2 in enumerate(sorted(mlb.classes_)):
            index_1 = list(mlb.classes_).index(class_1)
            index_2 = list(mlb.classes_).index(class_2)
            s_label_correlation[i, j] = label_correlation[index_1, index_2]

    header = "Columns: {}\nRows: {}".format(
        ','.join(sorted(mlb.classes_)), ','.join(sorted(mlb.classes_))
    )
    np.savetxt(
        X=j_v_similarity_matrix,
        fname='{}/{}'.format(direc, 'j_v_similarity_matrix.csv'),
        header=header, delimiter=','
    )
    np.savetxt(
        X=d_v_similarity_matrix,
        fname='{}/{}'.format(direc, 'd_v_similarity_matrix.csv'),
        header=header, delimiter=','
    )
    np.savetxt(
        X=s_label_correlation, fname='{}/{}'.format(
            direc, 'label_spearmanr.csv'),
        header=header, delimiter=','
    )

    # Compute label similarity heatmaps for the holdout set
    # -------------------------------------------------------------------- #
    holdout_labels = ('dephosphorylation', 'phosphorylation')
    holdout_label_features = {l: set() for l in holdout_labels}
    for idx, label in enumerate(mlb.classes_):
        if label in holdout_labels:
            selector = y_test[:, idx] == 1
            positive_cases = X_test[selector]
            for feature_string in positive_cases:
                unique = set(feature_string.split(','))
                holdout_label_features[label] |= unique

    j_t_similarity_matrix = np.zeros((2, len(mlb.classes_)))
    d_t_similarity_matrix = np.zeros((2, len(mlb.classes_)))
    for i, class_1 in enumerate(sorted(holdout_labels)):
        for j, class_2 in enumerate(sorted(mlb.classes_)):
            set_1 = holdout_label_features[class_1]
            set_2 = label_features[class_2]
            jaccard = len(set_1 & set_2) / len(set_1 | set_2)
            dice = 2 * len(set_1 & set_2) / (len(set_1) + len(set_2))
            j_t_similarity_matrix[i, j] = jaccard
            d_t_similarity_matrix[i, j] = dice

    header = "Columns: {}\nRows: {}".format(
        ','.join(sorted(mlb.classes_)), ','.join(sorted(holdout_labels))
    )
    np.savetxt(
        X=j_t_similarity_matrix,
        fname='{}/{}'.format(direc, 'j_t_similarity_matrix.csv'),
        header=header, delimiter=','
    )
    np.savetxt(
        X=d_t_similarity_matrix,
        fname='{}/{}'.format(direc, 'd_t_similarity_matrix.csv'),
        header=header, delimiter=','
    )

    # Compute some quick stats
    # -------------------------------------------------------------------- #
    with open('{}/{}'.format(direc, 'results_f1_ml.tsv'), 'wt') as fp:
        for label in mlb.classes:
            mean = binary_df.loc[(label, 'validation', 'Binary F1'), :].mean(
                axis=0, level=[0]
            ).mean()
            stdev = binary_df.loc[(label, 'validation', 'Binary F1'), :].mean(
                axis=0, level=[0]
            ).std()
            stderr = stdev / np.sqrt(n_iter)
            fp.write("{}\t{:.6f}\t{:.6f}\n".format(label, mean, stderr))

            if label in ("Phosphorylation", "Dephosphorylation"):
                mean = binary_df.loc[(label, 'holdout', 'Binary F1'), :].mean(
                    axis=0, level=[0]
                ).mean()
                stdev = binary_df.loc[(label, 'holdout', 'Binary F1'), :].mean(
                    axis=0, level=[0]
                ).std()
                stderr = stdev / np.sqrt(n_iter)
                fp.write("{} (HPRD)\t{:.6f}\t{:.6f}\n".format(
                    label, mean, stderr)
                )

        for metric, _ in multilabel_scoring_funcs:
            mean = multilabel_df.loc[('validation', metric), :].mean(
                axis=0, level=[0]
            ).mean()
            stdev = multilabel_df.loc[('validation', metric), :].mean(
                axis=0, level=[0]
            ).std()
            stderr = stdev / np.sqrt(n_iter)
            fp.write("{}\t{:.6f}\t{:.6f}\n".format(metric, mean, stderr))

    # ---------------- Rise and shine, it's plotting time! ---------------- #
    plot_heatmaps(
        "{}/heat_maps_jaccard.png".format(direc),
        labels=mlb.classes,
        correlation_matrix=s_label_correlation,
        similarity_matrix=j_v_similarity_matrix,
        dpi=350, format='png'
    )
    plot_heatmaps(
        "{}/heat_maps_dice.png".format(direc),
        labels=mlb.classes,
        correlation_matrix=s_label_correlation,
        similarity_matrix=d_v_similarity_matrix,
        dpi=350, format='png'
    )
