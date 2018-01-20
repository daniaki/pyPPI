"""
This script runs the bootstrap kfold validation experiments as used in
the publication.

Usage:
  validation.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
             [--induce] [--verbose] [--top=T]
             [--model=M] [--n_jobs=J] [--n_splits=S] [--n_iterations=I]
             [--h_iterations=H] [--directory=DIR]
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
  --verbose     Print intermediate output for debugging.
  --top=T       Top T features for each label to log [default: 25]
  --model=M         A binary classifier from Scikit-Learn implementing fit,
                    predict and predict_proba [default: LogisticRegression]
  --n_jobs=J        Number of processes to run in parallel [default: 1]
  --n_splits=S      Number of cross-validation splits [default: 5]
  --h_iterations=H  Number of hyperparameter tuning
                    iterations per fold [default: 60]
  --n_iterations=I  Number of bootstrap iterations [default: 5]
  --directory=DIR   Output directory [default: ./results/]
"""

import json
import logging
import pandas as pd
import numpy as np
import scipy as sp
from itertools import product
from operator import itemgetter
from collections import Counter
from datetime import datetime
from docopt import docopt
import warnings

from pyppi.base import parse_args, su_make_dir
from pyppi.data import load_network_from_path, load_ptm_labels
from pyppi.data import testing_network_path, training_network_path
from pyppi.data import get_term_description, ipr_name_map, pfam_name_map

from pyppi.models.utilities import get_coefs, top_n_features
from pyppi.models import make_classifier, get_parameter_distribution_for_model
from pyppi.models import supported_estimators
from pyppi.model_selection.scoring import fdr_score, specificity
from pyppi.model_selection.sampling import IterativeStratifiedKFold

from pyppi.data_mining.ontology import get_active_instance

from pyppi.database import begin_transaction
from pyppi.database.models import Interaction
from pyppi.database.managers import InteractionManager, format_interactions_for_sklearn

from sklearn.base import clone
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import (
    recall_score, make_scorer,
    label_ranking_average_precision_score,
    label_ranking_loss,
    confusion_matrix
)

MAX_SEED = 1000000
logger = logging.getLogger("scripts")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def train_fold(X, y, labels, fold_iter, use_binary, model,
               hyperparam_iter, rng, params):
    logger.info("Fitting fold {}.".format(fold_iter + 1))

    # Prepare all training and testing data
    vectorizer = CountVectorizer(
        binary=True if use_binary else False,
        lowercase=False, stop_words=[':', 'GO']
    )
    X = vectorizer.fit_transform(X)

    requires_dense = False
    estimators = []
    for i, label in enumerate(labels):
        logger.info("\tFitting label {}.".format(label))
        model_to_tune = make_classifier(
            algorithm=model,
            random_state=rng.randint(MAX_SEED)
        )
        clf = RandomizedSearchCV(
            estimator=model_to_tune,
            scoring='f1',
            error_score=0,
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=rng.randint(MAX_SEED)
            ),
            n_iter=hyperparam_iter,
            n_jobs=n_jobs,
            refit=True,
            random_state=rng.randint(MAX_SEED),
            param_distributions=params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X, y[:, i])
                requires_dense = False
            except TypeError:
                logger.info(
                    "Error fitting sparse input. Converting to dense input."
                )
                X = X.todense()
                clf.fit(X, y[:, i])
                requires_dense = True
        estimators.append(clf)
    return estimators, vectorizer, requires_dense


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
    use_feature_cache = args['use_cache']
    direc = args['directory']
    hyperparam_iter = args['h_iterations']
    get_top_n = args['top']
    use_binary = args['binary']

    # Set up the folder for each experiment run named after the current time
    folder = datetime.now().strftime("val_%y-%m-%d_%H-%M")
    direc = "{}/{}/".format(direc, folder)
    su_make_dir(direc)
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
    i_manager = InteractionManager(verbose=verbose, match_taxon_id=9606)

    with begin_transaction() as session:
        labels = i_manager.training_labels(session, include_holdout=False)
        training = i_manager.training_interactions(
            session, filter_out_holdout=True)
        testing = i_manager.holdout_interactions(
            session, filter_out_training=True)

    # Get the features into X, and multilabel y indicator format
    # -------------------------------------------------------------------- #
    logger.info("Preparing training and testing data.")
    X_train, y_train = format_interactions_for_sklearn(training, selection)
    X_test, y_test = format_interactions_for_sklearn(testing, selection)

    logging.info("Computing class distributions.")
    json.dump(
        Counter([l for ls in y_train for l in ls]),
        fp=open("{}/training_distribution.json".format(direc), 'w'),
        indent=4, sort_keys=True
    )
    json.dump(
        Counter([l for ls in y_test for l in ls]),
        fp=open("{}/testing_distribution.json".format(direc), 'w'),
        indent=4, sort_keys=True
    )

    mlb = MultiLabelBinarizer(classes=labels, sparse_output=False)
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    # Set up the numpy arrays and dictionarys for statistics etc
    # -------------------------------------------------------------------- #
    logger.info("Setting up preliminaries and the statistics arrays")
    logger.info("Found classes {}".format(', '.join(mlb.classes_)))
    n_classes = len(mlb.classes_)
    rng = np.random.RandomState(seed=42)
    max_seed = 1000000
    top_features = {
        "absolute": {
            l: {
                i: {
                    j: [] for j in range(n_splits)
                } for i in range(n_iter)
            } for l in mlb.classes_
        },
        "not_absolute": {
            l: {
                i: {
                    j: [] for j in range(n_splits)
                } for i in range(n_iter)
            } for l in mlb.classes_
        }
    }
    params = get_parameter_distribution_for_model(model)

    binary_scoring_funcs = [
        ('Binary F1', f1_score),
        ('Precision', precision_score),
        ('Recall', recall_score),
        ('Specificity', specificity),
        ('FDR', fdr_score)
    ]
    multilabel_scoring_funcs = [
        ('Label Ranking Loss', label_ranking_loss),
        ('Label Ranking Average Precision',
            label_ranking_average_precision_score),
        ('Macro (weighted) F1', f1_score),
        ('Macro (un-weighted) F1', f1_score)
    ]
    n_scorers = len(binary_scoring_funcs)
    n_ml_scorers = len(multilabel_scoring_funcs)

    # 2: position 0 is for validation, position 1 is for testing
    binary_statistics = np.zeros((n_classes, 2, n_scorers, n_iter, n_splits))
    multilabel_statistics = np.zeros((2, n_ml_scorers, n_iter, n_splits))

    # Begin the main show!
    # ------------------------------------------------------------------- #
    for bs_iter in range(n_iter):
        logger.info("Fitting bootstrap iteration {}.".format(bs_iter + 1))
        cv = IterativeStratifiedKFold(
            n_splits=n_splits, random_state=rng.randint(MAX_SEED)
        )
        cv = list(cv.split(X_train, y_train))

        fit_results = []
        for fold_iter, (train_idx, _) in enumerate(cv):
            clf_tuple = train_fold(
                X=X_train[train_idx],
                y=y_train[train_idx],
                labels=mlb.classes_,
                fold_iter=fold_iter,
                use_binary=use_binary,
                model=model,
                hyperparam_iter=hyperparam_iter,
                rng=rng,
                params=params
            )
            fit_results.append(clf_tuple)

        for fold_iter, ((_, validation_idx), (estimators, vectorizer, requires_dense)) in enumerate(zip(cv, fit_results)):
            logger.info(
                "Computing binary performance for fold {}.".format(fold_iter + 1))
            y_valid_f_pred = []
            y_test_f_pred = []
            y_valid_f_proba = []
            y_test_f_proba = []

            for clf, (label_idx, label) in zip(estimators, enumerate(mlb.classes_)):
                logger.info(
                    "\tComputing binary performance for label {}.".format(label))

                X_valid_l = vectorizer.transform(X_train[validation_idx])
                y_valid_l = y_train[validation_idx, label_idx]

                X_test_l = vectorizer.transform(X_test)
                y_test_l = y_test[:, label_idx]

                if requires_dense:
                    X_valid_l = X_valid_l.todense()
                    y_valid_l = y_valid_l.todense()
                    X_test_l = X_test_l.todense()
                    y_test_l = y_test_l.todense()

                # Validation scores in binary and probability format
                y_valid_l_pred = clf.predict(X_valid_l)
                y_valid_l_proba = clf.predict_proba(X_valid_l)

                # Held-out testing scores in binary and probability format
                y_test_l_pred = clf.predict(X_test_l)
                y_test_l_proba = clf.predict_proba(X_test_l)

                # Store these per label results in a list which we will
                # later use to stack into a multi-label array.
                y_valid_f_pred.append([[x] for x in y_valid_l_pred])
                y_valid_f_proba.append([[x[1]] for x in y_valid_l_proba])

                y_test_f_pred.append([[x] for x in y_test_l_pred])
                y_test_f_proba.append([[x[1]] for x in y_test_l_proba])

                # Perform scoring on the validation set and the external testing set.

                for func_idx, (func_name, func) in enumerate(binary_scoring_funcs):
                    if func_name in ['Specificity', 'FDR']:
                        scores_v = func(y_valid_l, y_valid_l_pred)
                        scores_t = func(y_test_l, y_test_l_pred)
                    else:
                        scores_v = func(
                            y_valid_l, y_valid_l_pred, average='binary')
                        scores_t = func(
                            y_test_l, y_test_l_pred, average='binary')
                    binary_statistics[label_idx, 0, func_idx,
                                      bs_iter, fold_iter] = scores_v
                    binary_statistics[label_idx, 1, func_idx,
                                      bs_iter, fold_iter] = scores_t

                # Get the top 20 features for this labels's run.
                top_n = top_n_features(
                    clf=clf,
                    go_dag=go_dag,
                    ipr_map=ipr_map,
                    pfam_map=pfam_map,
                    n=get_top_n,
                    absolute=False,
                    vectorizer=vectorizer
                )
                top_n_abs = top_n_features(
                    clf=clf,
                    go_dag=go_dag,
                    ipr_map=ipr_map,
                    pfam_map=pfam_map,
                    n=get_top_n,
                    absolute=True,
                    vectorizer=vectorizer
                )
                top_features["not_absolute"][label][bs_iter][fold_iter].extend(
                    top_n)
                top_features["absolute"][label][bs_iter][fold_iter].extend(
                    top_n_abs)

            logger.info("Computing fold mult-label performance.")
            # True scores in multi-label indicator format
            y_valid_f = y_train[validation_idx]
            y_test_f = y_test

            # Validation scores in multi-label indicator format
            y_valid_f_pred = np.hstack(y_valid_f_pred)
            y_valid_f_proba = np.hstack(y_valid_f_proba)

            # Testing scores in multi-label probability format
            y_test_f_pred = np.hstack(y_test_f_pred)
            y_test_f_proba = np.hstack(y_test_f_proba)

            for func_idx, (func_name, func) in enumerate(multilabel_scoring_funcs):
                if func_name == 'Macro (weighted) F1':
                    scores_v = func(y_valid_f, y_valid_f_pred,
                                    average='weighted')
                    scores_t = func(y_test_f, y_test_f_pred,
                                    average='weighted')
                elif func_name == 'Macro (un-weighted) F1':
                    scores_v = func(y_valid_f, y_valid_f_pred, average='macro')
                    scores_t = func(y_test_f, y_test_f_pred, average='macro')
                elif func_name == 'Label Ranking Average Precision':
                    scores_v = func(y_valid_f, y_valid_f_proba)
                    scores_t = func(y_test_f, y_test_f_proba)
                else:
                    scores_v = func(y_valid_f, y_valid_f_pred)
                    scores_t = func(y_test_f, y_test_f_pred)

                multilabel_statistics[0, func_idx,
                                      bs_iter, fold_iter] = scores_v
                multilabel_statistics[1, func_idx,
                                      bs_iter, fold_iter] = scores_t

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
        fp.write(",".join(mlb.classes_))

    logging.info("Writing top features to file.")
    with open('{}/{}'.format(direc, 'top_features.json'), 'wt') as fp:
        json.dump(top_features, fp, indent=4, sort_keys=True)

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
            index_1 = mlb.classes_.index(class_1)
            index_2 = mlb.classes_.index(class_2)
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
