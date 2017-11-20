"""
This script runs the bootstrap kfold validation experiments as used in
the publication.

Usage:
  validation.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
             [--use_cache] [--induce] [--verbose] [--abs] [--top=T]
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
  --induce      Use ULCA inducer over Gene Ontology.
  --verbose     Print intermediate output for debugging.
  --use_cache   Use cached features if available.
  --abs         Take the absolute value of feature weights when computing top
                features.
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

from pyppi.base import parse_args, su_make_dir
from pyppi.data import load_network_from_path, load_ptm_labels
from pyppi.data import testing_network_path, training_network_path
from pyppi.data import get_term_description, ipr_name_map, pfam_name_map

from pyppi.models.binary_relevance import get_coefs, top_n_features
from pyppi.models import make_classifier, get_parameter_distribution_form_model
from pyppi.models import supported_estimators
from pyppi.model_selection.scoring import fdr_score, specificity
from pyppi.model_selection.sampling import IterativeStratifiedKFold

from pyppi.data_mining.features import AnnotationExtractor
from pyppi.data_mining.uniprot import UniProt, get_active_instance
from pyppi.data_mining.tools import xy_from_interaction_frame
from pyppi.data_mining.ontology import get_active_instance as load_godag

from sklearn.base import clone
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import (
    recall_score, make_scorer,
    label_ranking_average_precision_score,
    label_ranking_loss,
    confusion_matrix
)

args = docopt(__doc__)
logging.captureWarnings(False)
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%m-%d-%Y %I:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
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
    abs_weights = args['abs']

    # Set up the folder for each experiment run named after the current time
    folder = datetime.now().strftime("val_%y-%m-%d_%H-%M")
    direc = "{}/{}/".format(direc, folder)
    su_make_dir(direc)
    json.dump(
        args, fp=open("{}/settings.json".format(direc), 'w'),
        indent=4, sort_keys=True
    )

    # Load all the training data, features etc.
    # ------------------------------------------------------------------- #
    logging.info("Loading training and testing data.")
    ipr_map = ipr_name_map(short_names=False)
    pfam_map = pfam_name_map()
    go_dag = load_godag()
    uniprot = get_active_instance(
        verbose=verbose,
        sprot_cache=False if use_feature_cache else True,
        trembl_cache=False if use_feature_cache else True
    )
    data_types = UniProt.data_types()
    labels = load_ptm_labels()
    annotation_ex = AnnotationExtractor(
        induce=induce,
        selection=selection,
        n_jobs=n_jobs,
        verbose=verbose,
        cache=use_feature_cache,
        backend='threading'
    )
    training = load_network_from_path(training_network_path)
    testing = load_network_from_path(testing_network_path)

    # Get the features into X, and multilabel y indicator format
    # -------------------------------------------------------------------- #
    logging.info("Preparing training and testing data.")
    mlb = MultiLabelBinarizer(classes=labels, sparse_output=False)
    X_train_ppis, y_train = xy_from_interaction_frame(training)
    X_test_ppis, y_test = xy_from_interaction_frame(testing)
    mlb.fit(y_train)

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

    X_train = annotation_ex.transform(X_train_ppis)
    X_test = annotation_ex.transform(X_test_ppis)
    y_train = mlb.transform(y_train)
    y_test = mlb.transform(y_test)

    # These take up around ~5GB so it's best to delete them from memory
    # -------------------------------------------------------------------- #
    del annotation_ex
    del uniprot

    # Set up the numpy arrays and dictionarys for statistics etc
    # -------------------------------------------------------------------- #
    logging.info("Setting up preliminaries and the statistics arrays")
    logging.info("Found classes {}".format(', '.join(mlb.classes)))
    n_classes = len(mlb.classes)
    seeds = range(n_iter)
    top_features = {
        l: {
            i: {
                j: [] for j in range(n_splits)
            } for i in range(n_iter)
        } for l in mlb.classes
    }
    params = get_parameter_distribution_form_model(model)

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
        logging.info("Fitting bootstrap iteration {}.".format(bs_iter + 1))
        cv = IterativeStratifiedKFold(
            n_splits=n_splits, random_state=seeds[bs_iter])

        for fold_iter, (train_idx, validation_idx) in \
                enumerate(cv.split(X_train, y_train)):
            logging.info("Fitting fold iteration {}.".format(fold_iter + 1))
            y_valid_f_pred = []
            y_test_f_pred = []
            y_valid_f_proba = []
            y_test_f_proba = []

            for label_idx, label in enumerate(mlb.classes):
                logging.info("Fitting label {}.".format(label))

                # Prepare all training and testing data
                logging.info("Preparing data.")
                vectorizer = CountVectorizer(binary=False)
                vectorizer.fit(X_train[train_idx])

                X_train_l = vectorizer.transform(X_train[train_idx])
                y_train_l = y_train[train_idx, label_idx]

                X_valid_l = vectorizer.transform(X_train[validation_idx])
                y_valid_l = y_train[validation_idx, label_idx]

                X_test_l = vectorizer.transform(X_test)
                y_test_l = y_test[:, label_idx]

                # Build and fit classifier
                logging.info("Fitting classifier.")
                base_est = make_classifier(
                    algorithm=model,
                    random_state=0
                )

                clf = RandomizedSearchCV(
                    estimator=base_est,
                    scoring='f1',
                    error_score=0,
                    cv=3,
                    n_iter=hyperparam_iter,
                    n_jobs=n_jobs,
                    refit=True,
                    random_state=0,
                    param_distributions=params,
                )
                try:
                    clf.fit(X_train_l, y_train_l)
                except TypeError:
                    logger.info(
                        "Error fitting sparse input. "
                        "Converting to dense input."
                    )
                    X_train_l = X_train_l.todense()
                    X_valid_l = X_valid_l.todense()
                    X_test_l = X_test_l.todense()
                    clf.fit(X_train_l, y_train_l)

                # Validation scores in binary and probability format
                logging.info("Making predictions on validation data.")
                y_valid_l_pred = clf.predict(X_valid_l)
                y_valid_l_proba = clf.predict_proba(X_valid_l)

                # Held-out testing scores in binary and probability format
                logging.info("Making predictions on testing data.")
                y_test_l_pred = clf.predict(X_test_l)
                y_test_l_proba = clf.predict_proba(X_test_l)

                # Store these per label results in a list which we will
                # later use to stack into a multi-label array.
                logging.info("Storing validation predictions.")
                y_valid_f_pred.append([[x] for x in y_valid_l_pred])
                y_valid_f_proba.append([[x[1]] for x in y_valid_l_proba])

                logging.info("Storing testing predictions.")
                y_test_f_pred.append([[x] for x in y_test_l_pred])
                y_test_f_proba.append([[x[1]] for x in y_test_l_proba])

                # Perform scoring on the validation set and the external
                #  testing set.
                logging.info("Computing fold label binary performance.")
                for func_idx, (func_name, func) in \
                        enumerate(binary_scoring_funcs):
                    if func_name in ['Specificity', 'FDR']:
                        scores_v = func(y_valid_l, y_valid_l_pred)
                        scores_t = func(y_test_l, y_test_l_pred)
                    else:
                        scores_v = func(
                            y_valid_l, y_valid_l_pred,
                            average='binary'
                        )
                        scores_t = func(
                            y_test_l, y_test_l_pred,
                            average='binary'
                        )
                    binary_statistics[
                        label_idx, 0, func_idx, bs_iter, fold_iter] = scores_v
                    binary_statistics[
                        label_idx, 1, func_idx, bs_iter, fold_iter] = scores_t

                logging.info("Computing top label features for fold.")
                # Get the top 20 features for this labels's run.
                top_n = top_n_features(
                    clf=clf,
                    go_dag=go_dag,
                    ipr_map=ipr_map,
                    pfam_map=pfam_map,
                    n=get_top_n,
                    absolute=abs_weights,
                    vectorizer=vectorizer
                )
                top_features[label][bs_iter][fold_iter].extend(top_n)

            logging.info("Computing fold mult-label performance.")
            # True scores in multi-label indicator format
            y_valid_f = y_train[validation_idx]
            y_test_f = y_test

            # Validation scores in multi-label indicator format
            y_valid_f_pred = np.hstack(y_valid_f_pred)
            y_valid_f_proba = np.hstack(y_valid_f_proba)

            # Testing scores in multi-label probability format
            y_test_f_pred = np.hstack(y_test_f_pred)
            y_test_f_proba = np.hstack(y_test_f_proba)

            for func_idx, (func_name, func) in \
                    enumerate(multilabel_scoring_funcs):
                if func_name == 'Macro (weighted) F1':
                    scores_v = func(
                        y_valid_f, y_valid_f_pred, average='weighted'
                    )
                    scores_t = func(
                        y_test_f, y_test_f_pred, average='weighted'
                    )
                elif func_name == 'Macro (un-weighted) F1':
                    scores_v = func(y_valid_f, y_valid_f_pred, average='macro')
                    scores_t = func(y_test_f, y_test_f_pred, average='macro')
                elif func_name == 'Label Ranking Average Precision':
                    scores_v = func(y_valid_f, y_valid_f_proba)
                    scores_t = func(y_test_f, y_test_f_proba)
                else:
                    scores_v = func(y_valid_f, y_valid_f_pred)
                    scores_t = func(y_test_f, y_test_f_pred)

                multilabel_statistics[
                    0, func_idx, bs_iter, fold_iter] = scores_v
                multilabel_statistics[
                    1, func_idx, bs_iter, fold_iter] = scores_t

    # Write out all the statistics to a multi-indexed dataframe
    # -------------------------------------------------------------------- #
    logging.info("Writing statistics to file.")

    # Binary Statistics
    # -------------------------------------------------------------------- #
    dim_a_size = len(mlb.classes) * 2 * len(binary_scoring_funcs)
    dim_b_size = n_iter * n_splits

    func_names = [n for n, _ in binary_scoring_funcs]
    iterables = [mlb.classes, ["validation", "holdout"], func_names]
    names = ['Labels', 'Condition', 'Metric']
    tuples = list(product(*iterables))
    index = pd.MultiIndex.from_tuples(tuples, names=names)

    names = ['Bootstrap Iteration', 'Fold Iteration']
    arrays = [
        ['B{}'.format(i) for i in range(n_iter)],
        ['F{}'.format(i) for i in range(n_splits)]
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
        ['B{}'.format(i) for i in range(n_iter)],
        ['F{}'.format(i) for i in range(n_splits)]
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
    logging.info("Writing label training order.")
    with open("{}/{}".format(direc, "label_order.csv"), 'wt') as fp:
        fp.write(",".join(mlb.classes))

    logging.info("Writing top features to file.")
    with open('{}/{}'.format(direc, 'top_features.json'), 'wt') as fp:
        json.dump(top_features, fp, indent=4, sort_keys=True)

# Compute label similarity heatmaps and label correlation heatmap
# -------------------------------------------------------------------- #
label_features = {l: set() for l in mlb.classes}
for idx, label in enumerate(mlb.classes):
    selector = y_train[:, idx] == 1
    positive_cases = X_train[selector]
    for feature_string in positive_cases:
        unique = set(feature_string.split(','))
        label_features[label] |= unique

j_v_similarity_matrix = np.zeros((len(mlb.classes), len(mlb.classes)))
d_v_similarity_matrix = np.zeros((len(mlb.classes), len(mlb.classes)))
for i, class_1 in enumerate(sorted(mlb.classes)):
    for j, class_2 in enumerate(sorted(mlb.classes)):
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
for i, class_1 in enumerate(sorted(mlb.classes)):
    for j, class_2 in enumerate(sorted(mlb.classes)):
        index_1 = mlb.classes.index(class_1)
        index_2 = mlb.classes.index(class_2)
        s_label_correlation[i, j] = label_correlation[index_1, index_2]


header = "Columns: {}\nRows: {}".format(
    ','.join(sorted(mlb.classes)), ','.join(sorted(mlb.classes))
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
    X=s_label_correlation, fname='{}/{}'.format(direc, 'label_spearmanr.csv'),
    header=header, delimiter=','
)

# Compute label similarity heatmaps for the holdout set
# -------------------------------------------------------------------- #
holdout_labels = ('dephosphorylation', 'phosphorylation')
holdout_label_features = {l: set() for l in holdout_labels}
for idx, label in enumerate(mlb.classes):
    if label in holdout_labels:
        selector = y_test[:, idx] == 1
        positive_cases = X_test[selector]
        for feature_string in positive_cases:
            unique = set(feature_string.split(','))
            holdout_label_features[label] |= unique

j_t_similarity_matrix = np.zeros((2, len(mlb.classes)))
d_t_similarity_matrix = np.zeros((2, len(mlb.classes)))
for i, class_1 in enumerate(sorted(holdout_labels)):
    for j, class_2 in enumerate(sorted(mlb.classes)):
        set_1 = holdout_label_features[class_1]
        set_2 = label_features[class_2]
        jaccard = len(set_1 & set_2) / len(set_1 | set_2)
        dice = 2 * len(set_1 & set_2) / (len(set_1) + len(set_2))
        print(class_1, class_2, jaccard)
        j_t_similarity_matrix[i, j] = jaccard
        d_t_similarity_matrix[i, j] = dice

header = "Columns: {}\nRows: {}".format(
    ','.join(sorted(mlb.classes)), ','.join(sorted(holdout_labels))
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
