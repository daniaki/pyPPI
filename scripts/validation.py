#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.

Usage:
  validation.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
             [--use_cache] [--induce] [--verbose]
             [--model=M] [--n_jobs=J] [--n_splits=S] [--n_iterations=I]
             [--directory=DIR]
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
  --model=M         A binary classifier from Scikit-Learn implementing fit,
                    predict and predict_proba [default: LogisticRegression]
  --n_jobs=J        Number of processes to run in parallel [default: 1]
  --n_splits=S      Number of cross-validation splits [default: 5]
  --n_iterations=I  Number of bootstrap iterations [default: 5]
  --directory=DIR   Output directory [default: ./results/]
"""

import numpy as np
from datetime import datetime
from docopt import docopt
args = docopt(__doc__)

from pyPPI.base import parse_args, su_make_dir, pretty_print_dict
from pyPPI.data import load_network_from_path, load_ptm_labels
from pyPPI.data import testing_network_path, training_network_path

from pyPPI.models.binary_relevance import BinaryRelevance
from pyPPI.models import make_classifier
from pyPPI.model_selection.scoring import MultilabelScorer, Statistics
from pyPPI.model_selection.experiment import KFoldExperiment, Bootstrap
from pyPPI.model_selection.sampling import IterativeStratifiedKFold

from pyPPI.data_mining.features import AnnotationExtractor
from pyPPI.data_mining.uniprot import UniProt, get_active_instance
from pyPPI.data_mining.tools import xy_from_interaction_frame

from sklearn.base import clone
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score, make_scorer


if __name__ == '__main__':
    args = parse_args(args)
    n_jobs = args['n_jobs']
    n_splits =args['n_splits']
    n_iter = args['n_iterations']
    induce = args['induce']
    verbose = args['verbose']
    selection = args['selection']
    model = args['model']
    use_feature_cache = args['use_cache']
    direc = args['directory']

    # Set up the folder for each experiment run named after the current time
    folder = datetime.now().strftime("val_%y-%m-%d_%H-%M-%S")
    direc = "{}/{}/".format(direc, folder)
    su_make_dir(direc)
    pretty_print_dict(args, open("{}/settings.json".format(direc)))

    print("Loading data...")
    uniprot = get_active_instance(verbose=verbose)
    data_types = UniProt.data_types()
    labels = load_ptm_labels()
    annotation_ex = AnnotationExtractor(
        induce=induce,
        selection=selection,
        n_jobs=n_jobs,
        verbose=verbose,
        cache=use_feature_cache
    )
    training = load_network_from_path(training_network_path)
    testing = load_network_from_path(testing_network_path)

    # Get the features into X, and multilabel y indicator format
    print("Preparing training and testing data...")
    mlb = MultiLabelBinarizer(classes=labels)
    X_train_ppis, y_train = xy_from_interaction_frame(training)
    X_test_ppis, y_test = xy_from_interaction_frame(testing)
    mlb.fit(y_train)

    X_train = annotation_ex.transform(X_train_ppis)
    X_test = annotation_ex.transform(X_test_ppis)
    y_train = mlb.transform(y_train)
    y_test = mlb.transform(y_test)

    # Make the estimators and BR classifier
    print("Making classifier...")
    param_distribution = {
        'C': np.arange(0.01, 10.01, step=0.01),
        'penalty': ['l1', 'l2']
    }
    random_cv = RandomizedSearchCV(
        cv=3,
        n_iter=60,
        param_distributions=param_distribution,
        estimator=make_classifier('LogisticRegression'),
        scoring=make_scorer(f1_score, greater_is_better=True)
    )
    estimators = [
        Pipeline(
            [('vectorizer', CountVectorizer(binary=False)),
             ('clf', clone(random_cv))]
        )
        for l in labels
    ]
    clf = BinaryRelevance(estimators, n_jobs=n_jobs)

    # Make the bootstrap and KFoldExperiments
    print("Setting up experiments...")
    cv = IterativeStratifiedKFold(n_splits=n_splits, shuffle=True)
    kf = KFoldExperiment(
        estimator=clf, cv=cv, n_jobs=n_splits,
        verbose=verbose, backend='threading'
    )
    bootstrap = Bootstrap(
        kfold_experiemnt=kf, n_iter=n_iter, n_jobs=n_jobs,
        verbose=verbose, backend='multiprocessing'
    )

    # Fit the data
    print("Fitting training data...")
    bootstrap.fit(X_train, y_train)

    # Make the scoring functions
    print("Evaluating performance...")
    f1_scorer = MultilabelScorer(f1_score)
    recall_scorer = MultilabelScorer(recall_score)
    precision_scorer = MultilabelScorer(precision_score)
    score_funcs = [recall_scorer, precision_scorer, f1_scorer]
    thresholds = [0.5, 0.5, 0.5]

    # Evaluate performance
    validation_data = bootstrap.validation_scores(
        X_train, y_train, score_funcs, thresholds, True, False
    )
    testing_data = bootstrap.held_out_scores(
        X_test, y_test, score_funcs, thresholds, True, False
    )

    # Put everything into a dataframe
    print("Saving statistics dataframes...")
    validation_stats = Statistics.statistics_from_data(
        data=validation_data,
        statistics_names=['Recall', 'Precision', 'F1'],
        classes=mlb.classes_,
        return_df=True
    )
    # Put everything into a dataframe
    testing_stats = Statistics.statistics_from_data(
        data=testing_data,
        statistics_names=['Recall', 'Precision', 'F1'],
        classes=mlb.classes_,
        return_df=True
    )
    validation_stats.to_csv(
        '{}/validation_stats.csv'.format(direc), sep='\t', index=False
    )
    testing_stats.to_csv(
        '{}/testing_stats.csv'.format(direc), sep='\t', index=False
    )

