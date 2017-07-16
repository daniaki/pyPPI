#!/usr/bin/env python

"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  predict_ppis.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
                  [--use_cache] [--retrain] [--induce] [--verbose]
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
  --use_cache   Use cached features if available.
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

import os
import json
import numpy as np
from datetime import datetime
from docopt import docopt

from pyppi.base import parse_args, su_make_dir
from pyppi.base import P1, P2, G1, G2
from pyppi.data import load_network_from_path, load_ptm_labels
from pyppi.data import full_training_network_path, generic_io
from pyppi.data import interactome_network_path, classifier_path

from pyppi.models.binary_relevance import BinaryRelevance
from pyppi.models import make_classifier

from pyppi.data_mining.features import AnnotationExtractor
from pyppi.data_mining.uniprot import UniProt, get_active_instance
from pyppi.data_mining.tools import xy_from_interaction_frame
from pyppi.data_mining.generic import edgelist_func, generic_to_dataframe
from pyppi.data_mining.tools import map_network_accessions

from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer

args = docopt(__doc__)

if __name__ == '__main__':
    args = parse_args(args)
    n_jobs = args['n_jobs']
    n_splits = args['n_splits']
    rcv_iter = args['n_iterations']
    induce = args['induce']
    verbose = args['verbose']
    selection = args['selection']
    model = args['model']
    use_feature_cache = args['use_cache']
    out_file = args['output']
    input_file = args['input']
    direc = args['directory']
    retrain = args['retrain']

    # Set up the folder for each experiment run named after the current time
    folder = datetime.now().strftime("pred_%y-%m-%d_%H-%M-%S")
    direc = "{}/{}/".format(direc, folder)
    su_make_dir(direc)
    json.dump(args, fp=open("{}/settings.json".format(direc), 'w'), indent=4,
              sort_keys=True)
    out_file = open("{}/{}".format(direc, out_file), "w")

    print("Loading feature data...")
    uniprot = get_active_instance(verbose=verbose)
    data_types = UniProt.data_types()
    selection = [
        data_types.GO_MF.value,
        data_types.GO_BP.value,
        data_types.GO_CC.value,
        data_types.INTERPRO.value,
        data_types.PFAM.value
    ]
    labels = load_ptm_labels()
    annotation_ex = AnnotationExtractor(
        induce=induce,
        selection=selection,
        n_jobs=n_jobs,
        verbose=verbose,
        cache=use_feature_cache
    )

    # Get the input edge-list ready
    if input_file == 'None':
        print("Loading interactome data...")
        testing = load_network_from_path(interactome_network_path)
    else:
        print("Loading custom ppi data...")
        testing = generic_to_dataframe(
            f_input=generic_io(input_file),
            parsing_func=edgelist_func,
            drop_nan=True,
            allow_self_edges=True,
            allow_duplicates=True
        )
        sources = set(p for df in testing for p in df.source.values)
        targets = set(p for df in testing for p in df.target.values)
        accessions = list(sources | targets)
        accession_mapping = uniprot.batch_map(accessions)
        testing = map_network_accessions(
            interactions=testing, accession_map=accession_mapping,
            drop_nan=True, allow_self_edges=True,
            allow_duplicates=False, min_counts=None, merge=False
        )

    # Get the features into X, and multilabel y indicator format
    print("Preparing training and testing data...")
    training = load_network_from_path(full_training_network_path)
    X_train_ppis, y_train = xy_from_interaction_frame(training)
    X_test_ppis, _ = xy_from_interaction_frame(testing)
    X_train = annotation_ex.transform(X_train_ppis)
    X_test = annotation_ex.transform(X_test_ppis)

    # Get all annotations used during training
    training_go = set([
        g.strip() for gs in X_train
        for g in x.split(',')
        if 'go' in g.strip().lower()
    ])
    training_pfam = set([
        g.strip() for gs in X_train
        for g in x.split(',')
        if 'pf' in g.strip().lower()
    ])
    training_ipr = set([
        g.strip() for gs in X_train
        for g in x.split(',')
        if 'ipr' in g.strip().lower()
    ])

    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit(y_train)
    y_train = mlb.transform(y_train)

    # Make the estimators and BR classifier
    if retrain or not os.path.isfile(classifier_path):
        print("Making classifier...")
        param_distribution = {
            'C': np.arange(0.01, 10.01, step=0.01),
            'penalty': ['l1', 'l2']
        }
        random_cv = RandomizedSearchCV(
            cv=n_splits,
            n_iter=rcv_iter,
            n_jobs=n_jobs,
            param_distributions=param_distribution,
            estimator=make_classifier(model),
            scoring=make_scorer(f1_score, greater_is_better=True)
        )
        estimators = [
            Pipeline(
                [('vectorizer', CountVectorizer(binary=False)),
                 ('clf', clone(random_cv))]
            )
            for l in labels
        ]
        clf = BinaryRelevance(estimators, n_jobs=1)

        # Fit the complete training data and make predictions.
        print("Fitting data...")
        clf.fit(X_train, y_train)
        joblib.dump(clf, classifier_path)

    print("Making predictions...")
    clf = joblib.load(classifier_path)
    predictions = clf.predict_proba(X_test)

    # Write the predictions to a tsv file
    print("Writing results to file...")
    header = "{p1}\t{p2}\t{g1}\t{g2}\t{classes}\tsum\n".format(
        p1=P1, p2=P2, g1=G1, g2=G2, classes='\t'.join(sorted(mlb.classes_))
    )
    out_file.write(header)
    acc = annotation_ex.accession_vocabulary[UniProt.accession_column()]
    genes = annotation_ex.accession_vocabulary[UniProt.data_types().GENE.value]
    accession_gene_map = {a: g for (a, g) in zip(acc, genes)}
    for (s, t), p_vec in zip(X_test_ppis, predictions):
        p_vec = [p for _, p in sorted(zip(mlb.classes_, p_vec))]
        g1 = accession_gene_map.get(s, ['-'])[0] or '-'
        g2 = accession_gene_map.get(t, ['-'])[0] or '-'

        # Compute the usability of each of the annotation sets
        annots = annotation_ex.transform([s, t])
        go = set([
            g.strip() for gs in annots
            for g in x.split(',')
            if 'go' in g.strip().lower()
        ])
        pf = set([
            g.strip() for gs in annots
            for g in x.split(',')
            if 'pf' in g.strip().lower()
        ])
        ipr = set([
            g.strip() for gs in annots
            for g in x.split(',')
            if 'ipr' in g.strip().lower()
        ])
        usability_go = (go & training_go) / (go | training_go)
        usability_pf = (pf & training_pfam) / (pf | training_pfam)
        usability_ipr = (ipr & training_ipr) / (ipr | training_ipr)

        sum_pr = sum(p_vec)
        line = "{s}\t{t}\t{g1}\t{g2}\t{classes}\t{sum_pr}\t{usability_go}" \
               "\t{usability_pf}\t{usability_ipr}\n".format(
                s=s, t=t, g1=g1, g2=g2, sum_pr=sum_pr,
                classes='\t'.join(['%.4f' % p for p in p_vec]),
                usability_go=usability_go,
                usability_pf=usability_pf,
                usability_ipr=usability_ipr)
        out_file.write(line)
    out_file.close()
