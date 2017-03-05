#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.
"""

import numpy as np

from pyPPI.base import make_arg_parser, P1, P2, G1, G2
from pyPPI.data import load_network_from_path, load_ptm_labels
from pyPPI.data import full_training_network_path
from pyPPI.data import interactome_network_path

from pyPPI.models.binary_relevance import BinaryRelevance
from pyPPI.models import make_classifier, supported_estimators

from pyPPI.data_mining.features import AnnotationExtractor
from pyPPI.data_mining.uniprot import UniProt, get_active_instance
from pyPPI.data_mining.tools import xy_from_interaction_frame

from pyPPI.network_analysis import P1, P2, G1, G2

from sklearn.base import clone
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer


if __name__ == '__main__':
    n_jobs = 2
    n_splits = 5
    rcv_iter = 1
    induce = True
    verbose = True
    use_feature_cache = True

    print("Loading data...")
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
    training = load_network_from_path(full_training_network_path)
    testing = load_network_from_path(interactome_network_path)

    # Get the features into X, and multilabel y indicator format
    print("Preparing training and testing data...")
    X_train_ppis, y_train = xy_from_interaction_frame(training)
    X_test_ppis, _ = xy_from_interaction_frame(testing)
    X_train = annotation_ex.transform(X_train_ppis)
    X_test = annotation_ex.transform(X_test_ppis)

    mlb = MultiLabelBinarizer(classes=labels)
    mlb.fit(y_train)
    y_train = mlb.transform(y_train)

    # Make the estimators and BR classifier
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

    # Fit the complete training data and make predictions.
    print("Fitting data...")
    clf.fit(X_train, y_train)

    print("Making predictions...")
    predictions = clf.predict_proba(X_test)

    # Write the predictions to a tsv file
    print("Writing results to file...")
    fp = open('./results/predictions.tsv', 'w')
    header = "{p1}\t{p2}\t{g1}\t{g2}\t{classes}\tsum\n".format(
        P1, P2, G1, G2, '\t'.join(sorted(mlb.classes_))
    )
    fp.write(header)
    acc = annotation_ex.accession_vocabulary[UniProt.accession_column()]
    genes = annotation_ex.accession_vocabulary[UniProt.data_types().GENE.value]
    accession_gene_map = {a: g for (a, g) in zip(acc, genes)}
    for (s, t), p_vec in zip(X_test_ppis, predictions):
        p_vec = [p for _, p in sorted(zip(mlb.classes_, p_vec))]
        g1 = accession_gene_map.get(s, ['-'])[0] or '-'
        g2 = accession_gene_map.get(t, ['-'])[0] or '-'
        sum_pr = sum(p_vec)
        line = "{s}\t{t}\t{g1}\t{g2}\t{classes}\t{sum_pr}\n".format(
            s=s, t=t, g1=g1, g2=g2, sum_pr=sum_pr,
            classes='\t'.join(['%.4f' % p for p in p_vec])
        )
        fp.write(line)
    fp.close()
    print("Done!")
