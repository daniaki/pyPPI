#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.
"""

import argparse
import numpy as np
import pandas as pd

from ..models.binary_relevance import BinaryRelevance
from ..data_mining.features import AnnotationExtractor
from ..models import make_classifier, supported_estimators
from ..model_selection.scoring import MultilabelScorer, Statistics
from ..model_selection.sampling import IterativeStratifiedKFold
from ..data import load_training_network, load_testing_network

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import *
from sklearn.metrics import *


if __name__ == '__main__':
    development_df = load_training_network()
    testing_df = load_testing_network()

    # Get the features into X, and multilabel y indicator format

    # Make the estimators and BR classifier
    clf = BinaryRelevance()

    # Make the bootstrap and KFoldExperiments

    # Fit the data

    # Make the scoring functions

    # Evaluate performance

    # Put everything into a dataframe
