#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.
"""

import pandas as pd
from pyPPI.network_analysis import InteractionNetwork

if __name__ == '__main__':
    label = 'myristoylation'
    threshold = 0.5
    interactions = pd.read_csv('./results/predictions.tsv')
    network = InteractionNetwork(interactions)
    network.induce_subnetwork_from_label(label, threshold, output=True)

