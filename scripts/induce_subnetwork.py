#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.
"""

from pyPPI.network_analysis import InteractionNetwork

if __name__ == '__main__':
    label = 'myristoylation'
    threshold = 0.5
    network = InteractionNetwork('./results/predictions.tsv')
    network.induce_subnetwork_from_label(label, threshold, output=True)

