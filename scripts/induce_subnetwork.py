#!/usr/bin/env python

"""
This script induces a subnetwork from the interactome predictions which
correspond to all edges labelled with `label` over the specified `threshold`
probability.

Usage:
  predict_interactome.py [--input=FILE] [--label=L] [--threshold=T] [--directory=DIR]
  predict_interactome.py -h | --help

Options:
  -h --help         Show this screen.
  --label=L         The label to induce the subnetwork from.
  --input=FILE      Input file [default: predictions.tsv]
  --directory=DIR   Input/Output directory [default: ./results/]
  --threhsold=T     Include all edges with a label probability over this
                    number [default: 0.5]
"""

from docopt import docopt

from pyPPI.base import parse_args
from pyPPI.network_analysis import InteractionNetwork

if __name__ == '__main__':
    args = parse_args(docopt(__doc__))
    label = args['label']
    threshold = args['threshold']
    network = InteractionNetwork(args['input'], args['directory'])
    network.induce_subnetwork_from_label(label, threshold)

