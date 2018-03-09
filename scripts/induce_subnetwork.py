
"""
This script induces a subnetwork from the interactome predictions which
correspond to all edges labelled with `label` over the specified `threshold`
probability.

Usage:
  induce_subnetwork.py [--input=FILE] [--label=L] [--threshold=T] 
                       [--pathway=FILE] [--gene_names] [--directory=DIR]
  induce_subnetwork.py -h | --help

Options:
  -h --help         Show this screen.
  --label=L         The label to induce the subnetwork from. [default: None]
  --pathway=FILE    List of uniprot/gene ids to induce from. [default: None]
  --gene_names      Supply this if your pathway file contains gene names.
  --input=FILE      Input file [default: predictions.tsv]
  --directory=DIR   Output directory [default: ./results/networks/]
  --threhsold=T     Include all edges with a label probability over this
                    number [default: 0.5]
"""

import pandas as pd
import logging

from docopt import docopt
from pyppi.base.log import create_logger
from pyppi.base.arg_parsing import parse_args
from pyppi.network_analysis import InteractionNetwork


logger = create_logger("scripts", logging.INFO)


if __name__ == "__main__":
    args = parse_args(docopt(__doc__))
    interaction_path = args['input']
    directory = args['directory']
    threshold = args['threshold']
    label = args['label']
    pathway = args['pathway']
    use_genes = args['gene_names']

    if interaction_path is None:
        raise ValueError(
            "Please supply a prediction file. "
            "Run predict_ppis.py to generate one."
        )

    if label and pathway:
        raise ValueError(
            "You cannot supply both a pathway and label at the same time."
        )

    if label or pathway:
        logger.info("Reading network from '{}'".format(interaction_path))
        interactions = pd.read_csv(interaction_path, sep='\t', na_values=None)
        network = InteractionNetwork(
            interactions=interactions,
            output_dir=directory,
            sep='\t'
        )
        if label:
            logger.info("Inducing from label '{}'".format(label))
            logger.info("Using threshold: {}".format(threshold))
            network.induce_subnetwork_from_label(label, threshold)

        elif pathway:
            logger.info(
                "Inducing from pathway [{}]".format(', '.join(pathway))
            )
            logger.info("Using genes: {}".format(use_genes))
            logger.info("Using threshold: {}".format(threshold))
            network.induce_subnetwork_from_pathway(
                pathway, threshold=threshold, genes=use_genes
            )
    else:
        raise ValueError(
            "Please supply either a label or pathway to induce from."
        )
