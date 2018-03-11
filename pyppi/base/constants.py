"""Definitions of some commonly used variables throughout the application."""

__all__ = [
    'P1', 'P2', 'G1', 'G2',
    'SOURCE', 'TARGET', 'LABEL',
    'PUBMED', 'EXPERIMENT_TYPE', 'NULL_VALUES',
]

import numpy as np


P1 = 'protein_a'
P2 = 'protein_b'
G1 = 'gene_a'
G2 = 'gene_b'
SOURCE = 'source'
TARGET = 'target'
LABEL = 'label'
PUBMED = 'pubmed'
EXPERIMENT_TYPE = 'experiment_type'
NULL_VALUES = (
    '', 'None', 'NaN', 'none', 'nan', '-', 'unknown', None, ' ', np.NaN
)
MAX_SEED = 1000000
