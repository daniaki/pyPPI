#!/usr/bin/env python

"""
Collection of utility operations that don't go anywhere else.
"""

import os
import numpy as np
import pandas as pd
import argparse


P1 = 'protein_a'
P2 = 'protein_b'
G1 = 'gene_a'
G2 = 'gene_b'
SOURCE = 'source'
TARGET = 'target'
LABEL = 'label'


def su_make_dir(path, mode=0o777):
    if not path or os.path.exists(path):
        print("{} already exists...".format(path))
    else:
        os.mkdir(path)
        os.chmod(path, mode)


def pretty_print_dict(dictionary, n_tabs=0):
    for k in sorted(dictionary.keys()):
        print('\t'*n_tabs + '{}:\t{}'.format(k, dictionary[k]))


def create_seeds(size, random_state):
    np.random.seed(random_state)
    ii32 = np.iinfo(np.int32)
    max_int = ii32.max
    seeds = np.random.random_integers(low=0, high=max_int, size=size)
    return seeds


def validate_term(term):
    if 'go' in term.lower():
        term = term.replace(':', '')
        term = term[0:2] + ':' + term[2:]
        return term.upper()
    return term.upper()


def concat_dataframes(dfs, reset_index=False):
    """
    Concatenate a list of dataframes.
    """
    combined = pd.DataFrame()
    for df in dfs:
        combined = pd.concat([combined, df], ignore_index=True)
    if reset_index:
        combined.reset_index(drop=True, inplace=True)
    return combined


def chunk_list(ls, n):
    """Yield successive n-sized chunks from l."""
    if n == 0:
        return []
    if n == 1:
        return ls
    ranges = list(range(0, len(ls), int(np.ceil(len(ls)/n))))
    tup_ranges = []
    for i in range(len(ranges)-1):
        tup_ranges.append((ranges[i], ranges[i+1]))
    tup_ranges.append((ranges[i+1], len(ls) + 1))
    for (i, j) in tup_ranges:
        yield ls[i: j]


class PPI(object):
    """
    Simple class to contain some basic functionality to represent a PPI
    instance.
    """

    def __init__(self, p1, p2):
        self.__proteins = tuple(sorted((str(p1), str(p2))))
        self.__p1 = self.__proteins[0]
        self.__p2 = self.__proteins[1]

    @property
    def p1(self):
        return self.__p1

    @property
    def p2(self):
        return self.__p2

    @property
    def proteins(self):
        return self.__proteins

    def __repr__(self):
        return 'PPI({}, {})'.format(self.__p1, self.__p2)

    def __str__(self):
        return 'PPI({}, {})'.format(self.__p1, self.__p2)

    def __hash__(self):
        return hash(self.__proteins)

    def __reversed__(self):
        return PPI(self.__p2, self.__p1)

    def __contains__(self, item):
        return item in self.__proteins

    def __len__(self):
        return len(self.__proteins)

    def __eq__(self, other):
        return self.__proteins == other.proteins

    def __ne__(self, other):
        return not self.__proteins == other.proteins

    def __le__(self, other):
        return self.__p1 <= other.p1

    def __ge__(self, other):
        return self.__p1 >= other.p1

    def __lt__(self, other):
        return self.__p1 < other.p1

    def __gt__(self, other):
        return self.__p1 > other.p1

    def __iter__(self):
        return iter(self.__proteins)


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--method",
        help="Sklearn Binary Classifier",
        type=str, default='LogisticRegression'
    )
    parser.add_argument(
        "-gsf", "--grid_folds",
        help="Number of Cross Validation folds for Grid-Search.",
        type=int, default=5
    )
    parser.add_argument(
        "-it", "--iter",
        help="Number of bootstrap iterations to run.",
        type=int, default=5
    )
    parser.add_argument(
        "-f", "--folds",
        help="Number of Cross Validation folds to run in each bootstrap.",
        type=int, default=5
    )
    parser.add_argument(
        "-n", "--jobs",
        help="Number of processes to spawn.",
        type=int, default=1
    )
    parser.add_argument(
        "-i", "--induce",
        help="Use ULCA GO term induction.",
        action='store_true', default=False
    )
    parser.add_argument(
        "-v", "--vectoriser",
        help="Sklearn text vectoriser method to use.",
        type=str, default='CountVectorizer'
    )
    parser.add_argument(
        "-b", "--binary",
        help="Set binary in CountVectorizer to True",
        action='store_true', default=False
    )
    parser.add_argument(
        "-csl", "--balanced",
        help="Use cost-sensitive learning",
        action='store_true', default=False
    )
    parser.add_argument(
        "-pf", "--pfam",
        help="Use pfam terms in fetures.",
        action='store_true', default=False
    )
    parser.add_argument(
        "-ipr", "--interpro",
        help="Use interpro terms in fetures.",
        action='store_true', default=False
    )
    parser.add_argument(
        "-bp", "--biological_process",
        help="Use GO Biological Process in fetures.",
        action='store_true', default=False
    )
    parser.add_argument(
        "-mf", "--molecular_function",
        help="Use GO Molecular Function in fetures.",
        action='store_true', default=False
    )
    parser.add_argument(
        "-cc", "--cellular_component",
        help="Use GO Cellular Component in fetures.",
        action='store_true', default=False
    )
    parser.add_argument(
        "-of", "--output_file",
        help="Use pfam terms in fetures.",
        type=str, default='/scripts/results/predictions.tsv'
    )
    return parser
