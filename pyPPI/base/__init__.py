#!/usr/bin/env python

"""
Collection of utility operations that don't go anywhere else.
"""

import os
import sys
import numpy as np
import pandas as pd

from ..data import load_ptm_labels
from ..data_mining.uniprot import UniProt
from ..models import supported_estimators

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


def pretty_print_dict(dictionary, n_tabs=0, fp=None):
    for k in sorted(dictionary.keys()):
        if fp:
            fp.write('\t'*n_tabs + '{}:\t{}\n'.format(k, dictionary[k]))
        else:
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


def query_doctop_dict(docopt_dict, key):
    if key in docopt_dict:
        return docopt_dict[key]
    else:
        return False


def parse_args(docopt_args):
    parsed = {}

    # String processing
    if query_doctop_dict(docopt_args, '--directory'):
        if os.path.isdir(docopt_args['--directory']):
            parsed['directory'] = docopt_args['--directory']
        else:
            parsed['directory'] = './'
    if query_doctop_dict(docopt_args, '--label'):
        if docopt_args['--label'] not in load_ptm_labels():
            print("Invalid label selection. Select one of: ".format(
                ' ,'.join(load_ptm_labels())
            ))
            sys.exit(0)
        parsed['label'] = docopt_args['--label']

    # Selection parsing
    selection = []
    if query_doctop_dict(docopt_args, '--interpro'):
        selection.append(UniProt.data_types().INTERPRO.value)
    if query_doctop_dict(docopt_args, '--pfam'):
        selection.append(UniProt.data_types().PFAM.value)
    if query_doctop_dict(docopt_args, '--mf'):
        selection.append(UniProt.data_types().GO_MF.value)
    if query_doctop_dict(docopt_args, '--cc'):
        selection.append(UniProt.data_types().GO_CC.value)
    if query_doctop_dict(docopt_args, '--bp'):
        selection.append(UniProt.data_types().GO_BP.value)
    if len(selection) == 0:
        print("Must have at least one feature.")
        sys.exit(0)
    parsed['selection'] = selection

    # bool parsing
    parsed['induce'] = query_doctop_dict(docopt_args, '--interpro')
    parsed['verbose'] = query_doctop_dict(docopt_args, '--verbose')
    parsed['use_cache'] = query_doctop_dict(docopt_args, '--use_cache')
    parsed['retrain'] = query_doctop_dict(docopt_args, '--retrain')
    parsed['binary'] = query_doctop_dict(docopt_args, '--binary')
    parsed['cost_sensitive'] = query_doctop_dict(
        docopt_args, '--cost_sensitive'
    )
    parsed['update_features'] = query_doctop_dict(
        docopt_args, '--update_features'
    )
    parsed['update_mapping'] = query_doctop_dict(
        docopt_args, '--update_mapping'
    )

    # Numeric parsing
    parsed['n_jobs'] = int(query_doctop_dict(docopt_args, '--n_jobs'))
    parsed['n_splits'] = int(query_doctop_dict(docopt_args, '--n_splits'))
    parsed['iterations'] = int(query_doctop_dict(docopt_args, '--iterations'))
    parsed['threshold'] = float(query_doctop_dict(docopt_args, '--threshold'))

    # Input/Output parsing
    if query_doctop_dict(docopt_args, '--output'):
        try:
            if query_doctop_dict(docopt_args, '--directory'):
                fp = open(
                    docopt_args['--directory'] + docopt_args['--output'], 'w')
            else:
                fp = open(docopt_args['--output'], 'w')
            fp.close()
            parsed['output'] = docopt_args['--output']
        except IOError as e:
            print(e)
            sys.exit(0)

    if query_doctop_dict(docopt_args, '--input') == 'None':
        parsed['input'] = docopt_args['--input']
    elif query_doctop_dict(docopt_args, '--input'):
        try:
            if query_doctop_dict(docopt_args, '--directory'):
                fp = open(
                    docopt_args['--directory'] + docopt_args['--input'], 'r')
            else:
                fp = open(docopt_args['--input'], 'r')
            fp.close()
            parsed['input'] = docopt_args['--input']
        except IOError as e:
            print(e)
            sys.exit(0)

    # Model parsing
    model = query_doctop_dict(docopt_args, '--model')
    if model and model not in supported_estimators():
        print('Classifier not supported. Please choose one of:'.format(
            '\t\n'.join(supported_estimators().keys())
        ))
        sys.exit(0)
    else:
        parsed['model'] = model

    return parsed