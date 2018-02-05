"""
Collection of utility operations that don't go anywhere else.
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import islice
import math

from ..data import load_ptm_labels
from ..models import supported_estimators

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


def su_make_dir(path, mode=0o777):
    if not path or os.path.exists(path):
        print("{} already exists...".format(path))
    else:
        os.mkdir(path)
        os.chmod(path, mode)


def pretty_print_dict(dictionary, n_tabs=1, fp=None):
    for k in sorted(dictionary.keys()):
        if fp:
            fp.write('\t' * n_tabs + '{}:\t{}\n'.format(k, dictionary[k]))
        else:
            print('\t' * n_tabs + '{}:\t{}'.format(k, dictionary[k]))


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


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def chunk_list(ls, n):
    """
    Split a list into n sublists.
    """
    if n < 1:
        raise ValueError("n must be greater than 0.")
    else:
        consumed = 0
        elements_per_sublist = int(math.ceil(len(ls) / n))
        while consumed < len(ls):
            sublist = take(elements_per_sublist, ls[consumed:])
            consumed += len(sublist)
            yield sublist


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
        return None


def parse_args(docopt_args):
    parsed = {}
    from ..database.models import Interaction

    # String processing
    if query_doctop_dict(docopt_args, '--directory'):
        if os.path.isdir(docopt_args['--directory']):
            parsed['directory'] = docopt_args['--directory']
        else:
            parsed['directory'] = './'

    if query_doctop_dict(docopt_args, '--label'):
        if docopt_args['--label'] not in load_ptm_labels():
            sys.stdout.write("Invalid label selection. Select one of: {}".format(
                ' ,'.join(load_ptm_labels())
            ))
            sys.exit(0)
        parsed['label'] = docopt_args['--label']

    if query_doctop_dict(docopt_args, '--backend'):
        backend = query_doctop_dict(docopt_args, '--backend')
        if backend not in ['threading', 'multiprocessing']:
            sys.stdout.write(
                "Backend must be one of 'threading' or 'multiprocessing'."
            )
            sys.exit(0)
        else:
            parsed['backend'] = backend

    # Selection parsing
    selection = []
    has_features = False
    if query_doctop_dict(docopt_args, '--interpro'):
        selection.append(Interaction.columns().INTERPRO.value)
        has_features = True
    if query_doctop_dict(docopt_args, '--pfam'):
        selection.append(Interaction.columns().PFAM.value)
        has_features = True
    if query_doctop_dict(docopt_args, '--mf'):
        has_features = True
        if query_doctop_dict(docopt_args, '--induce'):
            selection.append(Interaction.columns().ULCA_GO_MF.value)
        else:
            selection.append(Interaction.columns().GO_MF.value)
    if query_doctop_dict(docopt_args, '--cc'):
        has_features = True
        if query_doctop_dict(docopt_args, '--induce'):
            selection.append(Interaction.columns().ULCA_GO_CC.value)
        else:
            selection.append(Interaction.columns().GO_CC.value)
    if query_doctop_dict(docopt_args, '--bp'):
        has_features = True
        if query_doctop_dict(docopt_args, '--induce'):
            selection.append(Interaction.columns().ULCA_GO_BP.value)
        else:
            selection.append(Interaction.columns().GO_BP.value)
    if has_features and len(selection) == 0:
        sys.stdout.write("Must have at least one feature.")
        sys.exit(0)
    if has_features:
        parsed['selection'] = selection

    # bool parsing
    booleans = [
        '--abs', '--induce', '--verbose', '--retrain',
        '--binary', '--clear_cache', '--cost_sensitive'
    ]
    for arg in booleans:
        if query_doctop_dict(docopt_args, arg) is not None:
            parsed[arg[2:]] = query_doctop_dict(docopt_args, arg)

    # Numeric parsing
    if query_doctop_dict(docopt_args, '--n_jobs') is not None:
        n_jobs = int(query_doctop_dict(docopt_args, '--n_jobs')) or 1
        parsed['n_jobs'] = n_jobs
    if query_doctop_dict(docopt_args, '--n_splits') is not None:
        n_splits = int(query_doctop_dict(docopt_args, '--n_splits')) or 5
        parsed['n_splits'] = n_splits
    if query_doctop_dict(docopt_args, '--n_iterations') is not None:
        n_iterations = int(query_doctop_dict(
            docopt_args, '--n_iterations')) or 5
        parsed['n_iterations'] = n_iterations
    if query_doctop_dict(docopt_args, '--h_iterations') is not None:
        h_iterations = int(query_doctop_dict(
            docopt_args, '--h_iterations')) or 60
        parsed['h_iterations'] = h_iterations
    if query_doctop_dict(docopt_args, '--top') is not None:
        top_features = int(query_doctop_dict(docopt_args, '--top')) or 25
        parsed['top'] = top_features
    if query_doctop_dict(docopt_args, '--threshold') is not None:
        threshold = float(query_doctop_dict(docopt_args, '--threshold')) or 0.5
        parsed['threshold'] = threshold

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
            sys.stdout.write(e)
            sys.exit(0)

    if query_doctop_dict(docopt_args, '--input') is not None:
        if query_doctop_dict(docopt_args, '--input') == 'None':
            parsed['input'] = None
        elif query_doctop_dict(docopt_args, '--input'):
            try:
                fp = open(docopt_args['--input'], 'r')
                fp.close()
                parsed['input'] = docopt_args['--input']
            except IOError as e:
                sys.stdout.write(e)
                sys.exit(0)

    # Model parsing
    model = query_doctop_dict(docopt_args, '--model')
    if model is not None:
        if (model not in supported_estimators()):
            sys.stdout.write(
                'Classifier not supported. Please choose one of: {}'.format(
                    '\t\n'.join(supported_estimators().keys())
                ))
            sys.exit(0)
        else:
            parsed['model'] = model
    return parsed


def delete_cache():
    from ..data_mining.kegg import reset_kegg, reset_uniprot
    reset_kegg()
    reset_uniprot()
