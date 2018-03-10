"""
The `arg_parsing` module exports the method `parse_args` which is a utility
function for parsing the arguments orginially parsed by :module:`docopt`.
This can be used in scripts to parse user input.
"""

import sys
import os
import joblib


from .io import load_classifier
from .file_paths import classifier_path
from ..models.utilities import supported_estimators
from ..database.models import Interaction

__all__ = [
    'parse_args'
]


VALID_SELECTION = (
    Interaction.columns().GO_MF.value,
    Interaction.columns().GO_CC.value,
    Interaction.columns().GO_BP.value,
    Interaction.columns().ULCA_GO_MF.value,
    Interaction.columns().ULCA_GO_CC.value,
    Interaction.columns().ULCA_GO_BP.value,
    Interaction.columns().INTERPRO.value,
    Interaction.columns().PFAM.value
)


def _query_doctop_dict(docopt_dict, key):
    """
    Check the :module:`doctopt` dictionary for a key. Returns None is not 
    found, otherwise the key's value.
    """
    if key in docopt_dict:
        return docopt_dict[key]
    else:
        return None


def parse_args(docopt_args, require_features=False, predict_script=False):
    """
    Function to parse args that may be passed into one of the supplied scripts.

    Parameters:
    ----------
    docopt_args : dict
        Args initially parsed by docopt

    Returns
    -------
    dict 
        Dictionary of parsed arguments.
    """

    parsed = {}

    # String processing
    if _query_doctop_dict(docopt_args, '--directory'):
        if os.path.isdir(docopt_args['--directory']):
            parsed['directory'] = docopt_args['--directory']
        else:
            parsed['directory'] = './'

    if _query_doctop_dict(docopt_args, '--label'):
        label = docopt_args['--label']
        if label in (None, 'None'):
            parsed['label'] = None
        else:
            parsed['label'] = docopt_args['--label']

    if _query_doctop_dict(docopt_args, '--backend'):
        backend = _query_doctop_dict(docopt_args, '--backend')
        if backend not in ['threading', 'multiprocessing']:
            sys.stdout.write(
                "Backend must be one of 'threading' or 'multiprocessing'."
            )
            sys.exit(0)
        else:
            parsed['backend'] = backend

    # Selection parsing
    selection = []
    if _query_doctop_dict(docopt_args, '--interpro'):
        selection.append(Interaction.columns().INTERPRO.value)
    if _query_doctop_dict(docopt_args, '--pfam'):
        selection.append(Interaction.columns().PFAM.value)
    if _query_doctop_dict(docopt_args, '--mf'):
        if _query_doctop_dict(docopt_args, '--induce'):
            selection.append(Interaction.columns().ULCA_GO_MF.value)
        else:
            selection.append(Interaction.columns().GO_MF.value)
    if _query_doctop_dict(docopt_args, '--cc'):
        if _query_doctop_dict(docopt_args, '--induce'):
            selection.append(Interaction.columns().ULCA_GO_CC.value)
        else:
            selection.append(Interaction.columns().GO_CC.value)
    if _query_doctop_dict(docopt_args, '--bp'):
        if _query_doctop_dict(docopt_args, '--induce'):
            selection.append(Interaction.columns().ULCA_GO_BP.value)
        else:
            selection.append(Interaction.columns().GO_BP.value)

    # bool parsing
    booleans = [
        '--abs', '--induce', '--verbose', '--retrain',
        '--binary', '--clear_cache', '--cost_sensitive',
        '--gene_names', '--chain'
    ]
    for arg in booleans:
        if _query_doctop_dict(docopt_args, arg) is not None:
            parsed[arg[2:]] = _query_doctop_dict(docopt_args, arg)

    # If no classifier exists, check that a valid selection has been supplied.
    # Alternatively if a previous classifier does exist, and the user has not
    # requested a retrain and the supplied feature selection does not match
    # the saved one, then raise and error and quit.
    if predict_script:
        if not os.path.isfile(classifier_path):
            require_features = True
            if require_features and len(selection) == 0:
                sys.stdout.write("Must have at least one feature.")
                sys.exit(0)
            parsed['retrain'] = True  # couldn't find a default classifier
            parsed['selection'] = selection

        elif os.path.isfile(classifier_path):
            _, trained_selection = load_classifier(classifier_path)
            if not parsed['retrain'] and len(selection) and \
                    sorted(trained_selection) != selection:
                sys.stdout.write(
                    "It seems the saved classifier was trained using '{}', "
                    "but you supplied '{}'. Use argument --retrain if you "
                    "would like to train a new classifier on "
                    "different features.".format(
                        trained_selection, selection
                    )
                )
                sys.exit(0)
    else:
        if require_features and not len(selection):
            sys.stdout.write("Must have at least one feature.")
            sys.exit(0)
        parsed['selection'] = selection

    # Numeric parsing
    if _query_doctop_dict(docopt_args, '--n_jobs') is not None:
        n_jobs = int(_query_doctop_dict(docopt_args, '--n_jobs')) or 1
        parsed['n_jobs'] = n_jobs
    if _query_doctop_dict(docopt_args, '--n_splits') is not None:
        n_splits = int(_query_doctop_dict(docopt_args, '--n_splits')) or 5
        parsed['n_splits'] = n_splits
    if _query_doctop_dict(docopt_args, '--n_iterations') is not None:
        n_iterations = int(_query_doctop_dict(
            docopt_args, '--n_iterations')) or 3
        parsed['n_iterations'] = n_iterations
    if _query_doctop_dict(docopt_args, '--h_iterations') is not None:
        h_iterations = int(_query_doctop_dict(
            docopt_args, '--h_iterations')) or 30
        parsed['h_iterations'] = h_iterations
    if _query_doctop_dict(docopt_args, '--threshold') is not None:
        threshold = float(_query_doctop_dict(docopt_args, '--threshold'))
        parsed['threshold'] = threshold

    # Input/Output parsing
    if _query_doctop_dict(docopt_args, '--output'):
        try:
            if _query_doctop_dict(docopt_args, '--directory'):
                path = docopt_args['--directory'] + docopt_args['--output']
                open(path, 'w').close()
                os.remove(path)
            else:
                path = docopt_args['--output']
                open(path, 'w').close()
                os.remove(path)
            parsed['output'] = docopt_args['--output']
        except IOError as e:
            sys.stdout.write(e)
            sys.exit(0)

    if _query_doctop_dict(docopt_args, '--input') is not None:
        if _query_doctop_dict(docopt_args, '--input') == 'None':
            parsed['input'] = None
        elif _query_doctop_dict(docopt_args, '--input'):
            try:
                fp = open(docopt_args['--input'], 'r')
                fp.close()
                parsed['input'] = docopt_args['--input']
            except IOError as e:
                sys.stdout.write(e)
                sys.exit(0)

    if _query_doctop_dict(docopt_args, '--classifier') is not None:
        if _query_doctop_dict(docopt_args, '--classifier') == 'None':
            parsed['classifier'] = None
        elif _query_doctop_dict(docopt_args, '--classifier'):
            try:
                clf, sel, mlb = load_classifier(docopt_args['--classifier'])
                if len(sel) == 0:
                    raise ValueError("Saved selection cannot be empty")
                elif not all([s in VALID_SELECTION for s in sel]):
                    raise ValueError(
                        "Invalid feature selection '{}'. Select from "
                        "'{}'".format(
                            sel, VALID_SELECTION
                        )
                    )
                parsed['classifier'] = (clf, sel, mlb)
            except (IOError, ValueError) as e:
                sys.stdout.write(e)
                sys.exit(0)

    # Model parsing
    model = _query_doctop_dict(docopt_args, '--model')
    if model is not None:
        if model == "paper":
            parsed["model"] = model
        elif (model not in supported_estimators()):
            sys.stdout.write(
                'Classifier not supported. Please choose one of: {}'.format(
                    '\t\n'.join(supported_estimators().keys())
                ))
            sys.exit(0)
        else:
            parsed['model'] = model

    if _query_doctop_dict(docopt_args, '--pathway') is not None:
        if _query_doctop_dict(docopt_args, '--pathway') == 'None':
            parsed['pathway'] = None
        elif _query_doctop_dict(docopt_args, '--pathway'):
            try:
                fp = open(docopt_args['--pathway'], 'r')
                pathway = []
                for line in fp:
                    pathway.append(line.strip().upper())
                fp.close()
                parsed['pathway'] = pathway
            except IOError as e:
                sys.stdout.write(e)
                sys.exit(0)

    return parsed
