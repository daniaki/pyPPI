"""
Collection of utility operations that don't go anywhere else.
"""

__all__ = [
    'is_null',
    'su_make_dir',
    'take',
    'remove_duplicates',
    'get_term_description',
    'rename',
    'chunk_list',
    'delete_cache'
]

import os
import logging
import numpy as np
from itertools import islice
import math

from .constants import (
    NULL_VALUES, SOURCE, TARGET, EXPERIMENT_TYPE, PUBMED, LABEL
)


logger = logging.getLogger("pyppi")


def is_null(value):
    """Check if a value is null according to `NULL_VALUES` in 
    :module:`.constants`. This includes numpy/pandas `NaN`, `None` values
    and some other values considered null by this software that are
    enountered during edgelist parsing.

    Returns
    -------
    bool
        True if the value is considered null.
    """
    return str(value).strip() in NULL_VALUES


def su_make_dir(path, mode=0o777):
    """Make a directory at the path with read and write permisions"""
    if not path or os.path.exists(path):
        logger.info("Found existing directory {}.".format(path))
    else:
        os.mkdir(path)
        os.chmod(path, mode)


def take(n, iterable):
    """Return first n items of the iterable as a list

    Returns
    -------
    list
        List of size `n`, or the length of `iterable` if `n` is larger
        than the input.
    """
    return list(islice(iterable, n))


def remove_duplicates(seq):
    """Remove duplicates from a sequence preserving order.

    Returns
    -------
    list
        Return a list without duplicate entries.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_term_description(term, go_dag, ipr_map, pfam_map):
    """Takes an `InterPro`, `Gene Ontology` or `Pfam` annotation
    and returns it's description.

    Returns
    -------
    str
        String description or None if one could not be found.
    """
    term = term.upper()
    if 'IPR' in term:
        return ipr_map[term]
    elif 'PF' in term:
        return pfam_map[term]
    elif "GO" in term:
        term = term.replace('GO', 'GO:').replace('::', ':')
        return go_dag[term].name
    return None


def rename(term):
    """
    Re-format feature terms after they've been formated by the vectorizer.

    Parameters:
    ----------
    term : str
        Mutilated term in string format.

    Returns
    -------
    str
        The normalised term.
    """
    term = term.upper()
    if 'IPR' in term:
        return term
    elif 'PF' in term:
        return term
    elif 'GO' in term:
        return term.replace("GO", 'GO:').replace('::', ':')
    else:
        return term


def chunk_list(ls, n):
    """
    Split a list into n sublists.

    Return
    ------
    generator
        Generator of `n` sublists.
    """
    if not ls:
        return []

    ls = list(ls)
    n_elems = len(ls)
    sublists_returned = 0
    consumed = 0
    if n < 1:
        raise ValueError("n must be greater than 0.")
    if n > n_elems:
        raise ValueError("n must <= than the length of the sequence.")
    else:
        while sublists_returned < n:
            elem_per_sublist = (n_elems - consumed) / (n - sublists_returned)
            if elem_per_sublist % 2 == 0:
                step = int(elem_per_sublist)
                for i in range(consumed, n_elems, step):
                    sublists_returned += 1
                    yield ls[i: i + step]
            else:
                elem_per_sublist = int(np.ceil(elem_per_sublist))
                sublist = take(elem_per_sublist, ls[consumed:])
                consumed += len(sublist)
                sublists_returned += 1
                yield sublist


def generate_interaction_tuples(df):
    """Takes a :class:`pandas.DataFrame` object with the columns 'source',
    'target', 'label', 'pubmed', 'experiment_type' and combines the rows
    into a generator of tuples.

    Parameters:
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe to tuple-ize.

    Returns
    -------
    generator
        Generator of tuples of the form (accession, accession, label, 
        pmid list, psimi list). Accessions and labels will be `None` if
        null values are enountered.
    """

    zipped = zip(
        df[SOURCE],
        df[TARGET],
        df[LABEL],
        df[PUBMED],
        df[EXPERIMENT_TYPE]
    )
    for (uniprot_a, uniprot_b, label, pmids, psimis) in zipped:
        if is_null(uniprot_a):
            uniprot_a = None
        if is_null(uniprot_b):
            uniprot_b = None
        if is_null(label):
            label = None
        if is_null(pmids):
            pmids = [None]
        else:
            pmids = pmids.split(',')
        if is_null(psimis) or pmids == [None]:
            psimis = [None]
        else:
            psimis_groups = psimis.split(',')
            psimis = [
                [(None if is_null(p) else p) for p in group.split('|')]
                for group in psimis_groups
            ]

        assert len(pmids) == len(psimis)
        yield (uniprot_a, uniprot_b, label, pmids, psimis)


def delete_cache():
    """Wrapper to delete kegg and uniprot caches."""
    from ..data_mining.kegg import reset_kegg, reset_uniprot
    reset_kegg()
    reset_uniprot()
