"""
This module contains exports the two main functions required to make 
predictions on a list of UniProt edges or :class:`Interaction` instances,
and to parse an edge list of UniProt accessions into 
:class:`Interaction` instances. There are also submodules containing
utility functions for plotting, loading datasets, models and so on.
"""

import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sqlalchemy.orm import scoped_session

from ..base.constants import SOURCE, TARGET
from ..base.io import load_classifier
from ..base.utilities import remove_duplicates

from ..database import db_session
from ..database.exceptions import ObjectAlreadyExists
from ..database.models import Protein, Interaction
from ..database.utilities import (
    create_interaction, get_upid_to_protein_map,
    get_source_taget_to_interactions_map
)
from ..data_mining.features import compute_interaction_features
from ..data_mining.uniprot import (
    parse_record_into_protein, parallel_download,
    recent_accession
)

from .utilities import VALID_SELECTION, interactions_to_Xy_format

from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


logger = logging.getLogger("pyppi")


__all__ = [
    "plotting",
    "utilities",
    "get_or_create_interactions",
    "classify_interactions"
]


def _check_classifier_and_selection(classifier=None, selection=None):
    mlb = None
    if classifier is None:
        try:
            clf, selection, mlb = load_classifier()
        except IOError:
            logger.exception(
                "Default classifier not found. Run build_data.py script first."
            )
            raise
    else:
        clf = classifier
        if selection is None or not selection:
            raise ValueError(
                "If supplying your own classifier, please specify a list "
                "of the feature databases it was trained on. "
                "Choose from {}.".format(', '.join(VALID_SELECTION))
            )

    for val in selection:
        if val not in VALID_SELECTION:
            raise ValueError(
                "Invalid selection '{}'. Please select from: {}.".format(
                    val, ', '.join(VALID_SELECTION)
                )
            )

    return clf, selection, mlb


def _update_missing_protein_map(ppis, session, verbose=False, n_jobs=1,
                                taxon_id=9606):
    to_download = set()
    uniprot_ids = [upid for tup in ppis for upid in tup if upid is not None]

    # Keep taxon_id as None. Why? Someone has passed in a mouse protein
    # by accident. The protein map will return a None entry because
    # it doesn't have the same taxon_id. The next step will try to download
    # the record for the mouse protein thinking it doesn't exist in the
    # database. Keeping the taxon_id None will stop needless downloads.
    # Instead just do a filter step afterwards to remove non-matching ids.
    protein_map = get_upid_to_protein_map(uniprot_ids, taxon_id=None)
    items = list(protein_map.items())
    for (upid, protein) in items:
        if protein is None:
            to_download.add(upid)

    # Download missing records (matching taxon_id if supplied), and create
    # new protein instances if necessary. Two things to fix later on:
    #
    #   (1) Records might be downloaded that are actually more recent than
    #       what's in the database. For now it's ok to ignore the newer one.
    #   (2) Records may be downloaded that are already. in the database
    #       (which *should* be caught by elif statement no.2).
    #
    # So really, there's only one thing to fix. TODO: Fix (1).
    records = parallel_download(
        to_download, n_jobs=n_jobs, verbose=verbose, taxon_id=taxon_id
    )

    assert len(records) == len(to_download)
    for uniprot, record in zip(to_download, records):
        if record is None:
            protein_map[uniprot] = None
            continue
        # Check to see if the record downloaded matches a database
        # entry with a different UniProt ID. Can happen if the
        # accession provided are outdated.
        most_recent = recent_accession(record)
        existing_entry = Protein.get_by_uniprot_id(most_recent)
        if existing_entry is not None:
            protein_map[uniprot] = existing_entry
            if verbose:
                logger.info(
                    "UniProt '{}' already exists in the database "
                    "as '{}'".format(uniprot, most_recent)
                )
        else:
            protein = parse_record_into_protein(record)
            assert uniprot == protein.uniprot_id
            protein.save(session, commit=False)
            protein_map[uniprot] = protein

    # Check for any non-matching taxon_ids
    for upid, protein in protein_map.items():
        if (protein is not None) and (taxon_id is not None) and \
                (protein.taxon_id != taxon_id):
            if verbose:
                logger.warning(
                    "{} has a non-matching taxonomy id {}. "
                    "Expected {}. Adding associated interactions to "
                    "invalid.".format(upid, protein.taxon_id, taxon_id)
                )
            protein_map[upid] = None

    try:
        session.commit()
        return protein_map
    except:
        session.rollback()
        raise


def _create_missing_interactions(ppis, protein_map, session, verbose=False,
                                 taxon_id=9606, n_jobs=1):
    valid = []
    invalid = []
    id_ppis = []
    id_protein_map = {p.id: p for _, p in protein_map.items() if p is not None}

    for a, b in ppis:
        if a not in protein_map:
            invalid.append((a, b))
        elif b not in protein_map:
            invalid.append((a, b))
        elif protein_map[a] is None:
            invalid.append((a, b))
        elif protein_map[b] is None:
            invalid.append((a, b))
        elif (taxon_id is not None) and (protein_map[a].taxon_id != taxon_id):
            if verbose:
                logger.warning(
                    "{} in ({},{}) has a non-matching taxonomy {}. "
                    "Expected {}. Adding to invalid.".format(
                        a, a, b, protein_map[a].taxon_id, taxon_id
                    ))
            invalid.append((a, b))
        elif (taxon_id is not None) and (protein_map[b].taxon_id != taxon_id):
            if verbose:
                logger.warning(
                    "{} in ({},{}) has a non-matching taxonomy {}. "
                    "Expected {}. Adding to invalid.".format(
                        b, a, b, protein_map[b].taxon_id, taxon_id
                    ))
            invalid.append((a, b))
        else:
            id_ppis.append((protein_map.get(a).id, protein_map.get(b).id))

    if id_ppis:
        interactions = get_source_taget_to_interactions_map(id_ppis, taxon_id)
        new_interactions = {k: v for k, v in interactions.items() if v is None}
        feature_map = {}
        # The new interaction will need to have their features computed
        # Do this in parallel to speed things up.
        if new_interactions:
            if verbose:
                logger.info("Computing features for new interactions.")
            features = Parallel(n_jobs=n_jobs)(
                delayed(compute_interaction_features)(
                    id_protein_map[source], id_protein_map[target]
                )
                for (source, target) in new_interactions
            )
            for (source, target), features in zip(new_interactions, features):
                feature_map[(source, target)] = features

        for (a, b), instance in interactions.items():
            if instance is None:
                source = id_protein_map[a]
                target = id_protein_map[b]
                class_kwargs = feature_map[(a, b)]
                class_kwargs['is_training'] = False
                class_kwargs['is_interactome'] = False
                class_kwargs['is_holdout'] = False
                interaction = create_interaction(
                    source, target, labels=None,
                    verbose=verbose, **class_kwargs
                )
                if verbose:
                    logger.info("Creating new interaction ({},{})".format(
                        source.uniprot_id, target.uniprot_id
                    ))
                valid.append(interaction)
            else:
                valid.append(instance)

        try:
            session.add_all(valid)
            session.commit()
        except:
            session.rollback()
            raise

    return valid, invalid


def get_or_create_interactions(ppis, session=None, taxon_id=9606,
                               verbose=False, n_jobs=1):
    """Parse an iterable of interactions in valid and invalid.

    Parse an iterable of either :py:class:`Interaction` instances or
    edge `tuple` of UniProt string identifiers into :py:class:`Interaction`
    instances that are valid. Invalid interactions are those that do not match
    `taxon_id` if provided or those for which UniProt entries cannot be
    created or downloaded. This method will treat (A, B) the same as
    (B, A) since feature wise, they are identical. New interactions will be
    constructed with (A, B) sorted in increasing alpha-numeric order. 

    **Example:** (Q1234, P1234) will create an :py:class:`Interaction` instance
    with `source` being P1234 and `target` being Q1234.


    Parameters
    ----------
    ppis : `list` or :py:class:`pd.DataFrame`
        List of uppercase UniProt identifiers `tuples`, list of 
        :py:class:`Interaction` objects or a :py:class:`pd.DataFrame`
        with columns `source` and `target` containing uppercase UniProt 
        identifiers.

    taxon_id : `int`, Default: 9606
        A `UniProt` taxonomy identifier to indicate which organism your
        interactions are from. Interactions supplied with proteins not matching
        this code will be treated as invalid. If None, the taxonomy identifier
        will be ignored. it is strongly recommended that you only make
        predictions on interactions with the same identifier as those
        used during training.

    verbose : `boolean`, Default: True
        Logs intermediate messages to your console. Useful for debugging.

    session : :py:func: `scoped_session`
        A database session object that connects to a SQLite3 database file.
        Only supply this if you have experience with `SQLAlchemy` and you
        know what you are doing. Leave as `None` to use this package's
        database.

    n_jobs : `int`, Default: 1
        Number of processes to use when downloading new records and
        computing features for new interactions. This can provide a nice speed 
        boost for large input.

    Returns
    -------
    `tuple` : (`list`, `list`, `dict`)
        A tuple where the first element a list of :py:class:`Interaction`
        instances for each valid and unique interaction in `ppis`. 
        The second element is a list of invalid interactions. The last 
        element is a dictionary from input UniProt identifiers the most recent
        UniProt identifiers. A change may occur when input proteins are mapped 
        to newer accessions by the UniProt mapping service. Provided for your
        own record keeping.
    """
    if session is None:
        session = db_session

    if isinstance(ppis, pd.DataFrame):
        ppis = list(zip(ppis[SOURCE], ppis[TARGET]))
    else:
        ppis = list(ppis)

    if isinstance(ppis[0], Interaction):
        invalid = []
        interactions = []
        for entry in ppis:
            if not isinstance(entry, Interaction):
                invalid.append(entry)
            elif entry.taxon_id != taxon_id:
                invalid.append(entry)
            else:
                interactions.append(entry)
        return interactions, invalid, {}

    elif isinstance(ppis[0], (tuple, list)):
        invalid = []
        for ppi in ppis:
            if not isinstance(ppi, (tuple, list)):
                invalid.append(ppi)
                if verbose:
                    logger.warning(
                        "Invalid: '{}' is not list or tuple.".format(ppi)
                    )
            elif not len(ppi) == 2:
                invalid.append(ppi)
                if verbose:
                    logger.warning(
                        "Invalid: '{}' is not length 2.".format(ppi)
                    )
            elif any(not isinstance(elem, str) for elem in ppi):
                invalid.append(ppi)
                if verbose:
                    logger.warning(
                        "Invalid: '{}' has non string members.".format(ppi)
                    )

        for invalid_ppi in invalid:
            ppis.remove(invalid_ppi)

        unique_ppis = remove_duplicates(
            (tuple(sorted([str(a), str(b)])) for (a, b) in ppis)
        )
        if len(unique_ppis) == 0:
            return [], invalid, {}  # They're all invalid

        # Create new proteins not in the database and add them to the map.
        # Using a map because it's quicker than asking the database for each
        # instance. Could also do this with raw SQL to avoid having the whole
        # database in memory, but this is fine for now.
        protein_map = _update_missing_protein_map(
            ppis=unique_ppis, session=session, n_jobs=n_jobs,
            verbose=verbose, taxon_id=taxon_id
        )
        # Parse the interactions creating missing ones where required.
        interactions, invalid = _create_missing_interactions(
            ppis=unique_ppis, protein_map=protein_map, n_jobs=n_jobs,
            session=session, verbose=verbose, taxon_id=taxon_id
        )
        # _update_missing_protein_map and _create_missing_interactions
        # will add new instances to the current session. Commit these if
        # requested making sure to rollback changes if there's an error.
        try:
            session.commit()
            upid_new_upid = {
                k: None if v is None else v.uniprot_id
                for k, v in protein_map.items()
            }
            return interactions, invalid, upid_new_upid
        except:
            session.rollback()
            raise
    else:
        t = type(ppis[0]).__name__
        raise TypeError("Unexpected type %s at index 0 in ppis." % t)


def classify_interactions(ppis, proba=True, classifier=None, selection=None,
                          taxon_id=9606, verbose=True, session=None,
                          n_jobs=1):
    """Predict the labels of a list of interactions.

    Parameters
    ----------
    ppis : `list` or :py:class:`pd.DataFrame`
        List of uppercase UniProt identifiers `tuples`, list of 
        :py:class:`Interaction` objects or a :py:class:`pd.DataFrame`
        with columns `source` and `target` containing uppercase UniProt 
        identifiers.

    proba : `boolean`, default: `True`
        If true, predict label membership probabilities. Otherwise make
        binary predictions.

    classifier : `object`, Optional.
        Classifier object used to make predictions. If None, loads the default
        classifier if it exists throwing an `IOError` if it cannot be found.
        The classifier must be a either a :py:class:`Pipeline` where the 
        first step subclasses :py:class:`VectorizerMixin`, or be a meta
        classifier such as a :py:class:`OneVsRestClassifier` where the base
        estimator is a :py:class:`Pipeline` as before.

    selection : `list`, optional
        list of strings indicating feature databases the classifier has been
        trained on. Must be supplied if a custom classifier is supplied.
        If the wrong selection is supplied for your classifier, un-reliable
        predictions will be returned.

    taxon_id : `int`, Default: 9606
        A `UniProt` taxonomy identifier to indicate which organism your
        interactions are from. Interactions supplied with proteins not matching
        this code will be treated as invalid. If None, the taxonomy identifier
        will be ignored. it is strongly recommended that you only make
        predictions on interactions with the same identifier as those
        used during training.

    verbose : `boolean`, Default: True
        Logs intermediate messages to your console. Useful for debugging.

    session : :py:func: `scoped_session`
        A database session object that connects to a SQLite3 database file.
        Only supply this if you have experience with `SQLAlchemy` and you
        know what you are doing. Leave as `None` to use this package's
        database.

    n_jobs : `int`, Default: 1
        Number of processes to use when downloading new records and
        computing features for new interactions. This can provide a nice speed 
        boost for large input.

    Returns
    -------
    `tuple`
        Tuple of array-like (n_ppis, n_labels), `list`, `dict`, `list` or None
        A tuple where the first element if a numpy array of predictions
        for each valid and unique interaction in `ppis`. Second element
        is a list of invalid PPIs. The third element is a dictionary from 
        input UniProt identifiers the most recent UniProt identifiers. A 
        change may occur when input proteins are mapped  to newer accessions 
        by the UniProt mapping service. Provided for your own record keeping. 
        The final element is the ordering of labels of predictions if the 
        default classifier is used, otherwise it is `None`.
    """
    clf, selection, mlb = _check_classifier_and_selection(
        classifier, selection)
    valid, invalid, mapping = get_or_create_interactions(
        ppis, session, taxon_id, verbose, n_jobs
    )
    if mlb is not None:
        labels = mlb.classes
    else:
        labels = None

    # Make predictions with the saved/supplied model.
    if len(valid):
        X, _ = interactions_to_Xy_format(valid, selection)
        if proba:
            return clf.predict_proba(X), invalid, mapping, labels
        else:
            return clf.predict(X), invalid, mapping, labels
    else:
        return [], invalid, {}, labels
