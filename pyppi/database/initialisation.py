"""
The two methods `init_protein_database` and `init_interaction_database`
are two utility functions to set up the local database tables with the data
from the uniprot dumps and with interactions from the parsed networks.
"""

import logging

from Bio import SwissProt
from joblib import Parallel, delayed
from sqlalchemy.exc import IntegrityError

from ..data import uniprot_sprot, uniprot_trembl
from ..data_mining.features import compute_interaction_features
from ..data_mining.uniprot import parse_record_into_protein

from . import begin_transaction
from .models import Interaction
from .exceptions import ObjectAlreadyExists, ObjectNotFound

logger = logging.getLogger("pyppi")


def init_protein_table(record_handle=None, db_path=None):
    """
    Create entries for each uniprot accession in `accessions` matching
    the taxonomy id if `taxon_id` is not `None`.
    """
    if not record_handle:
        records = list(SwissProt.parse(uniprot_sprot())) + \
            list(SwissProt.parse(uniprot_trembl()))
    else:
        records = list(SwissProt.parse(record_handle))
    proteins = [parse_record_into_protein(r) for r in records]
    with begin_transaction(db_path=db_path) as session:
        for protein in proteins:
            protein.save(session, commit=True)
    return


def add_interactions_to_database(session, interactions, match_taxon_id=9606,
                                 n_jobs=1, verbose=False):
    """
    Create entries for each interaction tuple in `interactions` matching
    the taxonomy id if `taxon_is` is not `None`.
    """
    possible_exceptions = (
        ValueError, TypeError, AttributeError,
        ObjectAlreadyExists, ObjectNotFound, IntegrityError
    )
    valid = []
    entries = []
    invalid = []
    features = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(compute_interaction_features)(source, target)
        for (source, target, _, _, _, _) in interactions
    )

    for (source, target, is_holdout, is_training, is_interactome, label), features in \
            zip(interactions, features):
        entry = Interaction(
            source=source.id, target=target.id, is_holdout=is_holdout,
            is_training=is_training, is_interactome=is_interactome,
            label=label, **features
        )
        entries.append(entry)

    for (source, target, _, _, _, _), entry in zip(interactions, entries):
        try:
            if match_taxon_id is not None and \
                    source.taxon_id != match_taxon_id and \
                    target.taxon_id != match_taxon_id:
                raise ValueError(
                    "Non-matching taxonomy id {}. Expected {}.".format(
                        source.taxon_id, match_taxon_id)
                )

            entry.save(session, commit=True)
            valid.append(entry)

        except possible_exceptions as error:
            invalid.append((entry, str(error)))
            if verbose:
                logger.warning(
                    "Couldn't create interaction ({}, {}). Reason: {}.".format(
                        source.uniprot_id, target.uniprot_id, str(error)
                    )
                )

    return valid, invalid
