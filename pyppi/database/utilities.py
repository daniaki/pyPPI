"""
This module contains a collection of functions that perform common tasks
related to the database.
"""
import os
import gzip
from Bio import SwissProt
import logging
from collections import OrderedDict

from sqlalchemy import and_, or_
from sqlalchemy.orm.query import Query

from ..base.constants import SOURCE, TARGET, LABEL, EXPERIMENT_TYPE, PUBMED
from ..base.utilities import is_null, remove_duplicates
from ..data_mining.features import compute_interaction_features
from ..data_mining.uniprot import serialise_record
from ..data_mining.psimi import parse_miobo_file

from . import db_session
from .models import Interaction, Psimi, Protein, Pubmed
from .exceptions import ObjectNotFound
from .validators import (
    validate_interaction_does_not_exist, validate_same_taxonid,
    validate_source_and_target, validate_joint_id
)

__all__ = [
    "uniprotid_entry_map",
    "accession_entry_map",
    "filter_matching_taxon_ids",
    "training_interactions",
    "holdout_interactions",
    "full_training_network",
    "interactome_interactions",
    "labels_from_interactions",
    "get_upid_to_protein_map",
    "get_source_taget_to_interactions_map",
    "create_interaction",
    "proteins_from_dat",
    'psimi_from_obo',
    'pmids_from_list',
    'psimis_from_list'
]

logger = logging.getLogger("pyppi")


def uniprotid_entry_map():
    """Creates a `dict` mapping from UniProt accession to it's
    :class:`Protein` for all instances in the database.

    Returns
    -------
    `dict`
        A dictionary mapping from UniProt accession to it's Protein instance.
    """
    proteins = Protein.query.all()
    protein_map = {p.uniprot_id: p for p in proteins}
    return protein_map


def accession_entry_map(klass):
    """Creates a `dict` mapping from accessions to the owning instance.
    for all instances in the database represented by `klass`.

    Parameters
    ----------
    klass : :class:`Pubmed` or :class:`Psimi`
        A class having the accession attribute.

    Returns
    -------
    dict[str, object]
        A dictionary mapping from accession to the associated instance.
    """
    if not hasattr(klass, 'accession'):
        raise AttributeError(
            "`{}` does not have the attribute `accession`." % klass.__name__
        )
    items = klass.query.all()
    mapping = {e.accession: e for e in items}
    return mapping


def filter_matching_taxon_ids(query_set, taxon_id=None):
    """Filters out all instances with a taxonomy id that does not match
    `taxon_id` if taxon_id is not None. Taxonomy id must be one supported by
    `UniProt`.

    Parameters
    ----------
    query_set : :class:`Query`
        A query instance from `sqlalchemy`

    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`

    Returns
    -------
    :class:`Query`
        Filtered query instance.
    """
    if taxon_id is not None:
        return query_set.filter_by(taxon_id=taxon_id)
    else:
        return query_set


def training_interactions(strict=False, taxon_id=None):
    """Return all :class:`Interaction` instances with `is_training` set
    to `True`.

    Parameters
    ----------
    strict : bool, optional, default: False
        If strict, fiters out instances that also have `is_holdout` set to
        True.

    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id. Ignored if `None`.

    Returns
    -------
    :class:`Query`
        An sqlalchemy :class:`Query` instances containing
        :class:`Interaction` instances.
    """
    qs = Interaction.query.filter_by(is_training=True)
    if strict:
        qs = qs.filter_by(is_holdout=False)
    return filter_matching_taxon_ids(qs, taxon_id)


def holdout_interactions(strict=False, taxon_id=None):
    """Return all :class:`Interaction` instances with `is_holdout` set
    to `True`.

    Parameters
    ----------
    strict : bool, optional, default: False
        If strict, fiters out instances that also have `is_training` set to
        True.

    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id. Ignored if `None`.

    Returns
    -------
    :class:`Query`
        An sqlalchemy :class:`Query` instances containing
        :class:`Interaction` instances.
    """
    qs = Interaction.query.filter_by(is_holdout=True)
    if strict:
        qs = qs.filter_by(is_training=False)
    return filter_matching_taxon_ids(qs, taxon_id)


def full_training_network(taxon_id=None):
    """Return all :class:`Interaction` instances with `is_holdout` and
    `is_training` set to `True`.

    Parameters
    ----------
    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id. Ignored if `None`.

    Returns
    -------
    :class:`Query`
        An sqlalchemy :class:`Query` instances containing
        :class:`Interaction` instances.
    """
    qs = Interaction.query.filter(
        or_(
            Interaction.is_holdout == True,
            Interaction.is_training == True
        )
    )
    return filter_matching_taxon_ids(qs, taxon_id)


def interactome_interactions(taxon_id=None):
    """Return all :class:`Interaction` instances with `is_interaction`
    set to `True`.

    Parameters
    ----------
    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id. Ignored if `None`.

    Returns
    -------
    :class:`Query`
        An sqlalchemy :class:`Query` instances containing
        :class:`Interaction` instances.
    """
    qs = Interaction.query.filter_by(is_interactome=True)
    return filter_matching_taxon_ids(qs, taxon_id)


def labels_from_interactions(interactions=None, taxon_id=None):
    """Return all labels from an iterable of :class:`Interaction` instances.
    By default, :func:`full_training_network` will be called to obtain
    all training and holdout instances.

    Parameters
    ----------
    interactions : iterable, optional.
        An iterable of :class:`Interaction` objects. If `None`,
        :func:`full_training_network` is called to obtain interactions.

    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id before collecting labels.
        Ignored if `None`.

    Returns
    -------
    `list`
        A list of sorted labels.
    """
    if interactions is None:
        interactions = full_training_network(taxon_id).all()
    labels = set()
    for interaction in interactions:
        labels |= set(interaction.labels_as_list)
    return list(sorted(labels))


def get_upid_to_protein_map(uniprot_ids, taxon_id=None):
    """Builds a `dict` mapping from the given UniProt accession strings
    to their instances stored in the database, if it exists.

    Parameters
    ----------
    uniprot_ids : `list`
        List of UniProt accession.

    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id before collecting labels.
        Ignored if `None`.

    Returns
    -------
    `OrderedDict`
        Mapping from UniProt accession to :class:`Protein` or None.
    """
    uniprot_ids = remove_duplicates(
        [upid for upid in uniprot_ids if upid is not None]
    )
    mapping = OrderedDict()

    if taxon_id is not None:
        matches = Protein.query.filter(
            and_(
                Protein.uniprot_id.in_(uniprot_ids),
                Protein.taxon_id == taxon_id
            )
        )
    else:
        matches = Protein.query.filter(
            Protein.uniprot_id.in_(uniprot_ids)
        )

    for p in matches.all():
        mapping[p.uniprot_id] = p
    for upid in uniprot_ids:
        if upid not in mapping:
            mapping[upid] = None
    return mapping


def get_source_taget_to_interactions_map(id_ppis, taxon_id=None):
    """Builds a `dict` mapping from a tuple of integer ids representing
    the :class:`Protein` source and target primary keys to the associtated
    :class:`Interaction` instance if it exists.

    Parameters
    ----------
    id_ppis : `list`
        List of `tuples` of integer ids representing the source and target
        :class:`Protein`. The ids should be the protein's integer primary key.

    taxon_id : int, optional
        An integer taxonomy id supported by `UniProt`. Filter out interactions
        with a non-matching taxonomy id before collecting labels.
        Ignored if `None`.

    Returns
    -------
    `OrderedDict`
        Mapping from (int, int) source, target tuple to  :class:`Interaction`
        or None.
    """
    id_ppis = remove_duplicates(id_ppis)
    sources = [s for s, _ in id_ppis]
    targets = [t for _, t in id_ppis]
    mapping = OrderedDict()

    if taxon_id is not None:
        matches = Interaction.query.filter(
            and_(
                Interaction.source_.in_(sources),
                Interaction.target_.in_(targets),
                Interaction.taxon_id == taxon_id
            )
        )
    else:
        matches = Interaction.query.filter(
            and_(
                Interaction.source_.in_(sources),
                Interaction.target_.in_(targets)
            )
        )

    matches = {(m.source, m.target): m for m in matches.all()}
    for (source, target) in id_ppis:
        mapping[(source, target)] = matches.get((source, target), None)
    return mapping


def create_interaction(source, target, labels=None, session=None,
                       save=False, commit=False, verbose=False, **kwargs):
    """Create an :class:`Interaction` instance with `source` and `target`
    as the interactors.

    Parameters
    ----------
    source : int, str or :class:`Protein`
        Source protein of the interaction. It can either be the int `id`
        of the protein, the instance itself or the `uniprot_id` of the protein.
        If creating many interactions, it will be much faster to pass
        in the protein instance itself to avoid constant database queries
        during validation.

    target : int, str or :class:`Protein`
        Target protein of the interaction. It can either be the int `id`
        of the protein, the instance itself or the `uniprot_id` of the protein.
        If creating many interactions, it will be much faster to pass
        in the protein instance itself to avoid constant database queries
        during validation.

    label : str or list, optional, default: None
        A label or list of labels for this interaction.

    is_interactome : bool, optional, default: False
        Indicates if this interaction belongs to an interactome dataset.

    is_training : bool, optional, default: False
        Indicates if this interaction belongs to a training dataset. Must
        be labelled if this is set as True.

    is_holdout : bool, optional, default: False
        Indicates if this interaction belongs to a holdout dataset. Must
        be labelled if this is set as True.

    go_mf : str, optional, default: None
        A comma delimited string of Gene Ontology: Molecular Function
        annotations.

    go_bp : str, optional, default: None
        A comma delimited string of Gene Ontology: Biological Process
        annotations.

    go_cc : str, optional, default: None
        A comma delimited string of Gene Ontology: Cellular Component
        annotations.

    ulca_go_mf : str, optional, default: None
        A comma delimited string of Gene Ontology: Molecular Function
        annotations computed using feature induction.

    ulca_go_bp : str, optional, default: None
        A comma delimited string of Gene Ontology: Biological Process
        annotations computed using feature induction.

    ulca_go_cc : str, optional, default: None
        A comma delimited string of Gene Ontology: Cellular Component
        annotations computed using feature induction.

    interpro : str, optional, default: None
        A comma delimited string of InterPro domain annotations.

    pfam : str, optional, default: None
        A comma delimited string of Pfam annotations.

    keywords : str, optional, default: None
        A comma delimited string of keyword annotations.

    session : :class:`scoped_session`, optional.
        A session instance to save to. Leave as None to use the default
        session and save to the database located at `~/.pyppi/pyppi.db`

    commit : bool, default: False
        Commit attempts to save changes to the database, wrapped within
        an atomic transaction. If an error occurs, any changes will be
        rolledback. Ignored if `save` is False.

    save : bool, optional, default: False
        Save instance to the database.

    verbose : bool, default: False
        Log messages that occur during the call.

    Returns
    -------
    :class:`Interaction`
        The created interaction instance.
    """
    if session is None:
        session = db_session
    try:
        is_interactome = kwargs.get('is_interactome', False)
        is_holdout = kwargs.get('is_holdout', False)
        is_training = kwargs.get('is_training', False)
        go_mf = kwargs.get('go_mf', None)
        go_bp = kwargs.get('go_bp', None)
        go_cc = kwargs.get('go_cc', None)
        ulca_go_mf = kwargs.get('ulca_go_mf', None)
        ulca_go_bp = kwargs.get('ulca_go_bp', None)
        ulca_go_cc = kwargs.get('ulca_go_cc', None)
        pfam = kwargs.get('pfam', None)
        interpro = kwargs.get('interpro', None)
        keywords = kwargs.get('keywords', None)

        entry = Interaction(
            source=source, target=target, label=labels,
            is_interactome=is_interactome, is_training=is_training,
            is_holdout=is_holdout, go_mf=go_mf, go_bp=go_bp, go_cc=go_cc,
            ulca_go_mf=ulca_go_mf, ulca_go_bp=ulca_go_bp,
            ulca_go_cc=ulca_go_cc, interpro=interpro, pfam=pfam,
            keywords=keywords
        )
        if save:
            entry.save(session, commit=commit)
        return entry
    except Exception as e:
        if verbose:
            logger.exception(e)
        raise


def proteins_from_dat(file_path, session=None, verbose=False):
    """Parses dat files into protein records to be saved into the database. If
    the file path ends in `.gz` then gzip will be used to read from the file
    automatically. If an entry already exists in the database with the same
    `UniProt` identifier, then that entry will have all of it's attributes
    updated.

    Parameters
    ----------
    file_path : str
        The path to the `.dat` or `.dat.gz` file to read from.

    session : :class:`scoped_session`, optional.
        A session instance to save to. Leave as None to use the default
        session and save to the database located at `~/.pyppi/pyppi.db`

    verbose : bool, default: False
        Log messages that occur during the call.

    Returns
    -------
    `tuple`
        First element is a list of new :class:`Protein` instances created, the 
        second is a list of updated instances.
    """
    if session is None:
        session = db_session

    # Not memory efficient but speeds up the process since we don't
    # need to constantly query the database for entries. This
    # becomes a faster dict lookup.
    existing = uniprotid_entry_map()
    new_proteins = []
    updated_proteins = []

    if os.path.splitext(file_path)[-1] == '.gz':
        fp = gzip.open(file_path, 'rt')
    else:
        fp = open(file_path, 'rt')

    records = SwissProt.parse(fp)
    for record in records:
        params = serialise_record(record)
        if params is None:
            continue

        uniprot_id = params.get('uniprot_id')
        protein = existing.get(uniprot_id, None)
        if protein is None:
            if verbose:
                logger.info(
                    "Creating new entry '{}'.".format(uniprot_id)
                )
            protein = Protein(**params)
            new_proteins.append(protein)
        else:
            if verbose:
                logger.info(
                    "Updating entry '{}'.".format(uniprot_id)
                )
            for k, v in params.items():
                setattr(protein, k, v)
            updated_proteins.append(protein)

    try:
        session.add_all(new_proteins + updated_proteins)
        session.commit()
        fp.close()
        return new_proteins, updated_proteins
    except:
        if verbose:
            logger.exception("An error occured when trying to parse record.")
        session.rollback()
        fp.close()
        raise


def psimi_from_obo(file_path, session=None):
    """Parses a gzipped PSI-MI obo file into :class:`Psimi` entries in 
    the database. Will update existing enties.

    Parameters
    ----------
    file_path : str
        The path to the gzipped file to read from.

    session : :class:`scoped_session`, optional.
        A session instance to save to. Leave as None to use the default
        session and save to the database located at `~/.pyppi/pyppi.db`

    Returns
    -------
    `tuple`
        First element is a list of new :class:`Psimi` instances created, the 
        second is a list of updated instances.
    """

    if session is None:
        session = db_session

    mi_ont = parse_miobo_file(file_path)
    new = []
    updated = []

    # Not memory efficient but speeds up the process since we don't
    # need to constantly query the database for entries. This
    # becomes a faster dict lookup.
    psimi_map = {p.accession: p for p in Psimi.query.all()}
    for key, term in mi_ont.items():
        entry = psimi_map.get(key, None)
        if entry is None:
            entry = Psimi(accession=key, description=term.name)
            new.append(entry)
        else:
            entry.description = term.name
            updated.append(entry)

    try:
        session.add_all(new + updated)
        session.commit()
        return new, updated
    except:
        session.rollback()
        raise


def pmids_from_list(pmids, session=None):
    """Parses a set/list of pmids and adds them to the database. Updates 
    existing entries.

    Parameters
    ----------
    pmids : list or set
        List/set of uppercase pmids.

    session : :class:`scoped_session`, optional.
        A session instance to save to. Leave as None to use the default
        session and save to the database located at `~/.pyppi/pyppi.db`

    Returns
    -------
    `tuple`
        First element is a list of new :class:`Pubmed` instances created, the 
        second is a list of updated instances.
    """

    if session is None:
        session = db_session

    if not isinstance(pmids, (list, set)):
        raise TypeError("`pmids` must a list or set. Found {}.".format(
            type(pmids).__name__
        ))

    new = []
    updated = []
    pmids = list(pmids)
    # Not memory efficient but speeds up the process since we don't
    # need to constantly query the database for entries. This
    # becomes a faster dict lookup.
    existing = {p.accession: p for p in Pubmed.query.all()}
    for pmid in pmids:
        if is_null(pmid):
            continue
        pmid = pmid.strip().upper()
        entry = existing.get(pmid, None)
        if entry is None:
            entry = Pubmed(accession=pmid)
            new.append(entry)
        else:
            updated.append(entry)

    try:
        session.add_all(new + updated)
        session.commit()
        return new, updated
    except:
        session.rollback()
        raise


def psimis_from_list(psimi_tuples, session=None):
    """Parses a list/set of PSI-MI `(accession, description)` tuples and adds 
    them to the database. Updates existing entries.

    Parameters
    ----------
    psimi_tuples : list or set
        List or set of `2-tuples` `(MI:X accession, text description)`.

    session : :class:`scoped_session`, optional.
        A session instance to save to. Leave as None to use the default
        session and save to the database located at `~/.pyppi/pyppi.db`

    Returns
    -------
    `tuple`
        First element is a list of new :class:`Psimi` instances created, the 
        second is a list of updated instances.
    """

    if session is None:
        session = db_session

    if not isinstance(psimi_tuples, (list, set)):
        raise TypeError("`psimi` must a list or set. Found {}.".format(
            type(psimi_tuples).__name__
        ))

    new = []
    updated = []
    psimi_tuples = set(psimi_tuples)
    # Not memory efficient but speeds up the process since we don't
    # need to constantly query the database for entries. This
    # becomes a faster dict lookup.
    existing = {p.accession: p for p in Psimi.query.all()}
    for psimi, desc in psimi_tuples:
        if is_null(psimi):
            continue
        psimi = psimi.strip().upper()
        desc = desc.strip()

        entry = existing.get(psimi, None)
        if entry is None:
            entry = Psimi(accession=psimi, description=desc)
            new.append(entry)
        else:
            entry.description = desc
            updated.append(entry)

    try:
        session.add_all(new + updated)
        session.commit()
        return new, updated
    except:
        session.rollback()
        raise
