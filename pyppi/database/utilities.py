"""
The two methods `init_protein_database` and `init_interaction_database`
are two utility functions to set up the local database tables with the data
from the uniprot dumps and with interactions from the parsed networks.
"""

import logging

from Bio import SwissProt
from joblib import Parallel, delayed
from sqlalchemy.exc import IntegrityError

from ..base import SOURCE, TARGET, LABEL, EXPERIMENT_TYPE, PUBMED
from ..base import is_null

from ..data import uniprot_sprot, uniprot_trembl
from ..data_mining.features import compute_interaction_features
from ..data_mining.uniprot import parse_record_into_protein

from . import begin_transaction
from .models import Interaction, Psimi, Protein, Pubmed
from .managers import InteractionManager
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


def add_interaction(session, commit=False, verbose=False, psimi_ls=(),
                    pmid_ls=(), existing_interactions=None,
                    match_taxon_id=None, **class_kwargs):
    try:
        i_manager = InteractionManager(
            verbose=verbose, match_taxon_id=match_taxon_id
        )
        source = class_kwargs["source"]
        target = class_kwargs["target"]
        if isinstance(source, Protein):
            source = source.uniprot_id
        if isinstance(target, Protein):
            target = target.uniprot_id

        if existing_interactions is not None:
            entry = existing_interactions.get((source, target), None)
        else:
            entry = i_manager.get_by_source_target(session, source, target)

        if entry is not None:
            raise ObjectAlreadyExists(
                "Interaction ({}, {}) already exists.".format(
                    source, target)
            )
        else:
            entry = Interaction(
                source=class_kwargs["source"],
                target=class_kwargs["target"],
                label=class_kwargs["label"],
                is_interactome=class_kwargs["is_interactome"],
                is_training=class_kwargs["is_training"],
                is_holdout=class_kwargs["is_holdout"],
                go_mf=class_kwargs["go_mf"],
                go_bp=class_kwargs["go_bp"],
                go_cc=class_kwargs["go_cc"],
                ulca_go_mf=class_kwargs["ulca_go_mf"],
                ulca_go_bp=class_kwargs["ulca_go_bp"],
                ulca_go_cc=class_kwargs["ulca_go_cc"],
                interpro=class_kwargs["interpro"],
                pfam=class_kwargs["pfam"],
                keywords=class_kwargs["keywords"],
            )
            entry.save(session, commit=commit)
            for psimi in psimi_ls:
                entry.add_psimi_reference(psimi)
            for pmid in pmid_ls:
                entry.add_pmid_reference(pmid)
            return entry
    except Exception as e:
        logger.exception(e)
        raise


def update_interaction(session, commit=False, psimi_ls=(), pmid_ls=(),
                       replace_fields=False, override_boolean=False,
                       create_if_not_found=False, match_taxon_id=None,
                       verbose=False, update_features=True,
                       existing_interactions=None, **class_kwargs):
    try:
        i_manager = InteractionManager(
            verbose=verbose, match_taxon_id=match_taxon_id
        )
        source = class_kwargs["source"]
        target = class_kwargs["target"]
        if isinstance(source, Protein):
            source = source.uniprot_id
        if isinstance(target, Protein):
            target = target.uniprot_id

        if existing_interactions is not None:
            entry = existing_interactions.get((source, target), None)
        else:
            entry = i_manager.get_by_source_target(session, source, target)

        if entry is None and not create_if_not_found:
            raise ObjectNotFound(
                "Interaction ({}, {}) doesn't exist.".format(source, target)
            )
        elif entry is None and create_if_not_found:
            return add_interaction(
                session, commit=commit,
                existing_interactions=existing_interactions,
                psimi_ls=psimi_ls, pmid_ls=pmid_ls,
                **class_kwargs
            )

        if replace_fields:
            entry.label = class_kwargs["label"]
            entry.is_interactome = class_kwargs["is_interactome"]
            entry.is_training = class_kwargs["is_training"]
            entry.is_holdout = class_kwargs["is_holdout"]
            entry.go_mf = class_kwargs["go_mf"]
            entry.go_bp = class_kwargs["go_bp"]
            entry.go_cc = class_kwargs["go_cc"]
            entry.ulca_go_mf = class_kwargs["ulca_go_mf"]
            entry.ulca_go_bp = class_kwargs["ulca_go_bp"]
            entry.ulca_go_cc = class_kwargs["ulca_go_cc"]
            entry.interpro = class_kwargs["interpro"]
            entry.pfam = class_kwargs["pfam"]
            entry.keywords = class_kwargs["keywords"]

            entry.save(session, commit=commit)
            current_psimis = entry.psimi
            current_pmids = entry.pmid
            for psimi in current_psimis:
                entry.remove_psimi_reference(psimi)
            for pmid in current_pmids:
                entry.remove_pmid_reference(pmid)
            for psimi in psimi_ls:
                entry.add_psimi_reference(psimi)
            for pmid in pmid_ls:
                entry.add_pmid_reference(pmid)
        else:
            new_label = class_kwargs["label"]
            if new_label:
                entry.label = entry.labels_as_list + new_label.split(',')

            if override_boolean:
                entry.is_interactome = class_kwargs["is_interactome"]
                entry.is_training = class_kwargs["is_training"]
                entry.is_holdout = class_kwargs["is_holdout"]
            else:
                entry.is_interactome |= class_kwargs["is_interactome"]
                entry.is_training |= class_kwargs["is_training"]
                entry.is_holdout |= class_kwargs["is_holdout"]

            if update_features:
                annotation_keys = [
                    "go_mf", "go_bp", "go_cc",
                    "ulca_go_mf", "ulca_go_bp", "ulca_go_cc",
                    "interpro", "pfam", "keywords"
                ]
                for key in annotation_keys:
                    new_values = class_kwargs[key]
                    if new_values:
                        curr_values = getattr(entry, key)
                        if curr_values is not None:
                            value = curr_values.split(',') + \
                                new_values.split(',')
                            setattr(entry, key, value)
                        else:
                            setattr(entry, key, new_values)

            entry.save(session, commit=commit)
            for psimi in psimi_ls:
                entry.add_psimi_reference(psimi)
            for pmid in pmid_ls:
                entry.add_pmid_reference(pmid)

        return entry
    except Exception as e:
        logger.exception(e)
        raise


def pmid_string_to_list(session, pmids):
    if is_null(pmids):
        return []

    entries = []
    for pmid in pmids.split(','):
        entry = session.query(Pubmed).filter_by(
            accession=pmid
        ).first()
        if entry is None:
            raise ObjectNotFound("Pubmed {} does not exist.".format(pmid))
        else:
            entries.append(entry)
    return entries


def psimi_string_to_list(session, psimis):
    if is_null(psimis):
        return []

    entries = []
    for psimi in psimis.split(','):
        entry = session.query(Psimi).filter_by(
            accession=psimi
        ).first()
        if entry is None:
            raise ObjectNotFound("Psimi {} does not exist.".format(psimi))
        else:
            entries.append(entry)
    return entries


def generate_interaction_tuples(df):
    zipped = zip(
        df[SOURCE],
        df[TARGET],
        df[LABEL],
        df[PUBMED],
        df[EXPERIMENT_TYPE]
    )
    for (uniprot_a, uniprot_b, label, pmids, psimis) in zipped:
        if is_null(uniprot_a):
            raise ValueError("Source cannot be None")
        if is_null(uniprot_b):
            raise ValueError("Target cannot be None")
        if is_null(label):
            label = None
        if is_null(pmids):
            pmids = None
        if is_null(psimis):
            psimis = None

        yield (uniprot_a, uniprot_b, label, pmids, psimis)
