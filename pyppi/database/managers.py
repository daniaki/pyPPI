
"""
This module contains classes that interface with the different
tables within the database. If anything changes in the database, the code
resoponsible should be placed in here.
"""
import logging
import numpy as np
from enum import EnumMeta

from sqlalchemy.sql import and_

from .exceptions import ObjectAlreadyExists, ObjectNotFound
from .models import Protein, Interaction, _format_annotations

from ..data_mining.features import compute_interaction_features
from ..data_mining.features import split_string as split_
from ..data_mining.ontology import (
    get_up_to_lca, group_terms_by_ontology_type
)

logger = logging.getLogger("pyppi")


class ProteinManager(object):

    def __init__(self, verbose=False, match_taxon_id=9606):
        self.verbose = verbose
        self.match_taxon_id = match_taxon_id

    @property
    def columns(self):
        return Protein.columns

    def get_by_id(self, session, id):
        if not isinstance(id, int):
            raise TypeError("Expected type int for id. Found '{}'.".format(
                type(id).__name__
            ))
        try:
            return session.query(Protein).get(id)
        except:
            raise

    def get_by_uniprot_id(self, session, uniprot_id):
        query_set = session.query(Protein).filter(
            Protein.uniprot_id == uniprot_id
        )

        if query_set.count() == 0:
            return None

        elif query_set.count() > 0 and self.match_taxon_id is not None \
                and query_set[0].taxon_id is not None \
                and query_set[0].taxon_id != self.match_taxon_id:
            if self.verbose:
                logger.warning(
                    "Entry '{}' did not match taxonomy id '{}'. "
                    "Entry has taxonomy id '{}'.".format(
                        uniprot_id, self.match_taxon_id, query_set[0].taxon_id
                    )
                )
            return None
        else:
            return query_set[0]

    def entry_to_dict(self, session, id=None, uniprot_id=None, split=False):
        if id is not None:
            entry = self.get_by_id(session, id)
        elif uniprot_id is not None:
            entry = self.get_by_uniprot_id(session, uniprot_id)
        else:
            raise ValueError(
                "You must supply either an integer id (primary key) or "
                "the uniprot accession id."
            )
        if entry is None:
            return {}
        return {
            "go_bp": split_(entry.go_bp) if split else entry.go_bp,
            "go_mf": split_(entry.go_mf) if split else entry.go_mf,
            "go_cc": split_(entry.go_cc) if split else entry.go_cc,
            "interpro": split_(entry.interpro) if split else entry.interpro,
            "pfam": split_(entry.pfam) if split else entry.pfam,
            "keywords": split_(entry.keywords) if split else entry.keywords,
            "reviewed": entry.reviewed,
            "gene_id": entry.gene_id,
            "taxon_id": entry.taxon_id,
            "uniprot_id": entry.uniprot_id,
            "id": entry.id
        }

    def uniprotid_entry_map(self, session):
        proteins = session.query(Protein).all()
        protein_map = {p.uniprot_id: p for p in proteins}
        return protein_map


class InteractionManager(object):

    def __init__(self, verbose=False, match_taxon_id=9606):
        self.verbose = verbose
        self.match_taxon_id = match_taxon_id
        self.p_manager = ProteinManager(
            verbose=verbose, match_taxon_id=match_taxon_id
        )

    @property
    def columns(self):
        return Interaction.columns

    def taxon_ids_match(self, interaction):
        if self.match_taxon_id is None:
            return True
        else:
            return interaction.taxon_id == self.match_taxon_id

    def filter_matching_taxon_ids(self, query_set):
        if self.match_taxon_id is not None:
            if isinstance(query_set, list):
                return [i for i in query_set if self.taxon_ids_match(i)]
            else:
                return query_set.filter(
                    Interaction.taxon_id == self.match_taxon_id
                )
        else:
            return query_set

    def _get_protein(self, session, value, arg_name):
        if isinstance(value, str):
            return self.p_manager.get_by_uniprot_id(session, value)
        elif isinstance(value, int):
            return self.p_manager.get_by_id(session, value)
        elif isinstance(value, Protein):
            return value
        else:
            raise TypeError(
                "{} argument must be an 'int' or 'str'. "
                "Found '{}'.".format(arg_name, type(value).__name__)
            )

    def get_by_id(self, session, id):
        if not isinstance(id, int):
            raise TypeError("Expected type int for id. Found '{}'.".format(
                type(id).__name__
            ))
        entry = session.query(Interaction).get(id)
        if entry is not None and self.taxon_ids_match(entry):
            return entry
        return None

    def get_by_source_target(self, session, source, target):
        source = self._get_protein(session, source, 'source')
        target = self._get_protein(session, target, 'target')
        if source is None or target is None:
            return None
        query_set = session.query(Interaction).filter(and_(
            Interaction.source == source.id,
            Interaction.target == target.id
        ))
        if query_set.count() > 1:
            raise ValueError(
                "More than one Interaction found Database integrity "
                "has been compromised."
            )
        if self.match_taxon_id is not None:
            query_set = query_set.filter_by(taxon_id=self.match_taxon_id)
            if query_set.count() == 0:
                if self.verbose:
                    logger.info(
                        "No interactions matching ({}, {}) and taxonomy "
                        "id {} found.".format(
                            "Any" if source is None else source.uniprot_id,
                            "Any" if target is None else target.uniprot_id,
                            self.match_taxon_id,
                        )
                    )
                return None
            else:
                return query_set[0]
        else:
            return None

    def get_by_label(self, session, label=None):
        if label is not None:
            label = ','.join(
                sorted(set(
                    [l.strip().capitalize() for l in label.split(',')]
                ))
            )
        query_set = session.query(Interaction).filter(
            Interaction._label == label
        )
        return self.filter_matching_taxon_ids(query_set)

    def get_contains_label(self, session, label):
        label = _format_annotations(label, upper=True, allow_duplicates=False)
        label = ','.join([l.capitalize() for l in label])

        query_set = session.query(Interaction).filter(
            # Using the property 'label' doesn't seem to work with SQL.
            # Use _label directly instead.
            Interaction._label.contains(label)
        )
        return self.filter_matching_taxon_ids(query_set)

    def get_by_source(self, session, source):
        source = self._get_protein(session, source, 'source')
        query_set = session.query(Interaction).filter(
            Interaction.source == source.id
        )
        return self.filter_matching_taxon_ids(query_set)

    def get_by_target(self, session, target):
        target = self._get_protein(session, target, 'target')
        query_set = session.query(Interaction).filter(
            Interaction.target == target.id
        )
        return self.filter_matching_taxon_ids(query_set)

    def entry_to_dict(self, session, id=None,
                      source=None, target=None, split=False):
        if id is not None:
            entry = self.get_by_id(session, id)
        elif source is not None and target is not None:
            entry = self.get_by_source_target(session, source, target)
        else:
            raise ValueError(
                "You must supply either an integer id (primary key) or "
                "the source and target uniprot accession ids."
            )
        if entry is None:
            return {}
        source = self._get_protein(session, entry.source, arg_name="source")
        target = self._get_protein(session, entry.target, arg_name="target")
        return {
            "go_bp": split_(entry.go_bp) if split else entry.go_bp,
            "go_mf": split_(entry.go_mf) if split else entry.go_mf,
            "go_cc": split_(entry.go_cc) if split else entry.go_cc,
            "interpro": split_(entry.interpro) if split else entry.interpro,
            "pfam": split_(entry.pfam) if split else entry.pfam,
            "keywords": split_(entry.keywords) if split else entry.keywords,
            "ulca_go_bp": split_(entry.ulca_go_bp) if split else entry.ulca_go_bp,
            "ulca_go_mf": split_(entry.ulca_go_mf) if split else entry.ulca_go_mf,
            "ulca_go_cc": split_(entry.ulca_go_cc) if split else entry.ulca_go_cc,
            "source": source.uniprot_id,
            "target": target.uniprot_id,
            "is_holdout": entry.is_holdout,
            "is_training": entry.is_training,
            "is_interactome": entry.is_interactome,
            "label": split_(entry.label) if split else entry.label,
            "id": entry.id
        }

    def training_interactions(self, session, filter_out_holdout=False):
        qs = session.query(Interaction).filter(
            Interaction.is_training.is_(True)
        )
        if filter_out_holdout:
            qs = qs.filter_by(is_holdout=False)
        return self.filter_matching_taxon_ids(qs)

    def holdout_interactions(self, session, filter_out_training=False):
        qs = session.query(Interaction).filter(
            Interaction.is_holdout.is_(True)
        )
        if filter_out_training:
            qs = qs.filter_by(is_training=False)
        return self.filter_matching_taxon_ids(qs)

    def interactome_interactions(self, session, filter_out_training=False,
                                 filter_out_holdout=False):
        qs = session.query(Interaction).filter(
            Interaction.is_interactome.is_(True)
        )
        if filter_out_holdout:
            qs = qs.filter_by(is_holdout=False)
        if filter_out_training:
            qs = qs.filter_by(is_training=False)

        return self.filter_matching_taxon_ids(qs)

    def training_labels(self, session, include_holdout=False):
        labels = set()
        interactions = self.training_interactions(
            session, filter_out_holdout=not include_holdout)
        for interaction in interactions:
            if interaction.label is None:
                continue
            labels |= set([x for x in interaction.label.split(',') if x])
        return list(sorted(labels))


def format_interactions_for_sklearn(interactions, selection):
    X = list(range(len(list(interactions))))
    y = list(range(len(list(interactions))))
    for i, interaction in enumerate(interactions):
        x = ''
        label = interaction.label
        for attr in selection:
            try:
                value = getattr(interaction, attr)
            except:
                value = getattr(interaction, attr.value)
            if value:
                if x:
                    x = ','.join([x, value])
                else:
                    x = value

        if label is None:
            label = []
        else:
            label = label.split(',')

        X[i] = x
        y[i] = label

    return np.asarray(X), y
