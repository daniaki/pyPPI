
"""
This module contains the SQLAlchemy database definitions and related
functions to access and update the database files.
"""
import logging
import numpy as np
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Boolean, ForeignKey, Table, DateTime,
    UniqueConstraint
)
from sqlalchemy.orm import (
    relationship, mapper, validates, backref, Query, scoped_session
)
from sqlalchemy.sql import and_

from . import Base, db_session
from .exceptions import ObjectNotFound, ObjectAlreadyExists
from .validators import (
    validate_function, validate_gene_id, validate_keywords,
    validate_pfam_annotations, validate_interpro_annotations,
    validate_go_annotations, validate_function,
    validate_boolean, validate_joint_id, validate_labels,
    validate_training_holdout_is_labelled,  validate_taxon_id,
    validate_source_and_target, validate_protein,
    validate_same_taxonid, validate_uniprot_does_not_exist,
    validate_accession, validate_interaction_does_not_exist,
    validate_description
)

logger = logging.getLogger("pyppi")


class Protein(Base):
    """
    This is the ORM specification for the table `protein`. It is used as
    an object representation to store relevant aspects of a protein
    in a convenient format. All fields should be parsed from UniProt 
    database records.

    Parameters
    ----------
    id : int
        The integer primary key of the instance.

    uniprot_id : str
        The UniProt accession string.

    taxon_id : int
        An integer representing the UniProt taxonomy id. `9606` represents
        human.

    gene_id : str, optional, default: None
        The gene name related to this protein.

    go_mf : str, optional, default: None
        A comma delimited string of Gene Ontology: Molecular Function
        annotations.

    go_bp : str, optional, default: None
        A comma delimited string of Gene Ontology: Biological Process
        annotations.

    go_cc : str, optional, default: None
        A comma delimited string of Gene Ontology: Cellular Component
        annotations.

    interpro : str, optional, default: None
        A comma delimited string of InterPro domain annotations.

    pfam : str, optional, default: None
        A comma delimited string of Pfam annotations.

    keywords : str, optional, default: None
        A comma delimited string of keyword annotations.

    reviewed : bool, optional, default: False
        If `True`, this protein has status `Reviewed` according to UniProt
        and is a member of the SwissProt database. Otherwise, it treated as a
        TrEMBL entry.

    last_update : datetime or str, optional, default: None
        Either a datetime object which contains a month, day and year or a 
        string in the form DD-MMM-YYY indicating the last time UniProt updated
        the annotations for this entry. `MMM` refers to the three letter month
        code eg. JAN for January.

    last_release : int, optional, default: None
        An integer representing the UniProt release this entry was parsed from.

    Attributes
    ----------
    interactions : list
        Contains a list of associated interactions with an instance.

    Notes
    -----
    `last_update` and `last_release` should be updated whenever this entry 
    is updated with new annotations.

    """
    __tablename__ = "protein"

    id = Column('id', Integer, primary_key=True)
    uniprot_id = Column('uniprot_id', String, nullable=False, unique=True)
    taxon_id = Column('taxon_id', Integer, nullable=False)
    gene_id = Column('gene_id', String)
    go_mf = Column('go_mf', String)
    go_cc = Column('go_cc', String)
    go_bp = Column('go_bp', String)
    interpro = Column('interpro', String)
    pfam = Column('pfam', String)
    keywords = Column('keywords', String)
    function = Column('function', String)
    reviewed = Column('reviewed', Boolean, nullable=False, default=False)
    last_update = Column('last_update', DateTime, default=None)
    last_release = Column('last_release', Integer, default=None)

    interactions = relationship(
        "Interaction",
        primaryjoin=(
            "or_("
            "Protein.id==Interaction.source_,"
            "Protein.id==Interaction.target_"
            ")"
        )
    )

    def __init__(self, uniprot_id=None, gene_id=None, taxon_id=None,
                 go_mf=None, go_cc=None, go_bp=None, interpro=None, pfam=None,
                 keywords=None, reviewed=None, function=None, last_update=None,
                 last_release=None):
        self.uniprot_id = uniprot_id
        self.gene_id = gene_id
        self.taxon_id = taxon_id
        self.go_mf = go_mf
        self.go_cc = go_cc
        self.go_bp = go_bp
        self.interpro = interpro
        self.pfam = pfam
        self.reviewed = reviewed
        self.keywords = keywords
        self.function = function
        self.last_update = last_update
        self.last_release = last_release

    @staticmethod
    def columns():
        """Returns an Enum of columns.

        Returns
        -------
        Enum
            Enum values contain the column names.
        """
        class Columns(Enum):
            GO_MF = 'go_mf'
            GO_BP = 'go_bp'
            GO_CC = 'go_cc'
            PFAM = 'pfam'
            INTERPRO = 'interpro'
            REVIEWED = 'reviewed'
            KW = 'keywords'
            TAX = 'taxon_id'
            GENE = 'gene_id'
            UNIPROT = "uniprot_id"
            FUNCTION = "function"
            LAST_UPDATE = "last_update"
            LAST_RELEASE = "last_release"
        return Columns

    @classmethod
    def get_by_uniprot_id(cls, uniprot_id):
        """Return the instance associated with a particular UniProt accession.

        Parameters
        ----------
        uniprot_id : str
            A UniProt accession string.

        Returns
        -------
        :class:`Protein` or None
            A protein instance associated with this accession or None
            if no hits could be found.
        """
        if uniprot_id is None:
            return None
        upid = validate_accession(
            uniprot_id, klass=Protein, upper=True, check_exists=False
        )
        if not upid:
            return None
        return cls.query.filter_by(uniprot_id=upid).first()

    def __repr__(self):
        string = (
            "<Protein(id={}, uniprot_id={}, gene_id={}, "
            "taxon_id={}, reviewed={}, release={})>"
        )
        return string.format(
            self.id, self.uniprot_id, self.gene_id, self.taxon_id,
            self.reviewed, self.last_release
        )

    def __eq__(self, other):
        if not isinstance(other, Protein):
            raise TypeError("Cannot compare 'Protein' with '{}'".format(
                type(other).__name__
            ))
        else:
            return all([
                getattr(self, attr.value) == getattr(other, attr.value)
                for attr in Protein.columns()
            ])

    # ------------------------- validators --------------------------------- #
    @validates('uniprot_id')
    def _validate_uniprot_id(self, key, uniprot_id):
        if uniprot_id is not None:
            if hasattr(self, 'uniprot_id') and self.uniprot_id == uniprot_id:
                return uniprot_id
        return validate_accession(
            uniprot_id, klass=Protein, upper=True, check_exists=True
        )

    @validates('gene_id')
    def _validate_gene_id(self, key, gene_id):
        return validate_gene_id(gene_id)

    @validates('taxon_id')
    def _validate_taxon_id(self, key, id_):
        return validate_taxon_id(id_)

    @validates('reviewed')
    def _validate_reviewed(self, key, reviewed):
        return validate_boolean(reviewed)

    @validates(*['go_mf', 'go_cc', 'go_bp'])
    def _validate_go_annotations(self, key, values):
        return validate_go_annotations(
            values, upper=True, allow_duplicates=False
        )

    @validates('interpro')
    def _validate_interpro_annotations(self, key, values):
        return validate_interpro_annotations(
            values, upper=True, allow_duplicates=False
        )

    @validates('pfam')
    def _validate_pfam_annotations(self, key, values):
        return validate_pfam_annotations(
            values, upper=True, allow_duplicates=False
        )

    @validates('keywords')
    def _validate_keywords(self, key, values):
        return validate_keywords(values)

    @validates('function')
    def _validate_function(self, key, value):
        return validate_function(value)

    @validates('last_release')
    def _validate_last_release(self, key, value):
        if value is None:
            return None
        if not isinstance(value, int):
            raise TypeError("`last_release` must be an int.")
        elif value < 1:
            raise TypeError("`last_release` must be positive.")
        return value

    @validates('last_update')
    def _validate_last_update(self, key, value):
        if value is None:
            return None
        try:
            return datetime.strptime(value, '%d-%b-%Y')
        except TypeError:
            raise TypeError(
                "`last_update` must be str not {}".format(type(value).__name__)
            )
        except ValueError:
            raise ValueError(
                "`last_update` must be a string in the format "
                "DD-MONTH-YYYY, where `MONTH` is the first three letters "
                "of your desired month."
            )

    # ---------------------------- METHODS ------------------------------- #
    def save(self, session=None, commit=False):
        """
        Save entry by adding it to a session.

        Adds this instance to a session object or default global session.
        If commit is passed in, the instance is commited and saved to the 
        database.

        Parameters
        ----------
        session : :class:`scoped_session`, optional.
            A session instance to save to. Leave as None to use the default
            session and save to the database located at `~/.pyppi/pyppi.db`

        commit : bool, default: False
            Commit attempts to save changes to the database, wrapped within
            an atomic transaction. If an error occurs, any changes will be
            rolledback.
        """
        if session is None:
            session = db_session
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise
        return self

    def release_outdated(self, release):
        """Compares the release of this instance to that passed in. 

        Parameters
        ----------
        release : int
            Integer UniProt release number.

        Returns
        -------
        bool
            True if this release is earlier than that supplied. Defaults to 
            True if `last_release` is None.
        """
        if not isinstance(release, int):
            raise TypeError("`release` must be a positive int.")
        if self.last_release is None:
            return True
        return self.last_release < release

    def annotations_outdated(self, date):
        """Compares the date of the last annotation update of this instance
        to the date supplied.

        Parameters
        ----------
        release : datetime or str.
            A datetime object or a string in the form DD-MMM-YYY indicating 
            the last time UniProt updated the annotations for this entry. 
            `MMM` refers to the three letter month code eg. JAN for January.

        Returns
        -------
        bool
            True if the date of `last_update` is earlier than that supplied. 
            Defaults to True if `last_update` is None.
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%d-%b-%Y')
        elif not isinstance(date, datetime):
            raise TypeError("`date` must be a datetime instance.")
        if self.last_update is None:
            return True
        return self.last_update < date


class Interaction(Base):
    """
    This is the ORM specification for the table `interaction`. It is used as
    an object representation to store relevant aspects of an interaction
    between two proteins of the same taxonomy id in a convenient format. 

    Parameters
    ----------
    id : int
        The integer primary key of the instance.

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

    Attributes
    ----------
    joint_id : str
        A string identifier created during creation to uniquely identify
        this reaction using the protein integer ids. It is computed as
        the comma joined sorted integer ids of `source` and `target`.

    taxon_id : int
        An integer representing the UniProt taxonomy id. `9606` represents
        human. It is derived from the `source` protein.

    Notes
    -----
    Interactions are not directional, so `(A, B)` will be treated as 
    `(B, A)`.

    """
    __tablename__ = "interaction"

    # Create a unique identifier using the uniprot ids joint_id
    # into a string. This column is unique causing a contraint
    # failure if an (B, A) is added when (A, B) already exists.
    id = Column(Integer, primary_key=True)
    target_ = Column('target', ForeignKey("protein.id"), nullable=False)
    source_ = Column('source', ForeignKey("protein.id"), nullable=False)
    joint_id_ = Column('joint_id', String, unique=True, nullable=False)
    taxon_id = Column('taxon_id', Integer, nullable=False)

    is_training = Column(Boolean, nullable=False, default=False)
    is_holdout = Column(Boolean, nullable=False, default=False)
    is_interactome = Column(Boolean, nullable=False, default=False)

    label = Column('label', String)
    keywords = Column('keywords', String)
    go_mf = Column('go_mf', String)
    go_cc = Column('go_cc', String)
    go_bp = Column('go_bp', String)
    ulca_go_mf = Column('ulca_go_mf', String)
    ulca_go_cc = Column('ulca_go_cc', String)
    ulca_go_bp = Column('ulca_go_bp', String)
    interpro = Column('interpro', String)
    pfam = Column('pfam', String)

    @classmethod
    def get_by_interactors(cls, a, b):
        """Return the instance associated with two proteins if it exists. This
        will search for interactions matching (a, b) and (b, a) using the 
        `joint_id` column.

        Parameters
        ----------
        a : str, int or :class:`Protein`
            A UniProt accession string, int id of a :class:`Protein` instance
            or the id of a :class:`Protein` instance.

        b : str, int or :class:`Protein`
            A UniProt accession string, int id of a :class:`Protein` instance
            or the id of a :class:`Protein` instance.

        Returns
        -------
        :class:`Interaction` or None
            An interaction instance associated proteins `a` and `b` or
            `None` if no hits are found.
        """
        try:
            a = validate_protein(a)
            b = validate_protein(b)
            joint_id = validate_joint_id(a, b)
            return cls.query.filter(Interaction.joint_id_ == joint_id).first()
        except ObjectNotFound:
            return None

    @classmethod
    def get_by_label(cls, label):
        """Return the instances associated with the given label.

        Parameters
        ----------
        label : str
            A label in string format.

        Returns
        -------
        :class:`Query`
            A query instance containing matching instances that can be 
            further queried.
        """
        value = validate_labels(label)
        if value is None:
            None
        else:
            return cls.query.filter(cls.label.contains(value))

    @classmethod
    def get_by_source(cls, source):
        """Return the instances having the `source` column associated with
        the input source.

        Parameters
        ----------
        source : str, int or :class:`Protein`
            A UniProt accession string, int id of a :class:`Protein` instance
            or the id of a :class:`Protein` instance.

        Returns
        -------
        :class:`Query`
            A query instance containing matching instances that can be 
            further queried.
        """
        try:
            source = validate_protein(source)
            return cls.query.filter(cls.source_ == source)
        except ObjectNotFound:
            return None

    @classmethod
    def get_by_target(cls, target):
        """Return the instances having the `target` column associated with
        the input target.

        Parameters
        ----------
        target : str, int or :class:`Protein`
            A UniProt accession string, int id of a :class:`Protein` instance
            or the id of a :class:`Protein` instance.

        Returns
        -------
        :class:`Query`
            A query instance containing matching instances that can be 
            further queried.
        """
        try:
            target = validate_protein(target)
            return cls.query.filter(cls.target_ == target)
        except ObjectNotFound:
            return None

    @staticmethod
    def columns():
        """Returns an Enum of columns.

        Returns
        -------
        Enum
            Enum values contain the column names.
        """
        class Columns(Enum):
            ID = 'id'
            JOINT_ID = 'joint_id'
            LABEL = 'label'
            GO_MF = 'go_mf'
            GO_BP = 'go_bp'
            GO_CC = 'go_cc'
            ULCA_GO_MF = 'ulca_go_mf'
            ULCA_GO_BP = 'ulca_go_bp'
            ULCA_GO_CC = 'ulca_go_cc'
            PFAM = 'pfam'
            INTERPRO = 'interpro'
            KW = 'keywords'
            IS_HOLDOUT = 'is_holdout'
            IS_TRAINING = 'is_training'
            IS_INTERACTOME = 'is_interactome'
        return Columns

    def __init__(self, source=None, target=None, label=None,
                 is_interactome=None, is_training=None, is_holdout=None,
                 go_mf=None, go_cc=None, go_bp=None,
                 ulca_go_mf=None, ulca_go_cc=None, ulca_go_bp=None,
                 interpro=None, pfam=None, keywords=None):

        self.source = source
        self.target = target

        # Use the constructor inputs rather than class instance
        # because if the source/target are protein entries
        # these methods can avoid a database lookup.
        validate_interaction_does_not_exist(source, target)
        taxon_id = validate_same_taxonid(source, target)

        self.taxon_id = taxon_id
        self.joint_id = validate_joint_id(self.source, self.target)

        # Label must be set before these booleans so the validators
        # can check to see if training/holdout is labelled.
        self.label = label
        self.is_training = is_training
        self.is_holdout = is_holdout
        self.is_interactome = is_interactome

        self.go_mf = go_mf
        self.go_cc = go_cc
        self.go_bp = go_bp
        self.ulca_go_mf = ulca_go_mf
        self.ulca_go_cc = ulca_go_cc
        self.ulca_go_bp = ulca_go_bp
        self.interpro = interpro
        self.pfam = pfam
        self.keywords = keywords

    def __repr__(self):
        string = (
            "<Interaction("
            "id={}, source={}/{}, target={}/{}, training={}, holdout={}, "
            "interactome={}, label={}, taxon_id={}"
            ")"
            ">"
        )
        return string.format(
            self.id,
            Protein.query.get(self.source).uniprot_id, self.source,
            Protein.query.get(self.target).uniprot_id, self.target,
            self.is_training, self.is_holdout, self.is_interactome,
            self.label, self.taxon_id
        )

    def __eq__(self, other):
        if not isinstance(other, Interaction):
            raise TypeError("Cannot compare 'Interaction' with '{}'".format(
                type(other).__name__
            ))
        else:
            return all([
                getattr(self, attr.value) == getattr(other, attr.value)
                for attr in Interaction.columns()
            ])

    # ---------------------------- VALIDATORS ----------------------------- #
    @validates(*['go_mf', 'go_cc', 'go_bp'])
    def _validate_go_annotations(self, key, values):
        return validate_go_annotations(
            values, upper=True, allow_duplicates=True
        )

    @validates(*['ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp'])
    def _validate_ulca_go_annotations(self, key, values):
        return validate_go_annotations(
            values, upper=True, allow_duplicates=True
        )

    @validates('interpro')
    def _validate_interpro_annotations(self, key, values):
        return validate_interpro_annotations(
            values, upper=True, allow_duplicates=True
        )

    @validates('pfam')
    def _validate_pfam_annotations(self, key, values):
        return validate_pfam_annotations(
            values, upper=True, allow_duplicates=True
        )

    @validates('keywords')
    def _validate_keywords(self, key, values):
        return validate_keywords(values, allow_duplicates=True)

    @validates('taxon_id')
    def _validate_taxon_id(self, key, id_):
        return validate_taxon_id(id_)

    @validates(*['is_holdout', 'is_training'])
    def _validate_holdout_training(self, key, value):
        value = validate_boolean(value)
        if key == "is_holdout":
            validate_training_holdout_is_labelled(
                self.label, value, self.is_training
            )
        if key == "is_training":
            validate_training_holdout_is_labelled(
                self.label, self.is_holdout, value
            )
        return value

    @validates('is_interactome')
    def _validate_is_interactome(self, key, value):
        return validate_boolean(value)

    @validates('label')
    def _validate_label(self, key, label):
        value = validate_labels(label)
        validate_training_holdout_is_labelled(
            value, self.is_holdout, self.is_training
        )
        return value

    # ---------------------------- PROPERTIES ----------------------------- #
    @property
    def source(self):
        return self.source_

    @source.setter
    def source(self, value):
        if self.source is not None:
            raise AttributeError("Attribute `source` is immutable once set.")
        else:
            self.source_ = validate_protein(value)

    @property
    def target(self):
        return self.target_

    @target.setter
    def target(self, value):
        if self.target is not None:
            raise AttributeError("Attribute `target` is immutable once set.")
        else:
            self.target_ = validate_protein(value)

    @property
    def joint_id(self):
        return self.joint_id_

    @joint_id.setter
    def joint_id(self, value):
        if self.joint_id is not None:
            raise AttributeError("Attribute `joint_id` is immutable once set.")
        else:
            if value != validate_joint_id(self.source, self.target):
                raise AttributeError("Invalid `joint_id` {}".format(value))
            self.joint_id_ = value

    @property
    def labels_as_list(self):
        if self.label is None:
            return []
        else:
            return list(sorted(self.label.split(',')))

    # ---------------------------- METHODS ------------------------------- #
    def save(self, session=None, commit=False):
        """
        Save entry by adding it to a session.

        Adds this instance to a session object or default global session.
        If commit is passed in, the instance is commited and saved to the 
        database.

        Parameters
        ----------
        session : :class:`scoped_session`, optional.
            A session instance to save to. Leave as None to use the default
            session and save to the database located at `~/.pyppi/pyppi.db`

        commit : bool, default: False
            Commit attempts to save changes to the database, wrapped within
            an atomic transaction. If an error occurs, any changes will be
            rolledback.
        """
        if session is None:
            session = db_session
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise
        return self

    def references(self):
        """
        Returns an :class:`Query` object containing
        any references associated with this instance. To execute the
        query call `all()`.

        Returns
        -------
        :class:`Query`
            Query object containing references.
        """
        return Reference.query.filter(
            Reference.interaction_id == self.id)

    def pmids(self):
        """
        Returns an :class:`Query` object containing
        any pmids associated with this instance, or None if no pmids
        are associated.

        Returns
        -------
        :class:`Query` or None
            Query object containing pubmed entries or None if there are
            no rows returned.
        """
        entries = [
            r for rs in self.references().with_entities(Reference.pubmed_id)
            for r in rs
        ]
        if not entries:
            return None
        return Pubmed.query.filter(Pubmed.id.in_(entries))

    def experiment_types(self):
        """
        Returns an :class:`Query` object containing
        any psimi instances associated with this instance, or None if no
        instances are associated.

        Returns
        -------
        :class:`Query` or None
            Query object containing psimi entries or None if there are
            no rows returned.
        """
        entries = [
            r for rs in self.references().with_entities(Reference.psimi_id)
            for r in rs
        ]
        if not entries:
            return None
        return Psimi.query.filter(Psimi.id.in_(entries))

    def add_label(self, label):
        """Add a label to this instance.

        Parameters
        ----------
        label : str
            Label can be a comma delimited string or an iterable of 
            single strings. Labels will be validated by being `stripped of
            whitespace` and `capitalised`. Duplicate labels will not be
            added.
        """
        labels = validate_labels(label, return_list=True)
        existing = self.labels_as_list
        self.label = labels + existing  # should invoke validator
        return self

    def remove_label(self, label):
        """Remove a label from this instance.

        Parameters
        ----------
        label : str
            Label can be a comma delimited string or an iterable of 
            single strings. Labels will be validated by being `stripped of
            whitespace` and `capitalised`. 
        """
        remove = set(validate_labels(label, return_list=True))
        existing_labels = set(self.labels_as_list)
        self.label = list(existing_labels - remove)
        return self

    def add_reference(self, session=None, pmid=None, psimi=None, commit=False):
        """Add a reference to this instance. Optional commit and save it
        to the database.

        Parameters
        ----------
        pmid : int or :class:`Pubmed`
            Instance to associate this interaction with, or its int id.

        psimi : int or :class:`Psimi`, optional.
            Instance to associate this interaction with, or its int id.

        session : :class:`scoped_session`, optional.
            A session instance to save to. Leave as None to use the default
            session and save to the database located at `~/.pyppi/pyppi.db`

        commit : bool, default: False
            Commit attempts to save changes to the database, wrapped within
            an atomic transaction. If an error occurs, any changes will be
            rolledback.

        Returns
        -------
        :class:`Reference`
            The newly created reference.
        """
        if pmid is None:
            raise ValueError("`pmid` cannot be None.")
        ref = Reference(self, pmid, psimi)
        ref.save(session, commit=commit)
        return ref


class Pubmed(Base):
    """
    Pubmed schema definition. This is a basic table containing fields
    relating to a pubmed id. It simple has an integer column, auto-generated
    and auto-incremented, and an `accession` column representing the pubmed
    accession number.

    Parameters
    ----------
    id : int
        The integer primary key of the instance.

    accession : str
        Accession for this Pubmed instance.
    """
    __tablename__ = "pubmed"

    id = Column(Integer, primary_key=True)
    accession = Column(String, unique=True, nullable=False)

    def __init__(self, accession):
        self.accession = accession

    def __repr__(self):
        string = "<Pubmed(id={}, accession={})>"
        return string.format(
            self.id, self.accession
        )

    def references(self):
        """
        Returns an :class:`Query` object containing
        any references associated with this instance. To execute the
        query call `all()`.

        Returns
        -------
        :class:`Query`
            Query object containing references.
        """
        return Reference.query.filter(
            Reference.pubmed_id == self.id)

    def psimis(self):
        """
        Returns an :class:`Query` object containing
        any psimi instances associated with this instance, or None if no 
        instances are associated.

        Returns
        -------
        :class:`Query` or None
            Query object containing pubmed entries or None if there are
            no rows returned.
        """
        entries = [
            r for rs in self.references().with_entities(Reference.psimi_id)
            for r in rs
        ]
        if not entries:
            return None
        return Psimi.query.filter(Psimi.id.in_(entries))

    def interactions(self):
        """
        Returns an :class:`Query` object containing
        any interactions associated with this instance, or None if no
        interactions are associated.

        Returns
        -------
        :class:`Query` or None
            Query object containing pubmed entries or None if there are
            no rows returned.
        """
        entries = [
            r for rs in self.references().with_entities(
                Reference.interaction_id)
            for r in rs
        ]
        if not entries:
            return None
        return Interaction.query.filter(Interaction.id.in_(entries))

    def save(self, session=None, commit=False):
        """
        Save entry by adding it to a session.

        Adds this instance to a session object or default global session.
        If commit is passed in, the instance is commited and saved to the 
        database.

        Parameters
        ----------
        session : :class:`scoped_session`, optional.
            A session instance to save to. Leave as None to use the default
            session and save to the database located at `~/.pyppi/pyppi.db`

        commit : bool, default: False
            Commit attempts to save changes to the database, wrapped within
            an atomic transaction. If an error occurs, any changes will be
            rolledback.
        """
        if session is None:
            session = db_session
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise
        return self

    @validates('accession')
    def _validate_accession(self, key, value):
        return validate_accession(value, Pubmed, upper=True, check_exists=True)


class Psimi(Base):
    """
    PSIMI ontology schema definition. This is a basic table containing fields
    relating to a psi-mi experiment type term. It simple has an integer column,
    auto-generated and auto-incremented, an `accession` column representing
    the psi-mi accession number and `description` column, which is a plain
    text description.

    Parameters
    ----------
    id : int
        The integer primary key of the instance.

    accession : str
        Accession for this Pubmed instance.

    description : str, optional, default: None
        Plain text description of this Psi-mi entry.
    """
    __tablename__ = "psimi"

    id = Column(Integer, primary_key=True)
    accession = Column(String, unique=True, nullable=False)
    description = Column(String)

    def __init__(self, accession, description=None):
        self.accession = accession
        self.description = description

    def __repr__(self):
        string = "<Psimi(id={}, accession={}, desc={})>"
        return string.format(
            self.id, self.accession, self.description
        )

    def references(self):
        """
        Returns an :class:`Query` object containing
        any references associated with this instance. To execute the
        query call `all()`.

        Returns
        -------
        :class:`Query`
            Query object containing references.
        """
        return Reference.query.filter(
            Reference.psimi_id == self.id)

    def pmids(self):
        """
        Returns an :class:`Query` object containing
        any pmids associated with this instance, or None if no pmids
        are associated.

        Returns
        -------
        :class:`Query` or None
            Query object containing pubmed entries or None if there are
            no rows returned.
        """
        entries = [
            r for rs in self.references().with_entities(Reference.pubmed_id)
            for r in rs
        ]
        if not entries:
            return None
        return Pubmed.query.filter(Pubmed.id.in_(entries))

    def interactions(self):
        """
        Returns an :class:`Query` object containing
        any interactions associated with this instance, or None if no
        interactions are associated.

        Returns
        -------
        :class:`Query` or None
            Query object containing pubmed entries or None if there are
            no rows returned.
        """
        entries = [
            r for rs in self.references().with_entities(
                Reference.interaction_id)
            for r in rs
        ]
        if not entries:
            return None
        return Interaction.query.filter(Interaction.id.in_(entries))

    def save(self, session=None, commit=False):
        """
        Save entry by adding it to a session.

        Adds this instance to a session object or default global session.
        If commit is passed in, the instance is commited and saved to the 
        database.

        Parameters
        ----------
        session : :class:`scoped_session`, optional.
            A session instance to save to. Leave as None to use the default
            session and save to the database located at `~/.pyppi/pyppi.db`

        commit : bool, default: False
            Commit attempts to save changes to the database, wrapped within
            an atomic transaction. If an error occurs, any changes will be
            rolledback.
        """
        if session is None:
            session = db_session
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise
        return self

    @validates('accession')
    def _validate_accession(self, key, value):
        return validate_accession(value, Psimi, upper=True, check_exists=True)

    @validates('description')
    def _validate_description(self, key, value):
        return validate_description(value)


class Reference(Base):
    """
    A table to house references connecting interactions to a `Pubmed` id and
    a `psimi` experiment type annotation. There is a unique constraint 
    placed on all three columns `interaction_id`, `pubmed_id` and `psimi_id`.

    Parameters
    ----------
    interaction_id : int or :class:`Interaction`
        An integer id of an interaction instance or the instance itself.

    pubmed_id : int or :class:`Pubmed`
        An integer id of a pubmed instance or the instance itself.

    psimi_id : int or :class:`Psimi`
        An integer id of a psimi instance or the instance itself.

    Methods
    -------
    save(session=None, commit=False)
        Adds this instance to a session object or default global session.
        If commit is passed in, the instance is commited and saved to the 
        database.
    """

    __tablename__ = "reference"

    interaction_id = Column(
        Integer, ForeignKey('interaction.id'),
        primary_key=True, nullable=False
    )
    pubmed_id = Column(Integer, ForeignKey('pubmed.id'),
                       primary_key=True, nullable=False)
    psimi_id = Column(Integer, ForeignKey('psimi.id'),
                      primary_key=True, nullable=True)
    UniqueConstraint('interaction_id', 'pubmed_id', 'psimi_id')

    def __init__(self, interaction, pubmed, psimi):
        self.interaction_id = interaction
        self.pubmed_id = pubmed
        self.psimi_id = psimi

    def __repr__(self):
        return (
            "<Reference(interaction_id={}, pubmed_id={}, "
            "psimi_id={})>".format(
                self.interaction_id, self.pubmed_id, self.psimi_id
            ))

    def save(self, session=None, commit=False):
        """
        Save entry by adding it to a session.

        Adds this instance to a session object or default global session.
        If commit is passed in, the instance is commited and saved to the 
        database.

        Parameters
        ----------
        session : :class:`scoped_session`, optional.
            A session instance to save to. Leave as None to use the default
            session and save to the database located at `~/.pyppi/pyppi.db`

        commit : bool, default: False
            Commit attempts to save changes to the database, wrapped within
            an atomic transaction. If an error occurs, any changes will be
            rolledback.
        """
        if session is None:
            session = db_session
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise
        return self

    @validates('interaction_id')
    def _validate_interaction_id(self, key, value):
        if isinstance(value, Interaction):
            return value.id
        elif isinstance(value, (int, type(None))):
            return value
        else:
            raise TypeError(
                "Invalid type {} for `interaction_id`. Expected "
                "int or Interaction.".format(
                    type(value).__name__)
            )

    @validates('pubmed_id')
    def _validate_pubmed_id(self, key, value):
        if isinstance(value, Pubmed):
            return value.id
        elif isinstance(value, (int, type(None))):
            return value
        else:
            raise TypeError(
                "Invalid type {} for `pubmed_id`. Expected "
                "int or Pubmed.".format(
                    type(value).__name__)
            )

    @validates('psimi_id')
    def _validate_psimi_id(self, key, value):
        if isinstance(value, Psimi):
            return value.id
        elif isinstance(value, (int, type(None))):
            return value
        else:
            raise TypeError(
                "Invalid type {} for `psimi_id`. Expected "
                "int or Psimi.".format(
                    type(value).__name__)
            )
