
"""
This module contains the SQLAlchemy database definitions and related
functions to access and update the database files.
"""
from enum import Enum

from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship

from ..database import Base
from ..database.exceptions import ObjectNotFound, ObjectAlreadyExists


def _format_annotation(value, upper=True):
    if upper:
        return str(value).strip().upper()
    else:
        return str(value).strip()


def _format_annotations(values, upper=True, allow_duplicates=False):
    if isinstance(values, str):
        values = values.split(',')

    if values is None:
        return None

    if not allow_duplicates:
        return sorted(set(
            _format_annotation(value, upper) for value in values
            if _format_annotation(value, upper)
        ))
    else:
        return sorted(
            _format_annotation(value, upper) for value in values
            if _format_annotation(value, upper)
        )


def _check_annotations(values, dbtype=None):
    values = _format_annotations(values)

    if not dbtype:
        return
    if not values:
        return

    elif dbtype == "GO":
        all_valid = all(["GO:" in v for v in values])
    elif dbtype == "IPR":
        all_valid = all(["IPR" in v for v in values])
    elif dbtype == "PF":
        all_valid = all(["PF" in v for v in values])
    else:
        raise ValueError("Unrecognised dbtype '{}'".format(dbtype))

    if not all_valid:
        raise ValueError(
            "Annotations contain invalid values for database type {}".format(
                dbtype
            )
        )


pmid_interactions = Table('pmid_interactions', Base.metadata,
    Column('interaction_id', Integer, ForeignKey('interaction.id')),
    Column('pubmed_id', Integer, ForeignKey('pubmed.id'))
)


psimi_interactions = Table('psimi_interactions', Base.metadata,
    Column('interaction_id', Integer, ForeignKey('interaction.id')),
    Column('psimi_id', Integer, ForeignKey('psimi.id'))
)


class Protein(Base):
    """
    Protein schema definition. This is a basic table housing selected fields
    from the uniprot database dump files.
    """
    __tablename__ = "protein"

    id = Column('id', Integer, primary_key=True)
    uniprot_id = Column('uniprot_id', String, nullable=False, unique=True)
    taxon_id = Column('taxon_id', Integer, nullable=False)
    gene_id = Column('gene_id', String)
    _go_mf = Column('go_mf', String)
    _go_cc = Column('go_cc', String)
    _go_bp = Column('go_bp', String)
    _interpro = Column('interpro', String)
    _pfam = Column('pfam', String)
    _keywords = Column("keywords", String)
    reviewed = Column(Boolean, nullable=False)

    interactions = relationship(
        "Interaction", backref="protein",
        primaryjoin="or_("
        "Protein.id==Interaction.source, "
        "Protein.id==Interaction.target"
        ")"
    )

    error_msg = (
        "Expected a comma delimited string, list or set "
        "for argument {arg}. Found {type}."
    )

    def __init__(self, uniprot_id=None, gene_id=None, taxon_id=None,
                 go_mf=None, go_cc=None, go_bp=None, interpro=None, pfam=None,
                 keywords=None, reviewed=None):
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

    def __repr__(self):
        string = ("<Protein(id={}, uniprot_id={}, gene_id={}, taxon_id={})>")
        return string.format(
            self.id, self.uniprot_id, self.gene_id, self.taxon_id
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

    @staticmethod
    def columns():
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
        return Columns

    def save(self, session, commit=False):
        try:
            _check_annotations(self.go_bp, dbtype='GO')
            _check_annotations(self.go_cc, dbtype='GO')
            _check_annotations(self.go_mf, dbtype='GO')

            _check_annotations(self.interpro, dbtype='IPR')
            _check_annotations(self.pfam, dbtype='PF')

            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise

    @property
    def go_mf(self):
        return self._go_mf

    @go_mf.setter
    def go_mf(self, value):
        if not value:
            self._go_mf = None
        else:
            self._set_annotation_attribute("_go_mf", value)

    @property
    def go_cc(self):
        return self._go_cc

    @go_cc.setter
    def go_cc(self, value):
        if not value:
            self._go_cc = None
        else:
            self._set_annotation_attribute("_go_cc", value)

    @property
    def go_bp(self):
        return self._go_bp

    @go_bp.setter
    def go_bp(self, value):
        if not value:
            self._go_bp = None
        else:
            self._set_annotation_attribute("_go_bp", value)

    @property
    def interpro(self):
        return self._interpro

    @interpro.setter
    def interpro(self, value):
        if not value:
            self._interpro = None
        else:
            self._set_annotation_attribute("_interpro", value)

    @property
    def pfam(self):
        return self._pfam

    @pfam.setter
    def pfam(self, value):
        if not value:
            self._pfam = None
        else:
            self._set_annotation_attribute("_pfam", value)

    @property
    def keywords(self):
        return self._keywords

    @keywords.setter
    def keywords(self, value):
        if not value:
            self._keywords = None
        else:
            self._set_annotation_attribute("_keywords", value)
            if self._keywords is not None:
                self._keywords = ','.join(
                    [x.capitalize() for x in self.keywords.split(',')]
                )

    def _set_annotation_attribute(self, attr, value):
        accepted_types = [
            isinstance(value, str),
            isinstance(value, list),
            isinstance(value, set)
        ]
        if not any(accepted_types):
            raise TypeError(self.error_msg.format(
                arg=attr, type=type(value).__name__))
        else:
            value = _format_annotations(value, allow_duplicates=False)
            if not value:
                value = None
            else:
                setattr(self, attr, ','.join(value))


class Interaction(Base):
    """
    PPI schema definition. This is a basic table containing fields
    such as computed features.
    """
    __tablename__ = "interaction"

    id = Column(Integer, primary_key=True)
    is_training = Column(Boolean, nullable=False)
    is_holdout = Column(Boolean, nullable=False)
    is_interactome = Column(Boolean, nullable=False)
    source = Column(ForeignKey("protein.id"), nullable=False)
    target = Column(ForeignKey("protein.id"), nullable=False)
    combined = Column(String, unique=True, nullable=False)
    taxon_id = Column('taxon_id', Integer, nullable=False)
    _label = Column('label', String)
    _keywords = Column('keywords', String)
    _go_mf = Column('go_mf', String)
    _go_cc = Column('go_cc', String)
    _go_bp = Column('go_bp', String)
    _ulca_go_mf = Column('ulca_go_mf', String)
    _ulca_go_cc = Column('ulca_go_cc', String)
    _ulca_go_bp = Column('ulca_go_bp', String)
    _interpro = Column('interpro', String)
    _pfam = Column('pfam', String)

    # M-2-O relationships
    pmid = relationship(
        "Pubmed", backref="pmid_interactions", 
        uselist=True, secondary=pmid_interactions, lazy='joined'
    )
    psimi = relationship(
        "Psimi", backref="psimi_interactions", 
        uselist=True, secondary=psimi_interactions, lazy='joined'
    )

    def __init__(self, source=None, target=None, is_interactome=None,
                 is_training=None, is_holdout=None, label=None,
                 go_mf=None, go_cc=None, go_bp=None,
                 ulca_go_mf=None, ulca_go_cc=None, ulca_go_bp=None,
                 interpro=None, pfam=None, keywords=None):

        self.source = source
        self.target = target
        self.label = label
        self.is_training = is_training
        self.is_holdout = is_holdout
        self.is_interactome = is_interactome

        # Create a unique identifier using the uniprot ids combined
        # into a string. This column is unique causing a contraint
        # failure if an (B, A) is added when (A, B) already exists.
        self.combined = ','.join(
            _format_annotations(
                [source, target],
                allow_duplicates=True,
                upper=True
            )
        )

        self.keywords = keywords
        self.go_mf = go_mf
        self.go_cc = go_cc
        self.go_bp = go_bp
        self.ulca_go_mf = ulca_go_mf
        self.ulca_go_cc = ulca_go_cc
        self.ulca_go_bp = ulca_go_bp
        self.interpro = interpro
        self.pfam = pfam

    def __repr__(self):
        from ..database import begin_transaction
        with begin_transaction() as session:
            source_uid = session.query(Protein).get(self.source)
            target_uid = session.query(Protein).get(self.target)
        string = (
            "<Interaction("
            "id={}, source={} ({}), target={} ({}), training={}, holdout={}, "
            "interactome={}, label={}"
            ")"
            ">"
        )
        return string.format(
            self.id, self.source, None or source_uid.uniprot_id,
            self.target, None or target_uid.uniprot_id,
            self.is_training, self.is_holdout,
            self.is_interactome, self.label
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

    @staticmethod
    def columns():
        class Columns(Enum):
            ID = 'id'
            GO_MF = 'go_mf'
            GO_BP = 'go_bp'
            GO_CC = 'go_cc'
            ULCA_GO_MF = 'ulca_go_mf'
            ULCA_GO_BP = 'ulca_go_bp'
            ULCA_GO_CC = 'ulca_go_cc'
            PFAM = 'pfam'
            INTERPRO = 'interpro'
            LABEL = 'label'
            KW = 'keywords'
            COMBINED = 'combined'
            IS_HOLDOUT = 'is_holdout'
            IS_TRAINING = 'is_training'
            IS_INTERACTOME = 'is_interactome'
        return Columns

    def save(self, session, commit=False):
        try:
            _check_annotations(self.go_bp, dbtype='GO')
            _check_annotations(self.go_cc, dbtype='GO')
            _check_annotations(self.go_mf, dbtype='GO')

            _check_annotations(self.ulca_go_bp, dbtype='GO')
            _check_annotations(self.ulca_go_cc, dbtype='GO')
            _check_annotations(self.ulca_go_mf, dbtype='GO')

            _check_annotations(self.interpro, dbtype='IPR')
            _check_annotations(self.pfam, dbtype='PF')

            if (self.is_holdout or self.is_training) and not self.label:
                raise ValueError("Training/Holdout interaction must be labled")

            if not isinstance(self.is_holdout, bool):
                raise TypeError(
                    "is_holdout must be a boolean value."
                )
            if not isinstance(self.is_training, bool):
                raise TypeError(
                    "is_training must be a boolean value."
                )
            if not isinstance(self.is_interactome, bool):
                raise TypeError(
                    "is_interactome must be a boolean value."
                )

            invalid_source = session.query(Protein).filter_by(
                id=self.source
            ).count() == 0
            invalid_target = session.query(Protein).filter_by(
                id=self.target
            ).count() == 0
            already_exists = session.query(Interaction).filter_by(
                combined=self.combined
            ).count() != 0

            if self.source and invalid_source:
                raise ObjectNotFound(
                    "Source '{}' does not exist in table 'protein'.".format(
                        self.source
                    )
                )
            if self.target and invalid_target:
                raise ObjectNotFound(
                    "Target '{}' does not exist in table 'protein'.".format(
                        self.target
                    )
                )

            if self.source and self.target:
                source = session.query(Protein).get(self.source)
                target = session.query(Protein).get(self.target)
                if already_exists:
                    raise ObjectAlreadyExists(
                        "Interaction ({}, {}) already exists.".format(
                            source.uniprot_id, target.uniprot_id)
                    )
                same_taxon = (source.taxon_id == target.taxon_id)
                if not same_taxon:
                    raise ValueError(
                        "Source and target must have the same taxonomy ids. "
                    )
                else:
                    self.taxon_id = source.taxon_id

            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise

    @property
    def keywords(self):
        return self._keywords

    @keywords.setter
    def keywords(self, value):
        if not value:
            self._keywords = None
        else:
            self._set_annotation_attribute("_keywords", value)
            if self._keywords is not None:
                self._keywords = ','.join(
                    [x.capitalize() for x in self.keywords.split(',')]
                )

    @property
    def go_mf(self):
        return self._go_mf

    @go_mf.setter
    def go_mf(self, value):
        if not value:
            self._go_mf = None
        else:
            self._set_annotation_attribute("_go_mf", value)

    @property
    def go_cc(self):
        return self._go_cc

    @go_cc.setter
    def go_cc(self, value):
        if not value:
            self._go_cc = None
        else:
            self._set_annotation_attribute("_go_cc", value)

    @property
    def go_bp(self):
        return self._go_bp

    @go_bp.setter
    def go_bp(self, value):
        if not value:
            self._go_bp = None
        else:
            self._set_annotation_attribute("_go_bp", value)

    @property
    def ulca_go_mf(self):
        return self._ulca_go_mf

    @ulca_go_mf.setter
    def ulca_go_mf(self, value):
        if not value:
            self._ulca_go_mf = None
        else:
            self._set_annotation_attribute("_ulca_go_mf", value)

    @property
    def ulca_go_cc(self):
        return self._ulca_go_cc

    @ulca_go_cc.setter
    def ulca_go_cc(self, value):
        if not value:
            self._ulca_go_cc = None
        else:
            self._set_annotation_attribute("_ulca_go_cc", value)

    @property
    def ulca_go_bp(self):
        return self._ulca_go_bp

    @ulca_go_bp.setter
    def ulca_go_bp(self, value):
        if not value:
            self._ulca_go_bp = None
        else:
            self._set_annotation_attribute("_ulca_go_bp", value)

    @property
    def interpro(self):
        return self._interpro

    @interpro.setter
    def interpro(self, value):
        if not value:
            self._interpro = None
        else:
            self._set_annotation_attribute("_interpro", value)

    @property
    def pfam(self):
        return self._pfam

    @pfam.setter
    def pfam(self, value):
        if not value:
            self._pfam = None
        else:
            self._set_annotation_attribute("_pfam", value)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        valid_types = [
            isinstance(value, str),
            isinstance(value, list),
            isinstance(value, set),
            isinstance(value, type(None))
        ]
        if not any(valid_types):
            raise TypeError("Label must be list, str, set or None.")

        if not value:
            self._label = None
        else:
            if isinstance(value, str):
                value = value.split(',')
            labels = ','.join(sorted(
                set(v.strip().capitalize() for v in value if v.strip())
            ))
            if not labels:
                self._label = None
            else:
                self._label = labels

    def _set_annotation_attribute(self, attr, value):
        accepted_types = [
            isinstance(value, str),
            isinstance(value, list),
            isinstance(value, set)
        ]
        if not any(accepted_types):
            raise TypeError(self.error_msg.format(
                arg=attr, type=type(value).__name__)
            )
        else:
            value = _format_annotations(value, allow_duplicates=True)
            if not value:
                value = None
            else:
                setattr(self, attr, ','.join(value))

    @property
    def has_missing_data(self):
        return any([
            not bool(self.go_bp),
            not bool(self.go_mf),
            not bool(self.go_cc),
            not bool(self.ulca_go_bp),
            not bool(self.ulca_go_cc),
            not bool(self.ulca_go_mf),
            not bool(self.keywords),
            not bool(self.interpro),
            not bool(self.pfam)
        ])


class Pubmed(Base):
    """
    Pubmed schema definition. This is a basic table containing fields
    relating to a pubmed id. It simple has an integer column, auto-generated
    and auto-incremented, and an `accession` column representing the pubmed
    accession number.
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

    def save(self, session, commit=False):
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise

    @property
    def interactions(self):
        return self.pmid_interactions


class Psimi(Base):
    """
    PSIMI ontology schema definition. This is a basic table containing fields
    relating to a psi-mi experiment type term. It simple has an integer column, 
    auto-generated and auto-incremented, an `accession` column representing 
    the psi-mi accession number and `description` column, which is a plain
    text description.
    """
    __tablename__ = "psimi"

    id = Column(Integer, primary_key=True)
    accession = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=False)

    def __init__(self, accession, description):
        self.accession = accession
        self.description = description

    def __repr__(self):
        string = "<Psimi(id={}, accession={}, desc={})>"
        return string.format(
            self.id, self.accession, self.description
        )

    def save(self, session, commit=False):
        try:
            session.add(self)
            if commit:
                session.commit()
        except:
            session.rollback()
            raise

    @property
    def interactions(self):
        return self.psimi_interactions
