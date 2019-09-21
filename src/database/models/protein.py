import peewee
from hashlib import md5
from typing import Iterable, Optional

from ...constants import GeneOntologyCategory
from ...utilities import is_null
from ...settings import DATABASE
from .base import BaseModel, ForeignKeyConstraint
from .identifiers import PsimiIdentifier, PubmedIdentifier, UniprotIdentifier
from .metadata import (
    GeneOntologyTerm,
    GeneSymbol,
    InterproTerm,
    Keyword,
    PfamTerm,
)


__all__ = [
    "Protein",
    "UniprotRecord",
    "UniprotRecordIdentifier",
    "UniprotRecordGene",
]


class UniprotRecord(BaseModel):
    organism = peewee.IntegerField(
        null=False, default=None, help_text="NCBI organism code. Eg 9606."
    )
    sequence = peewee.TextField(
        null=False,
        default=None,
        help_text="The protein sequence in single letter format.",
    )
    reviewed = peewee.BooleanField(
        null=False,
        default=None,
        help_text="Protein has been reviewed (Swiss-prot).",
    )
    version = peewee.IntegerField(
        null=False, default=None, help_text="UniProt record version."
    )
    data_hash = peewee.CharField(
        null=False,
        unique=True,
        default=None,
        max_length=32,
        help_text="Unique hash of data fields to prevent duplicate entries.",
    )

    # --- M2M --- #
    go_annotations = peewee.ManyToManyField(
        model=GeneOntologyTerm, backref="uniprot_records"
    )
    interpro_annotations = peewee.ManyToManyField(
        model=InterproTerm, backref="uniprot_records"
    )
    pfam_annotations = peewee.ManyToManyField(
        model=PfamTerm, backref="uniprot_records"
    )
    keywords = peewee.ManyToManyField(model=Keyword, backref="uniprot_records")

    def __str__(self):
        return f"{self.primary_identifier}"

    def _select_go(self, category: str) -> peewee.ModelSelect:
        klass = UniprotRecord
        return (
            GeneOntologyTerm.select(klass, GeneOntologyTerm)
            .join(klass.go_annotations.get_through_model())
            .join(klass)
            .where(klass.id == self.id)
            .where(GeneOntologyTerm.category == category)
            .switch(klass)
        )

    @classmethod
    def get_by_identifier(
        cls,
        identifiers: Iterable[str],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Search protein data rows by UniProt identifier. This Will return all 
        rows that have primary or alias identifiers matching any given in 
        `identifiers`.
        
        Parameters
        ----------
        identifiers : Iterable[str]
            Uniprot accession strings.
        
        query : Optional[ModelSelect], optional
            Query to filter. Defaults to all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        query = query or cls.all()
        unique = set(i.lower() for i in identifiers)
        result: peewee.ModelSelect = (
            query.select(cls, UniprotRecordIdentifier)
            .join(
                UniprotRecordIdentifier,
                on=(cls.id == UniprotRecordIdentifier.record),
            )
            .join(
                UniprotIdentifier,
                on=(
                    UniprotRecordIdentifier.identifier == UniprotIdentifier.id
                ),
            )
            .where(peewee.fn.Lower(UniprotIdentifier.identifier).in_(unique))
            .switch(cls)
        )
        return result.distinct()

    @property
    def primary_identifier(self) -> Optional["UniprotRecord"]:
        return self.identifiers.filter(primary=True).first()

    @property
    def go_mf(self) -> peewee.ModelSelect:
        return self._select_go(GeneOntologyCategory.molecular_function)

    @property
    def go_bp(self) -> peewee.ModelSelect:
        return self._select_go(GeneOntologyCategory.biological_process)

    @property
    def go_cc(self) -> peewee.ModelSelect:
        return self._select_go(GeneOntologyCategory.cellular_component)

    def format_for_save(self):
        self.sequence = (
            None if is_null(self.sequence) else self.sequence.strip().upper()
        )
        self.data_hash = md5(
            str(
                (self.organism, self.sequence, self.reviewed, self.version)
            ).encode("utf-8")
        ).hexdigest()
        return super().format_for_save()


class UniprotRecordIdentifier(BaseModel):
    record = peewee.ForeignKeyField(
        model=UniprotRecord,
        null=False,
        default=None,
        backref="identifiers",
        on_delete=ForeignKeyConstraint.CASCADE,
        help_text="Related protein data instance.",
    )
    identifier = peewee.ForeignKeyField(
        model=UniprotIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="uniprot_record_relations",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text="Stable UniProt accession for data entry.",
    )
    primary = peewee.BooleanField(
        null=False,
        default=None,
        help_text=(
            "UniProt identifier relation to data entry (primary or secondary)."
        ),
    )

    def __str__(self):
        primary = "primary" if self.primary else "alternative"
        try:
            # Accessing the identifier field results in a database lookup.
            return f"{self.identifier} {primary}"
        except peewee.DoesNotExist:
            return f"{None} {primary}"

    class Meta:
        indexes = (
            # Unique index
            (("identifier_id", "record_id"), True),
        )

    def format_for_save(self):
        record_identifier: UniprotRecordIdentifier
        for record_identifier in self.record.identifiers:
            if (
                self.primary
                and record_identifier.primary
                and record_identifier.id != self.id
            ):
                raise ValueError(
                    f"Uniprot record with id '{self.record.id}' already has a "
                    f"primary identifier '{record_identifier.identifier}'. "
                    f"Each instance can have at most one primary identifier."
                )
        return super().format_for_save()


class UniprotRecordGene(BaseModel):
    RELATION_CHOICES = ("primary", "synonym", "orf")
    PRIMARY = RELATION_CHOICES[0]

    record = peewee.ForeignKeyField(
        model=UniprotRecord,
        null=False,
        default=None,
        backref="genes",
        on_delete=ForeignKeyConstraint.CASCADE,
        help_text="Related protein data instance.",
    )
    gene = peewee.ForeignKeyField(
        model=GeneSymbol,
        null=False,
        default=None,
        backref="uniprot_record_relations",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text="Related gene symbol instance.",
    )
    primary = peewee.BooleanField(
        null=False, default=None, help_text="Gene is primary for a data entry."
    )
    relation = peewee.CharField(
        null=False,
        default=None,
        max_length=32,
        help_text=(
            "Gene relation to protein data entry (primary, secondary, orf etc)."
        ),
    )

    def __str__(self):
        try:
            return f"{self.gene} {self.relation}"
        except peewee.DoesNotExist:
            return f"{None} {self.relation}"

    class Meta:
        indexes = (
            # Unique index
            (("gene_id", "record_id"), True),
        )

    def format_for_save(self):
        self.relation = (
            None if is_null(self.relation) else self.relation.lower()
        )
        if self.relation not in self.RELATION_CHOICES:
            choices = "'{}'".format("', '".join(self.RELATION_CHOICES))
            raise ValueError(
                f"'{self.relation}' is not a valid relation. Relation must be "
                f"one of {choices}."
            )
        self.primary = self.relation == self.PRIMARY

        # Check only one primary gene and gene symbol doesn't exist as
        # a primary
        relation: UniprotRecordGene
        for relation in self.record.genes:
            if self.primary and relation.primary and relation.id != self.id:
                raise ValueError(
                    f"Uniprot record with id '{self.record.id}' already has a "
                    f"primary gene '{relation.gene}'. Each instance "
                    f"can have at most one primary gene."
                )
        return super().format_for_save()


class Protein(BaseModel):
    identifier = peewee.ForeignKeyField(
        model=UniprotIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="proteins",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text="Stable UniProt accession for data entry.",
    )
    record = peewee.ForeignKeyField(
        model=UniprotRecord, 
        null=False, 
        default=None, 
        backref="proteins",
        on_delete=ForeignKeyConstraint.RESTRICT,
    )

    class Meta:
        # Unique index
        indexes = ((("identifier_id", "record_id"), True),)

    def __str__(self):
        try:
            # Accessing the identifier field results in a database lookup.
            return str(self.identifier)
        except peewee.DoesNotExist:
            return str(None)

    @classmethod
    def get_by_identifier(
        cls,
        identifiers: Iterable[str],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Search protein rows by UniProt identifier. This Will return all 
        rows that have an identifier matching any given in `identifiers`.
        
        Parameters
        ----------
        identifiers : Iterable[str]
            Uniprot accession strings.
        
        query : Optional[ModelSelect], optional
            Query to filter. Defaults to all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        query = query or cls.all()
        unique = set(i.lower() for i in identifiers)
        return (
            query.select(cls, UniprotIdentifier)
            .join(UniprotIdentifier)
            .where(peewee.fn.Lower(UniprotIdentifier.identifier).in_(unique))
            .switch(cls)
        )
