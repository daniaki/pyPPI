import peewee
from typing import Iterable, Optional

from ...constants import GeneOntologyCategory
from ...utilities import is_null
from ...settings import DATABASE
from .base import BaseModel
from .identifiers import PsimiIdentifier, PubmedIdentifier, UniprotIdentifier
from .metadata import (
    GeneOntologyTerm,
    GeneSymbol,
    InterproTerm,
    Keyword,
    PfamTerm,
)


__all__ = ["Protein", "ProteinData"]


class ProteinData(BaseModel):
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
    version = peewee.CharField(
        null=False,
        default=None,
        max_length=16,
        help_text="UniProt record version.",
    )

    # --- M2M --- #
    genes = peewee.ManyToManyField(
        model=GeneSymbol, backref="protein_data_set"
    )
    identifiers = peewee.ManyToManyField(
        model=UniprotIdentifier, backref="protein_data_set"
    )
    go_annotations = peewee.ManyToManyField(
        model=GeneOntologyTerm, backref="protein_data_set"
    )
    interpro_annotations = peewee.ManyToManyField(
        model=InterproTerm, backref="protein_data_set"
    )
    pfam_annotations = peewee.ManyToManyField(
        model=PfamTerm, backref="protein_data_set"
    )
    keywords = peewee.ManyToManyField(
        model=Keyword, backref="protein_data_set"
    )

    def _select_go(self, category: str) -> peewee.ModelSelect:
        return (
            GeneOntologyTerm.select(ProteinData, GeneOntologyTerm)
            .join(ProteinData.go_annotations.get_through_model())
            .join(ProteinData)
            .where(ProteinData.id == self.id)
            .where(GeneOntologyTerm.category == category)
            .switch(ProteinData)
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
        return (
            query.select(cls, UniprotIdentifier)
            .join(cls.identifiers.get_through_model())
            .join(UniprotIdentifier)
            .where(peewee.fn.Lower(UniprotIdentifier.identifier).in_(unique))
            .switch(cls)
        )

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
        return super().format_for_save()


class Protein(BaseModel):
    identifier = peewee.ForeignKeyField(
        model=UniprotIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="proteins",
    )
    data = peewee.ForeignKeyField(
        model=ProteinData, null=False, default=None, backref="proteins"
    )

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