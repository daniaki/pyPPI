import peewee
from typing import Iterable

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


__all__ = ["Protein"]


class Protein(BaseModel):
    identifier = peewee.ForeignKeyField(
        model=UniprotIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="proteins",
    )
    organism = peewee.IntegerField(
        null=False, default=None, help_text="Numeric organism code. Eg 9606."
    )
    sequence = peewee.TextField(
        null=False,
        default=None,
        help_text="The protein sequence in single letter format.",
    )
    reviewed = peewee.BooleanField(
        null=True,
        default=None,
        help_text="Protein has been reviewed (Swiss-prot).",
    )
    gene = peewee.ForeignKeyField(
        model=GeneSymbol,
        null=True,
        default=None,
        backref="proteins",
        help_text="Gene symbol related to this protein.",
    )

    # --- M2M --- #
    aliases = peewee.ManyToManyField(
        model=UniprotIdentifier, backref="alias_proteins"
    )
    go_annotations = peewee.ManyToManyField(
        model=GeneOntologyTerm, backref="proteins"
    )
    interpro_annotations = peewee.ManyToManyField(
        model=InterproTerm, backref="proteins"
    )
    pfam_annotations = peewee.ManyToManyField(
        model=PfamTerm, backref="proteins"
    )
    keywords = peewee.ManyToManyField(model=Keyword, backref="proteins")
    alt_genes = peewee.ManyToManyField(
        model=GeneSymbol, backref="proteins_alt"
    )

    def __str__(self):
        try:
            # Accessing the identifier field results in a database lookup.
            return str(self.identifier)
        except peewee.DoesNotExist:
            return str(None)

    def _select_go(self, category: str) -> peewee.ModelSelect:
        return (
            GeneOntologyTerm.select()
            .join(Protein.go_annotations.get_through_model())
            .join(Protein)
            .where(Protein.id == self.id)
            .select()
            .where(GeneOntologyTerm.category == category)
        )

    @classmethod
    def get_by_identifier(
        cls, identifiers: Iterable[str]
    ) -> peewee.ModelSelect:
        return (
            cls.select()
            .join(UniprotIdentifier)
            .where(
                peewee.fn.Upper(UniprotIdentifier.identifier)
                << set(i.upper() for i in identifiers)
            )
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
