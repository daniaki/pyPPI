import peewee

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
    gene = peewee.ForeignKeyField(
        model=GeneSymbol,
        null=True,
        default=None,
        backref="proteins",
        help_text="Gene symbol related to this protein.",
    )
    alt_genes = peewee.ManyToManyField(
        model=GeneSymbol, backref="proteins_alt"
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

    def __str__(self):
        try:
            return str(self.identifier)
        except peewee.DoesNotExist:
            return str(None)

    def _select_go(self, category: str):
        return (
            GeneOntologyTerm.select()
            .join(Protein.go_annotations.get_through_model())
            .join(Protein)
            .where(Protein.id == self.id)
            .select()
            .where(GeneOntologyTerm.category == category)
        )

    @property
    def go_mf(self):
        return self._select_go(GeneOntologyCategory.molecular_function)

    @property
    def go_bp(self):
        return self._select_go(GeneOntologyCategory.biological_process)

    @property
    def go_cc(self):
        return self._select_go(GeneOntologyCategory.cellular_component)

    def save(self, *args, **kwargs):
        self.sequence = (
            None if is_null(self.sequence) else self.sequence.strip().upper()
        )
        return super().save(*args, **kwargs)
