import peewee

from ...constants import GeneOntologyCategory
from .base import BaseModel
from .identifiers import (
    GeneOntologyIdentifier,
    InterproIdentifier,
    KeywordIdentifier,
    PfamIdentifier,
)


__all__ = [
    "AnnotationMixin",
    "GeneOntologyTerm",
    "InterproTerm",
    "PfamTerm",
    "Keyword",
    "GeneSymbol",
]


class AnnotationMixin:
    def to_str(self) -> str:
        return getattr(self, "identifier").identifier


class GeneOntologyTerm(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=GeneOntologyIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Unique identifier for this term.",
    )
    name = peewee.TextField(
        null=False, default=None, help_text="GO term name."
    )
    description = peewee.TextField(
        null=True, default=None, help_text="GO term description."
    )
    category = peewee.CharField(
        null=False,
        default=None,
        help_text="The GO category that the term belongs to.",
    )
    obsolete = peewee.BooleanField(
        null=True,
        default=None,
        help_text="Term is obsolete according to the GO.",
    )

    def save(self, *args, **kwargs):
        self.category = self.category.strip().capitalize()
        if len(self.category) == 1:
            self.category = GeneOntologyCategory.letter_to_category(
                self.category
            )
        if self.category not in set(GeneOntologyCategory.list()):
            raise ValueError(
                f"'{self.category}' is not a supported category. "
                f"Supported categories are {GeneOntologyCategory.list()}"
            )
        return super().save(*args, **kwargs)


class InterproTerm(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=InterproIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term.",
    )
    name = peewee.TextField(
        null=False, default=None, help_text="Short name description of a term."
    )
    entry_type = peewee.CharField(
        null=True,
        default=None,
        help_text="Interpro entry type.",
        max_length=32,
    )


class PfamTerm(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=PfamIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term.",
    )
    name = peewee.TextField(
        null=False, default=None, help_text="Short name description of a term."
    )
    description = peewee.TextField(
        null=False, default=None, help_text="Long form description of a term."
    )


class Keyword(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=KeywordIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term. Has form 'KW-d+'",
    )
    description = peewee.TextField(
        null=False, default=None, help_text="Keyword description."
    )


class GeneSymbol(BaseModel):
    text = peewee.CharField(
        null=False, default=None, unique=True, help_text="HGNC gene symbol."
    )

    def save(self, *args, **kwargs):
        self.text = self.text.strip().upper()
        return super().save(*args, **kwargs)
