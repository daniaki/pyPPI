import peewee

from ...constants import GeneOntologyCategory
from ...utilities import is_null
from .base import BaseModel
from .identifiers import (
    GeneOntologyIdentifier,
    InterproIdentifier,
    KeywordIdentifier,
    PfamIdentifier,
)


__all__ = [
    "Annotation",
    "GeneOntologyTerm",
    "InterproTerm",
    "PfamTerm",
    "Keyword",
    "GeneSymbol",
]


class Annotation(BaseModel):
    """
    Inherited by models which represent an annotation, with
    an identifier, name and description. For example GO 
    annotations.
    """

    identifier = None

    name = peewee.TextField(
        null=True, default=None, help_text="Name of the annotation."
    )
    description = peewee.TextField(
        null=True, default=None, help_text="Long form description of a term."
    )

    def __str__(self):
        try:
            return str(self.identifier)
        except peewee.DoesNotExist:
            return str(None)

    def save(self, *args, **kwargs):
        self.name = None if is_null(self.name) else self.name
        self.description = (
            None if is_null(self.description) else self.description
        )
        return super().save(*args, **kwargs)


class GeneOntologyTerm(Annotation):
    identifier = peewee.ForeignKeyField(
        model=GeneOntologyIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Unique identifier for this term.",
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


class InterproTerm(Annotation):
    identifier = peewee.ForeignKeyField(
        model=InterproIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term.",
    )
    entry_type = peewee.CharField(
        null=True,
        default=None,
        help_text="Interpro entry type.",
        max_length=32,
    )

    def save(self, *args, **kwargs):
        self.entry_type = (
            None
            if is_null(self.entry_type)
            else self.entry_type.strip().capitalize()
        )
        return super().save(*args, **kwargs)


class PfamTerm(Annotation):
    identifier = peewee.ForeignKeyField(
        model=PfamIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term.",
    )


class Keyword(Annotation):
    identifier = peewee.ForeignKeyField(
        model=KeywordIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term. Has form 'KW-d+'",
    )


class GeneSymbol(BaseModel):
    text = peewee.CharField(
        null=False, default=None, unique=True, help_text="HGNC gene symbol."
    )

    def __str__(self):
        return str(self.text)

    def save(self, *args, **kwargs):
        self.text = None if is_null(self.text) else self.text.strip().upper()
        return super().save(*args, **kwargs)
