from typing import Iterable, Optional, Callable, Any

import peewee

from ... import validators, utilities
from .base import BaseModel


__all__ = [
    "ExternalIdentifier",
    "GeneOntologyIdentifier",
    "PfamIdentifier",
    "InterproIdentifier",
    "KeywordIdentifier",
    "PubmedIdentifier",
    "PsimiIdentifier",
    "UniprotIdentifier",
]


class ExternalIdentifier(BaseModel):
    # Database name of the identifier.
    DB_NAME: Optional[str] = None

    identifier = peewee.CharField(
        null=False,
        default=None,
        unique=True,
        max_length=32,
        help_text="The unique identifier from an external database.",
    )
    dbname = peewee.CharField(
        null=False,
        default=None,
        max_length=16,
        help_text="The identifier's database name.",
    )

    def __str__(self):
        return str(self.identifier)

    @classmethod
    def get_by_identifier(
        cls, identifiers: Iterable[str]
    ) -> peewee.ModelSelect:
        """Select identifiers by string accession values."""
        return cls.select().where(
            peewee.fn.Upper(cls.identifier)
            << set(i.upper() for i in identifiers)
        )

    @classmethod
    def format(cls, identifier: str) -> str:
        """
        How to format identifier. Called after prefix is performed and must
        accept a single input and return a single string.
        """
        if utilities.is_null(identifier):
            raise ValueError(f"'identifier' cannot be null.")
        return identifier.strip().upper()

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        """
        How to validate identifier. Called after formatter is called and must
        accept a single input and return a single boolean.
        """
        raise NotImplementedError()

    @classmethod
    def validate(cls, identifier: str) -> str:
        if utilities.is_null(identifier):
            raise ValueError(f"'identifier' cannot be null.")
        if not cls.is_valid(identifier):
            """Validate an accession is valid, or raise a `ValueError`."""
            raise ValueError(
                f"'{identifier}' is not a valid {cls.__name__}' identifier."
            )
        return identifier

    def format_for_save(self):
        self.dbname = self.DB_NAME
        if self.DB_NAME is None:
            raise NotImplementedError("Concrete table must define DB_NAME.")
        self.identifier = self.validate(self.format(self.identifier))
        return super().format_for_save()


class GeneOntologyIdentifier(ExternalIdentifier):
    DB_NAME = "Gene Ontology"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_go(identifier) is not None


class PubmedIdentifier(ExternalIdentifier):
    DB_NAME = "PubMed"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_pubmed(identifier) is not None


class PsimiIdentifier(ExternalIdentifier):
    DB_NAME = "MI Ontology"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_psimi(identifier) is not None


class UniprotIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_uniprot(identifier) is not None


class KeywordIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt KW"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_keyword(identifier) is not None


class InterproIdentifier(ExternalIdentifier):
    DB_NAME = "InterPro"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_interpro(identifier) is not None


class PfamIdentifier(ExternalIdentifier):
    DB_NAME = "PFAM"

    @classmethod
    def is_valid(cls, identifier: str) -> bool:
        return validators.is_pfam(identifier) is not None
