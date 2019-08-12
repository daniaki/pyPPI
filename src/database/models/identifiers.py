from typing import Iterable, Optional

import peewee

from .base import BaseModel


__all__ = [
    "IdentifierMixin",
    "ExternalIdentifier",
    "GeneOntologyIdentifier",
    "PfamIdentifier",
    "InterproIdentifier",
    "KeywordIdentifier",
    "PubmedIdentifier",
    "PsimiIdentifier",
    "UniprotIdentifier",
]


class IdentifierMixin:
    def _get_identifier(self) -> str:
        identifier: Optional[str] = getattr(self, "identifier", None)
        if not getattr(self, "identifier", None):
            raise AttributeError(
                f"{self.__class__.__name__} is missing attribute 'identifier'."
            )
        if not isinstance(identifier, str):
            klass = type(identifier).__name__
            raise TypeError(
                f"Expected 'identifier' to be 'str'. Found '{klass}'"
            )
        return identifier

    def prefix(self, prefix: str, sep: str = ":") -> str:
        identifier = self._get_identifier()
        if not identifier.lower().startswith(f"{prefix.lower()}{sep}"):
            return f"{prefix}{sep}{identifier}"
        return identifier

    def unprefix(self, sep: str = ":") -> str:
        return self._get_identifier().split(sep)[-1]


class ExternalIdentifier(IdentifierMixin, BaseModel):
    # Database name of the identifier.
    DB_NAME: Optional[str] = None
    # Prefix for appending to an identifier if missing. For example the
    # GO in GO:<accession>.
    PREFIX: Optional[str] = None
    # How to separate identifier and prefix. For example the ':' between
    # GO and <accession>.
    SEP: Optional[str] = ":"

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

    def save(self, *args, **kwargs):
        if self.DB_NAME is None:
            raise NotImplementedError("Concrete table must define DB_NAME.")
        if self.PREFIX:
            self.identifier = self.prefix(self.PREFIX, self.SEP)
        self.identifier = self.identifier.upper()
        self.dbname = self.DB_NAME
        return super().save(*args, **kwargs)


class GeneOntologyIdentifier(ExternalIdentifier):
    DB_NAME = "Gene Ontology"
    PREFIX = "GO"
    SEP = ":"


class PubmedIdentifier(ExternalIdentifier):
    DB_NAME = "PubMed"
    PREFIX = "pubmed"
    SEP = ":"


class PsimiIdentifier(ExternalIdentifier):
    DB_NAME = "Psimi"
    PREFIX = "MI"
    SEP = ":"


class UniprotIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt"
    PREFIX = None
    SEP = None


class KeywordIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt KW"
    PREFIX = "KW"
    SEP = "-"


class InterproIdentifier(ExternalIdentifier):
    DB_NAME = "InterPro"
    PREFIX = None
    SEP = None


class PfamIdentifier(ExternalIdentifier):
    DB_NAME = "PFAM"
    PREFIX = None
    SEP = None
