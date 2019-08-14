from typing import Iterable, Optional, Callable

import peewee

from ... import validators
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

    def prefix(
        self, prefix: Optional[str] = None, sep: Optional[str] = None
    ) -> str:
        identifier = self._get_identifier()
        if not prefix:
            return identifier
        if not identifier.lower().startswith(f"{prefix.lower()}"):
            if sep:
                return f"{prefix}{sep}{identifier}"
            return f"{prefix}{identifier}"
        return identifier

    def unprefix(
        self, prefix: Optional[str] = None, sep: Optional[str] = None
    ) -> str:
        identifier = self._get_identifier()
        if not prefix:
            return identifier
        skip_to = len(prefix) + len(str(sep or ""))
        return self._get_identifier()[skip_to:]


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

    def __str__(self):
        return str(self.identifier)

    def format(self) -> str:
        """
        How to format identifier. Called after prefix is performed and must
        accept a single input and return a single string.
        """
        return str.upper(self.identifier)

    def validate(self) -> str:
        """
        How to validate identifier. Called after formatter is called and must
        accept a single input and return a single boolean.
        """
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        if self.DB_NAME is None:
            raise NotImplementedError("Concrete table must define DB_NAME.")

        if self.PREFIX:
            self.identifier = self.prefix(self.PREFIX, self.SEP)

        self.identifier = self.format()
        if not self.validate():
            raise ValueError(f"'{self.identifier}' is not a valid identifier.")

        self.dbname = self.DB_NAME
        return super().save(*args, **kwargs)


class GeneOntologyIdentifier(ExternalIdentifier):
    DB_NAME = "Gene Ontology"
    PREFIX = "GO"
    SEP = ":"

    def validate(self):
        return validators.is_go(self.identifier)


class PubmedIdentifier(ExternalIdentifier):
    DB_NAME = "PubMed"
    PREFIX = "PUBMED"
    SEP = ":"

    def validate(self):
        return validators.is_pubmed(self.identifier)


class PsimiIdentifier(ExternalIdentifier):
    DB_NAME = "Psimi"
    PREFIX = "MI"
    SEP = ":"

    def validate(self):
        return validators.is_psimi(self.identifier)


class UniprotIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt"
    PREFIX = None
    SEP = None

    def validate(self):
        return validators.is_uniprot(self.identifier)


class KeywordIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt KW"
    PREFIX = "KW"
    SEP = "-"

    def validate(self):
        return validators.is_keyword(self.identifier)


class InterproIdentifier(ExternalIdentifier):
    DB_NAME = "InterPro"
    PREFIX = "IPR"
    SEP = None

    def validate(self):
        return validators.is_interpro(self.identifier)


class PfamIdentifier(ExternalIdentifier):
    DB_NAME = "PFAM"
    PREFIX = "PF"
    SEP = None

    def validate(self):
        return validators.is_pfam(self.identifier)
