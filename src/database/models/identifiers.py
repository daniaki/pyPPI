from typing import Iterable, Optional, Callable

import peewee

from ... import validators
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


def add_prefix(
    identifier: str, prefix: Optional[str] = None, sep: Optional[str] = None
):
    """
    Adds a prefix separated by a delimiter to an accession. Prefix will
    only be added if it isn't already in the accession.

    Parameters
    ----------
    identifier : str
        String accession.
    prefix : Optional[str], optional
        Prefix to append. Identifier returned as is if not provided.
    sep : Optional[str], optional
        Delimiter to separate accession and prefix with.

    Returns
    -------
    str
    """
    if not prefix:
        return identifier
    if not identifier.lower().startswith(f"{prefix.lower()}"):
        if sep:
            return f"{prefix}{sep}{identifier}"
        return f"{prefix}{identifier}"
    return identifier


class ExternalIdentifier(BaseModel):
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

    @classmethod
    def get_by_identifier(
        cls, identifiers: Iterable[str]
    ) -> peewee.ModelSelect:
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
        return add_prefix(identifier, cls.PREFIX, cls.SEP).upper()

    @classmethod
    def validate(cls, identifier: str) -> str:
        """
        How to validate identifier. Called after formatter is called and must
        accept a single input and return a single boolean.
        """
        raise NotImplementedError()

    def format_for_save(self):
        if self.DB_NAME is None:
            raise NotImplementedError("Concrete table must define DB_NAME.")

        if self.identifier is None:
            raise TypeError(f"'{self.identifier}' is cannot be 'None'.")

        self.identifier = self.format(self.identifier)
        if not self.validate(self.identifier):
            raise ValueError(f"'{self.identifier}' is not a valid identifier.")

        self.dbname = self.DB_NAME

        return super().format_for_save()


class GeneOntologyIdentifier(ExternalIdentifier):
    DB_NAME = "Gene Ontology"
    PREFIX = "GO"
    SEP = ":"

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_go(identifier)


class PubmedIdentifier(ExternalIdentifier):
    DB_NAME = "PubMed"
    PREFIX = "PUBMED"
    SEP = ":"

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_pubmed(identifier)


class PsimiIdentifier(ExternalIdentifier):
    DB_NAME = "Psimi"
    PREFIX = "MI"
    SEP = ":"

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_psimi(identifier)


class UniprotIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt"
    PREFIX = None
    SEP = None

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_uniprot(identifier)


class KeywordIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt KW"
    PREFIX = "KW"
    SEP = "-"

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_keyword(identifier)


class InterproIdentifier(ExternalIdentifier):
    DB_NAME = "InterPro"
    PREFIX = "IPR"
    SEP = None

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_interpro(identifier)


class PfamIdentifier(ExternalIdentifier):
    DB_NAME = "PFAM"
    PREFIX = "PF"
    SEP = None

    @classmethod
    def validate(cls, identifier: str):
        return validators.is_pfam(identifier)
