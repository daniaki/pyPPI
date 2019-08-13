import re
from typing import Optional, Callable

from idutils import is_uniprot, uniprot_regexp

from .utilities import is_null

__all__ = [
    "psimi_re",
    "go_re",
    "interpro_re",
    "pfam_re",
    "uniprot_re",
    "pubmed_re",
    "keyword_re",
    "is_go",
    "is_pubmed",
    "is_psimi",
    "is_interpro",
    "is_pfam",
    "is_keyword",
    "validate_accession",
]


uniprot_re = re.compile(
    f"({uniprot_regexp.pattern[:-1]})", flags=re.IGNORECASE
)
psimi_re = re.compile(r"MI:\d{4}", flags=re.IGNORECASE)
pubmed_re = re.compile(r"((PUBMED:)?\d+)", flags=re.IGNORECASE)
go_re = re.compile(r"GO:\d{7}", flags=re.IGNORECASE)
interpro_re = re.compile(r"IPR\d{6}", flags=re.IGNORECASE)
pfam_re = re.compile(r"PF\d{5}", flags=re.IGNORECASE)
keyword_re = re.compile(r"KW-\d{4}", flags=re.IGNORECASE)


def is_pubmed(identifier):
    return pubmed_re.fullmatch(str(identifier))


def is_psimi(identifier):
    return psimi_re.fullmatch(str(identifier))


def is_go(identifier):
    return go_re.fullmatch(str(identifier))


def is_interpro(identifier):
    return interpro_re.fullmatch(str(identifier))


def is_pfam(identifier):
    return pfam_re.fullmatch(str(identifier))


def is_keyword(identifier):
    return keyword_re.fullmatch(str(identifier))


def validate_accession(
    accession: Optional[str],
    formatter: Callable = str.upper,
    validator: Callable = is_uniprot,
) -> Optional[str]:
    """
    Return None if an accession is invalid or null, else strip whitespace and
    apply the formatter callback.

    Parameters
    ----------
    accession : str | None
        Accession to validate.

    formatter : callable, optional.
        String formatting function. Should return a string value and accept
        a single string input.
    
    validator : callable, optional.
        Validator function to check if an accession is valid. Should return
        a single boolean value and accept a single input value.

    Returns
    -------
    str | None
    """
    if accession is None:
        return None
    elif is_null(accession):
        return None
    elif not validator(str(accession).strip()):
        return None
    else:
        return formatter(accession.strip())
