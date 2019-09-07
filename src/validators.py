import re
from typing import Optional, Callable, Pattern

from idutils import uniprot_regexp

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
]


uniprot_re = re.compile(r"({})(-\d+)?".format(uniprot_regexp.pattern[:-1]))
psimi_re = re.compile(r"MI:\d{4}")
pubmed_re = re.compile(r"\d+")
go_re = re.compile(r"GO:\d{7}")
interpro_re = re.compile(r"IPR\d{6}")
pfam_re = re.compile(r"PF\d{5}")
keyword_re = re.compile(r"KW-\d{4}")


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


def is_uniprot(identifier):
    return uniprot_re.fullmatch(str(identifier))