import itertools

from typing import Generator, List, Dict, Union, Set
from collections import OrderedDict

from ..utilities import validate_accession, is_null
from ..constants import PSIMI_NAME_TO_IDENTIFIER

from .types import Interaction
from . import open_file


def pina_sif_func(path: str) -> Generator[Interaction, None, None]:
    """
    Parsing function for bioplex tsv format.

    Parameters
    ----------
    path : str
        Path to file to parse.

    Returns
    -------
    Generator[Interaction]
        Interaction. Label is always `None`. There may be pubmed and psimi
        identifiers associated with each row.
    """
    source_idx = 0
    target_idx = 2

    with open_file(path, "rt") as handle:
        for line in handle:
            xs = line.strip().split(" ")
            source = validate_accession(xs[source_idx].strip().upper())
            target = validate_accession(xs[target_idx].strip().upper())
            if not (source and target):
                continue
            yield Interaction(source=source, target=target, labels=[])


def pina_mitab_func(path: str) -> Generator[Interaction, None, None]:
    """
    Parsing function for psimitab format files from `PINA2`.

    Parameters
    ----------
    path : str
        Path to file to parse.

    Returns
    -------
    Generator[Interaction]
        Interaction. Label is always `None`. There may be pubmed and psimi
        identifiers associated with each row.
    """
    uniprot_source_idx = 0
    uniprot_target_idx = 1
    d_method_idx = 6  # detection method
    pmid_idx = 8

    with open_file(path, "rt") as handle:
        handle.readline()  # Remove header
        for line in handle:
            xs = line.strip().split("\t")
            source = validate_accession(
                xs[uniprot_source_idx].split(":")[-1].strip().upper()
            )
            target = validate_accession(
                xs[uniprot_target_idx].split(":")[-1].strip().upper()
            )
            if not (source and target):
                continue

            pmids: List[str] = list(
                OrderedDict.fromkeys(
                    [
                        x.split(":")[-1]
                        for x in xs[pmid_idx].strip().split("|")
                        if not is_null(x.split(":")[-1])
                    ]
                ).keys()
            )
            psimi_ids: List[str] = list(
                OrderedDict.fromkeys(
                    [
                        x.split("(")[0]
                        for x in xs[d_method_idx].strip().split("|")
                        if not is_null(x.split("(")[0])
                    ]
                ).keys()
            )
            experiment_types: List[str] = [
                PSIMI_NAME_TO_IDENTIFIER[psimi_id] for psimi_id in psimi_ids
            ]

            yield Interaction(
                source=source,
                target=target,
                labels=[],
                pubmed_ids=pmids,
                psimi_ids=psimi_ids,
                experiment_types=experiment_types,
                databases=["PINA2"],
            )
