import csv
import itertools
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ..settings import LOGGER_NAME
from ..utilities import is_null
from ..validators import is_pubmed, is_uniprot, psimi_re, uniprot_re
from . import open_file, warn_if_isoform
from .types import InteractionData, InteractionEvidenceData

logger = logging.getLogger(LOGGER_NAME)


def parse_interactions(path: Union[str, Path]) -> List[InteractionData]:
    """
    Parsing function for psimitab format files from `PINA2`.

    Parameters
    ----------
    path : Union[str, Path]
        Path to file to parse.

    Returns
    -------
    List[InteractionData]
    """
    uniprot_source_column = '"ID(s) interactor A"'
    uniprot_target_column = '"ID(s) interactor B"'
    detection_method_column = '"Interaction detection method(s)"'
    pmid_column = '"Publication Identifier(s)"'
    taxid_A_column = '"Taxid interactor A"'
    taxid_B_column = '"Taxid interactor B"'

    interactions: List[InteractionData] = []

    with open_file(path, "rt") as handle:
        header = handle.readline().strip().split("\t")
        reader = csv.DictReader(f=handle, fieldnames=header, delimiter="\t")

        for row in reader:
            source_tax = row[taxid_A_column].strip()
            target_tax = row[taxid_B_column].strip()
            if ("9606" not in source_tax) or ("9606" not in target_tax):
                continue

            # These formats should contain ONE uniprot interactor in a
            # single line, or none. Continue parsing if the latter.
            source: Optional[str] = None
            target: Optional[str] = None

            source_match = uniprot_re.search(row[uniprot_source_column])
            target_match = uniprot_re.search(row[uniprot_target_column])
            if source_match:
                source = source_match.group().strip().upper()
            if target_match:
                target = target_match.group().strip().upper()

            if is_null(source) or is_null(target):
                # logger.warning(
                #     f"Edge {(source, target)} contains null UniProt "
                #     f"identifiers. Skipping line."
                # )
                continue

            warn_if_isoform(source, target)

            # These lines should be equal in length
            assert len(row[pmid_column].split("|")) == len(
                row[detection_method_column].split("|")
            )
            # Some PMIDs will be DOIs so ignore these.
            evidence_ids = [
                (pmid.split(":")[-1].strip().upper(), psimi.strip().upper())
                for (pmid, psimi) in zip(
                    row[pmid_column].split("|"),
                    row[detection_method_column].split("|"),
                )
                if is_pubmed(pmid.split(":")[-1].strip().upper())
            ]
            evidence: List[InteractionEvidenceData] = []
            for (pmid, psimi) in evidence_ids:
                match = psimi_re.match(psimi)
                evidence.append(
                    InteractionEvidenceData(
                        pubmed=pmid, psimi=match.group() if match else None
                    )
                )

            assert source is not None
            assert target is not None
            interactions.append(
                InteractionData(
                    source=source,
                    target=target,
                    evidence=list(
                        sorted(set(evidence), key=lambda e: hash(e))
                    ),
                    databases=["pina"],
                )
            )

    return interactions
