import csv
import itertools
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set, Union, Optional

from ..utilities import is_null
from ..validators import uniprot_re, psimi_re, is_uniprot, is_pubmed
from . import open_file
from .types import InteractionData, InteractionEvidenceData


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

            if not source or not target:
                continue

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
