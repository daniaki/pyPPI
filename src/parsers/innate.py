import csv
import itertools
import logging
from pathlib import Path
from typing import Generator, List, Optional, Union

from ..settings import LOGGER_NAME
from ..utilities import is_null
from ..validators import is_pubmed, is_uniprot, psimi_re, uniprot_re
from . import open_file, warn_if_isoform
from .types import InteractionData, InteractionEvidenceData

logger = logging.getLogger(LOGGER_NAME)

def parse_interactions(path: Union[str, Path]) -> List[InteractionData]:
    """
    Parsing function for psimitab format files issued by `InnateDB`.

    Parameters
    ----------
    path : str
        Path to file to parse.

    Returns
    -------
    List[InteractionData]
    """
    uniprot_source_column = "alias_A"
    uniprot_target_column = "alias_B"
    detection_method_column = "interaction_detection_method"
    pmid_column = "pmid"
    taxid_A_column = "ncbi_taxid_A"
    taxid_B_column = "ncbi_taxid_B"

    interactions: List[InteractionData] = []
    with open_file(path, "rt") as handle:
        # Remove header
        header = handle.readline().strip().split("\t")
        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")

        for row in reader:
            source_tax = row[taxid_A_column].strip()
            target_tax = row[taxid_B_column].strip()
            if ("9606" not in source_tax) or ("9606" not in target_tax):
                continue

            # These formats might contain multiple uniprot interactors in a
            # single line, or none. Continue parsing if the latter. Take
            # first identifier only.
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

            # These lines should be equal in length, unless one PSI-MI term
            # has been provided for multiple PMIDs.
            pmids = row[pmid_column].split("|")
            psimis = row[detection_method_column].split("|")
            try:
                assert len(pmids) == len(psimis)
                generator = zip(pmids, psimis)
            except AssertionError:
                # There should only be one MI term. Fail otherwise.
                assert len(psimis) == 1
                assert len(pmids) >= 1
                # Pad psimi list to even out length
                generator = zip(pmids, psimis * len(pmids))

            # Some PMIDs will be DOIs so ignore these.
            evidence_ids = [
                (pmid.split(":")[-1].strip().upper(), psimi.strip().upper())
                for (pmid, psimi) in generator
                if is_pubmed(pmid.split(":")[-1].strip().upper())
            ]

            # Iterate through the list of tuples, each tuple being a list of
            # accessions found within a line for each of the two proteins.
            evidence: List[InteractionEvidenceData] = []
            for pmid, psimi in evidence_ids:
                match = None if not psimi else psimi_re.search(psimi)
                evidence.append(
                    InteractionEvidenceData(
                        pubmed=pmid, psimi=None if not match else match.group()
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
                    databases=["innatedb"],
                )
            )

    return interactions
