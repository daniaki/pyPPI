import csv
import itertools
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Generator, List, Set, Union

from ..utilities import is_null
from ..validators import uniprot_re, psimi_re, pubmed_re, validate_accession
from . import open_file
from .types import InteractionData, InteractionEvidenceData


def parse_interactions(
    path: Union[str, Path]
) -> Generator[InteractionData, None, None]:
    """
    Parsing function for psimitab format files from `PINA2`.

    Parameters
    ----------
    path : Union[str, Path]
        Path to file to parse.

    Returns
    -------
    Generator[Interaction, None, None]
    """
    uniprot_source_column = '"ID(s) interactor A"'
    uniprot_target_column = '"ID(s) interactor B"'
    detection_method_column = '"Interaction detection method(s)"'
    pmid_column = '"Publication Identifier(s)"'
    taxid_A_column = '"Taxid interactor A"'
    taxid_B_column = '"Taxid interactor B"'

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
            sources = [
                m[0] for m in uniprot_re.findall(row[uniprot_source_column])
            ]
            targets = [
                m[0] for m in uniprot_re.findall(row[uniprot_target_column])
            ]
            if not sources or not targets:
                continue

            psimi_ids = psimi_re.findall(row[detection_method_column])
            pmids = [m[0] for m in pubmed_re.findall(row[pmid_column])]
            evidence: List[InteractionEvidenceData] = []
            if len(pmids):
                assert len(pmids) == len(psimi_ids)
                for pmid, psimi in zip(pmids, psimi_ids):
                    evidence.append(
                        InteractionEvidenceData(pubmed=pmid, psimi=psimi)
                    )

            yield InteractionData(
                source=sources[0],
                target=targets[0],
                organism=9606,
                evidence=list(sorted(set(evidence), key=lambda e: hash(e))),
                databases=["PINA2"],
            )
