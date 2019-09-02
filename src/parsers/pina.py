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
            sources: List[str] = [
                str(validate_accession(m[0]))
                for m in uniprot_re.findall(row[uniprot_source_column])
                if validate_accession(m[0]) is not None
            ]
            targets: List[str] = [
                str(validate_accession(m[0]))
                for m in uniprot_re.findall(row[uniprot_target_column])
                if validate_accession(m[0]) is not None
            ]
            if not sources or not targets:
                continue

            # These lines should be equal in length
            assert len(row[pmid_column].split("|")) == len(
                row[detection_method_column].split("|")
            )
            # Some PMIDs will be DOIs so ignore these.
            evidence_ids = [
                (pmid, psimi)
                for (pmid, psimi) in zip(
                    row[pmid_column].split("|"),
                    row[detection_method_column].split("|"),
                )
                if pubmed_re.fullmatch(pmid)
            ]
            evidence: List[InteractionEvidenceData] = []
            # Zip will remove trailing psimi
            for (pmid, psimi) in evidence_ids:
                match = psimi_re.match(psimi)
                evidence.append(
                    InteractionEvidenceData(
                        pubmed=pmid, psimi=match.group() if match else None
                    )
                )

            assert len(sources) == 1
            assert len(targets) == 1
            yield InteractionData(
                source=sources[0],
                target=targets[0],
                organism=9606,
                evidence=list(sorted(set(evidence), key=lambda e: hash(e))),
                databases=["PINA2"],
            )
