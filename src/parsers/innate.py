import csv
import itertools
from pathlib import Path
from typing import Generator, List, Union

from ..validators import psimi_re, pubmed_re, uniprot_re, validate_accession
from . import open_file
from .types import InteractionData, InteractionEvidenceData


def parse_interactions(
    path: Union[str, Path]
) -> Generator[InteractionData, None, None]:
    """
    Parsing function for psimitab format files issued by `InnateDB`.

    Parameters
    ----------
    path : str
        Path to file to parse.

    Returns
    -------
    Generator[Interaction, None, None]
    """
    uniprot_source_column = "alias_A"
    uniprot_target_column = "alias_B"
    detection_method_column = "interaction_detection_method"
    pmid_column = "pmid"
    taxid_A_column = "ncbi_taxid_A"
    taxid_B_column = "ncbi_taxid_B"

    # Remove header
    with open_file(path, "rt") as handle:
        header = handle.readline().strip().split("\t")
        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")

        for row in reader:
            source_tax = row[taxid_A_column].strip()
            target_tax = row[taxid_B_column].strip()
            if ("9606" not in source_tax) or ("9606" not in target_tax):
                continue

            # These formats might contain multiple uniprot interactors in a
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
                (pmid, psimi)
                for (pmid, psimi) in generator
                if pubmed_re.fullmatch(pmid)
            ]

            # Iterate through the list of tuples, each tuple being a list of
            # accessions found within a line for each of the two proteins.
            for source, target in itertools.product(sources, targets):
                evidence: List[InteractionEvidenceData] = []
                for pmid, psimi in evidence_ids:
                    match = None if not psimi else psimi_re.findall(psimi)
                    evidence.append(
                        InteractionEvidenceData(
                            pubmed=pmid, psimi=None if not match else match[0]
                        )
                    )

                yield InteractionData(
                    source=source,
                    target=target,
                    organism=9606,
                    evidence=list(
                        sorted(set(evidence), key=lambda e: hash(e))
                    ),
                    databases=["InnateDB"],
                )
