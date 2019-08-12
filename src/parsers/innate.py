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

            # Iterate through the list of tuples, each tuple being a list of
            # accessions found within a line for each of the two proteins.
            for source, target in itertools.product(sources, targets):
                source = validate_accession(source)
                target = validate_accession(target)
                if not (source and target):
                    continue

                evidence: List[InteractionEvidenceData] = []
                if len(pmids):
                    assert len(psimi_ids) == len(pmids)
                    for pmid, psimi in zip(pmids, psimi_ids):
                        evidence.append(
                            InteractionEvidenceData(pubmed=pmid, psimi=psimi)
                        )

                yield InteractionData(
                    source=source,
                    target=target,
                    organism=9606,
                    evidence=evidence,
                    databases=["InnateDB"],
                )
