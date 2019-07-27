import itertools
from typing import Generator, List

from ..utilities import validate_accession, is_null
from ..constants import PSIMI_NAME_TO_IDENTIFIER

from .types import Interaction
from . import open_file


def innate_mitab_func(path: str) -> Generator[Interaction, None, None]:
    """
    Parsing function for psimitab format files issued by `InnateDB`.

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
    uniprot_source_idx = 4
    uniprot_target_idx = 5
    source_idx = 2
    target_idx = 3
    d_method_idx = 6  # detection method
    pmid_idx = 8

    # Remove header
    with open_file(path, "rt") as handle:
        handle.readline()  # Remove header
        for line in handle:
            xs = line.strip().split("\t")
            ensembl_source = xs[source_idx].strip()
            ensembl_target = xs[target_idx].strip()
            if ("ENSG" not in ensembl_source) or (
                "ENSG" not in ensembl_target
            ):
                continue

            # These formats might contain multiple uniprot interactors in a
            # single line, or none. Continue parsing if the latter.
            source_ls = [
                elem.split(":")[1]
                for elem in xs[uniprot_source_idx].split("|")
                if ("uniprotkb" in elem and not "_" in elem)
            ]
            target_ls = [
                elem.split(":")[1]
                for elem in xs[uniprot_target_idx].split("|")
                if ("uniprotkb" in elem) and (not "_" in elem)
            ]

            if len(source_ls) < 1 or len(target_ls) < 1:
                continue

            d_method_line = xs[d_method_idx].strip()
            d_psimi = None
            if not is_null(d_method_line):
                _, d_method_text = d_method_line.strip().split("psi-mi:")
                _, d_psimi, _ = d_method_text.split('"')
                if is_null(d_psimi):
                    d_psimi = None
                else:
                    d_psimi.strip().upper()

            pmid_line = xs[pmid_idx].strip()
            pmid = None
            if not is_null(pmid_line):
                pmid = pmid_line.split(":")[-1]
                if is_null(pmid):
                    pmid = None
                else:
                    pmid.strip().upper()

            # Iterate through the list of tuples, each tuple being a list of
            # accessions found within a line for each of the two proteins.
            for source, target in itertools.product(source_ls, target_ls):
                source = validate_accession(source)
                target = validate_accession(target)
                if not (source and target):
                    continue

                experiment_types: List[str] = []
                if d_psimi:
                    experiment_types = [PSIMI_NAME_TO_IDENTIFIER[d_psimi]]

                pmids: List[str] = []
                if pmid:
                    pmids = [pmid]

                psimi_ids: List[str] = []
                if d_psimi:
                    psimi_ids = [d_psimi]

                else:
                    yield Interaction(
                        source=source,
                        target=target,
                        labels=[],
                        pubmed_ids=pmids,
                        psimi_ids=psimi_ids,
                        experiment_types=experiment_types,
                        databases=["InnateDB"],
                    )
