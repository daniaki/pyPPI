from pathlib import Path
from typing import Generator, Union

from ..utilities import validate_accession
from . import open_file
from .types import InteractionData


def bioplex_func(
    path: Union[str, Path]
) -> Generator[InteractionData, None, None]:
    """
    Parsing function for bioplex tsv format.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to file to parse.

    Returns
    -------
    Generator[InteractionData]
        Interaction. Label is always `None`.
    """
    source_idx = 2
    target_idx = 3

    with open_file(path, "rt") as handle:
        handle.readline()  # Remove header
        for line in handle:
            xs = line.strip().split("\t")
            source = validate_accession(xs[source_idx].strip().upper())
            target = validate_accession(xs[target_idx].strip().upper())
            if not (source and target):
                continue

            yield InteractionData(
                source=source,
                target=target,
                organism=9606,
                labels=[],
                databases=["BioPlex"],
            )
