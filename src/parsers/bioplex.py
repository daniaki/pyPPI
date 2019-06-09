from typing import Generator

from ..utilities import validate_accession

from .types import Interaction
from . import open_file


def bioplex_func(path: str) -> Generator[Interaction, None, None]:
    """
    Parsing function for bioplex tsv format.
    
    Parameters
    ----------
    path : str
        Path to file to parse.

    Returns
    -------
    Generator[Interaction]
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

            yield Interaction(
                source=source, target=target, label=None, database="BioPlex"
            )
