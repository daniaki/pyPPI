from typing import Generator

from ..utilities import validate_accession

from .types import Interaction
from . import open_file


def edgelist_func(
    path, database: str = None, sep: str = "\t"
) -> Generator[Interaction, None, None]:
    """
    Parsing function a generic edgelist file.
    
    Parameters
    ----------
    path : str
        Path to file to parse.

    database: str, optional
        The database the edgelist was downloaded from.

    sep: str, optional 
        File column separator.

    Returns
    -------
    Generator[Interaction]
        Interaction. Label is always a list of `None`.
    """
    source_idx = 0
    target_idx = 1

    with open_file(path, "rt") as handle:
        handle.readline()  # Remove header
        for line in handle:
            xs = line.strip().split(sep)
            source = validate_accession(xs[source_idx].strip().upper())
            target = validate_accession(xs[target_idx].strip().upper())
            if not (source and target):
                continue

            yield Interaction(
                source=source, target=target, label=None, database=database
            )
