from pathlib import Path
from typing import Generator, Iterable, Union

from ..validators import validate_accession

from .types import InteractionData
from . import open_file


def parse_interactions(
    path: Union[str, Path], databases: Iterable[str] = (), sep: str = "\t"
) -> Generator[InteractionData, None, None]:
    """
    Parsing function a generic edgelist file.
    
    Parameters
    ----------
    path : str | Path
        Path to file to parse.

    databases: list[str], optional
        The databases an edgelist was downloaded from.

    sep: str, optional 
        File column separator.

    Returns
    -------
    Generator[Interaction, None, None]
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

            yield InteractionData(
                source=source,
                target=target,
                organism=9606,
                databases=list(databases),
            )
