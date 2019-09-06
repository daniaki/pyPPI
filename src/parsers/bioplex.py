from pathlib import Path
from typing import List, Union, Optional

from ..validators import is_uniprot
from . import open_file
from .types import InteractionData


def parse_interactions(path: Union[str, Path]) -> List[InteractionData]:
    """
    Parsing function for bioplex tsv format.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to file to parse.

    Returns
    -------
    List[InteractionData]
    """
    source_idx = 2
    target_idx = 3

    interactions: List[InteractionData] = []
    with open_file(path, "rt") as handle:
        handle.readline()  # Remove header
        for line in handle:
            xs = line.strip().split("\t")

            source = xs[source_idx].strip().upper()
            target = xs[target_idx].strip().upper()

            if not (source and target):
                continue

            if not is_uniprot(source) or not is_uniprot(target):
                raise ValueError(
                    f"Edge '{(source, target)}' contains invalid UniProt "
                    f"identifiers."
                )

            interactions.append(
                InteractionData(
                    source=source, target=target, databases=["bioplex"]
                )
            )
    return interactions
