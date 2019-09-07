import logging
from pathlib import Path
from typing import List, Optional, Union

from ..settings import LOGGER_NAME
from ..utilities import is_null
from ..validators import is_uniprot
from . import open_file, warn_if_isoform
from .types import InteractionData

logger = logging.getLogger(LOGGER_NAME)


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

            if is_null(source) or is_null(target):
                # logger.warning(
                #     f"Edge {(source, target)} contains null UniProt "
                #     f"identifiers. Skipping line."
                # )
                continue

            warn_if_isoform(source, target)

            if (not is_uniprot(source)) or (not is_uniprot(target)):
                raise ValueError(
                    f"Edge {(source, target)} contains invalid UniProt "
                    f"identifiers."
                )

            assert source is not None
            assert target is not None
            interactions.append(
                InteractionData(
                    source=source, target=target, databases=["bioplex"]
                )
            )
    return interactions
