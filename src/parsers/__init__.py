import gzip
import logging
from pathlib import Path
from typing import Union, Callable

from ..settings import LOGGER_NAME

__all__ = [
    "bioplex",
    "hprd",
    "innate",
    "interpro",
    "kegg",
    "pfam",
    "pina",
    "types",
    "open_file",
]

logger = logging.getLogger(LOGGER_NAME)


def open_file(path: Union[str, Path], mode="rt"):
    """
    Wrapper for opening a file. Will switch to gzip if the file has a `.gz`
    extenstion.
    
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to open. Gzipped files supported.
    mode : str, optional
        File open mode, by default "rt"
    
    Returns
    -------
    [type]
        [description]
    """
    func: Callable
    if str(path).endswith("gz"):
        func = gzip.open
    else:
        func = open
    return func(path, mode=mode)


def warn_if_isoform(source, target):
    source_is_isoform = "-" in source
    target_is_isoform = "-" in target
    if source_is_isoform or target_is_isoform:
        logger.warning(
            f"Edge {(source, target)} contains UniProt isoform "
            f"identifiers."
        )
