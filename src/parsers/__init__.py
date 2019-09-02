import gzip
from pathlib import Path
from typing import Union, Callable


__all__ = [
    "bioplex",
    "edgelist",
    "hprd",
    "innate",
    "interpro",
    "kegg",
    "pfam",
    "pina",
    "types",
    "open_file",
]


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
