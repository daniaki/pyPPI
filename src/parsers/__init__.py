import os
import gzip


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
    "utilities",
    "open_file",
]


def open_file(path, mode="rt"):
    """
    Wrapper for opening a file. Will switch to gzip if the file has a `.gz`
    extenstion.
    
    Args:
        path (Path): A path to open.
        mode (str, optional): [description]. Defaults to "rt".
    
    Returns:
        [type]: [description]
    """
    if os.path.splitext(path)[-1] == "gz":
        func = gzip.open
    else:
        func = open

    return func(path, mode=mode)
