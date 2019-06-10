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
    if os.path.splitext(path)[-1] == "gz":
        func = gzip.open
    else:
        func = open

    return func(path, mode=mode)
