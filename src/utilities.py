"""
Collection of utility operations that don't go anywhere else.
"""

import os
import logging
import numpy as np
import math
import gzip
from urllib.request import urlretrieve

from itertools import islice
from typing import Sequence, List, Generator, Any
from collections import OrderedDict

from .constants import NULL_RE


__all__ = [
    "take",
    "chunks",
    "is_null",
    "su_make_dir",
    "remove_duplicates",
    "download_from_url",
    "validate_accession",
]


logger: logging.Logger = logging.getLogger("pyppi")


def is_null(value: Any) -> bool:
    """
    Check if a value is NaN/None/NA/empty etc.

    Returns
    -------
    bool
        True if the value is considered null.
    """
    return NULL_RE.fullmatch(str(value)) is not None


def su_make_dir(path: str, mode: int = 0o777) -> None:
    """Make a directory at the path with read and write permisions"""
    if not path or os.path.exists(path):
        logger.info("Found existing directory {}.".format(path))
    else:
        os.mkdir(path)
        os.chmod(path, mode)


def take(n: int, iterable: Sequence) -> List:
    """Return first n items of the iterable as a list."""
    return list(islice(iterable, n))


def remove_duplicates(seq: Sequence) -> List:
    """Remove duplicates from a sequence preserving order."""
    return list(OrderedDict.fromkeys(seq).keys())


def chunks(l: Sequence, n: int) -> Generator:
    """Yield successive n-sized chunks from l."""
    ls = list(l)
    for i in range(0, len(ls), n):
        yield ls[i : i + n]


def validate_accession(accession):
    """Return None if an accession is invalid, else strip and uppercase it."""
    if is_null(accession):
        return None
    else:
        return accession.strip().upper()


def download_from_url(url, save_path, compress=True):
    logger.info(f"Downloading file from {url}")
    if compress:
        tmp, info = urlretrieve(url)
        bytes_ = info["Content-Length"]
        logger.info(f"Compresing file with size {bytes_} bytes")
        with open(tmp, "rb") as f_in, gzip.open(save_path, "wb") as f_out:
            return f_out.writelines(f_in)
    else:
        return urlretrieve(url, save_path)
