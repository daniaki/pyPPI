"""
Collection of utility operations that don't go anywhere else.
"""

import gzip
import logging
import math
import os
import urllib.request
from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Sequence, Union

import numpy as np

from .constants import NULL_RE

__all__ = ["is_null", "download_from_url", "validate_accession"]


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


def validate_accession(
    accession: Optional[str], formatting: Callable = str.upper
) -> Optional[str]:
    """Return None if an accession is invalid, else strip and uppercase it."""
    if accession is None:
        return None
    elif is_null(accession):
        return None
    else:
        return formatting(accession.strip())


def download_from_url(
    url: str, save_path: Union[str, Path], compress: bool = True
):
    logger.info(f"Downloading file from {url}")
    if compress:
        tmp_file_path, _ = urllib.request.urlretrieve(url)
        logger.info(f"Compresing file.")
        with open(tmp_file_path, "rb") as f_in, gzip.open(
            save_path, "wb"
        ) as f_out:
            return f_out.writelines(f_in)
    else:
        return urllib.request.urlretrieve(url, save_path)
