"""
Collection of utility operations that don't go anywhere else.
"""

import gzip
import logging
import urllib.request
from pathlib import Path
from typing import Any, Union

from .constants import NULL_RE
from .settings import LOGGER_NAME

__all__ = ["is_null", "download_from_url"]


logger: logging.Logger = logging.getLogger(LOGGER_NAME)


def is_null(value: Any) -> bool:
    """
    Check if a value is NaN/None/NA/empty etc.

    Returns
    -------
    bool
        True if the value is considered null.
    """
    if not value:
        return True
    return NULL_RE.fullmatch(str(value)) is not None


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
