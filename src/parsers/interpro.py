from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

from . import open_file
from ..utilities import is_null
from .types import InterproTermData


def parse_entry_list(path: Union[str, Path]) -> List[InterproTermData]:
    """
    Parse the entry-list file into a list of Interpro data structures that
    can be used to create database rows.
    
    Parameters
    ----------
    path : Union[str, Path]
        Path or string path. Gzipped files supported.
    
    Returns
    -------
    List[InterproTermData]
    """
    terms: List[InterproTermData] = []
    with open_file(path, mode="rt") as handle:
        reader = DictReader(
            f=handle,
            fieldnames=handle.readline().strip().split("\t"),
            delimiter="\t",
        )
        for row in reader:
            identifier = (
                None if is_null(row["ENTRY_AC"]) else row["ENTRY_AC"].strip()
            )
            if not identifier:
                continue
            terms.append(
                InterproTermData(
                    identifier=identifier.strip().upper(),
                    description=(
                        None
                        if is_null(row["ENTRY_NAME"])
                        else row["ENTRY_NAME"].strip()
                    ),
                    entry_type=(
                        None
                        if is_null(row["ENTRY_TYPE"])
                        else row["ENTRY_TYPE"].strip()
                    ),
                )
            )
    return terms
