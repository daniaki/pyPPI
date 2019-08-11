from csv import DictReader
from dataclasses import dataclass
from typing import List, Dict

from . import open_file
from .types import InterproTermData


def parse_entry_list(path: str) -> List[InterproTermData]:
    terms: List[InterproTermData] = []
    with open_file(path, mode="rt") as handle:
        reader = DictReader(
            f=handle,
            fieldnames=["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"],
            delimiter="\t",
        )
        for row in reader:
            terms.append(
                InterproTermData(
                    identifier=row["ENTRY_AC"],
                    name=row["ENTRY_NAME"],
                    entry_type=row["ENTRY_TYPE"],
                )
            )
    return terms
