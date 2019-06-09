from csv import DictReader
from dataclasses import dataclass
from typing import List, Dict

from . import open_file


@dataclass
class ParsedInterproTerm:
    identifier: str
    name: str
    description: str
    term_type: str


def parse_entry_list(path: str) -> List[ParsedInterproTerm]:
    terms: List[ParsedInterproTerm] = []
    with open_file(path, mode="rt") as handle:
        reader = DictReader(
            f=handle,
            fieldnames=["ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"],
            delimiter="\t",
        )
        for row in reader:
            terms.append(
                ParsedInterproTerm(
                    identifier=row["ENTRY_AC"],
                    name=row["ENTRY_NAME"],
                    description=row["ENTRY_NAME"],
                    term_type=row["ENTRY_TYPE"],
                )
            )
    return terms
