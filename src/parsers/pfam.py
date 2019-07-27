from dataclasses import dataclass
from typing import Dict, List

from . import open_file


@dataclass
class ParsedPfamTerm:
    identifier: str
    name: str
    description: str


def parse_clans_file(path: str) -> List[ParsedPfamTerm]:
    pfam_terms: List[ParsedPfamTerm] = []
    with open_file(path, mode="rt") as handle:
        for line in handle:
            identifier, _, _, name, description = line.strip().split("\t")
            pfam_terms.append(
                ParsedPfamTerm(
                    identifier=identifier, name=name, description=description
                )
            )
    return pfam_terms
