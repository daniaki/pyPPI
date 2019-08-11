from dataclasses import dataclass
from typing import Dict, List

from . import open_file
from .types import PfamTermData


def parse_clans_file(path: str) -> List[PfamTermData]:
    pfam_terms: List[PfamTermData] = []
    with open_file(path, mode="rt") as handle:
        for line in handle:
            identifier, _, _, name, description = line.strip().split("\t")
            pfam_terms.append(
                PfamTermData(
                    identifier=identifier, name=name, description=description
                )
            )
    return pfam_terms
