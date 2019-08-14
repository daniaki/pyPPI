import csv
from pathlib import Path
from typing import Dict, List, Union

from . import open_file
from ..validators import is_pfam
from .types import PfamTermData


def parse_clans_file(path: Union[str, Path]) -> List[PfamTermData]:
    pfam_terms: List[PfamTermData] = []
    with open_file(path, mode="rt") as handle:
        reader = csv.DictReader(
            f=handle,
            fieldnames=["identifier", "clan", "family", "name", "description"],
            delimiter="\t",
        )
        for row in reader:
            identifier = row["identifier"].strip()
            name = row["name"].strip()
            description = row["description"].strip() or None
            if not is_pfam(identifier):
                continue
            pfam_terms.append(
                PfamTermData(
                    identifier=identifier, name=name, description=description
                )
            )
    return pfam_terms
