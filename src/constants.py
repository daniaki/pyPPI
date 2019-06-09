"""Definitions of some commonly used variables throughout the application."""

__all__ = ["P1", "P2", "G1", "G2", "null_re", "Columns"]

import re
import numpy as np
from typing import DefaultDict

from collections import defaultdict


P1 = "protein_a"
P2 = "protein_b"
G1 = "gene_a"
G2 = "gene_b"


class Columns:
    source = "source"
    target = "target"
    label = "label"
    pubmed = "pubmed"
    psimi = "psimi"


null_re = re.compile(
    r"^none|na|nan|n/a|undefined|unknown|null|\s+$", flags=re.IGNORECASE
)

UNIPROT_ORD_KEY: DefaultDict[str, int] = defaultdict(lambda: 9)


psimi_name_to_identifier = {
    "in vitro": "MI:0492",
    "invitro": "MI:0492",
    "in vivo": "MI:0493",
    "invivo": "MI:0493",
    "yeast 2-hybrid": "MI:0018",
}
