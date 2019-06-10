import re
import numpy as np
from pathlib import Path
from typing import DefaultDict

from collections import defaultdict

from .settings import DATA_DIR, NETWORKS_DIR, MODELS_DIR


__all__ = [
    "P1",
    "P2",
    "G1",
    "G2",
    "Columns",
    "Paths",
    "Urls",
    "NULL_RE",
    "UNIPROT_ORD_KEY",
    "PSIMI_NAME_TO_IDENTIFIER",
]


P1 = "protein_a"
P2 = "protein_b"
G1 = "gene_a"
G2 = "gene_b"


NULL_RE = re.compile(
    r"^none|na|nan|n/a|undefined|unknown|null|\s+$", flags=re.IGNORECASE
)

UNIPROT_ORD_KEY: DefaultDict[str, int] = defaultdict(lambda: 9)

PSIMI_NAME_TO_IDENTIFIER = {
    "in vitro": "MI:0492",
    "invitro": "MI:0492",
    "in vivo": "MI:0493",
    "invivo": "MI:0493",
    "yeast 2-hybrid": "MI:0018",
}


class Columns:
    source = "source"
    target = "target"
    label = "label"
    pubmed = "pubmed"
    psimi = "psimi"


class Paths:
    psimi_obo = DATA_DIR / "mi.obo.gz"
    go_obo = DATA_DIR / "go.obo.gz"
    interpro_entries = DATA_DIR / "entry.list"
    pfam_clans = DATA_DIR / "Pfam-A.clans.tsv.gz"

    # Networks
    hprd_ptms = NETWORKS_DIR / "POST_TRANSLATIONAL_MODIFICATIONS.txt"
    hprd_xref = NETWORKS_DIR / "HPRD_ID_MAPPINGS.txt"
    pina2_mitab = NETWORKS_DIR / "Homo-sapiens-20140521.tsv.gz"
    bioplex = NETWORKS_DIR / "BioPlex_interactionList_v4a.tsv.gz"
    innate_all = NETWORKS_DIR / "all.mitab.gz"
    innate_curated = NETWORKS_DIR / "innatedb_ppi.mitab.gz"

    # classifiers
    trained_models = MODELS_DIR


class Urls:
    interpro_entries = "ftp://ftp.ebi.ac.uk/pub/databases/interpro/entry.list"
    psimi_obo = "http://purl.obolibrary.org/obo/mi.obo"
    go_obo = "http://purl.obolibrary.org/obo/go.obo"
    pfam_clans = (
        "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/"
        "current_release/Pfam-A.clans.tsv.gz"
    )

    # Networks
    pina2_mitab = (
        "http://omics.bjcancer.org/pina/download/Homo%20sapiens-20140521.tsv"
    )
    bioplex = (
        "http://bioplex.hms.harvard.edu/data/BioPlex_interactionList_v4a.tsv"
    )
    innate_curated = (
        "http://www.innatedb.com/download/interactions/innatedb_ppi.mitab.gz"
    )
    innate_all = "http://www.innatedb.com/download/interactions/all.mitab.gz"
