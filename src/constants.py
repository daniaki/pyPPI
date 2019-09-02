import re
import numpy as np
from pathlib import Path
from typing import DefaultDict, Dict

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

PSIMI_NAME_TO_IDENTIFIER: Dict[str, str] = {
    "in vitro": "MI:0492",
    "invitro": "MI:0492",
    "in vivo": "MI:0493",
    "invivo": "MI:0493",
    "yeast 2-hybrid": "MI:0018",
}


class GeneOntologyCategory:
    molecular_function = "Molecular function"
    biological_process = "Biological process"
    cellular_component = "Cellular component"

    @classmethod
    def list(cls):
        return [
            cls.molecular_function,
            cls.biological_process,
            cls.cellular_component,
        ]

    @classmethod
    def letter_to_category(cls, letter: str) -> str:
        if letter.upper() == "C":
            return cls.cellular_component
        elif letter.upper() == "P":
            return cls.biological_process
        elif letter.upper() == "F":
            return cls.molecular_function
        else:
            raise ValueError(
                f"'{letter}' is not a supported shorthand category."
            )

    @classmethod
    def choices(cls):
        return [(c, c) for c in cls.list()]


class Columns:
    source: str = "source"
    target: str = "target"
    gene_source = "gene_source"
    gene_target = "gene_target"
    label: str = "label"
    pubmed: str = "pubmed"
    psimi: str = "psimi"
    experiment_type: str = "experiment_type"
    go_mf: str = "go_mf"
    go_bp: str = "go_bp"
    go_cc: str = "go_cc"
    interpro: str = "interpro"
    keyword: str = "keyword"
    pfam: str = "pfam"
    database: str = "database"


class Paths:
    go_obo: Path = DATA_DIR / "go.obo.gz"
    interpro_entries: Path = DATA_DIR / "entry.list"
    pfam_clans: Path = DATA_DIR / "Pfam-A.clans.tsv.gz"

    # Networks
    hprd_ptms: Path = NETWORKS_DIR / "POST_TRANSLATIONAL_MODIFICATIONS.txt"
    hprd_xref: Path = NETWORKS_DIR / "HPRD_ID_MAPPINGS.txt"
    pina2_mitab: Path = NETWORKS_DIR / "Homo-sapiens-20140521.tsv.gz"
    bioplex: Path = NETWORKS_DIR / "BioPlex_interactionList_v4a.tsv.gz"
    innate_all: Path = NETWORKS_DIR / "all.mitab.gz"
    innate_curated: Path = NETWORKS_DIR / "innatedb_ppi.mitab.gz"

    # Path where classifiers will be saved.
    trained_models: Path = MODELS_DIR
    kegg_cache: Path = DATA_DIR / "kegg.cache.gz"
    uniprot_cache: Path = DATA_DIR / "uniprot.cache.gz"


class Urls:
    interpro_entries: str = (
        "ftp://ftp.ebi.ac.uk/pub/databases/interpro/entry.list"
    )
    go_obo: str = "http://purl.obolibrary.org/obo/go.obo"
    pfam_clans: str = (
        "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/"
        "current_release/Pfam-A.clans.tsv.gz"
    )

    # Networks
    pina2_mitab: str = (
        "http://omics.bjcancer.org/pina/download/Homo%20sapiens-20140521.tsv"
    )
    bioplex: str = (
        "http://bioplex.hms.harvard.edu/data/BioPlex_interactionList_v4a.tsv"
    )
    innate_curated: str = (
        "http://www.innatedb.com/download/interactions/innatedb_ppi.mitab.gz"
    )
    innate_all: str = (
        "http://www.innatedb.com/download/interactions/all.mitab.gz"
    )
