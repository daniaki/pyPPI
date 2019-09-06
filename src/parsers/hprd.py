"""
This module provides functionality to mine interactions with labels from
HPRD flat files.
"""

from collections import OrderedDict, defaultdict
from itertools import product
from typing import (
    Any,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Iterable,
    Union,
)

from ..constants import PSIMI_NAME_TO_IDENTIFIER, UNIPROT_ORD_KEY, Columns
from ..utilities import is_null
from ..validators import is_pubmed, is_uniprot
from . import open_file
from .types import InteractionData, InteractionEvidenceData


__all__ = [
    "parse_interactions",
    "parse_xref_mapping",
    "parse_ptm",
    "PTMEntry",
    "HPRDXrefEntry",
]


SUBTYPES_TO_EXCLUDE: List[str] = []

__PTM_FIELDS: DefaultDict[str, Optional[Any]] = defaultdict(lambda: None)
__PTM_FIELDS["substrate_hprd_id"] = None
__PTM_FIELDS["substrate_gene_symbol"] = None
__PTM_FIELDS["substrate_isoform_id"] = None
__PTM_FIELDS["substrate_refseq_id"] = None
__PTM_FIELDS["site"] = None
__PTM_FIELDS["residue"] = None
__PTM_FIELDS["enzyme_name"] = None
__PTM_FIELDS["enzyme_hprd_id"] = None
__PTM_FIELDS["modification_type"] = None
__PTM_FIELDS["experiment_type"] = None
__PTM_FIELDS["reference_id"] = None  # pubmed ids
__PTM_INDEX = {k: i for (i, k) in enumerate(__PTM_FIELDS.keys())}

__HPRD_XREF_FIELDS: DefaultDict[str, Optional[Any]] = defaultdict(lambda: None)
__HPRD_XREF_FIELDS["hprd_id"] = None
__HPRD_XREF_FIELDS["gene_symbol"] = None
__HPRD_XREF_FIELDS["nucleotide_accession"] = None
__HPRD_XREF_FIELDS["protein_accession"] = None
__HPRD_XREF_FIELDS["entrezgene_id"] = None
__HPRD_XREF_FIELDS["omim_id"] = None
__HPRD_XREF_FIELDS["swissprot_id"] = None
__HPRD_XREF_FIELDS["main_name"] = None
__HPRD_XREF_INDEX = {k: i for (i, k) in enumerate(__HPRD_XREF_FIELDS.keys())}


class PTMEntry:
    """
    Class to hold row data from the Post translational mod text file.
    """

    def __init__(
        self,
        enzyme_hprd_id: Optional[str] = None,
        substrate_hprd_id: Optional[str] = None,
        modification_type: Optional[str] = None,
        reference_id: Union[str, Iterable[str]] = (),
        experiment_type: Union[str, Iterable[str]] = (),
        **kwargs,
    ):
        self.enzyme_hprd_id = enzyme_hprd_id
        if self.enzyme_hprd_id == "-":
            self.enzyme_hprd_id = None

        self.substrate_hprd_id = substrate_hprd_id
        if self.substrate_hprd_id == "-":
            self.substrate_hprd_id = None

        self.modification_type = modification_type
        if self.modification_type == "-":
            self.modification_type = None

        self.reference_id = reference_id
        if not self.reference_id:
            self.reference_id = []
        if isinstance(self.reference_id, str):
            if self.reference_id == "-":
                self.reference_id = []
            else:
                self.reference_id = [self.reference_id]

        self.experiment_type = experiment_type
        if not self.experiment_type:
            self.experiment_type = []
        if isinstance(self.experiment_type, str):
            if self.experiment_type == "-":
                self.experiment_type = []
            else:
                self.experiment_type = [self.experiment_type]

        self.__dict__.update(**kwargs)


class HPRDXrefEntry:
    """
    Class to hold row data from the HPRD text file.
    """

    def __init__(
        self,
        hprd_id: str,
        gene_symbol: Optional[str] = None,
        swissprot_id: Iterable[str] = (),
        **kwargs,
    ):
        self.hprd_id = hprd_id

        self.gene_symbol = gene_symbol
        if not self.gene_symbol or self.gene_symbol == "-":
            self.gene_symbol = None

        self.swissprot_id = swissprot_id
        if not self.swissprot_id or self.swissprot_id == "-":
            self.swissprot_id = []

        self.__dict__.update(**kwargs)


def parse_ptm(
    path: str, header: bool = False, sep: str = "\t"
) -> List[PTMEntry]:
    """
    Parse HPRD post_translational_modifications file.
    
    Parameters
    ----------
    path : str
        Open file handle pointing to the HPRD PTM file to parse.
    
    header : bool, optional.
        True if file has header. Default is False.
    
    sep : str, optional.
        Column separator value.
    
    Returns
    -------
    List[PTMEntry]
        List of PTMEntry objects.
    """
    with open_file(path, "rt") as handle:
        ptms = []
        if header:
            handle.readline()
        for line in handle:
            class_fields = __PTM_FIELDS.copy()
            xs = line.strip().split(sep)
            for k in __PTM_FIELDS.keys():
                raw_data: Optional[str] = xs[__PTM_INDEX[k]]
                if is_null(raw_data) or raw_data == "-":
                    raw_data = None

                data: Union[List[str], str, None] = []
                if k == "reference_id" and raw_data:
                    data = [x.strip() for x in raw_data.split(",")]
                elif k == "experiment_type" and raw_data:
                    data = [x.strip() for x in raw_data.split(";")]
                else:
                    data = raw_data or None

                class_fields[k] = data

            ptms.append(PTMEntry(**class_fields))
        return ptms


def parse_xref_mapping(
    path: str, header: bool = False, sep: str = "\t"
) -> Dict[str, HPRDXrefEntry]:
    """
    Parse a hprd mapping file into HPRDXref Objects.
    
    Parameters
    ----------
    path : str
        Open file handle pointing to the HPRD mapping file to parse.
    
    header : bool, optional
        True if file has header. Default is False.
    
    sep : str, optional
        Column separator value.
    
    Returns
    -------
    dict[str, HPRDXrefEntry]
        Dictionary of HPRDXrefEntry objects indexed by hprd accession.
    """
    with open_file(path, "rt") as handle:
        xrefs = {}
        if header:
            handle.readline()
        for line in handle:
            class_fields = __HPRD_XREF_FIELDS.copy()
            xs = line.strip().split(sep)
            for k in __HPRD_XREF_FIELDS.keys():
                raw_data: Optional[str] = xs[__HPRD_XREF_INDEX[k]]
                if is_null(raw_data) or raw_data == "-":
                    raw_data = None

                data: Union[List[str], str, None] = None
                if k == "swissprot_id" and raw_data:
                    data = [x.strip() for x in raw_data.split(",")]
                else:
                    data = raw_data or None

                class_fields[k] = data

            xrefs[xs[0]] = HPRDXrefEntry(**class_fields)
        return xrefs


def parse_interactions(
    ptms: Iterable[PTMEntry], xrefs=Dict[str, HPRDXrefEntry]
) -> Generator[InteractionData, None, None]:
    """
    Parse the FLAT_FILES from HPRD into a list of interactions with pubmed and
    experiment type annotations.
    
    Parameters
    ----------
    ptms : Iterable[PTMEntry]
        Path to the HPRD PTM file to parse.
    
    xrefs : Dict[str, HPRDXrefEntry]
        Path to the HPRD mapping file to parse.
    
    Returns
    -------
    list[types.InteractionData]
    """
    for ptm in ptms:
        label = None
        if ptm.modification_type is not None:
            label = ptm.modification_type.strip().capitalize()
            if not label:
                ptm.modification_type = None
                label = None

        if (
            (ptm.enzyme_hprd_id is None)
            or (ptm.substrate_hprd_id is None)
            or (label is None)
        ):
            continue

        # Remove duplicated pubmed ids
        evidence = [
            InteractionEvidenceData(pubmed=x.strip().upper())
            for x in set(ptm.reference_id)
            if not is_null(x) and is_pubmed(x.strip().upper())
        ]

        uniprot_sources = getattr(
            xrefs.get(ptm.enzyme_hprd_id, None), "swissprot_id"
        )
        uniprot_targets = getattr(
            xrefs.get(ptm.substrate_hprd_id, None), "swissprot_id"
        )

        if uniprot_sources and uniprot_targets:
            for (source, target) in product(uniprot_sources, uniprot_targets):
                source = source.strip().upper()
                target = target.strip().upper()

                if not is_uniprot(source):
                    raise ValueError(
                        f"Source '{source}' is not a valid UniProt identifier."
                    )
                if not is_uniprot(target):
                    raise ValueError(
                        f"Target '{target}' is not a valid UniProt identifier."
                    )

                yield InteractionData(
                    source=source,
                    target=target,
                    labels=[label.lower()],
                    evidence=evidence,
                    databases=["hprd"],
                )
