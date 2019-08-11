from typing import Dict, List

from goatools.obo_parser import GODag, GOTerm

from ..database.models import GeneOntologyTerm
from . import open_file
from .types import GeneOntologyTermData


def parse_go_obo(path: str) -> List[GeneOntologyTermData]:
    """
    Parses the GO obo saved at `path` into a list of `GeneOntologyTermData`
    instances.
    
    Parameters
    ----------
    path : str
        Path to load obo file from.
    
    Returns
    -------
    list[GeneOntologyTermData]
        A list of `GeneOntologyTermData` instances.
    """
    terms: List[GeneOntologyTermData] = []
    dag: GODag = GODag(obo_file=path)
    namespaces = {
        "BP": GeneOntologyTerm.Category.biological_process,
        "MF": GeneOntologyTerm.Category.molecular_function,
        "CC": GeneOntologyTerm.Category.cellular_compartment,
    }
    term: GOTerm
    for _, term in dag.items():
        terms.append(
            GeneOntologyTermData(
                identifier=term.item_id,
                name=term.name,
                obsolete=term.is_obsolete,
                description=None,
                category=namespaces[term.namespace],
            )
        )
    return terms

