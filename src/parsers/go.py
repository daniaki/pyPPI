from typing import Dict, List

from goatools.obo_parser import GODag, GOTerm

from ..constants import GeneOntologyCategory
from ..utilities import is_null
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
        "BP": GeneOntologyCategory.biological_process,
        "MF": GeneOntologyCategory.molecular_function,
        "CC": GeneOntologyCategory.cellular_component,
        "biological_process": GeneOntologyCategory.biological_process,
        "molecular_function": GeneOntologyCategory.molecular_function,
        "cellular_component": GeneOntologyCategory.cellular_component,
    }
    term: GOTerm
    for _, term in dag.items():
        terms.append(
            GeneOntologyTermData(
                identifier=term.item_id,
                name=None if is_null(term.name) else term.name.strip(),
                obsolete=term.is_obsolete,
                description=None,
                category=namespaces[term.namespace],
            )
        )
    return terms

