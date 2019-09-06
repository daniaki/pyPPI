import gzip
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from goatools.obo_parser import GODag, GOTerm

from ..constants import GeneOntologyCategory
from ..settings import LOGGER_NAME
from ..utilities import is_null
from ..validators import is_go
from . import open_file
from .types import GeneOntologyTermData

logger = logging.getLogger(LOGGER_NAME)


def parse_go_obo(path: Union[str, Path]) -> List[GeneOntologyTermData]:
    """
    Parses the GO obo saved at `path` into a list of `GeneOntologyTermData`
    instances.
    
    Parameters
    ----------`
    path : str
        Path to load obo file from.
    
    Returns
    -------
    list[GeneOntologyTermData]
        A list of `GeneOntologyTermData` instances.
    """
    dag: GODag
    if str(path).endswith(".gz"):
        # Unzip into a temp file.
        tmp = tempfile.NamedTemporaryFile(
            mode="wt", suffix=".obo", prefix="uzipped"
        )
        new_path = Path(tempfile.gettempdir()) / str(tmp.name)
        logger.info(f"Unzipping '{path}' into '{new_path}'")
        with gzip.open(path, "rt") as zipped:
            tmp.write(zipped.read())
        dag = GODag(obo_file=new_path)
        tmp.close()
    else:
        dag = GODag(obo_file=path)

    terms: List[GeneOntologyTermData] = []
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
        assert is_go(term.item_id.strip().upper())
        terms.append(
            GeneOntologyTermData(
                identifier=term.item_id.strip().upper(),
                name=None if is_null(term.name) else term.name.strip(),
                obsolete=term.is_obsolete,
                description=None,
                category=namespaces[term.namespace],
            )
        )

    return terms
