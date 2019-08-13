from pathlib import Path
from typing import Generator, Iterable, Union, Callable

from idutils import is_uniprot

from ..validators import validate_accession

from .types import InteractionData
from . import open_file


def parse_interactions(
    path: Union[str, Path],
    databases: Iterable[str] = (),
    sep: str = "\t",
    validator: Callable = is_uniprot,
    formatter: Callable = str.upper,
    header: bool = True,
) -> Generator[InteractionData, None, None]:
    """
    Parsing function a generic edgelist file.
    
    Parameters
    ----------
    path : str | Path
        Path to file to parse.

    databases: list[str], optional
        The databases an edgelist was downloaded from.

    sep: str, optional 
        File column separator.

    formatter : callable, optional.
        String formatting function. Should return a string value and accept
        a single string input.
    
    validator : callable, optional.
        Validator function to check if an accession is valid. Should return
        a single boolean value and accept a single input value.

    header : bool, optional.
        Set as `True` if the input file has a header line.

    Returns
    -------
    Generator[Interaction, None, None]
    """
    source_idx = 0
    target_idx = 1

    with open_file(path, "rt") as handle:
        if header:
            handle.readline()  # Remove header
        for line in handle:
            xs = line.strip().split(sep)
            source = validate_accession(
                accession=xs[source_idx].strip().upper(),
                formatter=formatter,
                validator=validator,
            )
            target = validate_accession(
                accession=xs[target_idx].strip().upper(),
                formatter=formatter,
                validator=validator,
            )
            if not (source and target):
                continue

            yield InteractionData(
                source=source,
                target=target,
                organism=9606,
                databases=list(databases),
            )
