import logging
from itertools import product
from typing import Dict, Generator, Iterable, List, Optional

import pandas as pd
import tqdm

from ..clients import kegg
from ..settings import LOGGER_NAME
from ..validators import is_uniprot
from .types import InteractionData

logger = logging.getLogger(LOGGER_NAME)


def parse_interactions(
    exclude_labels: Iterable[str] = (
        "indirect effect",
        "compound",
        "hidden compound",
        "state change",
        "missing interaction",
    ),
    organism: str = "hsa",
    show_progress: bool = True,
    client: Optional[kegg.Kegg] = None,
) -> List[InteractionData]

    # Set up client, download kegg -> uniprot mapping and all pathways.
    if not client:
        client = kegg.Kegg()

    logger.info("Downloading Kegg to UniProt mapping.")
    to_uniprot: Dict[str, List[str]] = client.convert("hsa", "uniprot")

    logger.info("Downloading HSA pathways.")
    pathway_accessions: List[str] = list(
        client.pathways(organism)["accession"]
    )
    pathways: Generator[kegg.KeggPathway, None, None] = (
        client.pathway_detail(pathway) for pathway in pathway_accessions
    )
    if show_progress:
        pathways = tqdm.tqdm(pathways, total=len(pathway_accessions))

    # For each pathway, yield the interaction data for each interaction.
    # Stack all dataframes into a single dataframe.
    interactions: pd.DataFrame = pd.concat(
        [p.interactions for p in pathways], axis=0, ignore_index=True
    )

    # Roll up each row so that labels are combined into a single list.
    # Aggregation is based on source and target columns.
    interactions = (
        interactions.drop_duplicates(subset=None, keep="first", inplace=False)
        .groupby(by=["source", "target"], as_index=False)
        .agg(lambda args: list(sorted(set(",".join(args).split(",")))))
        .dropna(axis=0, how="any", inplace=False)
    )

    logger.info("Generating interactions.")
    result: List[InteractionData] = []
    for row in interactions.to_dict("record"):
        sources = to_uniprot.get(row["source"], None)
        targets = to_uniprot.get(row["target"], None)
        labels = list(
            sorted(
                set(l.lower() for l in row["label"] if l not in exclude_labels)
            )
        )

        # Continue if there are no labels to parse for the interaction.
        if not labels:
            continue

        # If any KeggIDs do not map to uniprot, skip.
        if not (sources and targets):
            continue
        else:
            # Otherwise product the lists and yield an interaction for
            # each source and target.
            for (source, target) in product(sources, targets):
                source = source.strip().upper()
                target = target.strip().upper()

                if (not is_uniprot(source)) or (not is_uniprot(target)):
                    raise ValueError(
                        f"Edge '{(source, target)}' contains invalid UniProt "
                        f"identifiers."
                    )

                result.append(
                    InteractionData(
                        source=source,
                        target=target,
                        labels=labels,
                        databases=["kegg"],
                    )
                )
