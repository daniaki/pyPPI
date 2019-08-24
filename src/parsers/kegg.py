from itertools import product
from typing import Dict, Generator, Iterable, List, Optional

import tqdm
import pandas as pd

from ..clients import kegg
from .types import InteractionData


def parse_interactions(
    exclude_labels: Iterable[str] = (
        "indirect effect",
        "compound",
        "hidden compound",
        "state change",
        "missing interaction",
    ),
    show_progress: bool = True,
    client: Optional[kegg.Kegg] = None,
) -> Generator[InteractionData, None, None]:

    # Set up client, download kegg -> uniprot mapping and all pathways.
    if not client:
        client = kegg.Kegg()
    to_uniprot: Dict[str, List[str]] = client.convert("hsa", "uniprot")
    pathway_accessions: List[str] = list(client.pathways("hsa")["accession"])
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

    for row in interactions.to_dict("record"):
        sources = to_uniprot.get(row["source"], None)
        targets = to_uniprot.get(row["target"], None)
        labels = [l for l in row["label"] if l not in exclude_labels]

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
                yield InteractionData(
                    source=source,
                    target=target,
                    organism=9606,
                    labels=labels,
                    databases=["KEGG"],
                )
