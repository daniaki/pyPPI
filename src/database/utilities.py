from logging import getLogger
from typing import Iterable, List, Dict

from ..clients import uniprot
from ..parsers.types import InteractionData, GeneOntologyTermData
from ..settings import DATABASE, LOGGER_NAME

from . import models


logger = getLogger(LOGGER_NAME)


@DATABASE.atomic()
def create_go_terms(terms: Iterable[GeneOntologyTermData]):
    pass


@DATABASE.atomic()
def create_pfam_terms(terms):
    pass


@DATABASE.atomic()
def create_interpro_terms(terms):
    pass


@DATABASE.atomic()
def create_keywords(keywords):
    pass


@DATABASE.atomic()
def create_identifiers(identifiers, model):
    pass


@DATABASE.atomic()
def create_gene_symbols(symbols):
    pass


@DATABASE.atomic()
def create_experiment_types(types):
    pass


@DATABASE.atomic()
def create_interactions(interactions: Iterable[InteractionData]) -> None:
    # Collect all UniProt identifiers to pass to create_proteins.
    # create_proteins will download the uniprot entries and create the
    # co-responding the database models
    proteins: List[str] = []
    interaction: InteractionData
    for interaction in interactions:
        if interaction.source:
            proteins.append(interaction.source)
        if interaction.target:
            proteins.append(interaction.target)

    proteins: Dict[str, models.Protein] = create_proteins(proteins)

    for interaction in interactions:
        # Skip interactions for which no uniprot entries could be found.
        if proteins.get(interaction.source, None):
            logger.warning(
                f"Skipping interaction {interaction}. No source found "
                f"on UniProt."
            )
            continue
        if proteins.get(interaction.target, None):
            logger.warning(
                f"Skipping interaction {interaction}. No target found "
                f"on UniProt."
            )
            continue

        models.Interaction.update_or_create(
            source=proteins[interaction.source],
            target=proteins[interaction.target],
            organism=interaction.organism,
            labels=interaction.labels,
            psimi_ids=interaction.psimi_ids,
            pubmed_ids=interaction.pubmed_ids,
            experiment_types=interaction.experiment_types,
            databases=interaction.databases,
        )


@DATABASE.atomic()
def create_proteins(proteins: Iterable[str]) -> Dict[str, models.Protein]:
    client = uniprot.UniprotClient()
    proteins: List[uniprot.UniprotEntry] = client.get_entries(proteins)
    return {}
