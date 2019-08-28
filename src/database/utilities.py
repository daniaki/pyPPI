from logging import getLogger
from dataclasses import asdict
from typing import Iterable, List, Dict, Set, Any, Iterable, Optional

from peewee import ModelSelect
from peewee import fn

from ..clients.uniprot import UniprotClient, UniprotEntry
from ..parsers import types
from ..settings import DATABASE, LOGGER_NAME

from . import models


logger = getLogger(LOGGER_NAME)


@DATABASE.atomic()
def create_annotations(
    terms: Iterable[Any], dataclass: Any, model: models.Annotation
) -> ModelSelect:
    # First bulk-create/get existing identifiers
    identifiers: Dict[str, models.Annotation] = {
        str(i): i
        for i in create_identifiers(
            identifiers=set(term.identifier for term in terms), model=model
        )
    }

    # Select all models that already exist using the GO identifier
    existing: Dict[str, models.Annotation] = {
        str(i): i for i in model.get_by_identifier(identifiers.keys())
    }

    # Loop through each dataclass instance and create a list of models to
    # to bulk create.
    create: List[models.Annotation] = []
    # Term is a dataclass term defined in parser.types
    for term in terms:
        instance: Optional[models.Annotation] = existing.get(
            term.identifier, None
        )
        # Replace the str identifier in the dataclass instance with real
        # identifier so we can create a model instance from the asdict method
        # by argument unpacking.
        term.identifier = identifiers[term.identifier]
        if not instance:
            create.append(model(**asdict(term)))

    # Bulk update/create then return a query of all instances matching
    # the identifiers in the dataclass objects from terms parameter.
    models.GeneOntologyTerm.bulk_create(create)
    return (
        models.GeneOntologyTerm.select()
        .join(models.GeneOntologyIdentifier)
        .where(models.GeneOntologyIdentifier.identifier << identifiers)
    )


@DATABASE.atomic()
def create_identifiers(
    identifiers: Iterable[str], model: models.ExternalIdentifier
) -> ModelSelect:
    for identifier in identifiers:
        model.get_or_create(identifier=identifier.upper())
    return model.select().where(
        fn.Upper(model.identifier) << set(i.upper() for i in identifier)
    )


@DATABASE.atomic()
def create_gene_symbols(symbols: Iterable[str]) -> ModelSelect:
    for symbol in symbols:
        models.GeneSymbol.get_or_create(text=symbol.upper())
    return models.GeneSymbol.select().where(
        fn.Upper(models.GeneSymbol.text) << set(s.upper() for s in symbols)
    )


@DATABASE.atomic()
def create_proteins(proteins: Iterable[str]) -> ModelSelect:
    client = UniprotClient()
    entries: List[UniprotEntry] = client.get_entries(proteins)

    # Loop through and collec the identifiers, genes and all annotations to
    # bulk create first.
    annotations: Dict[str, Any] = {}
    entry: UniprotEntry
    for entry in entries:
        entry.

    # Bulk create and create lookup dictionaries indexed by identifier
    # for identifiers and annotations, and gene symbol for the genes.

    # Loop through and intialize protein models

    # Bulk create new proteins

    # Update M2M relationships for each protein


@DATABASE.atomic()
def create_interactions(interactions: Iterable[types.InteractionData]) -> None:
    pass
    # # Collect all UniProt identifiers to pass to create_proteins.
    # # create_proteins will download the uniprot entries and create the
    # # co-responding the database models
    # proteins: List[str] = []
    # interaction: InteractionData
    # for interaction in interactions:
    #     if interaction.source:
    #         proteins.append(interaction.source)
    #     if interaction.target:
    #         proteins.append(interaction.target)

    # proteins: Dict[str, models.Protein] = create_proteins(proteins)

    # for interaction in interactions:
    #     # Skip interactions for which no uniprot entries could be found.
    #     if proteins.get(interaction.source, None):
    #         logger.warning(
    #             f"Skipping interaction {interaction}. No source found "
    #             f"on UniProt."
    #         )
    #         continue
    #     if proteins.get(interaction.target, None):
    #         logger.warning(
    #             f"Skipping interaction {interaction}. No target found "
    #             f"on UniProt."
    #         )
    #         continue

    #     models.Interaction.update_or_create(
    #         source=proteins[interaction.source],
    #         target=proteins[interaction.target],
    #         organism=interaction.organism,
    #         labels=interaction.labels,
    #         psimi_ids=interaction.psimi_ids,
    #         pubmed_ids=interaction.pubmed_ids,
    #         experiment_types=interaction.experiment_types,
    #         databases=interaction.databases,
    #     )
