from logging import getLogger
from dataclasses import asdict
from collections import defaultdict
from typing import Iterable, List, Dict, Set, Any, Iterable, Optional

from peewee import ModelSelect
from peewee import fn

from ..clients.uniprot import UniprotClient, UniprotEntry
from ..parsers import types
from ..settings import DATABASE, LOGGER_NAME

from . import models


logger = getLogger(LOGGER_NAME)


@DATABASE.atomic()
def create_terms(
    terms: Iterable[Any], model: models.Annotation
) -> ModelSelect:
    # First bulk-create/get existing identifiers
    identifiers: Dict[str, models.Annotation] = {
        str(i): i
        for i in create_identifiers(
            identifiers=set(term.identifier for term in terms),
            model=getattr(model.identifier, "rel_model"),
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
        # Replace the str identifier in the dataclass instance with real
        # identifier so we can create a model instance from the asdict method
        # by argument unpacking.
        params = asdict(term)
        params["identifier"] = identifiers[term.identifier.upper()]
        if term.identifier.upper() not in existing:
            create.append(model(**params).format_for_save())

    # Bulk update/create then return a query of all instances matching
    # the identifiers in the dataclass objects from terms parameter.
    models.GeneOntologyTerm.bulk_create(create)
    return model.get_by_identifier(set(t.identifier.upper() for t in terms))


@DATABASE.atomic()
def create_identifiers(
    identifiers: Iterable[str], model: models.ExternalIdentifier
) -> ModelSelect:
    # Create new identifiers list with formatted accessions so we can
    # perform database lookups. This allows users to pass in un-prefixed ids
    instances = [model(identifier=i).format_for_save() for i in identifiers]
    identifiers = [str(i) for i in instances]
    existing = set(str(i) for i in model.get_by_identifier(identifiers))

    # Loop through and bulk create only the missing identifiers.
    create: List[models.ExternalIdentifier] = []
    instance: models.ExternalIdentifier
    for instance in instances:
        if str(instance) not in existing:
            create.append(instance)
    if create:
        model.bulk_create(create)

    return model.get_by_identifier(identifiers)


@DATABASE.atomic()
def create_gene_symbols(symbols: Iterable[str]) -> ModelSelect:
    # Create new symbols list with formatted text so we can
    # perform database lookups.
    instances = [models.GeneSymbol(text=s).format_for_save() for s in symbols]
    symbols = [str(i) for i in instances]
    existing = set(
        str(i)
        for i in models.GeneSymbol.select().where(
            fn.Upper(models.GeneSymbol.text) << set(symbols)
        )
    )

    # Loop through and bulk create only the missing instances.
    create: List[models.GeneSymbol] = []
    instance: models.GeneSymbol
    for instance in instances:
        if str(instance) not in existing:
            create.append(instance)
    if create:
        models.GeneSymbol.bulk_create(create)

    return models.GeneSymbol.select().where(
        fn.Upper(models.GeneSymbol.text) << set(symbols)
    )


@DATABASE.atomic()
def create_evidence(
    evidences: Iterable[types.InteractionEvidenceData]
) -> List[models.InteractionEvidence]:
    # Bulk create identifiers
    create_identifiers(
        set(e.pubmed for e in evidences), model=models.PubmedIdentifier
    )
    create_identifiers(
        set(e.psimi for e in evidences if e.psimi),
        model=models.PsimiIdentifier,
    )

    instances: List[models.InteractionEvidence] = []
    evidence: types.InteractionEvidenceData
    for evidence in evidences:
        # Format identifier strings before query.
        instance, _ = models.InteractionEvidence.get_or_create(
            pubmed=models.PubmedIdentifier.get(
                identifier=models.PubmedIdentifier(identifier=evidence.pubmed)
                .format_for_save()
                .identifier
            ),
            psimi=models.PsimiIdentifier.get_or_none(
                identifier=models.PsimiIdentifier(identifier=evidence.psimi)
                .format_for_save()
                .identifier
            ),
        )
        instances.append(instance)
    return instances


@DATABASE.atomic()
def create_proteins(proteins: Iterable[str]) -> List[models.Protein]:
    client = UniprotClient()
    entries: List[UniprotEntry] = client.get_entries(proteins)

    # Loop through and collect the identifiers, genes and all annotations to
    # bulk create first.
    terms: Dict[str, Set[Any]] = defaultdict(set)
    gene_symbols: Set[types.GeneData] = set()
    accessions: Set[str] = set()
    entry: UniprotEntry
    for entry in entries:
        accessions |= set(entry.accessions)
        terms["go"] |= set(entry.go_terms)
        terms["pfam"] |= set(entry.pfam_terms)
        terms["interpro"] |= set(entry.interpro_terms)
        terms["keyword"] |= set(entry.keywords)
        gene_symbols |= set(entry.genes)

    # Bulk create and create lookup dictionaries indexed by identifier
    # for identifiers and terms, and gene symbol for the genes.
    create_terms(terms["go"], model=models.GeneOntologyTerm)
    create_terms(terms["pfam"], model=models.PfamTerm)
    create_terms(terms["interpro"], model=models.InterproTerm)
    create_terms(terms["keyword"], model=models.Keyword)
    create_gene_symbols(set(g.symbol for g in gene_symbols))
    create_identifiers(accessions, model=models.UniprotIdentifier)

    # Loop through and intialize protein models
    protein_models: List[models.Protein] = []
    existing: Dict[str, models.Protein] = {
        str(protein): protein
        for protein in models.Protein.get_by_identifier(
            [e.primary_accession for e in entries if e.primary_accession]
        )
    }
    for entry in entries:
        # Check if a protein instance already exists and update. Otherwise
        # create a new instance.
        accession = entry.primary_accession

        # Check if a primary gene exists.
        gene: Optional[models.GeneSymbol] = None
        if entry.primary_gene:
            gene = models.GeneSymbol.get_or_none(
                text=entry.primary_gene.symbol
            )

        if accession not in existing:
            instance = models.Protein.create(
                identifier=models.UniprotIdentifier.get(identifier=accession),
                organism=entry.taxonomy,
                sequence=entry.sequence,
                reviewed=entry.reviewed,
                gene=gene,
            )
        else:
            instance = existing[accession]
            instance.organism = entry.taxonomy
            instance.sequence = entry.sequence
            instance.reviewed = entry.reviewed
            instance.save()

        # Update the M2M relations.
        instance.go_terms = models.GeneOntologyTerm.get_by_identifier(
            set(x.identifier for x in entry.go_terms)
        )
        instance.pfam_terms = models.PfamTerm.get_by_identifier(
            set(x.identifier for x in entry.pfam_terms)
        )
        instance.interpro_terms = models.InterproTerm.get_by_identifier(
            set(x.identifier for x in entry.interpro_terms)
        )
        instance.keywords = models.Keyword.get_by_identifier(
            set(x.identifier for x in entry.keywords)
        )
        # Do an upper-case filter on uniprot identifiers and gene symbols.
        # Use upper-case since the save method will call str.upper before
        # commiting to the database.
        instance.aliases = models.UniprotIdentifier.select().where(
            fn.Upper(models.UniprotIdentifier.identifier)
            << set(entry.alias_accessions)
        )
        instance.alt_genes = models.GeneSymbol.select().where(
            fn.Upper(models.UniprotIdentifier.identifier)
            << set(g.symbol.upper() for g in entry.synonym_genes)
        )

        # Append new/updated row to list to return to caller.
        protein_models.append(instance)

    return protein_models


@DATABASE.atomic()
def create_interactions(
    interactions: Iterable[types.InteractionData]
) -> List[models.Interaction]:
    # First aggregate all interaction data instances.
    aggregated: Dict[types.InteractionData, types.InteractionData] = dict()
    interaction: types.InteractionData
    for interaction in interactions:
        # Interactions are hashable based on source, target and organism code.
        # Order of source target is important.
        if interaction in aggregated:
            aggregated[interaction] += interaction
        else:
            aggregated[interaction] = interaction

    # Collect all UniProt identifiers to pass to create_proteins.
    # create_proteins will download the uniprot entries and create the
    # co-responding the database models
    uniprot_accessions: Set[str] = set()
    databases: Set[str] = set()
    labels: Set[str] = set()
    evidences: Set[types.InteractionEvidenceData] = set()
    # Loop over aggregated interactions (value, not the key).
    for _, interaction in aggregated.items():
        if interaction.source:
            uniprot_accessions.add(interaction.source)
        if interaction.target:
            uniprot_accessions.add(interaction.target)
        # Collect database and evidence terms
        evidences |= set(interaction.evidence)
        databases |= set(interaction.databases)
        labels |= set(interaction.labels)

    # Bulk create proteins, evidence terms and database names.
    proteins: Dict[str, models.Protein] = {
        str(protein): protein
        for protein in create_proteins(uniprot_accessions)
    }
    evidence_instances: Dict[int, models.InteractionEvidence] = {
        hash(evidence): evidence for evidence in create_evidence(evidences)
    }
    for db in databases:
        models.InteractionDatabase.get_or_create(name=db)
    for label in labels:
        models.InteractionLabel.get_or_create(text=label)

    # Loop through and update/create interactions.
    instances: List[models.Interaction] = []
    for _, interaction in aggregated.items():
        # Skip interactions for which no uniprot entries could be found.
        if proteins.get(interaction.source, None):
            logger.warning(
                f"Skipping interaction {interaction}. No source found "
                f"in UniProt."
            )
            continue
        if proteins.get(interaction.target, None):
            logger.warning(
                f"Skipping interaction {interaction}. No target found "
                f"in UniProt."
            )
            continue

        # Get/Create instance and update M2M fields.
        instance, _ = models.Interaction.get_or_create(
            source=proteins[interaction.source],
            target=proteins[interaction.target],
            organism=interaction.organism,
        )
        # Filter by capitalization since save will capitalize values before
        # commiting to databse.
        instance.labels = models.InteractionLabel.filter(
            fn.Capitalize(models.InteractionLabel.text)
            << set(i.capitalize() for i in interaction.labels)
        )
        instance.databases = models.InteractionDatabase.filter(
            fn.Capitalize(models.InteractionDatabase.text)
            << set(d.capitalize() for d in interaction.databases)
        )
        instance.evidence = [
            evidence_instances[hash(e)] for e in interaction.evidence
        ]
        instances.append(instance)

    return instances
