from collections import defaultdict
from dataclasses import asdict, astuple
from logging import getLogger
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from peewee import ModelSelect, fn

from ..clients.uniprot import UniprotClient, UniprotEntry
from ..parsers import types
from ..settings import DATABASE, LOGGER_NAME
from . import models

logger = getLogger(LOGGER_NAME)


@DATABASE.atomic()
def create_terms(
    terms: Iterable[Any], model: models.Annotation
) -> ModelSelect:
    """
    Bulk creates annotations for a given annotation model. Also bulk creates
    the required identifiers for each annotation instance.

    Existing annotations will NOT be updated.

    Parameters
    ----------
    terms : Iterable[Any]
        A term dataclass type from the `parsers.types` module.
    model : models.Annotation
        Database model to create rows for.

    Returns
    -------
    ModelSelect
    """
    # First bulk-create/get existing identifiers
    identifier_model: models.ExternalIdentifier = getattr(
        model.identifier, "rel_model"
    )
    identifiers: Dict[str, models.Annotation] = {
        str(i): i
        for i in create_identifiers(
            identifiers=set(term.identifier for term in terms),
            model=identifier_model,
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
    for term in set(terms):
        # Replace the str identifier in the dataclass instance with real
        # identifier so we can create a model instance from the asdict method
        # by argument unpacking.
        params = asdict(term)
        normalized = identifier_model.format(term.identifier)
        params["identifier"] = identifiers[normalized]
        if normalized not in existing:
            create.append(model(**params).format_for_save())

    # Bulk update/create then return a query of all instances matching
    # the identifiers in the dataclass objects from terms parameter.
    model.bulk_create(create, batch_size=250)
    # Filter by normalized identifiers
    return model.get_by_identifier(set(identifiers.keys()))


@DATABASE.atomic()
def create_identifiers(
    identifiers: Iterable[str], model: models.ExternalIdentifier
) -> ModelSelect:
    """
    Creates identifiers for the table defined by `model`.

    Parameters
    ----------
    identifiers : Iterable[str]
        Iterable of string accessions. These will be formatted.
    model : models.ExternalIdentifier
        Model to create identifiers for.

    Returns
    -------
    ModelSelect
    """
    # Create new identifiers list with formatted accessions so we can
    # perform database lookups.
    normalized: Set[str] = set(i.strip().upper() for i in identifiers)

    # Get all existing model identifiers
    existing: Set[str] = set(
        str(i) for i in model.get_by_identifier(normalized)
    )

    # Loop through and bulk create only the missing identifiers.
    create: List[models.ExternalIdentifier] = []
    for identifier in normalized:
        if identifier not in existing:
            create.append(model(identifier=identifier).format_for_save())
    if create:
        model.bulk_create(create, batch_size=250)

    return model.get_by_identifier(normalized)


@DATABASE.atomic()
def create_gene_symbols(symbols: Iterable[str]) -> ModelSelect:
    """
    Creates formated gene symbol rows from a list of gene symbols.

    New PubMed and MI Ontology identifiers will be bulk created.

    Parameters
    ----------
    symbols : Iterable[str]
        Symbols to create.

    Returns
    -------
    ModelSelect
    """
    # Create new symbols list with formatted text so we can
    # perform database lookups.
    normalized = set(s.strip().upper() for s in symbols)

    # Get all existing model identifiers. Table shouldn't be too big.
    existing = set(str(i) for i in models.GeneSymbol.all())

    # Loop through and bulk create only the missing instances.
    create: List[models.GeneSymbol] = []
    for symbol in normalized:
        if symbol not in existing:
            create.append(models.GeneSymbol(text=symbol).format_for_save())
    if create:
        models.GeneSymbol.bulk_create(create, batch_size=250)

    return models.GeneSymbol.select().where(
        fn.Upper(models.GeneSymbol.text) << set(normalized)
    )


@DATABASE.atomic()
def create_evidence(
    evidences: Iterable[types.InteractionEvidenceData]
) -> List[models.InteractionEvidence]:
    """
    Creates evidence instances from an iterable of `InteractionEvidenceData`.

    New formated PubMed and MI Ontology identifiers will be bulk created.

    Parameters
    ----------
    evidences : Iterable[types.InteractionEvidenceData]
        Items to create rows for.

    Returns
    -------
    List[models.InteractionEvidence]
    """
    # Bulk create identifiers
    create_identifiers(
        identifiers=set(e.pubmed for e in evidences),
        model=models.PubmedIdentifier,
    )
    create_identifiers(
        identifiers=set(e.psimi for e in evidences if e.psimi),
        model=models.PsimiIdentifier,
    )

    normalized = set(e.normalize() for e in evidences)
    instances: List[models.InteractionEvidence] = []
    evidence: types.InteractionEvidenceData
    for evidence in normalized:
        # Format identifier strings before query.
        print(evidence)
        instance, _ = models.InteractionEvidence.get_or_create(
            pubmed=models.PubmedIdentifier.get(identifier=evidence.pubmed),
            psimi=(
                None
                if not evidence.psimi
                else models.PsimiIdentifier.get(identifier=evidence.psimi)
            ),
        )
        instances.append(instance)
    return instances


def update_accessions(
    interactions: Iterable[types.InteractionData],
    mapping: Dict[str, Sequence[str]],
    keep_isoforms: bool = True,
) -> Iterable[types.InteractionData]:
    """
    Takes an iterable of `InteractionData` instances and maps their source and
    target accessions to the first element in the `mapping` dictionary value
    for accession. Interactions witj accessions that cannot be mapped will be 
    discarded. 

    Parameters
    ----------
    interactions : Iterable[types.InteractionData]
        Interactions to map.
    
    mapping : Dict[str, Sequence[str]]
        Mapping from a UniProt accession to a list of most recent UniProt
        accessions. The first element in this list will be considered the 
        canonical accession.

    keep_isoforms: bool, default True.
        Do not map interactions with an isoform UniProt identifier to it's
        primary stable identifier (the same but without the digit extension)

    Returns
    -------
    Iterable[types.InteractionData]
    """
    updated_interactions: List[types.InteractionData] = []
    interaction: types.InteractionData
    for interaction in interactions:
        new_sources = mapping.get(interaction.source, [])
        new_targets = mapping.get(interaction.target, [])

        if len(new_sources) and len(new_targets):
            primary_source = new_sources[0]
            primary_target = new_targets[0]

            if keep_isoforms:
                source_is_isoform = len(interaction.source.split("-")) == 2
                target_is_isoform = len(interaction.target.split("-")) == 2
                # Only retain isoform it the isoform identifier matches the
                # mapped primary identifier.
                source_isoform_matches_primary = (
                    interaction.source.split("-")[0] == primary_source
                )
                target_isoform_matches_primary = (
                    interaction.target.split("-")[0] == primary_target
                )
                if source_is_isoform and source_isoform_matches_primary:
                    primary_source = interaction.source
                if target_is_isoform and target_isoform_matches_primary:
                    primary_source = interaction.target

            updated_interactions.append(
                types.InteractionData(
                    # First item will be the most recent accession.
                    source=primary_source,
                    target=primary_target,
                    labels=interaction.labels,
                    evidence=interaction.evidence,
                    databases=interaction.databases,
                )
            )
        else:
            if not new_sources:
                logger.warning(
                    f"Could not find a mapping for source node "
                    f"'{interaction.source}'. Skipping interaction "
                    f"{asdict(interaction)}."
                )
            if not new_targets:
                logger.warning(
                    f"Could not find a mapping for target node "
                    f"'{interaction.target}'. Skipping interaction "
                    f"{asdict(interaction)}."
                )

    return updated_interactions


@DATABASE.atomic()
def create_proteins(
    proteins: Iterable[Tuple[str, UniprotEntry]]
) -> List[models.Protein]:
    """
    Create `Protein` database entries for each `UniprotEntry` instance.

    Metadata fields such as GO, Pfam, InterPro, and Keyword will be bulk
    created as well. 

    Existing proteins will have their metadata fields updated.

    Parameters
    ----------
    proteins : Iterable[Tuple[str, UniprotEntry]]
        Iterable of identifier and associated `UniprotEntry` tuples.

    Returns
    -------
    List[models.Protein]
    """
    # Loop through and collect the identifiers, genes and all annotations to
    # bulk create first.
    terms: Dict[str, Set[Any]] = defaultdict(set)
    gene_symbols: Set[types.GeneData] = set()
    accessions: Set[str] = set()
    # Result returned from `get_entries` is a generator since if can be
    # quite memory intensive to hold all XML records in memory at once. Parse
    # the required results into a dict for each:
    slim_entries: Dict[str, Dict[str, Any]] = {}
    proteins_to_create: Dict[str, str] = {}
    entry: UniprotEntry
    for (accession, entry) in proteins:
        # Add loop identifier since it may be an isoform identifier.
        normalized = models.UniprotIdentifier.format(accession)
        accessions |= set(entry.accessions + [normalized])

        terms["go"] |= set(entry.go_terms)
        terms["pfam"] |= set(entry.pfam_terms)
        terms["interpro"] |= set(entry.interpro_terms)
        terms["keyword"] |= set(entry.keywords)
        gene_symbols |= set(entry.genes)

        primary_ac = models.UniprotIdentifier.format(entry.primary_accession)
        if primary_ac in slim_entries:
            slim_entries[primary_ac]["accessions"] |= set(
                entry.accessions + [normalized]
            )
        else:
            slim_entries[primary_ac] = {
                "go_terms": set(entry.go_terms),
                "interpro_terms": set(entry.interpro_terms),
                "pfam_terms": set(entry.pfam_terms),
                "keywords": set(entry.keywords),
                "genes": set(entry.genes),
                "accessions": set(entry.accessions + [normalized]),
                "taxonomy": entry.taxonomy,
                "reviewed": entry.reviewed,
                "version": entry.version,
                "sequence": entry.sequence,
            }
        # Associate protein to create with data instance for later. Store
        # the primary accessions for each secondary accession used to access
        # the data record instance.
        proteins_to_create[normalized] = primary_ac

    # Bulk create and create lookup dictionaries indexed by identifier
    # for identifiers and terms, and gene symbol for the genes.
    logger.info("Populating annotation tables.")
    create_terms(terms["go"], model=models.GeneOntologyTerm)
    create_terms(terms["pfam"], model=models.PfamTerm)
    create_terms(terms["interpro"], model=models.InterproTerm)
    create_terms(terms["keyword"], model=models.Keyword)

    logger.info("Populating gene symbol table.")
    symbols = create_gene_symbols(set(g.symbol for g in gene_symbols))

    logger.info("Populating uniprot identifier table.")
    create_identifiers(accessions, model=models.UniprotIdentifier)

    # Loop through and intialize protein models from the slimmed down entries
    logger.info("Updating Protein data table.")
    protein_models: List[models.Protein] = []

    existing_symbols: Dict[str, models.GeneSymbol] = {
        str(symbol): symbol for symbol in symbols
    }
    existing_proteins: Dict[str, models.Protein] = {
        str(protein): protein
        for protein in models.Protein.get_by_identifier(slim_entries.keys())
    }
    identifiers: Dict[str, models.UniprotIdentifier] = {
        str(i): i
        for i in models.UniprotIdentifier.get_by_identifier(accessions)
    }

    slim_entry: Dict[str, Any]
    for primary_ac, slim_entry in list(slim_entries.items()):
        data = models.UniprotRecord.get_by_identifier(slim_entry["accessions"])

        # There should only be once instance for a set of primary and secondary
        # uniprot accessions.
        assert data.count() in (0, 1)

        if data.count() == 0:
            instance = models.UniprotRecord.create(
                organism=slim_entry["taxonomy"],
                sequence=slim_entry["sequence"],
                reviewed=slim_entry["reviewed"],
                version=slim_entry["version"],
            )
        else:
            instance = data.first()
            instance.organism = slim_entry["taxonomy"]
            instance.sequence = slim_entry["sequence"]
            instance.reviewed = slim_entry["reviewed"]
            instance.version = slim_entry["version"]
            instance.format_for_save()
            instance.save()

        # Replace dict with created/updated instance.
        slim_entries[primary_ac] = instance

        # Update the M2M relations.
        instance.go_annotations = models.GeneOntologyTerm.get_by_identifier(
            set(
                models.GeneOntologyIdentifier.format(x.identifier)
                for x in slim_entry["go_terms"]
            )
        )
        instance.pfam_annotations = models.PfamTerm.get_by_identifier(
            set(
                models.PfamIdentifier.format(x.identifier)
                for x in slim_entry["pfam_terms"]
            )
        )
        instance.interpro_annotations = models.InterproTerm.get_by_identifier(
            set(
                models.InterproIdentifier.format(x.identifier)
                for x in slim_entry["interpro_terms"]
            )
        )
        instance.keywords = models.Keyword.get_by_identifier(
            set(
                models.KeywordIdentifier.format(x.identifier)
                for x in slim_entry["keywords"]
            )
        )

        # Delete genes no longer associated with uniprot record.
        current_uniprot_genes = {str(g.gene): g for g in instance.genes}
        new_genes = set(g.symbol.upper() for g in slim_entry["genes"])
        uniprot_gene: models.UniprotRecordGene
        for uniprot_gene in instance.genes:
            if str(uniprot_gene.gene) not in new_genes:
                uniprot_gene.delete_instance()

        # Go through and update/create new genes associations
        gene: types.GeneData
        for gene in slim_entry["genes"]:
            symbol = gene.symbol.upper()
            if symbol in current_uniprot_genes:
                uniprot_gene = current_uniprot_genes[symbol]
                uniprot_gene.relation = gene.relation
                uniprot_gene.format_for_save()
                uniprot_gene.save()
            else:
                models.UniprotRecordGene.create(
                    record=instance,
                    relation=gene.relation,
                    gene=existing_symbols[gene.symbol.upper()],
                )

        # Delete identifiers no longer associated with uniprot record.
        current_record_identifiers = {
            str(i.identifier): i for i in instance.identifiers
        }
        record_identifier: models.UniprotRecordIdentifier
        for record_identifier in instance.identifiers:
            if (
                str(record_identifier.identifier)
                not in slim_entry["accessions"]
            ):
                record_identifier.delete_instance()

        # Go through and update/create new genes associations
        for accession in slim_entry["accessions"]:
            if accession in current_record_identifiers:
                record_identifier = current_record_identifiers[accession]
                record_identifier.primary = accession == primary_ac
                record_identifier.format_for_save()
                record_identifier.save()
            else:
                models.UniprotRecordIdentifier.create(
                    record=instance,
                    identifier=identifiers[accession],
                    primary=accession == primary_ac,
                )

    logger.info("Updating Protein table.")
    for accession, primary in proteins_to_create.items():
        if accession not in existing_proteins:
            protein_models.append(
                models.Protein.create(
                    identifier=identifiers[accession],
                    record=slim_entries[primary],
                )
            )
        else:
            protein_models.append(existing_proteins[accession])

    return protein_models


@DATABASE.atomic()
def create_interactions(
    interactions: Iterable[types.InteractionData],
) -> List[models.Interaction]:
    """
    Create a list of `Interaction` database rows and respective metadata fields.

    Expects the protein database table to be pre-filled. However additional
    metadata fields such as `databases`, `labels` and `evidence` will be 
    bulk created on the fly.

    Existing interactions will have their metadata fields updated.

    Parameters
    ----------
    interactions : Iterable[types.InteractionData]
        Iterable of interaction data instances.

    Raises
    ------
    peewee.IntegrityError
        Raised if `interactions` does not contain unique instances. Unique 
        meaning a unique `source` and `target` combination. 

    Returns
    -------
    List[models.Interaction]
    """
    # Collect all UniProt identifiers to pass to create_proteins.
    # create_proteins will download the uniprot entries and create the
    # co-responding the database models
    accessions: Set[str] = set()
    databases: Set[str] = set()
    labels: Set[str] = set()
    evidences: Set[types.InteractionEvidenceData] = set()

    # Loop over aggregated interactions (value, not the key).
    for interaction in set(interactions):
        # Collect database and evidence terms
        accessions.add(models.UniprotIdentifier.format(interaction.source))
        accessions.add(models.UniprotIdentifier.format(interaction.target))
        evidences |= set(interaction.evidence)
        databases |= set(interaction.databases)
        labels |= set(interaction.labels)

    # Bulk create proteins, evidence terms and database names.
    logger.info("Populating interaction evidence and metadata tables.")
    create_evidence(evidences)
    for db in databases:
        models.InteractionDatabase.get_or_create(name=db.lower())
    for label in labels:
        models.InteractionLabel.get_or_create(text=label.lower())

    # Loop through and update/create interactions.
    logger.info("Populating interaction table.")
    proteins: Dict[str, models.Protein] = {
        str(i): i for i in models.Protein.get_by_identifier(accessions)
    }

    instances: List[models.Interaction] = []
    for interaction in interactions:
        source = models.UniprotIdentifier.validate(
            models.UniprotIdentifier.format(interaction.source)
        )
        target = models.UniprotIdentifier.validate(
            models.UniprotIdentifier.format(interaction.target)
        )

        # Get/Create instance and update M2M fields.
        instance, _ = models.Interaction.get_or_create(
            source=proteins[source], target=proteins[target]
        )

        # Filter by capitalization since save will capitalize values before
        # commiting to databse.
        instance.labels = models.InteractionLabel.filter(
            fn.Lower(models.InteractionLabel.text)
            << set(i.lower() for i in interaction.labels)
        )
        instance.databases = models.InteractionDatabase.filter(
            fn.Lower(models.InteractionDatabase.name)
            << set(d.lower() for d in interaction.databases)
        )
        # Get related evidence terms by pubmed/psimi id.
        instance.evidence = [
            models.InteractionEvidence.get(
                pubmed=models.PubmedIdentifier.get(
                    identifier=models.PubmedIdentifier.format(e.pubmed)
                ),
                psimi=(
                    None
                    if not e.psimi
                    else models.PsimiIdentifier.get(
                        identifier=models.PsimiIdentifier.format(e.psimi)
                    )
                ),
            )
            for e in interaction.evidence
        ]
        instances.append(instance)

    return instances
