from .base import BaseModel
from .identifiers import (
    GeneOntologyIdentifier,
    PfamIdentifier,
    InterproIdentifier,
    UniprotIdentifier,
    PubmedIdentifier,
    PsimiIdentifier,
    KeywordIdentifier,
    ExternalIdentifier,
)
from .metadata import (
    GeneOntologyTerm,
    PfamTerm,
    InterproTerm,
    Keyword,
    GeneSymbol,
    Annotation,
)
from .protein import (
    Protein,
    UniprotRecord,
    UniprotRecordGene,
    UniprotRecordIdentifier,
)
from .interaction import (
    Interaction,
    InteractionDatabase,
    InteractionLabel,
    InteractionEvidence,
    InteractionPrediction,
    ClassifierModel,
)

__all__ = [
    # modules
    "base",
    "identifiers",
    "interaction",
    "metadata",
    "protein",
    "MODELS",
    # identifiers
    "GeneOntologyIdentifier",
    "PfamIdentifier",
    "InterproIdentifier",
    "UniprotIdentifier",
    "PubmedIdentifier",
    "PsimiIdentifier",
    "KeywordIdentifier",
    "ExternalIdentifier",
    # metadata
    "GeneOntologyTerm",
    "PfamTerm",
    "InterproTerm",
    "Keyword",
    "GeneSymbol",
    "Annotation",
    # protein
    "Protein",
    "UniprotRecord",
    "UniprotRecordGene",
    "UniprotRecordIdentifier",
    # interaction
    "Interaction",
    "InteractionDatabase",
    "InteractionLabel",
    "InteractionEvidence",
    "InteractionPrediction",
    "ClassifierModel",
]

MODELS = (
    # Identifiers
    GeneOntologyIdentifier,
    PfamIdentifier,
    InterproIdentifier,
    UniprotIdentifier,
    PubmedIdentifier,
    PsimiIdentifier,
    KeywordIdentifier,
    # Metadata
    GeneOntologyTerm,
    PfamTerm,
    InterproTerm,
    Keyword,
    GeneSymbol,
    # Protein
    Protein,
    UniprotRecord,
    UniprotRecordGene,
    UniprotRecordIdentifier,
    UniprotRecord.go_annotations.get_through_model(),
    UniprotRecord.interpro_annotations.get_through_model(),
    UniprotRecord.pfam_annotations.get_through_model(),
    UniprotRecord.keywords.get_through_model(),
    # Interaction
    Interaction,
    InteractionDatabase,
    InteractionLabel,
    InteractionEvidence,
    Interaction.labels.get_through_model(),
    Interaction.evidence.get_through_model(),
    Interaction.databases.get_through_model(),
    InteractionPrediction,
    ClassifierModel,
)
