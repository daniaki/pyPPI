from .base import BaseModel
from .identifiers import (
    GeneOntologyIdentifier,
    PfamIdentifier,
    InterproIdentifier,
    UniprotIdentifier,
    PubmedIdentifier,
    PsimiIdentifier,
    KeywordIdentifier,
    IdentifierMixin,
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
from .protein import Protein
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
    # identifiers
    "GeneOntologyIdentifier",
    "PfamIdentifier",
    "InterproIdentifier",
    "UniprotIdentifier",
    "PubmedIdentifier",
    "PsimiIdentifier",
    "KeywordIdentifier",
    "IdentifierMixin",
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
    Protein.aliases.get_through_model(),
    Protein.alt_genes.get_through_model(),
    Protein.go_annotations.get_through_model(),
    Protein.interpro_annotations.get_through_model(),
    Protein.pfam_annotations.get_through_model(),
    Protein.keywords.get_through_model(),
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
