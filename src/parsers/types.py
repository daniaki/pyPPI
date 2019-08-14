from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from ..constants import GeneOntologyCategory


@dataclass
class InteractionEvidenceData:
    pubmed: str
    psimi: Optional[str] = field(default=None)

    def __hash__(self):
        return hash((self.pubmed, self.psimi))


@dataclass
class InteractionData:
    source: str
    target: str
    organism: int
    labels: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    evidence: List[InteractionEvidenceData] = field(default_factory=list)


@dataclass
class GeneOntologyTermData:
    identifier: str
    category: str
    obsolete: bool
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.category not in GeneOntologyCategory.list():
            raise ValueError(f"'{self.category}' is not a valid GO category.")


@dataclass
class PfamTermData:
    identifier: str
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)


@dataclass
class InterproTermData:
    identifier: str
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    entry_type: Optional[str] = field(default=None)


@dataclass
class KeywordTermData:
    identifier: str
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
