from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field, astuple

from ..constants import GeneOntologyCategory


@dataclass(order=True)
class InteractionEvidenceData:
    pubmed: str
    psimi: Optional[str] = field(default=None, compare=False)

    def __hash__(self):
        return hash(astuple(self))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return super().__eq__(other)
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other


@dataclass
class InteractionData:
    source: str
    target: str
    organism: int
    labels: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    evidence: List[InteractionEvidenceData] = field(default_factory=list)

    def __hash__(self):
        return hash((self.source, self.target, self.organism))

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"unsupported operand type(s) for +: "
                f"'{type(self).__name__}' and '{type(other).__name__}'"
            )

        if hash(self) != hash(other):
            raise ValueError(
                f"Cannot add instances with different source/target values or "
                f"organism codes."
            )

        return InteractionData(
            source=self.source,
            target=self.target,
            organism=self.organism,
            labels=sorted(set(self.labels + other.labels)),
            databases=sorted(set(self.databases + other.databases)),
            evidence=sorted(set(self.evidence + other.evidence)),
        )


@dataclass
class GeneOntologyTermData:
    identifier: str
    category: str
    obsolete: Optional[bool] = field(default=None)
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)

    def __hash__(self):
        return hash(astuple(self))

    def __post_init__(self):
        if self.category not in GeneOntologyCategory.list():
            raise ValueError(f"'{self.category}' is not a valid GO category.")


@dataclass
class PfamTermData:
    identifier: str
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)

    def __hash__(self):
        return hash(astuple(self))


@dataclass
class InterproTermData:
    identifier: str
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    entry_type: Optional[str] = field(default=None)

    def __hash__(self):
        return hash(astuple(self))


@dataclass
class KeywordTermData:
    identifier: str
    name: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)

    def __hash__(self):
        return hash(astuple(self))
