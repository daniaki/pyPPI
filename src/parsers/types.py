from typing import List, Tuple, Optional
from dataclasses import dataclass, field

Source = str
Target = str
Label = str
Psimi = str
Pubmed = str
ExperimentType = str


@dataclass
class Interaction:
    source: str
    target: str
    label: Optional[str]
    database: Optional[str]
    psimi_ids: List[str] = field(default_factory=list)
    pmids: List[str] = field(default_factory=list)
    experiment_types: List[str] = field(default_factory=list)