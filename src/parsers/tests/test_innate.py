from pathlib import Path

from ..types import InteractionEvidenceData
from .. import innate


class TestInnateParser:
    def setup(self):
        self.base_path = Path(__file__).parent / "data" / "innate"

    def test_removes_duplicate_evidence(self):
        path = self.base_path / "innate.tsv"
        interactions = list(innate.parse_interactions(path))

        assert len(interactions) == 2
        assert interactions[0].source == "P55211"
        assert interactions[0].target == "P55212"
        assert interactions[0].organism == 9606
        assert interactions[0].databases == ["InnateDB"]
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(
                    pubmed="pubmed:11734640", psimi="MI:0114"
                ),
                InteractionEvidenceData(
                    pubmed="pubmed:11734640", psimi="MI:0118"
                ),
            ],
            key=lambda e: hash(e),
        )

        assert interactions[1].source == "P55211"
        assert interactions[1].target == "P55213"
        assert interactions[1].organism == 9606
        assert interactions[1].databases == ["InnateDB"]
        assert interactions[1].evidence == []

    def test_ignores_if_missing_source_or_target(self):
        path = self.base_path / "innate_no_uniprot.tsv"
        interactions = list(innate.parse_interactions(path))
        assert not interactions

    def test_ignores_if_non_human_interactor(self):
        path = self.base_path / "innate_non_human.tsv"
        interactions = list(innate.parse_interactions(path))

        assert len(interactions) == 1
        assert interactions[0].source == "P55200"
        assert interactions[0].target == "P55211"
        assert interactions[0].organism == 9606
        assert interactions[0].databases == ["InnateDB"]
        assert interactions[0].evidence == []

    def test_creates_one_interaction_for_each_source_and_target_per_line(self):
        path = self.base_path / "innate_multi_uniprot.tsv"
        interactions = list(innate.parse_interactions(path))
        assert len(interactions) == 4

        assert interactions[0].source == "P55212"
        assert interactions[0].target == "P55211"

        assert interactions[1].source == "P55211"
        assert interactions[1].target == "P55211"

        assert interactions[2].source == "P55211"
        assert interactions[2].target == "P55213"

        assert interactions[3].source == "P55211"
        assert interactions[3].target == "P55211"
