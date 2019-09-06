from pathlib import Path

from ..types import InteractionEvidenceData
from .. import pina


class TestPinaParser:
    def setup(self):
        self.base_path = Path(__file__).parent / "data" / "pina"

    def test_parses_valid_interactions(self):
        path = self.base_path / "pina.tsv"
        interactions = list(pina.parse_interactions(path))

        assert len(interactions) == 2
        assert interactions[0].source == "Q96BR9"
        assert interactions[0].target == "Q9BXS5"
        assert interactions[0].databases == ["pina"]
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(
                    pubmed="16189514", psimi="MI:0018"
                ),
                InteractionEvidenceData(
                    pubmed="16189514", psimi="MI:0096"
                ),
                InteractionEvidenceData(
                    pubmed="16189515", psimi="MI:0018"
                ),
            ],
            key=lambda e: hash(e),
        )

        assert interactions[1].source == "Q96BR7"
        assert interactions[1].target == "Q9BXS7"
        assert interactions[1].databases == ["pina"]
        assert interactions[1].evidence == []

    def test_ignores_if_missing_source_or_target(self):
        path = self.base_path / "pina_no_uniprot.tsv"
        interactions = list(pina.parse_interactions(path))
        assert not interactions

    def test_ignores_if_non_human_interactor(self):
        path = self.base_path / "pina_non_human.tsv"
        interactions = list(pina.parse_interactions(path))

        assert len(interactions) == 1
        assert interactions[0].source == "Q96BR9"
        assert interactions[0].target == "Q9BXS5"
        assert interactions[0].databases == ["pina"]
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(
                    pubmed="16189514", psimi="MI:0018"
                ),
                InteractionEvidenceData(
                    pubmed="16189514", psimi="MI:0096"
                ),
                InteractionEvidenceData(
                    pubmed="16189515", psimi="MI:0018"
                ),
            ],
            key=lambda e: hash(e),
        )

    def test_removes_doi_evidence(self):
        path = self.base_path / "pina_dois.tsv"
        interactions = list(pina.parse_interactions(path))

        assert len(interactions) == 2
        assert interactions[0].source == "Q96BR9"
        assert interactions[0].target == "Q9BXS5"
        assert interactions[0].databases == ["pina"]
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(
                    pubmed="16189514", psimi="MI:0096"
                ),
                InteractionEvidenceData(
                    pubmed="16189514", psimi="MI:0018"
                ),
            ],
            key=lambda e: hash(e),
        )

        assert interactions[1].source == "Q96BR7"
        assert interactions[1].target == "Q9BXS7"
        assert interactions[1].databases == ["pina"]
        assert interactions[1].evidence == []
