from pathlib import Path

import pytest

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
        assert interactions[0].databases == ["innatedb"]
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(pubmed="11734640", psimi="MI:0114"),
                InteractionEvidenceData(pubmed="11734640", psimi="MI:0118"),
            ],
            key=lambda e: hash(e),
        )

        assert interactions[1].source == "P55211"
        assert interactions[1].target == "P55213"
        assert interactions[1].databases == ["innatedb"]
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
        assert interactions[0].databases == ["innatedb"]
        assert interactions[0].evidence == []

    def test_uses_first_matching_uniprot_identifier(self):
        path = self.base_path / "innate_multi_uniprot.tsv"
        interactions = list(innate.parse_interactions(path))
        assert len(interactions) == 2

        assert interactions[0].source == "P55212"
        assert interactions[0].target == "P55211"

        assert interactions[1].source == "P55211"
        assert interactions[1].target == "P55213"

    def test_removes_doi_based_evidence(self):
        path = self.base_path / "innate_dois.tsv"
        interactions = list(innate.parse_interactions(path))
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(pubmed="11734641", psimi="MI:0114"),
                InteractionEvidenceData(pubmed="11734640", psimi="MI:0118"),
            ],
            key=lambda e: hash(e),
        )

    def test_if_more_pmid_than_psimis_uses_same_psimi_value(self):
        path = self.base_path / "innate_more_pmid_than_psimis.tsv"
        interactions = list(innate.parse_interactions(path))
        assert interactions[0].evidence == sorted(
            [
                InteractionEvidenceData(pubmed="11734640", psimi="MI:0114"),
                InteractionEvidenceData(pubmed="11734641", psimi="MI:0114"),
                InteractionEvidenceData(pubmed="11734642", psimi="MI:0114"),
            ],
            key=lambda e: hash(e),
        )
