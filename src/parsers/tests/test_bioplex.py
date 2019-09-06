from pathlib import Path

import pytest

from .. import bioplex


class TestBioPlexFunc:
    def test_parses_correctly(self):
        path = Path(__file__).parent / "data" / "bioplex" / "bioplex.tsv"
        interactions = list(bioplex.parse_interactions(path))

        assert interactions[0].source == "P00813"
        assert interactions[0].target == "A5A3E0"
        assert interactions[0].databases == ["bioplex"]

        assert interactions[1].source == "P00813"
        assert interactions[1].target == "Q562R1"
        assert interactions[1].databases == ["bioplex"]

    def test_filters_out_interactions_with_missing_source_or_target(self):
        path = (
            Path(__file__).parent / "data" / "bioplex" / "bioplex_missing.tsv"
        )
        interactions = list(bioplex.parse_interactions(path))
        assert len(interactions) == 0

    def test_value_error_invalid_source(self):
        path = (
            Path(__file__).parent
            / "data"
            / "bioplex"
            / "bioplex_invalid_source.tsv"
        )
        with pytest.raises(ValueError):
            list(bioplex.parse_interactions(path))

    def test_value_error_invalid_target(self):
        path = (
            Path(__file__).parent
            / "data"
            / "bioplex"
            / "bioplex_invalid_target.tsv"
        )
        with pytest.raises(ValueError):
            list(bioplex.parse_interactions(path))

