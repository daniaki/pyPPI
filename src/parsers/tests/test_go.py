from pathlib import Path

from ...constants import GeneOntologyCategory
from .. import go


class TestGOParser:
    def setup(self):
        self.base = Path(__file__).parent / "data" / "go"

    def test_parses_obo_into_terms(self):
        path = self.base / "basic.obo"
        terms = go.parse_go_obo(path)

        assert len(terms) == 241
        assert terms[0].identifier == "GO:0000003"
        assert terms[0].obsolete is False
        assert terms[0].name == "reproduction"
        assert terms[0].description is None
        assert terms[0].category == GeneOntologyCategory.biological_process

    def test_parses_gzipped_obo_into_terms(self):
        path = self.base / "basic.obo.gz"
        terms = go.parse_go_obo(path)

        assert len(terms) == 241
        assert terms[0].identifier == "GO:0000003"
        assert terms[0].obsolete is False
        assert terms[0].name == "reproduction"
        assert terms[0].description is None
        assert terms[0].category == GeneOntologyCategory.biological_process
