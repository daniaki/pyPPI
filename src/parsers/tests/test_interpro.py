from pathlib import Path

from .. import interpro


class TestInterproParser:
    def setup(self):
        self.base = Path(__file__).parent / "data" / "interpro"

    def test_skips_invalid_identifiers(self):
        path = self.base / "invalid_identifier.tsv"
        terms = interpro.parse_entry_list(path)
        assert len(terms) == 0

    def test_parses_identifiers(self):
        path = self.base / "basic.tsv"
        terms = interpro.parse_entry_list(path)

        assert len(terms) == 1
        assert terms[0].identifier == "IPR000126"
        assert terms[0].name is None
        assert terms[0].entry_type == "Active_site"
        assert (
            terms[0].description
            == "Serine proteases, V8 family, serine active site"
        )

    def test_sets_metadata_as_none_if_missing(self):
        path = self.base / "missing_meta.tsv"
        terms = interpro.parse_entry_list(path)

        assert len(terms) == 1
        assert terms[0].identifier == "IPR000126"
        assert terms[0].name is None
        assert terms[0].description is None
        assert terms[0].entry_type is None
