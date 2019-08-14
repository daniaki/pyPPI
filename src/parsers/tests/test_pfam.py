from pathlib import Path

from .. import pfam


class TestPfamParser:
    def setup(self):
        self.base = Path(__file__).parent / "data" / "pfam"

    def test_skips_non_pfam_identifiers(self):
        path = self.base / "invalid_identifier.tsv"
        terms = pfam.parse_clans_file(path)

        assert len(terms) == 1
        assert terms[0].identifier == "PF00001"
        assert terms[0].name == "7tm_1"
        assert (
            terms[0].description
            == "7 transmembrane receptor (rhodopsin family)"
        )

    def test_parses_pfam_identifiers(self):
        path = self.base / "clans.tsv"
        terms = pfam.parse_clans_file(path)

        assert len(terms) == 2
        assert terms[0].identifier == "PF00001"
        assert terms[0].name == "7tm_1"
        assert (
            terms[0].description
            == "7 transmembrane receptor (rhodopsin family)"
        )

        assert terms[1].identifier == "PF00002"
        assert terms[1].name == "7tm_2"
        assert (
            terms[1].description
            == "7 transmembrane receptor (Secretin family)"
        )

    def test_sets_desciption_as_none_if_empty(self):
        path = self.base / "missing_description.tsv"
        terms = pfam.parse_clans_file(path)

        assert len(terms) == 1
        assert terms[0].identifier == "PF00001"
        assert terms[0].name == "7tm_1"
        assert terms[0].description is None
