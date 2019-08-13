from pathlib import Path

from idutils import is_uniprot

from ..types import InteractionEvidenceData
from .. import edgelist


class TestInnateParser:
    def setup(self):
        self.base_path = Path(__file__).parent / "data" / "edgelist"

    def test_formats_using_formatter(self):
        path = self.base_path / "basic.csv"
        interactions = list(
            edgelist.parse_interactions(
                path=path,
                databases=["edgy"],
                sep=",",
                header=True,
                formatter=str.upper,
                validator=is_uniprot,
            )
        )

        assert len(interactions) == 2
        assert interactions[0].source == "P55211"
        assert interactions[0].target == "P55212"
        assert interactions[0].organism == 9606
        assert interactions[0].databases == ["edgy"]

        assert interactions[1].source == "P55212"
        assert interactions[1].target == "P55213"
        assert interactions[1].organism == 9606
        assert interactions[1].databases == ["edgy"]

    def test_does_not_trim_header_if_header_is_false(self):
        path = self.base_path / "no_header.csv"
        interactions = list(
            edgelist.parse_interactions(
                path=path,
                databases=["edgy"],
                sep=",",
                header=False,
                formatter=str.upper,
                validator=is_uniprot,
            )
        )
        assert len(interactions) == 2

    def test_skips_lines_with_invalid_accessions(self):
        path = self.base_path / "invalid_accession.csv"
        interactions = list(
            edgelist.parse_interactions(
                path=path,
                databases=["edgy"],
                sep=",",
                header=True,
                formatter=str.upper,
                validator=is_uniprot,
            )
        )
        assert not interactions
