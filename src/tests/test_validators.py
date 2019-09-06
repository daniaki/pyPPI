from .. import validators
from .test_utilities import null_values

from idutils import is_uniprot


class TestIsPubmed:
    def test_passes(self):
        assert validators.is_pubmed("16333295")

    def test_fails(self):
        assert not validators.is_pubmed("pubmed:16333295")
        assert not validators.is_pubmed("pubmed:16333295 Other")


class TestIsGO:
    def test_passes(self):
        assert validators.is_go("GO:1234567")

    def test_fails(self):
        assert not validators.is_go("GO:1234567 Other")


class TestIsInterpro:
    def test_passes(self):
        assert validators.is_interpro("IPR000169")

    def test_fails(self):
        assert not validators.is_interpro("IPR000169 other")


class TestIsPfam:
    def test_passes(self):
        assert validators.is_pfam("PF00005")

    def test_fails(self):
        assert not validators.is_pfam("random PF00005")


class TestIsKeyword:
    def test_passes(self):
        assert validators.is_keyword("KW-0001")

    def test_fails(self):
        assert not validators.is_keyword("random KW-0001")


class TestIsPsimi:
    def test_passes(self):
        assert validators.is_psimi("MI:0004")

    def test_fails(self):
        assert not validators.is_psimi("MI:0004 random")

