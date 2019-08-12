import pytest

from ....constants import GeneOntologyCategory
from ... import models
from .. import DatabaseTestMixin


class TestGeneOntologyTermModel(DatabaseTestMixin):
    def test_converts_single_letter_category_to_full_category(self):
        term = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(identifier="1"),
            name="",
            description="",
            category="C",
        )
        assert term.category == GeneOntologyCategory.cellular_compartment

    def test_raises_error_invalid_category(self):
        term = models.GeneOntologyTerm(
            identifier=models.GeneOntologyIdentifier.create(identifier="1"),
            name="",
            description="",
            category="m function",
        )
        with pytest.raises(ValueError):
            term.save()

    def test_capitalizes_and_strips_category(self):
        term = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(identifier="1"),
            name="",
            description="",
            category="  molecular function  ",
        )
        assert term.category == GeneOntologyCategory.molecular_function


class TestGeneSymbolModel(DatabaseTestMixin):
    def test_uppercases_and_strips_text(self):
        symbol = models.GeneSymbol.create(text="  brca1  ")
        assert symbol.text == "BRCA1"


class TestInteractionDatabaseModel(DatabaseTestMixin):
    def test_capitalizes_and_strips_text(self):
        instance = models.InteractionDatabase.create(name="  KEGG  ")
        assert instance.name == "Kegg"


class TestKeywordModel(DatabaseTestMixin):
    def test_appends_prefix(self):
        kw = models.Keyword.create(
            identifier=models.KeywordIdentifier.create(identifier="1"),
            description="",
        )
        assert kw.identifier.identifier == "KW-1"

        kw = models.Keyword.create(
            identifier=models.KeywordIdentifier.create(identifier="KW-2"),
            description="",
        )
        assert kw.identifier.identifier == "KW-2"
