import pytest

from peewee import IntegrityError

from ....constants import GeneOntologyCategory
from ....tests.test_utilities import null_values
from ... import models
from .. import DatabaseTestMixin


class TestAnnotationMixin(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.driver = models.Keyword
        self.identifier = models.KeywordIdentifier.create(identifier="KW-0001")

    def test_sets_name_as_none_if_falsey(self):
        instance = self.driver.create(identifier=self.identifier)
        for value in null_values:
            instance.name = value
            instance.save()
            assert instance.name is None

    def test_sets_description_as_none_if_falsey(self):
        instance = self.driver.create(identifier=self.identifier)
        for value in null_values:
            instance.description = value
            instance.save()
            assert instance.description is None

    def test_as_string_returns_identifier_string_or_na(self):
        instance = self.driver.create(identifier=self.identifier)
        assert str(instance) == str(self.identifier)

        instance.identifier = None
        assert str(instance) == str(None)

    def test_get_by_identifier_filters_by_uppercase(self):
        instance_1 = self.driver.create(identifier=self.identifier)
        instance_2 = self.driver.create(
            identifier=models.KeywordIdentifier.create(identifier="KW-0002")
        )
        instance_3 = self.driver.create(
            identifier=models.KeywordIdentifier.create(identifier="KW-0003")
        )

        query = self.driver.get_by_identifier(
            [
                str(instance_1.identifier).lower(),
                str(instance_2.identifier).capitalize(),
            ]
        )
        assert query.count() == 2
        assert instance_1 in query
        assert instance_2 in query
        assert instance_3 not in query


class TestGeneOntologyTermModel(DatabaseTestMixin):
    def test_converts_single_letter_category_to_full_category(self):
        term = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000001"
            ),
            name="",
            description="",
            category="C",
        )
        assert term.category == GeneOntologyCategory.cellular_component

    def test_raises_error_invalid_category(self):
        term = models.GeneOntologyTerm(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000001"
            ),
            name="",
            description="",
            category="m function",
        )
        with pytest.raises(ValueError):
            term.save()

    def test_capitalizes_and_strips_category(self):
        term = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000001"
            ),
            name="",
            description="",
            category="  molecular function  ",
        )
        assert term.category == GeneOntologyCategory.molecular_function


class TestInterproTerm(DatabaseTestMixin):
    def test_lowercases_and_strips_entry_type(self):
        instance = models.InterproTerm.create(
            identifier=models.InterproIdentifier.create(
                identifier="IPR000001"
            ),
            entry_type="  active site  ",
        )
        assert instance.entry_type == "active site"

    def test_sets_entry_text_as_none_if_null(self):
        instance = models.InterproTerm.create(
            identifier=models.InterproIdentifier.create(identifier="IPR000001")
        )
        for value in null_values:
            instance.entry_type = value
            instance.save()
            assert instance.entry_type is None


class TestGeneSymbolModel(DatabaseTestMixin):
    def test_uppercases_and_strips_text(self):
        symbol = models.GeneSymbol.create(text="  brca1  ")
        assert symbol.text == "BRCA1"

    def test_str_returns_text(self):
        symbol = models.GeneSymbol.create(text="A")
        assert str(symbol) == "A"

    def test_sets_text_as_none_if_null(self):
        for value in null_values:
            with pytest.raises(IntegrityError):
                models.GeneSymbol.create(text=value)


class TestKeywordModel(DatabaseTestMixin):
    def test_format_capitalizes(self):
        kw = models.Keyword.create(
            identifier=models.KeywordIdentifier.create(identifier="kw-0001"),
            description="",
        )
        assert kw.identifier.identifier == "KW-0001"
