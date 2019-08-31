import pytest
from peewee import ModelSelect

from ...parsers import types
from ...database import models
from ...constants import GeneOntologyCategory
from .. import utilities
from ..tests import DatabaseTestMixin


class TestCreateIdentifiers(DatabaseTestMixin):
    def test_creates_new_formatted_identifiers(self):
        identifier = "0001"
        query: ModelSelect = utilities.create_identifiers(
            identifiers=[identifier], model=models.KeywordIdentifier
        )
        assert query.count() == 1
        assert str(query.first()) == "KW-0001"

    def test_gets_existing_identifier(self):
        identifier = models.KeywordIdentifier.create(identifier="KW-0001")
        query: ModelSelect = utilities.create_identifiers(
            identifiers=[str(identifier)], model=models.KeywordIdentifier
        )
        assert query.count() == 1
        assert query.first().id == identifier.id


class TestCreateGeneSymbols(DatabaseTestMixin):
    def test_creates_new_symbol_in_uppercase(self):
        symbol = "brca1"
        query: ModelSelect = utilities.create_gene_symbols(symbols=[symbol])
        assert query.count() == 1
        assert str(query.first()) == symbol.upper()

    def test_gets_existing_symbol(self):
        symbol = models.GeneSymbol.create(text="brca1")
        query: ModelSelect = utilities.create_gene_symbols(
            symbols=[str(symbol)]
        )
        assert query.count() == 1
        assert query.first().id == symbol.id


class TestCreateAnnotations(DatabaseTestMixin):
    def test_creates_new_annotation_with_formatted_identifier(self):
        term = types.GeneOntologyTermData(
            identifier="go:0000001",
            category=GeneOntologyCategory.cellular_component,
            name="Energy production",
            obsolete=False,
            description=None,
        )
        query: ModelSelect = utilities.create_terms(
            terms=[term], model=models.GeneOntologyTerm
        )
        assert models.GeneOntologyIdentifier.count() == 1
        assert query.count() == 1

        instance: models.GeneOntologyTerm = query.first()
        assert str(instance.identifier) == "GO:0000001"
        assert instance.category == term.category
        assert instance.name == term.name
        assert instance.obsolete == term.obsolete
        assert instance.description == term.description


class TestCreateEvidence(DatabaseTestMixin):
    def test_creates_new_evidence_with_identifiers(self):
        evidence = types.InteractionEvidenceData(
            pubmed="1234", psimi="MI:0001"
        )

        assert models.PubmedIdentifier.count() == 0
        assert models.PsimiIdentifier.count() == 0

        query = utilities.create_evidence([evidence])
        assert len(query) == 1
        assert models.PubmedIdentifier.count() == 1
        assert models.PsimiIdentifier.count() == 1

        instance: models.InteractionEvidence = query[0]
        assert str(instance.pubmed) == "PUBMED:1234"
        assert str(instance.psimi) == "MI:0001"

    def test_gets_existing_evidence(self):
        instance = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="1234"),
            psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
        )
        evidence = types.InteractionEvidenceData(
            pubmed="1234", psimi="MI:0001"
        )

        assert models.PubmedIdentifier.count() == 1
        assert models.PsimiIdentifier.count() == 1

        query = utilities.create_evidence([evidence])
        assert len(query) == 1
        assert query[0].id == instance.id
        assert models.PubmedIdentifier.count() == 1
        assert models.PsimiIdentifier.count() == 1


# class

