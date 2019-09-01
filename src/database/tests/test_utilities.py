from pathlib import Path

import mock
import pytest
from bs4 import BeautifulSoup
from peewee import ModelSelect

from ...clients.uniprot import UniprotClient, UniprotEntry
from ...constants import GeneOntologyCategory
from ...database import models
from ...parsers import types
from .. import utilities
from ..tests import DatabaseTestMixin

DATA_DIR = Path(__file__).parent / "data"


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


class TestCreateProteins(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.entry = UniprotEntry(
            root=BeautifulSoup(
                open(DATA_DIR / "P38398.xml", "rt").read(), "xml"
            )
        )

    def assert_tables_empty(self):
        assert models.UniprotIdentifier.count() == 0
        assert models.GeneOntologyIdentifier.count() == 0
        assert models.GeneOntologyTerm.count() == 0
        assert models.InterproIdentifier.count() == 0
        assert models.InterproTerm.count() == 0
        assert models.PfamIdentifier.count() == 0
        assert models.PfamTerm.count() == 0
        assert models.KeywordIdentifier.count() == 0
        assert models.Keyword.count() == 0
        assert models.GeneSymbol.count() == 0

    def test_creates_new_protein(self):
        protein = "P38398"
        UniprotClient.get_entries = mock.MagicMock(return_value=[self.entry])

        # Check all tables are empty before call.
        self.assert_tables_empty()

        query = utilities.create_proteins([protein])
        assert len(query) == 1
        instance: models.Protein = query[0]
        # Check all tables have been populated.
        assert models.UniprotIdentifier.count() == len(self.entry.accessions)
        assert models.GeneOntologyIdentifier.count() == len(
            self.entry.go_terms
        )
        assert models.InterproIdentifier.count() == len(
            self.entry.interpro_terms
        )
        assert models.PfamIdentifier.count() == len(self.entry.pfam_terms)
        assert models.KeywordIdentifier.count() == len(self.entry.keywords)
        assert models.GeneSymbol.count() == len(self.entry.genes)

        # Check fields have been set appropriately.
        assert str(instance) == "P38398"
        assert instance.sequence == self.entry.sequence
        assert instance.organism == 9606
        assert instance.reviewed == self.entry.reviewed
        assert instance.aliases.count() == len(self.entry.alias_accessions)
        assert instance.go_annotations.count() == len(self.entry.go_terms)
        assert instance.interpro_annotations.count() == len(
            self.entry.interpro_terms
        )
        assert instance.pfam_annotations.count() == len(self.entry.pfam_terms)
        assert instance.keywords.count() == len(self.entry.keywords)
        assert instance.genes.count() == len(self.entry.genes)

    def test_updates_existing_protein(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P38398"),
            sequence="LMN",
            organism=1000,
            reviewed=False,
        )

        UniprotClient.get_entries = mock.MagicMock(return_value=[self.entry])

        query = utilities.create_proteins([str(protein)])
        assert len(query) == 1
        instance: models.Protein = query[0]
        # Check all tables have been populated.
        assert models.UniprotIdentifier.count() == len(self.entry.accessions)
        assert models.GeneOntologyIdentifier.count() == len(
            self.entry.go_terms
        )
        assert models.InterproIdentifier.count() == len(
            self.entry.interpro_terms
        )
        assert models.PfamIdentifier.count() == len(self.entry.pfam_terms)
        assert models.KeywordIdentifier.count() == len(self.entry.keywords)
        assert models.GeneSymbol.count() == len(self.entry.genes)

        # Check fields have been set appropriately.
        assert str(instance) == "P38398"
        assert instance.sequence == self.entry.sequence
        assert instance.organism == 9606
        assert instance.reviewed == self.entry.reviewed
        assert instance.aliases.count() == len(self.entry.alias_accessions)
        assert instance.go_annotations.count() == len(self.entry.go_terms)
        assert instance.interpro_annotations.count() == len(
            self.entry.interpro_terms
        )
        assert instance.pfam_annotations.count() == len(self.entry.pfam_terms)
        assert instance.keywords.count() == len(self.entry.keywords)
        assert instance.genes.count() == len(self.entry.genes)


class TestCreateInteractions(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.entry_a = UniprotEntry(
            root=BeautifulSoup(
                open(DATA_DIR / "P38398.xml", "rt").read(), "xml"
            )
        )
        self.entry_b = UniprotEntry(
            root=BeautifulSoup(
                open(DATA_DIR / "P02760.xml", "rt").read(), "xml"
            )
        )
        UniprotClient.get_entries = mock.MagicMock(
            return_value=[self.entry_a, self.entry_b]
        )

    def assert_tables_empty(self):
        assert models.UniprotIdentifier.count() == 0
        assert models.Interaction.count() == 0
        assert models.InteractionEvidence.count() == 0
        assert models.InteractionLabel.count() == 0
        assert models.InteractionDatabase.count() == 0
        assert models.PsimiIdentifier.count() == 0
        assert models.PubmedIdentifier.count() == 0
        assert models.GeneOntologyIdentifier.count() == 0
        assert models.GeneOntologyTerm.count() == 0
        assert models.InterproIdentifier.count() == 0
        assert models.InterproTerm.count() == 0
        assert models.PfamIdentifier.count() == 0
        assert models.PfamTerm.count() == 0
        assert models.KeywordIdentifier.count() == 0
        assert models.Keyword.count() == 0
        assert models.GeneSymbol.count() == 0

    def test_creates_new_interaction(self):
        interaction = types.InteractionData(
            source="P38398",
            target="P02760",
            organism=9606,
            labels=["Activation", "Methylation"],
            databases=["Kegg"],
            evidence=[
                types.InteractionEvidenceData(pubmed="123456", psimi="MI:0001")
            ],
        )

        self.assert_tables_empty()
        query = utilities.create_interactions([interaction])

        assert len(query) == 1
        instance: models.Interaction = query[0]

        assert instance.evidence.count() == 1
        assert str(instance.evidence.first().pubmed) == "PUBMED:123456"
        assert str(instance.evidence.first().psimi) == "MI:0001"
        assert instance.organism == 9606
        assert instance.compact == ("P38398", "P02760")
        assert sorted([str(l) for l in instance.labels]) == [
            "activation",
            "methylation",
        ]
        assert sorted([str(d) for d in instance.databases]) == ["kegg"]

    def test_creates_sets_m2m_fields_of_multiple_interactions(self):
        interactions = [
            types.InteractionData(
                source="P38398",
                target="P02760",
                organism=9606,
                labels=["Activation"],
                databases=["Kegg"],
                evidence=[
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0001"
                    ),
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0002"
                    )
                ],
            ),
            types.InteractionData(
                source="P38398",
                target="P38398",
                organism=9606,
                labels=["Activation", "Methylation"],
                databases=["Hprd"],
                evidence=[
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0002"
                    )
                ],
            ),
        ]

        self.assert_tables_empty()
        query = utilities.create_interactions(interactions)

        assert len(query) == 2

        # Check the m2m are set appropriately and not mixed.
        # @0
        assert sorted(
            [(str(e.pubmed), str(e.psimi)) for e in query[0].evidence],
            key=lambda e: e[-1],
        ) == [("PUBMED:123456", "MI:0001"), ("PUBMED:123456", "MI:0002")]
        assert query[0].organism == 9606
        assert query[0].compact == ("P38398", "P02760")
        assert sorted([str(l) for l in query[0].labels]) == [
            "activation",
        ]
        assert sorted([str(d) for d in query[0].databases]) == ["kegg"]

        # @1
        assert sorted(
            [(str(e.pubmed), str(e.psimi)) for e in query[1].evidence],
            key=lambda e: e[-1],
        ) == [("PUBMED:123456", "MI:0002")]
        assert query[1].organism == 9606
        assert query[1].compact == ("P38398", "P38398")
        assert sorted([str(l) for l in query[1].labels]) == [
            "activation", "methylation"
        ]
        assert sorted([str(d) for d in query[1].databases]) == ["hprd"]


    def test_updates_existing_interaction(self):
        interaction = models.Interaction.create(
            source=models.UniprotIdentifier.create(identifier="P38398"),
            target=models.UniprotIdentifier.create(identifier="P02760"),
            organism=1000,
        )
        interaction.labels = [models.InteractionLabel.create(text="a")]
        interaction.databases = [
            models.InteractionDatabase.create(name="database")
        ]
        interaction.evidence = [
            models.InteractionEvidence.create(
                pubmed=models.PubmedIdentifier.create(identifier="1"),
                psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
            )
        ]

        interactions = [
            types.InteractionData(
                source="P38398",
                target="P02760",
                organism=9606,
                labels=["Activation", "methylation"],
                databases=["Kegg", "HPRD"],
                evidence=[
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0001"
                    )
                ],
            )
        ]

        query = utilities.create_interactions(interactions)
        assert len(query) == 1
        instance: models.Interaction = query[0]

        assert instance.evidence.count() == 1
        assert models.InteractionEvidence.count() == 2
        assert sorted(
            [(str(e.pubmed), str(e.psimi)) for e in instance.evidence],
            key=lambda e: e[-1],
        ) == [("PUBMED:123456", "MI:0001")]

        assert instance.organism == 9606
        assert instance.compact == ("P38398", "P02760")

        assert sorted([str(l) for l in instance.labels]) == [
            "activation",
            "methylation",
        ]
        assert sorted([str(d) for d in instance.databases]) == ["hprd", "kegg"]

    def test_error_interaction_data_missing_source(self):
        interaction = types.InteractionData(
            source=None, target="P02760", organism=9606
        )
        with pytest.raises(ValueError):
            utilities.create_interactions([interaction])

    def test_error_interaction_data_missing_target(self):
        interaction = types.InteractionData(
            source="P38398", target=None, organism=9606
        )
        with pytest.raises(ValueError):
            utilities.create_interactions([interaction])

    def test_skips_interaction_if_source_not_found_on_uniprot(self):
        interaction = types.InteractionData(
            source="P38398", target="P02760", organism=9606
        )
        UniprotClient.get_entries = mock.MagicMock(return_value=[self.entry_b])
        query = utilities.create_interactions([interaction])
        assert len(query) == 0

    def test_skips_interaction_if_target_not_found_on_uniprot(self):
        interaction = types.InteractionData(
            source="P38398", target="P02760", organism=9606
        )
        UniprotClient.get_entries = mock.MagicMock(return_value=[self.entry_a])
        query = utilities.create_interactions([interaction])
        assert len(query) == 0

    def test_aggregates_interaction_data_structs(self):
        interactions = [
            types.InteractionData(
                source="P38398",
                target="P02760",
                organism=9606,
                labels=["Activation"],
                databases=["Kegg"],
                evidence=[
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0001"
                    )
                ],
            ),
            types.InteractionData(
                source="P38398",
                target="P02760",
                organism=9606,
                labels=["Activation", "Methylation"],
                databases=["Hprd"],
                evidence=[
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0002"
                    )
                ],
            ),
        ]

        self.assert_tables_empty()
        query = utilities.create_interactions(interactions)

        assert len(query) == 1
        instance: models.Interaction = query[0]

        assert instance.evidence.count() == 2
        assert models.InteractionEvidence.count() == 2
        assert sorted(
            [(str(e.pubmed), str(e.psimi)) for e in instance.evidence],
            key=lambda e: e[-1],
        ) == [("PUBMED:123456", "MI:0001"), ("PUBMED:123456", "MI:0002")]

        assert instance.organism == 9606
        assert instance.compact == ("P38398", "P02760")

        assert sorted([str(l) for l in instance.labels]) == [
            "activation",
            "methylation",
        ]
        assert sorted([str(d) for d in instance.databases]) == ["hprd", "kegg"]
