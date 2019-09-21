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
    def test_error_invalid_identifier(self):
        identifier = "0001"
        with pytest.raises(ValueError):
            utilities.create_identifiers(
                identifiers=[identifier], model=models.KeywordIdentifier
            )

    def test_creates_identifier(self):
        query: ModelSelect = utilities.create_identifiers(
            identifiers=["KW-0001"], model=models.KeywordIdentifier
        )
        assert query.count() == 1
        assert str(query.first()) == "KW-0001"

    def test_creates_identifiers_is_case_insensitive(self):
        query: ModelSelect = utilities.create_identifiers(
            identifiers=["kw-0001"], model=models.KeywordIdentifier
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

    def test_gets_existing_identifier_case_insensitive(self):
        identifier = models.KeywordIdentifier.create(identifier="KW-0001")
        query: ModelSelect = utilities.create_identifiers(
            identifiers=[str(identifier).lower()],
            model=models.KeywordIdentifier,
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


class TestCreateTerms(DatabaseTestMixin):
    def test_creates_new_annotation_case_insensitive(self):
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

    def test_returns_existing_case_insensitive(self):
        instance = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000001"
            ),
            category=GeneOntologyCategory.cellular_component,
            name="Energy production",
            obsolete=False,
            description=None,
        )

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
        assert instance.id == query.first().id


class TestCreateEvidence(DatabaseTestMixin):
    def test_creates_new_evidence_with_identifiers_case_insensitive(self):
        evidence = types.InteractionEvidenceData(
            pubmed="1234", psimi="mi:0001"
        )

        assert models.PubmedIdentifier.count() == 0
        assert models.PsimiIdentifier.count() == 0

        query = utilities.create_evidence([evidence])
        assert len(query) == 1
        assert models.PubmedIdentifier.count() == 1
        assert models.PsimiIdentifier.count() == 1

        instance: models.InteractionEvidence = query[0]
        assert str(instance.pubmed) == "1234"
        assert str(instance.psimi) == "MI:0001"

    def test_gets_existing_evidence_case_insensitive(self):
        instance = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="1234"),
            psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
        )
        evidence = [
            types.InteractionEvidenceData(pubmed="1234", psimi="mi:0001"),
            types.InteractionEvidenceData(pubmed="1234", psimi="MI:0001"),
        ]

        assert models.PubmedIdentifier.count() == 1
        assert models.PsimiIdentifier.count() == 1

        query = utilities.create_evidence(evidence)
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
        assert models.UniprotRecord.count() == 0
        assert models.UniprotRecordGene.count() == 0
        assert models.UniprotRecordIdentifier.count() == 0
        assert models.Protein.count() == 0
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

    def test_creates_with_isoform_identifier(self):
        # Check all tables are empty before call.
        self.assert_tables_empty()

        query = utilities.create_proteins([("P38398-1", self.entry)])

        assert len(query) == 1
        instance: models.Protein = query[0]
        # Check all tables have been populated.
        assert models.UniprotRecord.count() == 1
        assert models.UniprotIdentifier.count() == (
            len(self.entry.accessions) + 1
        )
        assert models.GeneOntologyIdentifier.count() == len(
            self.entry.go_terms
        )
        assert models.InterproIdentifier.count() == len(
            self.entry.interpro_terms
        )
        assert models.PfamIdentifier.count() == len(self.entry.pfam_terms)
        assert models.KeywordIdentifier.count() == len(self.entry.keywords)
        assert models.GeneOntologyTerm.count() == len(self.entry.go_terms)
        assert models.InterproTerm.count() == len(self.entry.interpro_terms)
        assert models.PfamTerm.count() == len(self.entry.pfam_terms)
        assert models.Keyword.count() == len(self.entry.keywords)
        assert models.GeneSymbol.count() == len(self.entry.genes)

        # Check fields have been set appropriately.
        assert str(instance) == "P38398-1"
        assert instance.record.sequence == self.entry.sequence
        assert instance.record.organism == 9606
        assert instance.record.reviewed == self.entry.reviewed
        assert instance.record.version == self.entry.version
        assert instance.record.identifiers.count() == (
            len(self.entry.accessions) + 1
        )

        # Check primary is correctly associated with first accession
        for record_identifier in instance.record.identifiers:
            if str(record_identifier.identifier) == "P38398":
                assert record_identifier.primary
            else:
                assert not record_identifier.primary

        assert instance.record.go_annotations.count() == len(
            self.entry.go_terms
        )
        assert instance.record.interpro_annotations.count() == len(
            self.entry.interpro_terms
        )
        assert instance.record.pfam_annotations.count() == len(
            self.entry.pfam_terms
        )
        assert instance.record.keywords.count() == len(self.entry.keywords)
        assert instance.record.genes.count() == len(self.entry.genes)

        for record_gene in instance.record.genes:
            if str(record_gene.gene) == "BRCA1":
                assert record_gene.relation == "primary"
                assert record_gene.primary == True
            else:
                assert record_gene.relation == "synonym"
                assert record_gene.primary == False

    def test_handles_duplicates(self):
        # Check all tables are empty before call.
        self.assert_tables_empty()
        query = utilities.create_proteins(
            [("P38398-1", self.entry), ("P38398", self.entry)]
        )
        assert models.UniprotRecord.count() == 1
        assert len(query) == 2

    def test_updates_existing_protein_data_case_insensitive_accession(self):
        instance = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P38398"),
            record=models.UniprotRecord.create(
                sequence="LMN", organism=1000, reviewed=False, version=1
            ),
        )

        # These should be updated
        models.UniprotRecordIdentifier.create(
            identifier=instance.identifier,
            record=instance.record,
            primary=False,
        )
        models.UniprotRecordGene.create(
            record=instance.record,
            gene=models.GeneSymbol.create(text="BRCA1"),
            relation="orf",
        )

        # These should be deleted by update
        models.UniprotRecordIdentifier.create(
            identifier=models.UniprotIdentifier.create(identifier="P12345"),
            record=instance.record,
            primary=True,
        )
        models.UniprotRecordGene.create(
            record=instance.record,
            gene=models.GeneSymbol.create(text="BRCA2"),
            relation="primary",
        )

        # case insensitive
        query = utilities.create_proteins([("P38398".lower(), self.entry)])
        assert len(query) == 1

        # Check all tables have been populated.
        instance: models.Protein = query[0]
        assert models.UniprotRecord.count() == 1
        assert models.UniprotIdentifier.count() == (
            len(self.entry.accessions) + 1
        )
        assert models.GeneOntologyIdentifier.count() == len(
            self.entry.go_terms
        )
        assert models.InterproIdentifier.count() == len(
            self.entry.interpro_terms
        )
        assert models.PfamIdentifier.count() == len(self.entry.pfam_terms)
        assert models.KeywordIdentifier.count() == len(self.entry.keywords)
        assert models.GeneOntologyTerm.count() == len(self.entry.go_terms)
        assert models.InterproTerm.count() == len(self.entry.interpro_terms)
        assert models.PfamTerm.count() == len(self.entry.pfam_terms)
        assert models.Keyword.count() == len(self.entry.keywords)
        assert models.GeneSymbol.count() == len(self.entry.genes) + 1

        # Check fields have been set appropriately.
        assert str(instance) == "P38398"
        assert instance.record.sequence == self.entry.sequence
        assert instance.record.organism == 9606
        assert instance.record.reviewed == self.entry.reviewed
        assert instance.record.version == self.entry.version
        assert instance.record.identifiers.count() == len(
            self.entry.accessions
        )

        assert "P12345" not in set(
            str(i.identifier for i in instance.record.identifiers)
        )

        # Check primary is correctly associated with first accession
        for record_identifier in instance.record.identifiers:
            if str(record_identifier.identifier) == "P38398":
                assert record_identifier.primary
            else:
                assert not record_identifier.primary

        assert instance.record.go_annotations.count() == len(
            self.entry.go_terms
        )
        assert instance.record.interpro_annotations.count() == len(
            self.entry.interpro_terms
        )
        assert instance.record.pfam_annotations.count() == len(
            self.entry.pfam_terms
        )
        assert instance.record.keywords.count() == len(self.entry.keywords)
        assert instance.record.genes.count() == len(self.entry.genes)
        assert "BRCA2" not in set(str(g.gene) for g in instance.record.genes)

        for record_gene in instance.record.genes:
            if str(record_gene.gene) == "BRCA1":
                assert record_gene.relation == "primary"
                assert record_gene.primary == True
            else:
                assert record_gene.relation == "synonym"
                assert record_gene.primary == False


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
        utilities.create_proteins(
            [
                (self.entry_a.primary_accession, self.entry_a),
                (self.entry_b.primary_accession, self.entry_b),
            ]
        )
        assert models.UniprotRecord.count() == 2

    def assert_tables_empty(self):
        assert models.Interaction.count() == 0
        assert models.InteractionEvidence.count() == 0
        assert models.InteractionLabel.count() == 0
        assert models.InteractionDatabase.count() == 0
        assert models.PsimiIdentifier.count() == 0
        assert models.PubmedIdentifier.count() == 0

    def test_creates_new_interaction(self):
        interaction = types.InteractionData(
            source="P38398",
            target="P02760",
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
        assert str(instance.evidence.first().pubmed) == "123456"
        assert str(instance.evidence.first().psimi) == "MI:0001"
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
                labels=["Activation"],
                databases=["Kegg"],
                evidence=[
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0001"
                    ),
                    types.InteractionEvidenceData(
                        pubmed="123456", psimi="MI:0002"
                    ),
                ],
            ),
            types.InteractionData(
                source="P38398",
                target="P38398",
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
        ) == [("123456", "MI:0001"), ("123456", "MI:0002")]
        assert query[0].compact == ("P38398", "P02760")
        assert sorted([str(l) for l in query[0].labels]) == ["activation"]
        assert sorted([str(d) for d in query[0].databases]) == ["kegg"]

        # @1
        assert sorted(
            [(str(e.pubmed), str(e.psimi)) for e in query[1].evidence],
            key=lambda e: e[-1],
        ) == [("123456", "MI:0002")]
        assert query[1].compact == ("P38398", "P38398")
        assert sorted([str(l) for l in query[1].labels]) == [
            "activation",
            "methylation",
        ]
        assert sorted([str(d) for d in query[1].databases]) == ["hprd"]

    def test_updates_existing_interaction_case_insensitive(self):
        proteins = models.Protein.all()
        interaction = models.Interaction.create(
            source=proteins[0], target=proteins[1]
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
                source="P38398".lower(),
                target="P02760".lower(),
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
        ) == [("123456", "MI:0001")]

        assert instance.compact == ("P38398", "P02760")

        assert sorted([str(l) for l in instance.labels]) == [
            "activation",
            "methylation",
        ]
        assert sorted([str(d) for d in instance.databases]) == ["hprd", "kegg"]

    def test_error_source_not_a_valid_accession(self):
        interaction = types.InteractionData(source="aaa", target="P02760")
        with pytest.raises(ValueError):
            utilities.create_interactions([interaction])

    def test_error_target_not_a_valid_accession(self):
        interaction = types.InteractionData(source="P38398", target="aaa")
        with pytest.raises(ValueError):
            utilities.create_interactions([interaction])

    # def test_skips_interaction_if_source_not_found_on_uniprot(self):
    #     interaction = types.InteractionData(source="P38398", target="P12345")
    #     query = utilities.create_interactions([interaction])
    #     assert len(query) == 0

    # def test_skips_interaction_if_target_not_found_on_uniprot(self):
    #     interaction = types.InteractionData(source="P12345", target="P02760")
    #     query = utilities.create_interactions([interaction])
    #     assert len(query) == 0


class TestUpdateAccessions:
    def test_removes_interactions_with_no_source_mapping(self):
        mapping = {"P38398": []}
        instance = types.InteractionData(source="P38398", target="P02760")
        assert not utilities.update_accessions([instance], mapping=mapping)

    def test_removes_interactions_with_no_target_mapping(self):
        mapping = {"P02760": []}
        instance = types.InteractionData(source="P38398", target="P02760")
        assert not utilities.update_accessions([instance], mapping=mapping)

    def test_maps_old_accession_to_first_item_in_mapping_result(self):
        mapping = {"P38398": ["A", "B"], "P02760": ["C", "D"]}
        instance = types.InteractionData(source="P38398", target="P02760")
        result = utilities.update_accessions([instance], mapping=mapping)

        assert len(result) == 1
        assert result[0].source == "A"
        assert result[0].target == "C"
