import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from peewee import IntegrityError

from ....constants import GeneOntologyCategory
from ... import models
from .. import DatabaseTestMixin


class TestInteractionDatabaseModel(DatabaseTestMixin):
    def test_formats_and_strips_text(self):
        instance = models.InteractionDatabase.create(name="  KEGG  ")
        assert instance.name == "kegg"


class TestInteractionLabelModel(DatabaseTestMixin):
    def test_formats_label(self):
        label = models.InteractionLabel.create(text="Activation")
        assert label.text == "activation"


class TestInteractionEvidence(DatabaseTestMixin):
    def test_str_returns_string_of_identifiers(self):
        psimi = models.PsimiIdentifier.create(identifier="MI:0001")
        pmid = models.PubmedIdentifier.create(identifier="8619402")
        evidence = models.InteractionEvidence.create(pubmed=pmid, psimi=psimi)
        assert str(evidence) == "8619402|MI:0001"

        evidence.psimi = None
        assert str(evidence) == "8619402|None"

        evidence.pubmed = None
        assert str(evidence) == str(None)

    def test_unique_index_on_psimi_and_pubmed(self):
        psimi = models.PsimiIdentifier.create(identifier="MI:0001")
        pmid = models.PubmedIdentifier.create(identifier="8619402")
        _ = models.InteractionEvidence.create(pubmed=pmid, psimi=psimi)

        # Test can create with same pmid but different pismi
        _ = models.InteractionEvidence.create(
            pubmed=pmid,
            psimi=models.PsimiIdentifier.create(identifier="MI:0002"),
        )

        with pytest.raises(IntegrityError):
            models.InteractionEvidence.create(pubmed=pmid, psimi=psimi)

    def test_filter_by_pubmed_and_psimi_returns_none_if_no_queries(self):
        _ = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="1"),
            psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
        )
        query = models.InteractionEvidence.filter_by_pubmed_and_psimi([])
        assert query.count() == 0

    def test_filter_by_pubmed_and_psimi_case_insensitive(self):
        pmid = models.PubmedIdentifier.create(identifier="1")
        e1 = models.InteractionEvidence.create(
            pubmed=pmid,
            psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
        )
        e2 = models.InteractionEvidence.create(
            pubmed=pmid,
            psimi=models.PsimiIdentifier.create(identifier="MI:0002"),
        )

        # Should prefix 'pubmed' to 1
        query = models.InteractionEvidence.filter_by_pubmed_and_psimi(
            [("1", "mi:0001"), ("2", "mi:0002")]
        )
        assert query.count() == 1
        assert e1 in query
        assert e2 not in query

    def test_filter_by_pubmed_and_psimi_null_psimi(self):
        e1 = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="1"),
            psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
        )
        e2 = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="2"), psimi=None
        )

        # Should prefix 'pubmed' to 1
        query = models.InteractionEvidence.filter_by_pubmed_and_psimi(
            [("2", None)]
        )
        assert query.count() == 1
        assert e1 not in query
        assert e2 in query

    def test_filter_by_pubmed_and_psimi_allows_null_and_non_null_psimi(self):
        e1 = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="1"),
            psimi=models.PsimiIdentifier.create(identifier="MI:0001"),
        )
        e2 = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="2"), psimi=None
        )

        # Should prefix 'pubmed' to 1
        query = models.InteractionEvidence.filter_by_pubmed_and_psimi(
            [("1", "mi:0001"), ("2", None)]
        )
        assert query.count() == 2
        assert e1 in query
        assert e2 in query


class TestInteractionModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.identifier_a = models.UniprotIdentifier.create(
            identifier="P12345"
        )
        self.identifier_b = models.UniprotIdentifier.create(
            identifier="P56785"
        )

        self.protein_a = models.Protein.create(
            identifier=self.identifier_a,
            record=models.UniprotRecord.create(
                organism=9606, sequence="MLPGA", reviewed=True, version=1
            ),
        )
        self.protein_b = models.Protein.create(
            identifier=self.identifier_b,
            record=models.UniprotRecord.create(
                organism=9606, sequence="EDALM", reviewed=True, version=1
            ),
        )

    def test_compact_returns_identifier_string_tuple(self):
        instance = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        assert instance.compact == (str(self.protein_a), str(self.protein_b))

        instance.target = None
        assert instance.compact == (str(self.protein_a), None)

        instance.source = None
        assert instance.compact == (None, None)

    def test_str_returns_str_of_compact(self):
        instance = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        assert str(instance) == str(instance.compact)

    def test_unique_index_on_source_target(self):
        _ = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        with pytest.raises(IntegrityError):
            models.Interaction.create(
                source=self.protein_a, target=self.protein_b
            )

    def test_to_dataframe_combines_annotations_into_csv_string_or_none_if_absent(
        self
    ):
        go1 = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000001"
            ),
            category=GeneOntologyCategory.molecular_function,
        )
        go2 = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000002"
            ),
            category=GeneOntologyCategory.molecular_function,
        )
        pfam = models.PfamTerm.create(
            identifier=models.PfamIdentifier.create(identifier="PF00001")
        )
        ipr = models.InterproTerm.create(
            identifier=models.InterproIdentifier.create(identifier="IPR000001")
        )
        self.protein_a.record.go_annotations = [go2, go1]
        self.protein_a.record.pfam_annotations = [pfam]
        self.protein_a.record.interpro_annotations = [ipr]

        models.Interaction.create(source=self.protein_a, target=self.protein_b)

        result = models.Interaction.to_dataframe()
        expected = pd.DataFrame(
            data=[
                {
                    "source": "P12345",
                    "target": "P56785",
                    "go_mf": "GO:0000001,GO:0000002",
                    "go_bp": None,
                    "go_cc": None,
                    "keyword": None,
                    "interpro": "IPR000001",
                    "pfam": "PF00001",
                    "label": None,
                    "evidence": None,
                    "database": None,
                }
            ],
            columns=[
                "source",
                "target",
                "go_mf",
                "go_bp",
                "go_cc",
                "keyword",
                "interpro",
                "pfam",
                "label",
                "evidence",
                "database",
            ],
            index=["P12345,P56785"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_to_dataframe_combines_metadata_into_csv_string_or_none_if_absent(
        self
    ):
        i = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        label_1 = models.InteractionLabel.create(text="activation")
        label_2 = models.InteractionLabel.create(text="methylation")
        i.labels = [label_1, label_2]

        psimi1 = models.PsimiIdentifier.create(identifier="MI:0001")
        psimi2 = models.PsimiIdentifier.create(identifier="MI:0002")
        pmid = models.PubmedIdentifier.create(identifier="8619402")
        evidence1 = models.InteractionEvidence.create(
            pubmed=pmid, psimi=psimi1
        )
        evidence2 = models.InteractionEvidence.create(
            pubmed=pmid, psimi=psimi2
        )
        database1 = models.InteractionDatabase.create(name="kegg")
        database2 = models.InteractionDatabase.create(name="hprd")

        # Add in reverse order to check if sorts
        i.evidence = [evidence2, evidence1]
        i.databases = [database2, database1]

        result = models.Interaction.to_dataframe()
        expected = pd.DataFrame(
            data=[
                {
                    "source": "P12345",
                    "target": "P56785",
                    "go_mf": None,
                    "go_bp": None,
                    "go_cc": None,
                    "keyword": None,
                    "interpro": None,
                    "pfam": None,
                    "label": "activation,methylation",
                    "evidence": "8619402|MI:0001,8619402|MI:0002",
                    "database": "hprd,kegg",
                }
            ],
            columns=[
                "source",
                "target",
                "go_mf",
                "go_bp",
                "go_cc",
                "keyword",
                "interpro",
                "pfam",
                "label",
                "evidence",
                "database",
            ],
            index=["P12345,P56785"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_filter_by_pmid(self):
        i1 = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        i2 = models.Interaction.create(
            source=self.protein_b, target=self.protein_a
        )

        e1 = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="1")
        )
        e2 = models.InteractionEvidence.create(
            pubmed=models.PubmedIdentifier.create(identifier="2")
        )

        i1.evidence = [e1]
        i2.evidence = [e2]

        query = models.Interaction.filter_by_pmid(["1"])
        assert query.count() == 1
        assert i1 in query
        assert i2 not in query

    def test_filter_by_label_case_insensitive(self):
        i1 = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        i2 = models.Interaction.create(
            source=self.protein_b, target=self.protein_a
        )

        l1 = models.InteractionLabel.create(text="activation")
        l2 = models.InteractionLabel.create(text="methylation")

        i1.labels = [l1]
        i2.labels = [l2]

        query = models.Interaction.filter_by_label(["Activation"])
        assert query.count() == 1
        assert i1 in query
        assert i2 not in query

    def test_filter_by_db_case_insensitive(self):
        i1 = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        i2 = models.Interaction.create(
            source=self.protein_b, target=self.protein_a
        )

        d1 = models.InteractionDatabase.create(name="kegg")
        d2 = models.InteractionDatabase.create(name="hprd")

        i1.databases = [d1]
        i2.databases = [d2]

        query = models.Interaction.filter_by_database(["KEGG"])
        assert query.count() == 1
        assert i1 in query
        assert i2 not in query

    def test_filter_by_source_case_insensitive(self):
        i1 = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        i2 = models.Interaction.create(
            source=self.protein_b, target=self.protein_a
        )

        query = models.Interaction.filter_by_source(
            [str(self.protein_a).upper()]
        )
        assert query.count() == 1
        assert i1 in query
        assert i2 not in query

    def test_filter_by_target_case_insensitive(self):
        i1 = models.Interaction.create(
            source=self.protein_a, target=self.protein_b
        )
        i2 = models.Interaction.create(
            source=self.protein_b, target=self.protein_a
        )

        query = models.Interaction.filter_by_target(
            [str(self.protein_b).upper()]
        )
        assert query.count() == 1
        assert i1 in query
        assert i2 not in query

    # def test_filter_by_edge_format_insensitive(self):
    #     i1 = models.Interaction.create(
    #         source=self.protein_a, target=self.protein_b
    #     )
    #     i2 = models.Interaction.create(
    #         source=self.protein_a, target=self.protein_a
    #     )

    #     query = models.Interaction.filter_by_edge(
    #         [
    #             (str(self.protein_a).lower(), str(self.protein_b).lower()),
    #             (str(self.protein_b), str(self.protein_a)),
    #         ]
    #     )
    #     assert query.count() == 1
    #     assert i1 in query
    #     assert i2 not in query

