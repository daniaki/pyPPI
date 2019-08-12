import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from peewee import IntegrityError

from ....constants import GeneOntologyCategory
from ... import models
from .. import DatabaseTestMixin


class TestInteractionLabelModel(DatabaseTestMixin):
    def test_capitalizes_label(self):
        label = models.InteractionLabel.create(text="activation")
        assert label.text == "Activation"


class TestInteractionEvidence(DatabaseTestMixin):
    def test_unique_index_on_source_target(self):
        psimi = models.PsimiIdentifier.create(identifier="1")
        pmid = models.PubmedIdentifier.create(identifier="1123")
        _ = models.InteractionEvidence.create(pubmed=pmid, psimi=psimi)
        with pytest.raises(IntegrityError):
            models.InteractionEvidence.create(pubmed=pmid, psimi=psimi)


class TestInteractionModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.protein_a = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P1234"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="MLPGA",
        )
        self.protein_b = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P5678"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA2"),
            sequence="EDALM",
        )

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
            identifier=models.GeneOntologyIdentifier.create(identifier="1"),
            name="",
            description="",
            category=GeneOntologyCategory.molecular_function,
        )
        go2 = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(identifier="2"),
            name="",
            description="",
            category=GeneOntologyCategory.molecular_function,
        )
        pfam = models.PfamTerm.create(
            identifier=models.PfamIdentifier.create(identifier="PF1"),
            name="",
            description="",
        )
        ipr = models.InterproTerm.create(
            identifier=models.InterproIdentifier.create(identifier="IPR1"),
            name="",
            description="",
        )
        self.protein_a.go_annotations.add([go2, go1])
        self.protein_a.pfam_annotations.add([pfam])
        self.protein_a.interpro_annotations.add([ipr])

        models.Interaction.create(
            source=self.protein_a, target=self.protein_b, organism=9606
        )

        result = models.Interaction.to_dataframe()
        expected = pd.DataFrame(
            data=[
                {
                    "source": "P1234",
                    "target": "P5678",
                    "go_mf": "GO:1,GO:2",
                    "go_bp": None,
                    "go_cc": None,
                    "keyword": None,
                    "interpro": "IPR1",
                    "pfam": "PF1",
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
            index=["P1234,P5678"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_to_dataframe_combines_metadata_into_csv_string_or_none_if_absent(
        self
    ):
        i = models.Interaction.create(
            source=self.protein_a, target=self.protein_b, organism=9606
        )
        label_1 = models.InteractionLabel.create(text="Activation")
        label_2 = models.InteractionLabel.create(text="Methylation")
        i.labels.add([label_1, label_2])

        psimi1 = models.PsimiIdentifier.create(identifier="1")
        psimi2 = models.PsimiIdentifier.create(identifier="2")
        pmid = models.PubmedIdentifier.create(identifier="1123")
        evidence1 = models.InteractionEvidence.create(
            pubmed=pmid, psimi=psimi1
        )
        evidence2 = models.InteractionEvidence.create(
            pubmed=pmid, psimi=psimi2
        )
        database1 = models.InteractionDatabase.create(name="Kegg")
        database2 = models.InteractionDatabase.create(name="hprd")

        # Add in reverse order to check if sorts
        i.evidence.add([evidence2, evidence1])
        i.databases.add([database2, database1])

        result = models.Interaction.to_dataframe()
        expected = pd.DataFrame(
            data=[
                {
                    "source": "P1234",
                    "target": "P5678",
                    "go_mf": None,
                    "go_bp": None,
                    "go_cc": None,
                    "keyword": None,
                    "interpro": None,
                    "pfam": None,
                    "label": "Activation,Methylation",
                    "evidence": "PMID:1123|MI:1,PMID:1123|MI:2",
                    "database": "Hprd,Kegg",
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
            index=["P1234,P5678"],
        )
        assert_frame_equal(result, expected, check_dtype=False)
