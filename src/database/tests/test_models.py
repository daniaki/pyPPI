import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from peewee import IntegrityError

from ...constants import GeneOntologyCategory
from .. import models
from . import DatabaseTestMixin


class TestBaseModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.instance = models.InteractionLabel.create(text="activation")

    def test_updates_modified_date_on_save(self):
        now = self.instance.modified
        self.instance.save()
        later = self.instance.modified
        assert later > now


class TestIdentifierMixin:
    class Driver(models.IdentifierMixin):
        def __init__(self, identifier: str):
            self.identifier = identifier

    def test_attribute_error_missing_identifier_attr_in_class(self):
        class MissingDriver(models.IdentifierMixin):
            pass

        with pytest.raises(AttributeError):
            MissingDriver()._get_identifier()
    
    def test_type_error_identifier_not_a_string(self):
        instance = self.Driver(identifier=1)
        with pytest.raises(TypeError):
            instance._get_identifier()
    
    def test_prefix_prepends_prefix_if_not_starts_with(self):
        value = self.Driver(identifier="hello world").prefix("GO", ":")
        assert value == "GO:hello world"

        value = self.Driver(identifier="go:hello world").prefix("GO", ":")
        assert value == "go:hello world"

    def test_unprefix_removes_prefix(self):
        value = self.Driver(identifier="go:hello world").unprefix(":")
        assert value == "hello world"


class TestExternalIdentiferModel(DatabaseTestMixin):
    def test_raises_not_implemented_error_db_name_attr_not_set(self):
        i = models.ExternalIdentifier()
        with pytest.raises(NotImplementedError):
            i.save()

    def test_prepends_prefix_if_defined(self):
        i = models.PubmedIdentifier.create(identifier="1234")
        assert i.identifier == "PMID:1234"

    def test_does_not_prepend_prefix_if_already_present(self):
        i = models.PubmedIdentifier.create(identifier="PMID:1234")
        assert i.identifier == "PMID:1234"

    def test_does_not_prepend_prefix_if_not_defined(self):
        i = models.UniprotIdentifier.create(identifier="P1234")
        assert i.identifier == "P1234"

    def test_uppercases_identifier(self):
        i = models.PubmedIdentifier.create(identifier="pmid:1234")
        assert i.identifier == "PMID:1234"

    def test_sets_db_name(self):
        i = models.PubmedIdentifier.create(identifier="pmid:1234")
        assert i.dbname == models.PubmedIdentifier.DB_NAME


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


class TestExperimentTypelModel(DatabaseTestMixin):
    def test_capitalizes_and_strips_text(self):
        instance = models.ExperimentType.create(text="  in vitro  ")
        assert instance.text == "In vitro"


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


class TestProteinModel(DatabaseTestMixin):
    def test_uppercases_sequence(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P1234"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="aaa",
        )
        assert protein.sequence == "AAA"


class TestInteractionLabelModel(DatabaseTestMixin):
    def test_capitalizes_label(self):
        label = models.InteractionLabel.create(text="activation")
        assert label.text == "Activation"


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
            source=self.protein_a, target=self.protein_b, organism=9606
        )
        with pytest.raises(IntegrityError):
            models.Interaction.create(
                source=self.protein_a, target=self.protein_b, organism=9606
            )

    def test_to_dataframe_combines_annotations_into_sorted_string_or_none_if_absent(
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
                    "psimi": None,
                    "pmid": None,
                    "experiment_type": None,
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
                "psimi",
                "pmid",
                "experiment_type",
                "database",
            ],
            index=["P1234,P5678"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_to_dataframe_combines_metadata_into_sorted_string_or_none_if_absent(
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
        e_type = models.ExperimentType.create(text="In vivo")
        database1 = models.InteractionDatabase.create(name="Kegg")
        database2 = models.InteractionDatabase.create(name="hprd")

        i.psimi_ids.add([psimi2, psimi1])
        i.pubmed_ids.add([pmid])
        i.experiment_types.add([e_type])
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
                    "psimi": "MI:1,MI:2",
                    "pmid": "PMID:1123",
                    "experiment_type": "In vivo",
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
                "psimi",
                "pmid",
                "experiment_type",
                "database",
            ],
            index=["P1234,P5678"],
        )
        assert_frame_equal(result, expected, check_dtype=False)

    def test_update_or_create_creates_new_instance(self):
        instance: models.Interaction = models.Interaction.update_or_create(
            source=self.protein_a,
            target=self.protein_b,
            organism=9606,
            labels=["Activation", "Methylation"],
            psimi_ids=["PSI:1"],
            pubmed_ids=["PMID:1"],
            experiment_types=["In vitro"],
            databases=["Kegg"],
        )
        assert instance.source.id == self.protein_a.id
        assert instance.target.id == self.protein_b.id
        assert instance.organism == 9606
        assert instance.labels.count() == 2
        assert instance.psimi_ids.count() == 1
        assert instance.pubmed_ids.count() == 1
        assert instance.experiment_types.count() == 1
        assert instance.databases.count() == 1

    def test_update_or_create_updates_existing_metadata(self):
        instance_1: models.Interaction = models.Interaction.update_or_create(
            source=self.protein_a,
            target=self.protein_b,
            organism=9606,
            labels=["Activation"],
            psimi_ids=["PSI:1"],
            pubmed_ids=["PMID:1"],
            experiment_types=["In vitro"],
            databases=["Kegg"],
        )

        instance_2: models.Interaction = models.Interaction.update_or_create(
            source=self.protein_a,
            target=self.protein_b,
            organism=9606,
            labels=["Methylation"],
            psimi_ids=["PSI:2"],
            pubmed_ids=["PMID:2"],
            experiment_types=["In vivo"],
            databases=["Hprd"],
        )

        assert instance_1.id == instance_2.id
        assert instance_1.source.id == self.protein_a.id
        assert instance_1.target.id == self.protein_b.id
        assert instance_1.organism == 9606
        assert instance_1.labels.count() == 2
        assert instance_1.psimi_ids.count() == 2
        assert instance_1.pubmed_ids.count() == 2
        assert instance_1.experiment_types.count() == 2
        assert instance_1.databases.count() == 2

    def test_update_or_create_updates_organism_if_not_none(self):
        instance = models.Interaction = models.Interaction.update_or_create(
            source=self.protein_a,
            target=self.protein_b,
            labels=["Activation"],
            psimi_ids=["PSI:1"],
            pubmed_ids=["PMID:1"],
            experiment_types=["In vitro"],
            databases=["Kegg"],
        )

        models.Interaction = models.Interaction.update_or_create(
            source=self.protein_a,
            target=self.protein_b,
            organism=123,
            labels=["Methylation"],
            psimi_ids=["PSI:2"],
            pubmed_ids=["PMID:2"],
            experiment_types=["In vivo"],
            databases=["Hprd"],
        )

        instance = instance.refresh()
        assert instance.organism == 123

