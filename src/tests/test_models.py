import pytest

from peewee import IntegrityError

from .. import models, settings

from . import BaseTestCase


class TestBaseModel(BaseTestCase):
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

    def test_prefix_prepends_prefix_if_not_starts_with(self):
        value = self.Driver(identifier="hello world").prefix("GO", ":")
        assert value == "GO:hello world"

        value = self.Driver(identifier="go:hello world").prefix("GO", ":")
        assert value == "go:hello world"

    def test_unprefix_removes_prefix(self):
        value = self.Driver(identifier="go:hello world").unprefix(":")
        assert value == "hello world"


class TestExternalIdentiferModel(BaseTestCase):
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


class TestGeneOntologyTermModel(BaseTestCase):
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
        assert term.category == "Molecular function"

    def test_converts_single_letter_to_category(self):
        check_for = enumerate(
            zip(list("FPC"), models.GeneOntologyTerm.Category.list())
        )
        for i, (letter, expected) in check_for:
            term = models.GeneOntologyTerm.create(
                identifier=models.GeneOntologyIdentifier.create(
                    identifier=str(i)
                ),
                name="",
                description="",
                category=letter,
            )
            assert term.category == expected


class TestGeneSymbolModel(BaseTestCase):
    def test_uppercases_and_strips_text(self):
        symbol = models.GeneSymbol.create(text="  brca1  ")
        assert symbol.text == "BRCA1"


class TestExperimentTypelModel(BaseTestCase):
    def test_capitalizes_and_strips_text(self):
        symbol = models.ExperimentType.create(text="  in vitro  ")
        assert symbol.text == "In vitro"


class TestKeywordModel(BaseTestCase):
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


class TestProteinModel(BaseTestCase):
    def test_uppercases_sequence(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P1234"),
            gene_name=models.GeneSymbol.create(text="BRCA1"),
            sequence="aaa",
        )
        assert protein.sequence == "AAA"


class TestInteractionLabelModel(BaseTestCase):
    def test_capitalizes_label(self):
        label = models.InteractionLabel.create(text="activation")
        assert label.text == "Activation"


class TestInteractionModel(BaseTestCase):
    def setup(self):
        super().setup()
        self.protein_a = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P1234"),
            gene_name=models.GeneSymbol.create(text="BRCA1"),
            sequence="MLPGA",
        )
        self.protein_b = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P5678"),
            gene_name=models.GeneSymbol.create(text="BRCA2"),
            sequence="EDALM",
        )

    def test_unique_index_on_source_target_organism(self):
        _ = models.Interaction.create(
            source=self.protein_a, target=self.protein_b, organism=9606
        )
        with pytest.raises(IntegrityError):
            models.Interaction.create(
                source=self.protein_a, target=self.protein_b, organism=9606
            )

    def test_unique_index_on_target_source_organism(self):
        models.Interaction.create(
            source=self.protein_a, target=self.protein_b, organism=9606
        )
        with pytest.raises(IntegrityError):
            # Switch order on source and target
            models.Interaction.create(
                source=self.protein_b, target=self.protein_a, organism=9606
            )

    def test_format_xy_returns_list_of_tuples(self):
        go1 = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(identifier="1"),
            name="",
            description="",
            category=models.GeneOntologyTerm.Category.molecular_function,
        )
        go2 = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(identifier="2"),
            name="",
            description="",
            category=models.GeneOntologyTerm.Category.molecular_function,
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
        self.protein_a.go_annotations.add([go1, go2])
        self.protein_a.pfam_annotations.add([pfam])
        self.protein_a.interpro_annotations.add([ipr])

        i = models.Interaction.create(
            source=self.protein_a, target=self.protein_b, organism=9606
        )
        label_1 = models.InteractionLabel.create(text="Activation")
        label_2 = models.InteractionLabel.create(text="Methylation")
        i.labels.add([label_1, label_2])

        data = list(models.Interaction.format_xy())
        expected = [
            (["GO:1", "GO:2", "PF1", "IPR1"], ["Activation", "Methylation"])
        ]
        assert data == expected
