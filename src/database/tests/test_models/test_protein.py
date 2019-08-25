import peewee
import pytest

from ....tests.test_utilities import null_values
from ... import models
from ....constants import GeneOntologyCategory
from .. import DatabaseTestMixin


class TestProteinModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.cc = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000001"
            ),
            category=GeneOntologyCategory.cellular_component,
        )
        self.bp = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000002"
            ),
            category=GeneOntologyCategory.biological_process,
        )
        self.mf = models.GeneOntologyTerm.create(
            identifier=models.GeneOntologyIdentifier.create(
                identifier="GO:0000003"
            ),
            category=GeneOntologyCategory.molecular_function,
        )

        self.protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P12345"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="aaa",
        )
        self.protein.go_annotations.add([self.cc, self.bp, self.mf])

    def test_str_returns_str_identifier_or_none(self):
        assert str(self.protein) == "P12345"
        self.protein.identifier = None
        assert str(self.protein) == str(None)

    def test_uppercases_sequence(self):
        assert self.protein.sequence == "AAA"

    def test_nulls_sequence_if_null(self):
        for value in null_values:
            self.protein.sequence = value
            with pytest.raises(peewee.IntegrityError):
                self.protein.save()

    def test_get_by_identifier_filters_by_uppercase(self):
        protein_2 = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P12346"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA2"),
            sequence="aaa",
        )
        query = models.Protein.get_by_identifier([str(self.protein.identifier)])
        assert query.count() == 1
        assert self.protein in query
        assert protein_2 not in query

    def test_go_cc_returns_cc_terms(self):
        assert self.protein.go_cc.count() == 1
        assert self.cc in self.protein.go_cc

    def test_go_mf_returns_cc_terms(self):
        assert self.protein.go_mf.count() == 1
        assert self.mf in self.protein.go_mf

    def test_go_bp_returns_cc_terms(self):
        assert self.protein.go_bp.count() == 1
        assert self.bp in self.protein.go_bp

