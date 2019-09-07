import peewee
import pytest

from ....tests.test_utilities import null_values
from ... import models
from ....constants import GeneOntologyCategory
from .. import DatabaseTestMixin


class TestProteinDataModel(DatabaseTestMixin):
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
        self.identifier_1 = models.UniprotIdentifier.create(
            identifier="P12345"
        )
        self.identifier_2 = models.UniprotIdentifier.create(
            identifier="Q54321"
        )

        self.instance = models.ProteinData.create(
            organism=9606, sequence="aaa", reviewed=True, version="1"
        )
        self.instance.go_annotations = [self.cc, self.bp, self.mf]
        self.instance.identifiers = [self.identifier_1, self.identifier_2]

    def test_uppercases_sequence(self):
        assert self.instance.sequence == "AAA"

    def test_nulls_sequence_if_null(self):
        for value in null_values:
            self.instance.sequence = value
            with pytest.raises(peewee.IntegrityError):
                self.instance.save()

    def test_get_by_identifier_filter_is_case_insensitive(self):
        instance = models.ProteinData.create(
            organism=9606, sequence="aaa", reviewed=True, version="2"
        )
        identifier = models.UniprotIdentifier.create(identifier="O54321")
        instance.identifiers = [identifier]

        query = models.ProteinData.get_by_identifier(
            [str(self.identifier_1).lower()]
        )
        assert query.count() == 1
        assert self.instance in query
        assert instance not in query

    def test_go_cc_returns_cc_terms(self):
        assert self.instance.go_cc.count() == 1
        assert self.cc in self.instance.go_cc

    def test_go_mf_returns_cc_terms(self):
        assert self.instance.go_mf.count() == 1
        assert self.mf in self.instance.go_mf

    def test_go_bp_returns_cc_terms(self):
        assert self.instance.go_bp.count() == 1
        assert self.bp in self.instance.go_bp


class TestProteinModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.identifier = models.UniprotIdentifier.create(identifier="P12345")
        self.data = models.ProteinData.create(
            organism=9606, sequence="aaa", reviewed=True, version="1"
        )
        self.data.identifiers = [self.identifier]
        self.instance = models.Protein.create(
            identifier=self.identifier, data=self.data
        )

    def test_str_returns_str_identifier(self):
        assert str(self.instance) == "P12345"

    def test_str_returns_none_if_identifier_is_none(self):
        self.instance.identifier = None
        assert str(self.instance) == str(None)

    def test_get_by_identifier_filter_is_case_insensitive(self):
        instance = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="O54321"),
            data=models.ProteinData.create(
                organism=9606, sequence="aaa", reviewed=True, version="1"
            ),
        )
        query = models.Protein.get_by_identifier([str(self.instance).lower()])
        assert query.count() == 1
        assert self.instance in query
        assert instance not in query
