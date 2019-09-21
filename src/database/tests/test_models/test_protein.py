import peewee
import pytest
from hashlib import md5

from ....tests.test_utilities import null_values
from ... import models
from ....constants import GeneOntologyCategory
from .. import DatabaseTestMixin


class TestUniprotRecordModel(DatabaseTestMixin):
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

        self.instance = models.UniprotRecord.create(
            organism=9606, sequence="aaa", reviewed=True, version=1
        )
        self.instance.go_annotations = [self.cc, self.bp, self.mf]

    def test_uppercases_sequence(self):
        assert self.instance.sequence == "AAA"

    def test_nulls_sequence_if_null(self):
        for value in null_values:
            self.instance.sequence = value
            with pytest.raises(peewee.IntegrityError):
                self.instance.save()

    def test_str_uses_primary_record_identifier(self):
        rid_1 = models.UniprotRecordIdentifier.create(
            identifier=self.identifier_1, record=self.instance, primary=True
        )
        assert str(rid_1.identifier) in str(self.instance)

    def test_primary_identifier_returns_primary_record_identifier(self):
        rid_1 = models.UniprotRecordIdentifier.create(
            identifier=self.identifier_1, record=self.instance, primary=True
        )
        rid_2 = models.UniprotRecordIdentifier.create(
            identifier=self.identifier_2, record=self.instance, primary=False
        )
        assert self.instance.primary_identifier.id == rid_1.id
        assert self.instance.primary_identifier.id != rid_2.id

    def test_sets_hash_on_save(self):
        assert (
            self.instance.data_hash
            == md5(
                str(
                    (
                        self.instance.organism,
                        self.instance.sequence,
                        self.instance.reviewed,
                        self.instance.version,
                    )
                ).encode("utf-8")
            ).hexdigest()
        )

    def test_get_by_identifier_filter_is_case_insensitive(self):
        other_instance = models.UniprotRecord.create(
            organism=9606, sequence="aaa", reviewed=True, version=2
        )
        models.UniprotRecordIdentifier.create(
            identifier=self.identifier_1, record=self.instance, primary=True
        )

        query = models.UniprotRecord.get_by_identifier(
            [str(self.identifier_1).lower()]
        )
        assert query.count() == 1
        assert self.instance in query
        assert other_instance not in query

    def test_go_cc_returns_cc_terms(self):
        assert self.instance.go_cc.count() == 1
        assert self.cc in self.instance.go_cc

    def test_go_mf_returns_cc_terms(self):
        assert self.instance.go_mf.count() == 1
        assert self.mf in self.instance.go_mf

    def test_go_bp_returns_cc_terms(self):
        assert self.instance.go_bp.count() == 1
        assert self.bp in self.instance.go_bp


class TestUniprotRecordIdentifier(DatabaseTestMixin):
    def test_str_uses_none_if_no_identifier(self):
        instance = models.UniprotRecordIdentifier()
        assert str(None) in str(instance)

    def test_error_if_creating_two_primary_ids_for_same_record(self):
        record = models.UniprotRecord.create(
            organism=9606, sequence="aaa", reviewed=True, version=1
        )
        id_1 = models.UniprotIdentifier.create(identifier="P12345")
        id_2 = models.UniprotIdentifier.create(identifier="P23456")
        models.UniprotRecordIdentifier.create(
            identifier=id_1, record=record, primary=True
        )
        with pytest.raises(ValueError):
            models.UniprotRecordIdentifier.create(
                identifier=id_2, record=record, primary=True
            )


class TestUniprotRecordGene(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.record = models.UniprotRecord.create(
            organism=9606, sequence="aaa", reviewed=True, version=1
        )

    def test_str_returns_none_if_identifier_is_none(self):
        instance = models.UniprotRecordGene(
            record=self.record, relation="PRIMARY"
        )
        assert str(None) in str(instance)

    def test_str_uses_gene_syumbol(self):
        instance = models.UniprotRecordGene.create(
            record=self.record,
            relation="PRIMARY",
            gene=models.GeneSymbol.create(text="BRCA1"),
        )
        assert "BRCA1" in str(instance)

    def test_format_lowers_relation(self):
        instance = models.UniprotRecordGene.create(
            record=self.record,
            relation="PRIMARY",
            gene=models.GeneSymbol.create(text="BRCA1"),
        )
        assert instance.relation == "primary"

    def test_format_raises_valueerror_invalid_relation_choice(self):
        with pytest.raises(ValueError):
            models.UniprotRecordGene.create(
                record=self.record,
                relation="unknown",
                gene=models.GeneSymbol.create(text="BRCA1"),
            )

    def test_format_infers_if_gene_is_primary(self):
        instance = models.UniprotRecordGene.create(
            record=self.record,
            relation="primary",
            gene=models.GeneSymbol.create(text="BRCA1"),
        )
        assert instance.primary

        instance = models.UniprotRecordGene.create(
            record=self.record,
            relation="orf",
            gene=models.GeneSymbol.create(text="BRCA2"),
        )
        assert not instance.primary

    def test_format_raises_error_more_than_one_primary_gene_for_record(self):
        models.UniprotRecordGene.create(
            record=self.record,
            relation="primary",
            gene=models.GeneSymbol.create(text="BRCA1"),
        )
        with pytest.raises(ValueError):
            models.UniprotRecordGene.create(
                record=self.record,
                relation="primary",
                gene=models.GeneSymbol.create(text="BRCA2"),
            )


class TestProteinModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.identifier = models.UniprotIdentifier.create(identifier="P12345")
        self.data = models.UniprotRecord.create(
            organism=9606, sequence="AML", reviewed=True, version=1
        )
        self.instance = models.Protein.create(
            identifier=self.identifier, record=self.data
        )

    def test_str_returns_str_identifier(self):
        assert str(self.instance) == "P12345"

    def test_str_returns_none_if_identifier_is_none(self):
        self.instance.identifier = None
        assert str(self.instance) == str(None)

    def test_get_by_identifier_filter_is_case_insensitive(self):
        other_instance = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="O54321"),
            record=models.UniprotRecord.create(
                organism=9606, sequence="AKL", reviewed=True, version=1
            ),
        )
        query = models.Protein.get_by_identifier([str(self.instance).lower()])
        assert query.count() == 1
        assert self.instance in query
        assert other_instance not in query
