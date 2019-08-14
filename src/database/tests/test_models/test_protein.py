import peewee
import pytest

from ....tests.test_utilities import null_values
from ... import models
from .. import DatabaseTestMixin


class TestProteinModel(DatabaseTestMixin):
    def test_str_returns_str_identifier_or_none(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P12345"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="aaa",
        )
        assert str(protein) == "P12345"

        protein.identifier = None
        assert str(protein) == str(None)

    def test_uppercases_sequence(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P12345"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="aaa",
        )
        assert protein.sequence == "AAA"

    def test_nulls_sequence_if_null(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P12345"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="AAA",
        )
        for value in null_values:
            protein.sequence = value
            with pytest.raises(peewee.IntegrityError):
                protein.save()
