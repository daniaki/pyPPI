from .. import DatabaseTestMixin
from ... import models


class TestProteinModel(DatabaseTestMixin):
    def test_uppercases_sequence(self):
        protein = models.Protein.create(
            identifier=models.UniprotIdentifier.create(identifier="P1234"),
            organism=9606,
            gene=models.GeneSymbol.create(text="BRCA1"),
            sequence="aaa",
        )
        assert protein.sequence == "AAA"
