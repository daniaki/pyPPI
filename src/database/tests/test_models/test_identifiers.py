from .. import DatabaseTestMixin

import pytest

from ....tests.test_utilities import null_values
from ... import models


class TestExternalIdentiferModel(DatabaseTestMixin):
    def test_not_implemented_error_for_is_valid(self):
        with pytest.raises(NotImplementedError):
            models.ExternalIdentifier.is_valid("")

    def test_str_returns_identifier_string(self):
        i = models.PubmedIdentifier.create(identifier="8619402")
        assert str(i) == "8619402"

    def test_raises_not_implemented_error_db_name_attr_not_set(self):
        i = models.ExternalIdentifier()
        with pytest.raises(NotImplementedError):
            i.save()

    def test_raises_valuerror_identifier_is_none(self):
        with pytest.raises(ValueError):
            models.PubmedIdentifier.create(identifier=None)

    def test_raises_value_error_identifier_does_not_validate(self):
        with pytest.raises(ValueError):
            models.PubmedIdentifier.create(identifier="aaa")

    def test_sets_db_name(self):
        i = models.PubmedIdentifier.create(identifier="8619402")
        assert i.dbname == models.PubmedIdentifier.DB_NAME

    def test_get_by_identifier_case_insensitive(self):
        i1 = models.UniprotIdentifier.create(identifier="P12345")
        _ = models.UniprotIdentifier.create(identifier="P12346")

        query = models.UniprotIdentifier.get_by_identifier([str(i1).lower()])
        assert query.count() == 1
        assert query.first().id == i1.id


class TestIdentifierValidators(DatabaseTestMixin):
    # model, valid identifier, invalid identifiers
    specs = (
        (
            models.InterproIdentifier,
            ("IPR000001",),
            ("IPR", "0000001", "random", "000001"),
        ),
        (
            models.GeneOntologyIdentifier,
            ("GO:0000001",),
            ("GO:", "000", "GO0000001", "random"),
        ),
        (models.PfamIdentifier, ("PF00001",), ("00001", "PF001", "random")),
        (models.PsimiIdentifier, ("MI:0001",), ("0001", "MI", "random")),
        (
            models.KeywordIdentifier,
            ("KW-0001",),
            ("0001", "KW0001", "00001", "KW-", "random"),
        ),
        (
            models.PubmedIdentifier,
            ("8619402", "12"),
            ("PUBMED:8619402", "pub:8619402", "pubmed:", "random"),
        ),
    )

    def test_fails_on_null_values(self):
        for model, _, _ in self.specs:
            for value in null_values:
                with pytest.raises(ValueError):
                    model.validate(identifier=value)

    def test_specs(self):
        for (model, valid, invalid) in self.specs:
            for valid_id in valid:
                model.validate(identifier=valid_id)

            for invalid_id in invalid:
                with pytest.raises(ValueError):
                    model.validate(identifier=invalid_id)

