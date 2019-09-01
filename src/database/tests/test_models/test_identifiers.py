from .. import DatabaseTestMixin

import pytest

from ....tests.test_utilities import null_values
from ... import models


class TestAddPrefix:
    def test_prefix_passthrough_if_prefix_not_defined(self):
        value = models.identifiers.add_prefix("0000001", None, ":")
        assert value == "0000001"

    def test_prefix_prepends_prefix_if_not_starts_with(self):
        value = models.identifiers.add_prefix("0000001", "GO", ":")
        assert value == "GO:0000001"
        # Check case insensitive
        value = models.identifiers.add_prefix("go:0000001", "GO", ":")
        assert value == "go:0000001"
        # Check no sep
        value = models.identifiers.add_prefix("000001", "IPR", None)
        assert value == "IPR000001"


class TestExternalIdentiferModel(DatabaseTestMixin):
    def test_not_implemented_error_validate(self):
        with pytest.raises(NotImplementedError):
            models.ExternalIdentifier.validate("")

    def test_str_returns_identifier_string(self):
        i = models.PubmedIdentifier.create(identifier="PUBMED:8619402")
        assert str(i) == "PUBMED:8619402"

    def test_raises_not_implemented_error_db_name_attr_not_set(self):
        i = models.ExternalIdentifier()
        with pytest.raises(NotImplementedError):
            i.save()

    def test_raises_typeerror_identifier_is_none(self):
        with pytest.raises(TypeError):
            models.PubmedIdentifier.create(identifier=None)

    def test_raises_value_error_identifier_does_not_validate(self):
        with pytest.raises(ValueError):
            models.PubmedIdentifier.create(identifier="aaa")

    def test_prepends_prefix_if_defined(self):
        i = models.PubmedIdentifier.create(identifier="8619402")
        assert i.identifier == "PUBMED:8619402"

    def test_does_not_prepend_prefix_if_already_present(self):
        i = models.PubmedIdentifier.create(identifier="PUBMED:8619402")
        assert i.identifier == "PUBMED:8619402"

    def test_does_not_prepend_prefix_if_not_defined(self):
        i = models.UniprotIdentifier.create(identifier="P12345")
        assert i.identifier == "P12345"

    def test_save_formats_identifier_as_upper_by_default(self):
        i = models.PubmedIdentifier.create(identifier="pubmed:8619402")
        assert i.identifier == "PUBMED:8619402"

    def test_sets_db_name(self):
        i = models.PubmedIdentifier.create(identifier="pubmed:8619402")
        assert i.dbname == models.PubmedIdentifier.DB_NAME

    def test_can_get_by_identifier(self):
        i1 = models.PubmedIdentifier.create(identifier="pubmed:8619402")
        _ = models.PubmedIdentifier.create(identifier="pubmed:8619403")

        query = models.PubmedIdentifier.get_by_identifier([str(i1)])
        assert query.count() == 1
        assert query.first().id == i1.id


class TestIdentifierValidators(DatabaseTestMixin):
    # model, valid identifier, invalid identifiers
    specs = (
        (
            models.InterproIdentifier,
            ("IPR000001",),
            ("IPR", "0000001", "random"),
        ),
        (
            models.GeneOntologyIdentifier,
            ("GO:0000001",),
            ("GO:", "000", "GO0000001", "random"),
        ),
        (models.PfamIdentifier, ("PF00001",), ("PF", "PF001", "random")),
        (models.PsimiIdentifier, ("MI:0001",), (":0001", "MI", "random")),
        (
            models.KeywordIdentifier,
            ("KW-0001",),
            ("KW0001", "00001", "KW-", "random"),
        ),
        (
            models.PubmedIdentifier,
            ("pubmed:8619402", "8619402"),
            ("pub:8619402", "pubmed:", "random"),
        ),
    )

    def test_fails_on_null_values(self):
        for model, _, _ in self.specs:
            for value in null_values:
                assert not model.validate(identifier=value)

    def test_specs(self):
        for (model, valid, invalid) in self.specs:
            for valid_id in valid:
                assert model.validate(identifier=valid_id) is not None

            for invalid_id in invalid:
                assert model.validate(identifier=invalid_id) is None

