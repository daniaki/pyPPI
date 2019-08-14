from .. import DatabaseTestMixin

import pytest

from ....tests.test_utilities import null_values
from ... import models


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

    def test_prefix_passthrough_if_prefix_not_defined(self):
        value = self.Driver(identifier="0000001").prefix(None, ":")
        assert value == "0000001"

    def test_prefix_prepends_prefix_if_not_starts_with(self):
        value = self.Driver(identifier="0000001").prefix("GO", ":")
        assert value == "GO:0000001"
        # Check case insensitive
        value = self.Driver(identifier="go:0000001").prefix("GO", ":")
        assert value == "go:0000001"
        # Check no sep
        value = self.Driver(identifier="000001").prefix("IPR", None)
        assert value == "IPR000001"

    def test_unprefix_passthrough_if_prefix_not_defined(self):
        value = self.Driver(identifier="GO:0000001").unprefix(None, ":")
        assert value == "GO:0000001"

    def test_unprefix_removes_prefix_with_sep(self):
        value = self.Driver(identifier="go:0000001").unprefix("GO", ":")
        assert value == "0000001"

    def test_unprefix_removes_prefix_without_sep(self):
        value = self.Driver(identifier="IPR000001").unprefix("IPR", None)
        assert value == "000001"


class TestExternalIdentiferModel(DatabaseTestMixin):
    def test_not_implemented_error_validate(self):
        with pytest.raises(NotImplementedError):
            models.ExternalIdentifier().validate()

    def test_str_returns_identifier_string(self):
        i = models.PubmedIdentifier.create(identifier="PUBMED:8619402")
        assert str(i) == "PUBMED:8619402"

    def test_raises_not_implemented_error_db_name_attr_not_set(self):
        i = models.ExternalIdentifier()
        with pytest.raises(NotImplementedError):
            i.save()

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


class TestIdentifierValidators(DatabaseTestMixin):
    # model, valid identifier, invalid identifiers
    specs = (
        (
            models.InterproIdentifier,
            ("IPR000001", "000001"),
            ("IPR", "0000001", "random"),
        ),
        (
            models.GeneOntologyIdentifier,
            ("GO:0000001", "0000001"),
            ("GO:", "000", "GO0000001", "random"),
        ),
        (
            models.PfamIdentifier,
            ("PF00001", "00001"),
            ("PF", "PF001", "random"),
        ),
        (
            models.PsimiIdentifier,
            ("MI:0001", "0001"),
            (":0001", "MI", "random"),
        ),
        (
            models.KeywordIdentifier,
            ("0001", "KW-0001"),
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
                with pytest.raises(ValueError):
                    model.create(identifier=value)

    def test_specs(self):
        for (model, valid, invalid) in self.specs:
            for valid_id in valid:
                instance = model.create(identifier=valid_id)
                assert instance.validate()
                instance.delete_instance()

            for invalid_id in invalid:
                with pytest.raises(ValueError):
                    model.create(identifier=invalid_id)

