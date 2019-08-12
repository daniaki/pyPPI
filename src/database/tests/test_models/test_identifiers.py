from .. import DatabaseTestMixin

import pytest

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
