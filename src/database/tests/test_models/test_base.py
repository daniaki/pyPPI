import mock

from .. import DatabaseTestMixin

from ...models.identifiers import UniprotIdentifier


class TestBaseModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.instance = UniprotIdentifier.create(identifier="P12345")

    def test_none_returns_no_rows(self):
        assert UniprotIdentifier.none().count() == 0

    def test_all_returns_all_rows(self):
        assert UniprotIdentifier.all().count() == 1

    def test_count_returns_all_row_count(self):
        assert UniprotIdentifier.count() == 1

    def test_updates_modified_date_on_save(self):
        now = self.instance.modified
        self.instance.save()
        later = self.instance.modified
        assert later > now

    def test_refresh_refreshes_fields_from_db(self):
        self.instance.identifier = "P12346"
        assert self.instance.is_dirty()
        self.instance = self.instance.refresh()
        assert not self.instance.is_dirty()

    def test_calls_format_for_save_on_child_class(self):
        instance = UniprotIdentifier(identifier="p12346")
        instance.format_for_save = mock.MagicMock(
            side_effect=instance.format_for_save
        )
        # Check if called but also call as a side_effect to avoid
        # integrity errors.
        instance.save()
        instance.format_for_save.assert_called()
        assert str(instance) == "P12346"
