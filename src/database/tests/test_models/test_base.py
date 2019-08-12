from .. import DatabaseTestMixin

from ...models.identifiers import UniprotIdentifier


class TestBaseModel(DatabaseTestMixin):
    def setup(self):
        super().setup()
        self.instance = UniprotIdentifier.create(identifier="A")

    def test_updates_modified_date_on_save(self):
        now = self.instance.modified
        self.instance.save()
        later = self.instance.modified
        assert later > now

    def test_refresh_refreshes_fields_from_db(self):
        self.instance.identifier = "B"
        assert self.instance.is_dirty()
        self.instance = self.instance.refresh()
        assert not self.instance.is_dirty()
