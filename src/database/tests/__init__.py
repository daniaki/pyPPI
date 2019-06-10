import peewee

from .. import models


__all__ = ["DatabaseTestMixin", "test_utilities", "test_models"]


class DatabaseTestMixin:
    def setup(self):
        # use an in-memory SQLite for tests.
        self.test_db = peewee.SqliteDatabase(":memory:")

        # Bind model classes to test db.
        self.test_db.bind(models.MODELS, bind_refs=True, bind_backrefs=True)
        self.test_db.connect()
        self.test_db.create_tables(models.MODELS)

    def teardown(self):
        # Not strictly necessary since SQLite in-memory databases only live
        # for the duration of the connection, and in the next step we close
        # the connection...but a good practice all the same.
        self.test_db.drop_tables(models.MODELS)

        # Close connection to db.
        self.test_db.close()

        # If we wanted, we could re-bind the models to their original
        # database here. But for tests this is probably not necessary.
