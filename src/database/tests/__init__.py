import peewee

from .. import models


__all__ = ["test_db", "DatabaseTestMixin", "test_utilities", "test_models"]


# use an in-memory SQLite for tests.
test_db = peewee.SqliteDatabase(":memory:")


class DatabaseTestMixin:
    def setup(self):
        # Bind model classes to test db.
        test_db.bind(models.MODELS, bind_refs=True, bind_backrefs=True)
        test_db.connect()
        test_db.create_tables(models.MODELS)

    def teardown(self):
        # Not strictly necessary since SQLite in-memory databases only live
        # for the duration of the connection, and in the next step we close
        # the connection...but a good practice all the same.
        test_db.drop_tables(models.MODELS)

        # Close connection to db.
        test_db.close()

        # If we wanted, we could re-bind the models to their original
        # database here. But for tests this is probably not necessary.
