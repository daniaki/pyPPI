
import os
import shutil

from unittest import TestCase
from Bio import SwissProt

from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm.exc import DetachedInstanceError

from ..database import create_session, delete_database
from ..database.models import (
    Protein, Interaction, Psimi, Pubmed, Reference
)

base_path = os.path.dirname(__file__)


class TestContextManager(TestCase):

    # Function signature: begin_transaction(db_path=None, echo=False)

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)

    def tearDown(self):
        delete_database(self.session)
        self.session.remove()
        self.session.close_all()
        self.engine.dispose()

    def test_create_session_creates_tables(self):
        from ..database import Base
        os.remove(self.db_path)  # remove database completely
        session, engine = create_session(self.db_path)

        tables = Base.metadata.tables.keys()
        self.assertIn("protein", tables)
        self.assertIn("interaction", tables)
        self.assertIn("pubmed", tables)
        self.assertIn("psimi", tables)
        self.assertIn("reference", tables)

        session.remove()
        session.close_all()
        engine.dispose()

    def test_can_commit_changes(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        obj.save(self.session, commit=True)
        self.assertEqual(Protein.query.count(), 1)

    def test_can_delete_database_with_session(self):
        obj1 = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        obj1.save(self.session, commit=True)
        obj2 = Interaction(
            source=obj1.id, target=obj1.id,
            is_interactome=True, is_holdout=False, is_training=False
        )
        obj2.save(self.session, commit=True)
        obj3 = Pubmed(accession='A')
        obj4 = Psimi(accession='A', description='hello')
        obj3.save(self.session, commit=True)
        obj4.save(self.session, commit=True)

        r1 = Reference(obj2, obj3, obj4)
        r1.save(self.session, commit=True)

        tables = [
            Protein, Interaction, Pubmed, Psimi, Reference
        ]
        for table in tables:
            self.assertEqual(table.query.count(), 1)

        delete_database(session=self.session)
        tables = [
            Protein, Interaction, Pubmed, Psimi, Reference
        ]
        for table in tables:
            self.assertEqual(table.query.count(), 0)

    def test_can_rollback_uncommited_changes(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        obj.save(self.session, commit=False)
        self.session.rollback()
        self.session.commit()
        self.assertEqual(Protein.query.count(), 0)

    def test_can_commit_after_rollback_without_failing_unique_constraint(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        obj.save(self.session, commit=False)
        self.session.rollback()
        self.assertEqual(Protein.query.count(), 0)
        obj.save(self.session, commit=True)
        self.assertEqual(Protein.query.count(), 1)

    def test_objects_persist_on_commit(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        obj.save(self.session, commit=False)
        self.assertEqual(Protein.query.count(), 0)
        self.session.commit()
        self.assertEqual(Protein.query.count(), 1)

    def test_saving_same_object_twice_doesnt_raise_error(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        obj.save(self.session, commit=True)

        self.session.commit()
        self.assertEqual(Protein.query.count(), 1)

        obj.save(self.session, commit=False)
        self.session.commit()
