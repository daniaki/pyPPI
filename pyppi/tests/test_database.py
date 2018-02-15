
import os
import shutil

from unittest import TestCase
from Bio import SwissProt

from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm.exc import DetachedInstanceError

from ..database import begin_transaction, delete_database, make_session
from ..database.models import (
    Protein, Interaction, Psimi, Pubmed,
    psimi_interactions, pmid_interactions
)

base_path = os.path.dirname(__file__)


class TestContextManager(TestCase):

    # Function signature: begin_transaction(db_path=None, echo=False)

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )

    def tearDown(self):
        with begin_transaction(db_path=self.db_path) as session:
            session.rollback()
            session.query(Protein).delete()
            session.query(Interaction).delete()
            session.query(Pubmed).delete()
            session.query(Psimi).delete()
            session.query(psimi_interactions).delete(
                synchronize_session=False)
            session.query(pmid_interactions).delete(
                synchronize_session=False)
            session.commit()

    def test_can_connect(self):
        with begin_transaction(db_path=self.db_path) as session:
            self.assertTrue(session.is_active, True)

    def test_make_session_creates_tables(self):
        session = make_session(db_path=self.db_path)
        os.remove(self.db_path)  # remove database completely
        session.close()

        session = make_session(db_path=self.db_path)
        inspector = Inspector.from_engine(session.get_bind())
        self.assertIn("protein", inspector.get_table_names())
        self.assertIn("interaction", inspector.get_table_names())
        self.assertIn("pubmed", inspector.get_table_names())
        self.assertIn("psimi", inspector.get_table_names())
        self.assertIn("pmid_interactions", inspector.get_table_names())
        self.assertIn("psimi_interactions", inspector.get_table_names())
        session.close()

    def test_begin_session_creates_tables(self):
        with begin_transaction(db_path=self.db_path) as session:
            os.remove(self.db_path)  # remove database completely

        with begin_transaction(db_path=self.db_path) as session:
            inspector = Inspector.from_engine(session.get_bind())
            self.assertIn("protein", inspector.get_table_names())
            self.assertIn("interaction", inspector.get_table_names())
            self.assertIn("pubmed", inspector.get_table_names())
            self.assertIn("psimi", inspector.get_table_names())
            self.assertIn("pmid_interactions", inspector.get_table_names())
            self.assertIn("psimi_interactions", inspector.get_table_names())

    def test_can_commit_changes(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        with begin_transaction(db_path=self.db_path) as session:
            obj.save(session, commit=True)

    def test_error_when_accessing_refreshed_field_from_expired_session(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        with begin_transaction(db_path=self.db_path) as session:
            obj.save(session, commit=True)
        with self.assertRaises(DetachedInstanceError):
            obj = Interaction(
                source=obj.id, target=obj.id,
                is_training=False, is_holdout=False, label=None,
                is_interactome=False
            )

    def test_can_delete_database_with_db_path(self):
        with begin_transaction(db_path=self.db_path) as session:
            obj1 = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
            obj1.save(session, commit=True)
            obj2 = Interaction(
                source=obj1.id, target=obj1.id,
                is_interactome=True, is_holdout=False, is_training=False
            )
            obj2.save(session, commit=True)
            obj3 = Pubmed(accession='A')
            obj4 = Psimi(accession='A', description='hello')
            obj3.save(session, commit=True)
            obj4.save(session, commit=True)

            obj2.add_pmid_reference(obj3)
            obj2.add_psimi_reference(obj4)
            obj2.save(session, commit=True)

            tables = [
                Protein, Interaction, Pubmed, Psimi,
                psimi_interactions, pmid_interactions
            ]
            for table in tables:
                self.assertEqual(session.query(table).count(), 1)

        with begin_transaction(db_path=self.db_path) as session:
            delete_database(db_path=self.db_path)
            tables = [
                Protein, Interaction, Pubmed, Psimi,
                psimi_interactions, pmid_interactions
            ]
            for table in tables:
                self.assertEqual(session.query(table).count(), 0)

    def test_can_delete_database_with_session(self):
        with begin_transaction(db_path=self.db_path) as session:
            obj1 = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
            obj1.save(session, commit=True)
            obj2 = Interaction(
                source=obj1.id, target=obj1.id,
                is_interactome=True, is_holdout=False, is_training=False
            )
            obj2.save(session, commit=True)
            obj3 = Pubmed(accession='A')
            obj4 = Psimi(accession='A', description='hello')
            obj3.save(session, commit=True)
            obj4.save(session, commit=True)

            obj2.add_pmid_reference(obj3)
            obj2.add_psimi_reference(obj4)
            obj2.save(session, commit=True)

            tables = [
                Protein, Interaction, Pubmed, Psimi,
                psimi_interactions, pmid_interactions
            ]
            for table in tables:
                self.assertEqual(session.query(table).count(), 1)

        with begin_transaction(db_path=self.db_path) as session:
            delete_database(session=session)
            tables = [
                Protein, Interaction, Pubmed, Psimi,
                psimi_interactions, pmid_interactions
            ]
            for table in tables:
                self.assertEqual(session.query(table).count(), 0)

    def test_can_rollback_uncommited_changes(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        with begin_transaction(db_path=self.db_path) as session:
            obj.save(session, commit=False)
            session.rollback()
            session.commit()
            self.assertEqual(session.query(Protein).count(), 0)

    def test_can_commit_after_rollback_without_failing_unique_constraint(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        with begin_transaction(db_path=self.db_path) as session:
            obj.save(session, commit=False)
            session.rollback()
            self.assertEqual(session.query(Protein).count(), 0)
            obj.save(session, commit=True)
            self.assertEqual(session.query(Protein).count(), 1)

    def test_session_commits_on_exit(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        with begin_transaction(db_path=self.db_path) as session:
            obj.save(session, commit=False)

        with begin_transaction(db_path=self.db_path) as session:
            self.assertEqual(session.query(Protein).count(), 1)

    def test_session_rollsback_error_on_exit(self):
        obj = Protein(uniprot_id=None, taxon_id=9606, reviewed=False)
        try:
            with begin_transaction(db_path=self.db_path) as session:
                obj.save(session, commit=False)
        except:
            self.assertEqual(session.query(Protein).count(), 0)

    def test_raise_error_not_a_valid_database_path(self):
        with self.assertRaises(Exception):
            with begin_transaction(db_path="/not/a/path/") as session:
                pass
