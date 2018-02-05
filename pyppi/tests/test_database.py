
import os
import shutil

from unittest import TestCase
from Bio import SwissProt

from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm.exc import DetachedInstanceError

from ..database import begin_transaction, delete_database, make_session
from ..database.initialisation import (
    add_interactions_to_database, init_protein_table,
)
from ..database.models import (
    Protein, Interaction, Psimi, Pubmed,
    psimi_interactions, pmid_interactions
)
from ..data_mining.uniprot import parse_record_into_protein

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


class TestInitialisationMethods(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.records = open(os.path.normpath(
            "{}/test_data/test_sprot_records.dat".format(base_path)
        ), 'rt')

    def tearDown(self):
        with begin_transaction(db_path=self.db_path) as session:
            session.rollback()
            session.execute("DROP TABLE {}".format(Protein.__tablename__))
            session.execute("DROP TABLE {}".format(Interaction.__tablename__))
        self.records.close()

    def test_can_init_protein_database_from_file(self):
        init_protein_table(
            record_handle=self.records, db_path=self.db_path
        )
        with begin_transaction(self.db_path) as session:
            self.assertEqual(session.query(Protein).count(), 3)

    def test_can_init_interaction_database(self):
        records = list(SwissProt.parse(self.records))
        proteins = [parse_record_into_protein(r) for r in records]
        with begin_transaction(db_path=self.db_path) as session:
            for protein in proteins:
                protein.save(session, commit=True)
            proteins = [session.query(Protein).get(p.id) for p in proteins]
            interactions = [
                (
                    proteins[0], proteins[1],
                    False, False, False, None
                ),
                (
                    proteins[1], proteins[1],
                    True, False, False, 'Activation'
                )
            ]
            valid, invalid = add_interactions_to_database(
                session, interactions, n_jobs=2
            )
            self.assertEqual(session.query(Interaction).count(), 2)
            self.assertEqual(len(valid), 2)
            self.assertEqual(len(invalid), 0)

    def test_add_interactions_will_not_add_invalid_interactions(self):
        records = list(SwissProt.parse(self.records))
        proteins = [parse_record_into_protein(r) for r in records]
        with begin_transaction(db_path=self.db_path) as session:
            for protein in proteins:
                protein.save(session, commit=True)
            proteins = [session.query(Protein).get(p.id) for p in proteins]
            interactions = [
                (
                    proteins[0], proteins[1],
                    False, False, False, None
                ),
                (
                    # reversed should cause a collision
                    proteins[1], proteins[0],
                    True, False, False, 'Activation'
                )
            ]
            valid, invalid = add_interactions_to_database(
                session, interactions, n_jobs=1
            )
            self.assertEqual(session.query(Interaction).count(), 1)
            self.assertEqual(len(valid), 1)
            self.assertEqual(len(invalid), 1)

    def test_add_interactions_will_not_add_nonmatching_taxon_interaction(self):
        records = list(SwissProt.parse(self.records))
        proteins = [parse_record_into_protein(r) for r in records]
        with begin_transaction(db_path=self.db_path) as session:
            for protein in proteins:
                protein.save(session, commit=True)
            proteins = [session.query(Protein).get(p.id) for p in proteins]
            interactions = [
                (proteins[0], proteins[1], False, False, False, None),
                (proteins[1], proteins[1], True, False, False, 'Activation')
            ]
            valid, invalid = add_interactions_to_database(
                session, interactions, n_jobs=1, match_taxon_id=0
            )
            self.assertEqual(session.query(Interaction).count(), 0)
            self.assertEqual(len(valid), 0)
            self.assertEqual(len(invalid), 2)

    def test_add_interactions_ignores_taxonid_matching_when_none(self):
        records = list(SwissProt.parse(self.records))
        proteins = [parse_record_into_protein(r) for r in records]
        with begin_transaction(db_path=self.db_path) as session:
            for protein in proteins:
                protein.save(session, commit=True)
            proteins = [session.query(Protein).get(p.id) for p in proteins]
            interactions = [
                (proteins[0], proteins[1], False, False, False, None),
                (proteins[1], proteins[1], True, False, False, 'Activation')
            ]
            valid, invalid = add_interactions_to_database(
                session, interactions, n_jobs=1, match_taxon_id=None
            )
            self.assertEqual(session.query(Interaction).count(), 2)
            self.assertEqual(len(valid), 2)
            self.assertEqual(len(invalid), 0)
