
import os

from unittest import TestCase
from Bio import SwissProt

from ..database import begin_transaction
from ..database.initialisation import (
    add_interactions_to_database, init_protein_table
)
from ..database.models import Protein, Interaction
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
            session.execute("DROP TABLE {}".format(Protein.__tablename__))

    def test_can_connect(self):
        with begin_transaction(db_path=self.db_path) as session:
            self.assertTrue(session.is_active, True)

    def test_can_commit_changes(self):
        obj = Protein(uniprot_id="abc", taxon_id=9606, reviewed=False)
        with begin_transaction(db_path=self.db_path) as session:
            obj.save(session, commit=True)

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
