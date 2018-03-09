
import os
import shutil
import pandas as pd

from collections import OrderedDict
from unittest import TestCase
from Bio import SwissProt

from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm.exc import DetachedInstanceError

from ..base.constants import (
    SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE, NULL_VALUES
)

from ..database import create_session, delete_database, cleanup_database
from ..database.exceptions import ObjectAlreadyExists, ObjectNotFound
from ..database.utilities import (
    filter_matching_taxon_ids,
    training_interactions,
    holdout_interactions,
    full_training_network,
    interactome_interactions,
    labels_from_interactions,
    get_upid_to_protein_map,
    get_source_taget_to_interactions_map,
    create_interaction
)
from ..database.models import (
    Protein, Interaction, Psimi, Pubmed
)

base_path = os.path.dirname(__file__)


class TestCreateInteractions(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.pmid_a = Pubmed(accession='A')
        self.pmid_b = Pubmed(accession='B')
        self.psmi_a = Psimi(accession='A', description='hello')
        self.psmi_b = Psimi(accession='B', description='world')

        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pmid_a.save(self.session, commit=True)
        self.pmid_b.save(self.session, commit=True)
        self.psmi_a.save(self.session, commit=True)
        self.psmi_b.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_integration_of_create_interaction(self):
        entry = create_interaction(
            source=self.pa,
            target=self.pb,
            labels=['Activation', 'activation', None, ' '],
            **{
                "is_interactome": False,
                "is_training": True,
                "is_holdout": None,
                'go_mf': "GO:00, , ,go:00",
                'go_bp': ['GO:02', ' ', 'GO:01'],
                'ulca_go_mf': "GO:01,GO:00,GO:02",
                'ulca_go_bp': [],
                'interpro': "IPR1,IPR1",
                'pfam': ['pf1'],
                'keywords': "protein,activator"
            }
        )
        self.assertEquals(entry.label, 'Activation')
        self.assertEquals(entry.go_mf, "GO:00,GO:00")
        self.assertEquals(entry.go_bp, "GO:01,GO:02")
        self.assertEquals(entry.go_cc, None)
        self.assertEquals(entry.ulca_go_mf, "GO:00,GO:01,GO:02")
        self.assertEquals(entry.ulca_go_bp, None)
        self.assertEquals(entry.interpro, "IPR1,IPR1")
        self.assertEquals(entry.pfam, "PF1")
        self.assertEquals(entry.keywords, "Activator,Protein")
        self.assertFalse(entry.is_interactome)
        self.assertTrue(entry.is_training)
        self.assertFalse(entry.is_holdout)
        self.assertEquals(entry.pmids(), None)
        self.assertEquals(entry.experiment_types(), None)

    def test_error_if_interaction_exists(self):
        ia = Interaction(self.pa, self.pb)
        ia.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            create_interaction(source=self.pa, target=self.pb)


class TestFilterMatchingTaxonId(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_removes_non_matching_taxonids(self):
        qs = filter_matching_taxon_ids(Protein.query, taxon_id=9606)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), self.pa)

    def test_ignores_taxonid_if_none(self):
        qs = filter_matching_taxon_ids(Protein.query, taxon_id=None)
        self.assertEqual(qs.count(), 2)


class TestTrainingInteractions(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_returns_all_training_matching_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_training=True)
        obj2 = Interaction(self.pa, self.pc, 'Activation', is_training=False)
        obj3 = Interaction(self.pb, self.pb, 'Activation', is_training=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        qs = training_interactions(taxon_id=9606)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), obj1)

    def test_strict_mode_removes_is_holdout(self):
        obj1 = Interaction(
            self.pa, self.pa, 'Activation',
            is_holdout=False, is_training=True
        )
        obj2 = Interaction(
            self.pc, self.pc, 'Activation',
            is_holdout=True, is_training=True
        )
        obj1.save(self.session, True)
        obj2.save(self.session, True)

        qs = training_interactions(strict=True, taxon_id=9606)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), obj1)

    def test_ignore_none_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_training=True)
        obj2 = Interaction(self.pa, self.pc, 'Activation', is_training=False)
        obj3 = Interaction(self.pb, self.pb, 'Activation', is_training=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        qs = training_interactions(taxon_id=None)
        self.assertEqual(qs.count(), 2)


class TestHoldoutInteractions(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_returns_all_holdout_matching_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_holdout=True)
        obj2 = Interaction(self.pa, self.pc, 'Activation', is_training=False)
        obj3 = Interaction(self.pb, self.pb, 'Activation', is_holdout=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        qs = holdout_interactions(taxon_id=9606)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), obj1)

    def test_strict_mode_removes_is_training(self):
        obj1 = Interaction(
            self.pa, self.pa, 'Activation',
            is_holdout=True, is_training=False
        )
        obj2 = Interaction(
            self.pc, self.pc, 'Activation',
            is_holdout=True, is_training=True
        )
        obj1.save(self.session, True)
        obj2.save(self.session, True)

        qs = holdout_interactions(strict=True, taxon_id=9606)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), obj1)

    def test_ignore_none_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_holdout=True)
        obj2 = Interaction(self.pa, self.pc, 'Activation', is_training=False)
        obj3 = Interaction(self.pb, self.pb, 'Activation', is_holdout=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        qs = holdout_interactions(taxon_id=None)
        self.assertEqual(qs.count(), 2)


class TestInteractomeInteractions(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_returns_all_holdout_matching_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, is_interactome=True)
        obj2 = Interaction(self.pa, self.pc, is_interactome=False)
        obj3 = Interaction(self.pb, self.pb, is_interactome=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        qs = interactome_interactions(taxon_id=9606)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first(), obj1)

    def test_ignore_none_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, is_interactome=True)
        obj2 = Interaction(self.pa, self.pc, is_interactome=False)
        obj3 = Interaction(self.pb, self.pb, is_interactome=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        qs = interactome_interactions(taxon_id=None)
        self.assertEqual(qs.count(), 2)


class TestFullTrainingNetwork(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pd = Protein(uniprot_id="D", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)
        self.pd.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_returns_all_holdout_and_training_matching_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_holdout=True)
        obj2 = Interaction(
            self.pa, self.pc, 'Activation', is_training=True, is_holdout=False)
        obj3 = Interaction(
            self.pc, self.pc, 'Activation', is_training=True, is_holdout=True)
        obj4 = Interaction(
            self.pc, self.pd, is_training=False, is_holdout=False)
        obj5 = Interaction(self.pb, self.pb, 'Activation', is_training=True)

        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)
        obj4.save(self.session, True)
        obj5.save(self.session, True)

        qs = full_training_network(taxon_id=9606)
        self.assertEqual(qs.count(), 3)
        self.assertEqual(qs.all(), [obj1, obj2, obj3])

    def test_ignore_none_taxonid(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_holdout=True)
        obj2 = Interaction(
            self.pa, self.pc, 'Activation', is_training=True, is_holdout=False)
        obj3 = Interaction(
            self.pc, self.pc, 'Activation', is_training=True, is_holdout=True)
        obj4 = Interaction(
            self.pc, self.pd, is_training=False, is_holdout=False)
        obj5 = Interaction(self.pb, self.pb, 'Activation', is_training=True)

        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)
        obj4.save(self.session, True)
        obj5.save(self.session, True)

        qs = full_training_network(taxon_id=None)
        self.assertEqual(qs.count(), 4)


class TestLabelsFromInteractions(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_returns_sorted(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_training=True)
        obj2 = Interaction(self.pc, self.pc, 'Inhibition', is_training=True)
        obj3 = Interaction(self.pa, self.pc, 'Acetylation', is_holdout=True)
        obj4 = Interaction(self.pb, self.pb, 'Activation', is_training=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)
        obj4.save(self.session, True)

        labels = labels_from_interactions([obj1, obj2, obj3, obj4])
        self.assertEqual(labels, ['Acetylation', 'Activation', 'Inhibition'])

    def test_returns_all_training_holdout_if_interactions_is_none(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_training=True)
        obj2 = Interaction(self.pc, self.pc, 'Inhibition', is_training=True)
        obj3 = Interaction(self.pa, self.pc, 'Acetylation', is_holdout=True)
        obj4 = Interaction(self.pb, self.pb, 'Binding')
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)
        obj4.save(self.session, True)

        labels = labels_from_interactions()
        self.assertEqual(labels, ['Acetylation', 'Activation', 'Inhibition'])

    def test_ignore_filters_taxon_id(self):
        obj1 = Interaction(self.pa, self.pa, 'Activation', is_training=True)
        obj2 = Interaction(self.pc, self.pc, 'Inhibition', is_training=True)
        obj3 = Interaction(self.pb, self.pb, 'Binding', is_training=True)
        obj1.save(self.session, True)
        obj2.save(self.session, True)
        obj3.save(self.session, True)

        labels = labels_from_interactions(taxon_id=9606)
        self.assertEqual(labels, ['Activation', 'Inhibition'])


class TestGetUpidProteinMap(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_map_contains_valid_uniprot_ids(self):
        mapping = get_upid_to_protein_map(['A', 'B', 'C'], taxon_id=9606)
        expected = {'A': self.pa, 'B': None, 'C': self.pc}
        self.assertEqual(mapping, expected)

    def test_ignores_taxon_id_if_none(self):
        mapping = get_upid_to_protein_map(['A', 'B', 'C'], taxon_id=None)
        expected = {'A': self.pa, 'B': self.pb, 'C': self.pc}
        for key, value in mapping.items():
            self.assertEqual(expected[key], value)

    def test_invalid_uniprot_id_default_to_none(self):
        mapping = get_upid_to_protein_map(['A', 'D'], taxon_id=9606)
        expected = {'A': self.pa, 'D': None}
        self.assertEqual(mapping, expected)


class TestSTInteractionMap(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pb = Protein(uniprot_id="B", taxon_id=0, reviewed=False)
        self.pc = Protein(uniprot_id="C", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)
        self.pc.save(self.session, commit=True)

        self.ia = Interaction(self.pa, self.pa)
        self.ib = Interaction(self.pb, self.pb)
        self.ic = Interaction(self.pa, self.pc)
        self.ia.save(self.session, commit=True)
        self.ib.save(self.session, commit=True)
        self.ic.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_map_contains_valid_st_tuples(self):
        mapping = get_source_taget_to_interactions_map(
            [(1, 1), (1, 3)], taxon_id=9606
        )
        expected = OrderedDict()
        expected[(1, 1)] = self.ia
        expected[(1, 3)] = self.ic
        self.assertEqual(mapping, expected)

    def test_maintains_order_of_input(self):
        mapping = get_source_taget_to_interactions_map(
            [(2, 3), (1, 1)], taxon_id=9606
        )
        expected = OrderedDict()
        expected[(2, 3)] = None
        expected[(1, 1)] = self.ia
        self.assertEqual(mapping, expected)

    def test_ignores_taxon_id_if_none(self):
        mapping = get_source_taget_to_interactions_map(
            [(1, 1), (1, 3), (2, 2)], taxon_id=None
        )
        expected = OrderedDict()
        expected[(1, 1)] = self.ia
        expected[(1, 3)] = self.ic
        expected[(2, 2)] = self.ib
        self.assertEqual(mapping, expected)

    def test_invalid_uniprot_id_default_to_none(self):
        mapping = get_source_taget_to_interactions_map(
            [(1, 1), (2, 3)], taxon_id=9606
        )
        expected = OrderedDict()
        expected[(1, 1)] = self.ia
        expected[(2, 3)] = None
        self.assertEqual(mapping, expected)
