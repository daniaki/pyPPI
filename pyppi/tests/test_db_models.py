#!/usr/bin/env python

import os
import shutil
from datetime import datetime

from unittest import TestCase
from sqlalchemy.exc import IntegrityError

from ..database import create_session, Base, delete_database
from ..database import cleanup_database
from ..database.models import Protein, Interaction, Psimi, Pubmed, Reference
from ..database.exceptions import ObjectNotFound, ObjectAlreadyExists
from ..database.exceptions import NonMatchingTaxonomyIds

base_path = os.path.dirname(__file__)


class TestProteinModel(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_get_by_uniprot_id(self):
        protein_1 = Protein("A", taxon_id=9606, reviewed=False)
        protein_1.save(self.session, commit=True)
        self.assertEqual(Protein.get_by_uniprot_id('B'), None)
        self.assertEqual(Protein.get_by_uniprot_id('a'), protein_1)

    def test_equality(self):
        protein_1 = Protein("A", taxon_id=9606, reviewed=False)
        protein_1.save(self.session, commit=True)
        protein_2 = Protein.query.get(protein_1.id)
        self.assertEqual(protein_1, protein_2)

    def test_inequality(self):
        protein_1 = Protein("A", taxon_id=9606, reviewed=False)
        protein_1.save(self.session, commit=True)
        protein_2 = Protein("B", taxon_id=0, reviewed=False)
        protein_2.save(self.session, commit=True)
        self.assertNotEqual(protein_1, protein_2)

    def test_error_uniprot_id_already_exists(self):
        obj1 = Protein(uniprot_id="abc", taxon_id=1, reviewed=False)
        obj1.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            obj2 = Protein(uniprot_id="abc", taxon_id=1, reviewed=False)

    def test_typeerr_adding_null_unirpotid_field(self):
        with self.assertRaises(TypeError):
            obj1 = Protein(uniprot_id=None, taxon_id=1, reviewed=False)

    def test_datetime_correctly_initalised(self):
        obj = Protein(
            uniprot_id='A', taxon_id=1, last_update='28-FEB-2018'
        )
        expected = datetime.strptime('28-FEB-2018', '%d-%b-%Y')
        self.assertEqual(obj.last_update, expected)

    def test_annotations_outdated_based_on_datetime(self):
        obj = Protein(
            uniprot_id='A', taxon_id=1, last_update='28-FEB-2018'
        )
        new = datetime.strptime('1-JUN-2018', '%d-%b-%Y')
        self.assertTrue(obj.annotations_outdated(new))

        same = datetime.strptime('28-FEB-2018', '%d-%b-%Y')
        self.assertFalse(obj.annotations_outdated(same))
        old = datetime.strptime('1-JUN-2017', '%d-%b-%Y')
        self.assertFalse(obj.annotations_outdated(old))

        # Keep this last.
        obj.last_update = None
        self.assertTrue(obj.annotations_outdated(new))

    def test_release_outdated(self):
        obj = Protein(
            uniprot_id='A', taxon_id=1, last_release=56
        )
        self.assertTrue(obj.release_outdated(57))
        self.assertFalse(obj.release_outdated(56))
        self.assertFalse(obj.release_outdated(5))

    def test_integration_of_validators(self):
        obj = Protein(
            uniprot_id='abc',
            gene_id='gene1',
            taxon_id=9606,
            reviewed=None,
            go_mf='go:001,go:001,GO:001',
            go_cc=[None, ' '],
            go_bp=['go:002', 'go:001'],
            interpro='ipr1,',
            pfam=[],
            keywords='should be captialised,comma split',
            function='    '
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.uniprot_id, "ABC")
        self.assertEqual(obj.gene_id, "gene1")
        self.assertEqual(obj.taxon_id, 9606)
        self.assertEqual(obj.reviewed, False)
        self.assertEqual(obj.go_mf, 'GO:001')
        self.assertEqual(obj.go_cc, None)
        self.assertEqual(obj.go_bp, 'GO:001,GO:002')
        self.assertEqual(obj.interpro, 'IPR1')
        self.assertEqual(obj.pfam, None)
        self.assertEqual(obj.keywords, 'Comma split,Should be captialised')
        self.assertEqual(obj.function, None)


class TestInteractionModel(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.a = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.b = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.c = Protein(uniprot_id="C", taxon_id=0, reviewed=False)
        self.a.save(self.session, commit=True)
        self.b.save(self.session, commit=True)
        self.c.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_integration_of_validators(self):
        obj = Interaction(
            source=2, target='A',
            label=['Inhibition', None, ' ', 'activation', 'ACTIVATION  '],
            go_mf='go:001,go:001,GO:001',
            go_cc=[None, ' '],
            go_bp=['go:002', 'go:001'],
            ulca_go_mf=['go:001 ', None, 'go:001', 'GO:001'],
            ulca_go_cc=[None, ' '],
            ulca_go_bp='  ',
            interpro='ipr1,',
            pfam=[],
            keywords='dog,,cat,cat ',
            is_holdout=True, is_interactome=False, is_training=False
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.source, 2)
        self.assertEqual(obj.target, 1)
        self.assertEqual(obj.taxon_id, 9606)
        self.assertEqual(obj.joint_id, '1,2')
        self.assertEqual(obj.label, 'Activation,Inhibition')
        self.assertEqual(obj.is_holdout, True)
        self.assertEqual(obj.is_interactome, False)
        self.assertEqual(obj.is_training, False)
        self.assertEqual(obj.go_mf, 'GO:001,GO:001,GO:001')
        self.assertEqual(obj.go_cc, None)
        self.assertEqual(obj.go_bp, 'GO:001,GO:002')
        self.assertEqual(obj.ulca_go_mf, 'GO:001,GO:001,GO:001')
        self.assertEqual(obj.ulca_go_cc, None)
        self.assertEqual(obj.ulca_go_bp, None)
        self.assertEqual(obj.interpro, 'IPR1')
        self.assertEqual(obj.pfam, None)
        self.assertEqual(obj.keywords, 'Cat,Cat,Dog')

    def test_can_add_new_label(self):
        obj = Interaction(source=self.a, target=self.b)
        obj.add_label('activation')
        self.assertEqual(obj.label, 'Activation')

    def test_labels_as_list_is_sorted(self):
        obj = Interaction(
            source=self.a,
            target=self.b,
            label='inhibition,activation'
        )
        self.assertEqual(obj.labels_as_list, ['Activation', 'Inhibition'])

    def test_labels_as_list_empty_if_label_is_none(self):
        obj = Interaction(source=self.a, target=self.b, label=None)
        self.assertEqual(obj.labels_as_list, [])

    def test_can_add_to_existing_label(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.add_label('phosphorylation')
        self.assertEqual(obj.label, 'Activation,Phosphorylation')

    def test_can_add_an_existing_label_without_duplicating(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.add_label('activation')
        self.assertEqual(obj.label, 'Activation')

    def test_setting_empty_string_does_nothing(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.add_label('')
        self.assertEqual(obj.label, 'Activation')

    def test_can_set_multilabel(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.add_label('activation,phosphorylation')
        self.assertEqual(obj.label, 'Activation,Phosphorylation')

    def test_can_remove_existing_label(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.remove_label('activation')
        self.assertEqual(obj.label, None)

    def test_can_remove_existing_labels_list(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            label='activation,inhibition'
        )
        obj.remove_label(['activation', 'inhibition'])
        self.assertEqual(obj.label, None)

    def test_can_remove_existing_label_not_case_sensitive(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.remove_label('ACTIVATION')
        self.assertEqual(obj.label, None)

    def test_remove_label_does_nothing_if_label_not_found(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.remove_label('hello')
        self.assertEqual(obj.label, 'Activation')

    def test_does_nothing_removing_empty_string_or_none(self):
        obj = Interaction(source=self.a, target=self.b, label='activation')
        obj.remove_label('')
        self.assertEqual(obj.label, 'Activation')
        obj.remove_label(None)
        self.assertEqual(obj.label, 'Activation')

    def test_interaction_equality(self):
        obj1 = Interaction(source=self.b, target=self.a)
        obj2 = Interaction(source=self.b, target=self.a)
        self.assertEqual(obj1, obj2)

    def test_interaction_equal_reversed_source_target(self):
        obj1 = Interaction(source=self.a, target=self.b)
        obj2 = Interaction(source=self.b, target=self.a)
        self.assertEqual(obj1, obj2)

    def test_interaction_inequality(self):
        obj1 = Interaction(source=self.b, target=self.a)
        obj2 = Interaction(source=self.b, target=self.a, label='activation')
        self.assertNotEqual(obj1, obj2)

    def test_objectexists_error_when_switching_source_target(self):
        obj1 = Interaction(source=self.a, target=self.b)
        obj1.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            obj2 = Interaction(source=self.b, target=self.a)

    def test_typeerr_null_source_or_target(self):
        with self.assertRaises(TypeError):
            obj = Interaction(source=None, target=self.b)
        with self.assertRaises(TypeError):
            obj = Interaction(source=self.a, target=None)

    def test_attrerr_trying_to_change_source_or_target(self):
        with self.assertRaises(AttributeError):
            obj = Interaction(source=self.a, target=self.b)
            obj.source = 'B'
        with self.assertRaises(AttributeError):
            obj = Interaction(source=self.a, target=self.b)
            obj.target = 'A'

    def test_constructor_accepts_uniprot_ids(self):
        obj = Interaction(source='A', target='B')
        self.assertEqual(obj.source, 1)
        self.assertEqual(obj.target, 2)

    def test_boolean_is_fields_default_to_False(self):
        obj = Interaction(source=self.a, target=self.b)
        obj.save(self.session, commit=True)
        self.assertEqual(obj.is_holdout, False)
        self.assertEqual(obj.is_interactome, False)
        self.assertEqual(obj.is_training, False)

    def test_valueerror_different_taxonomy_ids(self):
        with self.assertRaises(NonMatchingTaxonomyIds):
            obj = Interaction(source=self.a, target=self.c)

    def test_interaction_assumes_source_taxonid(self):
        obj = Interaction(source=self.a, target=self.b)
        obj.save(self.session, commit=True)
        self.assertEqual(obj.taxon_id, self.a.taxon_id)

    def test_create_valueerror_training_must_be_labeled(self):
        with self.assertRaises(ValueError):
            obj = Interaction(source=self.a, target=self.a)
            obj.is_training = True

        with self.assertRaises(ValueError):
            obj = Interaction(source=self.a, target=self.a)
            obj.is_holdout = True

        with self.assertRaises(ValueError):
            obj = Interaction(
                source=self.a, target=self.a, is_training=True,
                label='activation'
            )
            obj.label = None

    def test_can_add_reference(self):
        obj = Interaction(source=self.a, target=self.a)
        obj.save(self.session, commit=True)

        pm1 = Pubmed('A')
        pm1.save(self.session, commit=True)
        ps1 = Psimi('A')
        ps1.save(self.session, commit=True)
        r1 = obj.add_reference(self.session, pm1, ps1, commit=True)

        self.assertEqual(Reference.query.count(), 1)
        self.assertEqual(obj.references().all()[0], r1)
        self.assertEqual(obj.pmids().all()[0], pm1)
        self.assertEqual(obj.experiment_types().all()[0], ps1)

        self.assertEqual(pm1.interactions().all()[0], obj)
        self.assertEqual(pm1.psimis().all()[0], ps1)

        self.assertEqual(ps1.interactions().all()[0], obj)
        self.assertEqual(ps1.pmids().all()[0], pm1)

    def test_can_add_reference_with_null_psimi(self):
        obj = Interaction(source=self.a, target=self.a)
        obj.save(self.session, commit=True)

        pm1 = Pubmed('A')
        pm1.save(self.session, commit=True)
        r1 = obj.add_reference(self.session, pm1, None, commit=True)

        self.assertEqual(Reference.query.count(), 1)
        self.assertEqual(obj.references().all()[0], r1)
        self.assertEqual(obj.pmids().all()[0], pm1)
        self.assertEqual(obj.experiment_types().all(), [])

        self.assertEqual(pm1.interactions().all()[0], obj)
        self.assertEqual(pm1.psimis().all(), [])

    def test_save_can_update_fields(self):
        obj = Interaction(source=self.a, target=self.a)
        obj.save(self.session, commit=True)

        # Now update some fields
        obj = Interaction.query.first()
        obj.pfam = "PF002"
        obj.go_mf = "GO:000001"
        obj.save(self.session, commit=True)

        obj = Interaction.query.first()
        self.assertEqual(obj.go_mf, "GO:000001")
        self.assertEqual(obj.pfam, "PF002")

    def test_can_get_by_interactors(self):
        obj = Interaction(source=self.a, target=self.b, label=None)
        obj.save(self.session, commit=True)
        self.assertEqual(Interaction.get_by_interactors('A', 2), obj)
        self.assertEqual(Interaction.get_by_interactors(0, 2), None)

    def test_can_get_by_label(self):
        obj1 = Interaction(
            source=self.a, target=self.b, label='Activation,Phosphorylation'
        )
        obj1.save(self.session, commit=True)
        obj2 = Interaction(
            source=self.a, target=self.a, label='Phosphorylation'
        )
        obj2.save(self.session, commit=True)

        self.assertEqual(Interaction.get_by_label('activation').count(), 1)
        self.assertEqual(Interaction.get_by_label('activation').first(), obj1)
        self.assertEqual(Interaction.get_by_label(' '), None)

    def test_can_get_by_source(self):
        obj1 = Interaction(source=self.a, target=self.b)
        obj2 = Interaction(source=self.b, target=self.b)
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        self.assertEqual(Interaction.get_by_source('A').count(), 1)
        self.assertEqual(Interaction.get_by_source('C').count(), 0)
        self.assertEqual(Interaction.get_by_source('D'), None)
        with self.assertRaises(TypeError):
            Interaction.get_by_source(None)
            Interaction.get_by_source([])

    def test_can_get_by_target(self):
        obj1 = Interaction(source=self.a, target=self.b)
        obj2 = Interaction(source=self.b, target=self.b)
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        self.assertEqual(Interaction.get_by_target(2).count(), 2)
        self.assertEqual(Interaction.get_by_target(1).count(), 0)
        self.assertEqual(Interaction.get_by_target(45), None)
        with self.assertRaises(TypeError):
            Interaction.get_by_target(None)
            Interaction.get_by_target([])


class TestPubmedModel(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)
        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.pb.save(self.session, commit=True)
        self.ia = Interaction(self.pa, self.pb)
        self.ia.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_cannot_save_two_pubmeds_with_same_accession(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            pb = Pubmed(accession="A")

    def test_cannot_update_two_pubmeds_with_same_accession(self):
        pa = Pubmed(accession="A")
        pb = Pubmed(accession="B")
        pa.save(self.session, commit=True)
        pb.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            pb.accession = 'A'


class TestPsimiModel(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)

        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.pb.save(self.session, commit=True)
        self.ia = Interaction(self.pa, self.pb)
        self.ia.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_cannot_save_two_psimis_with_same_accession(self):
        pa = Psimi(accession="A", description="blah")
        pa.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            pb = Psimi(accession="A", description="blah")

    def test_cannot_update_two_psimis_with_same_accession(self):
        pa = Psimi(accession="A", description="blah")
        pb = Psimi(accession="B", description="blah")
        pa.save(self.session, commit=True)
        pb.save(self.session, commit=True)
        with self.assertRaises(ObjectAlreadyExists):
            pb.accession = 'A'
