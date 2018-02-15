#!/usr/bin/env python

import os
import shutil

from unittest import TestCase
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, Query

from ..database import begin_transaction, Base
from ..database.managers import ProteinManager
from ..database.models import Protein, Interaction, Psimi, Pubmed
from ..database.models import _check_annotations, _format_annotations
from ..database.exceptions import ObjectNotFound, ObjectAlreadyExists

base_path = os.path.dirname(__file__)


class TestAnnotationValidation(TestCase):

    def test_removes_duplicate_annotations(self):
        value = "1,1"
        expected = ["1"]
        result = _format_annotations(value, allow_duplicates=False)
        self.assertEqual(result, expected)

    def test_does_not_uppercase_when_upper_is_false(self):
        value = "dog"
        expected = ["dog"]
        result = _format_annotations(value, upper=False)
        self.assertEqual(result, expected)

    def test_allows_duplicate_annotations(self):
        value = "1,1"
        expected = ["1", "1"]
        result = _format_annotations(value, allow_duplicates=True)
        self.assertEqual(result, expected)

    def test_alpha_orders_annotations(self):
        value = "2,1"
        expected = ["1", "2"]
        result = _format_annotations(value)
        self.assertEqual(result, expected)

    def test_uppercases_annotations(self):
        value = "dog"
        expected = ["DOG"]
        result = _format_annotations(value)
        self.assertEqual(result, expected)

    def test_removes_blank(self):
        value = "1,,"
        expected = ["1"]
        result = _format_annotations(value)
        self.assertEqual(result, expected)

    def test_strips_whitespace(self):
        value = "   1   "
        expected = ["1"]
        result = _format_annotations(value)
        self.assertEqual(result, expected)

    def test_splits_on_comma(self):
        value = "1;2, 3"
        expected = ["1;2", "3"]
        result = _format_annotations(value)
        self.assertEqual(result, expected)

    def test_valuerror_check_annotations_invalid_dbtype(self):
        with self.assertRaises(ValueError):
            _check_annotations(["IPR201", "GO:00001"], dbtype="HELLO")

    def test_check_annotations_ignores_falsey_values(self):
        _check_annotations([], dbtype="GO")
        _check_annotations("", dbtype="GO")

    def test_valueerror_not_go_annotations(self):
        with self.assertRaises(ValueError):
            _check_annotations(["IPR201", "GO:00001"], dbtype="GO")

    def test_valueerror_not_interpro_annotations(self):
        with self.assertRaises(ValueError):
            _check_annotations(["IPR201", "GO:00001"], dbtype="IPR")

    def test_valueerror_not_pfam_annotations(self):
        with self.assertRaises(ValueError):
            _check_annotations(["IPR201", "PF00001"], dbtype="PF")

    # The following tests should pass without raising any errors
    def test_check_go_annotations(self):
        _check_annotations(["GO:00002", "GO:00001"], dbtype="GO")

    def test_check_interpro_annotations(self):
        _check_annotations(["GO:00002", "GO:00001"], dbtype="GO")

    def test_check_pfam_annotations(self):
        _check_annotations(["GO:00002", "GO:00001"], dbtype="GO")


class TestProteinModel(TestCase):

    def setUp(self):
        self.url = "sqlite:///" + os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.engine = create_engine(self.url, echo=False)
        Base.metadata.create_all(self.engine)
        self.session = Session(bind=self.engine)

    def tearDown(self):
        self.session.rollback()
        self.session.query(Interaction).delete()
        self.session.query(Protein).delete()
        self.session.query(Pubmed).delete()
        self.session.query(Psimi).delete()
        self.session.execute("DROP TABLE pmid_interactions")
        self.session.execute("DROP TABLE psimi_interactions")
        self.session.commit()
        self.session.close()

    def test_equality(self):
        protein_1 = Protein("A", taxon_id=9606, reviewed=False)
        protein_1.save(self.session, commit=True)
        protein_2 = self.session.query(Protein).get(protein_1.id)
        self.assertEqual(protein_1, protein_2)

    def test_inequality(self):
        protein_1 = Protein("A", taxon_id=9606, reviewed=False)
        protein_1.save(self.session, commit=True)
        protein_2 = Protein("B", taxon_id=0, reviewed=False)
        protein_2.save(self.session, commit=True)
        self.assertNotEqual(protein_1, protein_2)

    def test_integrityerror_adding_duplicate_ids(self):
        self.session.add(Protein(
            uniprot_id="abc", taxon_id=1, reviewed=False)
        )
        self.session.commit()
        with self.assertRaises(IntegrityError):
            self.session.add(Protein(
                uniprot_id="abc", taxon_id=1, reviewed=False)
            )
            self.session.commit()

    def test_integrityerror_adding_null_id_field(self):
        with self.assertRaises(IntegrityError):
            self.session.add(
                Protein(uniprot_id=None, taxon_id=1, reviewed=False)
            )
            self.session.commit()

    def test_integrityerror_adding_null_taxonid(self):
        with self.assertRaises(IntegrityError):
            self.session.add(
                Protein(uniprot_id='abc', taxon_id=None, reviewed=False)
            )
            self.session.commit()

    def test_can_add_accession(self):
        obj = Protein(
            uniprot_id="abc", taxon_id=1,
            pfam="pf002,PF001,PF001,", reviewed=False
        )
        self.session.add(obj)
        self.session.commit()

        obj = self.session.query(Protein).first()
        self.assertEqual(self.session.query(Protein).count(), 1)
        self.assertEqual(obj.pfam, "PF001,PF002")
        self.assertEqual(obj.keywords, None)
        self.assertEqual(obj.go_mf, None)

    def test_setter_typeerror_not_str_list_or_set(self):
        value = 1
        with self.assertRaises(TypeError):
            obj = Protein(
                uniprot_id="abc", taxon_id=1,
                go_mf=value, go_cc=value, go_bp=value,
                interpro=value, pfam=value, reviewed=False
            )
            self.session.commit()

    def test_keywords_are_capitalized(self):
        obj = Protein(
            uniprot_id="abc", taxon_id=1, reviewed=False,
            keywords="dog,CAT,CAt"
        )
        self.session.commit()
        self.assertEqual(obj.keywords, "Cat,Dog")

    def test_save_method_updates_fields(self):
        obj = Protein(
            uniprot_id="abc", taxon_id=1,
            pfam="pf002,PF001,PF001,", reviewed=False
        )
        obj.save(self.session, commit=True)
        self.assertEqual(self.session.query(Protein).count(), 1)

        obj.pfam = "pf002"
        obj.uniprot_id = "ab"
        obj.save(self.session, commit=True)

        obj = self.session.query(Protein).first()
        self.assertEqual(self.session.query(Protein).count(), 1)
        self.assertEqual(obj.pfam, "PF002")
        self.assertEqual(obj.uniprot_id, "ab")

    def test_save_method_does_not_change_unaltered_fields(self):
        obj = Protein(
            uniprot_id="abc", taxon_id=1,
            pfam="pf002,PF001,PF001,", reviewed=False
        )
        obj.save(self.session, commit=True)
        self.assertEqual(self.session.query(Protein).count(), 1)

        obj.pfam = "pf002"
        obj.save(self.session, commit=True)

        obj = self.session.query(Protein).first()
        self.assertEqual(self.session.query(Protein).count(), 1)
        self.assertEqual(obj.pfam, "PF002")
        self.assertEqual(obj.uniprot_id, "abc")

    def test_save_method_update_doesnt_fail_unique_constraint(self):
        obj = Protein(
            uniprot_id="abc", taxon_id=1,
            pfam="pf002,PF001,PF001,", reviewed=False
        )
        obj.save(self.session, commit=True)
        self.assertEqual(self.session.query(Protein).count(), 1)

        # Same object shouldn't fail constraint on unique uniprot_id
        obj.pfam = "pf002"
        obj.save(self.session, commit=True)

        obj = self.session.query(Protein).first()
        self.assertEqual(self.session.query(Protein).count(), 1)
        self.assertEqual(obj.pfam, "PF002")
        self.assertEqual(obj.uniprot_id, "abc")


class TestInteractionModel(TestCase):

    def setUp(self):
        self.url = "sqlite:///" + os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.engine = create_engine(self.url, echo=False)
        Base.metadata.create_all(self.engine)
        self.session = Session(bind=self.engine, )

        self.a = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.b = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.c = Protein(uniprot_id="C", taxon_id=0, reviewed=False)
        self.session.add_all([self.a, self.b, self.c])
        self.session.commit()

    def tearDown(self):
        self.session.rollback()
        self.session.query(Interaction).delete()
        self.session.query(Protein).delete()
        self.session.query(Pubmed).delete()
        self.session.query(Psimi).delete()
        self.session.execute("DROP TABLE pmid_interactions")
        self.session.execute("DROP TABLE psimi_interactions")
        self.session.commit()
        self.session.close()

    def test_can_add_interaction(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
            pfam="PF002,pf001,,"
        )
        obj.save(self.session, commit=True)

        obj = self.session.query(Interaction).first()
        self.assertEqual(self.session.query(Interaction).count(), 1)
        self.assertEqual(obj.source, self.a.id)
        self.assertEqual(obj.target, self.b.id)
        self.assertEqual(obj.pfam, "PF001,PF002")
        self.assertEqual(obj.go_mf, None)
        self.assertEqual(obj.label, None)

    def test_can_add_new_label(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label=None
        )
        obj.add_label('activation')
        self.assertEqual(obj.label, 'Activation')

    def test_labels_as_list_is_sorted(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation,inhibition'
        )
        self.assertEqual(
            obj.labels_as_list, list(sorted(['Activation', 'Inhibition']))
        )

    def test_can_get_singleton_label_as_list(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        self.assertEqual(
            obj.labels_as_list, ['Activation']
        )

    def test_labels_as_list_empty_if_label_is_none(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label=None
        )
        self.assertEqual(
            obj.labels_as_list, []
        )

    def test_can_add_to_existing_label(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='Phosphorylation'
        )
        obj.add_label('activation')
        self.assertEqual(obj.label, 'Activation,Phosphorylation')

    def test_can_add_an_existing_label_without_duplicating(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='Phosphorylation'
        )
        obj.add_label('Phosphorylation')
        self.assertEqual(obj.label, 'Phosphorylation')

    def test_raises_typeerror_when_setting_nonstring_label(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label=None
        )
        with self.assertRaises(TypeError):
            obj.add_label(['Activation'])

    def test_raises_value_when_setting_empty_string(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label=None
        )
        with self.assertRaises(ValueError):
            obj.add_label('')

    def test_raises_value_when_setting_multilabel(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label=None
        )
        with self.assertRaises(ValueError):
            obj.add_label('activation,activation')

    def test_can_remove_existing_label(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        obj.remove_label('activation')
        self.assertEqual(obj.label, None)

    def test_can_remove_existing_label_not_case_sensitive(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        obj.remove_label('ACTIVATION')
        self.assertEqual(obj.label, None)

    def test_remove_label_does_nothing_if_label_not_found(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        obj.remove_label('hello')
        self.assertEqual(obj.label, 'Activation')

    def test_can_remove_existing_duplicated_label(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation,activation'
        )
        obj.remove_label('activation')
        self.assertEqual(obj.label, None)

    def test_raises_typeerror_when_removing_nonstring_label(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        with self.assertRaises(TypeError):
            obj.remove_label(['Activation'])

    def test_raises_value_when_removing_empty_string(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        with self.assertRaises(ValueError):
            obj.remove_label('')

    def test_raises_value_when_removing_multilabel(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label='activation'
        )
        with self.assertRaises(ValueError):
            obj.remove_label('activation,activation')

    def test_combined_attr_is_a_string_of_sorted_ids_comma_separated(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        expected = ','.join(sorted([str(self.a.id), str(self.b.id)]))
        self.assertEqual(obj.combined, expected)

    def test_can_retrieve_interaction_from_accession(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        obj.save(self.session, commit=True)
        self.assertEqual(self.a.interactions[0], obj)

    def test_filters_out_nones_from_label_list(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False,
            label=[None, None, 'Label']
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.label, 'Label')

    def test_interaction_equality(self):
        obj1 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        obj2 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        self.assertEqual(obj1, obj2)

    def test_interaction_equal_reversed_source_target(self):
        obj1 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        obj2 = Interaction(
            source=self.b.id,
            target=self.a.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        self.assertEqual(obj1, obj2)

    def test_interaction_inequality(self):
        obj1 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        obj2 = Interaction(
            source=self.a.id,
            target=self.a.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        self.assertNotEqual(obj1, obj2)

    def test_objectexists_error_when_switching_source_target(self):
        obj1 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        obj2 = Interaction(
            source=self.b.id,
            target=self.a.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        with self.assertRaises(ObjectAlreadyExists):
            obj1.save(self.session, commit=True)
            obj2.save(self.session, commit=True)

    def test_objectexists_error_raised(self):
        obj1 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        obj2 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        with self.assertRaises(ObjectAlreadyExists):
            obj1.save(self.session, commit=True)
            obj2.save(self.session, commit=True)

    def test_integrityerror_null_source(self):
        obj = Interaction(
            source=None,
            target=self.b.id,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        with self.assertRaises(IntegrityError):
            obj.save(self.session, commit=True)

    def test_integrityerror_null_target(self):
        obj = Interaction(
            source=self.a.id,
            target=None,
            is_holdout=False,
            is_interactome=False,
            is_training=False
        )
        with self.assertRaises(IntegrityError):
            obj.save(self.session, commit=True)

    def test_typeerror_null_holdout(self):
        with self.assertRaises(TypeError):
            obj = Interaction(
                source=self.a.id,
                target=self.b.id,
                is_holdout=None,
                is_interactome=False,
                is_training=False
            )
            obj.save(self.session, commit=True)

    def test_typeerror_null_training(self):
        with self.assertRaises(TypeError):
            obj = Interaction(
                source=self.a.id,
                target=self.b.id,
                is_holdout=False,
                is_interactome=False,
                is_training=None
            )
            obj.save(self.session, commit=True)

    def test_typeerror_null_interactome(self):
        with self.assertRaises(TypeError):
            obj = Interaction(
                source=self.a.id,
                target=self.b.id,
                is_holdout=False,
                is_interactome=None,
                is_training=False
            )
            obj.save(self.session, commit=True)

    def test_cannot_create_interaction_linking_to_nonexistant_accession(self):
        with self.assertRaises(ObjectNotFound):
            obj = Interaction(
                source=self.a.id,
                target=99999,
                is_holdout=False,
                is_interactome=False,
                is_training=False
            )
            obj.save(self.session, commit=True)

        with self.assertRaises(ObjectNotFound):
            obj = Interaction(
                source=99999,
                target=self.a.id,
                is_holdout=False,
                is_interactome=False,
                is_training=False
            )
            obj.save(self.session, commit=True)

    def test_failed_save_rollsback(self):
        with self.assertRaises(IntegrityError):
            obj = Interaction(
                source=self.a.id,
                target=None,
                is_holdout=False,
                is_interactome=False,
                is_training=False
            )
            obj.save(self.session, commit=True)
        self.assertEqual(self.session.query(Interaction).count(), 0)

    def test_has_missing_data(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            go_bp="GO:000001", go_cc="GO:000001", go_mf="GO:000001",
            ulca_go_bp="GO:000001", ulca_go_cc="GO:000001",
            ulca_go_mf="GO:000001", interpro="", pfam="PF001"
        )
        obj.save(self.session, commit=True)
        self.assertTrue(obj.has_missing_data)

    def test_does_not_have_missing_data(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            go_bp="GO:000001", go_cc="GO:000001", go_mf="GO:000001",
            ulca_go_bp="GO:000001", ulca_go_cc="GO:000001",
            ulca_go_mf="GO:000001", interpro="IPR001", pfam="PF001",
            keywords="hello"
        )
        obj.save(self.session, commit=True)
        self.assertFalse(obj.has_missing_data)

    def test_valueerror_different_taxonomy_ids(self):
        obj = Interaction(
            source=self.a.id, target=self.c.id,
            is_holdout=False, is_training=False, is_interactome=False
        )
        with self.assertRaises(ValueError):
            obj.save(self.session, commit=True)

    def test_valueerror_mixed_annotations(self):
        obj = Interaction(
            source=self.a.id, target=self.c.id,
            is_holdout=False, is_training=False, is_interactome=False,
            go_bp="GO:000001,IPR001"
        )
        with self.assertRaises(ValueError):
            obj.save(self.session, commit=True)

    def test_sorts_and_capitalizes_labels(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            label="inhibition,activation"
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.label, "Activation,Inhibition")

    def test_no_duplicate_labels(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            label="activation,activation"
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.label, "Activation")

    def test_keywords_are_capitalized_and_duplicated(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            keywords="dog,dog,cat"
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.keywords, "Cat,Dog,Dog")

    def test_no_strip_whitespace_labels(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            label="activation , "
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.label, "Activation")

    def test_interaction_assumes_source_target_taxonid(self):
        obj = Interaction(
            source=self.a.id, target=self.b.id,
            is_holdout=False, is_training=False, is_interactome=False,
            label="activation"
        )
        obj.save(self.session, commit=True)
        self.assertEqual(obj.taxon_id, self.a.taxon_id)

    def test_create_typeerror_label_not_string_list_or_set(self):
        with self.assertRaises(TypeError):
            obj = Interaction(
                source=self.a.id, target=self.c.id,
                is_holdout=False, is_training=False, is_interactome=False,
                label=7
            )

    def test_create_valueerror_training_must_be_labeled(self):
        with self.assertRaises(ValueError):
            obj = Interaction(
                source=self.a.id, target=self.c.id,
                is_holdout=False, is_training=True, is_interactome=False,
                label=None
            )
            obj.save(self.session, commit=True)

    def test_create_valueerror_holdout_must_be_labeled(self):
        with self.assertRaises(ValueError):
            obj = Interaction(
                source=self.a.id, target=self.c.id,
                is_holdout=True, is_training=False, is_interactome=False,
                label=None
            )
            obj.save(self.session, commit=True)

    def test_save_can_update_fields(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
            pfam="PF002,pf001"
        )
        obj.save(self.session, commit=True)
        self.assertEqual(self.session.query(Interaction).count(), 1)

        obj = self.session.query(Interaction).first()
        self.assertEqual(obj.source, self.a.id)
        self.assertEqual(obj.target, self.b.id)
        self.assertEqual(obj.pfam, "PF001,PF002")
        self.assertEqual(obj.go_mf, None)
        self.assertEqual(obj.label, None)

        # Now update some fields
        obj.pfam = "PF002"
        obj.go_mf = "GO:000001"
        obj.save(self.session, commit=True)

        obj = self.session.query(Interaction).first()
        self.assertEqual(obj.go_mf, "GO:000001")
        self.assertEqual(obj.pfam, "PF002")

    def test_cannot_update_source_target_to_a_preexisting_source_target(self):
        obj1 = Interaction(
            source=self.b.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj1.save(self.session, commit=True)

        obj2 = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj2.save(self.session, commit=True)

        # Change source/target to match obj1
        obj2.source = self.b.id
        with self.assertRaises(ObjectAlreadyExists):
            obj2.save(self.session, commit=True)

    def test_update_source_target_updates_taxonid(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)
        obj.source = self.c.id
        obj.target = self.c.id
        obj.save(self.session, commit=True)
        self.assertEqual(obj.taxon_id, 0)

    def test_update_source_target_updates_combined(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)
        obj.source = self.b.id
        obj.save(self.session, commit=True)
        self.assertEqual(obj.combined, '2,2')

    def test_error_update_different_taxonid(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)
        obj.source = self.c.id

        with self.assertRaises(ValueError):
            obj.save(self.session, commit=True)

    def test_can_append_to_pmid_and_will_update_pmid_interactions(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)

        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)

        obj.add_pmid_reference(pa)
        obj = self.session.query(Interaction).first()
        pa = self.session.query(Pubmed).first()
        self.assertEqual(obj.pmid, [pa])
        self.assertEqual(pa.interactions, [obj])

    def test_TE_add_or_remove_non_pubmed_reference(self):
        pa = Psimi(accession="A", description="")
        pa.save(self.session, commit=True)
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)

        with self.assertRaises(TypeError):
            obj.add_pmid_reference(pa)
        with self.assertRaises(TypeError):
            obj.remove_pmid_reference(pa)

    def test_TE_add_or_remove_non_psimi_reference(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)

        with self.assertRaises(TypeError):
            obj.add_psimi_reference(pa)
        with self.assertRaises(TypeError):
            obj.add_psimi_reference(pa)

    def test_cannot_append_duplicate_pmids(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)

        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)
        obj.add_pmid_reference(pa)
        obj.add_pmid_reference(pa)
        self.assertEqual(obj.pmid, [pa])
        self.assertEqual(pa.interactions, [obj])

    def test_cannot_append_duplicate_psimis(self):
        pa = Psimi(accession="A", description='blah')
        pa.save(self.session, commit=True)

        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)
        obj.add_psimi_reference(pa)
        obj.add_psimi_reference(pa)
        self.assertEqual(obj.psimi, [pa])
        self.assertEqual(pa.interactions, [obj])

    def test_can_append_to_psimi_and_will_update_psimi_interactions(self):
        pa = Psimi(accession="A", description='blah')
        pa.save(self.session, commit=True)

        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)
        obj.add_psimi_reference(pa)

        obj = self.session.query(Interaction).first()
        pa = self.session.query(Psimi).first()
        self.assertEqual(obj.psimi, [pa])
        self.assertEqual(pa.interactions, [obj])

    def test_can_remove_non_existing_pmid_or_psimi_without_error(self):
        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)

        psimi_1 = Psimi(accession="A", description='blah')
        psimi_2 = Psimi(accession="B", description='blah')
        psimi_1.save(self.session, commit=True)
        psimi_2.save(self.session, commit=True)
        obj.add_psimi_reference(psimi_1)
        obj.remove_psimi_reference(psimi_2)
        self.assertTrue(obj.psimi, [psimi_1])

        pubmed_1 = Pubmed(accession="A")
        pubmed_2 = Pubmed(accession="B")
        pubmed_1.save(self.session, commit=True)
        pubmed_2.save(self.session, commit=True)
        obj.add_pmid_reference(pubmed_1)
        obj.remove_pmid_reference(pubmed_2)
        self.assertTrue(obj.pmid, [pubmed_1])

    def test_remove_pmid_and_will_update_pmid_interactions(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)

        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)

        # Check everything has updated first
        obj.pmid.append(pa)
        self.assertEqual(obj.pmid, [pa])
        self.assertEqual(pa.interactions, [obj])

        # Now remove and refresh
        obj.remove_pmid_reference(pa)
        self.assertEqual(obj.pmid, [])
        self.assertEqual(pa.interactions, [])

    def test_remove_psimi_and_will_update_psimi_interactions(self):
        pa = Psimi(accession="A", description='blah')
        pa.save(self.session, commit=True)

        obj = Interaction(
            source=self.a.id,
            target=self.b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        obj.save(self.session, commit=True)

        # Check everything has updated first
        obj.psimi.append(pa)
        self.assertEqual(obj.psimi, [pa])
        self.assertEqual(pa.interactions, [obj])

        # Now remove and refresh
        obj.remove_psimi_reference(pa)
        self.assertEqual(obj.psimi, [])
        self.assertEqual(pa.interactions, [])


class TestPubmedModel(TestCase):

    def setUp(self):
        self.url = "sqlite:///" + os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.engine = create_engine(self.url, echo=False)
        Base.metadata.create_all(self.engine)
        self.session = Session(bind=self.engine, )

        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.pb.save(self.session, commit=True)
        self.ia = Interaction(
            source=self.pa.id,
            target=self.pb.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        self.ia.save(self.session, commit=True)

    def tearDown(self):
        self.session.rollback()
        self.session.query(Interaction).delete()
        self.session.query(Protein).delete()
        self.session.query(Pubmed).delete()
        self.session.query(Psimi).delete()
        self.session.execute("DROP TABLE pmid_interactions")
        self.session.execute("DROP TABLE psimi_interactions")
        self.session.commit()
        self.session.close()

    def test_cannot_save_two_pubmeds_with_same_accession(self):
        pa = Pubmed(accession="A")
        pb = Pubmed(accession="A")
        with self.assertRaises(IntegrityError):
            pa.save(self.session, commit=True)
            pb.save(self.session, commit=True)

    def test_cannot_update_two_pubmeds_with_same_accession(self):
        pa = Pubmed(accession="A")
        pb = Pubmed(accession="B")
        pa.save(self.session, commit=True)
        pb.save(self.session, commit=True)
        with self.assertRaises(IntegrityError):
            pb.accession = 'A'
            pb.save(self.session, commit=True)

    def test_append_to_interactions_and_will_update_interaction_pmid(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)
        pa.interactions.append(self.ia)

        ia = self.session.query(Interaction).first()
        self.assertEqual(ia.pmid, [pa])

    def test_remove_interactions_and_will_update_interaction_pmid(self):
        pa = Pubmed(accession="A")
        pa.save(self.session, commit=True)

        pa.interactions.append(self.ia)
        ia = self.session.query(Interaction).first()
        self.assertEqual(ia.pmid, [pa])

        pa.interactions.remove(self.ia)
        ia = self.session.query(Interaction).first()
        self.assertEqual(ia.pmid, [])


class TestPsimiModel(TestCase):

    def setUp(self):
        self.url = "sqlite:///" + os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.engine = create_engine(self.url, echo=False)
        Base.metadata.create_all(self.engine)
        self.session = Session(bind=self.engine, )

        self.pa = Protein(uniprot_id="A", taxon_id=9606, reviewed=False)
        self.pa.save(self.session, commit=True)
        self.pb = Protein(uniprot_id="B", taxon_id=9606, reviewed=False)
        self.pb.save(self.session, commit=True)
        self.ia = Interaction(
            source=self.pa.id,
            target=self.pb.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False,
        )
        self.ia.save(self.session, commit=True)

    def tearDown(self):
        self.session.rollback()
        self.session.query(Interaction).delete()
        self.session.query(Protein).delete()
        self.session.query(Pubmed).delete()
        self.session.query(Psimi).delete()
        self.session.execute("DROP TABLE pmid_interactions")
        self.session.execute("DROP TABLE psimi_interactions")
        self.session.commit()
        self.session.close()

    def test_cannot_save_two_psimis_with_same_accession(self):
        pa = Psimi(accession="A", description="blah")
        pb = Psimi(accession="A", description="blah")
        with self.assertRaises(IntegrityError):
            pa.save(self.session, commit=True)
            pb.save(self.session, commit=True)

    def test_cannot_update_two_psimis_with_same_accession(self):
        pa = Psimi(accession="A", description="blah")
        pb = Psimi(accession="B", description="blah")
        pa.save(self.session, commit=True)
        pb.save(self.session, commit=True)
        with self.assertRaises(IntegrityError):
            pb.accession = 'A'
            pb.save(self.session, commit=True)

    def test_can_append_to_interactions_and_will_update_interaction_psimis(self):
        pa = Psimi(accession="A", description="blah")
        pa.save(self.session, commit=True)
        pa.interactions.append(self.ia)

        ia = self.session.query(Interaction).first()
        self.assertEqual(ia.psimi, [pa])

    def test_can_remove_interactions_and_will_update_interaction_psmi(self):
        pa = Psimi(accession="A", description="blah")
        pa.save(self.session, commit=True)

        pa.interactions.append(self.ia)
        ia = self.session.query(Interaction).first()
        self.assertEqual(ia.psimi, [pa])

        pa.interactions.remove(self.ia)
        ia = self.session.query(Interaction).first()
        self.assertEqual(ia.psimi, [])

