#!/usr/bin/env python

import os

from unittest import TestCase
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, Query

from ..database.models import Protein, Interaction, Base
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
        self.session.execute("DROP TABLE Protein")
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
        self.session.execute("DROP TABLE interaction")
        self.session.execute("DROP TABLE Protein")
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
