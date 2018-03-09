import os
from unittest import TestCase

from ..database import delete_database, create_session, cleanup_database
from ..database.exceptions import ObjectAlreadyExists, ObjectNotFound
from ..database.exceptions import NonMatchingTaxonomyIds
from ..database.models import Protein, Interaction, Psimi, Pubmed
from ..database.validators import (
    format_annotation,
    format_annotations,
    format_label,
    format_labels,
    validate_annotations,
    validate_go_annotations,
    validate_pfam_annotations,
    validate_interpro_annotations,
    validate_keywords,
    validate_function,
    validate_protein,
    validate_source_and_target,
    validate_same_taxonid,
    validate_labels,
    validate_boolean,
    validate_joint_id,
    validate_interaction_does_not_exist,
    validate_training_holdout_is_labelled,
    validate_gene_id,
    validate_taxon_id,
    validate_accession,
    validate_description,
    validate_accession_does_not_exist,
    validate_uniprot_does_not_exist
)

base_path = os.path.dirname(__file__)
db_path = os.path.normpath("{}/databases/test.db".format(base_path))


class TestFormatAnnotation(TestCase):

    def test_returns_none_if_input_is_none(self):
        result = format_annotation(None)
        self.assertIsNone(result)

    def test_returns_none_empty_string(self):
        result = format_annotation(' ')
        self.assertIsNone(result)

    def test_upper_cases_by_default(self):
        result = format_annotation('go111')
        self.assertEqual(result, 'GO111')

    def test_lower_cases_if_upper_is_false(self):
        result = format_annotation('go111', upper=False)
        self.assertEqual(result, 'go111')

    def test_strips_white_space(self):
        result = format_annotation(' go111 ')
        self.assertEqual(result, 'GO111')

    def test_type_err_not_str_or_none(self):
        with self.assertRaises(TypeError):
            format_annotation(1)
        with self.assertRaises(TypeError):
            format_annotation([])


class TestFormatAnnotations(TestCase):

    def test_removes_duplicate_annotations(self):
        value = "1,1"
        expected = ["1"]
        result = format_annotations(value, allow_duplicates=False)
        self.assertEqual(result, expected)

    def test_does_not_uppercase_when_upper_is_false(self):
        value = "dog"
        expected = ["dog"]
        result = format_annotations(value, upper=False)
        self.assertEqual(result, expected)

    def test_allows_duplicate_annotations(self):
        value = "1,1"
        expected = ["1", "1"]
        result = format_annotations(value, allow_duplicates=True)
        self.assertEqual(result, expected)

    def test_alpha_orders_annotations(self):
        value = "2,1"
        expected = ["1", "2"]
        result = format_annotations(value)
        self.assertEqual(result, expected)

    def test_uppercases_annotations(self):
        value = "dog"
        expected = ["DOG"]
        result = format_annotations(value)
        self.assertEqual(result, expected)

    def test_removes_blank(self):
        value = "1,,"
        expected = ["1"]
        result = format_annotations(value)
        self.assertEqual(result, expected)

    def test_strips_whitespace(self):
        value = "   1   "
        expected = ["1"]
        result = format_annotations(value)
        self.assertEqual(result, expected)

    def test_splits_on_comma(self):
        value = "1;2, 3"
        expected = ["1;2", "3"]
        result = format_annotations(value)
        self.assertEqual(result, expected)

    def test_returns_none_values_is_none(self):
        self.assertIsNone(format_annotations(None))

    def test_removes_none_and_empty_strings(self):
        self.assertIsNone(format_annotations([None, ' ']))

    def test_returns_none_no_valid_values(self):
        self.assertIsNone(format_annotations([' ']))

    def test_typeerror_not_list_set_none_or_str(self):
        with self.assertRaises(TypeError):
            format_annotations(1)


class TestFormatLabel(TestCase):

    def test_returns_none_if_input_is_none(self):
        result = format_label(None)
        self.assertIsNone(result)

    def test_returns_none_empty_string(self):
        result = format_label(' ')
        self.assertIsNone(result)

    def test_cap_by_default(self):
        result = format_label('activation')
        self.assertEqual(result, 'Activation')

    def test_non_cap_if_cap_is_false(self):
        result = format_label('activation', capitalize=False)
        self.assertEqual(result, 'activation')

    def test_strips_white_space(self):
        result = format_label(' activation ')
        self.assertEqual(result, 'Activation')

    def test_converts_int_to_string(self):
        result = format_label(1)
        self.assertEqual(result, '1')

    def test_type_err_not_str_int_or_none(self):
        with self.assertRaises(TypeError):
            format_annotation(1.1)
        with self.assertRaises(TypeError):
            format_annotation([])


class TestFormatLabels(TestCase):

    def test_removes_duplicates(self):
        value = "activation,activation"
        expected = ["Activation"]
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_does_not_cap_when_cap_is_false(self):
        value = "activation"
        expected = ["activation"]
        result = format_labels(value, capitalize=False)
        self.assertEqual(result, expected)

    def test_removes_duplicate(self):
        value = "activation,activation"
        expected = ["Activation"]
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_alpha_orders(self):
        value = "inhibition,activation"
        expected = ["Activation", 'Inhibition']
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_captializes(self):
        value = "activation"
        expected = ["Activation"]
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_removes_blank(self):
        value = "activation,,"
        expected = ["Activation"]
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_strips_whitespace(self):
        value = "   activation   "
        expected = ["Activation"]
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_splits_on_comma(self):
        value = "activation;inhibition, acetylation"
        expected = ["Acetylation", "Activation;inhibition"]
        result = format_labels(value)
        self.assertEqual(result, expected)

    def test_returns_none_values_is_none(self):
        self.assertIsNone(format_labels(None))

    def test_removes_none_and_empty_strings(self):
        self.assertIsNone(format_labels([None, ' ']))

    def test_returns_none_no_valid_values(self):
        self.assertIsNone(format_labels([' ']))

    def test_typeerror_not_list_set_none_or_str(self):
        with self.assertRaises(TypeError):
            format_labels(1)


class TestValidateAnnotations(TestCase):

    def test_valuerror_check_annotations_invalid_dbtype(self):
        with self.assertRaises(ValueError):
            validate_annotations(["IPR201", "GO:00001"], dbtype="HELLO")

    def test_check_annotations_ignores_falsey_values(self):
        self.assertIsNone(validate_annotations("", dbtype="GO"))
        self.assertIsNone(validate_annotations([], dbtype="GO"))
        self.assertIsNone(validate_annotations([' '], dbtype="GO"))

    def test_valueerror_not_go_annotations(self):
        with self.assertRaises(ValueError):
            validate_annotations(["IPR201", "GO:00001"], dbtype="GO")

    def test_valueerror_not_interpro_annotations(self):
        with self.assertRaises(ValueError):
            validate_annotations(["IPR201", "GO:00001"], dbtype="IPR")

    def test_valueerror_not_pfam_annotations(self):
        with self.assertRaises(ValueError):
            validate_annotations(["IPR201", "PF00001"], dbtype="PF")

    def test_check_go_annotations(self):
        result = validate_annotations(["GO:00002", "GO:00001"], dbtype="GO")
        self.assertEqual(["GO:00001", "GO:00002"], result)

    def test_check_interpro_annotations(self):
        result = validate_annotations(
            ["IPR1", "ipr1"], dbtype="INTERPRO", upper=True,
            allow_duplicates=True
        )
        self.assertEqual(["IPR1", "IPR1"], result)

    def test_check_pfam_annotations(self):
        result = validate_annotations(
            ["pf1", " ", None, 'PF1'], dbtype="PFAM", upper=True,
            allow_duplicates=False
        )
        self.assertEqual(["PF1"], result)


class TestValidateGO(TestCase):

    def test_joins_splits_on_comma(self):
        result = validate_go_annotations(["GO:00002", "GO:00001"])
        self.assertEqual("GO:00001,GO:00002", result)
        result = validate_go_annotations("GO:00002,GO:00001")
        self.assertEqual("GO:00001,GO:00002", result)

    def test_returns_none_no_valid_annotations(self):
        self.assertIsNone(validate_go_annotations([None, '', ' ']))
        self.assertIsNone(validate_go_annotations(None))
        self.assertIsNone(validate_go_annotations(" "))


class TestValidatePfam(TestCase):

    def test_joins_splits_on_comma(self):
        result = validate_pfam_annotations(["PF00002", "PF00001"])
        self.assertEqual("PF00001,PF00002", result)
        result = validate_pfam_annotations("PF00002,PF00001")
        self.assertEqual("PF00001,PF00002", result)

    def test_returns_none_no_valid_annotations(self):
        self.assertIsNone(validate_pfam_annotations([None, '', ' ']))
        self.assertIsNone(validate_pfam_annotations(None))
        self.assertIsNone(validate_pfam_annotations(" "))


class TestValidateInterpro(TestCase):

    def test_joins_splits_on_comma(self):
        result = validate_interpro_annotations(["IPR00002", "IPR00001"])
        self.assertEqual("IPR00001,IPR00002", result)
        result = validate_interpro_annotations("IPR00001,IPR00002")
        self.assertEqual("IPR00001,IPR00002", result)

    def test_returns_none_no_valid_annotations(self):
        self.assertIsNone(validate_interpro_annotations([None, '', ' ']))
        self.assertIsNone(validate_interpro_annotations(None))
        self.assertIsNone(validate_interpro_annotations(" "))


class TestValidateKeywords(TestCase):

    def test_typeerror_not_list_set_none_or_str(self):
        with self.assertRaises(TypeError):
            validate_keywords(1)

    def test_joins_on_comma(self):
        result = validate_keywords(["dog", "cat"])
        self.assertEqual("Cat,Dog", result)

    def test_splits_on_comma(self):
        result = validate_keywords("dog,cat")
        self.assertEqual("Cat,Dog", result)

    def test_removes_duplicates(self):
        result = validate_keywords(["dog", "dog"])
        self.assertEqual("Dog", result)

    def test_removes_white_space(self):
        result = validate_keywords('dog, , ')
        self.assertEqual("Dog", result)

    def test_allows_duplicates(self):
        result = validate_keywords('dog,dog', allow_duplicates=True)
        self.assertEqual("Dog,Dog", result)

    def test_returns_none_no_valid_keywords(self):
        self.assertIsNone(validate_keywords([None, '', ' ']))
        self.assertIsNone(validate_keywords(None))
        self.assertIsNone(validate_keywords(" "))


class TestValidateFunction(TestCase):

    def test_typeerror_not_none_or_str(self):
        with self.assertRaises(TypeError):
            validate_function([])
            validate_function(1)

    def test_removes_white_space(self):
        result = validate_function('dog ')
        self.assertEqual("dog", result)

    def test_returns_none_no_valid_input(self):
        self.assertIsNone(validate_function(None))
        self.assertIsNone(validate_function(" "))


class TestValidateDescription(TestCase):

    def test_typeerror_not_none_or_str(self):
        with self.assertRaises(TypeError):
            validate_function([])
            validate_function(1)

    def test_removes_white_space(self):
        result = validate_function('dog ')
        self.assertEqual("dog", result)

    def test_returns_none_no_valid_input(self):
        self.assertIsNone(validate_function(None))
        self.assertIsNone(validate_function(" "))


class TestValidateLabels(TestCase):
    def test_joins_splits_on_comma(self):
        result = validate_labels(["activation", "inhibition"])
        self.assertEqual("Activation,Inhibition", result)
        result = validate_labels("activation,inhibition")
        self.assertEqual("Activation,Inhibition", result)

    def test_returns_none_no_valid_labels(self):
        self.assertIsNone(validate_labels([None, '', ' ']))
        self.assertIsNone(validate_labels(None))
        self.assertIsNone(validate_labels(" "))

    def test_returns_list_if_requesteed(self):
        self.assertEqual(
            validate_labels([None, '', ' '], return_list=True), []
        )
        self.assertEqual(
            validate_labels("Activation,Inhibition", return_list=True),
            ["Activation", "Inhibition"]
        )


class TestValidateJointId(TestCase):

    def test_sorts_and_joins_on_comma(self):
        self.assertEqual(validate_joint_id(2, 1), '1,2')

    def test_typeerror_not_int(self):
        with self.assertRaises(TypeError):
            validate_joint_id([], 1)
            validate_joint_id(2, '1')


class TestTrainingHoldoutMustBeLabelled(TestCase):

    def test_error_label_is_none_but_training_holdout_is_true(self):
        with self.assertRaises(ValueError):
            validate_training_holdout_is_labelled(None, True, False)
        with self.assertRaises(ValueError):
            validate_training_holdout_is_labelled(None, False, True)

        # Should not raise error
        validate_training_holdout_is_labelled(None, False, False)


class TestValidateGeneId(TestCase):

    def test_typeerror_not_none_or_str(self):
        with self.assertRaises(TypeError):
            validate_gene_id([])
            validate_gene_id(1)

    def test_removes_white_space(self):
        result = validate_gene_id(' EGFR ')
        self.assertEqual("EGFR", result)

    def test_returns_none_no_valid_input(self):
        self.assertIsNone(validate_gene_id(None))
        self.assertIsNone(validate_gene_id(" "))


class TestValidateTaxonId(TestCase):

    def test_typeerror_not_int(self):
        with self.assertRaises(TypeError):
            validate_taxon_id([])
            validate_taxon_id('1')


class TestValidateBoolean(TestCase):

    def test_typeerror_not_bool(self):
        with self.assertRaises(TypeError):
            validate_boolean([])
            validate_boolean('1')

    def test_none_converted_to_false(self):
        self.assertEqual(validate_boolean(None), False)


class TestValidateInteractionDoesNotExist(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.pa = Protein(uniprot_id='A', taxon_id=9606)
        self.pb = Protein(uniprot_id='B', taxon_id=9606)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)

        self.ia = Interaction(
            source=self.pa.id, target=self.pb.id
        )
        self.ia.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_objectexist_error_if_exists(self):
        with self.assertRaises(ObjectAlreadyExists):
            validate_interaction_does_not_exist(1, 2)
        with self.assertRaises(ObjectAlreadyExists):
            validate_interaction_does_not_exist(2, 1)
        validate_interaction_does_not_exist(1, 1)  # should not raise error


class TestValidateSameTaxonId(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.pa = Protein(uniprot_id='A', taxon_id=9606)
        self.pb = Protein(uniprot_id='B', taxon_id=0)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_error_not_same_taxon(self):
        with self.assertRaises(NonMatchingTaxonomyIds):
            validate_same_taxonid('A', 'B')
        with self.assertRaises(NonMatchingTaxonomyIds):
            validate_same_taxonid(1, 2)
        with self.assertRaises(NonMatchingTaxonomyIds):
            validate_same_taxonid(1, self.pb)
        with self.assertRaises(NonMatchingTaxonomyIds):
            validate_same_taxonid(self.pa, 'B')

    def test_returns_taxon_id_if_valid(self):
        result = validate_same_taxonid('A', 'A')
        self.assertEqual(result, 9606)

    def test_returns_taxon_id_if_valid_proteins_passed(self):
        result = validate_same_taxonid(self.pa, self.pa)
        self.assertEqual(result, 9606)


class TestValidateSourceAndTarget(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.pa = Protein(uniprot_id='A', taxon_id=9606)
        self.pb = Protein(uniprot_id='B', taxon_id=0)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_returns_ids_from_accession(self):
        s, t = validate_source_and_target('A', 'B')
        self.assertEqual(s, 1)
        self.assertEqual(t, 2)

    def test_returns_ids_from_proteinn(self):
        s, t = validate_source_and_target(self.pa, self.pb)
        self.assertEqual(s, 1)
        self.assertEqual(t, 2)

    def test_returns_ids_from_int(self):
        s, t = validate_source_and_target(1, 2)
        self.assertEqual(s, 1)
        self.assertEqual(t, 2)

    def test_raise_error_not_existing(self):
        with self.assertRaises(ObjectNotFound):
            validate_source_and_target(1, 0)
        with self.assertRaises(ObjectNotFound):
            validate_source_and_target('A', 'C')


class TestValidateProtein(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.pa = Protein(uniprot_id='A', taxon_id=9606)
        self.pb = Protein(uniprot_id='B', taxon_id=0)
        self.pa.save(self.session, commit=True)
        self.pb.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_typeerr_not_protein_str_or_int(self):
        with self.assertRaises(TypeError):
            validate_protein(None)

    def test_object_not_found_error_non_existing(self):
        with self.assertRaises(ObjectNotFound):
            validate_protein('C')
        with self.assertRaises(ObjectNotFound):
            validate_protein(0)

    def test_value_error_protein_with_none_id(self):
        with self.assertRaises(ValueError):
            self.pa.id = None
            validate_protein(self.pa)

    def test_returns_id(self):
        result = validate_protein(self.pa)
        self.assertEqual(result, 1)

        result = validate_protein('B')
        self.assertEqual(result, 2)

        result = validate_protein(1)
        self.assertEqual(result, 1)

    def test_returns_instance_if_true(self):
        result = validate_protein(1, return_instance=True)
        self.assertEqual(result, self.pa)


class TestValidateAndCheckUniprotIdAndAccession(TestCase):
    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.pa = Protein(uniprot_id='A', taxon_id=9606)
        self.pubmed = Pubmed(accession='PM1')
        self.psimi = Psimi(accession='PSI1', description="hello")

        self.pa.save(self.session, commit=True)
        self.pubmed.save(self.session, commit=True)
        self.psimi.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_type_err_not_string(self):
        with self.assertRaises(TypeError):
            validate_accession(None)
        with self.assertRaises(TypeError):
            validate_accession(1)

    def test_value_err_empty_string(self):
        with self.assertRaises(ValueError):
            validate_accession(" ")

    def test_object_exists_err_if_check_exists_true_protein(self):
        validate_accession("r", Protein, upper=False, check_exists=True)
        with self.assertRaises(ObjectAlreadyExists):
            validate_accession("a", Protein, upper=True, check_exists=True)

    def test_no_object_exists_err_if_unirpot_does_not_exist_protein(self):
        value = validate_accession("C", Protein, check_exists=True)
        self.assertEqual(value, "C")

    def test_object_exists_err_if_check_exists_true_pubmed(self):
        with self.assertRaises(ObjectAlreadyExists):
            validate_accession("PM1", Pubmed, check_exists=True)

    def test_no_object_exists_err_if_unirpto_does_not_exist_pubmed(self):
        value = validate_accession("PM2", Pubmed, check_exists=True)
        self.assertEqual(value, "PM2")

    def test_object_exists_err_if_check_exists_true_psimi(self):
        with self.assertRaises(ObjectAlreadyExists):
            validate_accession("PSI1", Psimi, check_exists=True)

    def test_no_object_exists_err_if_unirpto_does_not_exist_psimi(self):
        value = validate_accession("PSI2", Psimi, check_exists=True)
        self.assertEqual(value, "PSI2")

    def test_no_object_exists_err_if_check_exists_false(self):
        value = validate_accession("A", Protein, check_exists=False)
        self.assertEqual(value, 'A')

    def test_strip_white_space(self):
        value = validate_accession(" a ", Protein, check_exists=False)
        self.assertEqual(value, "A")

    def test_strip_does_not_upper_case_if_upper_is_false(self):
        value = validate_accession(
            "a", Protein, upper=False, check_exists=False)
        self.assertEqual(value, "a")


class TestUniProtAndAccessionDoesNotExist(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.pa = Protein(uniprot_id='A', taxon_id=9606)
        self.pubmed = Pubmed(accession='PM1')
        self.psimi = Psimi(accession='PSI1', description="hello")

        self.pa.save(self.session, commit=True)
        self.pubmed.save(self.session, commit=True)
        self.psimi.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_attr_err_class_doesnt_have_accession_attr(self):
        with self.assertRaises(AttributeError):
            validate_accession_does_not_exist(None, None)

    def test_err_doesnt_have_accession_attr_pubmed(self):
        validate_accession_does_not_exist("PM2", Pubmed)
        with self.assertRaises(ObjectAlreadyExists):
            validate_accession_does_not_exist("PM1", Pubmed)

    def test_err_doesnt_have_accession_attr_psimi(self):
        validate_accession_does_not_exist("PSI2", Psimi)
        with self.assertRaises(ObjectAlreadyExists):
            validate_accession_does_not_exist("PSI1", Psimi)

    def test_attr_err_class_doesnt_have_uniprot_id_attr(self):
        with self.assertRaises(AttributeError):
            validate_uniprot_does_not_exist(None, None)

    def test_err_doesnt_have_accession_attr_protein(self):
        validate_uniprot_does_not_exist("B", Protein)
        with self.assertRaises(ObjectAlreadyExists):
            validate_uniprot_does_not_exist("A", Protein)
