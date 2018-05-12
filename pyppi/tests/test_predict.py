import os
import numpy as np
from itertools import product
from Bio import SwissProt
from unittest import TestCase

from ..database import create_session, delete_database, cleanup_database
from ..database.models import Interaction, Protein
from ..database.utilities import create_interaction
from ..database.exceptions import ObjectAlreadyExists

from ..data_mining.uniprot import parse_record_into_protein
from ..data_mining.features import compute_interaction_features
from ..data_mining.ontology import get_active_instance

from ..models.binary_relevance import MixedBinaryRelevanceClassifier

from ..predict import _check_classifier_and_selection
from ..predict import _update_missing_protein_map
from ..predict import _create_missing_interactions
from ..predict import classify_interactions

from ..predict.utilities import load_dataset, DEFAULT_SELECTION

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

base_path = os.path.dirname(__file__)
db_path = os.path.normpath("{}/databases/test.db".format(base_path))

test_obo_file = '{}/{}'.format(base_path, "test_data/test_go.obo.gz")
dag = get_active_instance(filename=test_obo_file)


class TestCheckClassifierAndSelection(TestCase):

    def test_valueerror_custom_classifier_no_selection(self):
        with self.assertRaises(ValueError):
            _check_classifier_and_selection(classifier=1, selection=None)

    def test_valueerror_invalid_selection(self):
        with self.assertRaises(ValueError):
            _check_classifier_and_selection(classifier=1, selection=['1'])


class TestUpdateMissingProteinMap(TestCase):

    def setUp(self):
        self.session, self.engine = create_session(db_path)
        self.p1 = Protein(uniprot_id='A', taxon_id=9606, reviewed=True)
        self.p2 = Protein(uniprot_id='B', taxon_id=1, reviewed=True)
        self.p1.save(self.session, commit=True)
        self.p2.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_adds_new_proteins_to_map(self):
        ppis = [('A', 'A'), ('A', 'P31946')]
        pm = _update_missing_protein_map(ppis, self.session)
        expected = {
            'A': self.p1,
            'P31946': Protein.query.get(3)
        }
        self.assertEqual(expected, pm)

    def test_adds_invalid_proteins_as_none(self):
        ppis = [('A', 'A'), ('A', 'P3194')]
        pm = _update_missing_protein_map(ppis, self.session, verbose=False)
        expected = {
            'A': self.p1,
            'P3194': None
        }
        self.assertEqual(expected, pm)

    def test_proteins_different_taxonid_added_as_none(self):
        ppis = [('A', 'A'), ('B', 'A')]
        pm = _update_missing_protein_map(ppis, self.session, verbose=False)
        expected = {
            'A': self.p1,
            'B': None
        }
        self.assertEqual(expected, pm)

        ppis = [('A', 'A'), ('Q3TYD4', 'A')]
        pm = _update_missing_protein_map(ppis, self.session, verbose=False)
        expected = {
            'A': self.p1,
            'Q3TYD4': None
        }
        self.assertEqual(expected, pm)

    def test_ignores_taxonid_if_none(self):
        ppis = [('A', 'A'), ('B', 'A')]
        pm = _update_missing_protein_map(
            ppis, self.session, taxon_id=None, verbose=False)
        expected = {
            'A': self.p1,
            'B': self.p2
        }
        self.assertEqual(expected, pm)

        ppis = [('A', 'A'), ('Q3TYD4', 'A')]
        pm = _update_missing_protein_map(
            ppis, self.session, taxon_id=None, verbose=False)
        expected = {
            'A': self.p1,
            'Q3TYD4': Protein.query.get(3)
        }
        self.assertEqual(expected, pm)

    def test_ignores_none_input(self):
        ppis = [(None, 'A')]
        pm = _update_missing_protein_map(ppis, self.session, verbose=False)
        expected = {
            'A': self.p1,
        }
        self.assertEqual(expected, pm)


class TestCreateMissingInteractions(TestCase):

    def setUp(self):
        self.session, self.engine = create_session(db_path)
        delete_database(self.session)

        self.p1 = Protein(uniprot_id='A', taxon_id=9606, reviewed=True)
        self.p2 = Protein(uniprot_id='B', taxon_id=9606, reviewed=True)
        self.p3 = Protein(uniprot_id='C', taxon_id=0, reviewed=True)
        self.p4 = Protein(uniprot_id='D', taxon_id=0, reviewed=True)
        self.p1.save(self.session, commit=True)
        self.p2.save(self.session, commit=True)
        self.p3.save(self.session, commit=True)
        self.p4.save(self.session, commit=True)
        self.i1 = Interaction(source=self.p1, target=self.p2)
        self.i1.save(session=self.session, commit=True)
        self.p_map = {p.uniprot_id: p for p in Protein.query.all()}

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_existing_interactions_returned_and_invalid_is_empty_list(self):
        valid, invalid = _create_missing_interactions(
            ppis=[('A', 'B')],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [self.i1])
        self.assertEqual(invalid, [])

    def test_interaction_with_none_source_added_to_invalid(self):
        valid, invalid = _create_missing_interactions(
            ppis=[(None, 'B')],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [])
        self.assertEqual(invalid, [(None, 'B')])

    def test_interaction_with_none_target_added_to_invalid(self):
        valid, invalid = _create_missing_interactions(
            ppis=[('A', None)],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [])
        self.assertEqual(invalid, [('A', None)])

    def test_interaction_with_differing_taxonid_added_to_invalid(self):
        valid, invalid = _create_missing_interactions(
            ppis=[('C', 'D')],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [])
        self.assertEqual(invalid, [('C', 'D')])

        valid, invalid = _create_missing_interactions(
            ppis=[('C', 'A')],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [])
        self.assertEqual(invalid, [('C', 'A')])

        valid, invalid = _create_missing_interactions(
            ppis=[('A', 'D')],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [])
        self.assertEqual(invalid, [('A', 'D')])

    def test_new_interactions_created(self):
        valid, invalid = _create_missing_interactions(
            ppis=[('A', 'A')],
            protein_map=self.p_map,
            session=self.session
        )
        self.assertEqual(valid, [Interaction.query.get(2)])
        self.assertEqual(invalid, [])


class TestMakePredictions(TestCase):
    # This class implicitly also tests parse_predictions since
    # make_predictions is essentially a wrapper for parse_predictions
    # TODO: Separate these tests.
    def setUp(self):
        self.records = open(os.path.normpath(
            "{}/test_data/test_sprot_records.dat".format(base_path)
        ), 'rt')
        self.session, self.engine = create_session(db_path)
        delete_database(self.session)

        self.proteins = []
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            protein.save(self.session, commit=True)
            self.proteins.append(protein)

        self.labels = ['Activation', 'Inhibition', 'Acetylation']
        self.interactions = []
        for protein_a, protein_b in product(self.proteins, self.proteins):
            class_kwargs = compute_interaction_features(protein_a, protein_b)
            label = '{},{}'.format(
                self.labels[protein_a.id - 1],
                self.labels[protein_b.id - 1]
            )
            try:
                interaction = create_interaction(
                    protein_a, protein_b, labels=label, session=self.session,
                    verbose=False, save=True, commit=True, **class_kwargs
                )
                self.interactions.append(interaction)
            except ObjectAlreadyExists:
                continue

        self.X, self.y, _ = load_dataset(
            self.interactions, self.labels, selection=DEFAULT_SELECTION
        )
        base = Pipeline(steps=[
            ('vectorizer', CountVectorizer(
                lowercase=False, stop_words=[':', 'GO'])),
            ('estimator', LogisticRegression(random_state=0))
        ])
        self.clf = MixedBinaryRelevanceClassifier(
            [clone(base) for _ in range(len(self.labels))]
        )
        self.clf.fit(self.X, self.y)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)
        self.records.close()

    def test_can_make_proba_predictions_on_existing_interactions(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = (
            self.clf.predict_proba(self.X), []
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_can_make_binary_predictions_on_existing_interactions(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        predictions = classify_interactions(
            ppis, proba=False, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = (
            self.clf.predict(self.X), []
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_can_make_predictions_on_list_of_interaction_objects(self):
        ppis = self.interactions
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = (
            self.clf.predict_proba(self.X), []
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_ignores_None_or_not_interaction_objects(self):
        ppis = [self.interactions[0], None, '1']
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = (
            self.clf.predict_proba([self.X[0]]), [None, '1']
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_returns_empty_list_no_valid_interactions(self):
        ppis = [(1, 2), (1, 2, 3), None, '1']
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = ([], [(1, 2), (1, 2, 3), None, '1'])
        self.assertEqual(predictions[0], expected[0])
        self.assertEqual(predictions[1], expected[1])

    def test_typeerror_first_elem_not_interaction_or_tuple(self):
        with self.assertRaises(TypeError):
            ppis = [1, None, '1']
            classify_interactions(
                ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
                taxon_id=9606, verbose=False, session=self.session
            )

    def test_creates_new_interactions(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        delete_database(self.session)
        classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        self.assertEqual(Interaction.query.count(), 6)

    def test_removed_duplicate_interactions_interactions(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        ppis.append(ppis[0])
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = (
            self.clf.predict_proba(self.X), []
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_invalid_ppis_added_to_invalid(self):
        ppis = [('A', 'B'), ('Q04917', 'X')]
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = ([], ppis)
        self.assertEqual(predictions[0], expected[0])
        self.assertEqual(predictions[1], expected[1])

    def test_non_matching_taxonid_of_existing_ppis_added_to_invalid(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=0, verbose=False, session=self.session
        )
        expected = ([], ppis)
        self.assertEqual(predictions[0], expected[0])
        self.assertEqual(predictions[1], expected[1])

    def test_non_matching_taxonid_of_new_ppis_added_to_invalid(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        delete_database(self.session)
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=0, verbose=False, session=self.session
        )
        expected = ([], ppis)
        self.assertEqual(predictions[0], expected[0])
        self.assertEqual(predictions[1], expected[1])

    def test_taxonid_ignored_if_None(self):
        ppis = [
            (
                Protein.query.get(i.source).uniprot_id,
                Protein.query.get(i.target).uniprot_id
            )
            for i in self.interactions
        ]
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=None, verbose=False, session=self.session
        )
        expected = (
            self.clf.predict_proba(self.X), []
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_ignores_duplicate_entries(self):
        ppi_1 = (
            Protein.query.get(self.interactions[0].source).uniprot_id,
            Protein.query.get(self.interactions[0].target).uniprot_id
        )
        ppi_2 = (
            Protein.query.get(self.interactions[0].target).uniprot_id,
            Protein.query.get(self.interactions[0].source).uniprot_id
        )
        ppis = [ppi_1, ppi_2]
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )

        X, _, _ = load_dataset(
            [self.interactions[0]], self.labels, selection=DEFAULT_SELECTION
        )
        expected = (
            self.clf.predict_proba(X), []
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_multiple_choice_uniprot_ids_get_put_in_invalid(self):
        ppis = [('Q8NDH8', 'P0CG12')]
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )
        expected = ([], [('P0CG12', 'Q8NDH8')])
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])

    def test_outdated_accessions_map_to_most_recent_entries(self):
        ppis = [('A8K9K2', 'A8K9K2')]  # maps to P31946
        entry = Protein.get_by_uniprot_id('P31946')
        interaction = Interaction.get_by_interactors(entry, entry)
        predictions = classify_interactions(
            ppis, proba=True, classifier=self.clf, selection=DEFAULT_SELECTION,
            taxon_id=9606, verbose=False, session=self.session
        )

        X, _, _ = load_dataset(
            [interaction], self.labels, selection=DEFAULT_SELECTION
        )
        expected = (
            self.clf.predict_proba(X), [], {'A8K9K2': 'P31946'}
        )
        self.assertTrue(np.all(np.isclose(predictions[0], expected[0])))
        self.assertEqual(predictions[1], expected[1])
        self.assertEqual(predictions[2], expected[2])
