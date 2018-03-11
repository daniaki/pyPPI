import os
import numpy as np
from numpy.random import RandomState
from unittest import TestCase

from ..base.constants import MAX_SEED
from ..database import create_session, delete_database, cleanup_database
from ..database.models import Interaction, Protein

from ..models.binary_relevance import MixedBinaryRelevanceClassifier
from ..models.classifier_chain import KRandomClassifierChains
from ..models.utilities import publication_ensemble
from ..models.utilities import get_parameter_distribution_for_model

from ..predict.utilities import (
    DEFAULT_SELECTION,
    load_dataset,
    paper_model,
    validation_model,
    interactions_to_Xy_format,
    load_training_dataset,
    load_validation_dataset,
    load_interactome_dataset,
    train_paper_model,
    save_to_arff
)

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

base_path = os.path.dirname(__file__)
db_path = os.path.normpath("{}/databases/test.db".format(base_path))
max_int = MAX_SEED


class TestInteractionsToXyFormat(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = Protein.query.get(self.protein_a.id)
        self.protein_b = Protein.query.get(self.protein_b.id)
        self.protein_c = Protein.query.get(self.protein_c.id)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_can_conversion_to_Xy_removes_colon_from_GO_terms(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label=None, go_mf='GO:1', go_bp='GO:2', go_cc=None,
            interpro='IPR4', pfam='PF5', ulca_go_mf='GO:6', ulca_go_bp='GO:7',
            ulca_go_cc='GO:8'
        )
        obj2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=True, is_training=False, is_interactome=True,
            label='activation,hello', go_mf='GO:1', go_bp='GO:2', go_cc='GO:3',
            interpro=None, pfam='PF5', ulca_go_mf='GO:6', ulca_go_bp='GO:7',
            ulca_go_cc='GO:8'
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)

        X, y = interactions_to_Xy_format(
            [obj1, obj2],
            selection=[
                Interaction.columns().GO_MF,
                Interaction.columns().GO_BP,
                Interaction.columns().GO_CC,

                Interaction.columns().INTERPRO,
                Interaction.columns().PFAM,

                Interaction.columns().ULCA_GO_MF,
                Interaction.columns().ULCA_GO_BP,
                Interaction.columns().ULCA_GO_CC,
            ]
        )

        result_y = [[], ['Activation', "Hello"]]
        result_x = [
            'GO1,GO2,IPR4,PF5,GO6,GO7,GO8',
            'GO1,GO2,GO3,PF5,GO6,GO7,GO8',
        ]
        self.assertEqual(result_x, list(X))
        self.assertEqual(result_y, y)


# ------------- VALIDATION MODEL ------------------------ #
class TestValidationModel(TestCase):

    # BinaryRel -> RandomizedGS -> Pipeline(vec, est)

    def test_creates_n_estimators(self):
        clf = validation_model(labels=[0, 1, 2])
        self.assertEqual(len(clf.estimators), 3)

    def test_creates_pipeline_with_count_vectorizer(self):
        clf = validation_model(labels=[0, 1, 2])
        for i in range(3):
            self.assertIsInstance(clf.estimators[i], RandomizedSearchCV)
            self.assertIsInstance(clf.estimators[i].estimator, Pipeline)
            self.assertIsInstance(
                clf.estimators[i].estimator.steps[0][1], CountVectorizer)

    def test_vectorizer_has_correct_parameters(self):
        clf = validation_model(labels=[0, 1, 2])
        for i in range(3):
            vec = clf.estimators[i].estimator.steps[0][1]
            self.assertEqual(vec.stop_words, None)
            self.assertEqual(vec.binary, True)
            self.assertEqual(vec.lowercase, False)

    def test_correctly_sets_up_grid_search(self):
        clf = validation_model(
            labels=[0, 1, 2], rcv_splits=5, rcv_iter=50, scoring='accurracy',
            n_jobs_br=1, n_jobs_gs=2, random_state=0, n_jobs_model=3,
            model="RandomForestClassifier"
        )
        params = get_parameter_distribution_for_model(
            "RandomForestClassifier", step='estimator'
        )

        rng = RandomState(0)
        model_random_state = rng.randint(max_int)
        cv_random_state = rng.randint(max_int)
        rcv_random_state = rng.randint(max_int)

        for i in range(3):
            self.assertEqual(clf.n_jobs, 1)

            rgs_est = clf.estimators[i]
            cv = rgs_est.cv
            pipe = rgs_est.estimator
            model = rgs_est.estimator.steps[1][1]

            self.assertIsInstance(pipe, Pipeline)
            self.assertEqual(pipe.steps[0][0], 'vectorizer')
            self.assertEqual(pipe.steps[0][1].binary, True)
            self.assertEqual(pipe.steps[1][0], 'estimator')

            self.assertEqual(rgs_est.param_distributions, params)
            self.assertEqual(rgs_est.random_state, rcv_random_state)
            self.assertEqual(rgs_est.n_iter, 50)
            self.assertEqual(rgs_est.error_score, 0.0)
            self.assertEqual(rgs_est.n_jobs, 2)
            self.assertEqual(rgs_est.refit, True)
            self.assertEqual(rgs_est.scoring, 'accurracy')

            self.assertEqual(cv.n_splits, 5)
            self.assertEqual(cv.random_state, cv_random_state)

            self.assertIsInstance(model, RandomForestClassifier)
            self.assertEqual(model.random_state, model_random_state)
            self.assertEqual(model.n_jobs, 3)


# ------------- PAPER MODEL ------------------------ #
class TestPaperModel(TestCase):

    # BinaryRel -> RandomizedGS -> Pipeline(vec, est)

    def test_paper_model_creates_n_estimators(self):
        clf = paper_model(labels=[0, 1, 2])
        self.assertEqual(len(clf.estimators), 3)

    def test_creates_pipeline_with_count_vectorizer(self):
        clf = paper_model(labels=[0, 1, 2])
        for i in range(3):
            self.assertIsInstance(clf.estimators[i], RandomizedSearchCV)
            self.assertIsInstance(clf.estimators[i].estimator, Pipeline)
            self.assertIsInstance(
                clf.estimators[i].estimator.steps[0][1], CountVectorizer)

    def test_vectorizer_has_correct_parameters(self):
        clf = paper_model(labels=[0, 1, 2])
        for i in range(3):
            vec = clf.estimators[i].estimator.steps[0][1]
            self.assertEqual(vec.stop_words, None)
            self.assertEqual(vec.binary, True)
            self.assertEqual(vec.lowercase, False)

    def test_correctly_sets_up_grid_search(self):
        clf = paper_model(
            labels=[0, 1, 2], rcv_splits=5, rcv_iter=50, scoring='f1',
            n_jobs_br=1, n_jobs_gs=2, n_jobs_model=4, random_state=0
        )
        params = get_parameter_distribution_for_model(
            "LogisticRegression", step='estimator'
        )
        params['vectorizer__binary'] = [False, True]

        rng = RandomState(0)
        model_random_state = rng.randint(max_int)
        cv_random_state = rng.randint(max_int)
        rcv_random_state = rng.randint(max_int)

        for i in range(3):
            self.assertEqual(clf.n_jobs, 1)

            rgs_est = clf.estimators[i]
            cv = rgs_est.cv
            pipe = rgs_est.estimator
            model = rgs_est.estimator.steps[1][1]

            self.assertIsInstance(pipe, Pipeline)
            self.assertEqual(pipe.steps[0][0], 'vectorizer')
            self.assertEqual(pipe.steps[1][0], 'estimator')

            self.assertEqual(rgs_est.param_distributions, params)
            self.assertEqual(rgs_est.random_state, rcv_random_state)
            self.assertEqual(rgs_est.n_iter, 50)
            self.assertEqual(rgs_est.error_score, 0.0)
            self.assertEqual(rgs_est.n_jobs, 2)
            self.assertEqual(rgs_est.refit, True)
            self.assertEqual(rgs_est.scoring, 'f1')

            self.assertEqual(cv.n_splits, 5)
            self.assertEqual(cv.random_state, cv_random_state)

            self.assertIsInstance(model, LogisticRegression)
            self.assertEqual(model.random_state, model_random_state)
            self.assertEqual(model.n_jobs, 4)


class TestLoadInteractomeDataset(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = Protein.query.get(self.protein_a.id)
        self.protein_b = Protein.query.get(self.protein_b.id)
        self.protein_c = Protein.query.get(self.protein_c.id)

        self.i1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label=None, go_mf='GO:1', go_bp='GO:2', go_cc=None,
            interpro='IPR4', pfam='PF5', ulca_go_mf='GO:6', ulca_go_bp='GO:7',
            ulca_go_cc='GO:8'
        )
        self.i2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=True, is_training=False, is_interactome=True,
            label='activation,hello', go_mf='GO:1', go_bp='GO:2', go_cc='GO:3',
            interpro=None, pfam='PF5', ulca_go_mf='GO:6', ulca_go_bp='GO:7',
            ulca_go_cc='GO:8'
        )
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_keeps_holdout(self):
        X = load_interactome_dataset()
        result_x = ['GO1,GO2,IPR4,PF5', 'GO1,GO3,GO2,PF5']
        self.assertEqual(result_x, list(X))

    def test_keeps_training(self):
        self.i1.is_holdout = False
        self.i2.is_holdout = False
        self.i2.is_training = True

        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)

        X = load_interactome_dataset()
        result_x = ['GO1,GO2,IPR4,PF5', 'GO1,GO3,GO2,PF5']
        self.assertEqual(result_x, list(X))

    def test_returns_empty_list_if_no_interactions_found(self):
        self.i1.is_interactome = False
        self.i2.is_interactome = False
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)
        X = load_interactome_dataset()

        result_x = []
        self.assertEqual(result_x, list(X))

    def test_returns_empty_list_non_matching_taxon(self):
        self.i1.is_interactome = False
        self.i2.is_interactome = False
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)
        X = load_interactome_dataset(taxon_id=0)

        result_x = []
        self.assertEqual(result_x, list(X))

    def test_correct_selection(self):
        self.i1.is_holdout = False
        self.i2.is_holdout = False
        self.i2.is_training = True

        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)

        X = load_interactome_dataset(selection=['pfam'])
        result_x = ['PF5', 'PF5']
        self.assertEqual(result_x, list(X))


class TestLoadTrainingDataset(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = Protein.query.get(self.protein_a.id)
        self.protein_b = Protein.query.get(self.protein_b.id)
        self.protein_c = Protein.query.get(self.protein_c.id)

        self.i1 = Interaction(  # interactome only
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label=None, go_mf='GO:11', go_bp='GO:12', go_cc='GO:13',
            interpro='IPR11', pfam='PF11'
        )
        self.i2 = Interaction(  # training only
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label='activation', go_mf='GO:21', go_bp='GO:22', go_cc='GO:23',
            interpro='IPR21', pfam='PF21'
        )
        self.i3 = Interaction(  # holdout only
            source=self.protein_c.id, target=self.protein_c.id,
            is_holdout=True, is_training=False, is_interactome=False,
            label='activation,inhibition', go_mf='GO:31', go_bp='GO:32',
            go_cc='GO:33', interpro='IPR31', pfam='PF31'
        )
        self.i4 = Interaction(  # holdout and training and interactome
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=True, is_training=True, is_interactome=True,
            label='inhibition', go_mf='GO:41', go_bp='GO:42',
            go_cc='GO:43', interpro='IPR41', pfam='PF41'
        )
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)
        self.i3.save(self.session, commit=True)
        self.i4.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_keeps_holdout_and_interactome_that_are_training(self):
        data = load_training_dataset()
        result_x = [
            'GO21,GO23,GO22,IPR21,PF21',
            'GO31,GO33,GO32,IPR31,PF31',
            'GO41,GO43,GO42,IPR41,PF41'
        ]
        result_y = [[1, 0], [1, 1], [0, 1]]
        labels = ['Activation', 'Inhibition']
        self.assertEqual(result_x, list(data['training'][0]))
        self.assertEqual(list(data['training'][1][0]), result_y[0])
        self.assertEqual(list(data['training'][1][1]), result_y[1])
        self.assertEqual(list(data['training'][1][2]), result_y[2])
        self.assertEqual(labels, list(data['labels']))

    def test_returns_empty_dict_if_no_interactions_found(self):
        self.i2.is_training = False
        self.i3.is_holdout = False
        self.i4.is_training = False
        self.i4.is_holdout = False
        self.i2.label = None
        self.i3.label = None
        self.i4.label = None
        self.i2.save(self.session, commit=True)
        self.i3.save(self.session, commit=True)
        self.i4.save(self.session, commit=True)
        data = load_training_dataset()
        self.assertEqual(data, {})

    def test_returns_empty_dict_non_matching_taxon(self):
        data = load_training_dataset(taxon_id=0)
        self.assertEqual(data, {})

    def test_correct_selection(self):
        data = load_training_dataset(selection=['pfam'])
        result_x = [
            'PF21',
            'PF31',
            'PF41'
        ]
        result_y = [[1, 0], [1, 1], [0, 1]]
        labels = ['Activation', 'Inhibition']
        self.assertEqual(result_x, list(data['training'][0]))
        self.assertEqual(list(data['training'][1][0]), result_y[0])
        self.assertEqual(list(data['training'][1][1]), result_y[1])
        self.assertEqual(list(data['training'][1][2]), result_y[2])
        self.assertEqual(labels, list(data['labels']))


class TestLoadValidationDataset(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = Protein.query.get(self.protein_a.id)
        self.protein_b = Protein.query.get(self.protein_b.id)
        self.protein_c = Protein.query.get(self.protein_c.id)

        self.i1 = Interaction(  # interactome only
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label=None, go_mf='GO:11', go_bp='GO:12', go_cc='GO:13',
            interpro='IPR11', pfam='PF11'
        )
        self.i2 = Interaction(  # training only
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label='activation', go_mf='GO:21', go_bp='GO:22', go_cc='GO:23',
            interpro='IPR21', pfam='PF21'
        )
        self.i3 = Interaction(  # holdout only
            source=self.protein_c.id, target=self.protein_c.id,
            is_holdout=True, is_training=False, is_interactome=False,
            label='activation', go_mf='GO:31', go_bp='GO:32',
            go_cc='GO:33', interpro='IPR31', pfam='PF31'
        )
        self.i4 = Interaction(  # holdout and training and interactome
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=True, is_training=True, is_interactome=True,
            label='phosphorylation', go_mf='GO:41', go_bp='GO:42',
            go_cc='GO:43', interpro='IPR41', pfam='PF41'
        )
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)
        self.i3.save(self.session, commit=True)
        self.i4.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_keeps_holdout_and_interactome_that_are_training(self):
        data = load_validation_dataset()

        train_x = ['GO21,GO23,GO22,IPR21,PF21']
        train_y = [[1]]
        test_x = ['GO31,GO33,GO32,IPR31,PF31']
        test_y = [[1]]
        labels = ['Activation']
        self.assertEqual(train_x, list(data['training'][0]))
        self.assertEqual(list(data['training'][1][0]), train_y[0])

        self.assertEqual(test_x, list(data['testing'][0]))
        self.assertEqual(list(data['testing'][1][0]), test_y[0])

        self.assertEqual(labels, list(data['labels']))

    def test_returns_empty_dict_if_no_holdout_found(self):
        self.i2.is_training = False
        self.i3.is_holdout = False
        self.i4.is_training = False
        self.i4.is_holdout = False
        self.i2.label = None
        self.i3.label = None
        self.i4.label = None
        self.i2.save(self.session, commit=True)
        self.i3.save(self.session, commit=True)
        self.i4.save(self.session, commit=True)
        data = load_validation_dataset()
        self.assertEqual(data, {})

    def test_returns_empty_dict_if_no_training_found(self):
        self.i2.is_training = False
        self.i4.is_training = False
        self.i4.is_holdout = False
        self.i2.label = None
        self.i4.label = None
        self.i2.save(self.session, commit=True)
        self.i4.save(self.session, commit=True)
        data = load_validation_dataset()
        self.assertEqual(data, {})

    def test_returns_empty_dict_non_matching_taxon(self):
        data = load_validation_dataset(taxon_id=0)
        self.assertEqual(data, {})

    def test_correct_selection(self):
        data = load_validation_dataset(selection=['pfam'])
        train_x = ['PF21']
        train_y = [[1]]
        test_x = ['PF31']
        test_y = [[1]]
        labels = ['Activation']

        self.assertEqual(train_x, list(data['training'][0]))
        self.assertEqual(list(data['training'][1][0]), train_y[0])

        self.assertEqual(test_x, list(data['testing'][0]))
        self.assertEqual(list(data['testing'][1][0]), test_y[0])

        self.assertEqual(labels, list(data['labels']))


class TestLoadDataset(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = Protein.query.get(self.protein_a.id)
        self.protein_b = Protein.query.get(self.protein_b.id)
        self.protein_c = Protein.query.get(self.protein_c.id)

        self.i1 = Interaction(  # interactome only
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label=None, go_mf='GO:11', go_bp='GO:12', go_cc='GO:13',
            interpro='IPR11', pfam='PF11'
        )
        self.i2 = Interaction(  # training only
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label='activation,inhibition', go_mf='GO:21', go_bp='GO:22',
            go_cc='GO:23', interpro='IPR21', pfam='PF21'
        )
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)
        self.interactions = [self.i1, self.i2]
        self.labels = ['Activation', 'Inhibition']

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_binarize_labels_if_labels_supplied(self):
        data = load_dataset(self.interactions, self.labels)
        X = ['GO11,GO13,GO12,IPR11,PF11', 'GO21,GO23,GO22,IPR21,PF21']
        y = [[0, 0], [1, 1]]
        self.assertEqual(X, list(data[0]))
        self.assertEqual(y[0], list(data[1][0]))
        self.assertEqual(y[1], list(data[1][1]))
        self.assertEqual(['Activation', 'Inhibition'], data[2].classes)

    def test_returns_none_if_no_interactions_supplied(self):
        data = load_dataset([])
        self.assertEqual(data, (None, None))

    def test_doesnt_binarize_labels_if_no_labels_supplied(self):
        data = load_dataset(self.interactions)
        X = ['GO11,GO13,GO12,IPR11,PF11', 'GO21,GO23,GO22,IPR21,PF21']
        self.assertEqual(X, list(data[0]))
        self.assertEqual(data[1][0], [])
        self.assertEqual(data[1][1], ['Activation', 'Inhibition'])

    def test_correct_selection(self):
        data = load_dataset(self.interactions, self.labels, selection=['pfam'])
        X = ['PF11', 'PF21']
        y = [[0, 0], [1, 1]]
        self.assertEqual(X, list(data[0]))
        self.assertEqual(y[0], list(data[1][0]))
        self.assertEqual(y[1], list(data[1][1]))
        self.assertEqual(['Activation', 'Inhibition'], data[2].classes)


class TestTrainPaperModel(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(session=self.session)

        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = Protein.query.get(self.protein_a.id)
        self.protein_b = Protein.query.get(self.protein_b.id)
        self.protein_c = Protein.query.get(self.protein_c.id)

        self.i1 = Interaction(  # interactome only
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=False, is_training=True, is_interactome=True,
            label='activation', go_mf='GO:11', go_bp='GO:12', go_cc='GO:13',
            interpro='IPR11', pfam='PF11'
        )
        self.i2 = Interaction(  # training only
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label='inhibition', go_mf='GO:21', go_bp='GO:22', go_cc='GO:23',
            interpro='IPR21', pfam='PF21'
        )
        self.i3 = Interaction(  # holdout only
            source=self.protein_c.id, target=self.protein_c.id,
            is_holdout=True, is_training=False, is_interactome=False,
            label='activation,inhibition', go_mf='GO:31', go_bp='GO:32',
            go_cc='GO:33', interpro='IPR31', pfam='PF31'
        )
        self.i4 = Interaction(  # holdout and training and interactome
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=True, is_training=True, is_interactome=True,
            label='inhibition', go_mf='GO:41', go_bp='GO:42',
            go_cc='GO:43', interpro='IPR41', pfam='PF41'
        )
        self.i1.save(self.session, commit=True)
        self.i2.save(self.session, commit=True)
        self.i3.save(self.session, commit=True)
        self.i4.save(self.session, commit=True)

    def tearDown(self):
        delete_database(session=self.session)
        cleanup_database(self.session, self.engine)

    def test_works(self):
        import warnings
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        with warnings.catch_warnings():
            clf, selection = train_paper_model(rcv_iter=10, rcv_splits=2)

    def test_raise_error_no_training_data(self):
        delete_database(self.session)
        with self.assertRaises(ValueError):
            train_paper_model(rcv_iter=10, rcv_splits=2)
