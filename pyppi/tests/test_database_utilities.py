
import os
import shutil
import pandas as pd

from unittest import TestCase
from Bio import SwissProt

from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm.exc import DetachedInstanceError

from ..base import SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE, NULL_VALUES

from ..database import begin_transaction, delete_database, make_session
from ..database.exceptions import ObjectAlreadyExists, ObjectNotFound
from ..database.utilities import (
    init_protein_table, generate_interaction_tuples,
    add_interaction, update_interaction, pmid_string_to_list,
    psimi_string_to_list
)
from ..database.models import (
    Protein, Interaction, Psimi, Pubmed,
    psimi_interactions, pmid_interactions
)

base_path = os.path.dirname(__file__)


class TestInitialisationMethod(TestCase):

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


class TestAddUpdateTrainingInteractions(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session = make_session(db_path=self.db_path)
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
        delete_database(self.session, db_path=self.db_path)
        self.session.close()

    def test_can_add_interaction(self):
        entry = add_interaction(
            session=self.session,
            commit=True,
            psimi_ls=(),
            pmid_ls=(),
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": True,
                "is_training": True,
                "is_holdout": True,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEqual(self.session.query(Interaction).count(), 1)
        self.assertEquals(entry.label, 'Activation')
        self.assertEquals(entry.go_mf, "GO:00")
        self.assertEquals(entry.go_bp, "GO:01")
        self.assertEquals(entry.go_cc, "GO:02")
        self.assertEquals(entry.ulca_go_mf, "GO:03")
        self.assertEquals(entry.ulca_go_bp, "GO:04")
        self.assertEquals(entry.ulca_go_cc, "GO:05")
        self.assertEquals(entry.interpro, "IPR1")
        self.assertEquals(entry.pfam, "PF1")
        self.assertEquals(entry.keywords, "Protein")
        self.assertTrue(entry.is_interactome)
        self.assertTrue(entry.is_training)
        self.assertTrue(entry.is_holdout)
        self.assertEquals(entry.pmid, [])
        self.assertEquals(entry.psimi, [])

    def test_error_if_interaction_exists_in_add_interaction(self):
        entry = add_interaction(
            session=self.session,
            commit=True,
            psimi_ls=(),
            pmid_ls=(),
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        with self.assertRaises(ObjectAlreadyExists):
            print("Here")
            add_interaction(
                session=self.session,
                commit=True,
                psimi_ls=(),
                pmid_ls=(),
                **{
                    "source": self.pa,
                    "target": self.pb,
                    "label": "activation",
                    "is_interactome": False,
                    "is_training": True,
                    "is_holdout": False,
                    'go_mf': "GO:00",
                    'go_bp': "GO:01",
                    'go_cc': "GO:02",
                    'ulca_go_mf': "GO:03",
                    'ulca_go_bp': "GO:04",
                    'ulca_go_cc': "GO:05",
                    'interpro': "IPR1",
                    'pfam': "PF1",
                    'keywords': "protein"
                }
            )

    def test_add_interaction_creates_psimi_references(self):
        entry = add_interaction(
            session=self.session,
            commit=True,
            psimi_ls=[self.psmi_a],
            pmid_ls=(),
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEqual(self.session.query(Interaction).count(), 1)
        self.assertEquals(entry.psimi, [self.psmi_a])

    def test_add_ineraction_creates_pmid_references(self):
        entry = add_interaction(
            session=self.session,
            commit=True,
            psimi_ls=(),
            pmid_ls=[self.pmid_a],
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEqual(self.session.query(Interaction).count(), 1)
        self.assertEquals(entry.pmid, [self.pmid_a])

    def test_update_interaction_creates_if_not_found(self):
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=True,
            psimi_ls=(),
            pmid_ls=(),
            override_boolean=True,
            create_if_not_found=True,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": True,
                "is_training": True,
                "is_holdout": True,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEqual(self.session.query(Interaction).count(), 1)
        self.assertEquals(entry.label, 'Activation')
        self.assertEquals(entry.go_mf, "GO:00")
        self.assertEquals(entry.go_bp, "GO:01")
        self.assertEquals(entry.go_cc, "GO:02")
        self.assertEquals(entry.ulca_go_mf, "GO:03")
        self.assertEquals(entry.ulca_go_bp, "GO:04")
        self.assertEquals(entry.ulca_go_cc, "GO:05")
        self.assertEquals(entry.interpro, "IPR1")
        self.assertEquals(entry.pfam, "PF1")
        self.assertEquals(entry.keywords, "Protein")
        self.assertTrue(entry.is_interactome)
        self.assertTrue(entry.is_training)
        self.assertTrue(entry.is_holdout)
        self.assertEquals(entry.pmid, [])
        self.assertEquals(entry.psimi, [])

    def test_update_error_if_not_found_and_create_is_false(self):
        with self.assertRaises(ObjectNotFound):
            entry = update_interaction(
                session=self.session,
                commit=True,
                replace_fields=True,
                psimi_ls=(),
                pmid_ls=(),
                override_boolean=True,
                create_if_not_found=False,
                **{
                    "source": self.pa,
                    "target": self.pb,
                    "label": "activation",
                    "is_interactome": False,
                    "is_training": True,
                    "is_holdout": False,
                    'go_mf': "GO:00",
                    'go_bp': "GO:01",
                    'go_cc': "GO:02",
                    'ulca_go_mf': "GO:03",
                    'ulca_go_bp': "GO:04",
                    'ulca_go_cc': "GO:05",
                    'interpro': "IPR1",
                    'pfam': "PF1",
                    'keywords': "protein"
                }
            )

    def test_update_replaces_fields(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label=None, go_mf="GO:002"
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=True,
            psimi_ls=(),
            pmid_ls=(),
            override_boolean=False,
            create_if_not_found=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": True,
                "is_training": True,
                "is_holdout": True,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEquals(entry.label, 'Activation')
        self.assertEquals(entry.go_mf, "GO:00")
        self.assertEquals(entry.go_bp, "GO:01")
        self.assertEquals(entry.go_cc, "GO:02")
        self.assertEquals(entry.ulca_go_mf, "GO:03")
        self.assertEquals(entry.ulca_go_bp, "GO:04")
        self.assertEquals(entry.ulca_go_cc, "GO:05")
        self.assertEquals(entry.interpro, "IPR1")
        self.assertEquals(entry.pfam, "PF1")
        self.assertEquals(entry.keywords, "Protein")
        self.assertTrue(entry.is_interactome)
        self.assertTrue(entry.is_training)
        self.assertTrue(entry.is_holdout)
        self.assertEquals(entry.pmid, [])
        self.assertEquals(entry.psimi, [])

    def test_update_replaces_references(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label=None
        )
        obj.save(self.session, commit=True)
        obj.add_pmid_reference(self.pmid_a)
        obj.add_psimi_reference(self.psmi_a)

        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=True,
            psimi_ls=[self.psmi_b],
            pmid_ls=[self.pmid_b],
            override_boolean=False,
            create_if_not_found=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEquals(entry.pmid, [self.pmid_b])
        self.assertEquals(entry.psimi, [self.psmi_b])

    def test_update_appends_new_label_if_replace_fields_is_false(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label="Phosphorylation"
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEquals(entry.label, 'Activation,Phosphorylation')

    def test_update_adds_label_if_current_label_is_None(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label=None
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEquals(entry.label, 'Activation')

    def test_update_does_nothing_if_new_label_is_none(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label='activation'
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": None,
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEquals(entry.label, 'Activation')

    def test_update_overrides_boolean_fields(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=True, is_holdout=True,
            is_training=True, label="Activation"
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            psimi_ls=(),
            pmid_ls=(),
            replace_fields=False,
            override_boolean=True,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": None,
                "is_interactome": False,
                "is_training": False,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertFalse(entry.is_interactome)
        self.assertFalse(entry.is_training)
        self.assertFalse(entry.is_holdout)

    def test_update_uses_or_to_update_boolean_when_not_override(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=True,
            is_training=False, label="Activation"
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            psimi_ls=(),
            pmid_ls=(),
            replace_fields=False,
            override_boolean=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": None,
                "is_interactome": True,
                "is_training": False,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertTrue(entry.is_interactome)
        self.assertFalse(entry.is_training)
        self.assertTrue(entry.is_holdout)

    def test_update_appends_annotations_if_not_replace(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label=None,
            go_mf="GO:01", go_bp="GO:02", go_cc="GO:03",
            ulca_go_mf="GO:01", ulca_go_bp="GO:02", ulca_go_cc='GO:03',
            interpro="IPR1", pfam=None, keywords="hello",
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=False,
            update_features=True,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": True,
                "is_training": True,
                "is_holdout": True,
                'go_mf': "GO:01",
                'go_bp': "GO:22",
                'go_cc': "GO:33",
                'ulca_go_mf': "GO:11",
                'ulca_go_bp': "GO:22",
                'ulca_go_cc': None,
                'interpro': "IPR2",
                'pfam': "PF2",
                'keywords': "world"
            }
        )
        self.assertEquals(entry.go_mf, "GO:01,GO:01")
        self.assertEquals(entry.go_bp, "GO:02,GO:22")
        self.assertEquals(entry.go_cc, "GO:03,GO:33")
        self.assertEquals(entry.ulca_go_mf, "GO:01,GO:11")
        self.assertEquals(entry.ulca_go_bp, "GO:02,GO:22")
        self.assertEquals(entry.ulca_go_cc, "GO:03")
        self.assertEquals(entry.interpro, "IPR1,IPR2")
        self.assertEquals(entry.pfam, "PF2")
        self.assertEquals(entry.keywords, "Hello,World")

    def test_update_ignores_features_if_update_feature_is_false(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label=None,
            go_mf="GO:01", go_bp="GO:02", go_cc="GO:03",
            ulca_go_mf="GO:01", ulca_go_bp="GO:02", ulca_go_cc='GO:03',
            interpro="IPR1", pfam="PF1", keywords="hello",
        )
        obj.save(self.session, commit=True)
        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=False,
            update_features=False,
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": True,
                "is_training": True,
                "is_holdout": True,
                'go_mf': "GO:11",
                'go_bp': "GO:22",
                'go_cc': "GO:33",
                'ulca_go_mf': "GO:11",
                'ulca_go_bp': "GO:22",
                'ulca_go_cc': "GO:33",
                'interpro': "IPR2",
                'pfam': "PF2",
                'keywords': "world"
            }
        )
        self.assertEquals(entry.go_mf, "GO:01")
        self.assertEquals(entry.go_bp, "GO:02")
        self.assertEquals(entry.go_cc, "GO:03")
        self.assertEquals(entry.ulca_go_mf, "GO:01")
        self.assertEquals(entry.ulca_go_bp, "GO:02")
        self.assertEquals(entry.ulca_go_cc, "GO:03")
        self.assertEquals(entry.interpro, "IPR1")
        self.assertEquals(entry.pfam, "PF1")
        self.assertEquals(entry.keywords, "Hello")

    def test_update_appends_new_references(self):
        obj = Interaction(
            source=self.pa, target=self.pb,
            is_interactome=False, is_holdout=False,
            is_training=False, label=None
        )
        obj.save(self.session, commit=True)
        obj.add_pmid_reference(self.pmid_a)
        obj.add_psimi_reference(self.psmi_a)

        entry = update_interaction(
            session=self.session,
            commit=True,
            replace_fields=False,
            psimi_ls=[self.psmi_b],
            pmid_ls=[self.pmid_b],
            **{
                "source": self.pa,
                "target": self.pb,
                "label": "activation",
                "is_interactome": False,
                "is_training": True,
                "is_holdout": False,
                'go_mf': "GO:00",
                'go_bp': "GO:01",
                'go_cc': "GO:02",
                'ulca_go_mf': "GO:03",
                'ulca_go_bp': "GO:04",
                'ulca_go_cc': "GO:05",
                'interpro': "IPR1",
                'pfam': "PF1",
                'keywords': "protein"
            }
        )
        self.assertEquals(
            list(sorted(entry.pmid, key=lambda x: x.accession)),
            [self.pmid_a, self.pmid_b]
        )
        self.assertEquals(
            list(sorted(entry.psimi, key=lambda x: x.accession)),
            [self.psmi_a, self.psmi_b]
        )


class TestParsingAndGenerator(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session = make_session(db_path=self.db_path)
        self.psi_a = Psimi(accession="MI:1", description="blah")
        self.psi_b = Psimi(accession="MI:2", description="blah")
        self.pm_a = Pubmed(accession="1")
        self.pm_b = Pubmed(accession="2")
        self.psi_a.save(self.session, commit=True)
        self.psi_b.save(self.session, commit=True)
        self.pm_a.save(self.session, commit=True)
        self.pm_b.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session, db_path=self.db_path)
        self.session.close()

    def test_pmid_to_list_returns_existing_items(self):
        result = pmid_string_to_list(self.session, "1,2")
        self.assertEquals(result, [self.pm_a, self.pm_b])

    def test_pmid_to_list_raises_error_if_not_found(self):
        with self.assertRaises(ObjectNotFound):
            pmid_string_to_list(self.session, "3")

    def test_pmid_to_list_returns_empty_if_null_value_pass(self):
        for value in NULL_VALUES:
            result = pmid_string_to_list(self.session, value)
            self.assertEquals(result, [])

    def test_psimi_to_list_returns_existing_items(self):
        result = psimi_string_to_list(self.session, "MI:1,MI:2")
        self.assertEquals(result, [self.psi_a, self.psi_b])

    def test_psimi_to_list_returns_empty_if_null_value_pass(self):
        for value in NULL_VALUES:
            result = psimi_string_to_list(self.session, value)
            self.assertEquals(result, [])

    def test_psimi_to_list_raises_error_if_not_found(self):
        with self.assertRaises(ObjectNotFound):
            psimi_string_to_list(self.session, "MI:3")

    def test_can_generate_tuples(self):
        df = pd.DataFrame({
            SOURCE: ['A', 'B'],
            TARGET: ['A', 'B'],
            LABEL: ['1', '2'],
            PUBMED: ['01', '02'],
            EXPERIMENT_TYPE: ['MI:1', 'MI:2']
        })
        result = list(generate_interaction_tuples(df))
        self.assertEqual(result[0], ('A', 'A', '1', '01', 'MI:1'))
        self.assertEqual(result[1], ('B', 'B', '2', '02', 'MI:2'))

    def test_generate_tuples_formats_none(self):
        for value in NULL_VALUES:
            df = pd.DataFrame({
                SOURCE: ['A'],
                TARGET: ['A'],
                LABEL: [value],
                PUBMED: [value],
                EXPERIMENT_TYPE: [value]
            })
            result = list(generate_interaction_tuples(df))
            self.assertEqual(result[0], ('A', 'A', None, None, None))

    def test_generate_tuples_raises_error_none_type_source(self):
        for value in NULL_VALUES:
            df = pd.DataFrame({
                SOURCE: [value],
                TARGET: ['A'],
                LABEL: ['1'],
                PUBMED: ['2'],
                EXPERIMENT_TYPE: ['3']
            })
            with self.assertRaises(ValueError):
                list(generate_interaction_tuples(df))

    def test_generate_tuples_raises_error_none_type_target(self):
        for value in NULL_VALUES:
            df = pd.DataFrame({
                SOURCE: ['A'],
                TARGET: [value],
                LABEL: ['1'],
                PUBMED: ['2'],
                EXPERIMENT_TYPE: ['3']
            })
            with self.assertRaises(ValueError):
                list(generate_interaction_tuples(df))
