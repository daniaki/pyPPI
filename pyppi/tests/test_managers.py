
import os

from unittest import TestCase
from Bio import SwissProt

from ..database import make_session, delete_database
from ..database.models import Protein, Interaction
from ..database.exceptions import ObjectAlreadyExists, ObjectNotFound
from ..database.managers import ProteinManager, InteractionManager
from ..database.managers import format_interactions_for_sklearn
from ..data_mining.uniprot import parse_record_into_protein
from ..data_mining.ontology import parse_obo12_file

base_path = os.path.dirname(__file__)
test_obo_file = '{}/{}'.format(base_path, "test_data/test_go.obo.gz")
dag = parse_obo12_file(test_obo_file)


class TestProteinManager(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.records = open(os.path.normpath(
            "{}/test_data/test_sprot_records.dat".format(base_path)
        ), 'rt')

        self.session = make_session(db_path=self.db_path)
        delete_database(session=self.session)

        self.manager = ProteinManager(match_taxon_id=9606, verbose=False)
        self.protein = Protein("A", taxon_id=9606, reviewed=False)
        self.protein.save(self.session, commit=True)
        self.protein = self.session.query(Protein).get(self.protein.id)

    def tearDown(self):
        delete_database(session=self.session)
        self.session.close()
        self.records.close()

    def test_get_by_uniprot_id_works(self):
        entry = self.manager.get_by_uniprot_id(self.session, "A")
        self.assertEqual(entry, self.protein)

    def test_get_by_id_works(self):
        entry = self.manager.get_by_id(self.session, self.protein.id)
        self.assertEqual(entry, self.protein)

    def test_get_by_id_returns_none_if_id_does_not_exist(self):
        entry = self.manager.get_by_id(self.session, 99)
        self.assertTrue(entry is None)

    def test_get_by_id_raises_typeerror_if_id_not_int(self):
        with self.assertRaises(TypeError):
            self.manager.get_by_id(self.session, '99')

    def test_get_by_uniprot_id_returns_None_non_matching_taxon_id(self):
        new_protein = Protein("B", taxon_id=0, reviewed=False)
        new_protein.save(self.session, commit=True)
        result = self.manager.get_by_uniprot_id(self.session, "B")
        self.assertTrue(result is None)

    def test_correctly_return_dict_from_entry(self):
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            break
        protein.save(self.session, commit=True)
        result = self.manager.entry_to_dict(
            self.session, uniprot_id="P31946", split=True
        )

        expected = {
            'go_bp': [
                'GO:0000165', 'GO:0006605', 'GO:0016032', 'GO:0035308',
                'GO:0035329', 'GO:0043085', 'GO:0043488', 'GO:0045744',
                'GO:0045892', 'GO:0051220', 'GO:0051291', 'GO:0061024',
                'GO:1900740'
            ],
            'reviewed': True,
            'keywords': [
                '3d-structure', 'Acetylation', 'Alternative initiation',
                'Complete proteome', 'Cytoplasm', 'Direct protein sequencing',
                'Host-virus interaction', 'Isopeptide bond', 'Nitration',
                'Phosphoprotein', 'Polymorphism', 'Reference proteome',
                'Ubl conjugation'
            ],
            'gene_id': 'YWHAB',
            'interpro': ['IPR000308', 'IPR023409', 'IPR023410', 'IPR036815'],
            'uniprot_id': 'P31946',
            'go_cc': [
                'GO:0005634', 'GO:0005737', 'GO:0005739', 'GO:0005829',
                'GO:0005925', 'GO:0016020', 'GO:0017053', 'GO:0030659',
                'GO:0042470', 'GO:0043234', 'GO:0048471', 'GO:0070062'
            ],
            'pfam': ['PF00244'],
            'go_mf': [
                'GO:0003714', 'GO:0008022', 'GO:0019899', 'GO:0019904',
                'GO:0032403', 'GO:0042802', 'GO:0042826', 'GO:0045296',
                'GO:0050815', 'GO:0051219'
            ],
            "taxon_id": 9606,
            "id": 2
        }
        self.assertEqual(result, expected)

    def test_protein_to_dict_returns_empty_dict_when_entry_not_found(self):
        self.manager.allow_download = False
        result = self.manager.entry_to_dict(self.session, id=0, split=True)
        self.assertEqual(result, {})

    def test_entry_to_dict_raises_valueerror_if_neither_supplied(self):
        with self.assertRaises(ValueError):
            self.manager.entry_to_dict(self.session, split=True)


class TestInteractionManager(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.records = open(os.path.normpath(
            "{}/test_data/test_sprot_records.dat".format(base_path)
        ), 'rt')

        self.session = make_session(db_path=self.db_path)
        delete_database(session=self.session)

        self.i_manager = InteractionManager(verbose=False, match_taxon_id=9606)
        self.protein_a = Protein(uniprot_id="A", taxon_id=9606, reviewed=True)
        self.protein_b = Protein(uniprot_id="B", taxon_id=9606, reviewed=True)
        self.protein_c = Protein(uniprot_id="C", taxon_id=9606, reviewed=True)
        self.protein_a.save(self.session, commit=True)
        self.protein_b.save(self.session, commit=True)
        self.protein_c.save(self.session, commit=True)
        self.protein_a = self.session.query(Protein).get(self.protein_a.id)
        self.protein_b = self.session.query(Protein).get(self.protein_b.id)
        self.protein_c = self.session.query(Protein).get(self.protein_c.id)

    def tearDown(self):
        delete_database(session=self.session)
        self.session.close()
        self.records.close()

    def test_get_protein_can_correctly_delegate_and_find_entry(self):
        result = self.i_manager._get_protein(self.session, 'A', 'source')
        self.assertEqual(result, self.protein_a)

        result = self.i_manager._get_protein(self.session, 1, 'source')
        self.assertEqual(result, self.protein_a)

        result = self.i_manager._get_protein(self.session, 99, 'source')
        self.assertTrue(result is None)

        result = self.i_manager._get_protein(
            self.session, self.protein_a, 'source'
        )
        self.assertEqual(result, self.protein_a)

        with self.assertRaises(TypeError):
            self.i_manager._get_protein(self.session, [], 'source')

    def test_filter_taxonid_filters_non_matching_interactions(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        self.i_manager.match_taxon_id = 0
        obj.save(self.session, commit=True)
        qs = self.session.query(Interaction).filter(Interaction.id == obj.id)
        result = self.i_manager.filter_matching_taxon_ids(qs)
        self.assertEqual(result.count(), 0)

    def test_filter_taxonid_keeps_matching_interactions(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        qs = self.session.query(Interaction).filter(Interaction.id == obj.id)
        result = self.i_manager.filter_matching_taxon_ids(qs)
        self.assertEqual(result.count(), 1)

    def test_filter_taxonid_keeps_interactions_when_match_taxon_id_None(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        self.i_manager.match_taxon_id = None
        obj.save(self.session, commit=True)
        qs = self.session.query(Interaction).filter(Interaction.id == obj.id)
        result = self.i_manager.filter_matching_taxon_ids(qs)
        self.assertEqual(result.count(), 1)

    def test_can_get_by_source_target(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_a.id,
            target=self.protein_b.id
        )
        self.assertEqual(result, obj)

    def test_can_get_by_source_target_when_objects_are_passed_in(self):
        obj = Interaction(
            source=self.protein_a, target=self.protein_b,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_a.id,
            target=self.protein_b.id
        )
        self.assertEqual(result, obj)

    def test_can_get_by_source_target_return_none_not_found(self):
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_a.id,
            target=self.protein_b.id
        )
        self.assertIs(result, None)

    def test_get_by_source_target_returns_none_if_source_not_found(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_b.id,
            target=self.protein_b.id
        )
        self.assertTrue(result is None)

    def test_get_by_source_target_returns_none_if_target_not_found(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_b.id,
            target=self.protein_a.id
        )
        self.assertTrue(result is None)

    def test_get_by_source_target_returns_none_if_taxonid_not_matching(self):
        self.i_manager.match_taxon_id = 0
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_a.id,
            target=self.protein_b.id
        )
        self.assertTrue(result is None)

    def test_get_by_source_target_returns_doesnt_filter_if_taxon_id_None(self):
        self.i_manager.match_taxon_id = None
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.get_by_source_target(
            session=self.session, source=self.protein_a.id,
            target=self.protein_b.id
        )
        self.assertEqual(result.id, obj.id)

    def test_can_get_all_interactions_by_label(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False,
            label="activation,phosphorylation"
        )
        obj2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_training=False, is_holdout=False, label="activation",
            is_interactome=False
        )
        obj3 = Interaction(
            source=self.protein_b.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )

        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        obj3.save(self.session, commit=True)

        results = self.i_manager.get_by_label(self.session, 'activation')
        self.assertEqual(results.count(), 1)
        self.assertEqual(results[0], obj2)

        results = self.i_manager.get_by_label(
            self.session, 'activation,phosphorylation')
        self.assertEqual(results.count(), 1)
        self.assertEqual(results[0], obj1)

        results = self.i_manager.get_by_label(self.session, None)
        self.assertEqual(results.count(), 1)
        self.assertEqual(results[0], obj3)

    def test_get_by_label_returns_empty_qset_if_label_not_found(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False,
            label="activation,phosphorylation"
        )
        obj.save(self.session, commit=True)
        results = self.i_manager.get_by_label(self.session, 'Lalaru')
        self.assertEqual(results.count(), 0)

    def test_get_by_label_returns_empty_list_if_taxonid_not_matching(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False,
            label="activation"
        )
        self.i_manager.match_taxon_id = 0
        obj.save(self.session, commit=True)
        results = self.i_manager.get_by_label(self.session, 'activation')
        self.assertEqual(results.count(), 0)

    def test_can_get_interactions_containing_label_substring(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False,
            label="activation,phosphorylation"
        )
        obj.save(self.session, commit=True)
        results = self.i_manager.get_contains_label(
            self.session, 'phosphorylation')
        self.assertEqual(results.count(), 1)
        self.assertEqual(results[0], obj)

    def test_contains_label_returns_empty_list_if_taxonid_not_matching(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False,
            label="activation,phosphorylation"
        )
        self.i_manager.match_taxon_id = 0
        obj.save(self.session, commit=True)
        results = self.i_manager.get_contains_label(
            self.session, 'phosphorylation')
        self.assertEqual(results.count(), 0)

    def test_get_by_source_returns_all_interactions_matching_source(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        obj2 = Interaction(
            source=self.protein_b.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        results = self.i_manager.get_by_source(self.session, self.protein_a)
        self.assertEqual(results.count(), 1)

    def test_get_by_source_returns_empty_list_non_matching_taxonids(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = 0
        obj.save(self.session, commit=True)
        results = self.i_manager.get_by_source(self.session, self.protein_a)
        self.assertEqual(results.count(), 0)

    def test_get_by_source_ignore_taxonid_when_set_to_none(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = None
        obj.save(self.session, commit=True)
        results = self.i_manager.get_by_source(self.session, self.protein_a)
        self.assertEqual(results.count(), 1)

    def test_get_by_target_ignore_taxonid_when_set_to_none(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = None
        obj.save(self.session, commit=True)
        results = self.i_manager.get_by_target(self.session, self.protein_b)
        self.assertEqual(results.count(), 1)

    def test_get_by_target_returns_all_interactions_matching_source(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        obj2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False

        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        results = self.i_manager.get_by_target(self.session, self.protein_b)
        self.assertEqual(results.count(), 1)

    def test_get_by_target_returns_empty_list_non_matching_taxonids(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = 0
        obj.save(self.session, commit=True)
        results = self.i_manager.get_by_target(self.session, self.protein_b)
        self.assertEqual(results.count(), 0)

    def test_taxonids_match_can_detect_non_matching_taxon_ids(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = 1
        self.assertFalse(self.i_manager.taxon_ids_match(obj))

    def test_taxonids_match_ignores_taxonid_if_none(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = None
        self.assertTrue(self.i_manager.taxon_ids_match(obj))

    def test_taxonids_match_can_detect_matching_taxon_ids(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, label=None,
            is_interactome=False
        )
        self.i_manager.match_taxon_id = None
        self.assertTrue(self.i_manager.taxon_ids_match(obj))

    def test_get_by_id_works(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False
        )
        obj.save(self.session, commit=True)
        entry = self.i_manager.get_by_id(self.session, obj.id)
        self.assertEqual(entry, obj)

    def test_get_by_id_works_when_match_taxon_is_none(self):
        self.i_manager.match_taxon_id = None
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False
        )
        obj.save(self.session, commit=True)
        entry = self.i_manager.get_by_id(self.session, obj.id)
        self.assertEqual(entry, obj)

    def test_get_by_id_matches_taxonid(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False
        )
        self.i_manager.match_taxon_id = 0
        obj.save(self.session, commit=True)
        entry = self.i_manager.get_by_id(self.session, obj.id)
        self.assertIs(entry, None)

    def test_get_by_id_returns_none_if_id_does_not_exist(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False
        )
        obj.save(self.session, commit=True)
        entry = self.i_manager.get_by_id(self.session, 99)
        self.assertIs(entry, None)

    def test_get_by_id_raises_typeerror_if_id_not_int(self):
        with self.assertRaises(TypeError):
            self.i_manager.get_by_id(self.session, '99')

    def test_correctly_return_dict_from_entry(self):
        obj = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_training=False, is_holdout=False, is_interactome=False
        )
        obj.save(self.session, commit=True)
        result = self.i_manager.entry_to_dict(self.session, id=1, split=True)
        expected = {
            'go_bp': None,
            'go_cc': None,
            'go_mf': None,
            'id': 1,
            'interpro': None,
            'is_holdout': False,
            'is_training': False,
            'is_interactome': False,
            'keywords': None,
            'label': None,
            'pfam': None,
            'source': 'A',
            'target': 'B',
            'ulca_go_bp': None,
            'ulca_go_cc': None,
            'ulca_go_mf': None,
            "pmid": [],
            "psimi": []
        }
        self.assertEqual(result, expected)

    def test_protein_to_dict_returns_empty_dict_when_entry_not_found(self):
        result = self.i_manager.entry_to_dict(self.session, id=0, split=True)
        self.assertEqual(result, {})

    def test_entry_to_dict_raises_valueerror_if_neither_id_st_supplied(self):
        with self.assertRaises(ValueError):
            self.i_manager.entry_to_dict(self.session, split=True)

    def test_can_filter_training_interactions(self):
        obj1 = Interaction(
            source=self.protein_a.id,
            target=self.protein_b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False
        )
        obj2 = Interaction(
            source=self.protein_b.id,
            target=self.protein_b.id,
            is_holdout=False,
            is_training=True,
            is_interactome=False,
            label="activation"
        )
        obj3 = Interaction(
            source=self.protein_a.id,
            target=self.protein_a.id,
            is_holdout=True,
            is_training=True,
            is_interactome=False,
            label="activation"
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        obj3.save(self.session, commit=True)

        result = self.i_manager.training_interactions(
            self.session, keep_holdout=False
        )
        self.assertEqual(result.count(), 1)
        self.assertEqual(result[0], obj2)

        result = self.i_manager.training_interactions(
            self.session, keep_holdout=True
        )
        self.assertEqual(result.count(), 2)
        self.assertEqual(result[0], obj2)
        self.assertEqual(result[1], obj3)

    def test_can_filter_holdout_interactions(self):
        obj1 = Interaction(
            source=self.protein_a.id,
            target=self.protein_b.id,
            is_holdout=False,
            is_training=False,
            is_interactome=False
        )
        obj2 = Interaction(
            source=self.protein_b.id,
            target=self.protein_b.id,
            is_holdout=True,
            is_training=False,
            is_interactome=False,
            label="activation"
        )
        obj3 = Interaction(
            source=self.protein_a.id,
            target=self.protein_a.id,
            is_holdout=True,
            is_training=True,
            is_interactome=False,
            label="activation"
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        obj3.save(self.session, commit=True)

        result = self.i_manager.holdout_interactions(
            self.session, keep_training=False
        )
        self.assertEqual(result.count(), 1)
        self.assertEqual(result[0], obj2)

        result = self.i_manager.holdout_interactions(
            self.session, keep_training=True
        )
        self.assertEqual(result.count(), 2)
        self.assertEqual(result[0], obj2)
        self.assertEqual(result[1], obj3)

    def test_can_filter_interactome_interactions(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=False, is_training=False, is_interactome=True
        )
        obj2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=True, is_training=False, is_interactome=True,
            label="activation"
        )
        obj3 = Interaction(
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=True,
            label="activation"
        )
        obj4 = Interaction(
            source=self.protein_c.id, target=self.protein_c.id,
            is_holdout=False, is_training=False, is_interactome=False,
            label="activation"
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        obj3.save(self.session, commit=True)
        obj4.save(self.session, commit=True)

        result = self.i_manager.interactome_interactions(
            self.session, keep_holdout=False, keep_training=False
        )
        self.assertEqual(result.count(), 1)
        self.assertEqual(result[0], obj1)

        result = self.i_manager.interactome_interactions(
            self.session, keep_holdout=True, keep_training=False
        )
        self.assertEqual(result.count(), 2)
        self.assertEqual(result[0], obj1)
        self.assertEqual(result[1], obj2)

        result = self.i_manager.interactome_interactions(
            self.session, keep_holdout=False, keep_training=True
        )
        self.assertEqual(result.count(), 2)
        self.assertEqual(result[0], obj1)
        self.assertEqual(result[1], obj3)

    def test_can_format_interactions_for_sklearn(self):
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

        X, y = format_interactions_for_sklearn(
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
            'GO:1,GO:2,IPR4,PF5,GO:6,GO:7,GO:8',
            'GO:1,GO:2,GO:3,PF5,GO:6,GO:7,GO:8',
        ]
        self.assertEqual(result_x, list(X))
        self.assertEqual(result_y, y)

    def test_can_get_all_labels_from_training_instances(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label='methylation'
        )
        obj2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=True, is_training=True, is_interactome=False,
            label="activation,phosphorylation"
        )
        obj3 = Interaction(
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label="activation,dephosphorylation,"
        )
        obj4 = Interaction(
            source=self.protein_c.id, target=self.protein_c.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label='inhibition'
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        obj3.save(self.session, commit=True)
        obj4.save(self.session, commit=True)

        labels = list(sorted(self.i_manager.training_labels(
            self.session, include_holdout=False
        )))
        expected = list(
            sorted(['Activation', 'Dephosphorylation', 'Inhibition'])
        )
        self.assertEqual(labels, expected)

    def test_get_labels_from_training_instances_includes_holdout(self):
        obj1 = Interaction(
            source=self.protein_a.id, target=self.protein_b.id,
            is_holdout=False, is_training=False, is_interactome=True,
            label='methylation'
        )
        obj2 = Interaction(
            source=self.protein_a.id, target=self.protein_a.id,
            is_holdout=True, is_training=True, is_interactome=False,
            label="activation,phosphorylation"
        )
        obj3 = Interaction(
            source=self.protein_b.id, target=self.protein_b.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label="activation,dephosphorylation,"
        )
        obj4 = Interaction(
            source=self.protein_c.id, target=self.protein_c.id,
            is_holdout=False, is_training=True, is_interactome=False,
            label='inhibition'
        )
        obj1.save(self.session, commit=True)
        obj2.save(self.session, commit=True)
        obj3.save(self.session, commit=True)
        obj4.save(self.session, commit=True)

        labels = list(sorted(self.i_manager.training_labels(
            self.session, include_holdout=True
        )))
        expected = list(
            sorted(['Activation', 'Dephosphorylation',
                    'Inhibition', 'Phosphorylation'])
        )
        self.assertEqual(labels, expected)
