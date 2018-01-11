import os
from unittest import TestCase
from Bio import SwissProt

from ..database import begin_transaction
from ..database.models import Protein
from ..data_mining.features import compute_interaction_features
from ..data_mining.ontology import get_active_instance
from ..data_mining.uniprot import parse_record_into_protein


base_path = os.path.dirname(__file__)
test_obo_file = '{}/{}'.format(base_path, "test_data/test_go.obo.gz")
dag = get_active_instance(filename=test_obo_file)


class TestComputeInteractionFeatures(TestCase):

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
        self.records.close()

    def test_can_compute_features_for_source_target(self):
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            break

        # Same examples used in test_ontology
        protein.go_mf = ["GO:0001618"]
        protein.go_bp = ["GO:0007154", "GO:0050794"]
        protein.go_cc = ["GO:0016459"]

        with begin_transaction(db_path=self.db_path) as session:
            protein.save(session, commit=True)
            protein = session.query(Protein).get(protein.id)  # Refresh

        features = compute_interaction_features(protein, protein)
        go_mf = features['go_mf']
        go_bp = features['go_bp']
        go_cc = features['go_cc']
        ulca_go_mf = features['ulca_go_mf']
        ulca_go_bp = features['ulca_go_bp']
        ulca_go_cc = features['ulca_go_cc']
        interpro = features['interpro']
        pfam = features['pfam']
        keywords = features['keywords']

        # Regular go features
        self.assertEqual(
            sorted(go_mf),
            sorted(protein.go_mf.split(',') + protein.go_mf.split(','))
        )
        self.assertEqual(
            sorted(go_bp),
            sorted(protein.go_bp.split(',') + protein.go_bp.split(','))
        )
        self.assertEqual(
            sorted(go_cc),
            sorted(protein.go_cc.split(',') + protein.go_cc.split(','))
        )

        # Other features
        self.assertEqual(
            sorted(interpro),
            sorted(protein.interpro.split(',') +
                   protein.interpro.split(','))
        )
        self.assertEqual(
            sorted(pfam),
            sorted(protein.pfam.split(',') + protein.pfam.split(','))
        )
        self.assertEqual(
            sorted(keywords),
            sorted(protein.keywords.split(',') +
                   protein.keywords.split(','))
        )

        # Incuded go features
        self.assertEqual(
            sorted(ulca_go_bp),
            sorted([
                "GO:0008150", "GO:0008150",
                "GO:0065007", "GO:0065007",
                "GO:0050789", "GO:0050789",
                "GO:0050794", "GO:0050794",
                "GO:0007154", "GO:0007154",
                "GO:0009987", "GO:0009987",

                # induced from mf part_of relationship
                "GO:0046718", "GO:0046718"
            ])
        )
        self.assertEqual(
            sorted(ulca_go_mf),
            sorted([
                "GO:0001618", "GO:0001618"
            ])
        )
        self.assertEqual(
            sorted(ulca_go_cc),
            sorted([
                "GO:0016459", "GO:0016459",
                "GO:0015629", "GO:0015629",
                "GO:0044430", "GO:0044430"
            ])
        )

    def test_compute_features_returns_None_if_source_is_None(self):
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            break

        with begin_transaction(db_path=self.db_path) as session:
            protein.save(session, commit=True)
            protein = session.query(Protein).get(protein.id)  # Refresh
            self.assertIsNone(compute_interaction_features(None, protein))

    def test_compute_features_returns_None_if_target_is_None(self):
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            break

        with begin_transaction(db_path=self.db_path) as session:
            protein.save(session, commit=True)
            protein = session.query(Protein).get(protein.id)  # Refresh
            self.assertIsNone(compute_interaction_features(protein, None))

    def test_compute_features_return_empty_list_if_features_are_empty(self):
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            break

        protein.go_mf = None
        protein.go_bp = None
        protein.go_cc = None
        protein.interpro = None
        protein.pfam = None
        protein.keywords = None

        with begin_transaction(db_path=self.db_path) as session:
            protein.save(session, commit=True)
            protein = session.query(Protein).get(protein.id)  # Refresh

        features = compute_interaction_features(protein, protein)
        expected = dict(
            go_mf=[], go_bp=[], go_cc=[],
            ulca_go_mf=[], ulca_go_bp=[], ulca_go_cc=[],
            interpro=[], pfam=[], keywords=[]
        )
        self.assertEqual(expected, features)

    def test_compute_features_maps_to_alts_to_stable_ontology_terms(self):
        for record in SwissProt.parse(self.records):
            protein = parse_record_into_protein(record)
            break

        protein.go_mf = 'GO:0000975'
        protein.go_bp = None
        protein.go_cc = None
        protein.interpro = None
        protein.pfam = None
        protein.keywords = None

        features = compute_interaction_features(protein, protein)
        self.assertEqual(features['go_mf'], ['GO:0044212', 'GO:0044212'])
        self.assertTrue('GO:0000975' not in features['ulca_go_mf'])
