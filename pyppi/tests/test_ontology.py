import os
from unittest import TestCase

from ..data_mining.ontology import get_active_instance
from ..data_mining.ontology import (
    get_up_to_lca,
    get_lca_of_terms,
    group_terms_by_ontology_type,
    filter_obsolete_terms,
    parse_obo12_file
)

base_path = os.path.dirname(__file__)
test_obo_file = '{}/{}'.format(base_path, "test_data/test_go.obo.gz")


class TestULCAInducer(TestCase):

    def setUp(self):
        self.dag = parse_obo12_file(test_obo_file)

    def test_ulca_inducer_case_1(self):
        expected = [
            "GO:0008150", "GO:0008150",
            "GO:0009987", "GO:0009987",
            "GO:0065007", "GO:0065007",
            "GO:0050896",
            "GO:0050789", "GO:0050789",
            "GO:0007154",
            "GO:0050794", "GO:0050794",
            "GO:0051716"
        ]
        p1 = ["GO:0007154", "GO:0050794"]
        p2 = ["GO:0050794", "GO:0051716"]

        induced = get_up_to_lca(p1, p2)
        self.assertEqual(list(sorted(expected)), list(sorted(induced)))

    def test_ulca_inducer_case_2(self):
        expected = [
            "GO:0008150", "GO:0008150",
            "GO:0009987",
            "GO:0065007",
            "GO:0050789",
            "GO:0050794",
            "GO:0007154"
        ]
        p1 = ["GO:0007154"]
        p2 = ["GO:0050794"]

        induced = get_up_to_lca(p1, p2)
        self.assertEqual(list(sorted(expected)), list(sorted(induced)))

    def test_lca_returns_original_input_if_no_common_ancestor(self):
        expected = ["GO:0008150", "GO:0008150"]
        p1 = ["GO:0008150"]
        p2 = ["GO:0008150"]

        induced = get_up_to_lca(p1, p2)
        self.assertEqual(list(sorted(expected)), list(sorted(induced)))

    def test_lca_returns_original_input_if_empty_input(self):
        expected = ["GO:0008150"]
        p1 = ["GO:0008150"]
        p2 = []

        induced = get_up_to_lca(p1, p2)
        self.assertEqual(list(sorted(expected)), list(sorted(induced)))

        expected = ["GO:0008150"]
        p1 = []
        p2 = ["GO:0008150"]

        induced = get_up_to_lca(p1, p2)
        self.assertEqual(list(sorted(expected)), list(sorted(induced)))

    def test_can_parse_obo_file(self):
        self.assertEqual(len(self.dag), 49209)
        expected = str({
            "id": "GO:0007165",
            "name": 'signal transduction',
            "namespace": 'biological_process',
            "is_a": sorted(['GO:0009987', 'GO:0050794']),
            "part_of": sorted(['GO:0051716', 'GO:0023052', 'GO:0007154']),
            "is_obsolete": False
        })
        self.assertEqual(str(self.dag["GO:0007165"]), expected)

    def test_can_parse_all_is_a_entries(self):
        self.assertEqual(len(self.dag), 49209)
        expected = sorted(['GO:0009987', 'GO:0050794'])
        result = sorted([t.id for t in self.dag["GO:0007165"].is_a])
        self.assertEqual(result, expected)

    def test_can_parse_all_part_of_entries(self):
        self.assertEqual(len(self.dag), 49209)
        expected = sorted(['GO:0051716', 'GO:0023052', 'GO:0007154'])
        result = sorted([t.id for t in self.dag["GO:0007165"].part_of])
        self.assertEqual(result, expected)

    def test_parser_correctly_sets_obsolete_status(self):
        self.assertTrue(self.dag["GO:0030939"].is_obsolete)

    def test_goterm_hash_is_has_of_id(self):
        term = self.dag["GO:0030939"]
        self.assertEqual(hash(term), hash("GO:0030939"))

    def test_has_parent(self):
        term = self.dag["GO:0007165"]
        self.assertTrue(term.has_parent(self.dag["GO:0007154"]))

    def test_has_ancestor(self):
        term = self.dag["GO:0007165"]
        self.assertTrue(term.has_ancestor(self.dag["GO:0008150"]))

    def test_can_get_all_parents_of_a_term(self):
        term = self.dag["GO:0007165"]
        result = sorted([t.id for t in term.all_parents])
        expected = sorted([
            "GO:0007154",
            "GO:0050789",
            "GO:0008150",
            "GO:0009987",
            "GO:0051716",
            "GO:0023052",
            "GO:0050794",
            "GO:0050896",
            "GO:0065007",
        ])
        self.assertEqual(result, expected)

    def test_correctly_computes_term_depth(self):
        term = self.dag["GO:0007165"]
        self.assertEqual(term.depth, 4)

        term = self.dag["GO:0023052"]
        self.assertEqual(term.depth, 1)

        term = self.dag["GO:0008150"]
        self.assertEqual(term.depth, 0)

    def test_can_get_lca_of_terms(self):
        terms = [
            self.dag["GO:0007154"],
            self.dag["GO:0050794"],
            self.dag["GO:0051716"]
        ]
        result = get_lca_of_terms(terms)
        expected = [self.dag["GO:0008150"]]
        self.assertEqual(result, expected)

        terms = [
            self.dag["GO:0007154"],
            self.dag["GO:0051716"]
        ]
        result = get_lca_of_terms(terms)
        expected = [self.dag["GO:0009987"]]
        self.assertEqual(result, expected)

        terms = [
            self.dag["GO:0007165"]
        ]
        result = get_lca_of_terms(terms)
        expected = list(set([
            self.dag["GO:0050794"]
        ]))
        self.assertEqual(result, expected)

    def test_get_lca_of_terms_returns_None_if_no_common_ancestors(self):
        terms = [
            self.dag["GO:0008150"],
            self.dag["GO:0008150"],
        ]
        result = get_lca_of_terms(terms)
        self.assertIsNone(result)

    def test_can_group_by_ontology(self):
        grouped = group_terms_by_ontology_type(
            term_ids=["GO:0008150", "GO:0104005", "GO:0016459"],
            max_count=None
        )
        expected = {
            'cc': ["GO:0016459"],
            'bp': ["GO:0008150"],
            'mf': ["GO:0104005"],
        }
        self.assertEqual(grouped, expected)

    def test_can_filter_max_count(self):
        grouped = group_terms_by_ontology_type(
            term_ids=[
                "GO:0008150", "GO:0008150", "GO:0008150",
                "GO:0104005", "GO:0016459", "GO:0016459"
            ],
            max_count=2
        )
        expected = {
            'cc': ["GO:0016459", "GO:0016459"],
            'bp': ["GO:0008150", "GO:0008150"],
            'mf': ["GO:0104005"],
        }
        self.assertEqual(grouped, expected)

    def test_max_count_filter_ignored_when_none(self):
        grouped = group_terms_by_ontology_type(
            term_ids=[
                "GO:0008150", "GO:0008150", "GO:0008150",
                "GO:0104005", "GO:0016459", "GO:0016459"
            ],
            max_count=None
        )
        expected = {
            'cc': ["GO:0016459", "GO:0016459"],
            'bp': ["GO:0008150", "GO:0008150", "GO:0008150"],
            'mf': ["GO:0104005"],
        }
        self.assertEqual(grouped, expected)

    def test_can_filter_obsolete_terms(self):
        result = filter_obsolete_terms(["GO:0000005", "GO:0000006"])
        expected = ["GO:0000006"]
        self.assertEqual(result, expected)

    def test_can_parse_alt_ids(self):
        self.assertEqual(self.dag['GO:0000975'].id, 'GO:0044212')
