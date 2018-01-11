import os
import pandas as pd
import numpy as np
from unittest import TestCase

from ..base import SOURCE, TARGET, LABEL
from ..data_mining.tools import (
    _format_label, _format_labels,

    xy_from_interaction_frame,
    labels_from_interaction_frame,
    ppis_from_interaction_frame,

    make_interaction_frame,
    map_network_accessions,

    remove_nan,
    remove_intersection,
    remove_labels,
    remove_min_counts,
    remove_self_edges,
    merge_labels,
    remove_duplicates,

    process_interactions
)


class TestFormatLabel(TestCase):

    def test_strip_whitespace(self):
        label = ' hello '
        expected = "hello"
        self.assertEqual(expected, _format_label(label))

    def test_lowercase(self):
        label = 'Hello'
        expected = "hello"
        self.assertEqual(expected, _format_label(label))

    def test_replace_whitespace_with_dash(self):
        label = 'Hello world'
        expected = "hello-world"
        self.assertEqual(expected, _format_label(label))


class TestFormatLabels(TestCase):

    def test_splits_on_comma(self):
        labels = ['hello,world']
        expected = [["hello", "world"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_removes_duplicates_from_each_label_if_true(self):
        labels = ['hello,hello']
        expected = [["hello"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_doesnt_remove_duplicates_from_each_label_if_false(self):
        labels = ['hello,hello']
        expected = [["hello", "hello"]]
        result = _format_labels(labels, remove_duplicates_after_split=False)
        self.assertEqual(expected, result)

    def test_sorts_each_label_after_split(self):
        labels = ['world,hello']
        expected = [["hello", "world"]]
        result = _format_labels(labels)
        self.assertEqual(expected, result)

    def test_doesnt_sort_each_label_after_split_if_false(self):
        labels = ['world,hello']
        expected = [["world", "hello"]]
        result = _format_labels(labels, sort_after_split=False)
        self.assertEqual(expected, result)

    def test_joins_labels_if_rejoin_is_true(self):
        labels = ['world,hello']
        expected = ["hello,world"]
        result = _format_labels(labels, rejoin_after_split=True)
        self.assertEqual(expected, result)


class TestXYFromInteractionFrame(TestCase):

    def test_xy_from_iframe_returns_ppis_in_same_order(self):
        iframe = pd.DataFrame(
            {SOURCE: ['A', 'B'], TARGET: ['B', 'A'], LABEL: ['-', '-']}
        )
        ppis, _ = xy_from_interaction_frame(iframe)
        expected = [("A", "B"), ("B", "A")]
        self.assertEqual(ppis, expected)

    def test_xy_from_iframe_does_not_alter_duplicate_rows(self):
        iframe = pd.DataFrame(
            {SOURCE: ['A', 'A'], TARGET: ['A', 'A'], LABEL: ['-', '-']}
        )
        ppis, labels = xy_from_interaction_frame(iframe)
        self.assertEqual(ppis, [("A", "A"), ("A", "A")])
        self.assertEqual(labels, [['-'], ['-']])

    def test_xy_from_iframe_returns_ppis_in_same_order_as_input(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: ['-', '-']}
        )
        ppis, _ = xy_from_interaction_frame(iframe)
        expected = [("B", "B"), ("A", "A")]
        self.assertEqual(ppis, expected)

    def test_xy_from_iframe_returns_labels_in_same_order_as_input(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: ['dog', 'cat']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['dog'], ['cat']]
        self.assertEqual(labels, expected)

    def test_xy_from_iframe_returns_sorted_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['A'], LABEL: ['dog,cat']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['cat', 'dog']]
        self.assertEqual(labels, expected)

    def test_xy_from_iframe_splits_labels_on_comma(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['cat', 'dog']]
        self.assertEqual(labels, expected)

    def test_xy_from_iframe_replaces_whitespace_for_dash_in_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog cat']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['dog-cat']]
        self.assertEqual(labels, expected)

    def test_xy_from_iframe_capitalizes_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['activation,inhibition']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['activation', 'inhibition']]
        self.assertEqual(labels, expected)

    def test_xy_from_iframe_removes_duplicate_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['activation,activation']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['activation']]
        self.assertEqual(labels, expected)

    def test_xy_from_iframe_strips_trailing_whitespace_from_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: [' activation ']}
        )
        _, labels = xy_from_interaction_frame(iframe)
        expected = [['activation']]
        self.assertEqual(labels, expected)


class TestMakeInteractionFrame(TestCase):

    def test_sorts_source_and_target_alphabetically(self):
        sources = ['B']
        targets = ['A']
        labels = ['-']

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['-']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_formats_labels(self):
        sources = ['B']
        targets = ['A']
        labels = ['World hello']

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['world-hello']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))


class TestRemoveNaN(TestCase):

    def test_removes_row_if_source_is_None_values(self):
        sources = [None, 'A']
        targets = ['A', 'B']
        labels = ['1', '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_removes_row_if_target_is_None_values(self):
        sources = ['A', 'A']
        targets = ['None', 'B']
        labels = ['1', '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_removes_row_if_label_is_None_values(self):
        sources = ['A', 'A']
        targets = ['B', 'B']
        labels = [None, '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_also_removes_nan_values(self):
        sources = ['A', 'A']
        targets = ['B', 'B']
        labels = [np.NaN, '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_resets_index(self):
        sources = ['A', 'A']
        targets = ['B', 'B']
        labels = [None, '2']
        result = remove_nan(make_interaction_frame(
            sources, targets, labels)
        ).index
        expected = pd.Index([0])
        self.assertTrue(expected.equals(result))


class TestRemoveIntersection(TestCase):

    def test_removes_ppis_appearing_in_other(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['1'])
        result = remove_intersection(iframe1, iframe2)
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['A'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_doesnt_removes_if_label_is_different(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['2'])
        result = remove_intersection(iframe1, iframe2)
        expected = iframe1
        self.assertTrue(expected.equals(result))

    def test_doesnt_differentiate_direction_of_interaction(self):
        iframe1 = make_interaction_frame(['B'], ['A'], ['1'])
        iframe2 = make_interaction_frame(['A'], ['B'], ['1'])
        result = remove_intersection(iframe1, iframe2)
        self.assertTrue(result.empty)

    def test_resets_index(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['1'])
        result = remove_intersection(iframe1, iframe2).index
        expected = pd.Index([0])
        self.assertTrue(expected.equals(result))


class TestRemoveLabels(TestCase):

    def test_removes_rows_with_label_in_exclusion_list(self):
        iframe = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        result = remove_labels(iframe, ['1'])
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['A'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_case_sensitive(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: ['a', 'A']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_labels(iframe, ['a'])
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['A'], LABEL: ['A']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_exclude_labels_doesnt_exclude_multilabel_samples(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['a,b']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_labels(iframe, ['a'])
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['a,b']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_resets_index(self):
        iframe = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        result = remove_labels(iframe, ['1']).index
        expected = pd.Index([0])
        self.assertTrue(expected.equals(result))


class TestRemoveLabelWithMinCount(TestCase):

    def test_remove_labels_with_count_less_than_min_count(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], ['1', '2', '1']
        )
        result = remove_min_counts(iframe, min_count=2)
        expected = pd.DataFrame(
            {SOURCE: ['B', 'C'], TARGET: ['B', 'C'], LABEL: ['1', '1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(expected.equals(result))

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], ['1', '2', '1']
        )
        result = remove_min_counts(iframe, min_count=2).index
        expected = pd.Index([0, 1])
        self.assertTrue(expected.equals(result))


class TestRemoveSelfEdges(TestCase):

    def test_removes_self_edges(self):
        iframe = make_interaction_frame(
            ['B'], ['B'], ['1']
        )
        result = remove_self_edges(iframe)
        self.assertTrue(result.empty)

    def test_doesnt_remove_regular_edges(self):
        iframe = make_interaction_frame(
            ['A'], ['B'], ['1']
        )
        result = remove_self_edges(iframe)
        self.assertTrue(result.equals(iframe))

    def test_ignores_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['1', '2']
        )
        result = remove_self_edges(iframe)
        self.assertTrue(result.empty)

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'A'], ['1', '2']
        )
        result = remove_self_edges(iframe).index
        expected = pd.Index([0])
        self.assertTrue(expected.equals(result))


class TestMergeLabels(TestCase):

    def test_joins_labels_by_comma(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['cat', 'dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(expected.equals(result))

    def test_sorts_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(expected.equals(result))

    def test_removes_duplicate_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(expected.equals(result))

    def test_removes_whitespace(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], [' dog ', ' cat ']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(expected.equals(result))

    def test_ignores_ppi_direction(self):
        iframe = make_interaction_frame(
            ['A', 'B'], ['B', 'A'], ['dog', 'cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(expected.equals(result))

    def test_lowercases_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['DOG', 'DOG']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(expected.equals(result))

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        expected = pd.Index([0])
        result = merge_labels(iframe).index
        self.assertTrue(expected.equals(result))


class TestRemoveDuplicates(TestCase):

    def test_removes_identical_rows(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(expected.equals(result))

    def test_doesnt_remove_if_label_is_different(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B', 'B'], TARGET: ['B', 'B'], LABEL: ['dog', 'cat']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(expected.equals(result))

    def test_can_detect_merged_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog,cat', 'dog,cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(expected.equals(result))

    def test_can_detect_merged_labels_out_of_order(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog,cat', 'cat,dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(expected.equals(result))

    def test_removes_reverse_duplicates(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['A', 'B'], ['dog', 'dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(expected.equals(result))

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        expected = pd.Index([0])
        result = remove_duplicates(iframe).index
        self.assertTrue(expected.equals(result))


class TestProcessInteractions(TestCase):

    def setUp(self):
        base_dir = os.path.dirname(__file__)
        path = os.path.normpath(
            '{}/test_data/{}'.format(base_dir, "test_network.tsv")
        )
        self.file = open(path, 'rt')
        self.file.readline()

    def tearDown(self):
        self.file.close()

    def test_process_interactions_integration_test_1(self):
        interactions = []
        for line in self.file:
            s, t, l = line.strip().split(',')
            interactions.append((s, t, l))
        iframe = make_interaction_frame(
            [s for s, _, _ in interactions],
            [t for _, t, _ in interactions],
            [l for _, _, l in interactions]
        )
        result = process_interactions(
            iframe, drop_nan=True, allow_self_edges=False,
            allow_duplicates=False, exclude_labels=['inhibition'],
            min_counts=2, merge=True
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O95398', 'O43749', 'Q13131'],
                TARGET: ['P51828', 'O60262', 'Q8NHA4'],
                LABEL: [
                    'activation,binding/association',
                    'activation',
                    'activation,binding/association'
                ]
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(result.equals(expected))


class TestMapNetwork(TestCase):

    def test_can_map_one_to_one(self):
        network_map = {
            'A': ['C'],
            'B': ['D'],
            'C': ['C']
        }
        iframe = pd.DataFrame(
            {
                SOURCE: ['A', 'B', 'C'],
                TARGET: ['C', 'B', 'A'],
                LABEL: ['-', '-', '-']
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'D', 'C'],
                TARGET: ['C', 'D', 'C'],
                LABEL: ['-', '-', '-']
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=False, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(result.equals(expected))

    def test_can_map_one_to_many(self):
        network_map = {
            'A': ['C', 'D'],
        }
        iframe = pd.DataFrame(
            {
                SOURCE: ['A'],
                TARGET: ['A'],
                LABEL: ['-']
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'C', 'C', 'D'],
                TARGET: ['C', 'D', 'D', 'D'],
                LABEL: ['-', '-', '-', '-']
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=False, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(result.equals(expected))
