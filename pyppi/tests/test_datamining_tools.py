import os
import pandas as pd
import numpy as np
from unittest import TestCase

from ..base import SOURCE, TARGET, LABEL, NULL_VALUES
from ..data_mining.tools import (
    _format_label, _format_labels, _split_label,
    _null_to_none, _make_ppi_tuples,

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


def dataframes_are_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    # df1 = df1.replace(to_replace=NULL_VALUES, value=str(None), inplace=False)
    # df2 = df2.replace(to_replace=NULL_VALUES, value=str(None), inplace=False)
    return df1.equals(df2)


class TestNullToNone(TestCase):

    def test_can_convert_null_to_none(self):
        for value in NULL_VALUES:
            self.assertIs(None, _null_to_none(value))


class TestMakePPITuples(TestCase):

    def test_sorts_by_alpha(self):
        self.assertEqual(
            [('A', 'B')],
            _make_ppi_tuples(['B'], ['A'])
        )

    def test_converts_null_to_none(self):
        for value in NULL_VALUES:
            self.assertEqual(
                [tuple(sorted(['None', 'A']))],
                _make_ppi_tuples([value], ['A'])
            )


class TestSplitLabel(TestCase):

    def test_strip_whitespace(self):
        label = ' hello , world '
        expected = ['hello', 'world']
        self.assertEqual(expected, _split_label(label))

    def test_splits_on_comma_by_default(self):
        label = 'hello,world'
        expected = ['hello', 'world']
        self.assertEqual(expected, _split_label(label))

    def test_splits_on_sep(self):
        label = 'hello|,world'
        expected = ['hello', ',world']
        self.assertEqual(expected, _split_label(label, sep='|'))

    def test_strips_white_space(self):
        label = ' hello world '
        expected = ["hello world"]
        self.assertEqual(expected, _split_label(label))

    def test_can_handle_none(self):
        for value in NULL_VALUES:
            self.assertEqual([str(None)], _split_label(value))


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

    def test_can_handle_none(self):
        for value in NULL_VALUES:
            self.assertEqual(str(None).lower(), _format_label(value))


class TestFormatLabels(TestCase):

    def test_splits_on_comma(self):
        labels = ['hello,world']
        expected = [["hello", "world"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_splits_get_own_list(self):
        labels = ['hello,world', 'hello,world']
        expected = [["hello", "world"], ["hello", "world"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_removes_duplicates_from_each_label_if_true(self):
        labels = ['hello,hello']
        expected = [["hello"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_removes_null_duplicates(self):
        for value in NULL_VALUES:
            labels = ['{},none'.format(value)]
            expected = [[str(None).lower()]]
            result = _format_labels(labels)
            self.assertEqual(expected, result)

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

    def test_can_handle_null_values(self):
        for value in NULL_VALUES:
            self.assertEqual([[str(None).lower()]], _format_labels([value]))


class TestLabelsFromInteractionFrame(TestCase):

    def test_converts_null_to_None(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {SOURCE: ['A', 'A'], TARGET: ['A', 'A'], LABEL: ['cat', value]}
            )
            labels = labels_from_interaction_frame(iframe)
            self.assertEqual(labels, [['cat'], [None]])

    def test_converts_multilabel_null_to_None(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {
                    SOURCE: ['A', 'A'], TARGET: ['A', 'A'],
                    LABEL: ['{},cat'.format(value), '{},whale'.format(value)]
                }
            )
            labels = labels_from_interaction_frame(iframe)
            self.assertEqual(labels, [['cat', None], [None, 'whale']])

    def test_returns_label_row_in_same_order_as_input(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: ['dog', 'cat']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['dog'], ['cat']]
        self.assertEqual(labels, expected)

    def test_returns_sorted_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['A'], LABEL: ['dog,cat']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['cat', 'dog']]
        self.assertEqual(labels, expected)

    def test_splits_labels_on_comma(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['cat', 'dog']]
        self.assertEqual(labels, expected)

    def test_replaces_whitespace_for_dash_in_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog cat']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['dog-cat']]
        self.assertEqual(labels, expected)

    def test_removes_duplicate_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['activation,activation']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['activation']]
        self.assertEqual(labels, expected)

    def test_strips_trailing_whitespace_from_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: [' activation ']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['activation']]
        self.assertEqual(labels, expected)


class TestPPIsFromInteractionFrame(TestCase):

    def test_returns_ppis_in_same_order(self):
        iframe = pd.DataFrame(
            {SOURCE: ['A', 'B'], TARGET: ['B', 'A'], LABEL: [None, None]}
        )
        ppis = ppis_from_interaction_frame(iframe)
        expected = [("A", "B"), ("B", "A")]
        self.assertEqual(ppis, expected)

    def test_does_not_alter_duplicate_rows(self):
        iframe = pd.DataFrame(
            {SOURCE: ['A', 'A'], TARGET: ['A', 'A'], LABEL: [None, None]}
        )
        ppis = ppis_from_interaction_frame(iframe)
        self.assertEqual(ppis, [("A", "A"), ("A", "A")])

    def test_returns_ppis_in_same_order_as_input(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: [None, None]}
        )
        ppis = ppis_from_interaction_frame(iframe)
        expected = [("B", "B"), ("A", "A")]
        self.assertEqual(ppis, expected)

    def test_returns_null_values_as_None(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {SOURCE: [value, 'A'], TARGET: ['B', 'A'], LABEL: ['1', '2']}
            )
            ppis = ppis_from_interaction_frame(iframe)
            expected = [(None, "B"), ("A", "A")]
            self.assertEqual(ppis, expected)


class TestXYFromInteractionFrame(TestCase):
    
    def test_integration(self):
        iframe = pd.DataFrame(
            {
                SOURCE: ['B', None], TARGET: ['B', 'A'], 
                LABEL: ['None,,cat', 'dog,cat,Dog,blue whale,None,none']
            }
        )
        ppis, labels = xy_from_interaction_frame(iframe)
        self.assertEqual(ppis, [("B", "B"), (None, "A")])
        self.assertEqual(
            labels, [['cat', None], ['blue-whale', 'cat', 'dog', None]]
        )


class TestMakeInteractionFrame(TestCase):

    def test_sorts_source_and_target_alphabetically(self):
        sources = ['B']
        targets = ['A']
        labels = ['1']

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_converts_null_value_to_NaN(self):
        sources = ['B'] * len(NULL_VALUES)
        targets = ['A'] * len(NULL_VALUES)
        labels = list(NULL_VALUES)

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {
                SOURCE: ['A'] * len(NULL_VALUES),
                TARGET: ['B'] * len(NULL_VALUES),
                LABEL: [None] * len(NULL_VALUES)
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_formats_labels(self):
        sources = ['B']
        targets = ['A']
        labels = ['World hello']

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['world-hello']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_correctly_creates_additional_columns_in_alpha_key_order(self):
        sources = ['B']
        targets = ['A']
        labels = ['activation']
        extra_1 = ["pmid:xxx"]
        extra_2 = ["MI:xxx"]
        extra = {"pubmed": extra_1, "psimi": extra_2}

        result = make_interaction_frame(sources, targets, labels, **extra)
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['B'], LABEL: ['activation'],
                'psimi': ["MI:xxx"], 'pubmed': ["pmid:xxx"]
            },
            columns=[SOURCE, TARGET, LABEL, 'psimi', "pubmed"]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_additional_columns_preserve_list_order(self):
        sources = ['A', 'C']
        targets = ['A', 'C']
        labels = ['activation', 'activation']
        extra_1 = ["pmid:1", "pmid:2"]
        extra_2 = ["MI:1", "MI:2"]
        extra = {"pubmed": extra_1, "psimi": extra_2}

        result = make_interaction_frame(sources, targets, labels, **extra)
        expected = pd.DataFrame(
            {
                SOURCE: ['A', 'C'], TARGET: ['A', 'C'],
                LABEL: ['activation', 'activation'],
                'psimi': ["MI:1", "MI:2"],
                'pubmed': ["pmid:1", "pmid:2"],
            },
            columns=[SOURCE, TARGET, LABEL, 'psimi', "pubmed"]
        )
        self.assertTrue(dataframes_are_equal(expected, result))


class TestRemoveNaN(TestCase):

    def test_removes_row_if_source_is_None(self):
        sources = [None, 'A']
        targets = ['A', 'B']
        labels = ['1', '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_row_if_target_is_None(self):
        sources = ['A', 'A']
        targets = [None, 'B']
        labels = ['1', '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_row_if_label_is_None(self):
        sources = ['A', 'A']
        targets = ['B', 'B']
        labels = [None, '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removed_row_if_contains_anything_from_NULL_VALUES(self):
        for value in NULL_VALUES:
            sources = ['A', 'A']
            targets = ['B', 'B']
            labels = [value, '2']
            result = remove_nan(
                make_interaction_frame(sources, targets, labels)
            )
            expected = pd.DataFrame(
                {SOURCE: ['A'], TARGET: ['B'], LABEL: ['2']},
                columns=[SOURCE, TARGET, LABEL]
            )
            self.assertTrue(dataframes_are_equal(expected, result))

    def test_resets_index(self):
        sources = ['A', 'A']
        targets = ['B', 'B']
        labels = [None, '2']
        result = remove_nan(make_interaction_frame(
            sources, targets, labels)
        ).index
        expected = pd.Index([0])
        self.assertTrue(expected.equals(result))

    def test_does_not_remove_row_if_column_not_in_subset(self):
        sources = ['B']
        targets = ['A']
        labels = ['activation']
        extra_1 = [None]
        extra_2 = [None]
        extra = {"pubmed": extra_1, "psimi": extra_2}

        result = remove_nan(
            make_interaction_frame(sources, targets, labels, **extra)
        )
        expected = make_interaction_frame(
            ['A'], ['B'], ['activation'],
            **{'psimi': [None], 'pubmed': [None]}
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_does_not_modify_value_of_other_columns(self):
        sources = [None, 'A']
        targets = ['B', 'A']
        labels = ['activation', 'activation']
        extra_1 = [None] * 2
        extra_2 = [None] * 2
        extra = {"pubmed": extra_1, "psimi": extra_2}

        result = remove_nan(
            make_interaction_frame(sources, targets, labels, **extra)
        )
        expected = make_interaction_frame(
            ['A'], ['A'], ['activation'],
            **{'psimi': [None], 'pubmed': [None]}
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_row_if_column_included_in_subset(self):
        sources = ['B']
        targets = ['A']
        labels = ['activation']
        extra_1 = [None]
        extra_2 = ['MI:xxx']
        extra = {"pubmed": extra_1, "psimi": extra_2}

        result = remove_nan(
            make_interaction_frame(sources, targets, labels, **extra),
            subset=[SOURCE, TARGET, LABEL, 'pubmed']
        )
        self.assertTrue(result.empty)


class TestRemoveIntersection(TestCase):

    def test_removes_ppis_appearing_in_other(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['1'])
        result, removed = remove_intersection(iframe1, iframe2, use_label=True)
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['A'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        removed_exp = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))
        self.assertTrue(dataframes_are_equal(removed_exp, removed))

    def test_removes_ppis_with_null_source_or_target(self):
        iframe1 = make_interaction_frame(['B', 'A'], [None, 'A'], ['1', '2'])
        iframe2 = make_interaction_frame([None], ['B'], ['1'])
        result, removed = remove_intersection(iframe1, iframe2)

        expected = make_interaction_frame(['A'], ['A'], ['2'])
        removed_exp = make_interaction_frame(['B'], [None], ['1'])
        self.assertTrue(dataframes_are_equal(expected, result))
        self.assertTrue(dataframes_are_equal(removed_exp, removed))

    def test_removes_ppis_with_null_label_if_use_label_is_true(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], [None, '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], [None])
        result, removed = remove_intersection(iframe1, iframe2)

        expected = make_interaction_frame(['A'], ['A'], ['2'])
        removed_exp = make_interaction_frame(['B'], ['B'], [None])
        self.assertTrue(dataframes_are_equal(expected, result))
        self.assertTrue(dataframes_are_equal(removed_exp, removed))

    def test_doesnt_removes_if_label_is_different_and_use_label_is_true(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['2'])
        result, removed = remove_intersection(iframe1, iframe2, use_label=True)
        expected = iframe1
        self.assertTrue(dataframes_are_equal(expected, result))
        self.assertTrue(removed.empty)

    def test_removes_dupe_if_label_is_different_but_use_label_is_false(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['3'])
        result, removed = remove_intersection(
            iframe1, iframe2, use_label=False)
        expected = make_interaction_frame(['A'], ['A'], ['2'])
        removed_exp = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))
        self.assertTrue(dataframes_are_equal(removed_exp, removed))

    def test_doesnt_differentiate_direction_of_interaction(self):
        iframe1 = make_interaction_frame(['B'], ['A'], ['1'])
        iframe2 = make_interaction_frame(['A'], ['B'], ['1'])
        result, removed = remove_intersection(iframe1, iframe2, use_label=True)
        self.assertTrue(result.empty)
        self.assertTrue(removed.equals(iframe1))

    def test_resets_index(self):
        iframe1 = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        iframe2 = make_interaction_frame(['B'], ['B'], ['1'])
        result, removed = remove_intersection(iframe1, iframe2, use_label=True)
        self.assertTrue(result.index.equals(pd.Index([0])))
        self.assertTrue(removed.index.equals(pd.Index([0])))

    def test_extra_columns_are_ignored(self):
        iframe1 = make_interaction_frame(
            ['B', 'A'], ['B', 'A'], ['1', '2'],
            **{"col1": ['XXX', 'ZZZ'], 'col2': ['XXX', 'ZZZ']}
        )
        iframe2 = make_interaction_frame(
            ['B'], ['B'], ['1'],
            **{"col1": ['ZZZ'], 'col2': ['ZZZ']}
        )
        result, removed = remove_intersection(iframe1, iframe2, use_label=True)
        expected = make_interaction_frame(
            ['A'], ['A'], ['2'],
            **{"col1": ['ZZZ'], 'col2': ['ZZZ']}
        )
        removed_exp = make_interaction_frame(
            ['B'], ['B'], ['1'],
            **{"col1": ['XXX'], 'col2': ['XXX']}
        )
        self.assertTrue(dataframes_are_equal(expected, result))
        self.assertTrue(dataframes_are_equal(removed_exp, removed))


class TestRemoveLabels(TestCase):

    def test_removes_rows_with_label_in_exclusion_list(self):
        iframe = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        result = remove_labels(iframe, ['1'])
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['A'], LABEL: ['2']},
            columns=[SOURCE, TARGET, LABEL]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

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
        self.assertTrue(dataframes_are_equal(expected, result))

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
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_exclude_labels_doesnt_exclude_nan_labels(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {SOURCE: ['B'], TARGET: ['B'], LABEL: [value]},
                columns=[SOURCE, TARGET, LABEL]
            )
            result = remove_labels(iframe, ['a'])
            expected = pd.DataFrame(
                {SOURCE: ['B'], TARGET: ['B'], LABEL: [value]},
                columns=[SOURCE, TARGET, LABEL]
            )
            self.assertTrue(dataframes_are_equal(expected, result))

    def test_exclude_labels_excludes_nan_labels(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {SOURCE: ['B'], TARGET: ['B'], LABEL: [value]},
                columns=[SOURCE, TARGET, LABEL]
            )
            result = remove_labels(iframe, NULL_VALUES)
            self.assertTrue(result.empty)

    def test_resets_index(self):
        iframe = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        result = remove_labels(iframe, ['1']).index
        self.assertTrue(result.equals(pd.Index([0])))

    def test_extra_columns_are_ignored(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['B', 'A'], ['1', '2'],
            **{"col1": ['XXX', 'ZZZ'], 'col2': ['XXX', 'ZZZ']}
        )
        result = remove_labels(iframe, ['1'])
        expected = make_interaction_frame(
            ['A'], ['A'], ['2'],
            **{"col1": ['ZZZ'], 'col2': ['ZZZ']}
        )
        self.assertTrue(dataframes_are_equal(expected, result))


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
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_doesnt_error_if_NULL_VALUES_present(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], [None, None, None]
        )
        result = remove_min_counts(iframe, min_count=2)
        self.assertTrue(dataframes_are_equal(iframe, result))

    def test_null_also_treated_as_a_label(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], [None, None, None]
        )
        result = remove_min_counts(iframe, min_count=4)
        self.assertTrue(result.empty)

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], ['1', '2', '1']
        )
        result = remove_min_counts(iframe, min_count=2).index
        self.assertTrue(pd.Index([0, 1]).equals(result))

    def test_extra_columns_are_ignored(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], ['1', '2', '1'],
            **{"col1": ['X', 'Z', 'Y']}
        )
        result = remove_min_counts(iframe, min_count=2)
        expected = make_interaction_frame(
            ['B', 'C'], ['B', 'C'], ['1', '1'],
            **{"col1": ['X', 'Y']}
        )
        self.assertTrue(dataframes_are_equal(expected, result))


class TestRemoveSelfEdges(TestCase):

    def test_removes_self_edges(self):
        iframe = make_interaction_frame(
            ['B'], ['B'], ['1']
        )
        result = remove_self_edges(iframe)
        self.assertTrue(result.empty)

    def test_removes_self_edges_that_are_nan(self):
        iframe = make_interaction_frame(
            [None], [None], ['1']
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
        self.assertTrue(pd.Index([0]).equals(result))

    def test_extra_columns_are_ignored(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], ['1', '2', '1'],
            **{"col1": ['X', 'Z', 'Y']}
        )
        result = remove_self_edges(iframe)
        self.assertTrue(result.empty)


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
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_sorts_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_doesnt_merge_non_null_with_null(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                ['B', 'B'], ['B', 'B'], [value, 'cat']
            )
            expected = pd.DataFrame(
                {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat']},
                columns=[SOURCE, TARGET, LABEL]
            )
            result = merge_labels(iframe)
            self.assertTrue(dataframes_are_equal(expected, result))

    def test_can_merge_already_merged_labels(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['A', 'B'], ['dog,cat', 'eel,dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['cat,dog,eel']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_can_merge_null_labels(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['A', 'B'], [None, None]
        )
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: [None]},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_duplicate_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_whitespace(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], [' dog ', ' cat ']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_ignores_ppi_direction(self):
        iframe = make_interaction_frame(
            ['A', 'B'], ['B', 'A'], ['dog', 'cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_lowercases_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['DOG', 'DOG']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = merge_labels(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        result = merge_labels(iframe).index
        self.assertTrue(result.equals(pd.Index([0])))

    def test_additional_columns_are_comma_joined(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ['1111|2222', "3333|4444"], "col2": ["merge", "me"]}
        )
        result = merge_labels(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": ['1111|2222,3333|4444'], "col2": ["me,merge"]}
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_row_data_order_is_not_mixed(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['B', 'A'], ['dog', 'cat'],
            **{"col1": ['1111|2222', "3333|4444"]}
        )
        result = merge_labels(iframe)
        self.assertTrue(result.equals(iframe))

    def test_premerged_extra_columns_can_be_merged_again(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['A', 'A'], ['dog', 'cat'],
            **{"col1": ['3333|4444,1111|2222', "3333|4444"]}
        )
        expected = make_interaction_frame(
            ['A'], ['B'], ['cat,dog'],
            **{"col1": ['1111|2222,3333|4444']}
        )
        result = merge_labels(iframe)
        self.assertTrue(result.equals(expected))

    def test_additional_columns_default_to_None_if_not_present(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['A', 'A'], ['dog', 'cat'],
            **{"col1": ['', 'NaN']}
        )
        result = merge_labels(iframe)
        expected = make_interaction_frame(
            ['A'], ['B'], ['cat,dog'],
            **{"col1": [None]}
        )
        self.assertTrue(result.equals(expected))

    def test_additional_columns_filter_out_nan(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['A', 'A'], ['dog', 'cat'],
            **{"col1": ['1', 'NaN']}
        )
        result = merge_labels(iframe)
        expected = make_interaction_frame(
            ['A'], ['B'], ['cat,dog'],
            **{"col1": ['1']}
        )
        self.assertTrue(result.equals(expected))


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
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_doesnt_remove_if_label_is_different(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B', 'B'], TARGET: ['B', 'B'], LABEL: ['dog', 'cat']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_can_handle_null_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], [None, None]
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: [None]},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_can_detect_merged_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog,cat', 'dog,cat']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_can_detect_merged_labels_out_of_order(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog,cat', 'cat,dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_reverse_duplicates(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['A', 'B'], ['dog', 'dog']
        )
        expected = pd.DataFrame(
            {SOURCE: ['A'], TARGET: ['B'], LABEL: ['dog']},
            columns=[SOURCE, TARGET, LABEL]
        )
        result = remove_duplicates(iframe)
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        result = remove_duplicates(iframe).index
        self.assertTrue(pd.Index([0]).equals(result))

    def test_additional_columns_marged_when_removing_duplicate(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ['1111|2222', "3333|4444"]}
        )
        result = remove_duplicates(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": ['1111|2222,3333|4444']}
        )
        self.assertTrue(result.equals(expected))

    def test_additional_columns_duplicates_removed(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ['1111|2222', "1111|2222"]}
        )
        result = remove_duplicates(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": ['1111|2222']}
        )
        self.assertTrue(result.equals(expected))

    def test_additional_columns_sorted(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ['3|4', "1|2"]}
        )
        result = remove_duplicates(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": ['1|2,3|4']}
        )
        self.assertTrue(result.equals(expected))

    def test_additional_columns_removes_trailing_whitespace(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ["   1|2   ", "   1|2   "]}
        )
        result = remove_duplicates(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": ["1|2"]}
        )
        self.assertTrue(result.equals(expected))

    def test_additional_columns_default_to_None_if_not_present(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ["", ""]}
        )
        result = remove_duplicates(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": [None]}
        )
        self.assertTrue(result.equals(expected))

    def test_additional_columns_filter_out_nan(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog'],
            **{"col1": ["1", None]}
        )
        result = remove_duplicates(iframe)
        expected = make_interaction_frame(
            ['B'], ['B'], ['dog'],
            **{"col1": ['1']}
        )
        self.assertTrue(result.equals(expected))


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

    def test_default_drop_nan_uses_source_target_and_label_as_column(self):
        iframe = make_interaction_frame(
            [None, 'B', 'C'],
            ['A', None, 'C'],
            ['1', '2', None],
            **{'meta': ['a', 'b', 'c']}
        )
        result = process_interactions(
            iframe, drop_nan='default', allow_self_edges=True,
            allow_duplicates=True, exclude_labels=None,
            min_counts=None, merge=False
        )
        self.assertTrue(result.empty)

    def test_normalises_nan_values(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                [value, 'B', 'C'],
                ['A', value, 'C'],
                ['1', '2', value],
                **{'meta': ['a', 'b', 'c']}
            )
            expected = make_interaction_frame(
                [None, 'B', 'C'],
                ['A', None, 'C'],
                ['1', '2', None],
                **{'meta': ['a', 'b', 'c']}
            )
            result = process_interactions(
                iframe, drop_nan=None, allow_self_edges=True,
                allow_duplicates=True, exclude_labels=None,
                min_counts=None, merge=False
            )
            self.assertTrue(dataframes_are_equal(expected, result))

    def test_drop_nan_bypasses_when_none_is_suppled(self):
        iframe = make_interaction_frame(
            [None, 'B', 'C'],
            ['A', None, 'B'],
            ['1', '2', None],
            **{'meta': ['a', 'b', 'c']}
        )
        result = process_interactions(
            iframe, drop_nan=None, allow_self_edges=True,
            allow_duplicates=True, exclude_labels=None,
            min_counts=None, merge=False
        )
        self.assertTrue(result.equals(iframe))

    def test_drop_nan_uses_columns_when_supplied(self):
        iframe = make_interaction_frame(
            [None, 'B', 'C'],
            ['A', None, 'C'],
            ['1', '2', None],
            **{'meta': ['a', 'b', 'c']}
        )
        result = process_interactions(
            iframe, drop_nan=[SOURCE, TARGET], allow_self_edges=True,
            allow_duplicates=True, exclude_labels=None,
            min_counts=None, merge=False
        )
        expected = make_interaction_frame(
            ['C'], ['C'], [None], **{'meta': ['c']}
        )
        self.assertTrue(result.equals(expected))

    def test_process_interactions_integration_test(self):
        interactions = []
        for line in self.file:
            s, t, l, x, y = line.strip().split(',')
            interactions.append((s, t, l, x, y))
        iframe = make_interaction_frame(
            [s for s, *_ in interactions],
            [t for _, t, *_ in interactions],
            [l for _, _, l, *_ in interactions],
            **{
                "pmid": [x for _, _, _, x, _ in interactions],
                "psimi": [y for _, _, _, _, y in interactions]
            }
        )
        result = process_interactions(
            iframe, drop_nan='default', allow_self_edges=False,
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
                ],
                "pmid": ["1,2", "3", None],
                "psimi": ["MI:1,MI:2", "MI:3", "MI:3"]
            },
            columns=[SOURCE, TARGET, LABEL, "pmid", "psimi"]
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
                LABEL: [None, None, None]
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'D', 'C'],
                TARGET: ['C', 'D', 'C'],
                LABEL: [None, None, None]
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=None, allow_self_edges=True,
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
                LABEL: [None]
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'C', 'C', 'D'],
                TARGET: ['C', 'D', 'D', 'D'],
                LABEL: [None] * 4
            },
            columns=[SOURCE, TARGET, LABEL]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=None, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(result.equals(expected))

    def test_will_correctly_map_meta_columns(self):
        network_map = {
            'A': ['C', 'D'],
        }
        iframe = pd.DataFrame(
            {
                SOURCE: ['A'],
                TARGET: ['A'],
                LABEL: [None],
                'meta': ['mi:xxx']
            },
            columns=[SOURCE, TARGET, LABEL, 'meta']
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'C', 'C', 'D'],
                TARGET: ['C', 'D', 'D', 'D'],
                LABEL: [None] * 4,
                'meta': ['mi:xxx', 'mi:xxx', 'mi:xxx', 'mi:xxx']
            },
            columns=[SOURCE, TARGET, LABEL, 'meta']
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=None, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(result.equals(expected))
