import os
import pandas as pd
import numpy as np
from unittest import TestCase

from ..base.constants import (
    SOURCE, TARGET, LABEL, NULL_VALUES, PUBMED, EXPERIMENT_TYPE
)
from ..data_mining.tools import (
    _format_label, _format_labels, _split_label,
    _null_to_none, _make_ppi_tuples,

    xy_from_interaction_frame,
    labels_from_interaction_frame,
    ppis_from_interaction_frame,

    make_interaction_frame,
    map_network_accessions,

    remove_nan,
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
            self.assertEqual([None], _split_label(value))

    def test_filters_nones(self):
        for value in NULL_VALUES:
            self.assertEqual([None], _split_label(
                '{},{}'.format(value, value)))


class TestFormatLabel(TestCase):

    def test_strip_whitespace(self):
        label = ' hello '
        expected = "Hello"
        self.assertEqual(expected, _format_label(label))

    def test_caps(self):
        label = 'Hello'
        expected = "Hello"
        self.assertEqual(expected, _format_label(label))

    def test_replace_whitespace_with_dash(self):
        label = 'Hello world'
        expected = "Hello-world"
        self.assertEqual(expected, _format_label(label))

    def test_can_handle_none(self):
        for value in NULL_VALUES:
            self.assertEqual(None, _format_label(value))


class TestFormatLabels(TestCase):

    def test_splits_on_comma(self):
        labels = ['hello,world']
        expected = [["Hello", "World"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_splits_get_own_list(self):
        labels = ['hello,world', 'hello,world']
        expected = [["Hello", "World"], ["Hello", "World"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_removes_duplicates_from_each_label_if_true(self):
        labels = ['hello,hello']
        expected = [["Hello"]]
        self.assertEqual(expected, _format_labels(labels))

    def test_removes_null_duplicates(self):
        for value in NULL_VALUES:
            labels = ['{},none'.format(value)]
            expected = [[None]]
            result = _format_labels(labels)
            self.assertEqual(expected, result)

    def test_sorts_each_label_after_split(self):
        labels = ['world,hello']
        expected = [["Hello", "World"]]
        result = _format_labels(labels)
        self.assertEqual(expected, result)

    def test_joins_if_true(self):
        labels = ['world,hello']
        expected = ["Hello,World"]
        result = _format_labels(labels, join=True)
        self.assertEqual(expected, result)

    def test_can_handle_null_values(self):
        for value in NULL_VALUES:
            self.assertEqual([[None]], _format_labels([value]))


class TestLabelsFromInteractionFrame(TestCase):

    def test_converts_null_to_None(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {SOURCE: ['A', 'A'], TARGET: ['A', 'A'], LABEL: ['cat', value]}
            )
            labels = labels_from_interaction_frame(iframe)
            self.assertEqual(labels, [['Cat'], [None]])

    def test_removes_multilabel_null(self):
        for value in NULL_VALUES:
            iframe = pd.DataFrame(
                {
                    SOURCE: ['A', 'A'], TARGET: ['A', 'A'],
                    LABEL: ['{},cat'.format(value), '{},whale'.format(value)]
                }
            )
            labels = labels_from_interaction_frame(iframe)
            self.assertEqual(labels, [['Cat'], ['Whale']])

    def test_returns_label_row_in_same_order_as_input(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: ['dog', 'cat']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['Dog'], ['Cat']]
        self.assertEqual(labels, expected)

    def test_returns_sorted_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['A'], LABEL: ['dog,cat']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['Cat', 'Dog']]
        self.assertEqual(labels, expected)

    def test_splits_labels_on_comma(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['cat,dog']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['Cat', 'Dog']]
        self.assertEqual(labels, expected)

    def test_replaces_whitespace_for_dash_in_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['dog cat']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['Dog-cat']]
        self.assertEqual(labels, expected)

    def test_removes_duplicate_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: ['activation,activation']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['Activation']]
        self.assertEqual(labels, expected)

    def test_strips_trailing_whitespace_from_labels(self):
        iframe = pd.DataFrame(
            {SOURCE: ['B'], TARGET: ['B'], LABEL: [' activation ']}
        )
        labels = labels_from_interaction_frame(iframe)
        expected = [['Activation']]
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
            labels, [['Cat'], ['Blue-whale', 'Cat', 'Dog']]
        )


class TestMakeInteractionFrame(TestCase):

    def test_sorts_source_and_target_alphabetically(self):
        sources = ['B']
        targets = ['A']
        labels = ['1']

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['B'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
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
                LABEL: [None] * len(NULL_VALUES),
                PUBMED: [None] * len(NULL_VALUES),
                EXPERIMENT_TYPE: [None] * len(NULL_VALUES)
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_formats_labels(self):
        sources = ['B']
        targets = ['A']
        labels = ['World hello']

        result = make_interaction_frame(sources, targets, labels)
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['B'], LABEL: ['World-hello'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_pmids_and_psimis_are_not_sorted(self):
        sources = ['B'] * 2
        targets = ['A'] * 2
        labels = ['activation'] * 2
        extra_1 = ["1001", "1000"]
        extra_2 = ["MI:2", "MI:1"]
        result = make_interaction_frame(
            sources, targets, labels, pmids=extra_1, psimis=extra_2)
        expected = pd.DataFrame(
            {
                SOURCE: ['A']*2, TARGET: ['B']*2, LABEL: ['Activation']*2,
                EXPERIMENT_TYPE: ["MI:2", "MI:1"], PUBMED: ["1001", "1000"]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_source_target_are_not_sorted_if_sort_is_false(self):
        sources = ['B'] * 2
        targets = ['A'] * 2
        labels = ['activation'] * 2
        extra_1 = ["1001", "1000"]
        extra_2 = ["MI:2", "MI:1"]
        result = make_interaction_frame(
            sources, targets, labels, pmids=extra_1, psimis=extra_2, sort=False)
        expected = pd.DataFrame(
            {
                SOURCE: ['B']*2, TARGET: ['A']*2, LABEL: ['Activation']*2,
                EXPERIMENT_TYPE: ["MI:2", "MI:1"], PUBMED: ["1001", "1000"]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))


class TestRemoveNaN(TestCase):

    def test_removes_row_if_source_is_None(self):
        sources = [None, 'A']
        targets = ['A', 'B']
        labels = ['1', '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['B'], LABEL: ['2'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_row_if_target_is_None(self):
        sources = ['A', 'A']
        targets = [None, 'B']
        labels = ['1', '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['B'], LABEL: ['2'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_row_if_label_is_None(self):
        sources = ['A', 'A']
        targets = ['B', 'B']
        labels = [None, '2']
        result = remove_nan(make_interaction_frame(sources, targets, labels))
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['B'], LABEL: ['2'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
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
                {
                    SOURCE: ['A'], TARGET: ['B'], LABEL: ['2'],
                    PUBMED: [None], EXPERIMENT_TYPE: [None]
                },
                columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
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

    def test_ignores_null_values_in_pmid_and_psimi(self):
        sources = ['B']
        targets = ['A']
        labels = ['activation']
        extra_1 = [None]
        extra_2 = [None]

        result = remove_nan(
            make_interaction_frame(sources, targets, labels, extra_1, extra_2)
        )
        expected = make_interaction_frame(
            ['A'], ['B'], ['Activation'], extra_1, extra_2
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_removes_row_if_column_included_in_subset(self):
        for value in NULL_VALUES:
            sources = ['B']
            targets = ['A']
            labels = ['activation']
            extra_1 = [value]
            extra_2 = [value]

            result = remove_nan(
                make_interaction_frame(
                    sources, targets, labels, extra_1, extra_2),
                subset=[PUBMED, EXPERIMENT_TYPE]
            )
            self.assertTrue(result.empty)


class TestRemoveLabels(TestCase):

    def test_removes_rows_with_label_in_exclusion_list(self):
        iframe = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        result = remove_labels(iframe, ['1'])
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['A'], LABEL: ['2'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_case_sensitive(self):
        iframe = pd.DataFrame(
            {
                SOURCE: ['B', 'A'], TARGET: ['B', 'A'], LABEL: ['cat', 'Cat'],
                PUBMED: [None]*2, EXPERIMENT_TYPE: [None]*2
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        result = remove_labels(iframe, ['cat'])
        expected = pd.DataFrame(
            {
                SOURCE: ['A'], TARGET: ['A'], LABEL: ['Cat'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_exclude_labels_doesnt_exclude_multilabel_samples(self):
        iframe = make_interaction_frame(['B'], ['B'], ['Cat,Dog'])
        result = remove_labels(iframe, ['Cat'])
        expected = pd.DataFrame(
            {
                SOURCE: ['B'], TARGET: ['B'], LABEL: ['Cat,Dog'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(dataframes_are_equal(expected, result))

    def test_exclude_labels_excludes_nan_labels(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(['B'], ['B'], [value])
            result = remove_labels(iframe, NULL_VALUES)
            self.assertTrue(result.empty)

    def test_resets_index(self):
        iframe = make_interaction_frame(['B', 'A'], ['B', 'A'], ['1', '2'])
        result = remove_labels(iframe, ['1']).index
        self.assertTrue(result.equals(pd.Index([0])))


class TestRemoveLabelWithMinCount(TestCase):

    def test_remove_labels_with_count_less_than_min_count(self):
        iframe = make_interaction_frame(
            ['B', 'A', 'C'], ['B', 'A', 'C'], ['1', '2', '1']
        )
        result = remove_min_counts(iframe, min_count=2)
        expected = pd.DataFrame(
            {
                SOURCE: ['B', 'C'], TARGET: ['B', 'C'], LABEL: ['1', '1'],
                PUBMED: [None] * 2, EXPERIMENT_TYPE: [None] * 2
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
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

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'A'], ['1', '2']
        )
        result = remove_self_edges(iframe).index
        self.assertTrue(pd.Index([0]).equals(result))


class TestMergeLabels(TestCase):

    def test_joins_labels_by_comma(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Cat', 'Dog']
        )
        result = merge_labels(iframe)
        self.assertEqual(list(result[LABEL]), ['Cat,Dog'])

    def test_sorts_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'cat']
        )
        result = merge_labels(iframe)
        self.assertEqual(list(result[LABEL]), ['Cat,Dog'])

    def test_doesnt_merge_non_null_with_null(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                ['B', 'B'], ['B', 'B'], [value, 'Cat']
            )
            result = merge_labels(iframe)
            self.assertEqual(list(result[LABEL]), ['Cat'])

    def test_can_merge_already_merged_labels(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['A', 'B'], ['dog,cat', 'eel,dog']
        )
        merged = merge_labels(iframe)
        expected = ['Cat,Dog,Eel']
        result = list(merged[LABEL])
        self.assertEqual(result, expected)

    def test_can_merge_null_labels(self):
        for value1 in NULL_VALUES:
            for value2 in NULL_VALUES:
                iframe = make_interaction_frame(
                    ['B', 'A'], ['A', 'B'], [value1, value2]
                )
                merged = merge_labels(iframe)
                expected = [None]
                result = list(merged[LABEL])
                self.assertEqual(result, expected)

    def test_removes_duplicate_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog']
        )
        merged = merge_labels(iframe)
        expected = ['Dog']
        result = list(merged[LABEL])
        self.assertEqual(result, expected)

    def test_removes_whitespace(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], [' Dog ', ' Cat ']
        )
        merged = merge_labels(iframe)
        expected = ['Cat,Dog']
        result = list(merged[LABEL])
        self.assertEqual(result, expected)

    def test_ignores_ppi_direction(self):
        iframe = make_interaction_frame(
            ['A', 'B'], ['B', 'A'], ['Dog', 'Cat']
        )
        merged = merge_labels(iframe)
        expected = ['Cat,Dog']
        result = list(merged[LABEL])
        self.assertEqual(result, expected)
        self.assertEqual(list(merged[SOURCE]), ['A'])
        self.assertEqual(list(merged[TARGET]), ['B'])

    def test_capitalises_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['DOG', 'DOG']
        )
        merged = merge_labels(iframe)
        expected = ['Dog']
        result = list(merged[LABEL])
        self.assertEqual(result, expected)

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog']
        )
        result = merge_labels(iframe).index
        self.assertTrue(result.equals(pd.Index([0])))

    def test_pmids_are_comma_joined(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1001'], None
        )
        merged = merge_labels(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000,1001'])
        self.assertEqual(psimis, ['None,None'])

    def test_psimi_order_is_not_mixed(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1001'], [None, 'MI:1|MI:2']
        )
        merged = merge_labels(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000,1001'])
        self.assertEqual(psimis, ['None,MI:1|MI:2'])

    def test_premerged_psimi_pmids_can_be_merged_again(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000,1001', '1001'], ['None,None', 'MI:1|MI:2']
        )
        merged = merge_labels(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000,1001'])
        self.assertEqual(psimis, ['None,MI:1|MI:2'])

        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000,1001', '1001'], ['MI:1,None', 'MI:1|MI:2']
        )
        merged = merge_labels(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000,1001'])
        self.assertEqual(psimis, ['MI:1,MI:1|MI:2'])

    def test_psimi_pmids_filter_out_nan(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
                ['1000', '1001'], [value, value]
            )
            merged = merge_labels(iframe)
            pmids = list(merged[PUBMED])
            psimis = list(merged[EXPERIMENT_TYPE])
            self.assertEqual(pmids, ['1000,1001'])
            self.assertEqual(psimis, ['None,None'])

    def test_nan_pmids_filtered_out(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
                [value, '1001'], ['mi:1', 'mi:2']
            )
            merged = merge_labels(iframe)
            pmids = list(merged[PUBMED])
            psimis = list(merged[EXPERIMENT_TYPE])
            self.assertEqual(pmids, ['1001'])
            self.assertEqual(psimis, ['MI:2'])

    def test_psimi_group_has_no_duplicates_and_is_sorted(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1000'], [None, 'MI:3|MI:2|mi:2']
        )
        merged = merge_labels(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, ['MI:2|MI:3'])

    def test_psimi_group_is_uppercase(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1000'], [None, '|mi:2']
        )
        merged = merge_labels(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, ['MI:2'])


class TestRemoveDuplicates(TestCase):

    def test_removes_identical_rows(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog']
        )
        result = remove_duplicates(iframe)
        self.assertEqual(list(result[LABEL]), ['Dog'])

    def test_doesnt_remove_if_label_is_different(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Cat']
        )
        result = remove_duplicates(iframe)
        self.assertEqual(list(result[LABEL]), ['Dog', 'Cat'])

    def test_can_handle_null_labels(self):
        for value1 in NULL_VALUES:
            for value2 in NULL_VALUES:
                iframe = make_interaction_frame(
                    ['B', 'B'], ['B', 'B'], [value1, value2]
                )
                result = remove_duplicates(iframe)
                self.assertEqual(list(result[LABEL]), [None])

    def test_can_detect_merged_labels(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog,Cat', 'Dog,Cat']
        )
        result = remove_duplicates(iframe)
        self.assertEqual(list(result[LABEL]), ['Cat,Dog'])

    def test_can_detect_merged_labels_out_of_order(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog,cat', 'cat,dog']
        )
        result = remove_duplicates(iframe)
        self.assertEqual(list(result[LABEL]), ['Cat,Dog'])

    def test_removes_reverse_duplicates(self):
        iframe = make_interaction_frame(
            ['B', 'A'], ['A', 'B'], ['dog', 'dog']
        )
        result = remove_duplicates(iframe)
        self.assertEqual(list(result[LABEL]), ['Dog'])

    def test_resets_index(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['dog', 'dog']
        )
        result = remove_duplicates(iframe).index
        self.assertTrue(pd.Index([0]).equals(result))

    def test_pmids_merged_when_removing_duplicate(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1001'], None
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000,1001'])
        self.assertEqual(psimis, ['None,None'])

    def test_pmid_duplicates_removed(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1000'], None
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, [None])

    def test_psimis_are_group_sorted(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1000'], ['mi:2', 'mi:1']
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, ['MI:1|MI:2'])

    def test_additional_columns_removes_trailing_whitespace(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            [' 1000', '1000 '], [' mi:2 ', ' mi:1 ']
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, ['MI:1|MI:2'])

    def test_psimi_order_is_not_mixed(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1001', '1000'], ['MI:0', 'MI:1|MI:2']
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1001,1000'])
        self.assertEqual(psimis, ['MI:0,MI:1|MI:2'])

    def test_premerged_psimi_pmids_can_be_merged_again(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000,1001', '1001'], ['MI:1,None', 'MI:1|MI:2']
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000,1001'])
        self.assertEqual(psimis, ['MI:1,MI:1|MI:2'])

    def test_psimi_pmids_filter_out_nan(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
                ['1000', '1001'], [value, value]
            )
            merged = remove_duplicates(iframe)
            pmids = list(merged[PUBMED])
            psimis = list(merged[EXPERIMENT_TYPE])
            self.assertEqual(pmids, ['1000,1001'])
            self.assertEqual(psimis, ['None,None'])

    def test_nan_pmids_filtered_out(self):
        for value in NULL_VALUES:
            iframe = make_interaction_frame(
                ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
                [value, '1001'], ['mi:1', 'mi:2']
            )
            merged = remove_duplicates(iframe)
            pmids = list(merged[PUBMED])
            psimis = list(merged[EXPERIMENT_TYPE])
            self.assertEqual(pmids, ['1001'])
            self.assertEqual(psimis, ['MI:2'])

    def test_psimi_group_has_no_duplicates_and_is_sorted(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1000'], [None, 'MI:3|MI:2|mi:2']
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, ['MI:2|MI:3'])

    def test_psimi_group_is_uppercase(self):
        iframe = make_interaction_frame(
            ['B', 'B'], ['B', 'B'], ['Dog', 'Dog'],
            ['1000', '1000'], [None, '|mi:2']
        )
        merged = remove_duplicates(iframe)
        pmids = list(merged[PUBMED])
        psimis = list(merged[EXPERIMENT_TYPE])
        self.assertEqual(pmids, ['1000'])
        self.assertEqual(psimis, ['MI:2'])


class TestProcessInteractions(TestCase):

    # def test_default_drop_nan_uses_source_target_and_label_as_column(self):
    #     iframe = make_interaction_frame(
    #         [None, 'B', 'C'],
    #         ['A', 'C', 'C'],
    #         ['1', '2', '3'],
    #         ['1', '2', None],
    #         ['1', None, '3']
    #     )
    #     expected = make_interaction_frame(
    #         ['B', 'C'],
    #         ['C', 'C'],
    #         ['2', '3'],
    #         ['2', None],
    #         [None, '3']
    #     )
    #     result = process_interactions(
    #         iframe, drop_nan='default', allow_self_edges=True,
    #         allow_duplicates=True, exclude_labels=None,
    #         min_counts=None, merge=False
    #     )
    #     self.assertTrue(dataframes_are_equal(result, expected))

    # def test_normalises_nan_values(self):
    #     for value in NULL_VALUES:
    #         iframe = make_interaction_frame(
    #             [value, 'B', 'C'],
    #             ['A', value, 'C'],
    #             ['1', '2', value],
    #             [value] * 3,
    #             [value] * 3
    #         )
    #         expected = make_interaction_frame(
    #             [None, 'B', 'C'],
    #             ['A', None, 'C'],
    #             ['1', '2', None],
    #             [None] * 3,
    #             [None] * 3
    #         )
    #         result = process_interactions(
    #             iframe, drop_nan=None, allow_self_edges=True,
    #             allow_duplicates=True, exclude_labels=None,
    #             min_counts=None, merge=False
    #         )
    #         self.assertTrue(dataframes_are_equal(expected, result))

    # def test_drop_nan_bypasses_when_none_is_suppled(self):
    #     iframe = make_interaction_frame(
    #         [None, 'B', 'C'],
    #         ['A', None, 'B'],
    #         ['1', '2', None],
    #         [None] * 3,
    #         [None] * 3
    #     )
    #     result = process_interactions(
    #         iframe, drop_nan=None, allow_self_edges=True,
    #         allow_duplicates=True, exclude_labels=None,
    #         min_counts=None, merge=False
    #     )
    #     self.assertTrue(dataframes_are_equal(iframe, result))

    # def test_drop_nan_uses_columns_when_supplied(self):
    #     iframe = make_interaction_frame(
    #         [None, 'B', 'C'],
    #         ['A', 'C', 'C'],
    #         ['1', '2', '3'],
    #         ['1', '2', '3'],
    #         ['1', None, '3']
    #     )
    #     result = process_interactions(
    #         iframe, drop_nan=[SOURCE, PUBMED, EXPERIMENT_TYPE],
    #         allow_self_edges=True,
    #         allow_duplicates=True, exclude_labels=None,
    #         min_counts=None, merge=False
    #     )
    #     expected = make_interaction_frame(
    #         ['C'], ['C'], ['3'], ['3'], ['3']
    #     )
    #     self.assertTrue(dataframes_are_equal(expected, result))

    def test_process_interactions_integration_test(self):
        interactions = []
        from io import StringIO
        file_ = StringIO(
            "source\ttarget\tlabel\tpmid\tpsimi\n"
            "O95398\tP51828\tactivation\t1\tMI:1\n"
            "O95398\tP51828\tbinding/association\t2,1\tNone,MI:2\n"
            "O43749\tO60262\tactivation\t3\tMI:3\n"
            "Q8NHA4\tQ13131\tactivation\t-\t-\n"
            "Q8NHA4\tQ13131\tbinding/association\tNaN\tMI:3\n"
            "P35626\tQ8NHA4\tphosphorylation\t\t\n"
            "Q8NHA4\tP35626\tphosphorylation\t\t\n"
            "P54840\tQ13131\tinhibition\t4\tMI:4\n"
            "P54840\tP54840\tinhibition\t4\tMI:4\n"
            "P35626\tO95398\tNone\tNone\tNone\n"
        )
        file_.readline()

        for line in file_:
            s, t, l, x, y = line.split('\t')
            interactions.append((s, t, l, x, y))

        iframe = make_interaction_frame(
            [s.strip() for s, *_ in interactions],
            [t.strip() for _, t, *_ in interactions],
            [l.strip() for _, _, l, *_ in interactions],
            [x.strip() for _, _, _, x, _ in interactions],
            [y.strip() for _, _, _, _, y in interactions]
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
                    'Activation,Binding/association',
                    'Activation',
                    'Activation,Binding/association'
                ],
                PUBMED: ["1,2", "3", None],
                EXPERIMENT_TYPE: ["MI:1|MI:2,None", "MI:3", None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
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
                LABEL: [None, None, None],
                PUBMED: [None, None, None],
                EXPERIMENT_TYPE: [None, None, None]

            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'D', 'C'],
                TARGET: ['C', 'D', 'C'],
                LABEL: [None, None, None],
                PUBMED: [None, None, None],
                EXPERIMENT_TYPE: [None, None, None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=None, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(dataframes_are_equal(result, expected))

    def test_can_map_one_to_many(self):
        network_map = {
            'A': ['C', 'D'],
        }
        iframe = pd.DataFrame(
            {
                SOURCE: ['A'],
                TARGET: ['A'],
                LABEL: [None],
                PUBMED: [None],
                EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'C', 'C', 'D'],
                TARGET: ['C', 'D', 'D', 'D'],
                LABEL: [None] * 4,
                PUBMED: [None] * 4,
                EXPERIMENT_TYPE: [None] * 4
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=None, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(dataframes_are_equal(result, expected))

    def test_will_correctly_map_meta_columns(self):
        network_map = {
            'A': ['C', 'D'],
        }
        iframe = pd.DataFrame(
            {
                SOURCE: ['A'],
                TARGET: ['A'],
                LABEL: [None],
                PUBMED: ['1,2'],
                EXPERIMENT_TYPE: ['MI:1|MI:2,None']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['C', 'C', 'C', 'D'],
                TARGET: ['C', 'D', 'D', 'D'],
                LABEL: [None] * 4,
                PUBMED: ['1,2'] * 4,
                EXPERIMENT_TYPE: ['MI:1|MI:2,None'] * 4
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        result = map_network_accessions(
            iframe, network_map, drop_nan=None, allow_self_edges=True,
            allow_duplicates=True, min_counts=None, merge=False
        )
        self.assertTrue(dataframes_are_equal(result, expected))
