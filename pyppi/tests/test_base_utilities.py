import os
import shutil
import pandas as pd
from unittest import TestCase
from collections import namedtuple

from ..base.constants import (
    NULL_VALUES, SOURCE, TARGET, EXPERIMENT_TYPE, PUBMED, LABEL
)
from ..base.utilities import (
    is_null, su_make_dir, get_term_description,
    take, remove_duplicates, rename, chunk_list,
    generate_interaction_tuples

)

from ..database import create_session, delete_database, cleanup_database
from ..database.exceptions import ObjectAlreadyExists, ObjectNotFound
from ..database.models import (
    Protein, Interaction, Psimi, Pubmed
)

base_path = os.path.dirname(__file__)


class TestUtilityFunctions(TestCase):

    def test_null_recognises_each_entry_in_NULL_VALUES(self):
        for v in NULL_VALUES:
            self.assertTrue(is_null(v))

    def test_can_make_directory_with_full_read_write_rights(self):
        path = os.path.normpath(
            base_path + '/test_data/' + '/test_dir/'
        )
        su_make_dir(path)
        self.assertTrue(os.access(path, os.W_OK))
        self.assertTrue(os.access(path, os.R_OK))
        shutil.rmtree(path)

    def test_take_returns_first_n(self):
        seq = [1, 2, 3, 4]
        result = take(2, seq)
        expected = [1, 2]
        self.assertEqual(result, expected)

    def test_take_returns_whole_list_if_n_greater_than_len(self):
        seq = [1, 2, 3, 4]
        result = take(10, seq)
        self.assertEqual(result, seq)

    def test_take_returns_empty_list_if_n_is_zero(self):
        seq = [1, 2, 3, 4]
        result = take(0, seq)
        self.assertEqual(result, [])

    def test_can_remove_duplicates_in_order(self):
        seq = [1, 1, 'dog', 'Dog', 'dog', 'cat']
        result = remove_duplicates(seq)
        self.assertEqual(result, [1, 'dog', 'Dog', 'cat'])

    def test_get_term_desc_uses_upper_case_lookup(self):
        Node = namedtuple('Node', ['name'])
        dag = {'GO:001': Node(name='go 1')}
        ipr_name_map = {"IPR1": 'ipr 1'}
        pfam_name_map = {'PF1': 'pf 1'}

        result = get_term_description('ipr1', dag, ipr_name_map, pfam_name_map)
        self.assertTrue(result, 'ipr 1')

        result = get_term_description('PF1', dag, ipr_name_map, pfam_name_map)
        self.assertTrue(result, 'pf 1')

    def test_get_term_desc_can_search_go_terms_without_colon(self):
        Node = namedtuple('Node', ['name'])
        dag = {'GO:001': Node(name='go 1')}
        ipr_name_map = {"IPR1": 'ipr 1'}
        pfam_name_map = {'PF1': 'pf 1'}

        result = get_term_description(
            'go001', dag, ipr_name_map, pfam_name_map)
        self.assertTrue(result, 'go 1')

        result = get_term_description(
            'go:001', dag, ipr_name_map, pfam_name_map)
        self.assertTrue(result, 'go 1')

    def test_rename_upper_cases(self):
        result = rename("ipr1")
        self.assertTrue(result, 'IPR1')

        result = rename("pf1")
        self.assertTrue(result, 'PF1')

        result = rename("go:1")
        self.assertTrue(result, 'GO:1')

    def test_rename_re_colonises_go_terms(self):
        result = rename("go1")
        self.assertTrue(result, 'GO:1')

    def test_rename_doesnt_add_double_colons(self):
        result = rename("go:1")
        self.assertTrue(result, 'GO:1')

    def test_chunk_list_splits_list_int_n_sublists(self):
        seq = list(range(100))
        chunks = chunk_list(seq, 10)
        for i, chunk in enumerate(chunks):
            self.assertEqual(len(chunk), 10)
            self.assertEqual(chunk, seq[10*i: 10*(i + 1)])

    def test_chunk_list_last_chunk_takes_remainder1(self):
        seq = list(range(95))  # should split into sublists of 10 except last
        chunks = list(chunk_list(seq, 10))
        total = 0
        self.assertEqual(len(chunks), 10)
        for i, chunk in enumerate(chunks):
            if i < 5:
                self.assertEqual(len(chunk), 10)
                total += len(chunk)
            else:
                self.assertEqual(len(chunk), 9)
                total += len(chunk)

        self.assertEqual(total, len(seq))

    def test_chunk_list_last_chunk_takes_remainder2(self):
        seq = list(range(101))  # should split into sublists of 10 except last
        chunks = list(chunk_list(seq, 10))
        total = 0

        self.assertEqual(len(chunks), 10)
        for i, chunk in enumerate(chunks):
            if i == 0:
                self.assertEqual(len(chunk), 11)
                total += len(chunk)
            else:
                self.assertEqual(len(chunk), 10)
                total += len(chunk)

        self.assertEqual(total, len(seq))

    def test_chunk_list_last_chunk_takes_remainder3(self):
        seq = list(range(91))
        chunks = list(chunk_list(seq, 10))
        total = 0

        self.assertEqual(len(chunks), 10)
        for i, chunk in enumerate(chunks):
            if i == 0:
                self.assertEqual(len(chunk), 10)
                total += len(chunk)
            else:
                self.assertEqual(len(chunk), 9)
                total += len(chunk)

        self.assertEqual(total, len(seq))

    def test_chunk_return_whole_list_if_n_is_1(self):
        seq = list(range(100))
        chunks = list(chunk_list(seq, 1))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), len(seq))

    def test_chunk_return_individual_items_in_list_if_n_equal_len(self):
        seq = list(range(100))
        chunks = list(chunk_list(seq, 100))
        self.assertEqual(len(chunks), 100)

    def test_chunk_return_valueerror_n_less_than_1(self):
        with self.assertRaises(ValueError):
            list(chunk_list(range(100), 0))

    def test_chunk_return_valueerror_n_gt_than_len(self):
        with self.assertRaises(ValueError):
            list(chunk_list(range(100), 101))


class TestTupleGenerator(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        self.psi_a = Psimi(accession="MI:1", description="blah")
        self.psi_b = Psimi(accession="MI:2", description="blah")
        self.pm_a = Pubmed(accession="1")
        self.pm_b = Pubmed(accession="2")
        self.psi_a.save(self.session, commit=True)
        self.psi_b.save(self.session, commit=True)
        self.pm_a.save(self.session, commit=True)
        self.pm_b.save(self.session, commit=True)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_can_generate_tuples(self):
        df = pd.DataFrame({
            SOURCE: ['A', 'B'],
            TARGET: ['A', 'B'],
            LABEL: ['1', '2'],
            PUBMED: ['1000,1001', '1000'],
            EXPERIMENT_TYPE: ['None,MI:1|MI:2', 'MI:1|MI:2']
        })
        result = list(generate_interaction_tuples(df))
        self.assertEqual(
            result[0],
            ('A', 'A', '1', ['1000', '1001'], [[None], ['MI:1', 'MI:2']])
        )
        self.assertEqual(
            result[1],
            ('B', 'B', '2', ['1000'], [['MI:1', 'MI:2']])
        )

    def test_generate_tuples_formats_none(self):
        for value in NULL_VALUES:
            df = pd.DataFrame({
                SOURCE: ['A'],
                TARGET: ['A'],
                LABEL: [value],
                PUBMED: [value],
                EXPERIMENT_TYPE: ['1']
            })
            result = list(generate_interaction_tuples(df))
            self.assertEqual(result[0], ('A', 'A', None, [None], [None]))
