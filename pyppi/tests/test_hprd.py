
import os
import pandas as pd
import numpy as np
from unittest import TestCase

from ..base.constants import LABEL, SOURCE, TARGET, PUBMED, EXPERIMENT_TYPE
from ..database.models import Protein
from ..database import create_session, cleanup_database, delete_database
from ..data_mining.tools import make_interaction_frame
from ..data_mining.hprd import (
    parse_ptm,
    parse_hprd_mapping,
    hprd_to_dataframe
)

base_path = os.path.dirname(__file__)


class TestHPRDModule(TestCase):

    def setUp(self):
        self.hprd_mapping = open(
            "{}/test_data/{}".format(base_path, 'hprd_mapping.tsv'),
            'rt'
        )
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        p1 = Protein(uniprot_id='P48595', taxon_id=9606, reviewed=True)
        p2 = Protein(uniprot_id='B3KRC2', taxon_id=9606, reviewed=False)
        p3 = Protein(uniprot_id='Q99685', taxon_id=9606, reviewed=True)
        p4 = Protein(uniprot_id='O15393', taxon_id=9606, reviewed=True)
        for p in [p1, p2, p3, p4]:
            p.save(self.session, commit=True)

        self.handles = []

    def tearDown(self):
        self.hprd_mapping.close()
        for h in self.handles:
            h.close()
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_can_parse_hprd_line_into_ptm_entry(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_ptm_parse_test.tsv'),
            'rt'
        )
        self.handles.append(file_input)

        ptms = parse_ptm(
            file_input=file_input,
            header=False,
            col_sep='\t'
        )
        self.assertEqual(len(ptms), 2)

        ptm1 = ptms[0]
        self.assertEqual(ptm1.substrate_hprd_id, '03635')
        self.assertEqual(ptm1.substrate_gene_symbol, 'SERPINB10')
        self.assertEqual(ptm1.substrate_isoform_id, '03635_1')
        self.assertEqual(ptm1.substrate_refseq_id, 'NP_005015.1')
        self.assertEqual(ptm1.site, '307')
        self.assertEqual(ptm1.residue, 'S')
        self.assertEqual(ptm1.enzyme_name, '-')
        self.assertEqual(ptm1.enzyme_hprd_id, '-')
        self.assertEqual(ptm1.modification_type, 'Phosphorylation')
        self.assertEqual(ptm1.experiment_type, ['in vivo', 'in vitro'])
        self.assertEqual(ptm1.reference_id, ['17287340'])

        ptm2 = ptms[1]
        self.assertEqual(ptm2.substrate_hprd_id, '03637')
        self.assertEqual(ptm2.substrate_gene_symbol, 'TMPRSS2')
        self.assertEqual(ptm2.substrate_isoform_id, '03637_1')
        self.assertEqual(ptm2.substrate_refseq_id, 'NP_005647.3')
        self.assertEqual(ptm2.site, '255')
        self.assertEqual(ptm2.residue, 'R')
        self.assertEqual(ptm2.enzyme_name, 'TMPRSS2')
        self.assertEqual(ptm2.enzyme_hprd_id, '03637')
        self.assertEqual(ptm2.modification_type, 'Proteolytic Cleavage')
        self.assertEqual(ptm2.experiment_type, ['in vitro'])
        self.assertEqual(ptm2.reference_id, ['11245484'])

    def test_can_parse_mapping_into_mapping_entry(self):
        xrefs = parse_hprd_mapping(
            file_input=self.hprd_mapping,
            header=False,
            col_sep='\t'
        )
        self.assertEqual(len(xrefs), 3)

        xref1 = xrefs['03635']
        self.assertEqual(xref1.hprd_id, '03635')
        self.assertEqual(xref1.gene_symbol, 'SERPINB10')
        self.assertEqual(xref1.nucleotide_accession, 'NM_005024.1')
        self.assertEqual(xref1.protein_accession, 'NP_005015.1')
        self.assertEqual(xref1.entrezgene_id, '5273')
        self.assertEqual(xref1.omim_id, '602058')
        self.assertEqual(xref1.swissprot_id, ['P48595'])
        self.assertEqual(xref1.main_name, 'Protease inhibitor 10')

        xref2 = xrefs['17574']
        self.assertEqual(xref2.hprd_id, '17574')
        self.assertEqual(xref2.gene_symbol, 'MGLL')
        self.assertEqual(xref2.nucleotide_accession, 'NM_007283.5')
        self.assertEqual(xref2.protein_accession, 'NP_009214.1')
        self.assertEqual(xref2.entrezgene_id, '11343')
        self.assertEqual(xref2.omim_id, '609699')
        self.assertEqual(xref2.swissprot_id, ['B3KRC2', 'Q99685', 'Q6IBG9'])
        self.assertEqual(
            xref2.main_name,
            'Monoglyceride lipase'
        )

        xref3 = xrefs['03637']
        self.assertEqual(xref3.hprd_id, '03637')
        self.assertEqual(xref3.gene_symbol, 'TMPRSS2')
        self.assertEqual(xref3.nucleotide_accession, 'NM_005656.3')
        self.assertEqual(xref3.protein_accession, 'NP_005647.3')
        self.assertEqual(xref3.entrezgene_id, '7113')
        self.assertEqual(xref3.omim_id, '602060')
        self.assertEqual(xref3.swissprot_id, ['O15393'])
        self.assertEqual(
            xref3.main_name,
            'Transmembrane serine protease 2'
        )

    def test_hprd_replaces_unreviewed_not_existing_uniprot_with_None(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_unreviewed.tsv'),
            'rt'
        )
        self.handles.append(file_input)

        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan=None,
            mapping_input=self.hprd_mapping,
            allow_duplicates=True
        )
        expected = make_interaction_frame(
            [None, 'O15393', None], ['O15393', 'Q99685', 'O15393'],
            ['phosphorylation'] * 3, ['19608861']*3, [None]*3,
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_keeps_duplicates_if_allow_is_true(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_duplicate.tsv'),
            'rt'
        )
        self.handles.append(file_input)

        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan='default',
            mapping_input=self.hprd_mapping,
            allow_duplicates=True
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393', 'O15393'], TARGET: ['Q99685', 'Q99685'],
                LABEL: ['Acetylation', 'Acetylation'],
                EXPERIMENT_TYPE: [None, None],
                PUBMED: ['19608861', '19608861']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_filters_duplicates_if_allow_is_false(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_duplicate.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan='default',
            mapping_input=self.hprd_mapping,
            allow_duplicates=False
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393'], TARGET: ['Q99685'],
                LABEL: ['Acetylation'],
                EXPERIMENT_TYPE: [None],
                PUBMED: ['19608861']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_allows_self_edges_if_true(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_self_edges.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan='default',
            mapping_input=self.hprd_mapping,
            allow_self_edges=True
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393'], TARGET: ['O15393'],
                LABEL: ['Acetylation'],
                EXPERIMENT_TYPE: [None],
                PUBMED: ['19608861']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_will_drop_nan_if_drop_nan_supplied(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_drop_nan.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan='default',
            mapping_input=self.hprd_mapping,
            allow_self_edges=True
        )
        self.assertTrue(iframe.empty)

    def test_hprd_will_ignore_nan_if_drop_nan_is_none(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_drop_nan.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        expected = make_interaction_frame(
            [None, 'O15393'], ['O15393', 'O15393'], ['Acetylation', None],
            ['19608861', '19608861'], [None, None]
        )
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan=None,
            mapping_input=self.hprd_mapping,
            allow_self_edges=True
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_will_ignore_nan_if_drop_nan_is_empty(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_drop_nan.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        expected = make_interaction_frame(
            [None, 'O15393'], ['O15393', 'O15393'], ['Acetylation', None],
            ['19608861', '19608861'], [None, None]
        )
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan=[],
            mapping_input=self.hprd_mapping,
            allow_self_edges=True
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_filters_self_edges_if_false(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_self_edges.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan='default',
            mapping_input=self.hprd_mapping,
            allow_self_edges=False
        )
        self.assertTrue(iframe.empty)

    def test_hprd_to_dataframe_filters_nan(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_drop_nan.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            drop_nan='default',
            ptm_input=file_input,
            mapping_input=self.hprd_mapping,
        )
        self.assertTrue(iframe.empty)

    def test_hprd_to_dataframe_can_exclude_labels(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_exclude_label.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            drop_nan='default',
            ptm_input=file_input,
            mapping_input=self.hprd_mapping,
            exclude_labels=['acetylation']
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393'], TARGET: ['Q99685'],
                LABEL: ['Phosphorylation'],
                EXPERIMENT_TYPE: [None],
                PUBMED: ['19608861']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_does_not_exclude_multilabel_samples(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_exclude_label_ml.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            drop_nan='default',
            ptm_input=file_input,
            mapping_input=self.hprd_mapping,
            exclude_labels=['acetylation']
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393'], TARGET: ['Q99685'],
                LABEL: ['Acetylation,Phosphorylation'],
                EXPERIMENT_TYPE: [None],
                PUBMED: ['19608861']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_can_merge_labels(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_merge.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            drop_nan='default',
            ptm_input=file_input,
            mapping_input=self.hprd_mapping,
            merge=True
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393'], TARGET: ['Q99685'],
                LABEL: ['Acetylation,Phosphorylation'],
                EXPERIMENT_TYPE: ['None,None'],
                PUBMED: ['19608861,17808861']
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_min_count_filter_less_than_min_count(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_min_count.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            ptm_input=file_input,
            drop_nan='default',
            mapping_input=self.hprd_mapping,
            min_label_count=2
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393', 'P48595'], TARGET: ['Q99685'] * 2,
                LABEL: ['Acetylation'] * 2,
                EXPERIMENT_TYPE: [None] * 2,
                PUBMED: ['19608861'] * 2
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))

    def test_hprd_to_dataframe_min_count_filter_occurs_before_merge(self):
        file_input = open(
            "{}/test_data/{}".format(base_path, 'hprd_mincount_merge.tsv'),
            'rt'
        )
        self.handles.append(file_input)
        iframe = hprd_to_dataframe(
            drop_nan='default',
            ptm_input=file_input,
            mapping_input=self.hprd_mapping,
            min_label_count=2,
            merge=True
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O15393', 'P48595'], TARGET: ['Q99685'] * 2,
                LABEL: ['Acetylation'] * 2,
                EXPERIMENT_TYPE: [None] * 2,
                PUBMED: ['19608861'] * 2
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        self.assertTrue(expected.equals(iframe))
