
import os
import pandas as pd
from unittest import TestCase

from ..base.constants import LABEL, SOURCE, TARGET, PUBMED, EXPERIMENT_TYPE
from ..base.constants import NULL_VALUES
from ..database import create_session, delete_database, cleanup_database
from ..database.models import Protein
from ..data_mining.kegg import (
    download_pathway_ids,
    pathway_to_dataframe,
    pathways_to_dataframe,
    keggid_to_uniprot
)

base_path = os.path.dirname(__file__)


def dataframes_are_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    df1 = df1.replace(to_replace=NULL_VALUES, value=str(None), inplace=False)
    df2 = df2.replace(to_replace=NULL_VALUES, value=str(None), inplace=False)
    return df1.equals(df2)


class TestKeggModule(TestCase):

    def setUp(self):
        self.db_path = os.path.normpath(
            "{}/databases/test.db".format(base_path)
        )
        self.records = open(os.path.normpath(
            "{}/test_data/test_sprot_records.dat".format(base_path)
        ), 'rt')
        self.pathway_iframe = pd.read_csv(
            "{}/test_data/hsa05202.csv".format(base_path)
        )
        self.session, self.engine = create_session(self.db_path)
        delete_database(self.session)

    def tearDown(self):
        delete_database(self.session)
        cleanup_database(self.session, self.engine)
        self.records.close()

    def test_download_human_pathways_all_strings(self):
        pathways = download_pathway_ids('hsa')
        self.assertTrue(len(pathways) > 0)
        self.assertTrue(
            all([isinstance(s, str) for s in pathways])
        )

    def test_can_download_non_hsa_pathways(self):
        pathways = download_pathway_ids('mus')
        self.assertTrue(len(pathways) > 0)
        self.assertTrue(
            all([isinstance(s, str) for s in pathways])
        )

    def test_can_parse_pathway(self):
        df = pathway_to_dataframe("path:hsa05202", org='hsa')
        self.assertTrue(dataframes_are_equal(self.pathway_iframe, df))

    def test_parse_pathway_filters_out_non_matching_ids(self):
        df = pathway_to_dataframe("path:hsa05202", org='mus')
        self.assertTrue(df.empty)

    def test_keggid_to_uniprot_maps_to_swissprot(self):
        protein = Protein('P43403', taxon_id=9606, reviewed=True)
        iframe = pd.DataFrame(
            {
                SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['P43403'], TARGET: ['P43403'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        protein.save(self.session, commit=True)
        result = keggid_to_uniprot(iframe, trembl=False)
        self.assertTrue(expected.equals(result))

    def test_keggid_to_uniprot_maps_to_multiple_uniprot(self):
        protein1 = Protein('P63092', taxon_id=9606, reviewed=True)
        protein2 = Protein('O95467', taxon_id=9606, reviewed=True)
        protein3 = Protein('A0A0S2Z3H8', taxon_id=9606, reviewed=False)
        iframe = pd.DataFrame(
            {
                SOURCE: ['hsa:2778'], TARGET: ['hsa:2778'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O95467', 'O95467', 'O95467', 'P63092'],
                TARGET: ['O95467', 'P63092', 'P63092', 'P63092'],
                LABEL: ['1', '1', '1', '1'],
                PUBMED: [None] * 4,
                EXPERIMENT_TYPE: [None] * 4
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )

        protein1.save(self.session, commit=True)
        protein2.save(self.session, commit=True)
        protein3.save(self.session, commit=True)
        result = keggid_to_uniprot(iframe, trembl=False)
        self.assertTrue(expected.equals(result))

    def test_keggid_to_uniprot_maps_filters_trembl(self):
        protein = Protein('P43403', taxon_id=9606, reviewed=False)
        iframe = pd.DataFrame(
            {
                SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        protein.save(self.session, commit=True)
        result = keggid_to_uniprot(iframe, trembl=False)
        self.assertTrue(result.empty)

    def test_keggid_to_uniprot_keeps_trembl_if_true(self):
        protein = Protein('P43403', taxon_id=9606, reviewed=False)
        iframe = pd.DataFrame(
            {
                SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['P43403'], TARGET: ['P43403'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )

        protein.save(self.session, commit=True)
        result = keggid_to_uniprot(iframe, trembl=True)
        self.assertTrue(expected.equals(result))

    def test_keggid_to_uniprot_ignores_accessions_not_in_database(self):
        iframe = pd.DataFrame(
            {
                SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1'],
                PUBMED: [None], EXPERIMENT_TYPE: [None]
            },
            columns=[SOURCE, TARGET, LABEL, PUBMED, EXPERIMENT_TYPE]
        )
        result = keggid_to_uniprot(iframe, verbose=True)
        self.assertTrue(result.empty)
