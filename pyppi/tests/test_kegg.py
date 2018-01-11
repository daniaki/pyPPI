
import os
import pandas as pd
from unittest import TestCase

from ..base import LABEL, SOURCE, TARGET
from ..database import begin_transaction
from ..database.managers import ProteinManager
from ..database.models import Protein
from ..data_mining.kegg import (
    download_pathway_ids,
    pathway_to_dataframe,
    pathways_to_dataframe,
    keggid_to_uniprot
)

base_path = os.path.dirname(__file__)


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

    def tearDown(self):
        with begin_transaction(db_path=self.db_path) as session:
            session.rollback()
            session.execute("DROP TABLE {}".format(Protein.__tablename__))
        self.records.close()

    def test_download_human_pathways_all_strings(self):
        pathways = download_pathway_ids('hsa')
        self.assertTrue(len(pathways) > 0)
        self.assertTrue(
            all([isinstance(s, str) for s in pathways])
        )

    def test_can_download_non_pathways(self):
        pathways = download_pathway_ids('mus')
        self.assertTrue(len(pathways) > 0)
        self.assertTrue(
            all([isinstance(s, str) for s in pathways])
        )

    def test_can_parse_pathway(self):
        pathways = download_pathway_ids('hsa')
        for p in pathways:
            if p == "path:hsa05202":
                df = pathway_to_dataframe(p)
        self.assertTrue(self.pathway_iframe.equals(df))

    def test_keggid_to_uniprot_maps_to_swissprot(self):
        protein = Protein('P43403', taxon_id=9606, reviewed=True)
        iframe = pd.DataFrame(
            {SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {SOURCE: ['P43403'], TARGET: ['P43403'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )

        with begin_transaction(self.db_path) as session:
            protein.save(session, commit=True)
            result = keggid_to_uniprot(
                session, iframe, trembl=False
            )
        self.assertTrue(expected.equals(result))

    def test_keggid_to_uniprot_maps_to_multiple_uniprot(self):
        protein1 = Protein('P63092', taxon_id=9606, reviewed=True)
        protein2 = Protein('O95467', taxon_id=9606, reviewed=True)
        protein3 = Protein('A0A0S2Z3H8', taxon_id=9606, reviewed=False)
        iframe = pd.DataFrame(
            {SOURCE: ['hsa:2778'], TARGET: ['hsa:2778'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {
                SOURCE: ['O95467', 'O95467', 'O95467', 'P63092'],
                TARGET: ['O95467', 'P63092', 'P63092', 'P63092'],
                LABEL: ['1', '1', '1', '1']
            },
            columns=[SOURCE, TARGET, LABEL]
        )

        with begin_transaction(self.db_path) as session:
            protein1.save(session, commit=True)
            protein2.save(session, commit=True)
            protein3.save(session, commit=True)
            result = keggid_to_uniprot(
                session, iframe, trembl=False
            )
        self.assertTrue(expected.equals(result))

    def test_keggid_to_uniprot_maps_filters_trembl(self):
        protein = Protein('P43403', taxon_id=9606, reviewed=False)
        iframe = pd.DataFrame(
            {SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        with begin_transaction(self.db_path) as session:
            protein.save(session, commit=True)
            result = keggid_to_uniprot(
                session, iframe, trembl=False
            )
        self.assertTrue(result.empty)

    def test_keggid_to_uniprot_keeps_trembl_if_true(self):
        protein = Protein('P43403', taxon_id=9606, reviewed=False)
        iframe = pd.DataFrame(
            {SOURCE: ['hsa:7535'], TARGET: ['hsa:7535'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )
        expected = pd.DataFrame(
            {SOURCE: ['P43403'], TARGET: ['P43403'], LABEL: ['1']},
            columns=[SOURCE, TARGET, LABEL]
        )

        with begin_transaction(self.db_path) as session:
            protein.save(session, commit=True)
            result = keggid_to_uniprot(
                session, iframe, trembl=True
            )

        self.assertTrue(expected.equals(result))
