import os
import time

from unittest import TestCase
from Bio import SwissProt

from ..database import create_session, delete_database, cleanup_database
from ..database.models import Protein
from ..data_mining.uniprot import (
    parallel_download, download_record,
    parse_record_into_protein,
    go_terms, interpro_terms, pfam_terms,
    keywords, gene_name, recent_accession, taxonid,
    review_status, batch_map, function
)

base_path = os.path.dirname(__file__)


class TestUniProtMethods(TestCase):

    def setUp(self):
        self.records = open(os.path.normpath(
            "{}/test_data/test_sprot_records.dat".format(base_path)
        ), 'rt')
        self.db_path = '{}/databases/test.db'.format(base_path)
        self.session, self.engine = create_session(self.db_path)

    def tearDown(self):
        self.records.close()
        delete_database(self.session)
        cleanup_database(self.session, self.engine)

    def test_parses_gomf_correctly(self):
        for record in SwissProt.parse(self.records):
            result = go_terms(record, ont="mf")
            break
        expected = [
            'GO:0045296',
            'GO:0019899',
            'GO:0042826',
            'GO:0042802',
            'GO:0051219',
            'GO:0050815',
            'GO:0008022',
            'GO:0032403',
            'GO:0019904',
            'GO:0003714',
        ]
        self.assertEqual(result, expected)

    def test_parses_gobp_correctly(self):
        for record in SwissProt.parse(self.records):
            result = go_terms(record, ont="bp")
            break
        expected = [
            'GO:0051220',
            'GO:0035329',
            'GO:0000165',
            'GO:0061024',
            'GO:0045744',
            'GO:0035308',
            'GO:0045892',
            'GO:0043085',
            'GO:1900740',
            'GO:0051291',
            'GO:0006605',
            'GO:0043488',
            'GO:0016032',
        ]
        self.assertEqual(result, expected)

    def test_parses_gocc_correctly(self):
        for record in SwissProt.parse(self.records):
            result = go_terms(record, ont="cc")
            break
        expected = [
            'GO:0005737',
            'GO:0030659',
            'GO:0005829',
            'GO:0070062',
            'GO:0005925',
            'GO:0042470',
            'GO:0016020',
            'GO:0005739',
            'GO:0005634',
            'GO:0048471',
            'GO:0043234',
            'GO:0017053',
        ]
        self.assertEqual(result, expected)

    def test_parses_interpro_correctly(self):
        for record in SwissProt.parse(self.records):
            result = interpro_terms(record)
            break
        expected = [
            'IPR000308',
            'IPR023409',
            'IPR036815',
            'IPR023410',
        ]
        self.assertEqual(result, expected)

    def test_parses_pfam_correctly(self):
        for record in SwissProt.parse(self.records):
            result = pfam_terms(record)
            break
        expected = ['PF00244']
        self.assertEqual(result, expected)

    def test_parses_keywords_correctly(self):
        for record in SwissProt.parse(self.records):
            result = keywords(record)
            break
        expected = [
            '3D-structure', 'Acetylation', 'Alternative initiation',
            'Complete proteome', 'Cytoplasm', 'Direct protein sequencing',
            'Host-virus interaction', 'Isopeptide bond', 'Nitration',
            'Phosphoprotein', 'Polymorphism', 'Reference proteome',
            'Ubl conjugation'
        ]
        self.assertEqual(result, expected)

    def test_parses_review_status_correctly(self):
        for record in SwissProt.parse(self.records):
            result = review_status(record)
            break
        expected = 'Reviewed'
        self.assertEqual(result, expected)

    def test_parses_gene_name_correctly(self):
        for record in SwissProt.parse(self.records):
            result = gene_name(record)
            break
        expected = 'YWHAB'
        self.assertEqual(result, expected)

    def test_parses_taxonid_correctly(self):
        for record in SwissProt.parse(self.records):
            result = taxonid(record)
            break
        expected = 9606
        self.assertEqual(result, expected)

    def test_parses_recent_accession_correctly(self):
        for record in SwissProt.parse(self.records):
            result = recent_accession(record)
            break
        expected = 'P31946'
        self.assertEqual(result, expected)

    def test_parses_function_correctly(self):
        for record in SwissProt.parse(self.records):
            result = function(record)
            break
        self.assertIn("Adapter protein implicated in the regulation", result)

    def test_parses_function_as_None_for_entry_with_no_comment(self):
        for record in SwissProt.parse(self.records):
            r = record
            break
        r.comments = [x for x in r.comments if "FUNCTION: " not in x]
        result = function(r)
        expected = None
        self.assertEqual(result, expected)

    def test_can_parse_record_into_protein_objects(self):
        for record in SwissProt.parse(self.records):
            obj = parse_record_into_protein(record)
            break
        self.assertEqual(obj.uniprot_id, "P31946")
        self.assertEqual(obj.gene_id, "YWHAB")
        self.assertEqual(obj.reviewed, True)

    def test_returns_none_when_parsing_None_record(self):
        self.assertIsNone(parse_record_into_protein(None))

    def test_download_returns_None_if_taxids_not_matching(self):
        record = download_record('P48193', wait=1, retries=0)  # Mouse
        self.assertEqual(record, None)

    def test_download_returns_None_if_record_not_found(self):
        record = download_record('abc', wait=1, retries=0)  # Invalid Protein
        self.assertEqual(record, None)

    def test_can_parallel_download(self):
        accessions = ['P30443', 'O75581', 'P51828']
        records = parallel_download(accessions, n_jobs=3, wait=1, retries=0)
        entries = [parse_record_into_protein(r) for r in records]
        self.assertEqual(entries[0].uniprot_id, accessions[0])
        self.assertEqual(entries[1].uniprot_id, accessions[1])
        self.assertEqual(entries[2].uniprot_id, accessions[2])

    def test_batch_map_keeps_unreviewed(self):
        protein1 = Protein(
            uniprot_id='P50224', taxon_id=9606, reviewed=False)
        protein2 = Protein(
            uniprot_id='P0DMN0', taxon_id=9606, reviewed=True)
        protein3 = Protein(
            uniprot_id='P0DMM9', taxon_id=9606, reviewed=False)
        protein1.save(self.session, commit=True)
        protein2.save(self.session, commit=True)
        protein3.save(self.session, commit=True)

        mapping = batch_map(
            session=self.session, accessions=['P50224'], keep_unreviewed=True,
            match_taxon_id=None
        )
        self.assertEqual(mapping, {"P50224": ['P0DMM9', 'P0DMN0']})

    def test_batch_map_filters_unreviewed(self):
        protein1 = Protein(
            uniprot_id='P50224', taxon_id=9606, reviewed=False)
        protein2 = Protein(
            uniprot_id='P0DMN0', taxon_id=9606, reviewed=True)
        protein3 = Protein(
            uniprot_id='P0DMM9', taxon_id=9606, reviewed=False)
        protein1.save(self.session, commit=True)
        protein2.save(self.session, commit=True)
        protein3.save(self.session, commit=True)

        mapping = batch_map(
            session=self.session, accessions=['P50224'], keep_unreviewed=False,
            match_taxon_id=None
        )
        self.assertEqual(mapping, {"P50224": ["P0DMN0"]})

    def test_batch_map_filters_non_matching_taxon_ids(self):
        protein1 = Protein(
            uniprot_id='P50224', taxon_id=9606, reviewed=False)
        protein2 = Protein(
            uniprot_id='P0DMN0', taxon_id=9606, reviewed=True)
        protein3 = Protein(
            uniprot_id='P0DMM9', taxon_id=9606, reviewed=False)
        protein1.save(self.session, commit=True)
        protein2.save(self.session, commit=True)
        protein3.save(self.session, commit=True)

        mapping = batch_map(
            session=self.session, accessions=['P50224'], keep_unreviewed=False,
            match_taxon_id=0
        )
        self.assertEqual(mapping, {"P50224": []})

    def test_batch_map_filters_keeps_matching_taxon_ids(self):
        protein1 = Protein(
            uniprot_id='P50224', taxon_id=9606, reviewed=False)
        protein2 = Protein(
            uniprot_id='P0DMN0', taxon_id=9606, reviewed=True)
        protein3 = Protein(
            uniprot_id='P0DMM9', taxon_id=9606, reviewed=False)
        protein1.save(self.session, commit=True)
        protein2.save(self.session, commit=True)
        protein3.save(self.session, commit=True)

        mapping = batch_map(
            session=self.session, accessions=['P50224'], keep_unreviewed=True,
            match_taxon_id=9606
        )
        self.assertEqual(mapping, {"P50224": ['P0DMM9', 'P0DMN0']})

    def test_batch_map_downloads_missing_records(self):
        mapping = batch_map(
            session=self.session, accessions=['P50224'], keep_unreviewed=True,
            match_taxon_id=9606, allow_download=True
        )
        self.assertEqual(mapping, {"P50224": ['P0DMM9', 'P0DMN0']})

    def test_batch_map_doesnt_save_invalid_record(self):
        mapping = batch_map(
            session=self.session, accessions=['P50224'], match_taxon_id=0,
            allow_download=True
        )
        self.assertEqual(mapping, {"P50224": []})

    def test_batch_return_empty_list_if_accession_maps_to_invalid_record(self):
        mapping = batch_map(
            session=self.session, accessions=['Q02248'], match_taxon_id=9606
        )
        self.assertEqual(mapping, {"Q02248": []})
