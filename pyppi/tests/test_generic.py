from io import StringIO
from unittest import TestCase

from ..data_mining.generic import (
    validate_accession,
    edgelist_func,
    bioplex_func,
    pina_func,
    mitab_func
)


class TestValidateAccession(TestCase):

    def test_returns_none_if_invalid(self):
        self.assertIsNone(validate_accession(''))
        self.assertIsNone(validate_accession(' '))
        self.assertIsNone(validate_accession('-'))
        self.assertIsNone(validate_accession('unknown'))

    def test_returns_upper_case_if_valid(self):
        self.assertEqual(validate_accession('a'), 'A')


class TestEdgeListFunc(TestCase):

    def test_elist_func_parses_correctly(self):
        fp = StringIO(
            "source\ttarget\n"
            "a\tb\n"
            "b\ta\n"
        )
        sources, targets, labels = edgelist_func(fp)
        self.assertEqual(sources, ['A', 'B'])
        self.assertEqual(targets, ['B', 'A'])
        self.assertEqual(labels, ['-', '-'])

    def test_elist_func_returns_none_for_invalid_accessions(self):
        fp = StringIO(
            "source\ttarget\n"
            "a\tunknown\n"
            "b\t-\n"
        )
        sources, targets, labels = edgelist_func(fp)
        self.assertEqual(sources, ['A', 'B'])
        self.assertEqual(targets, [None, None])
        self.assertEqual(labels, ['-', '-'])


class TestBioPlexFunc(TestCase):

    def test_parses_correctly(self):
        fp = StringIO(
            "GeneA\tGeneB\tUniprotA\tUniprotB\tSymbolA\tSymbolB\tp(Wrong)\tp(No Interaction)\tp(Interaction)\n"
            "100\t728378\tP00813\tA5A3E0\tADA\tPOTEF\t2.3\t0.0\t0.99\n"
            "100\t345651\tP00813\tQ562R1\tADA\tACTBL2\t9.7\t0.21\t0.78\n"
        )
        sources, targets, labels = bioplex_func(fp)
        self.assertEqual(sources, ['P00813', 'P00813'])
        self.assertEqual(targets, ['A5A3E0', 'Q562R1'])
        self.assertEqual(labels, ['-', '-'])

    def test_returns_none_for_invalid_accessions(self):
        fp = StringIO(
            "GeneA\tGeneB\tUniprotA\tUniprotB\tSymbolA\tSymbolB\tp(Wrong)\tp(No Interaction)\tp(Interaction)\n"
            "100\t728378\t \tA5A3E0\tADA\tPOTEF\t2.3\t0.0\t0.99\n"
            "100\t345651\tP00813\t \tADA\tACTBL2\t9.7\t0.21\t0.78\n"
        )
        sources, targets, labels = bioplex_func(fp)
        self.assertEqual(sources, [None, 'P00813'])
        self.assertEqual(targets, ['A5A3E0', None])
        self.assertEqual(labels, ['-', '-'])


class TestPina2Func(TestCase):

    def test_parses_correctly(self):
        fp = StringIO(
            "A0AV47 pp A0AV47\n"
            "A0FGR8 pp A0FGR9\n"
        )
        sources, targets, labels = pina_func(fp)
        self.assertEqual(sources, ['A0AV47', 'A0FGR8'])
        self.assertEqual(targets, ['A0AV47', 'A0FGR9'])
        self.assertEqual(labels, ['-', '-'])


class TestMitabFunc(TestCase):

    def test_parses_correctly(self):
        fp = StringIO(
            '#unique_identifier_A	unique_identifier_B	alt_identifier_A	alt_identifier_B	alias_A	alias_B	interaction_detection_method	author	pmid	ncbi_taxid_A	ncbi_taxid_B	interaction_type	source_database	idinteraction_in_source_db	confidence_score	expansion_method	biological_role_A	biological_role_B	exp_role_A	exp_role_B	interactor_type_A	interactor_type_B	xrefs_A	xrefs_B	xrefs_interaction	annotations_A	annotations_B	annotations_interaction	ncbi_taxid_host_organism	parameters_interaction	creation_date	update_date	checksum_A	checksum_B	checksum_interaction	negative	features_A	features_B	stoichiometry_A	stoichiometry_B	participant_identification_method_A	participant_identification_method_B\n'
            'innatedb:IDBG-25842	innatedb:IDBG-82738	ensembl:ENSG00000154589	ensembl:ENSG00000136869	uniprotkb:LY96_HUMAN|refseq:NP_056179|uniprotkb:Q9Y6Y9|refseq:NP_001182726|hgnc:LY96(display_short)	refseq:NP_612564|refseq:NP_612567|uniprotkb:O00206|uniprotkb:TLR4_HUMAN|refseq:NP_003257|hgnc:TLR4(display_short)	psi-mi:"MI:0007"(anti tag coimmunoprecipitation)	Shimazu et al. (1999)	pubmed:10359581	taxid:9606(Human)	taxid:9606(Human)	psi-mi:"MI:0915"(physical association)	MI:0974(innatedb)	innatedb:IDB-113240	lpr:3|hpr:3|np:1|	-	psi-mi:"MI:0499"(unspecified role)	psi-mi:"MI:0499"(unspecified role)	psi-mi:"MI:0498"(prey)	psi-mi:"MI:0496"(bait)	psi-mi:"MI:0326"(protein)	psi-mi:"MI:0326"(protein)	-	-	-	-	-	-	taxid:10090	-	2008/03/30	2008/03/30	-	-	-	false	-	-	-	-	psi-mi:"MI:0363"(inferred by author)	psi-mi:"MI:0363"(inferred by author)\n'
            'innatedb:IDBG-25713	innatedb:IDBG-82738	ensembl:ENSG00000172936	ensembl:ENSG00000136869	refseq:NP_002459|refseq:NP_001166039|refseq:NP_001166038|uniprotkb:MYD88_HUMAN|refseq:NP_001166040|uniprotkb:Q99836|refseq:NP_001166037|hgnc:MYD88(display_short)	refseq:NP_612564|refseq:NP_612567|uniprotkb:O00206|uniprotkb:TLR4_HUMAN|refseq:NP_003257|hgnc:TLR4(display_short)	psi-mi:"MI:0007"(anti tag coimmunoprecipitation)	Chaudhary et al. (2007)	pubmed:17228323	taxid:9606(Human)	taxid:9606(Human)	psi-mi:"MI:0915"(physical association)	MI:0974(innatedb)	innatedb:IDB-113241	lpr:5|hpr:5|np:1|	-	psi-mi:"MI:0499"(unspecified role)	psi-mi:"MI:0499"(unspecified role)	psi-mi:"MI:0496"(bait)	psi-mi:"MI:0498"(prey)	psi-mi:"MI:0326"(protein)	psi-mi:"MI:0326"(protein)	-	-	-	-	-	-	taxid:9606	-	2010/02/28	2010/02/28	-	-	-	false	-	-	-	-	psi-mi:"MI:0363"(inferred by author)	psi-mi:"MI:0363"(inferred by author)\n'
        )
        sources, targets, labels = mitab_func(fp)
        self.assertEqual(sources, ['Q9Y6Y9', 'Q99836'])
        self.assertEqual(targets, ['O00206', 'O00206'])
        self.assertEqual(labels, ['-', '-'])
