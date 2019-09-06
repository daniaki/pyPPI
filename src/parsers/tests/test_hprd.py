from pathlib import Path
from typing import List

import pytest

from .. import hprd
from ..types import InteractionEvidenceData


class TestPTMEntry:
    def test_sets_enzyme_hprd_id_as_none_if_dash(self):
        entry = hprd.PTMEntry(enzyme_hprd_id="-")
        assert entry.enzyme_hprd_id is None

    def test_sets_substrate_hprd_id_as_none_if_dash(self):
        entry = hprd.PTMEntry(substrate_hprd_id="-")
        assert entry.substrate_hprd_id is None

    def test_sets_modification_type_as_none_if_dash(self):
        entry = hprd.PTMEntry(modification_type="-")
        assert entry.modification_type is None

    def test_converts_reference_id_to_list_if_str(self):
        entry = hprd.PTMEntry(reference_id="1234")
        assert entry.reference_id == ["1234"]

    def test_converts_reference_id_to_list_if_falsey(self):
        entry = hprd.PTMEntry(reference_id="")
        assert entry.reference_id == []

        entry = hprd.PTMEntry(reference_id=None)
        assert entry.reference_id == []

        entry = hprd.PTMEntry(reference_id="-")
        assert entry.reference_id == []

    def test_converts_experiment_type_to_list_if_str(self):
        entry = hprd.PTMEntry(experiment_type="1234")
        assert entry.experiment_type == ["1234"]

    def test_converts_experiment_type_to_list_if_falsey(self):
        entry = hprd.PTMEntry(experiment_type="")
        assert entry.experiment_type == []

        entry = hprd.PTMEntry(experiment_type=None)
        assert entry.experiment_type == []

        entry = hprd.PTMEntry(experiment_type="-")
        assert entry.experiment_type == []

    def test_adds_kwargs_to_dict(self):
        entry = hprd.PTMEntry(random_arg="Hello world")
        assert getattr(entry, "random_arg") == "Hello world"


class TestHPRDXrefEntry:
    def test_sets_falsey_gene_symbol_as_none(self):
        entry = hprd.HPRDXrefEntry(hprd_id=1, gene_symbol="")
        assert entry.gene_symbol is None

        entry = hprd.HPRDXrefEntry(hprd_id=1, gene_symbol="-")
        assert entry.gene_symbol is None

    def test_sets_falsey_swissprot_id_as_empty_list(self):
        entry = hprd.HPRDXrefEntry(hprd_id=1, swissprot_id="")
        assert entry.swissprot_id == []

        entry = hprd.HPRDXrefEntry(hprd_id=1, swissprot_id="-")
        assert entry.swissprot_id == []

    def test_adds_kwargs_to_dict(self):
        entry = hprd.HPRDXrefEntry(hprd_id=1, random_arg="Hello world")
        assert getattr(entry, "random_arg") == "Hello world"


class TestParsePTM:
    def test_parse_ptm_file_no_header(self):
        path = Path(__file__).parent / "data" / "hprd" / "hprd_ptms.tsv"
        entries: List[hprd.PTMEntry] = hprd.parse_ptm(
            path=path, header=False, sep="\t"
        )
        assert len(entries) == 2

        ptm1 = entries[0]
        assert ptm1.substrate_hprd_id == "03635"
        assert ptm1.substrate_gene_symbol == "SERPINB10"
        assert ptm1.substrate_isoform_id == "03635_1"
        assert ptm1.substrate_refseq_id == "NP_005015.1"
        assert ptm1.site == "307"
        assert ptm1.residue == "S"
        assert ptm1.enzyme_name == None
        assert ptm1.enzyme_hprd_id == None
        assert ptm1.modification_type == "Phosphorylation"
        assert ptm1.reference_id == ["17287340"]

        ptm2 = entries[1]
        assert ptm2.substrate_hprd_id == "03637"
        assert ptm2.substrate_gene_symbol == "TMPRSS2"
        assert ptm2.substrate_isoform_id == "03637_1"
        assert ptm2.substrate_refseq_id == "NP_005647.3"
        assert ptm2.site == "255"
        assert ptm2.residue == "R"
        assert ptm2.enzyme_name == "TMPRSS2"
        assert ptm2.enzyme_hprd_id == "03637"
        assert ptm2.modification_type == "Proteolytic Cleavage"
        assert ptm2.reference_id == ["11245484", "17287340"]

    def test_parse_ptm_file_with_header_header_clipped(self):
        path = Path(__file__).parent / "data" / "hprd" / "hprd_ptms.tsv"
        entries: List[hprd.PTMEntry] = hprd.parse_ptm(
            path=path, header=True, sep="\t"
        )
        assert len(entries) == 1

    def test_can_parse_mapping_into_mapping_entries_no_header(self):
        path = Path(__file__).parent / "data" / "hprd" / "hprd_mapping.tsv"
        xrefs: List[hprd.HPRDXrefEntry] = hprd.parse_xref_mapping(
            path=path, header=False, sep="\t"
        )
        assert len(xrefs) == 3

        xref1 = xrefs["03635"]
        assert xref1.hprd_id == "03635"
        assert xref1.gene_symbol == "SERPINB10"
        assert xref1.nucleotide_accession == "NM_005024.1"
        assert xref1.protein_accession == "NP_005015.1"
        assert xref1.entrezgene_id == "5273"
        assert xref1.omim_id == "602058"
        assert xref1.swissprot_id == []
        assert xref1.main_name == "Protease inhibitor 10"

        xref2 = xrefs["17574"]
        assert xref2.hprd_id == "17574"
        assert xref2.gene_symbol == "MGLL"
        assert xref2.nucleotide_accession == "NM_007283.5"
        assert xref2.protein_accession == "NP_009214.1"
        assert xref2.entrezgene_id == "11343"
        assert xref2.omim_id == "609699"
        assert xref2.swissprot_id == ["B3KRC2", "Q99685", "Q6IBG9"]
        assert xref2.main_name == "Monoglyceride lipase"

        xref3 = xrefs["03637"]
        assert xref3.hprd_id == "03637"
        assert xref3.gene_symbol == "TMPRSS2"
        assert xref3.nucleotide_accession == "NM_005656.3"
        assert xref3.protein_accession == "NP_005647.3"
        assert xref3.entrezgene_id == "7113"
        assert xref3.omim_id == "602060"
        assert xref3.swissprot_id == ["O15393"]
        assert xref3.main_name == "Transmembrane serine protease 2"

    def test_parse_mapping_file_with_header_header_clipped(self):
        path = Path(__file__).parent / "data" / "hprd" / "hprd_mapping.tsv"
        xrefs: List[hprd.HPRDXrefEntry] = hprd.parse_xref_mapping(
            path=path, header=True, sep="\t"
        )
        assert len(xrefs) == 2


class TestParseInteractions:
    def setup(self):
        self.ptms = [
            hprd.PTMEntry(
                substrate_hprd_id="03635",
                substrate_gene_symbol="SERPINB10",
                substrate_isoform_id="03635_1",
                substrate_refseq_id="NP_005015.1",
                site="307",
                residue="S",
                enzyme_name="SERPINB10",
                enzyme_hprd_id="03635",
                modification_type="Phosphorylation",
                experiment_type=["in vivo", "in vitro"],
                reference_id=["17287340"],
            ),
            hprd.PTMEntry(
                substrate_hprd_id="03637",
                substrate_gene_symbol="TMPRSS2",
                substrate_isoform_id="03637_1",
                substrate_refseq_id="NP_005647.3",
                site="255",
                residue="R",
                enzyme_name="TMPRSS2",
                enzyme_hprd_id="03637",
                modification_type="Proteolytic Cleavage",
                experiment_type=["in vivo", "in vitro"],
                reference_id=["11245484", "11245484"],
            ),
        ]
        self.xrefs = {
            "03635": hprd.HPRDXrefEntry(
                hprd_id="03635",
                gene_symbol="SERPINB10",
                nucleotide_accession="NM_005024.1",
                protein_accession="NP_005015.1",
                entrezgene_id="5273",
                omim_id="602058",
                swissprot_id=["B3KRC2", "Q99685", "Q6IBG9"],
                main_name="Protease inhibitor 10",
            ),
            "03637": hprd.HPRDXrefEntry(
                hprd_id="03637",
                gene_symbol="TMPRSS2",
                nucleotide_accession="NM_005656.3",
                protein_accession="NP_005647.3",
                entrezgene_id="7113",
                omim_id="602060",
                swissprot_id=["O15393"],
                main_name="Transmembrane serine protease 2",
            ),
        }

    def test_skips_if_missing_enzyme_id(self):
        ptm1 = hprd.PTMEntry(
            enzyme_hprd_id="-",
            substrate_hprd_id="03637",
            modification_type="A",
        )
        interactions = list(hprd.parse_interactions(ptms=[ptm1], xrefs=[]))
        assert len(interactions) == 0

    def test_skips_if_missing_substrate_id(self):
        ptm1 = hprd.PTMEntry(
            enzyme_hprd_id="03637",
            substrate_hprd_id="-",
            modification_type="A",
        )
        interactions = list(hprd.parse_interactions(ptms=[ptm1], xrefs=[]))
        assert len(interactions) == 0

    def test_skips_if_missing_label(self):
        ptm1 = hprd.PTMEntry(
            enzyme_hprd_id="03637",
            substrate_hprd_id="03637",
            modification_type=" ",
        )
        interactions = list(hprd.parse_interactions(ptms=[ptm1], xrefs=[]))
        assert len(interactions) == 0

    def test_skips_source_missing_uniprot_mapping(self):
        # Will filter out ptm@0
        self.xrefs[self.ptms[0].enzyme_hprd_id].swissprot_id = []
        interactions = list(
            hprd.parse_interactions(ptms=self.ptms, xrefs=self.xrefs)
        )
        assert len(interactions) == 1

    def test_skips_target_missing_uniprot_mapping(self):
        # Will filter out ptm@0
        self.xrefs[self.ptms[0].substrate_hprd_id].swissprot_id = []
        interactions = list(
            hprd.parse_interactions(ptms=self.ptms, xrefs=self.xrefs)
        )
        assert len(interactions) == 1

    def test_one_interaction_for_each_id_in_swissprot_id(self):
        interactions = list(
            hprd.parse_interactions(ptms=[self.ptms[0]], xrefs=self.xrefs)
        )
        assert len(interactions) == 9

    def test_sets_database_as_hprd(self):
        interactions = list(
            hprd.parse_interactions(ptms=[self.ptms[1]], xrefs=self.xrefs)
        )
        assert interactions[0].databases == ["hprd"]

    def test_lowercases_labels(self):
        interactions = list(
            hprd.parse_interactions(ptms=[self.ptms[1]], xrefs=self.xrefs)
        )
        assert interactions[0].labels == ["proteolytic cleavage"]

    def test_removes_duplicate_reference_ids(self):
        self.ptms[1].reference_id = ["12345", "12345"]
        interactions = list(
            hprd.parse_interactions(ptms=[self.ptms[1]], xrefs=self.xrefs)
        )
        assert interactions[0].evidence == [
            InteractionEvidenceData(pubmed="12345")
        ]

    def test_removes_falsey_reference_ids(self):
        self.ptms[1].reference_id = [" "]
        interactions = list(
            hprd.parse_interactions(ptms=[self.ptms[1]], xrefs=self.xrefs)
        )
        assert interactions[0].evidence == []

    def test_error_invalid_source(self):
        self.ptms[0].enzyme_hprd_id = "03635"
        self.xrefs["03635"].swissprot_id = ["AAA"]
        with pytest.raises(ValueError):
            hprd.parse_interactions(ptms=[self.ptms[0]], xrefs=self.xrefs)

    def test_error_invalid_target(self):
        self.ptms[0].substrate_hprd_id = "03635"
        self.xrefs["03635"].swissprot_id = ["AAA"]
        with pytest.raises(ValueError):
            hprd.parse_interactions(ptms=[self.ptms[0]], xrefs=self.xrefs)

    def test_formats_uniprot_ids(self):
        self.ptms[0].enzyme_hprd_id = "03635"
        self.ptms[0].substrate_hprd_id = "03637"
        self.xrefs["03635"].swissprot_id = [" p12345 "]
        self.xrefs["03637"].swissprot_id = [" p12346 "]

        interactions = list(
            hprd.parse_interactions(ptms=[self.ptms[0]], xrefs=self.xrefs)
        )
        assert len(interactions) == 1
        assert interactions[0].source == "P12345"
        assert interactions[0].target == "P12346"
