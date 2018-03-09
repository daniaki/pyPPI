"""
This module contains the paths to all data files used throughout other modules.
"""

__all__ = [
    "PATH",
    "hprd_mappings_txt",
    "hprd_ptms_txt",
    "uniprot_trembl_dat",
    "uniprot_sprot_dat",
    "default_db_path",
    "obo_file",
    "psimi_obo_file",
    "ipr_names_path",
    "pfam_names_path",
    "kegg_network_path",
    "hprd_network_path",
    "pina2_network_path",
    "bioplex_network_path",
    "innate_i_network_path",
    "innate_c_network_path",
    "testing_network_path",
    "training_network_path",
    "full_training_network_path",
    "interactome_network_path",
    "bioplex_v4_path",
    "innate_c_mitab_path",
    "innate_i_mitab_path",
    "pina2_sif_path",
    "feature_cache_path",
    "uniprot_map_path",
    "classifier_path",
]

import os

PATH = os.path.normpath(os.path.join(os.path.expanduser('~'), '.pyppi/'))

uniprot_trembl_dat = os.path.join(PATH, 'uniprot_trembl_human.dat.gz')
uniprot_sprot_dat = os.path.join(PATH, 'uniprot_sprot_human.dat.gz')
default_db_path = os.path.join(PATH, 'pyppi.db')
obo_file = os.path.join(PATH, 'go.obo.gz')
psimi_obo_file = os.path.join(PATH, 'mi.obo.gz')
ipr_names_path = os.path.join(PATH, 'entry.list')
pfam_names_path = os.path.join(PATH, 'Pfam-A.clans.tsv.gz')

hprd_ptms_txt = os.path.join(
    PATH, 'networks/POST_TRANSLATIONAL_MODIFICATIONS.txt')
hprd_mappings_txt = os.path.join(PATH, 'networks/HPRD_ID_MAPPINGS.txt')
kegg_network_path = os.path.join(PATH, 'networks/kegg_network.tsv')
hprd_network_path = os.path.join(PATH, 'networks/hprd_network.tsv')
pina2_network_path = os.path.join(PATH, 'networks/pina2_network.tsv')
bioplex_network_path = os.path.join(PATH, 'networks/bioplex_network.tsv')
innate_i_network_path = os.path.join(PATH, 'networks/innate_i_network.tsv')
innate_c_network_path = os.path.join(PATH, 'networks/innate_c_network.tsv')
testing_network_path = os.path.join(PATH, 'networks/testing_network.tsv')
training_network_path = os.path.join(PATH, 'networks/training_network.tsv')
full_training_network_path = os.path.join(
    PATH, 'networks/full_training_network.tsv')
interactome_network_path = os.path.join(
    PATH, 'networks/interactome_network.tsv')

bioplex_v4_path = os.path.join(
    PATH, 'networks/BioPlex_interactionList_v4a.tsv.gz')
innate_c_mitab_path = os.path.join(PATH, 'networks/innatedb_ppi.mitab.gz')
innate_i_mitab_path = os.path.join(PATH, 'networks/all.mitab.gz')
pina2_sif_path = os.path.join(PATH, 'networks/Homo-sapiens-20140521.tsv.gz')
pina2_mitab_path = os.path.join(PATH, 'networks/Homo-sapiens-20140521.tsv.gz')

feature_cache_path = os.path.join(PATH, 'features.json.gz')
uniprot_map_path = os.path.join(PATH, 'accession_map.json')
classifier_path = os.path.join(PATH, 'classifier.pkl')


# urls TODO: Add HPRD links to this
interpro_names_url = "ftp://ftp.ebi.ac.uk/pub/databases/interpro/entry.list"
mi_obo_url = "http://purl.obolibrary.org/obo/mi.obo"
go_obo_url = "http://purl.obolibrary.org/obo/go.obo"
uniprot_sp_human_url = (
    "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/taxonomic_divisions/uniprot_sprot_human.dat.gz"
)
uniprot_tr_human_url = (
    "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/taxonomic_divisions/uniprot_trembl_human.dat.gz"
)
pfam_clans_url = (
    "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"
)
pina2_mitab_url = (
    "http://omics.bjcancer.org/pina/download/Homo%20sapiens-20140521.tsv"
)
bioplex_url = (
    "http://bioplex.hms.harvard.edu/data/BioPlex_interactionList_v4a.tsv"
)
innate_curated_url = (
    "http://www.innatedb.com/download/interactions/innatedb_ppi.mitab.gz"
)
innate_imported_url = (
    "http://www.innatedb.com/download/interactions/all.mitab.gz"
)
