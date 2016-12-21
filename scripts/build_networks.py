#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.
"""
import pandas as pd

from pyPPI.data import bioplex_network_path, pina2_network_path
from pyPPI.data import bioplex_v4, pina2, innate_curated, innate_imported
from pyPPI.data import innate_i_network_path, innate_c_network_path
from pyPPI.data import interactome_network_path
from pyPPI.data import kegg_network_path, hprd_network_path
from pyPPI.data import load_go_dag
from pyPPI.data import load_uniprot_accession_map, save_uniprot_accession_map
from pyPPI.data import testing_network_path, training_network_path
from pyPPI.data import save_network_to_path
from pyPPI.data import save_ptm_labels
from pyPPI.data import save_accession_features, save_ppi_features

from pyPPI.data_mining.features import AnnotationExtractor
from pyPPI.data_mining.generic import bioplex_func, mitab_func, pina_func
from pyPPI.data_mining.generic import generic_to_dataframe
from pyPPI.data_mining.hprd import hprd_to_dataframe
from pyPPI.data_mining.tools import process_interactions
from pyPPI.data_mining.tools import remove_intersection, remove_labels
from pyPPI.data_mining.tools import map_network_accessions
from pyPPI.data_mining.uniprot import UniProt, get_active_instance
from pyPPI.data_mining.kegg import download_pathway_ids, pathways_to_dataframe

if __name__ == '__main__':
    uniprot = get_active_instance(verbose=True)
    data_types = UniProt.data_types()
    selection = [data_types.GO, data_types.INTERPRO, data_types.PFAM]
    pathways = download_pathway_ids('hsa')
    update = False
    n_jobs = 4

    # Construct all the networks
    print("Building KEGG interactions...")
    kegg = pathways_to_dataframe(
        pathway_ids=pathways,
        map_to_uniprot=True,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    print("Building HPRD interactions...")
    hprd = hprd_to_dataframe(
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    print("Building Interactome interactions...")
    bioplex = generic_to_dataframe(
        f_input=bioplex_v4(),
        parsing_func=bioplex_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )
    pina2 = generic_to_dataframe(
        f_input=pina2(),
        parsing_func=pina_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )
    innate_c = generic_to_dataframe(
        f_input=innate_curated(),
        parsing_func=mitab_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )
    innate_i = generic_to_dataframe(
        f_input=innate_imported(),
        parsing_func=mitab_func,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False
    )

    print("Mapping to most recent uniprot accessions...")
    # Get a set of all the unique uniprot accessions
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]
    sources = set(p for df in networks for p in df.source.values)
    targets = set(p for df in networks for p in df.target.values)
    accessions = list(sources | targets)

    if update:
        accession_mapping = uniprot.batch_map(accessions)
        save_uniprot_accession_map(accession_mapping)
    else:
        try:
            accession_mapping = load_uniprot_accession_map()
        except IOError:
            accession_mapping = uniprot.batch_map(accessions)
            save_uniprot_accession_map(accession_mapping)

    print("Mapping each network to the most recent uniprot accessions...")
    kegg = map_network_accessions(
        interactions=kegg, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    hprd = map_network_accessions(
        interactions=hprd, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )

    pina2 = map_network_accessions(
        interactions=pina2, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    bioplex = map_network_accessions(
        interactions=bioplex, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    innate_c = map_network_accessions(
        interactions=innate_c, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    innate_i = map_network_accessions(
        interactions=innate_i, accession_map=accession_mapping,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=False
    )
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]

    print("Building features for each protein and PPI...")
    ae = AnnotationExtractor(
        uniprot=uniprot,
        godag=load_go_dag(),
        induce=True,
        selection=selection,
        n_jobs=1,
        verbose=True,
        cache=True
    )

    for df in networks:
        ppis = list(zip(df.source, df.target))
        ae.fit(ppis)

    print("Saving networks and feature files...")
    save_network_to_path(kegg, kegg_network_path)
    save_network_to_path(hprd, hprd_network_path)
    save_network_to_path(pina2, pina2_network_path)
    save_network_to_path(bioplex, bioplex_network_path)
    save_network_to_path(innate_i, innate_i_network_path)
    save_network_to_path(innate_c, innate_c_network_path)

    test_labels = ['dephosphorylation', 'phosphorylation']
    train_labels = [l for l in hprd.label if l not in test_labels]
    train_hprd = remove_labels(hprd, test_labels)

    testing = remove_intersection(remove_labels(hprd, train_labels), kegg)
    testing = process_interactions(
        interactions=testing, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    training = process_interactions(
        interactions=pd.concat([kegg, train_hprd], ignore_index=True),
        drop_nan=True, allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    ptm_labels = set(
        l for merged in list(training.label) + list(testing.label)
        for l in merged.split(',')
    )
    save_ptm_labels(ptm_labels)

    interactome = pd.concat(
        [bioplex, pina2, innate_i, innate_c], ignore_index=True
    )
    interactome = process_interactions(
        interactions=interactome, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=None, merge=True
    )
    save_network_to_path(interactome, interactome_network_path)
    save_network_to_path(training, training_network_path)
    save_network_to_path(testing, testing_network_path)
