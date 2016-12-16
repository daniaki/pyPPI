#!/usr/bin/env python

"""
This script runs the bootstrap kfold validation experiments as used in
the publication.
"""

import argparse
import numpy as np
import pandas as pd

from data import load_go_dag
from data import bioplex_v4, pina2, innate_curated, innate_imported
from data import testing_network_path, training_network_path
from data import bioplex_network_path, pina2_network_path
from data import innate_i_network_path, innate_c_network_path
from data import kegg_network_path, hprd_network_path
from data import interactome_network_path

from data_mining.uniprot import UniProt
from data_mining.features import AnnotationExtractor
from data_mining.hprd import hprd_to_dataframe
from data_mining.kegg import download_pathway_ids, pathways_to_dataframe
from data_mining.generic import bioplex_func, mitab_func, pina_func
from data_mining.generic import generic_to_dataframe
from data_mining.tools import write_to_edgelist, map_network_accessions
from data_mining.tools import process_interactions
from data_mining.tools import remove_intersection, remove_labels

if __name__ == '__main__':
    uniprot = UniProt(verbose=True)
    data_types = UniProt.data_types()
    selection = [data_types.GO, data_types.INTERPRO, data_types.PFAM]
    pathways = download_pathway_ids('hsa')

    # Construct all the networks
    print("Building HPRD interactions...")
    hprd = hprd_to_dataframe(
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False,
        min_label_count=5,
        merge=True
    )

    print("Building KEGG interactions...")
    kegg = pathways_to_dataframe(
        pathway_ids=pathways,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False,
        min_label_count=5,
        merge=True,
        uniprot=True
    )

    print("Building Interactome interactions...")
    bioplex = generic_to_dataframe(
        f_input=bioplex_v4(),
        parsing_func=bioplex_func,
        allow_self_edges=True,
        allow_duplicates=False
    )
    pina2 = generic_to_dataframe(
        f_input=pina2(),
        parsing_func=pina_func,
        allow_self_edges=True,
        allow_duplicates=False
    )
    innate_c = generic_to_dataframe(
        f_input=innate_curated(),
        parsing_func=mitab_func,
        allow_self_edges=True,
        allow_duplicates=False
    )
    innate_i = generic_to_dataframe(
        f_input=innate_imported(),
        parsing_func=mitab_func,
        allow_self_edges=True,
        allow_duplicates=False
    )

    print("Mapping to most recent uniprot accessions...")
    # Get a set of all the unique uniprot accessions
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]
    sources = set(p for df in networks for p in df.source.values)
    targets = set(p for df in networks for p in df.source.targets)

    # Download the uniprot data for all these accessions
    unique = sources | targets
    accession_data = uniprot.features_to_dataframe(sources | targets)
    new_accessions = {k: v[0] for k, v in zip(
        unique, accession_data.alt.values) if str(v[0]) != str(None)}

    print("Mapping each network to the most recent uniprot accessions...")
    kegg = map_network_accessions(
        interactions=kegg, accession_map=new_accessions,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=5, merge=True
    )
    hprd = map_network_accessions(
        interactions=hprd, accession_map=new_accessions,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=5, merge=True
    )

    pina2 = map_network_accessions(
        interactions=pina2, accession_map=new_accessions,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=True
    )
    bioplex = map_network_accessions(
        interactions=bioplex, accession_map=new_accessions,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=True
    )
    innate_c = map_network_accessions(
        interactions=innate_c, accession_map=new_accessions,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=True
    )
    innate_i = map_network_accessions(
        interactions=innate_i, accession_map=new_accessions,
        drop_nan=True, allow_self_edges=True,
        allow_duplicates=False, min_counts=None, merge=True
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

    # for df in networks:
    #     ppis = list(zip(df.source, df.target))
    #     ae.fit(ppis)

    print("Saving networks and feature files...")
    write_to_edgelist(kegg, kegg_network_path())
    write_to_edgelist(hprd, hprd_network_path())
    write_to_edgelist(pina2, pina2_network_path())
    write_to_edgelist(bioplex, bioplex_network_path())
    write_to_edgelist(innate_i, innate_i_network_path())
    write_to_edgelist(innate_c, innate_c_network_path())

    test_labels = ['dephosphorylation', 'phosphorylation']
    train_labels = [l for l in hprd.label if l not in test_labels]
    train_hprd = remove_labels(hprd, test_labels)

    testing = remove_intersection(remove_labels(hprd, train_labels), kegg)
    training = process_interactions(
        interactions=pd.concat([kegg, train_hprd], ignore_index=True),
        drop_nan=True, allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    interactome = pd.concat(
        [bioplex, pina2, innate_i, innate_c], ignore_index=True
    )
    interactome = process_interactions(
        interactions=interactome, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=None, merge=True
    )
    write_to_edgelist(interactome, interactome_network_path())
    write_to_edgelist(training, training_network_path())
    write_to_edgelist(testing, testing_network_path())
