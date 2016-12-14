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
from data import interactome_network_path

from data_mining.uniprot import UniProt
from data_mining.features import AnnotationExtractor
from data_mining.hprd import hprd_to_dataframe
from data_mining.kegg import download_pathway_ids, pathways_to_dataframe
from data_mining.generic import bioplex_func, mitab_func, pina_func
from data_mining.generic import generic_to_dataframe
from data_mining.tools import write_to_edgelist, map_network_accessions

if __name__ == '__main__':
    uniprot = UniProt(verbose=True)
    data_types = UniProt.data_types()
    selection = [data_types.GO, data_types.INTERPRO, data_types.PFAM]
    pathways = download_pathway_ids('hsa')

    # Construct all the networks
    hprd = hprd_to_dataframe(
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False,
        min_label_count=5,
        merge=True
    )
    kegg = pathways_to_dataframe(
        pathway_ids=pathways,
        drop_nan=True,
        allow_self_edges=True,
        allow_duplicates=False,
        min_label_count=5,
        merge=True,
        uniprot=True
    )
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
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]

    # Get a set of all the unique uniprot accessions
    sources = set(p for df in networks for p in df.source.values)
    targets = set(p for df in networks for p in df.source.targets)
    unique = sources | targets

    # Download the uniprot data for all these accessions
    accession_data = uniprot.features_to_dataframe(unique)
    new_accessions = {k: v[0] for k, v in zip(
        unique, accession_data.alt.values) if str(v[0]) != str(None)}

    # Map each network to the new uniprot accessions
    kegg = map_network_accessions(kegg, new_accessions)
    hprd = map_network_accessions(hprd, new_accessions)
    pina2 = map_network_accessions(pina2, new_accessions)
    bioplex = map_network_accessions(bioplex, new_accessions)
    innate_c = map_network_accessions(innate_c, new_accessions)
    innate_i = map_network_accessions(innate_i, new_accessions)
    networks = [kegg, hprd, bioplex, pina2, innate_i, innate_c]

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
        ppis = zip(df.source, df.target)
        ae.fit(ppis)
