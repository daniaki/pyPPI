#!/usr/bin/env python

"""
This script runs classifier training over the entire training data and then
output predictions over the interactome.

Usage:
  build_data.py [--interpro] [--pfam] [--mf] [--cc] [--bp]
                [--update_features] [--update_mapping]
                [--induce] [--verbose] [--n_jobs=J] [--backend=B]
  build_data.py -h | --help

Options:
  -h --help     Show this screen.
  --interpro    Use interpro domains in features.
  --pfam        Use Pfam domains in features.
  --mf          Use Molecular Function Gene Ontology in features.
  --cc          Use Cellular Compartment Gene Ontology in features.
  --bp          Use Biological Process Gene Ontology in features.
  --induce      Use ULCA inducer over Gene Ontology.
  --verbose     Print intermediate output for debugging.
  --n_jobs=J            Number of processes to run in parallel [default: 1]
  --backend=B           Python concurrent backend [default: multiprocessing]
  --update_features     Delete old feature cache and create a new one.
  --update_mapping      Delete old accession mapping and create a new one.
"""

import os
import pandas as pd
from docopt import docopt
args = docopt(__doc__)

from pyPPI.base import PPI, parse_args
from pyPPI.data import bioplex_network_path, pina2_network_path
from pyPPI.data import bioplex_v4, pina2, innate_curated, innate_imported
from pyPPI.data import innate_i_network_path, innate_c_network_path
from pyPPI.data import interactome_network_path, full_training_network_path
from pyPPI.data import kegg_network_path, hprd_network_path
from pyPPI.data import load_uniprot_accession_map, save_uniprot_accession_map
from pyPPI.data import testing_network_path, training_network_path
from pyPPI.data import save_network_to_path
from pyPPI.data import save_ptm_labels
from pyPPI.data import ppi_features_path, accession_features_path
from pyPPI.data import annotation_extractor_path
from pyPPI.data import pickle_pd_object

from pyPPI.data_mining.features import AnnotationExtractor
from pyPPI.data_mining.generic import bioplex_func, mitab_func, pina_func
from pyPPI.data_mining.generic import generic_to_dataframe
from pyPPI.data_mining.hprd import hprd_to_dataframe
from pyPPI.data_mining.tools import process_interactions, LABEL
from pyPPI.data_mining.tools import remove_intersection, remove_labels
from pyPPI.data_mining.tools import map_network_accessions
from pyPPI.data_mining.uniprot import UniProt, get_active_instance
from pyPPI.data_mining.kegg import download_pathway_ids, pathways_to_dataframe

if __name__ == '__main__':
    args = parse_args(args)
    uniprot = get_active_instance(verbose=True)
    data_types = UniProt.data_types()
    pathways = download_pathway_ids('hsa')
    n_jobs = args['n_jobs']
    induce = args['induce']
    verbose = args['verbose']
    selection = args['selection']
    update_features = args['update_features']
    update_mapping = args['update_mapping']
    backend = args['backend']

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

    if update_mapping:
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
    networks = [hprd, kegg, bioplex, pina2, innate_i, innate_c]

    print("Building features for each protein and PPI...")
    ae = AnnotationExtractor(
        induce=True,
        selection=selection,
        n_jobs=n_jobs,
        verbose=True,
        cache=not update_features,
        backend=backend,
    )

    cond_1 = os.path.isfile(ppi_features_path)
    cond_2 = os.path.isfile(accession_features_path)
    if update_features or (not cond_1) or (not cond_2):
        tuple_gens = [zip(n.source, n.target) for n in networks]
        ppis = [(a, b) for tuples in tuple_gens for (a, b) in tuples]
        ae.fit(ppis)

        # Sanity check
        unique_ppis = set()
        unique_acc = set()
        for df in networks:
            ppis = list(zip(df.source, df.target))
            unique_ppis |= set(PPI(a, b) for (a, b) in ppis)
            unique_acc |= set(a for a, _ in ppis) | set(b for _, b in ppis)
        assert ae.accession_vocabulary.shape[0] == len(unique_acc)
        assert ae.ppi_vocabulary.shape[0] == len(unique_ppis)
        ae.cache()

    print("Saving networks and feature files...")
    save_network_to_path(kegg, kegg_network_path)
    save_network_to_path(hprd, hprd_network_path)
    save_network_to_path(pina2, pina2_network_path)
    save_network_to_path(bioplex, bioplex_network_path)
    save_network_to_path(innate_i, innate_i_network_path)
    save_network_to_path(innate_c, innate_c_network_path)

    hprd_test_labels = ['dephosphorylation', 'phosphorylation']
    hprd_train_labels = set([l for l in hprd[LABEL]
                             if l not in hprd_test_labels])
    train_hprd = remove_labels(hprd, hprd_test_labels)
    training = pd.concat([kegg, train_hprd], ignore_index=True)
    testing = remove_intersection(remove_labels(hprd, hprd_train_labels), kegg)
    full_training = pd.concat([training, testing], ignore_index=True)

    print("Building and saving processed networks...")
    testing = process_interactions(
        interactions=testing, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    training = process_interactions(
        interactions=training,
        drop_nan=True, allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    full_training = process_interactions(
        interactions=full_training,
        drop_nan=True, allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=5, merge=True
    )
    labels = list(training[LABEL]) + list(testing[LABEL])
    ptm_labels = set(l for merged in labels for l in merged.split(','))
    save_ptm_labels(ptm_labels)
    interactome_networks = [bioplex, pina2, innate_i, innate_c]
    interactome = pd.concat(interactome_networks, ignore_index=True)
    interactome = process_interactions(
        interactions=interactome, drop_nan=True,
        allow_duplicates=False, allow_self_edges=True,
        exclude_labels=None, min_counts=None, merge=True
    )
    save_network_to_path(interactome, interactome_network_path)
    save_network_to_path(training, training_network_path)
    save_network_to_path(testing, testing_network_path)
    save_network_to_path(full_training, full_training_network_path)
