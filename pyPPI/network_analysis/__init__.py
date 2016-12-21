#!/usr/bin/env python

"""
This module contains a class to parse a ppi predictions output file in tsv
format and perform network related operations on it.
"""

import os
import igraph
import tempfile
import pandas as pd
import numpy as np

from ..data import load_training_network
from ..data import ptm_labels


def _igraph_from_tuples(v_names):
    # igraph will store the accessions under the 'name' attribute and
    # use its own integer number system to form vertex ids
    tmp = tempfile.mktemp(suffix='.ncol', prefix='graph_', dir='tmp/')
    fp = open(tmp, "w")
    for (name_a, name_b) in v_names:
        fp.write("{} {}\n".format(name_a, name_b))
    fp.close()
    # will store node names under the 'name' vertex attribute.
    g = igraph.read(tmp, format='ncol', directed=False, names=True)
    os.remove(tmp)
    return g


def _igraph_vid_attr_table(igraph_g, attr):
    vid_prop_lookup = {}
    for v in igraph_g.vs:
        data = v[attr]
        vid_prop_lookup[v.index] = data
    return vid_prop_lookup


def _node_in_training_set(node):
    train_df = load_training_network()
    train_nodes = set(train_df['p1'].values) | set(train_df['p2'])
    return node in train_nodes


def _edge_in_training_set(edge):
    train_df = load_training_network()
    p1, p2 = sorted(edge)
    train_edges = set([sorted([p1, p2]) for p1, p2 in
                       set(train_df['uniprot'].values)])
    return (p1, p2) in train_edges


class InteractionNetwork(object):
    """
    Utility class to ecapsulate all network related operations for predictions
    over some interaction network.

    Parameters
    -----------
    interactions, pd.DataFrame or string.
        The dataframe containing prediction output in the PTSV format or
        a string directing the path containing the PTSV file.

    Attributes
    -----------
    interactions_: pd.DataFrame
        The dataframe from the input parameter or loaded from the input
        parameter. Contains columns p1, p2, g1, g2 for the accession, and
        a series of ptm labels for each predicted label.

    columns_: List[str]
        List of columns in `interactions_`

    gene_names_: Dictionary[str, str]
        A mapping of protein accession to gene accession
        found in `interactions_`
    """

    def __init__(self, interactions):
        if not isinstance(interactions, pd.DataFrame):
            interactions = pd.read_csv(interactions, sep='\t')
        self.interactions_ = interactions
        self.columns_ = list(interactions.columns)
        self.gene_names_ = {}
        p1_g1 = zip((interactions['p1'].values, interactions['g1'].values))
        p2_g2 = zip((interactions['p2'].values, interactions['g2'].values))
        for (p1, g1), (p2, g2) in zip(p1_g1, p2_g2):
            self.gene_names_[p1] = g1
            self.gene_names_[p2] = g2

    def induce_subnetwork_from_label(self, label, threshold=0.5, output=False):
        """
        Filter the interactions to contain those with predicted `label` at or
        over the `threshold`.

        :param label: sting
            A ptm label.
        :param threshold: float, optional
            Minimum prediction probability.
        :param output: boolean, optional
            Write interaction, node and edge attribute files to disk.

        :return: Self
        """
        if label not in self.columns_:
            raise ValueError("{} was not found in the possible "
                             "labels.".format(label))
        if not isinstance(threshold, float):
            raise ValueError("Threshold must be a float.")

        label_idx = self.interactions_[
            self.interactions_[label] >= threshold
        ].index.values
        if output:
            self._write_cytoscape_files(label_idx, label)
        return self

    def induce_subnetwork_from_pathway(self, accesion_list, threshold,
                                       genes=False, output=False):
        """
        Filter the interactions to contain any interaction with both
        accessions in `accesion_list` and with predictions at or
        over the `threshold`.

        :param accesion_list: sting
            A ptm label.
        :param threshold: float, optional
            Minimum prediction probability.
        :param genes: boolean, optional
            Use gene identifier accessions.
        :param output: boolean, optional
            Write interaction, node and edge attribute files to disk.

        :return: Self
        """
        if not isinstance(threshold, float):
            raise ValueError("Threshold must be a float.")
        df = self.interactions_
        accesion_list = set(accesion_list)

        a, b = ('p1', 'p2')
        if genes:
            a, b = ('g1', 'g2')
        edges = [sorted([p1, p2]) for (p1, p2)
                 in zip(df[a].values, df[b].values)]
        edge_idx = np.asarray(
            [i for i, (p1, p2) in enumerate(edges)
             if p1 in accesion_list and p2 in accesion_list]
        )
        if len(edge_idx) == 0:
            ValueError("No subnetwork could be induced with the given"
                       "pathway list.")

        # Filter for the interactions with a max probability greater
        # than `threshold`.
        df = df.loc[edge_idx, ]
        sel = (df.loc[:, ptm_labels()].max(axis=1) >= threshold).values
        edge_idx = df[sel].index.values

        if len(edge_idx) == 0:
            ValueError("Threshold set too high and no subnetwork could be "
                       "induced with the given pathway list.")
        if output:
            self._write_cytoscape_files(edge_idx, label='pathway')
        return edge_idx

    def _write_cytoscape_files(self, noa_path, eda_path,
                               pp_path, idx_selection):
        """
        Compute some node and edge attributes and write these to
        files that can be loaded in cytoscape.
        """
        df = self.interactions_.loc[idx_selection, ]
        edges = [sorted([p1, p2]) for (p1, p2)
                 in zip(df['p1'].values, df['p2'].values)]

        # vid is the vertex id from igraph
        graph = _igraph_from_tuples(edges)
        vid_name_loopup = _igraph_vid_attr_table(graph, attr='name')
        vids = [v.index for v in graph.vs]

        # Compute some selected node-attributes,
        # Write the noa (noda-attribute) file.
        degree = graph.degree()
        nodes = set([p for tup in edges for p in tup])
        node_in_training = [_node_in_training_set(n) for n in nodes]
        accessions = [vid_name_loopup[vid] for vid in vids]
        gene_names = [self.gene_names_[a] for a in accessions]
        n_nodes = len(nodes)
        betweenness = [(2 * betwn)/(n_nodes*n_nodes - 3*n_nodes + 2)
                       for betwn in graph.betweenness()]
        cyto_n_attrs = pd.DataFrame({
            'Name': accessions,
            'Node in training': node_in_training,
            'Node betweeness': betweenness,
            'Node degree': degree,
            'Gene Name': gene_names
        })
        cyto_n_attrs.to_csv(noa_path, sep='\t')

        # Compute some selected edge-attributes a,
        # Write the eda (edge-attribute) file.
        edge_in_training = [_edge_in_training_set(e) for e in edges]
        cyto_e_attrs = {'{}-pr'.format(l): df[l].values for l in ptm_labels()}
        cyto_e_attrs['Name'] = ['{} pp {}'.format(p1, p2) for p1, p2 in edges]
        cyto_e_attrs['Edge In Training'] = edge_in_training
        cyto_e_attrs = pd.DataFrame(data=cyto_e_attrs)
        cyto_e_attrs.to_csv(eda_path, sep='\t')

        cyto_interactions = pd.DataFrame({
            'source': [p1 for p1, _ in edges],
            'target': [p2 for _, p2 in edges],
            'interaction': ['pp' for _ in edges]
        })
        cyto_interactions.to_csv(pp_path, sep='\t')
        return self
