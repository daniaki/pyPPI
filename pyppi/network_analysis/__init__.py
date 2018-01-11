#!/usr/bin/env python

"""
This module contains a class to parse a ppi predictions output file in tsv
format and perform network related operations on it.
"""

import os
import tempfile
from collections import OrderedDict as Od

import igraph
import numpy as np
import pandas as pd

from ..base import P1, P2, G1, G2, SOURCE, TARGET
from ..base import PPI
from ..data import load_network_from_path, full_training_network_path
from ..data import load_ptm_labels


def _igraph_from_tuples(v_names):
    # igraph will store the accessions under the 'name' attribute and
    # use its own integer number system to form vertex ids
    tmp = tempfile.mktemp(suffix='.ncol', prefix='graph_')
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
    train_df = load_network_from_path(full_training_network_path)
    train_nodes = set(train_df[SOURCE]) | set(train_df[TARGET])
    return node in train_nodes


def _edge_in_training_set(edge):
    train_df = load_network_from_path(full_training_network_path)
    p1, p2 = sorted(edge)
    train_edges = zip(train_df[SOURCE], train_df[TARGET])
    train_edges = set([tuple(PPI(s, t)) for (s, t) in train_edges])
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

    def __init__(self, interactions, output_dir='./'):
        interactions = pd.read_csv(interactions, sep='\t')
        self.interactions_ = interactions
        self.columns_ = list(interactions.columns)
        self.gene_names_ = {}
        self.output_dir_ = output_dir
        self.edges_ = list(zip(interactions[P1], interactions[G1]))
        p1_g1 = zip(interactions[P1], interactions[G1])
        p2_g2 = zip(interactions[P2], interactions[G2])
        for (p1, g1), (p2, g2) in zip(p1_g1, p2_g2):
            self.gene_names_[p1] = g1
            self.gene_names_[p2] = g2

    def induce_subnetwork_from_label(self, label, threshold=0.5):
        """
        Filter the interactions to contain those with predicted `label` at or
        over the `threshold`.

        :param label: string
            A ptm label.
        :param threshold: float, optional
            Minimum prediction probability.

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
        pp_path = '{}/{}_pp.tsv'.format(self.output_dir_, label)
        noa_path = '{}/{}_node_attrs.noa'.format(self.output_dir_, label)
        eda_path = '{}/{}_edge_attrs.eda'.format(self.output_dir_, label)
        self._write_cytoscape_files(
            pp_path=pp_path,
            noa_path=noa_path,
            eda_path=eda_path,
            idx_selection=label_idx
        )
        return self

    def induce_subnetwork_from_pathway(self, accesion_list, threshold,
                                       genes=False):
        """
        Filter the interactions to contain any interaction with both
        accessions in `accesion_list` and with predictions at or
        over the `threshold`.

        :param accesion_list: string
            A list of uniprot/gene accessions to induce a network from.
            Network will induce all edges incident upon these accessions.
        :param threshold: float, optional
            Minimum prediction probability.
        :param genes: boolean, optional
            Use gene identifier accessions.

        :return: Self
        """
        if not isinstance(threshold, float):
            raise ValueError("Threshold must be a float.")
        df = self.interactions_
        accesion_list = set(accesion_list)

        a, b = (P1, P2)
        if genes:
            a, b = (G1, G2)
        edges = [sorted([p1, p2]) for (p1, p2) in zip(df[a], df[b])]
        edge_idx = np.asarray(
            [i for i, (p1, p2) in enumerate(edges)
             if p1 in accesion_list and p2 in accesion_list]  # or?
        )
        if len(edge_idx) == 0:
            ValueError("No subnetwork could be induced with the given"
                       "pathway list.")

        # Filter for the interactions with a max probability greater
        # than `threshold`.
        df = df.loc[edge_idx, ]
        sel = (df.loc[:, load_ptm_labels()].max(axis=1) >= threshold).values
        edge_idx = df[sel].index.values

        if len(edge_idx) == 0:
            ValueError("Threshold set too high and no subnetwork could be "
                       "induced with the given pathway list.")
        label = 'pathway'
        pp_path = '{}/{}_pp.tsv'.format(self.output_dir_, label)
        noa_path = '{}/{}_node_attrs.noa'.format(self.output_dir_, label)
        eda_path = '{}/{}_edge_attrs.eda'.format(self.output_dir_, label)
        self._write_cytoscape_files(
            pp_path=pp_path,
            noa_path=noa_path,
            eda_path=eda_path,
            idx_selection=edge_idx
        )
        return edge_idx

    def _write_cytoscape_files(self, noa_path, eda_path,
                               pp_path, idx_selection):
        """
        Compute some node and edge attributes and write these to
        files that can be loaded in cytoscape.
        """
        df = self.interactions_.loc[idx_selection, ]
        edges = [sorted([p1, p2]) for (p1, p2)
                 in zip(df[P1].values, df[P2].values)]

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
        betweenness = [(2 * betwn) / (n_nodes * n_nodes - 3 * n_nodes + 2)
                       for betwn in graph.betweenness()]
        cyto_n_attrs = pd.DataFrame(Od([
            ('Name', accessions),
            ('Node in training', node_in_training),
            ('Node betweeness', betweenness),
            ('Node degree', degree),
            ('Gene Name', gene_names)
        ]))
        cyto_n_attrs.to_csv(noa_path, sep='\t', index=False)

        # Compute some selected edge-attributes a,
        # Write the eda (edge-attribute) file.

        edge_in_training = [_edge_in_training_set(e) for e in edges]
        cyto_e_attrs = [('%s-pr' % l, df[l].values) for l in load_ptm_labels()]
        cyto_e_attrs += [
            ('Name', ['{} pp {}'.format(p1, p2) for p1, p2 in edges])
        ]
        cyto_e_attrs += [
            ('Edge In Training', edge_in_training)
        ]
        cyto_e_attrs = pd.DataFrame(data=Od(cyto_e_attrs))
        cyto_e_attrs.to_csv(eda_path, sep='\t', index=False)

        cyto_interactions = pd.DataFrame(Od([
            ('source', [p1 for p1, _ in edges]),
            ('target', [p2 for _, p2 in edges]),
            ('interaction', ['pp' for _ in edges])
        ]))
        cyto_interactions.to_csv(pp_path, sep='\t', index=False)
        return self
