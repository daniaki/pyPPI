#!/usr/bin/env python

"""
This module contains a class to parse a ppi predictions output file in tsv
format and perform network related operations on it.
"""

from collections import OrderedDict as Od

import numpy as np
import pandas as pd

from ..base import P1, P2, G1, G2, SOURCE, TARGET
from ..database import begin_transaction
from ..database.managers import InteractionManager, ProteinManager


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

    def __init__(self, interactions, sep=',', output_dir='./'):
        interactions = pd.read_csv(interactions, sep=sep)
        self.interactions_ = interactions
        self.columns_ = list(interactions.columns)
        self.gene_names_ = {}
        self.output_dir_ = output_dir
        self.sep = sep
        self.edges_ = list(zip(interactions[P1], interactions[G1]))
        p1_g1 = zip(interactions[P1], interactions[G1])
        p2_g2 = zip(interactions[P2], interactions[G2])
        for (p1, g1), (p2, g2) in zip(p1_g1, p2_g2):
            self.gene_names_[p1] = g1
            self.gene_names_[p2] = g2

        i_manager = InteractionManager(verbose=False, match_taxon_id=None)
        p_manager = ProteinManager(verbose=False, match_taxon_id=None)
        self.training_edges = set()
        self.training_nodes = set()
        with begin_transaction() as session:
            self.labels = i_manager.training_labels(
                session, include_holdout=True
            )
            training_edges = i_manager.training_interactions(
                session, filter_out_holdout=False
            )
            for interaction in training_edges:
                a = p_manager.get_by_id(session, interaction.source).uniprot_id
                b = p_manager.get_by_id(session, interaction.target).uniprot_id
                a, b = sorted((a, b))
                self.training_edges.add((a, b))

        for (a, b) in self.training_edges:
            self.training_nodes.add(a)
            self.training_nodes.add(b)

    def node_in_training_set(self, node):
        return node in self.training_nodes

    def edge_in_training_set(self, edge):
        p1, p2 = sorted(edge)
        return (p1, p2) in self.training_edges

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
            raise ValueError(
                "{} is not a valid label. Please choose from {}.".format(
                    label, ', '.join(self.labels))
            )
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

    def induce_subnetwork_from_pathway(self, accession_list, threshold,
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
        accesion_list = set(accession_list)

        a, b = (P1, P2)
        if genes:
            a, b = (G1, G2)
        edges = [sorted([p1, p2]) for (p1, p2) in zip(df[a], df[b])]
        edge_idx = np.asarray(
            [i for i, (p1, p2) in enumerate(edges)
             if p1 in accesion_list or p2 in accesion_list]  # or?
        )
        if len(edge_idx) == 0:
            ValueError("No subnetwork could be induced with the given"
                       "pathway list.")

        # Filter for the interactions with a max probability greater
        # than `threshold`.
        df = df.loc[edge_idx, ]
        sel = (df.loc[:, self.labels].max(axis=1) >= threshold).values
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
        return self

    def _write_cytoscape_files(self, noa_path, eda_path,
                               pp_path, idx_selection):
        """
        Compute some node and edge attributes and write these to
        files that can be loaded in cytoscape.
        """
        df = self.interactions_.loc[idx_selection, ]
        edges = [sorted([p1, p2]) for (p1, p2)
                 in zip(df[P1].values, df[P2].values)]

        # Compute some selected node-attributes,
        # Write the noa (noda-attribute) file.
        accessions = sorted(set([p for tup in edges for p in tup]))
        gene_names = [self.gene_names_[a] for a in accessions]
        node_in_training = [self.node_in_training_set(a) for a in accessions]
        cyto_n_attrs = pd.DataFrame(Od([
            ('Name', accessions),
            ('Node in training', node_in_training),
            ('Gene Name', gene_names)
        ]))
        cyto_n_attrs.to_csv(noa_path, sep=self.sep, index=False)

        # Compute some selected edge-attributes a,
        # Write the eda (edge-attribute) file.
        edge_in_training = [self.edge_in_training_set(e) for e in edges]
        cyto_e_attrs = [('%s-pr' % l, df[l].values) for l in self.labels]
        cyto_e_attrs += [
            ('Name', ['{} pp {}'.format(p1, p2) for p1, p2 in edges])
        ]
        cyto_e_attrs += [
            ('Edge In Training', edge_in_training)
        ]
        cyto_e_attrs = pd.DataFrame(data=Od(cyto_e_attrs))
        cyto_e_attrs.to_csv(eda_path, sep=self.sep, index=False)

        cyto_interactions = pd.DataFrame(Od([
            ('source', [p1 for p1, _ in edges]),
            ('target', [p2 for _, p2 in edges]),
            ('interaction', ['pp' for _ in edges])
        ]))
        cyto_interactions.to_csv(pp_path, sep=self.sep, index=False)
        return self
