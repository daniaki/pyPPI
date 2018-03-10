#!/usr/bin/env python

"""
This module contains a class to parse a ppi predictions output file in tsv
format and perform network related operations on it.
"""

from collections import OrderedDict as Od

import numpy as np
import pandas as pd

from ..base.constants import P1, P2, G1, G2, SOURCE, TARGET
from ..database.models import Protein
from ..database.utilities import full_training_network


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
    interactions_: :class:`pd.DataFrame`
        The dataframe from the input parameter or loaded from the input
        parameter. Contains columns p1, p2, g1, g2 for the accession, and
        a series of ptm labels for each predicted label.

    columns_: list[str]
        List of columns in `interactions_`

    gene_names_: dict[str, str]
        A mapping of protein accession to gene accession
        found in `interactions_`

    edges_ : list
        List of tuples of `UniProt` accession parsed from the `source`
        and `target` columns

    output_dir_ : str
        Output directory.

    training_edges : set
        Edges that are part of the training network according the 
        database. These are the :class:`Interaction`s in the parsed file that
        have instances with `is_training` set to True.

    training_nodes : set
        All :class:`Protein`s in the supplied interactions which appear in 
        :class:`Interaction`s with `is_training` set to True.
    """

    def __init__(self, interactions, sep='\t', output_dir='./'):
        self.interactions_ = interactions
        self.columns_ = list(interactions.columns)
        self.gene_names_ = {}
        self.output_dir_ = output_dir
        self.sep = sep
        self.edges_ = list(zip(interactions[P1], interactions[G1]))
        self.training_edges = set()
        self.training_nodes = set()

        p1_g1 = zip(interactions[P1], interactions[G1])
        p2_g2 = zip(interactions[P2], interactions[G2])
        for (p1, g1), (p2, g2) in zip(p1_g1, p2_g2):
            self.gene_names_[p1] = g1
            self.gene_names_[p2] = g2

        self.labels = [
            c[:-3] for c in interactions.columns
            if ('-pr' in c) and ('sum' not in c) and ('max' not in c)
        ]

        training_edges = full_training_network(taxon_id=None)
        for interaction in training_edges:
            a = Protein.query.get(interaction.source).uniprot_id
            b = Protein.query.get(interaction.target).uniprot_id
            a, b = sorted((a, b))
            self.training_edges.add((a, b))

        for (a, b) in self.training_edges:
            self.training_nodes.add(a)
            self.training_nodes.add(b)

    def _validate_threshold(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(
                "Threshold must be a float. Found {}.".format(type(value))
            )

    def _label_to_column(self, label):
        if not label.endswith('-pr'):
            return label + '-pr'
        else:
            return label

    def node_in_training_set(self, node):
        return node in self.training_nodes

    def edge_in_training_set(self, edge):
        p1, p2 = sorted(edge)
        return (p1, p2) in self.training_edges

    def induce_subnetwork_from_label(self, label, threshold=0.5):
        """
        Filter the interactions to contain those with predicted `label` at or
        over the `threshold`.

        Parameters
        ----------
        label: string
            A PTM label seen in `interactions_`

        threshold: float, optional, default: 0.5
            Minimum prediction probability.

        Returns
        -------
        :class:`InteractionNetwork`
            Returns self
        """
        if label not in self.labels:
            raise ValueError(
                "{} is not a valid label. Please choose from {}.".format(
                    label, ', '.join(self.labels))
            )
        column = self._label_to_column(label)

        threshold = self._validate_threshold(threshold)
        label_idx = self.interactions_[
            self.interactions_[column] >= threshold
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

        Parameters
        ----------
        accesion_list: string
            A list of uniprot/gene accessions to induce a network from.
            Network will induce all edges incident upon these accessions.

        threshold: float
            Minimum prediction probability.

        genes: boolean, optional, default: False
            Use gene identifier accessions in `interactions_` instead. Set this
            to True if your `accession_list` is a list of gene names.

        Returns
        -------
        :class:`InteractionNetwork`
            Returns self
        """
        df = self.interactions_
        accesion_list = set(accession_list)
        threshold = self._validate_threshold(threshold)

        a, b = (P1, P2)
        if genes:
            a, b = (G1, G2)
        edges = [tuple(sorted([p1, p2])) for (p1, p2) in zip(df[a], df[b])]
        edge_idx = np.asarray(
            [
                i for i, (p1, p2) in enumerate(edges)
                if (p1 in accesion_list) or (p2 in accesion_list)
            ]
        )
        if len(edge_idx) == 0:
            raise ValueError(
                "No subnetwork could be induced with the given"
                "pathway list."
            )

        # Filter for the interactions with a max probability greater
        # than `threshold`.
        df = df.loc[edge_idx, ]
        sel = (df['max-pr'] >= threshold).values
        edge_idx = df[sel].index.values

        if len(edge_idx) == 0:
            raise ValueError(
                "Threshold set too high and no subnetwork could be "
                "induced with the given pathway list."
            )

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
            ('name', accessions),
            ('node in training', node_in_training),
            ('gene name', gene_names)
        ]))
        cyto_n_attrs.to_csv(noa_path, sep=self.sep, index=False)

        # Compute some selected edge-attributes a,
        # Write the eda (edge-attribute) file.
        columns = ['source', 'target', 'name', 'edge in training', 'max-pr']
        cyto_e_attrs = {}
        cyto_e_attrs['source'] = [p1 for p1, _ in edges]
        cyto_e_attrs['target'] = [p2 for _, p2 in edges]
        cyto_e_attrs['name'] = ['{} pp {}'.format(p1, p2) for p1, p2 in edges]
        cyto_e_attrs['edge in training'] = [
            self.edge_in_training_set(e) for e in edges
        ]
        cyto_e_attrs['max-pr'] = list(df['max-pr'].values)
        for label in self.labels:
            column = self._label_to_column(label)
            cyto_e_attrs[column] = df[column].values
            columns.append(column)

        cyto_interactions = pd.DataFrame(cyto_e_attrs, columns=columns)
        cyto_interactions.to_csv(pp_path, sep=self.sep, index=False)
        return self
