#!/usr/bin/python

"""
Created on 23-02-2016
@author: Daniel Esposito
@contact: desposito@student.unimelb.edu.au
"""

import numpy as np
from functools import reduce


# ------------------------------------------------------ #
#
#                  UTILITY OPERATIONS
#
# ------------------------------------------------------ #
def get_relationship_terms(dag, term, rs_type='part_of'):
    part_of_terms = set()
    try:
        relationships = dag[term].relationship
    except AttributeError:
        relationships = []

    for rs_item in relationships:
        xs = rs_item.split(' ')
        relationship = xs[0]
        terms = set([x for x in xs if 'GO:' in x.upper()])
        if relationship.lower() == rs_type:
            part_of_terms |= terms
    return part_of_terms


def id_to_node(term, dag):
    return dag[term]


def frequency_distribution(corpus):
    dist = {k: 0 for k in set(corpus)}
    n = len(corpus)
    for w in corpus:
        dist[w] += 1.0
    dist = {k: (v/n) for k, v in dist.items()}
    return dist


def entropy(word, distribution):
    if word not in distribution.keys():
        return 0.0
    p = distribution[word]
    return -p * np.log2(p)


def sentence_entropy(terms, distribution):
    return np.sum([entropy(t, distribution) for t in terms])


def validate_term_sets(term_sets):
    for ts in term_sets:
        if isinstance(ts, set):
            raise ValueError("term set is a set not a list.")
    term_sets = [ts for ts in term_sets if len(ts) > 0]
    return term_sets


def term_count(t, ts):
    return len([x for x in ts if x == t])


def fill_out(terms, go_dag, corpus, verbose=0):
    """For terms not seen in corpus but are in the terms list,
    append the lowest parent that is in the corpus

    :param terms: Term list to fillout
    :param go_dag: go dag loaded from obo file.
    :param corpus: corpus of go annotations appearing in the training set.
    :param verbose: print intermediate output.

    :return: List of original plus added terms.
    """
    terms = [x for x in terms if 'GO' in x.upper()]
    present_terms = [x for x in terms if x in corpus]
    absent_terms = [x for x in terms if x not in corpus]
    extra_induced_terms = []
    if len(terms) == 0:
        return terms

    for term in absent_terms:
        ancestors = get_all_parents(go_dag, term)
        ancestors = sorted(ancestors, key=lambda x: go_dag[x].depth,
                           reverse=True)
        leafiest_depth = -1  # value for when nothing is found
        for a in ancestors:
            if (a in corpus) and (a not in present_terms) and \
                    (a not in extra_induced_terms):
                leafiest_depth = go_dag[a].depth
                break

        # Add equally depthed ancestors if possible
        ancestors = [a for a in ancestors if go_dag[a] == leafiest_depth]
        for a in ancestors:
            if (a in corpus) and (a not in present_terms) \
                    and (a not in extra_induced_terms):
                # multiply by how many times it would have been
                # induced if it were originally preset.
                extra_induced_terms += [a] * term_count(term, absent_terms)

    new_terms = present_terms + absent_terms + extra_induced_terms
    if verbose:
        print("Terms already in corpus: {}/{}".format(
            len(present_terms), len(new_terms)))
        print("Additional terms: {}/{}".format(
            len(extra_induced_terms), len(new_terms)))
    return new_terms


def get_all_parents(dag, term):
    parents = set(dag[term].get_all_parents())

    # Get all the 'part_of' terms
    part_of_terms = get_relationship_terms(dag, term)
    for p in parents:
        part_ofs = get_relationship_terms(dag, p)
        part_of_terms |= part_ofs

    # Get all the parents of the 'part_of' terms.
    part_of_terms_parents = set()
    for pot in part_of_terms:
        part_of_terms_parents |= get_all_parents(dag, pot)
    part_of_terms |= part_of_terms_parents

    parents |= part_of_terms
    # print "All parents:"
    # for t in sorted(parents, key=lambda t: dag[t].name):
    #     print '\t {} --> {}'.format(t, dag[t].name)

    return parents


# ------------------------------------------------------ #
#
#                  GODAG METHODS
#
# ------------------------------------------------------ #
def get_deepest_term(term_set, go_dag):
    depths = [go_dag[t].depth for t in term_set]
    depths = {t: d for (t, d) in zip(term_set, depths)}
    try:
        max_depth = max(depths.values())
        deepest_terms = [k for (k, v) in depths.items() if v == max_depth]
    except ValueError:
        deepest_terms = []
    return deepest_terms


def get_all_ancestors_for_set(term_set, go_dag):
    parents = set()  # add term_set?
    for term in term_set:
        parents |= get_all_parents(go_dag, term)
    return parents


def get_all_common_ancestors_for_set(term_set, go_dag):
    parents = set()  # add term_set?
    for term in term_set:
        if len(parents) < 1:
            parents |= get_all_parents(go_dag, term)
        else:
            parents &= get_all_parents(go_dag, term)
    return parents


def get_all_ancestors_for_sets(term_sets, go_dag):
    a_ancestors = list(get_all_ancestors_for_set(term_sets[0], go_dag))
    for ts in term_sets[1:]:
        a_ancestors += list(get_all_ancestors_for_set(ts, go_dag))
    return a_ancestors


def get_all_commmon_ancestors_for_sets(term_sets, go_dag):
    c_ancestors = get_all_ancestors_for_set(term_sets[0], go_dag)
    for ts in term_sets[1:]:
        c_ancestors &= get_all_ancestors_for_set(ts, go_dag)

    # print "Common:"
    # for t in set(c_ancestors):
    #     print '\t {} --> {}'.format(t, go_dag[t].name)

    return c_ancestors


# ------------------------------------------------------ #
#
#                 FILTER OPERATIONS
#
# ------------------------------------------------------ #
def keep_gt_than_depth(min_depth, term_set, go_dag):
    filtered_terms = [t for t in term_set if go_dag[t].depth > min_depth]
    # return set(filtered_terms)
    return filtered_terms


def has_parents(parents, term, go_dag):
    term_parents = get_all_parents(go_dag, term)
    # return sorted(term_parents & parents) == sorted(parents)
    return len(term_parents & set(parents)) >= 1


def filter_has_parents(parents, term_set, go_dag):
    filtered_terms = [term for term in term_set if
                      has_parents(parents, term, go_dag)]
    return filtered_terms


# ------------------------------------------------------ #
#
#                  ALGORITHM IMPLMENTATIONS
#
# ------------------------------------------------------ #
def get_only_lca(term_sets, go_dag):
    """
    Compute the LCAs for two sets of string GO accessions
    """
    term_sets = validate_term_sets(term_sets)
    if len(term_sets) < 2:
        return []

    all_common_ancestors = get_all_commmon_ancestors_for_sets(term_sets,
                                                              go_dag)
    return get_deepest_term(all_common_ancestors, go_dag) * len(term_sets)


def get_lca_and_friends(term_sets, go_dag):
    """
    Compute union of sets of string GO accessions and their LCA(s)
    """
    term_sets = validate_term_sets(term_sets)
    if len(term_sets) == 0:
        return []
    if len(term_sets) == 1:
        return term_sets[0]

    leaf_terms = reduce(lambda x, y: x + y, term_sets)
    return leaf_terms + get_only_lca(term_sets, go_dag)


def get_up_to_lca(term_sets, go_dag):
    """
    Compute union of the two sets of string GO accessions, their LCAs \n
    and all the terms along the path to the LCAs.
    """
    term_sets = validate_term_sets(term_sets)
    if len(term_sets) == 0:
        return []
    if len(term_sets) == 1:
        return term_sets[0]

    # print 'Inducing on size: {}'.format(len(term_sets))

    lcas = get_only_lca(term_sets, go_dag)
    if len(lcas) == 0:
        # If there's no LCA then no terms can be induced.
        # This will happen if a protein is
        # Not annotated with anything for a particular ontology namespace.
        induced_terms = []
    else:
        lca_depth = go_dag[lcas[0]].depth
        all_ancestors = get_all_ancestors_for_sets(term_sets, go_dag)
        induced_terms = keep_gt_than_depth(lca_depth, all_ancestors, go_dag)
        induced_terms = filter_has_parents(lcas, induced_terms, go_dag)

    leaf_terms = reduce(lambda x, y: x + y, term_sets)
    ulca = induced_terms + leaf_terms + lcas

    # print "Lowest Common:"
    # for t in lcas:
    #     print '\t {} --> {}'.format(t, go_dag[t].name)
    #
    # print "Induced:"
    # def count(t, ts):
    #     return len([i for i in ts if i == t])
    #
    # induced_count = {k:count(k, ulca) for k in ulca}
    # for t in sorted(set(ulca), key=lambda x: go_dag[x].name):
    #     print '\t {}:{} --> {}'.format(t, induced_count[t], go_dag[t].name)

    return ulca


def get_without_lca(term_sets, go_dag):
    """
    Computes get_up_to_lca but removes the LCA terms.
    """
    term_sets = validate_term_sets(term_sets)
    if len(term_sets) == 0:
        return []
    if len(term_sets) == 1:
        return term_sets[0]

    ulca = get_up_to_lca(term_sets, go_dag)
    lca = get_only_lca(term_sets, go_dag)

    return [t for t in ulca if t not in lca]
