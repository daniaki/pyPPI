#!/usr/bin/python


import gzip
from functools import reduce
from operator import itemgetter

from ..data import obo_file

__GODAG__ = None


def get_active_instance(**kwargs):
    global __GODAG__
    if __GODAG__ is None:
        filename = kwargs.get("filename", obo_file)
        __GODAG__ = parse_obo12_file(filename)
    return __GODAG__


# ------------------------------------------------------ #
#
#                         OBO PARSER
#
# ------------------------------------------------------ #
GODag = dict


class GOTerm(object):

    def __init__(self, id, name, namespace, is_a, part_of, is_obsolete):
        self.id = id
        self.name = name
        self.namespace = namespace
        self.is_a = set(is_a)
        self.part_of = set(part_of)
        self.has_part = set()
        self.has_a = set()
        self.is_obsolete = is_obsolete
        self._depth = None

    def __str__(self):
        return str({
            "id": self.id,
            "name": self.name,
            "namespace": self.namespace,
            "is_a": sorted([x.id for x in self.is_a]),
            "part_of": sorted([x.id for x in self.part_of]),
            "is_obsolete": self.is_obsolete
        })

    def __le__(self, other):
        return self.id <= other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    def has_parent(self, p):
        return p in self.parents

    def has_ancestor(self, a):
        return a in self.all_parents

    @property
    def parents(self):
        return self.is_a | self.part_of

    @property
    def all_parents(self):
        all_parents = set()
        queue = list(self.parents)
        while queue:
            node = queue.pop()
            all_parents.add(node)
            queue += list(node.parents)
        return all_parents

    @property
    def depth(self):
        if self._depth is not None:
            return self._depth
        else:
            if not self.parents:
                return 0
            self._depth = max([t.depth for t in self.parents]) + 1
            return self._depth


def process_go_term(fp):
    id_ = None
    term = None
    part_of = []
    is_a = []
    line = "[Term]"
    is_obsolete = False
    alt_ids = []

    while line.strip() != "":
        line = fp.readline().strip()
        if line.startswith("id:"):
            _, id_ = [x.strip() for x in line.split('id: ')]

        elif line.startswith("alt_id:"):
            _, alt_id = [x.strip() for x in line.split('alt_id: ')]
            alt_ids += [alt_id]

        elif line.startswith("name:"):
            _, name = [x.strip() for x in line.split('name: ')]

        elif line.startswith("namespace:"):
            _, namespace = [x.strip() for x in line.split('namespace: ')]

        elif line.startswith("is_a:"):
            _, is_a_term = [x.strip() for x in line.split('is_a: ')]
            is_a_term, _ = [x.strip() for x in is_a_term.split(' ! ')]
            is_a.append(is_a_term)

        elif line.startswith("relationship: part_of"):
            _, part_of_term = [
                x.strip() for x in line.split('relationship: part_of ')
            ]
            part_of_term, _ = [x.strip() for x in part_of_term.split(' ! ')]
            part_of.append(part_of_term)

        elif line.startswith("is_obsolete"):
            _, is_obsolete = [x.strip() for x in line.split('is_obsolete: ')]
            is_obsolete = bool(is_obsolete)

        else:
            continue

    term = GOTerm(id_, name, namespace, is_a, part_of, is_obsolete)
    return id_, alt_ids, term


def parse_obo12_file(filename):
    """
    Parses all Term objects into a dictionary of GOTerms. Each GOTerm
    contains a small subset of the possible keys: id, name, namespace, is_a,
    part_of and is_obsolete.
    """
    dag = GODag()
    alt_id_map = {}
    with gzip.open(filename, 'rt') as fp:
        for line in fp:
            line = line.strip()
            if "format-version" in line:
                _, version = [x.strip() for x in line.split(":")]
                version = float(version)
                if version != 1.2:
                    raise ValueError("Parser only supports version 1.2.")
            elif "[Term]" in line:
                tid, alt, term = process_go_term(fp)
                alt_id_map[tid] = alt
                dag[tid] = term
            else:
                continue

    # Turn the string ids into object references.
    for _, item in dag.items():
        is_a_term_ids = item.is_a
        is_a_terms = []
        part_of_term_ids = item.part_of
        part_of_terms = []

        for t_id in is_a_term_ids:
            is_a_terms.append(dag[t_id])
            dag[t_id].has_a.add(item)

        for t_id in part_of_term_ids:
            part_of_terms.append(dag[t_id])
            dag[t_id].has_part.add(item)

        item.is_a = set(is_a_terms)
        item.part_of = set(part_of_terms)

    for tid, alts in alt_id_map.items():
        term = dag[tid]
        for alt_tid in alts:
            dag[alt_tid] = term

    return dag


def group_terms_by_ontology_type(term_ids, max_count=None):
    dag = get_active_instance()
    cc_terms = []
    bp_terms = []
    mf_terms = []

    for t in term_ids:
        if dag[t].namespace == "biological_process":
            if max_count is not None and bp_terms.count(t) >= max_count:
                continue
            else:
                bp_terms.append(t)
        elif dag[t].namespace == "cellular_component":
            if max_count is not None and cc_terms.count(t) >= max_count:
                continue
            else:
                cc_terms.append(t)
        elif dag[t].namespace == "molecular_function":
            if max_count is not None and mf_terms.count(t) >= max_count:
                continue
            else:
                mf_terms.append(t)
        else:
            raise ValueError("Term %s doesn't belong to any ontology." % t)

    return {'mf': mf_terms, 'bp': bp_terms, 'cc': cc_terms}


def filter_obsolete_terms(term_ids):
    dag = get_active_instance()
    return [tid for tid in term_ids if dag[tid].is_obsolete is False]

# ------------------------------------------------------ #
#
#                  ULCA Inducer
#
# ------------------------------------------------------ #


def get_lca_of_terms(terms):
    if not terms:
        return None

    dag = get_active_instance()
    parents = [t.all_parents for t in terms]
    common_parents = reduce(lambda x, y: x & y, parents)
    if not common_parents:
        return None
    depths = [(t, t.depth) for t in common_parents]
    _, max_depth = max(depths, key=itemgetter(1))
    lcas = set([t for t in common_parents if t.depth == max_depth])
    return list(lcas)


def get_up_to_lca(p1, p2):
    if not p1 or not p2:
        return p1 + p2

    dag = get_active_instance()
    p1 = [dag[t] for t in p1]
    p2 = [dag[t] for t in p2]
    lcas = get_lca_of_terms([t for ts in [p1, p2] for t in ts])
    if lcas is None:
        return [t.id for t in p1 + p2]

    induced_terms = []
    for term_set in [p1, p2]:
        induced_terms_for_set = set()
        for term in term_set:
            induced = [
                p for p in term.all_parents
                if all([p.has_ancestor(lca) for lca in lcas])
            ]
            induced_terms_for_set |= set(induced)
        induced_terms += list(induced_terms_for_set)

    induced_terms += lcas * 2
    induced_terms += p1
    induced_terms += p2
    return [t.id for t in induced_terms]
