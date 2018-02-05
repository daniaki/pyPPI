#!/usr/bin/python


import gzip
from pyppi.data import psimi_obo_file

__PSIMI_GRAPH__ = None


def get_active_instance(**kwargs):
    global __PSIMI_GRAPH__
    if __PSIMI_GRAPH__ is None:
        filename = kwargs.get("filename", psimi_obo_file)
        __PSIMI_GRAPH__ = parse_miobo_file(filename)
    return __PSIMI_GRAPH__


# ------------------------------------------------------ #
#
#                         OBO PARSER
#
# ------------------------------------------------------ #
MiOntology = dict


class Term(object):

    def __init__(self, id, name, is_obsolete):
        self.id = id
        self.name = name
        self.is_obsolete = is_obsolete


def process_term(fp):
    id_ = None
    term = None
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

        elif line.startswith("is_obsolete"):
            _, is_obsolete = [x.strip() for x in line.split('is_obsolete: ')]
            is_obsolete = bool(is_obsolete)

        else:
            continue

    term = Term(id_, name, is_obsolete)
    return id_, alt_ids, term


def parse_miobo_file(filename):
    """
    Parses all Term objects into a dictionary of GOTerms. Each GOTerm
    contains a small subset of the possible keys: id, name, namespace, is_a,
    part_of and is_obsolete.
    """
    graph = MiOntology()
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
                tid, alt, term = process_term(fp)
                alt_id_map[tid] = alt
                graph[tid] = term
            else:
                continue

    # Turn the string ids into object references.
    for tid, alts in alt_id_map.items():
        term = graph[tid]
        for alt_tid in alts:
            graph[alt_tid] = term

    return graph
