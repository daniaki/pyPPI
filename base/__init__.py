#!/usr/bin/env python

"""
Collection of utility operations that don't go anywhere else.
"""

import multiprocessing
import numpy as np
import tempfile
import igraph
import os


def process_queue(func, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, func(x)))


def parallel_map(func, iterator_to_parallelize, n_jobs):
    q_in = multiprocessing.Queue(maxsize=1)
    q_out = multiprocessing.Queue()

    # Build the list of processes.
    proc = [
        multiprocessing.Process(target=process_queue, args=(func, q_in, q_out))
        for _ in range(n_jobs)
    ]

    # Make each process a daemon to intercept calls, and start each one
    for p in proc:
        p.daemon = True
        p.start()

    # Send all the items in the iterator_to_parallelize to the job queue
    sent = [q_in.put(obj=(i, x))
            for i, x in enumerate(iterator_to_parallelize)]

    # Initialise the Queue
    [q_in.put((None, None)) for _ in range(n_jobs)]

    # Get the results from the current Queue
    accumulator = [q_out.get() for _ in range(len(sent))]

    # Join the spawned processes so they can be released.
    [p.join() for p in proc]

    return [x for i, x in sorted(accumulator, key=lambda item: item[0])]


def su_make_dir(path, mode=0o777):
    if not path or os.path.exists(path):
        print("{} already exists...".format(path))
    else:
        os.mkdir(path)
        os.chmod(path, mode)


def pretty_print_dict(dictionary, n_tabs=0):
    for k in sorted(dictionary.keys()):
        print('\t'*n_tabs + '{}:\t{}'.format(k, dictionary[k]))


def create_seeds(size):
    np.random.seed(42)
    ii32 = np.iinfo(np.int32)
    max_int = ii32.max
    seeds = np.random.random_integers(low=0, high=max_int, size=size)
    return seeds


def chunks(ls, n):
    """Yield successive n-sized chunks from l."""
    if n == 0:
        return []
    if n == 1:
        return ls

    ranges = list(range(0, len(ls), int(np.ceil(len(ls)/n))))
    tup_ranges = []
    for i in range(len(ranges)-1):
        tup_ranges.append((ranges[i], ranges[i+1]))
    tup_ranges.append((ranges[i+1], len(ls) + 1))
    for (i, j) in tup_ranges:
        yield ls[i: j]


def igraph_from_tuples(v_names):
    tmp = tempfile.mktemp(suffix='.ncol', prefix='graph_', dir='tmp/')
    fp = open(tmp, "w")
    for (name_a, name_b) in v_names:
        fp.write("{} {}\n".format(name_a, name_b))
    fp.close()
    # will store node names under the 'name' vertex attribute.
    g = igraph.read(tmp, format='ncol', directed=False, names=True)
    os.remove(tmp)
    return g


def igraph_vid_attr_table(igraph_g, attr):
    vid_prop_lookup = {}
    for v in igraph_g.vs:
        data = v[attr]
        vid_prop_lookup[v.index] = data
    return vid_prop_lookup


def write_cytoscape_attr_file(attr_name, attr_tuples, edge, fp):
    if len(attr_tuples) == 0:
        raise ValueError("attr_tuples must not be empty.")
    if edge:
        if len(attr_tuples[0]) != 3:
            raise ValueError("Tuples must have length 3 for edge attr files.")
        fp.write('Name\t{}\n'.format(attr_name.replace(" ", '-')))
        fp.write('\n'.join(('{} (pp) {}\t{}'.format(a, b, attr)
                            for (a, b, attr) in attr_tuples)))
    else:
        if len(attr_tuples[0]) != 2:
            raise ValueError("Tuples must have length 2 for node attr files.")
        fp.write('Name\t{}\n'.format(attr_name.replace(" ", '-')))
        fp.write('\n'.join(('{}\t{}'.format(a, attr)
                            for (a, attr) in attr_tuples)))
    fp.close()


def write_interactions(tuples, fp):
    fp.write('source\ttarget\tinteraction\n')
    fp.write('\n'.join(('{}\t{}\tpp'.format(a, b) for (a, b) in tuples)))
    fp.close()

