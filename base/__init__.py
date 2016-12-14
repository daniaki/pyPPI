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


def chunk_list(ls, n):
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


class PPI(object):
    """
    Simple class to contain some basic functionality to represent a PPI
    instance.
    """

    def __init__(self, p1, p2):
        self.__proteins = tuple(sorted((str(p1), str(p2))))
        self.__p1 = self.__proteins[0]
        self.__p2 = self.__proteins[1]

    @property
    def p1(self):
        return self.__p1

    @property
    def p2(self):
        return self.__p2

    @property
    def proteins(self):
        return self.__proteins

    def __repr__(self):
        return 'PPI({}, {})'.format(self.__p1, self.__p2)

    def __str__(self):
        return 'PPI({}, {})'.format(self.__p1, self.__p2)

    def __hash__(self):
        return hash(self.__proteins)

    def __reversed__(self):
        return PPI(self.__p2, self.__p1)

    def __contains__(self, item):
        return item in self.__proteins

    def __len__(self):
        return len(self.__proteins)

    def __eq__(self, other):
        return self.__proteins == other.proteins

    def __ne__(self, other):
        return not self.__proteins == other.proteins

    def __le__(self, other):
        return self.__p1 <= other.p1

    def __ge__(self, other):
        return self.__p1 >= other.p1

    def __lt__(self, other):
        return self.__p1 < other.p1

    def __gt__(self, other):
        return self.__p1 > other.p1

    def __iter__(self):
        return iter(self.__proteins)