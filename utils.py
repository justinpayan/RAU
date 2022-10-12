import lap
import math
import numpy as np
import os

from collections import defaultdict
from itertools import product
from queue import Queue
from sortedcontainers import SortedList
import networkx as nx
# from floyd_warshall import floyd_warshall_single_core


import time


def load_dset(dname, seed, data_dir="."):
    tpms = np.load(os.path.join(data_dir, "data", dname, "scores.npy"))
    covs = np.load(os.path.join(data_dir, "data", dname, "covs.npy"))
    loads = np.load(os.path.join(data_dir, "data", dname, "loads.npy"))

    rng = np.random.default_rng(seed)

    tpms = np.clip(tpms, 0, np.inf)
    tpms /= np.max(tpms)

    # Sample the "true" scores.
    # For now, assume that 10% of papers are underestimated by .2 and 10% are overestimated by .2
    # We will do a collaborative filtering approach later
    # true_scores = tpms.copy()
    # n = true_scores.shape[1]
    # underest_papers = rng.permutation(n)[:math.ceil(.2*n)]
    # overest_papers = rng.permutation(n)[:math.ceil(.2*n)]
    # true_scores[:, underest_papers] += .05
    # true_scores[:, overest_papers] -= .05
    # true_scores = np.clip(true_scores, 0, np.inf)
    # true_scores /= np.max(true_scores)

    # One way to sample true scores would be to assume papers are either liked or disliked
    noisy_tpms = tpms + rng.normal(-0.05, 0.05, tpms.shape)
    noisy_tpms = np.clip(noisy_tpms, 0, 1)
    true_scores = noisy_tpms
    # true_scores = rng.uniform(0, 1, size=tpms.shape) < noisy_tpms

    return tpms, true_scores, covs, loads

# # If the path ends with a cycle, decycle.
# # The path starts with a paper. We need to return the (reviewer, paper)
# # edges that make up this path.
# def decycle_and_return_edges(path, has_cycle):
#
#

# # Finds a simple cycle or maximal path of paper-reviewer edges
# # that have not been rounded
# def find_path(fractional_alloc):
#     frac_edges = np.where((0 < fractional_alloc) * (fractional_alloc < 1))
#     paper_reviewer_edges = defaultdict(set)
#     reviewer_paper_edges = defaultdict(set)
#     for (r, p) in zip(frac_edges[0].tolist(), frac_edges[1].tolist()):
#         paper_reviewer_edges[p].add(r)
#         reviewer_paper_edges[r].add(p)
#
#     path = [frac_edges[0][1]]
#     paper_or_reviewer = "paper"
#     edge_maps = {"paper": paper_reviewer_edges,
#                  "reviewer": reviewer_paper_edges}
#
#     for _ in range(fractional_alloc.shape[0] * fractional_alloc.shape[1]):
#         curr = path[-1]
#
#         edges = edge_maps[paper_or_reviewer]
#
#         if curr not in edges:
#             return decycle_and_return_edges(path, False)
#
#         for next in edges[curr]:
#             if next not in path:
#                 path.append(next)
#                 break
#             else:
#                 path.append(next)
#                 return decycle_and_return_edges(path, True)
#
#         if paper_or_reviewer == "paper":
#             paper_or_reviewer = "reviewer"
#         else:
#             paper_or_reviewer = "paper"
#
#     # We should not get here
#     assert False


# Implements https://ieeexplore.ieee.org/abstract/document/1181955
# Use bvn.cpp from https://github.com/theryanl/mitigating_manipulation_via_randomized_reviewer_assignment/blob/master/core/bvn.cpp
def bvn(fractional_alloc):
    # # While there are fractional edges, find a simple cycle or maximal path
    # while np.any((0 < fractional_alloc) * (fractional_alloc < 1)):
    #     # Find a simple cycle or maximal path
    #     path = find_path(fractional_alloc)
    with open("fractional_alloc.txt", 'w') as f:
        m, n = fractional_alloc.shape
        f.write("%d %d\n" % (m, n))
        f.write("1\n"*m)
        asst_str = ""
        for r in range(m):
            for p in range(n):
                assn = 0
                if not np.isclose(fractional_alloc[r, p], 0):
                    assn = fractional_alloc[r, p]
                asst_str += "%d %d %.6f\n" % (r, p+m, assn)
        f.write(asst_str[:-1])

    os.system("/mnt/nfs/scratch1/jpayan/MinimalBidding/a.out < fractional_alloc.txt > output_bvn.txt")

    rounded_alloc = np.zeros(fractional_alloc.shape)
    with open("output_bvn.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            r, p = line.strip().split()
            r = int(r)
            p = int(p) - m
            rounded_alloc[r, p] = 1
    return rounded_alloc




