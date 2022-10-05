import lap
import math
import numpy as np
import os

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
    true_scores = tpms.copy()
    n = true_scores.shape[1]
    underest_papers = rng.permutation(n)[:math.ceil(.2*n)]
    overest_papers = rng.permutation(n)[:math.ceil(.2*n)]
    true_scores[:, underest_papers] += .2
    true_scores[:, overest_papers] -= .2
    true_scores = np.clip(true_scores, 0, np.inf)
    true_scores /= np.max(true_scores)

    # One way to sample true scores would be to assume papers are either liked or disliked
    # noisy_tpms = tpms + rng.normal(-0.1, 0.1, tpms.shape)
    # noisy_tpms = np.clip(noisy_tpms, 0, 1)
    # true_scores = rng.uniform(0, 1, size=tpms.shape) < noisy_tpms

    return tpms, true_scores, covs, loads
