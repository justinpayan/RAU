
import numpy as np

from autoassigner import *


def pr4a(pra, covs, loads, iter_limit):
    # Normalize the affinities so they're between 0 and 1
    pra[np.where(pra < 0)] = np.float64(0.0)
    pra = pra/np.max(pra)

    pr4a_instance = auto_assigner(pra, demand=covs[0], ability=loads, iter_limit=iter_limit)
    pr4a_instance.fair_assignment()

    alloc = pr4a_instance.fa

    return alloc


def solve_esw(affinity_scores, covs, loads):
    # Use the PeerReview4All algorithm to maximize egalitarian social welfare
    iter_limit = 1

    alloc = pr4a(affinity_scores, covs, loads.astype(np.int64), iter_limit)

    matrix_alloc = np.zeros(affinity_scores.shape)
    for p in alloc:
        for r in alloc[p]:
            matrix_alloc[r, p] = 1

    print(np.sum(matrix_alloc, axis=0))
    values = np.sum(matrix_alloc * affinity_scores, axis=0)
    print(values)
    # Return the egalitarian welfare and the allocation as a matrix
    return np.min(values), matrix_alloc