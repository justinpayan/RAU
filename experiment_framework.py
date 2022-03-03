import random

from utils import *


def run_experiment(dset_name, query_model, solver, seed, lamb):
    tpms, true_bids, covs, loads = load_dset(dset_name)

    m = loads.shape[0]

    rng = np.random.default_rng(seed)
    random.seed(seed)

    total_bids = 0

    # Iterate through the reviewers.
    for r in sorted(range(m), key=lambda x: random.random()):
        num_bids = rng.poisson(lamb)
        total_bids += num_bids

        for _ in range(num_bids):
            # For each bid, pick the next paper to query based on the model
            query = query_model.get_query(r)
            print("Next query for reviewer %d was %d" % (r, query))
            query_model.update(r, query, true_bids[r, query])

    # Solve for the objective using the final v_tilde
    expected_obj, alloc = solver(query_model.v_tilde, covs, loads)
    return expected_obj, alloc, total_bids