import random
from tqdm import tqdm

from utils import *


def run_experiment(dset_name, query_model, solver, seed, lamb, data_dir, num_procs):
    tpms, true_bids, covs, loads = load_dset(dset_name, data_dir)

    print("%d reviewers, %d papers" % tpms.shape)

    m = loads.shape[0]

    rng = np.random.default_rng(seed)
    random.seed(seed)

    total_bids = 0

    # Iterate through the reviewers.
    for r in tqdm(sorted(range(m), key=lambda x: random.random())):
        num_bids = rng.poisson(lamb)
        total_bids += num_bids

        # For each bid, pick the next paper to query based on the model
        updated = True
        for _ in range(num_bids):
            # query = query_model.get_queries_parallel(r)[0]
            if updated:
                queries = query_model.get_query_parallel(r)
            # query = query_model.get_query(r)
            # queries = query_model.get_query(r)
            # query = query_model.get_query(r)
            query = queries.pop(0)
            updated = query_model.update(r, query, int(true_bids[r, query]))
            print("Next query for reviewer %d was %d" % (r, query))
            # for query in queries[:num_bids]:
            #     query_model.update(r, query, int(true_bids[r, query]))
            print("E[USW] right now: ", query_model.curr_expected_value)

        # for _ in range(num_bids):

        # query = query_model.get_query(r)
        # queries = query_model.get_queries(r)
        # print("Next query for reviewer %d was %d" % (r, query))
        # for query in queries[:num_bids]:
        #     query_model.update(r, query, int(true_bids[r, query]))
        # print("E[USW] right now: ", query_model.curr_expected_value)

    # Solve for the objective using the final v_tilde
    np.save(os.path.join(data_dir, "v_tilde_%s" % dset_name), query_model.v_tilde)
    # expected_obj, alloc = solver(query_model.v_tilde, covs, loads)
    # return expected_obj, alloc, total_bids