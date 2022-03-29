import random
from tqdm import tqdm

from utils import *


def run_experiment(dset_name, query_model, seed, lamb, data_dir):
    tpms, true_bids, covs, loads = load_dset(dset_name, seed, data_dir)

    print("%d reviewers, %d papers" % tpms.shape)
    print("Query model: %s" % query_model)
    print("Lambda: %d" % lamb)
    print("Dataset: %s" % dset_name)
    print("Seed: %d" % seed)

    m = loads.shape[0]

    rng = np.random.default_rng(seed)
    random.seed(seed)

    total_bids = 0

    # Iterate through the reviewers.
    for r in tqdm(sorted(range(m), key=lambda x: random.random())):
        num_bids = rng.poisson(lamb)
        total_bids += num_bids

        # For each bid, pick the next paper to query based on the model
        for _ in range(num_bids):
            query = query_model.get_query(r)
            query_model.update(r, query, int(true_bids[r, query]))
            print("Next query for reviewer %d was %d" % (r, query))

    os.makedirs(os.path.join(data_dir, "v_tildes"), exist_ok=True)

    v_tilde_fname = os.path.join(data_dir, "v_tildes", "v_tilde_%s_%s_%d" % (dset_name, query_model, seed))

    print("Used %d bids. Saving in %s" % (total_bids, v_tilde_fname))

    # Save the final v_tilde, to be loaded and solved using Gurobi separately
    np.save(v_tilde_fname, query_model.v_tilde)
