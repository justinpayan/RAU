
from experiment_framework import run_experiment
from query_models import *
from solve_esw import solve_esw
from solve_usw import solve_usw
from utils import *

import argparse
import sys


def basic_baselines(dset_name, obj):
    if obj == "USW":
        solver = solve_usw
    elif obj == "ESW":
        solver = solve_esw
    else:
        print("obj must be USW or ESW")
        sys.exit(0)

    tpms, true_bids, covs, loads = load_dset(dset_name)
    print("Solving for max E[%s] using TPMS scores" % obj)
    expected_obj, alloc = solver(tpms, covs, loads)

    true_obj = 0
    if obj == "USW":
        true_obj = np.sum(alloc * true_bids)
    elif obj == "ESW":
        true_obj = np.min(np.sum(alloc * true_bids, axis=0))

    print("Solving for max %s using true bids" % obj)
    opt, _ = solver(true_bids, covs, loads)

    print("\n*******************\n*******************\n*******************\n")
    print("Stats for ", dset_name)
    print("E[%s] from using TPMS scores: %.2f" % (obj, expected_obj))
    print("True %s from using TPMS scores: %.2f" % (obj, true_obj))
    print("Optimal %s: %.2f" % (obj, opt))


def query_model(dset_name, obj, lamb, seed):
    tpms, true_bids, covs, loads = load_dset(dset_name)
    if obj == "USW":
        solver = solve_usw
    else:
        print("USW is the only allowed objective right now")
        sys.exit(0)
    query_model = GreedyMaxQueryModel(tpms, covs, loads, solver, dset_name)
    # query_model = VarianceReductionQueryModel(tpms, covs, loads, solver, dset_name)
    # query_model = SuperStarQueryModel(tpms, dset_name)
    # query_model = RandomQueryModel(tpms)
    expected_obj, alloc, total_bids = run_experiment(dset_name, query_model, solver, seed, lamb)

    true_obj = np.sum(alloc * true_bids)

    print("Number of reviewers: %d" % loads.shape[0])
    print("Number of bids issued: %d" % total_bids)
    print("E[%s] from using this query model: %.2f" % (obj, expected_obj))
    print("True %s from using TPMS scores: %.2f" % (obj, true_obj))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str)
    parser.add_argument("--lamb", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--obj", type=str, default="USW")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dset_name = args.dset_name
    lamb = args.lamb
    seed = args.seed
    obj = args.obj

    query_model(dset_name, obj, lamb, seed)


