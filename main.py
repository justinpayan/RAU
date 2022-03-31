
from experiment_framework import run_experiment
from query_models import *
from collections import Counter
import matplotlib.pyplot as plt
# from solve_esw import solve_esw
from solve_usw import solve_usw, solve_usw_gurobi
from utils import *

import argparse
import sys


def basic_baselines(dset_name, seed, data_dir, obj):
    if obj == "USW":
        solver = solve_usw_gurobi
    else:
        print("obj must be USW")
        sys.exit(0)

    print("Dataset: %s" % dset_name)

    tpms, true_bids, covs, loads = load_dset(dset_name, seed, data_dir)
    print("Solving for max E[%s] using TPMS scores" % obj)
    expected_obj, alloc = solver(tpms, covs, loads)

    os.makedirs(os.path.join(data_dir, "saved_init_expected_usw"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "saved_init_max_usw_soln"), exist_ok=True)

    np.save(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d" % (dset_name, seed)), expected_obj)
    np.save(os.path.join(data_dir, "saved_init_max_usw_soln", "%s_%d" % (dset_name, seed)), alloc)

    true_obj = 0
    if obj == "USW":
        true_obj = np.sum(alloc * true_bids)

    print("Solving for max %s using true bids" % obj)
    opt, opt_alloc = solver(true_bids, covs, loads)

    np.save(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d_true" % (dset_name, seed)), true_obj)
    np.save(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d_opt" % (dset_name, seed)), opt)
    np.save(os.path.join(data_dir, "saved_init_max_usw_soln", "%s_%d_opt" % (dset_name, seed)), opt_alloc)

    print("\n*******************\n*******************\n*******************\n")
    print("Stats for ", dset_name)
    print("E[%s] from using TPMS scores: %.2f" % (obj, expected_obj))
    print("True %s from using TPMS scores: %.2f" % (obj, true_obj))
    print("Optimal %s: %.2f" % (obj, opt))
    print("Min number of papers per reviewer (init): %d" % np.min(np.sum(alloc, axis=1)))
    print("Max number of papers per reviewer (init): %d" % np.max(np.sum(alloc, axis=1)))
    print("Min number of papers per reviewer (opt): %d" % np.min(np.sum(opt_alloc, axis=1)))
    print("Max number of papers per reviewer (opt): %d" % np.max(np.sum(opt_alloc, axis=1)))


def query_model(dset_name, obj, lamb, seed, data_dir, query_model_type):
    tpms, true_bids, covs, loads = load_dset(dset_name, seed, data_dir)
    if obj == "USW":
        # For now, don't use a solver, we can just expect to load the starting solutions from disk,
        # and at the end we will write out the v_tilde for input to gurobi separately.
        solver = None
        # solver = solve_usw
    else:
        print("USW is the only allowed objective right now")
        sys.exit(0)

    if query_model_type == "greedymax":
        query_model = GreedyMaxQueryModel(tpms, covs, loads, solver, data_dir, dset_name)
        # query_model = GreedyMaxQueryModelParallel(tpms, covs, loads, solver, dset_name, data_dir, num_procs)
    elif query_model_type == "var":
        query_model = VarianceReductionQueryModel(tpms, covs, loads, solver, dset_name)
    elif query_model_type == "supergreedymax":
        query_model = SuperStarGreedyMaxQueryModel(tpms, covs, loads, solver, dset_name, data_dir, k=30, b=3, d=4, beam_sz=10, max_iters=2)
    elif query_model_type == "random":
        query_model = RandomQueryModel(tpms, dset_name)
    elif query_model_type == "superstar":
        query_model = SuperStarQueryModel(tpms, dset_name)
    elif query_model_type == "tpms":
        query_model = TpmsQueryModel(tpms, dset_name)
    elif query_model_type == "uncertainty":
        pass
        # query_model = UncertaintyQueryModel(tpms, dset_name)

    run_experiment(dset_name, query_model, seed, lamb, data_dir)


def check_stats(dset_name, data_dir):
    bid_cts = []
    for seed in range(10):
        tpms, true_bids, covs, loads = load_dset(dset_name, seed, data_dir)
        num_bids_per_rev = np.sum(true_bids, axis=1).tolist()
        bid_cts.extend(num_bids_per_rev)
    plt.hist(bid_cts, density=True, bins=40)
    plt.ylabel("Probability Density")
    plt.xlabel("Number of Acceptable Papers")
    plt.savefig("densplotnumbidsperrev_%s" % dset_name)


def final_solver_swarm(dset_name, obj, lamb, seed, data_dir, query_model):
    tpms, true_bids, covs, loads = load_dset(dset_name, seed, data_dir)
    v_tilde = np.load(os.path.join(data_dir, "v_tildes", "v_tilde_%s_%s_%d.npy" % (dset_name, query_model, seed)))
    if obj == "USW":
        solver = solve_usw_gurobi
    else:
        print("USW is the only allowed objective right now")
        sys.exit(0)
    expected_obj, alloc = solver(v_tilde, covs, loads)
    true_obj = np.sum(alloc * true_bids)

    print("Number of reviewers: %d" % loads.shape[0])
    # print("Number of bids issued: %d" % total_bids)
    print("E[%s] from using this query model: %.2f" % (obj, expected_obj))
    print("True %s from using this model: %.2f" % (obj, true_obj))

    with open(os.path.join(data_dir, "v_tildes", "expected_obj_%s_%s_%d" % (dset_name, query_model, seed)), 'w') as f:
        f.write(str(expected_obj))

    with open(os.path.join(data_dir, "v_tildes", "true_obj_%s_%s_%d" % (dset_name, query_model, seed)), 'w') as f:
        f.write(str(true_obj))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--lamb", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--obj", type=str, default="USW")
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--query_model", type=str, default="random")
    parser.add_argument("--mode", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dset_name = args.dset_name
    data_dir = args.data_dir
    lamb = args.lamb
    seed = args.seed
    obj = args.obj
    num_procs = args.num_procs
    query_model_type = args.query_model
    mode = args.mode

    if mode == "query_exps":
        query_model(dset_name, obj, lamb, seed, data_dir, query_model_type)
    elif mode == "basic_baselines":
        basic_baselines(dset_name, seed, data_dir, obj)
    elif mode == "final_solver":
        final_solver_swarm(dset_name, obj, lamb, seed, data_dir, query_model_type)
    elif mode == "check_stats":
        check_stats(dset_name, data_dir)

