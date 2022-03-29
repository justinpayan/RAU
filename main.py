
from experiment_framework import run_experiment
from query_models import *
# from solve_esw import solve_esw
from solve_usw import solve_usw, solve_usw_gurobi
from utils import *

import argparse
import sys


def basic_baselines(dset_name, seed, data_dir, obj):
    if obj == "USW":
        solver = solve_usw_gurobi
    elif obj == "ESW":
        solver = solve_esw
    else:
        print("obj must be USW or ESW")
        sys.exit(0)

    print("Dataset: %s" % dset_name)

    tpms, true_bids, covs, loads = load_dset(dset_name, seed, data_dir)
    print("Solving for max E[%s] using TPMS scores" % obj)
    expected_obj, alloc = solver(tpms, covs, loads)

    os.makedirs(os.path.join("saved_init_expected_usw"), exist_ok=True)
    os.makedirs(os.path.join("saved_init_max_usw_soln"), exist_ok=True)

    np.save(os.path.join("saved_init_expected_usw", dset_name), expected_obj)
    np.save(os.path.join("saved_init_max_usw_soln", dset_name), alloc)

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
        query_model = GreedyMaxQueryModel(tpms, covs, loads, solver, dset_name)
        # query_model = GreedyMaxQueryModelParallel(tpms, covs, loads, solver, dset_name, data_dir, num_procs)
    elif query_model_type == "var":
        query_model = VarianceReductionQueryModel(tpms, covs, loads, solver, dset_name)
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

