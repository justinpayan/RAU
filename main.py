
from solve_usw import solve_usw_gurobi
from solve_max_min import get_worst_case, solve_max_min
from solve_max_expected_min import solve_max_expected_min
from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=31415)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dset_name = args.dset_name
    data_dir = args.data_dir
    seed = args.seed

    # Load in the estimated data matrix, and the true data matrix
    tpms, true_scores, covs, loads = load_dset(dset_name, seed, data_dir)

    # Save the data used for this run
    np.save("true_scores_%s_%d.npy" % (dset_name, seed), true_scores)

    # Run the max-min model
    error_bound = np.sqrt(np.sum((tpms - true_scores)**2)) * 1.0
    print("Error bound is: ", error_bound)
    fractional_alloc_max_min = solve_max_min(tpms, covs, loads, error_bound)
    np.save("fractional_max_min_alloc_%s_%d.npy" % (dset_name, seed), fractional_alloc_max_min)
    alloc_max_min = bvn(fractional_alloc_max_min)
    # alloc_max_min = fractional_alloc_max_min
    np.save("max_min_alloc_%s_%d.npy" % (dset_name, seed), alloc_max_min)

    # Run the baseline, which is just TPMS
    print("Solving for max USW using TPMS scores")
    objective_score, alloc = solve_usw_gurobi(tpms, covs, loads)

    np.save("tpms_alloc_%s_%d.npy" % (dset_name, seed), alloc)

    true_obj = np.sum(alloc * true_scores)

    print("Solving for max USW using true bids")
    opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

    np.save("opt_alloc_%s_%d.npy" % (dset_name, seed), opt_alloc)

    print(loads)
    print(np.sum(alloc_max_min, axis=1))
    print(covs)
    print(np.sum(alloc_max_min, axis=0))
    print(np.all(np.sum(alloc_max_min, axis=1) <= loads))
    print(np.all(np.sum(alloc_max_min, axis=0) == covs))
    true_obj_max_min = np.sum(alloc_max_min * true_scores)

    worst_s = get_worst_case(alloc, tpms, error_bound)
    worst_case_obj_tpms = np.sum(worst_s * alloc)

    worst_s = get_worst_case(alloc_max_min, tpms, error_bound)
    worst_case_obj_max_min = np.sum(worst_s * alloc_max_min)

    print("\n*******************\n*******************\n*******************\n")
    print("Stats for %s with seed %d" % (dset_name, seed))
    print("Optimal USW: %.2f" % opt)
    print("\n")
    print("Estimated USW from using TPMS scores: %.2f" % objective_score)
    print("True USW from using TPMS scores: %.2f" % true_obj)
    print("Worst case USW from using TPMS scores: %.2f" % worst_case_obj_tpms)
    print("Efficiency loss from using TPMS scores (percent of opt): %.2f" % (100 * (opt - true_obj) / opt))
    print("\n")
    print("True USW from max_min optimizer: %.2f" % true_obj_max_min)
    print("Worst case USW from max_min optimizer: %.2f" % worst_case_obj_max_min)
    print("Efficiency loss for max_min (percent of opt): %.2f" % (100 * (opt - true_obj_max_min) / opt))
    # print("True USW from max_expected_min optimizer: %.2f" % true_obj_max_min)
    # print("Worst case USW from max_expected_min optimizer: %.2f" % worst_case_obj_max_min)
    # print("Efficiency loss for max_expected_min (percent of opt): %.2f" % (100 * (opt - true_obj_max_min) / opt))
