
from solve_usw import solve_usw_gurobi
from solve_min_max import solve_min_max, get_worst_case
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

    # Run the baseline, which is just TPMS
    print("Solving for max USW using TPMS scores")
    objective_score, alloc = solve_usw_gurobi(tpms, covs, loads)

    true_obj = np.sum(alloc * true_scores)

    print("Solving for max USW using true bids")
    opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

    # Run the min-max model
    error_bound = np.sqrt(np.sum((tpms - true_scores)**2)) * 1.1
    print(error_bound)
    alloc_min_max = solve_min_max(tpms, covs, loads, error_bound)
    true_obj_min_max = np.sum(alloc_min_max * true_scores)

    worst_s = get_worst_case(alloc, tpms, error_bound)
    worst_case_obj_tpms = np.sum(worst_s * alloc)

    worst_s = get_worst_case(alloc_min_max, tpms, error_bound)
    worst_case_obj_min_max = np.sum(worst_s * alloc_min_max)

    print("\n*******************\n*******************\n*******************\n")
    print("Stats for ", dset_name)
    print("Estimated USW from using TPMS scores: %.2f" % objective_score)
    print("True USW from using TPMS scores: %.2f" % true_obj)
    print("Worst case USW from using TPMS scores: %.2f" % worst_case_obj_tpms)
    print("Optimal USW: %.2f" % opt)
    print("Efficiency loss (percent of opt): %.2f" % (100 * (opt - true_obj) / opt))

    print("Worst case USW from min_max optimizer: %.2f" % worst_case_obj_min_max)
    print("True USW from min_max optimizer: %.2f" % true_obj_min_max)
    print("Efficiency loss for min_max (percent of opt): %.2f" % (100 * (opt - true_obj_min_max) / opt))
