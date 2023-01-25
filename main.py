import pickle

from solve_usw import solve_usw_gurobi
from solve_max_min import solve_max_min
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
    # seed = args.seed

    noise_model = "ellipse"

    for alpha in np.arange(0, 1.01, .1):
        for seed in range(10):
            fname = "stat_dict_%s_%d_%.1f.pkl" % (dset_name, seed, alpha)
            if not os.path.isfile(fname):
                # Load in the estimated data matrix, and the true data matrix
                tpms, true_scores, covs, loads, error_distrib, u_mag = load_dset(dset_name,
                                                                                 seed,
                                                                                 data_dir,
                                                                                 noise_model=noise_model,
                                                                                 alpha=alpha)

                # Save the data used for this run
                np.save("true_scores_%s_%d_%.1f.npy" % (dset_name, seed, alpha), true_scores)

                # Run the max-min model
                # fractional_alloc_max_min = solve_max_min_project_each_step(tpms, covs, loads, error_bound)
                fractional_alloc_max_min = solve_max_min(tpms, covs, loads, error_distrib, u_mag, noise_model=noise_model)

                np.save("fractional_max_min_alloc_%s_%d_%.1f.npy" % (dset_name, seed, alpha), fractional_alloc_max_min)
                # alloc_max_min = best_of_n_bvn(fractional_alloc_max_min, tpms, error_bound, n=10)
                alloc_max_min = bvn(fractional_alloc_max_min)
                np.save("max_min_alloc_%s_%d_%.1f.npy" % (dset_name, seed, alpha), alloc_max_min)

                # Run the baseline, which is just TPMS
                print("Solving for max USW using TPMS scores")
                objective_score, alloc = solve_usw_gurobi(tpms, covs, loads)

                np.save("tpms_alloc_%s_%d_%.1f.npy" % (dset_name, seed, alpha), alloc)

                true_obj = np.sum(alloc * true_scores)

                print("Solving for max USW using true bids")
                opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

                np.save("opt_alloc_%s_%d_%.1f.npy" % (dset_name, seed, alpha), opt_alloc)

                print(loads)
                print(np.sum(alloc_max_min, axis=1))
                print(covs)
                print(np.sum(alloc_max_min, axis=0))
                print(np.all(np.sum(alloc_max_min, axis=1) <= loads))
                print(np.all(np.sum(alloc_max_min, axis=0) == covs))
                true_obj_max_min = np.sum(alloc_max_min * true_scores)

                worst_s = get_worst_case(alloc, tpms, error_distrib, u_mag, noise_model=noise_model)
                worst_case_obj_tpms = np.sum(worst_s * alloc)

                worst_s = get_worst_case(alloc_max_min, tpms, error_distrib, u_mag, noise_model=noise_model)
                worst_case_obj_max_min = np.sum(worst_s * alloc_max_min)

                stat_dict = {}
                print("\n*******************\n*******************\n*******************\n")
                print("Stats for %s with seed %d and alpha=%.1f" % (dset_name, seed, alpha))
                print("Optimal USW: %.2f" % opt)
                stat_dict['opt_usw'] = opt
                print("\n")
                print("Estimated USW from using TPMS scores: %.2f" % objective_score)
                stat_dict['est_usw_tpms'] = objective_score
                print("True USW from using TPMS scores: %.2f" % true_obj)
                stat_dict['true_usw_tpms'] = true_obj
                print("Worst case USW from using TPMS scores: %.2f" % worst_case_obj_tpms)
                stat_dict['worst_usw_tpms'] = worst_case_obj_tpms
                print("Efficiency loss from using TPMS scores (percent of opt): %.2f" % (100 * (opt - true_obj) / opt))
                print("\n")
                print("True USW from max_min optimizer: %.2f" % true_obj_max_min)
                stat_dict['true_usw_maxmin'] = true_obj_max_min
                print("Worst case USW from max_min optimizer: %.2f" % worst_case_obj_max_min)
                stat_dict['worst_usw_maxmin'] = worst_case_obj_max_min
                print("Efficiency loss for max_min (percent of opt): %.2f" % (100 * (opt - true_obj_max_min) / opt))

                with open(fname, 'wb') as f:
                    pickle.dump(stat_dict, f)
