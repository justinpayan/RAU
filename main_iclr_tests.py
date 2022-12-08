import pickle
import math

from solve_usw import solve_usw_gurobi
from solve_max_min import get_worst_case, solve_max_min, solve_max_min_project_each_step
from solve_max_expected_min import solve_max_expected_min
from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir

    noise_model = "ellipse"

    for year in range(2018, 2023):
        for seed in range(10):
            fname = "stat_dict_iclr_%d_%d.pkl" % (year, seed)
            if not os.path.isfile(fname):
                # Load in the ellipse
                std_devs = np.load(os.path.join("data", "iclr", "scores_sigma_iclr_%d.npy" % year))
                error_distrib = 2 * std_devs
                u_mag = 1
                means = np.load(os.path.join("data", "iclr", "scores_mu_iclr_%d.npy" % year))

                # Take a subsample of the reviewers and papers
                m, n = means.shape
                sampled_revs = np.random.choice(range(m), math.floor(.9*m))
                sampled_paps = np.random.choice(range(n), math.floor(.9*n))

                error_distrib = error_distrib[sampled_revs, :][:, sampled_paps]
                means = means[sampled_revs, :][:, sampled_paps]

                covs = np.ones(math.floor(.9*n)) * 3
                loads = np.ones(math.floor(.9*m)) * 6

                # Save the data used for this run
                np.save("error_distrib_iclr_%d_%d.npy" % (year, seed), error_distrib)
                np.save("means_iclr_%d_%d.npy" % (year, seed), means)

                # Run the max-min model
                # fractional_alloc_max_min = solve_max_min_project_each_step(tpms, covs, loads, error_bound)
                fractional_alloc_max_min = solve_max_min(means, covs, loads, error_distrib, u_mag, noise_model=noise_model)

                np.save("fractional_max_min_alloc_iclr_%d_%d.npy" % (year, seed), fractional_alloc_max_min)
                # alloc_max_min = best_of_n_bvn(fractional_alloc_max_min, tpms, error_bound, n=10)
                alloc_max_min = bvn(fractional_alloc_max_min)
                np.save("max_min_alloc_iclr_%d_%d.npy" % (year, seed), alloc_max_min)

                # Run the baseline, which is just TPMS
                print("Solving for max USW using TPMS scores")
                objective_score, alloc = solve_usw_gurobi(means, covs, loads)

                np.save("tpms_alloc_iclr_%d_%d.npy" % (year, seed), alloc)

                # true_obj = np.sum(alloc * true_scores)

                # print("Solving for max USW using true bids")
                # opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

                # np.save("opt_alloc_%s_%d_%.1f.npy" % (dset_name, seed, alpha), opt_alloc)

                print(loads)
                print(np.sum(alloc_max_min, axis=1))
                print(covs)
                print(np.sum(alloc_max_min, axis=0))
                print(np.all(np.sum(alloc_max_min, axis=1) <= loads))
                print(np.all(np.sum(alloc_max_min, axis=0) == covs))

                worst_s = get_worst_case(alloc, means, error_distrib, u_mag, noise_model=noise_model)
                worst_case_obj_tpms = np.sum(worst_s * alloc)

                worst_s = get_worst_case(alloc_max_min, means, error_distrib, u_mag, noise_model=noise_model)
                worst_case_obj_max_min = np.sum(worst_s * alloc_max_min)

                stat_dict = {}
                print("\n*******************\n*******************\n*******************\n")
                print("Stats for ICLR %d with seed %d" % (year, seed))
                # print("Optimal USW: %.2f" % opt)
                # stat_dict['opt_usw'] = opt
                print("\n")
                print("Estimated USW from using TPMS scores: %.2f" % objective_score)
                stat_dict['est_usw_tpms'] = objective_score
                # print("True USW from using TPMS scores: %.2f" % true_obj)
                # stat_dict['true_usw_tpms'] = true_obj
                print("Worst case USW from using TPMS scores: %.2f" % worst_case_obj_tpms)
                stat_dict['worst_usw_tpms'] = worst_case_obj_tpms
                # print("Efficiency loss from using TPMS scores (percent of opt): %.2f" % (100 * (opt - true_obj) / opt))
                print("\n")
                # print("True USW from max_min optimizer: %.2f" % true_obj_max_min)
                # stat_dict['true_usw_maxmin'] = true_obj_max_min
                print("Worst case USW from max_min optimizer: %.2f" % worst_case_obj_max_min)
                stat_dict['worst_usw_maxmin'] = worst_case_obj_max_min
                # print("Efficiency loss for max_min (percent of opt): %.2f" % (100 * (opt - true_obj_max_min) / opt))

                with open(fname, 'wb') as f:
                    pickle.dump(stat_dict, f)
