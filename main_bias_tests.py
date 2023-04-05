import pickle
import math

import numpy as np

from solve_usw import solve_usw_gurobi
from solve_max_min import solve_max_min
from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_dummy_revs", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    num_dummy_revs = args.num_dummy_revs*10
    seed = args.seed

    noise_model = "ellipse"

    fname = "stat_dict_dummy_revs_%d_%d.pkl" % (num_dummy_revs, seed)
    if not os.path.isfile(os.path.join(data_dir, "outputs", fname)):
        # Load in the ellipse
        # std_devs = np.load(os.path.join(data_dir, "data", "iclr", "scores_sigma_iclr_%d.npy" % year))
        # means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_%d.npy" % year))
        orig_means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_2018.npy"))
        # Sample a set of small std deviations for these reviewer-paper pairs. We will assume there is almost no noise.
        std_devs = np.zeros(orig_means.shape)

        # Add on the dummy reviewers
        m, n = orig_means.shape

        gen = np.random.default_rng(seed=seed)
        true_mean_dummies = .1
        std_dev_of_dummies = .3
        new_revs = np.clip(gen.normal(loc=true_mean_dummies, scale=std_dev_of_dummies, size=(num_dummy_revs, n)), 0, 1)
        means = np.vstack((orig_means, new_revs))
        std_devs = np.vstack((std_devs, np.ones(new_revs.shape)*std_dev_of_dummies))

        true_scores = np.vstack((orig_means, np.ones(new_revs.shape)*true_mean_dummies))

        m, n = means.shape

        covs = np.ones(math.floor(n)) * 3
        loads = np.ones(math.floor(m)) * 6

        # Save the data used for this run
        np.save(os.path.join(data_dir, "outputs", "std_devs_dummy_revs_%d_%d.npy" % (num_dummy_revs, seed)), std_devs)
        np.save(os.path.join(data_dir, "outputs", "means_dummy_revs_%d_%d.npy" % (num_dummy_revs, seed)), means)

        # Run the max-min model
        # fractional_alloc_max_min = solve_max_min_project_each_step(tpms, covs, loads, error_bound)
        fractional_alloc_max_min = solve_max_min(means, covs, loads, std_devs,
                                                 noise_model=noise_model, dykstra=True, caching=False)

        np.save(os.path.join(data_dir, "outputs", "fractional_max_min_alloc_dummy_revs_%d_%d.npy" % (num_dummy_revs, seed)), fractional_alloc_max_min)
        # alloc_max_min = best_of_n_bvn(fractional_alloc_max_min, tpms, error_bound, n=10)
        alloc_max_min = bvn(fractional_alloc_max_min)
        np.save(os.path.join(data_dir, "outputs", "max_min_alloc_dummy_revs_%d_%d.npy" % (num_dummy_revs, seed)), alloc_max_min)

        # Run the baseline, which is just TPMS
        print("Solving for max USW using TPMS scores", flush=True)
        objective_score, alloc = solve_usw_gurobi(means, covs, loads)

        np.save(os.path.join(data_dir, "outputs", "tpms_alloc_dummy_revs_%d_%d.npy" % (num_dummy_revs, seed)), alloc)

        true_obj = np.sum(alloc * true_scores)

        true_obj_max_min = np.sum(alloc_max_min * true_scores)

        print("Solving for max USW using true bids")
        opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

        np.save("opt_alloc_dummy_revs_%d_%d.npy" % (num_dummy_revs, seed), opt_alloc)

        print(loads)
        print(np.sum(alloc_max_min, axis=1))
        print(covs)
        print(np.sum(alloc_max_min, axis=0))
        print(np.all(np.sum(alloc_max_min, axis=1) <= loads))
        print(np.all(np.sum(alloc_max_min, axis=0) == covs))

        # worst_s = get_worst_case(alloc, means, std_devs, noise_model=noise_model)
        # worst_case_obj_tpms = np.sum(worst_s * alloc)
        #
        # worst_s = get_worst_case(alloc_max_min, means, std_devs, noise_model=noise_model)
        # worst_case_obj_max_min = np.sum(worst_s * alloc_max_min)

        stat_dict = {}
        print("\n*******************\n*******************\n*******************\n")
        print("Stats for Dummy Revs on ICLR 2018. Num dummies is %d, seed is %d" % (num_dummy_revs, seed))
        print("Optimal USW: %.2f" % opt)
        stat_dict['opt_usw'] = opt
        print("\n")
        print("Estimated USW from using TPMS scores: %.2f" % objective_score)
        stat_dict['est_usw_tpms'] = objective_score
        print("True USW from using TPMS scores: %.2f" % true_obj)
        stat_dict['true_usw_tpms'] = true_obj
        # print("Worst case USW from using TPMS scores: %.2f" % worst_case_obj_tpms)
        # stat_dict['worst_usw_tpms'] = worst_case_obj_tpms
        print("Efficiency loss from using TPMS scores (percent of opt): %.2f" % (100 * (opt - true_obj) / opt))
        print("\n")
        print("True USW from max_min optimizer: %.2f" % true_obj_max_min)
        stat_dict['true_usw_maxmin'] = true_obj_max_min
        # print("Worst case USW from max_min optimizer: %.2f" % worst_case_obj_max_min)
        # stat_dict['worst_usw_maxmin'] = worst_case_obj_max_min
        print("Efficiency loss for max_min (percent of opt): %.2f" % (100 * (opt - true_obj_max_min) / opt))

        with open(os.path.join(data_dir, "outputs", fname), 'wb') as f:
            pickle.dump(stat_dict, f)
