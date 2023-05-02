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
    parser.add_argument("--num_dummies", type=int)
    parser.add_argument("--conf", type=str, default="midl")
    parser.add_argument("--revs_or_paps", type=str, default="revs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    num_dummies = args.num_dummies
    seed = args.seed
    conf = args.conf
    revs_or_paps = args.revs_or_paps

    noise_model = "ellipse"

    if revs_or_paps == "revs":
        fname = "stat_dict_dummy_revs_%s_%d_%d.pkl" % (conf, num_dummies, seed)
        gen = np.random.default_rng(seed=seed)
        # Load in the ellipse
        # std_devs = np.load(os.path.join(data_dir, "data", "iclr", "scores_sigma_iclr_%d.npy" % year))
        # means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_%d.npy" % year))
        orig_means = np.clip(np.load(os.path.join(data_dir, "data", conf, "scores.npy")), 0, 1)
        m, n = orig_means.shape
        # Sample a set of small std deviations for these reviewer-paper pairs. We will assume there is almost no noise.
        std_dev_of_real = .02
        std_devs = np.ones(orig_means.shape)*std_dev_of_real
        noisy_means = orig_means + gen.normal(loc=0, scale=std_dev_of_real, size=(m, n))

        # Add on the dummy reviewers
        true_mean_dummies = .1
        std_dev_of_dummies = .15
        new_revs = gen.normal(loc=true_mean_dummies, scale=std_dev_of_dummies, size=(num_dummies, n))
        means = np.vstack((noisy_means, new_revs))
        std_devs = np.vstack((std_devs, np.ones(new_revs.shape)*std_dev_of_dummies))

        true_scores = np.vstack((orig_means, np.ones(new_revs.shape)*true_mean_dummies))
    elif revs_or_paps == "paps":
        # We are just picking a subset of papers and randomly adding some noise to their estimated scores.
        fname = "stat_dict_dummy_paps_%s_%d_%d.pkl" % (conf, num_dummies, seed)
        gen = np.random.default_rng(seed=seed)
        # Load in the ellipse
        # std_devs = np.load(os.path.join(data_dir, "data", "iclr", "scores_sigma_iclr_%d.npy" % year))
        # means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_%d.npy" % year))
        orig_means = np.clip(np.load(os.path.join(data_dir, "data", conf, "scores.npy")), 0, 1)
        m, n = orig_means.shape
        # Sample a set of small std deviations for these reviewer-paper pairs. We will assume there is almost no noise.

        # Let's actually say that some of the reviewers get the artificial bump

        std_dev_of_real = .02
        std_dev_of_dummies = .15
        dummy_paps = np.random.choice(n, num_dummies, replace=False)
        top_revs = np.argsort(orig_means, axis=0)[::-1]
        mid_revs = top_revs[20:30, dummy_paps]
        # dummy_revs = np.random.choice(m, num_dummies, replace=False)
        std_devs = np.ones(orig_means.shape) * std_dev_of_real
        loc = np.zeros(orig_means.shape)

        for p_idx, p in enumerate(dummy_paps):
            std_devs[mid_revs[:, p_idx], [p]*10] = 0
            loc[mid_revs[:, p_idx], [p]*10] = 2*std_dev_of_dummies

        means = orig_means + gen.normal(loc=loc, scale=std_devs)

        for p_idx, p in enumerate(dummy_paps):
            std_devs[mid_revs[:, p_idx], [p]*10] = std_dev_of_dummies

        true_scores = orig_means.copy()

    m, n = means.shape

    covs = np.ones(math.floor(n)) * 3
    loads = np.ones(math.floor(m)) * 4

    if not os.path.isfile(os.path.join(data_dir, "outputs", fname)):
        # Save the data used for this run
        np.save(os.path.join(data_dir, "outputs", "std_devs_dummy_%s_%s_%d_%d.npy" % (revs_or_paps, conf, num_dummies, seed)), std_devs)
        np.save(os.path.join(data_dir, "outputs", "means_dummy_%s_%s_%d_%d.npy" % (revs_or_paps, conf, num_dummies, seed)), means)

        # Run the max-min model
        # fractional_alloc_max_min = solve_max_min_project_each_step(tpms, covs, loads, error_bound)
        fractional_alloc_max_min = solve_max_min(means, covs, loads, std_devs,
                                                 noise_model=noise_model, dykstra=True, caching=False)

        np.save(os.path.join(data_dir, "outputs", "fractional_max_min_alloc_dummy_%s_%s_%d_%d.npy" % (revs_or_paps, conf, num_dummies, seed)), fractional_alloc_max_min)
        # alloc_max_min = best_of_n_bvn(fractional_alloc_max_min, tpms, error_bound, n=10)
        alloc_max_min = bvn(fractional_alloc_max_min)
        np.save(os.path.join(data_dir, "outputs", "max_min_alloc_dummy_%s_%s_%d_%d.npy" % (revs_or_paps, conf, num_dummies, seed)), alloc_max_min)

        # Run the baseline, which is just TPMS
        print("Solving for max USW using TPMS scores", flush=True)
        objective_score, alloc = solve_usw_gurobi(means, covs, loads)

        np.save(os.path.join(data_dir, "outputs", "tpms_alloc_dummy_%s_%s_%d_%d.npy" % (revs_or_paps, conf, num_dummies, seed)), alloc)

        true_obj = np.sum(alloc * true_scores)

        true_obj_max_min = np.sum(alloc_max_min * true_scores)

        print("Solving for max USW using true bids")
        opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

        if revs_or_paps == "paps":
            true_obj_on_dummies = np.sum(alloc[:, dummy_paps] * true_scores[:, dummy_paps])
            true_obj_max_min_on_dummies = np.sum(alloc_max_min[:, dummy_paps] * true_scores[:, dummy_paps])
            opt_obj_on_dummies = np.sum(opt_alloc[:, dummy_paps] * true_scores[:, dummy_paps])

        np.save("opt_alloc_dummy_%s_%d_%d.npy" % (revs_or_paps, num_dummies, seed), opt_alloc)

        # Check if any dummy revs were used
        dummies_used_in_opt = 0
        if revs_or_paps == "revs":
            dummies_used_in_opt = np.any(np.where(opt_alloc)[0] >= m - num_dummies)

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
        print("Stats for Dummy %s on %s. Num dummies is %d, seed is %d" % (revs_or_paps, conf, num_dummies, seed))
        print("Optimal USW: %.2f" % opt)
        stat_dict['opt_usw'] = opt
        print("Dummies used in opt? ", dummies_used_in_opt)
        stat_dict['dummies_used_in_opt'] = dummies_used_in_opt
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

        if revs_or_paps == "paps":
            # Compare the welfare on only the papers that had the noise added
            stat_dict['true_obj_dummies'] = true_obj_on_dummies
            stat_dict['true_obj_max_min_dummies'] = true_obj_max_min_on_dummies
            stat_dict['opt_obj_dummies'] = opt_obj_on_dummies

            print('true_obj_dummies ', true_obj_on_dummies)
            print('true_obj_max_min_on_dummies ', true_obj_max_min_on_dummies)
            print('opt_obj_on_dummies ', opt_obj_on_dummies)

        with open(os.path.join(data_dir, "outputs", fname), 'wb') as f:
            pickle.dump(stat_dict, f)
