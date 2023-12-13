import pickle
import math

from copy import deepcopy

from solve_gesw import solve_gesw_gurobi
from solve_usw import solve_usw_gurobi
from solve_max_min import solve_max_min, solve_max_min_alt, solve_max_min_gesw, get_worst_case_gesw_standalone, get_worst_case_usw_standalone
from utils import *
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--r_idx", type=int)

    return parser.parse_args()


def remap(group_labels):
    label_map = {}
    max_group_label = int(np.max(group_labels))
    sorted_bad = sorted((set(range(max_group_label)) - set(group_labels.tolist())))
    sorted_bad.append(max_group_label + 1)
    ctr = 0
    for idx, i in enumerate(sorted_bad):
        while idx + ctr < i:
            label_map[idx + ctr] = ctr
            ctr += 1

    final_clusters = deepcopy(group_labels)
    for idx in range(final_clusters.shape[0]):
        final_clusters[idx] = label_map[final_clusters[idx]]
    return final_clusters


def gesw(group_labels, alloc, scores):
    n_groups = np.max(group_labels)
    gusws = []
    for i in range(n_groups):
        idxs = np.where(group_labels == i)[0]
        sub_alloc = alloc[:, idxs]
        sub_scores = scores[:, idxs]
        gusws.append(np.sum(sub_alloc * sub_scores) / idxs.shape[0])
    return min(gusws)


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    year = args.year
    algo = args.algo
    r_idx = args.r_idx

    noise_model = "ellipse"

    fname = "stat_dict_iclr_trunc_beta_%d_%d_%s.pkl" % (year, r_idx, algo)
    # if not os.path.isfile(os.path.join(data_dir, "outputs", fname)):
    # Load in the ellipse
    # std_devs = np.load(os.path.join(data_dir, "data", "iclr", "scores_sigma_iclr_%d.npy" % year))
    # means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_%d.npy" % year))

    # std_devs = np.load(os.path.join(data_dir, "data", "iclr", "sigma_trunc_%d.npy" % year))
    # # means = np.load(os.path.join(data_dir, "data", "iclr", "mu_trunc_%d.npy" % year))
    # means = np.load(os.path.join(data_dir, "data", "iclr", "mu_hat_%d.npy" % year))
    group_labels = np.load(os.path.join(data_dir, "data", "iclr", "group_labels_trunc_%d.npy" % year))
    # r_values = np.load(os.path.join(data_dir, "data", "iclr", "r_vals_%d.npy" % year))

    std_devs = np.load(os.path.join(data_dir, "data", "iclr", "sigma_hat_beta_%d.npy" % year))
    means = np.load(os.path.join(data_dir, "data", "iclr", "mu_hat_beta_%d.npy" % year))
    r_values = np.load(os.path.join(data_dir, "data", "iclr", "r_vals_beta_%d.npy" % year))

    r = r_values[r_idx]
    delta_values = np.load(os.path.join(data_dir, "data", "iclr", "delta_vals.npy"))
    if r_idx < len(delta_values):
        delta = delta_values[r_idx]
    else:
        delta = 0.0

    print("delta = %.4f, r_idx = %d, r = %.4f" % (delta_values[r_idx], r_idx, r))

    # group_labels = np.load(os.path.join(data_dir, "data", "iclr", "group_ids_%d.npy" % year))

    # Take a subsample of the reviewers and papers
    m, n = means.shape
    # sample_frac = .05
    # np.random.seed(1)
    # sampled_revs = np.random.choice(range(m), math.floor(sample_frac * m))
    # sampled_paps = np.random.choice(range(n), math.floor(sample_frac * n))
    #
    # print(sampled_revs)
    # print(sampled_paps)
    #
    # group_labels = remap(group_labels[sampled_paps])
    # std_devs = std_devs[sampled_revs, :][:, sampled_paps]
    # means = means[sampled_revs, :][:, sampled_paps]
    # covs = np.ones(math.floor(sample_frac * n)) * 3
    # loads = np.ones(math.floor(sample_frac * m)) * 6

    # m, n = math.floor(sample_frac * m), math.floor(sample_frac * n)

    covs = np.ones(n) * 3
    loads = np.ones(m) * 6

    # Save the data used for this run
    run_name = uuid.uuid1()
    print("Run name: ", run_name)

    # np.save(os.path.join(data_dir, "outputs", "std_devs_iclr_%d_%d.npy" % (year, run_name)), std_devs)
    # np.save(os.path.join(data_dir, "outputs", "means_iclr_%d_%d.npy" % (year, run_name)), means)

    # Run the max-min model
    # fractional_alloc_max_min = solve_max_min_project_each_step(tpms, covs, loads, error_bound)
    # fractional_alloc_max_min = solve_max_min(means, covs, loads, std_devs, noise_model=noise_model)

    # np.save(os.path.join(data_dir, "outputs", "fractional_max_min_alloc_iclr_%d_%d.npy" % (year, seed)), fractional_alloc_max_min)
    # alloc_max_min = best_of_n_bvn(fractional_alloc_max_min, tpms, error_bound, n=10)
    # alloc_max_min = bvn(fractional_alloc_max_min)
    # np.save(os.path.join(data_dir, "outputs", "max_min_alloc_iclr_%d_%d.npy" % (year, seed)), alloc_max_min)

    # Run the baseline, which is just TPMS
    run_type = "iclr_trunc_beta"
    max_iter = 50
    st = time.time()
    fractional_alloc_max_min = None
    if algo == "LP":
        print("Solving for max USW", flush=True)
        est_usw, alloc = solve_usw_gurobi(means, covs, loads)
        np.save(os.path.join(data_dir, "outputs", "tpms_alloc_%s_%d.npy" % (run_type, year)), alloc)

    elif algo == "GESW":
        print("Solving for max GESW", flush=True)
        print(group_labels)
        est_usw, alloc = solve_gesw_gurobi(means, covs, loads, group_labels)
        np.save(os.path.join(data_dir, "outputs", "gesw_alloc_%s_%d.npy" % (run_type, year)), alloc)

    # elif algo == "RRA":
    #     print("Solving for max robust USW using Elitas RRA", flush=True)
    #     fractional_alloc_max_min = solve_max_min_alt(means, covs, loads, std_devs, integer=False, rsquared=None,
    #                                                  check=False)
    #     np.save(os.path.join(data_dir, "outputs", "rra_frac_alloc_%s_%d.npy" % (run_type, year)), fractional_alloc_max_min)
    #     alloc = bvn(fractional_alloc_max_min, run_name)
    #     est_usw = np.sum(alloc * means)
    #     np.save(os.path.join(data_dir, "outputs", "rra_alloc_%s_%d.npy" % (run_type, year)), alloc)
    elif algo == "RRA":
        print("Solving for max robust USW using Elitas RRA in CVXPY", flush=True)
        fractional_alloc_max_min = solve_max_min_alt(means, covs, loads, std_devs, r)
        np.save(os.path.join(data_dir, "outputs", "rra_frac_alloc_%s_%d.npy" % (run_type, year)), fractional_alloc_max_min)
        alloc = bvn(fractional_alloc_max_min, run_name)
        est_usw = np.sum(alloc * means)
        np.save(os.path.join(data_dir, "outputs", "rra_alloc_%s_%d.npy" % (run_type, year)), alloc)

    elif algo == "RRA_ORIG":
        print("Solving for max robust USW using the original RRA formulation with r=%d" % r, flush=True)
        fractional_alloc_max_min = solve_max_min(means, covs, loads, std_devs, noise_model=noise_model, caching=True,
                                                 dykstra=True, run_name=run_name, r=r, max_iter=max_iter)
        np.save(os.path.join(data_dir, "outputs", "rra_orig_frac_alloc_%s_%d_%d.npy" % (run_type, r_idx, year)), fractional_alloc_max_min)
        alloc = bvn(fractional_alloc_max_min, run_name)
        est_usw = np.sum(alloc * means)
        np.save(os.path.join(data_dir, "outputs", "rra_orig_alloc_%s_%d_%d.npy" % (run_type, r_idx, year)), alloc)

    elif algo == "RRA_GESW":
        print("Solving for max robust GESW using the original RRA formulation with r=%d" % r, flush=True)
        fractional_alloc_max_min = solve_max_min_gesw(means, covs, loads, std_devs, group_labels,
                                                      noise_model=noise_model,
                                                      dykstra=True, run_name=run_name, r=r, max_iter=max_iter*10)
        np.save(os.path.join(data_dir, "outputs", "rra_gesw_frac_alloc_%s_%d_%d.npy" % (run_type, r_idx, year)), fractional_alloc_max_min)
        alloc = bvn(fractional_alloc_max_min, run_name)
        est_usw = np.sum(alloc * means)
        np.save(os.path.join(data_dir, "outputs", "rra_gesw_alloc_%s_%d_%d.npy" % (run_type, r_idx, year)), alloc)

    print("Solver took %s secs" % (time.time() - st))

    est_gesw = gesw(group_labels, alloc, means)

    # print("Solving for max USW using true bids")
    # opt, opt_alloc = solve_usw_gurobi(true_scores, covs, loads)

    # np.save("opt_alloc_%s_%d_%.1f.npy" % (dset_name, seed, alpha), opt_alloc)

    print(loads)
    # print(np.sum(alloc_max_min, axis=1))
    print(covs)
    # print(np.sum(alloc_max_min, axis=0))
    # print(np.all(np.sum(alloc_max_min, axis=1) <= loads))
    # print(np.all(np.sum(alloc_max_min, axis=0) == covs))

    # worst_s = get_worst_case(alloc, means, std_devs, noise_model=noise_model)
    # worst_case_obj_tpms = np.sum(worst_s * alloc)

    # worst_s = get_worst_case(alloc_max_min, means, std_devs, noise_model=noise_model)
    # worst_case_obj_max_min = np.sum(worst_s * alloc_max_min)

    true_usws = []
    true_gesws = []

    # with open(os.path.join(data_dir, "data", "iclr", "test_set_%d.pkl" % year), 'rb') as f:
    #     test_set = pickle.load(f)
    with open(os.path.join(data_dir, "data", "iclr", "test_set_beta_%d.pkl" % year), 'rb') as f:
        test_set = pickle.load(f)

    for true_scores in test_set:
        true_usws.append(np.sum(alloc * true_scores) / n)
        true_gesws.append(gesw(group_labels, alloc, true_scores))

    # Compute the worst case USW and GESW at all r values for this allocation
    worst_case_usws = []
    worst_case_gesws = []
    for alt_r_value in r_values:
        print("getting worst case for r=%.4f" % alt_r_value, flush=True)
        worst_case_usws.append(get_worst_case_usw_standalone(alloc, means, std_devs, alt_r_value))
        worst_case_gesws.append(get_worst_case_gesw_standalone(alloc, means, std_devs, alt_r_value, group_labels))

    worst_case_usws_frac = []
    worst_case_gesws_frac = []
    if fractional_alloc_max_min is not None:
        for alt_r_value in r_values:
            print("getting worst case for frac alloc for r=%.4f" % alt_r_value, flush=True)
            worst_case_usws_frac.append(get_worst_case_usw_standalone(fractional_alloc_max_min, means, std_devs, alt_r_value))
            worst_case_gesws_frac.append(get_worst_case_gesw_standalone(fractional_alloc_max_min, means, std_devs, alt_r_value, group_labels))

    stat_dict = {'true_usws_full': true_usws, 'true_gesws_full': true_gesws,
                 'worst_usws_full': worst_case_usws, 'worst_gesws_full': worst_case_gesws,
                 'worst_usws_frac_full': worst_case_usws_frac, 'worst_gesws_frac_full': worst_case_gesws_frac}
    print("\n*******************\n*******************\n*******************\n")
    print("Stats for ICLR %d with algo %s" % (year, algo))
    # print("Optimal USW: %.2f" % opt)
    # stat_dict['opt_usw'] = opt
    print("\n")
    print("Estimated USW: %.2f" % (est_usw / n))
    stat_dict['est_usw'] = est_usw / n
    print("True USW: %.2f \\pm %.2f" % (np.mean(true_usws), np.std(true_usws)))
    stat_dict['true_usw'] = np.mean(true_usws)

    print("Estimated GESW: %.2f" % est_gesw)
    stat_dict['est_gesw'] = est_gesw
    print("True GESW: %.2f \\pm %.2f" % (np.mean(true_gesws), np.std(true_gesws)))
    stat_dict['true_gesw'] = np.mean(true_gesws)

    print("\n")

    with open(os.path.join(data_dir, "outputs", fname), 'wb') as f:
        pickle.dump(stat_dict, f)

