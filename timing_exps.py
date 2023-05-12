import pickle
import math
import time

from solve_usw import solve_usw_gurobi
from solve_max_min import solve_max_min
from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument('--caching', action='store_true')
    parser.add_argument('--no-caching', dest='caching', action='store_false')
    parser.set_defaults(caching=True)
    parser.add_argument('--dykstra', action='store_true')
    parser.add_argument('--no-dykstra', dest='dykstra', action='store_false')
    parser.set_defaults(dykstra=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    year = args.year
    seed = args.seed
    caching = args.caching
    dykstra = args.dykstra

    noise_model = "ellipse"

    print("Running timing test with caching %s and dykstra %s" % (caching, dykstra))
    print("Year: %d\n Seed: %d" % (year, seed))

    fname = "stat_dict_iclr_%d_%d_%d_%d.pkl" % (year, seed, caching, dykstra)
    if not os.path.isfile(os.path.join(data_dir, "outputs", fname)):
        # Load in the ellipse
        std_devs = np.load(os.path.join(data_dir, "data", "iclr", "scores_sigma_iclr_%d.npy" % year))
        means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_%d.npy" % year))

        # Take a subsample of the reviewers and papers
        m, n = means.shape
        sampled_revs = np.random.choice(range(m), math.floor(.9*m))
        sampled_paps = np.random.choice(range(n), math.floor(.9*n))

        std_devs = std_devs[sampled_revs, :][:, sampled_paps]
        means = means[sampled_revs, :][:, sampled_paps]

        covs = np.ones(math.floor(.9*n)) * 3
        loads = np.ones(math.floor(.9*m)) * 6

        # Run the max-min model
        # fractional_alloc_max_min = solve_max_min_project_each_step(tpms, covs, loads, error_bound)
        fractional_alloc_max_min, adv_times, proj_times = solve_max_min(means, covs, loads, std_devs, caching=caching, dykstra=dykstra, noise_model=noise_model)

        np.save(os.path.join(data_dir, "outputs", "fractional_max_min_alloc_iclr_%d_%d_%d_%d.npy" % (year, seed, caching, dykstra)), fractional_alloc_max_min)
        # alloc_max_min = best_of_n_bvn(fractional_alloc_max_min, tpms, error_bound, n=10)
        st = time.time()
        alloc_max_min = bvn(fractional_alloc_max_min)
        bvn_time = time.time() - st
        np.save(os.path.join(data_dir, "outputs", "max_min_alloc_iclr_%d_%d_%d_%d.npy" % (year, seed, caching, dykstra)), alloc_max_min)

        print(loads)
        print(np.sum(alloc_max_min, axis=1))
        print(covs)
        print(np.sum(alloc_max_min, axis=0))
        print(np.all(np.sum(alloc_max_min, axis=1) <= loads))
        print(np.all(np.sum(alloc_max_min, axis=0) == covs))

        st = time.time()
        worst_s = get_worst_case(alloc_max_min, means, std_devs, noise_model=noise_model)
        worst_case_obj_max_min = np.sum(worst_s * alloc_max_min)
        final_adv_time = time.time() - st

        stat_dict = {}
        print("\n*******************\n*******************\n*******************\n")
        print("Stats for ICLR %d with seed %d" % (year, seed))
        print("\n")
        print("Worst case USW from max_min optimizer: %.2f" % worst_case_obj_max_min)
        print("Timing info adv_times: %s\nproj_times: %s\nbvn_time: %s\nfinal_adv_time: %s" %
              (adv_times, proj_times, bvn_time, final_adv_time))
        stat_dict['worst_usw_maxmin'] = worst_case_obj_max_min
        stat_dict['adv_times'] = adv_times
        stat_dict['proj_times'] = proj_times
        stat_dict['bvn_time'] = bvn_time
        stat_dict['final_adv_time'] = final_adv_time

        with open(os.path.join(data_dir, "outputs", fname), 'wb') as f:
            pickle.dump(stat_dict, f)
