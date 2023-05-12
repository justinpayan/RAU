import pickle
import math

from baselines import *
from utils import *

import time

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--year", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    year = args.year
    seed = args.seed

    sample_size = .6

    noise_model = "ellipse"

    np.random.seed(seed)

    fname = "stat_dict_iclr_baselines_%d_%d.pkl" % (year, seed)
    if not os.path.isfile(os.path.join(data_dir, "outputs", fname)):
        # Load in the ellipse
        std_devs = np.load(os.path.join(data_dir, "data", "iclr", "scores_sigma_iclr_%d.npy" % year))
        means = np.load(os.path.join(data_dir, "data", "iclr", "scores_mu_iclr_%d.npy" % year))

        # Take a subsample of the reviewers and papers
        m, n = means.shape
        sampled_revs = np.random.choice(range(m), math.floor(sample_size*m))
        sampled_paps = np.random.choice(range(n), math.floor(sample_size*n))

        std_devs = std_devs[sampled_revs, :][:, sampled_paps]
        means = means[sampled_revs, :][:, sampled_paps]

        covs = np.ones(math.floor(sample_size*n)) * 3
        loads = np.ones(math.floor(sample_size*m)) * 6

        # Run the baselines -- PR4A, FairFlow, and FairSeq
        print("Solving FairSeq", flush=True)
        fairseq_alloc = fairseq(means, covs, loads)

        np.save(os.path.join(data_dir, "outputs", "fairseq_alloc_iclr_%d_%d.npy" % (year, seed)), fairseq_alloc)

        print("Solving PR4A", flush=True)
        pr4a_alloc = pr4a(means, covs, loads)

        np.save(os.path.join(data_dir, "outputs", "pr4a_alloc_iclr_%d_%d.npy" % (year, seed)), pr4a_alloc)

        print("Solving FairFlow", flush=True)
        fairflow_alloc = fairflow(means, covs, loads)

        np.save(os.path.join(data_dir, "outputs", "fairflow_alloc_iclr_%d_%d.npy" % (year, seed)), fairflow_alloc)

        worst_s = get_worst_case(fairseq_alloc, means, std_devs, noise_model=noise_model)
        worst_case_obj_fairseq = np.sum(worst_s * fairseq_alloc)

        worst_s = get_worst_case(pr4a_alloc, means, std_devs, noise_model=noise_model)
        worst_case_obj_pr4a = np.sum(worst_s * pr4a_alloc)

        worst_s = get_worst_case(fairflow_alloc, means, std_devs, noise_model=noise_model)
        worst_case_obj_fairflow = np.sum(worst_s * fairflow_alloc)

        stat_dict = {}
        print("\n*******************\n*******************\n*******************\n")
        print("Stats for ICLR %d with seed %d" % (year, seed))
        print("\n")

        print("Worst case USW from FairSeq: %.2f" % worst_case_obj_fairseq)
        stat_dict['worst_usw_fairseq'] = worst_case_obj_fairseq
        print("\n")
        print("Worst case USW from PR4A: %.2f" % worst_case_obj_pr4a)
        stat_dict['worst_usw_pr4a'] = worst_case_obj_pr4a
        print("\n")
        print("Worst case USW from FairFlow: %.2f" % worst_case_obj_fairflow)
        stat_dict['worst_usw_fairflow'] = worst_case_obj_fairflow
        print("\n")

        with open(os.path.join(data_dir, "outputs", fname), 'wb') as f:
            pickle.dump(stat_dict, f)
