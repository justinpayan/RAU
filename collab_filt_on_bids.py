import pickle
import math

import numpy as np

from solve_usw import solve_usw_gurobi
from solve_max_min import solve_max_min
from bid_reconstruction import reconstruct_bids
from utils import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--decay_factor", type=int)
    parser.add_argument("--year", type=int, default=2020)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    decay_factor = args.decay_factor
    seed = args.seed
    year = args.year

    noise_model = "ellipse"

    estimator_fname = "bid_est_%d_%d_%d" % (year, decay_factor, seed)
    prob_dist_fname = "prob_dist_%d_%d_%d" % (year, decay_factor, seed)
    error_size_fname = "error_size_%d_%d_%d" % (year, decay_factor, seed)
    if not os.path.isfile(os.path.join(data_dir, "outputs", estimator_fname + ".npy")):
        # load in the sampled bids
        bid_mask_fname = os.path.join(data_dir, "data",
                                      "bid_reconstruction", "bid_mask_%d_%d_%d.npy" % (year, decay_factor, seed))
        true_bids_fname = os.path.join(data_dir, "data",
                                       "bid_reconstruction", "true_bids_%d_%d_%d.npy" % (year, decay_factor, seed))
        bid_mask = np.load(bid_mask_fname)
        true_bids = np.load(true_bids_fname)

        sampled_bids = bid_mask * true_bids
        sampled_bids[1-bid_mask] = np.nan

        # We also need to load in the keyword cosine sim scores. This is our most basic estimator
        fname = os.path.join(data_dir, "data", "bid_reconstruction", "keyword_cosine_sims_%d.npy" % year)
        keyword_based_cos_sim_estimator = np.load(fname)

        # We construct another estimator using the co-occurrence matrix of the keyword clusters,
        # and then we take each similarity between rev and pap to be the mixture of the similarities of the
        # clusters represented in each.
        fname = os.path.join(data_dir, "data", "bid_reconstruction", "rev_mixtures_%d.npy" % year)
        rev_mixtures = np.load(fname)
        fname = os.path.join(data_dir, "data", "bid_reconstruction", "pap_mixtures_%d.npy" % year)
        pap_mixtures = np.load(fname)
        fname = os.path.join(data_dir, "data", "bid_reconstruction", "keyword_cluster_correlations_%d.npy" % year)
        keyword_cluster_correlations = np.load(fname)

        # Construct the estimator
        keyword_cluster_correlation_estimator = np.dot(rev_mixtures, keyword_cluster_correlations)
        keyword_cluster_correlation_estimator = np.dot(keyword_cluster_correlation_estimator, pap_mixtures.transpose())

        # TODO: Now we need to basically construct the learned estimators and optimize them. I think this
        # TODO: should be done inside a function, where we can pass the fixed estimators and any auxiliary info.
        reconstructed_bid_matrix, empirical_error, generalization_error = reconstruct_bids(
            sampled_bids,
            [keyword_based_cos_sim_estimator,
             keyword_cluster_correlation_estimator],
            [(rev_mixtures, pap_mixtures)]
        )
