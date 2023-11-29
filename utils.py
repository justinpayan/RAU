import numpy as np
import os
import uuid

from solve_max_min import get_worst_case

from solve_usw import solve_usw_gurobi


def load_dset(dname, seed, data_dir=".", noise_model="ball", alpha=0.5):
    assert noise_model in ["ball", "ellipse"]

    tpms = np.load(os.path.join(data_dir, "data", dname, "scores.npy"))
    covs = np.load(os.path.join(data_dir, "data", dname, "covs.npy"))
    loads = np.load(os.path.join(data_dir, "data", dname, "loads.npy"))

    rng = np.random.default_rng(seed)

    tpms = np.clip(tpms, 0, np.inf)
    tpms /= np.max(tpms)

    # Sample the "true" scores.
    # For now, assume that 10% of papers are underestimated by .2 and 10% are overestimated by .2
    # We will do a collaborative filtering approach later
    # true_scores = tpms.copy()
    # n = true_scores.shape[1]
    # underest_papers = rng.permutation(n)[:math.ceil(.2*n)]
    # overest_papers = rng.permutation(n)[:math.ceil(.2*n)]
    # true_scores[:, underest_papers] += .05
    # true_scores[:, overest_papers] -= .05
    # true_scores = np.clip(true_scores, 0, np.inf)
    # true_scores /= np.max(true_scores)

    if noise_model == "ball":
        # One way to sample true scores would be to assume papers are either liked or disliked
        noisy_tpms = tpms + rng.normal(-0.05, 0.05, tpms.shape)
        noisy_tpms = np.clip(noisy_tpms, 0, 1)
        true_scores = noisy_tpms
        # true_scores = rng.uniform(0, 1, size=tpms.shape) < noisy_tpms

        error_distrib = np.sqrt(np.sum((tpms - true_scores) ** 2)) * 1.0
        print("Error estimate is: ", error_distrib)

        return tpms, true_scores, covs, loads, error_distrib, None
    elif noise_model == "ellipse":
        # Let's assume the noise is the same, but we just know more about it.
        # Maybe this will need to change later.
        _, alloc = solve_usw_gurobi(tpms, covs, loads)
        # noise = rng.normal(-.2, 0.05, tpms.shape)
        # noise[alloc < 0.5] = 0
        # noisy_tpms = tpms + noise

        num_assts = np.where(alloc > 0.5)[0].shape[0]
        num_nonassts = np.where(alloc < 0.5)[0].shape[0]
        num_errors_on_non_tpms = num_assts*alpha
        num_errors_on_tpms = num_assts*(1-alpha)

        error_distrib = np.zeros(tpms.shape)
        error_distrib[alloc > 0.5] = rng.uniform(size=num_assts)
        error_distrib[alloc > 0.5] /= error_distrib[alloc > 0.5].sum()
        error_distrib[alloc > 0.5] *= num_errors_on_tpms

        error_distrib[alloc < 0.5] = rng.uniform(size=num_nonassts)
        error_distrib[alloc < 0.5] /= error_distrib[alloc < 0.5].sum()
        error_distrib[alloc < 0.5] *= num_errors_on_non_tpms

        # Ensure that the L2 norm of u is = u_mag
        # And that the portion of u on the TPMS assignments is 1-alpha.
        u_mag = 50

        u = rng.uniform(size=tpms.shape)
        u[alloc > 0.5] *= (1 - alpha) / u[alloc > 0.5].sum()
        u[alloc < 0.5] *= alpha / u[alloc < 0.5].sum()

        u *= u_mag / np.sqrt(np.sum(u**2))

        print("L2 norm of u should be %.2f, it is %.2f" % (u_mag, np.sqrt(np.sum(u**2))))
        print("Concentration of u on TPMS assignments should be %.2f, it is %.2f" %
              (1-alpha, np.sum(u[alloc > 0.5])/np.sum(u)))
        print("Sum of error_distrib on TPMS assignments should be %.2f, it is %.2f" %
              (num_errors_on_tpms, error_distrib[alloc > 0.5].sum()))
        print("Sum of error_distrib on non-TPMS assignments should be %.2f, it is %.2f" %
              (num_errors_on_non_tpms, error_distrib[alloc < 0.5].sum()))

        noisy_tpms = tpms.copy()
        noisy_tpms = noisy_tpms - error_distrib * u
        # noisy_tpms = tpms + rng.normal(-0.05, 0.05, tpms.shape)
        noisy_tpms = np.clip(noisy_tpms, 0, 1)
        true_scores = noisy_tpms

        print("Error estimate is: ", error_distrib)

        return tpms, true_scores, covs, loads, error_distrib, u_mag


# Implements https://ieeexplore.ieee.org/abstract/document/1181955
# Use bvn.cpp from https://github.com/theryanl/mitigating_manipulation_via_randomized_reviewer_assignment/blob/master/core/bvn.cpp
def bvn(fractional_alloc, run_name):
    # # While there are fractional edges, find a simple cycle or maximal path
    # while np.any((0 < fractional_alloc) * (fractional_alloc < 1)):
    #     # Find a simple cycle or maximal path
    #     path = find_path(fractional_alloc)
    tmp_fname = str(run_name) + ".txt"
    with open(tmp_fname, 'w') as f:
        m, n = fractional_alloc.shape
        f.write("%d %d\n" % (m, n))
        f.write("1\n"*m)
        asst_str = ""
        for r in range(m):
            for p in range(n):
                assn = 0
                if not np.isclose(fractional_alloc[r, p], 0):
                    assn = fractional_alloc[r, p]
                asst_str += "%d %d %.6f\n" % (r, p+m, np.abs(assn))
        f.write(asst_str[:-1])

    os.system("/mnt/nfs/scratch1/jpayan/MinimalBidding/a.out < %s > output_%s" % (tmp_fname, tmp_fname))

    rounded_alloc = np.zeros(fractional_alloc.shape)
    with open("output_%s" % tmp_fname, 'r') as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            r, p = line.strip().split()
            r = int(r)
            p = int(p) - m
            rounded_alloc[r, p] = 1
    return rounded_alloc


# Run bvn n times. Check the worst-case objective value for each run, and take the best.
def best_of_n_bvn(fractional_alloc, tpms, error_bound, n=10):
    print("Sampling %d allocations" % n)
    best_alloc = None
    best_worst_case = -np.inf
    for i in range(n):
        rounded_alloc = bvn(fractional_alloc)
        worst_s = get_worst_case(rounded_alloc, tpms, error_bound)
        worst_case_obj = np.sum(worst_s * rounded_alloc)
        print("Worst case obj value is %.2f" % worst_case_obj)
        if worst_case_obj > best_worst_case:
            best_alloc = rounded_alloc
            best_worst_case = worst_case_obj
    print("Found obj value %.2f" % best_worst_case)
    return best_alloc

