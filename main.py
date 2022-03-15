
from experiment_framework import run_experiment
from query_models import *
from solve_esw import solve_esw
from solve_usw import solve_usw
from utils import *


def basic_baselines():
    dset_name = "cvpr"
    solver = solve_usw
    obj = "USW"

    tpms, true_bids, covs, loads = load_dset(dset_name)
    print("Solving for max E[%s] using TPMS scores" % obj)
    expected_obj, alloc = solver(tpms, covs, loads)

    true_obj = 0
    if obj == "USW":
        true_obj = np.sum(alloc * true_bids)
    elif obj == "ESW":
        true_obj = np.min(np.sum(alloc * true_bids, axis=0))

    print("Solving for max %s using true bids" % obj)
    opt, _ = solver(true_bids, covs, loads)

    print("\n*******************\n*******************\n*******************\n")
    print("Stats for ", dset_name)
    print("E[%s] from using TPMS scores: %.2f" % (obj, expected_obj))
    print("True %s from using TPMS scores: %.2f" % (obj, true_obj))
    print("Optimal %s: %.2f" % (obj, opt))


def query_model():
    dset_name = "midl"
    tpms, true_bids, covs, loads = load_dset(dset_name)
    solver = solve_usw
    obj = "USW"
    seed = 31415
    lamb = 5
    query_model = GreedyMaxQueryModel(tpms, covs, loads, solver, dset_name)
    # query_model = VarianceReductionQueryModel(tpms, covs, loads, solver, dset_name)
    # query_model = SuperStarQueryModel(tpms, dset_name)
    # query_model = RandomQueryModel(tpms)
    expected_obj, alloc, total_bids = run_experiment(dset_name, query_model, solver, seed, lamb)

    true_obj = np.sum(alloc * true_bids)

    print("Number of reviewers: %d" % loads.shape[0])
    print("Number of bids issued: %d" % total_bids)
    print("E[%s] from using this query model: %.2f" % (obj, expected_obj))
    print("True %s from using TPMS scores: %.2f" % (obj, true_obj))


if __name__ == "__main__":
    query_model()


