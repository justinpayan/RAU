
import numpy as np
from solve_usw import solve_usw_gurobi
import cvxpy as cp
import time

from utils import bvn


def get_worst_case(alloc, tpms, error_bound):
    s = cp.Variable(tpms.shape[0]*tpms.shape[1])

    soc_constraint = [cp.SOC(error_bound, s - tpms.ravel())]
    prob = cp.Problem(cp.Minimize(alloc.ravel().T @ s),
                      soc_constraint + [s >= np.zeros(s.shape), s <= np.ones(s.shape)])
    prob.solve(solver='SCS')

    return s.value.reshape(tpms.shape)


def project_to_feasible(alloc, covs, loads):
    # The allocation probably violates the coverage and reviewer load bounds.
    # Find the allocation with the smallest L2 distance from the current one such
    # that the constraints are satisfied
    x = cp.Variable(shape=alloc.shape)
    m, n = alloc.shape
    cost = cp.sum_squares(x - alloc)
    n_vec = np.ones((n, 1))
    m_vec = np.ones((m, 1))
    constraints = [x @ n_vec <= loads.reshape((m, 1)),
                   x.T @ m_vec == covs.reshape((n, 1)),
                   x >= np.zeros(x.shape),
                   x <= np.ones(x.shape)]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    return x.value


# Consider the tpms matrix as the center point, and then we assume
# that the L2 error is not more than "error_bound". We can then run subgradient ascent to figure
# out the maximin assignment where we worst-case over expected value over sampling the
# discrete assignments and then computing the worst case true scores within "error_bound" of
# the tpms scores.
def solve_max_expected_min(tpms, covs, loads, error_bound):
    st = time.time()
    print("Solving for initial max USW alloc")
    _, alloc = solve_usw_gurobi(tpms, covs, loads)

    print("Solving min max: %s elapsed" % (time.time() - st))

    t = 0
    converged = False
    max_iter = 20
    num_samples = 5

    while not converged and t < max_iter:
        rate = 1/(t+1)

        # Take samples of the possible discrete assignments

        alloc_samples = []
        if t == 0:
            alloc_samples.append(alloc)
        else:
            for _ in range(num_samples):
                alloc_samples.append(bvn(alloc))

        grads = []
        worst_s_values = []
        for sample in alloc_samples:
            # Compute the worst-case S matrix using second order cone programming
            print("Computing worst case S matrix")
            print("%s elapsed" % (time.time() - st))
            worst_s = get_worst_case(sample, tpms, error_bound)
            worst_s_values.append(worst_s)

            diff = np.sqrt(np.sum((worst_s - tpms)**2))
            assert diff-1e-2 <= error_bound

            # Update the allocation
            # 1, compute the gradient (I think it's just the value of the worst s, but check to be sure).
            # 2, update using the rate parameter times the gradient.
            # 3, project to the set of allocations that meet all the hard constraints

            grads.append(worst_s)

        grads = np.array(grads)
        alloc_grad = np.mean(grads, axis=0)
        old_alloc = alloc.copy()
        alloc = alloc + rate * alloc_grad

        # Project to the set of feasible allocations
        print("Projecting to feasible set: %s elapsed" % (time.time() - st))
        alloc = project_to_feasible(alloc, covs, loads)

        # Check for convergence, update t
        update_amt = np.linalg.norm(alloc - old_alloc)
        converged = np.isclose(0, update_amt, atol=1e-6)
        t += 1

        if t % 1 == 0:
            print("Step %d" % t)
            print("Est. obj value from previous step: ", np.mean([np.sum(a*s) for a, s in zip(alloc_samples, worst_s_values)]))
            print("%s elapsed" % (time.time() - st))

    return alloc
