
import numpy as np
from solve_usw import solve_usw_gurobi
import cvxpy as cp


def get_worst_case(alloc, tpms, error_bound):
    s = cp.Variable(tpms.shape[0]*tpms.shape[1])

    soc_constraint = [cp.SOC(error_bound, s - tpms.ravel())]
    prob = cp.Problem(cp.Minimize(alloc.ravel().T @ s),
                      soc_constraint + [s >= np.zeros(s.shape), s <= np.ones(s.shape)])
    prob.solve()

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
# out the maximin assignment where we worst-case over the true scores within "error_bound" of
# the tpms scores.
def solve_min_max(tpms, covs, loads, error_bound):
    _, alloc = solve_usw_gurobi(tpms, covs, loads)

    print("Solving min max")

    t = 0
    converged = False
    max_iter = 50

    while not converged and t < max_iter:
        rate = 1/(1+t)

        # Compute the worst-case S matrix using second order cone programming
        worst_s = get_worst_case(alloc, tpms, error_bound)

        diff = np.sqrt(np.sum((worst_s - tpms)**2))
        assert diff-1e-5 <= error_bound

        # Update the allocation
        # 1, compute the gradient (I think it's just the value of the worst s, but check to be sure).
        # 2, update using the rate parameter times the gradient.
        # 3, project to the set of allocations that meet all the hard constraints

        old_alloc = alloc.copy()

        alloc_grad = worst_s

        alloc = alloc + rate * alloc_grad

        # Project to the set of feasible allocations
        alloc = project_to_feasible(alloc, covs, loads)

        # Check for convergence, update t
        update_amt = np.linalg.norm(alloc - old_alloc)
        converged = np.isclose(0, update_amt, atol=1e-6)
        t += 1

        if t % 1 == 0:
            print("Step %d" % t)
            print("Obj value: ", np.sum(old_alloc*worst_s))

    return alloc

