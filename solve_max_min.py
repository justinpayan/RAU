import numpy as np
import os
import cvxpy as cp
import time


# Implements https://ieeexplore.ieee.org/abstract/document/1181955
# Use bvn.cpp from https://github.com/theryanl/mitigating_manipulation_via_randomized_reviewer_assignment/blob/master/core/bvn.cpp
def bvn(fractional_alloc):
    # # While there are fractional edges, find a simple cycle or maximal path
    # while np.any((0 < fractional_alloc) * (fractional_alloc < 1)):
    #     # Find a simple cycle or maximal path
    #     path = find_path(fractional_alloc)
    with open("fractional_alloc.txt", 'w') as f:
        m, n = fractional_alloc.shape
        f.write("%d %d\n" % (m, n))
        f.write("1\n" * m)
        asst_str = ""
        for r in range(m):
            for p in range(n):
                assn = 0
                if not np.isclose(fractional_alloc[r, p], 0):
                    assn = fractional_alloc[r, p]
                asst_str += "%d %d %.6f\n" % (r, p + m, np.abs(assn))
        f.write(asst_str[:-1])

    os.system("/mnt/nfs/scratch1/jpayan/MinimalBidding/a.out < fractional_alloc.txt > output_bvn.txt")

    rounded_alloc = np.zeros(fractional_alloc.shape)
    with open("output_bvn.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            r, p = line.strip().split()
            r = int(r)
            p = int(p) - m
            rounded_alloc[r, p] = 1
    return rounded_alloc


def get_worst_case(alloc, tpms, error_distrib, u_mag, noise_model="ball"):
    if noise_model == "ball":
        s = cp.Variable(tpms.shape[0] * tpms.shape[1])

        soc_constraint = [cp.SOC(u_mag, s - tpms.ravel())]
        prob = cp.Problem(cp.Minimize(alloc.ravel().T @ s),
                          soc_constraint + [s >= np.zeros(s.shape), s <= np.ones(s.shape)])
        prob.solve(solver='SCS')

        return s.value.reshape(tpms.shape)
    elif noise_model == "ellipse":
        u = cp.Variable(tpms.shape[0] * tpms.shape[1])

        soc_constraint = [cp.SOC(u_mag, u)]
        prob = cp.Problem(cp.Minimize(alloc.ravel().T @ (tpms.ravel() + cp.multiply(error_distrib.ravel(), u))),
                          soc_constraint + [(tpms.ravel() + cp.multiply(error_distrib.ravel(), u)) >= np.zeros(u.shape),
                                            (tpms.ravel() + cp.multiply(error_distrib.ravel(), u)) <= np.ones(u.shape)])
        prob.solve(solver='SCS')

        return tpms + error_distrib * u.value.reshape(tpms.shape)


# def project_to_feasible(alloc, covs, loads, use_verbose=False):
#     # The allocation probably violates the coverage and reviewer load bounds.
#     # Find the allocation with the smallest L2 distance from the current one such
#     # that the constraints are satisfied
#     x = cp.Variable(shape=alloc.shape)
#     m, n = alloc.shape
#     cost = cp.sum_squares(x - alloc)
#     n_vec = np.ones((n, 1))
#     m_vec = np.ones((m, 1))
#     constraints = [x @ n_vec <= loads.reshape((m, 1)),
#                    x.T @ m_vec == covs.reshape((n, 1)),
#                    x >= np.zeros(x.shape),
#                    x <= np.ones(x.shape)]
#     prob = cp.Problem(cp.Minimize(cost), constraints)
#     prob.solve(verbose=use_verbose)
#     return x.value

def project_to_feasible(alloc, covs, loads, use_verbose=False):
    # The allocation probably violates the coverage and reviewer load bounds.
    # Find the allocation with the smallest L2 distance from the current one such
    # that the constraints are satisfied

    # This version uses Dykstra's projection algorithm to repeatedly project onto
    # each constraint and converge to the point in all constraint sets with smallest
    # Euclidean distance.
    m, n = alloc.shape

    z_lb = np.zeros(alloc.shape)
    z_ub = np.zeros(alloc.shape)
    z_pap_cov = np.zeros(alloc.shape)
    z_rev_load = np.zeros(alloc.shape)

    u = alloc.copy()

    converged = False

    while not converged:
        # Project to each constraint
        # LB
        new_u = (u + z_lb).copy()
        proj_new_u = np.clip(new_u, 0, None)
        z_lb = new_u - proj_new_u
        u = proj_new_u

        print("Violation of LB")
        print(np.abs(np.sum(u[u < 0])))

        # UB
        new_u = (u + z_ub).copy()
        proj_new_u = np.clip(new_u, None, 1)
        z_ub = new_u - proj_new_u
        u = proj_new_u

        print("Violation of UB")
        print(np.sum(np.clip(u - np.ones(u.shape), 0, None)))

        # Paper coverage
        new_u = (u + z_pap_cov).copy()
        true_cov = np.sum(alloc, axis=0)
        proj_new_u = u - ((true_cov - covs) / m).reshape((1, -1))
        z_pap_cov = new_u - proj_new_u
        u = proj_new_u

        print("Violation of paper cov")
        print(np.sum(np.abs(np.sum(u, axis=0) - covs)))

        # Reviewer load bounds
        new_u = (u + z_rev_load).copy()
        true_load = np.sum(alloc, axis=1)
        proj_new_u = u - (np.clip(true_load - loads, 0, None) / n).reshape((-1, 1))
        z_rev_load = new_u - proj_new_u
        u = proj_new_u

        print("Violation of review loads")
        print(np.sum(np.clip(np.sum(u, axis=1) - loads, 0, None)), flush=True)

        # Let's say we've converged when none of the constraints is too far violated.
        print("Violation of LB")
        print(np.abs(np.sum(u[u < 0])))
        print("Violation of UB")
        print(np.sum(np.clip(u - np.ones(u.shape), 0, None)))
        print("Violation of paper cov")
        print(np.sum(np.abs(np.sum(u, axis=0) - covs)))
        print("Violation of review loads")
        print(np.sum(np.clip(np.sum(u, axis=1) - loads, 0, None)), flush=True)
        if np.abs(np.sum(u[u < 0])) < .1 and \
                np.sum(np.clip(u - np.ones(u.shape), 0, None)) < .1 and \
                np.sum(np.abs(np.sum(u, axis=0) - covs)) < .1 and \
                np.sum(np.clip(np.sum(u, axis=1) - loads, 0, None)) < .1:
            converged = True

    return u


def project_to_integer(alloc, covs, loads, use_verbose=False):
    # The allocation is likely not integral.
    # Find the integer allocation with the smallest L2 distance from the current one
    x = cp.Variable(shape=alloc.shape, integer=True)
    m, n = alloc.shape
    cost = cp.sum_squares(x - alloc)
    n_vec = np.ones((n, 1))
    m_vec = np.ones((m, 1))
    constraints = [x @ n_vec <= loads.reshape((m, 1)),
                   x.T @ m_vec == covs.reshape((n, 1)),
                   x >= np.zeros(x.shape),
                   x <= np.ones(x.shape)]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=use_verbose)
    return x.value


# Consider the tpms matrix as the center point, and then we assume
# that the L2 error is not more than "error_bound". We can then run subgradient ascent to figure
# out the maximin assignment where we worst-case over the true scores within "error_bound" of
# the tpms scores.
def solve_max_min(tpms, covs, loads, error_distrib, u_mag, noise_model="ball"):
    assert noise_model in ["ball", "ellipse"]

    st = time.time()
    print("Solving for initial max USW alloc", flush=True)
    # _, alloc = solve_usw_gurobi(tpms, covs, loads)
    alloc = np.random.randn(tpms.shape[0], tpms.shape[1])
    alloc = project_to_feasible(alloc, covs, loads)

    global_opt_obj = 0.0
    global_opt_alloc = alloc.copy()

    print("Solving max min: %s elapsed" % (time.time() - st), flush=True)

    converged = False
    max_iter = 300

    # Init params for grad asc

    # For adagrad
    # cache = np.zeros(tpms.shape)
    # lr = .01
    # eps = 1e-4

    # For vanilla
    t = 0
    lr = 1
    steps_no_imp = 0

    while not converged and t < max_iter:
        # Compute the worst-case S matrix using second order cone programming
        print("Computing worst case S matrix")
        print("%s elapsed" % (time.time() - st), flush=True)

        # worst case depends on noise model.
        worst_s = get_worst_case(alloc, tpms, error_distrib, u_mag, noise_model=noise_model)

        if noise_model == "ball":
            diff = np.sqrt(np.sum((worst_s - tpms) ** 2))
            assert diff - 1e-2 <= u_mag
        elif noise_model == "ellipse":
            diff = np.abs(worst_s - tpms)
            empirical_u = (1 / (error_distrib + 1e-9)) * diff
            empirical_u_mag = np.sqrt(np.sum(empirical_u ** 2))
            assert np.all(empirical_u_mag - 1e-2 < u_mag)

        # Update the allocation
        # 1, compute the gradient (I think it's just the value of the worst s, but check to be sure).
        # 2, update using the rate parameter times the gradient.
        # 3, project to the set of allocations that meet all the hard constraints

        old_alloc = alloc.copy()

        alloc_grad = worst_s

        # vanilla update
        if t % 10 == 0:
            lr /= 2
        alloc = alloc + lr * alloc_grad

        # Project to the set of feasible allocations
        print("Projecting to feasible set: %s elapsed" % (time.time() - st), flush=True)
        alloc = project_to_feasible(alloc, covs, loads)

        # alloc = bvn(alloc)

        # Check for convergence, update t
        # update_amt = np.linalg.norm(alloc - old_alloc)
        # converged = np.isclose(0, update_amt, atol=1e-6)
        t += 1

        prev_obj_val = np.sum(old_alloc * worst_s)
        if prev_obj_val > global_opt_obj:
            global_opt_obj = prev_obj_val
            global_opt_alloc = old_alloc
        else:
            steps_no_imp += 1

        if steps_no_imp > 10:
            return global_opt_alloc

        if t % 1 == 0:
            print("Step %d" % t)
            print("Obj value: ", np.sum(old_alloc * worst_s))
            print("%s elapsed" % (time.time() - st), flush=True)

    return global_opt_alloc


def solve_max_min_project_each_step(tpms, covs, loads, error_bound):
    st = time.time()
    print("Solving for initial max USW alloc")
    # _, alloc = solve_usw_gurobi(tpms, covs, loads)
    alloc = np.random.randn(tpms.shape[0], tpms.shape[1])
    alloc = project_to_feasible(alloc, covs, loads)
    alloc = project_to_integer(alloc, covs, loads)

    global_opt_obj = 0.0
    global_opt_alloc = alloc.copy()

    print("Solving max min: %s elapsed" % (time.time() - st))

    t = 0
    converged = False
    max_iter = 100

    while not converged and t < max_iter:
        # rate = 1/(t+1)

        # Compute the worst-case S matrix using second order cone programming
        print("Computing worst case S matrix")
        print("%s elapsed" % (time.time() - st))
        worst_s = get_worst_case(alloc, tpms, error_bound)

        diff = np.sqrt(np.sum((worst_s - tpms) ** 2))
        assert diff - 1e-2 <= error_bound

        # Update the allocation
        # 1, compute the gradient (I think it's just the value of the worst s, but check to be sure).
        # 2, update using the rate parameter times the gradient.
        # 3, project to the set of allocations that meet all the hard constraints

        old_alloc = alloc.copy()
        alloc_grad = worst_s

        rate = 1 / (t + 1)
        updated = False
        while not updated:
            alloc = old_alloc + rate * alloc_grad

            # Project to the set of feasible allocations
            print("Projecting to integer: %s elapsed" % (time.time() - st))
            alloc = project_to_integer(alloc, covs, loads)

            # Check for convergence, update t
            update_amt = np.linalg.norm(alloc - old_alloc)
            if np.isclose(0, update_amt, atol=1e-6):
                rate *= 2
            else:
                updated = True

        prev_obj_val = np.sum(old_alloc * worst_s)
        if prev_obj_val > global_opt_obj:
            global_opt_obj = prev_obj_val
            global_opt_alloc = old_alloc
        t += 1

        if t % 1 == 0:
            print("Step %d" % t)
            print("Obj value from prev step: ", prev_obj_val)
            print("%s elapsed" % (time.time() - st))

    # return project_to_integer(alloc, covs, loads)
    return global_opt_alloc
