import numpy as np
import os
import cvxpy as cp
import time

from scipy.stats.distributions import chi2

from solve_usw import solve_usw_gurobi

import gurobipy as gp

import numpy as np
from scipy.stats import chi2


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


def get_worst_case(alloc, tpms, std_devs, noise_model="ball"):
    if noise_model == "ball":
        # s = cp.Variable(tpms.shape[0] * tpms.shape[1])
        #
        # soc_constraint = [cp.SOC(u_mag, s - tpms.ravel())]
        # prob = cp.Problem(cp.Minimize(alloc.ravel().T @ s),
        #                   soc_constraint + [s >= np.zeros(s.shape), s <= np.ones(s.shape)])
        # prob.solve(solver='SCS')
        #
        # return s.value.reshape(tpms.shape)
        return np.zeros(alloc.shape)
    elif noise_model == "ellipse":
        u = cp.Variable(tpms.shape[0] * tpms.shape[1])

        soc_constraint = [cp.SOC(np.sqrt(chi2.ppf(.95, alloc.size)), cp.multiply(u-tpms.ravel(), 1/std_devs.ravel()))]
        prob = cp.Problem(cp.Minimize(alloc.ravel().T @ u),
                          soc_constraint + [u >= np.zeros(u.shape), u <= np.ones(u.shape)])
        # prob.solve(solver='SCS')
        print(np.sqrt(chi2.ppf(.95, alloc.size)))
        print(np.sum(((tpms.ravel()-tpms.ravel()) * (1/std_devs.ravel()))**2))
        prob.solve()
        print()
        print(u.value)

        return u.value.reshape(tpms.shape)


def project_to_feasible_exact(alloc, covs, loads, use_verbose=False):
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
    prob.solve(verbose=use_verbose)
    return x.value

def project_to_feasible(alloc, covs, loads, max_iter=np.inf):
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
    t = 0

    while not converged and t < max_iter:
        t += 1

        # Project to each constraint
        # LB
        new_u = (u + z_lb).copy()
        proj_new_u = np.clip(new_u, 0, None)
        z_lb = new_u - proj_new_u
        u = proj_new_u

        # UB
        new_u = (u + z_ub).copy()
        proj_new_u = np.clip(new_u, None, 1)
        z_ub = new_u - proj_new_u
        u = proj_new_u

        # Paper coverage
        new_u = (u + z_pap_cov).copy()
        true_cov = np.sum(new_u, axis=0)
        proj_new_u = new_u - ((true_cov - covs) / m).reshape((1, -1))
        z_pap_cov = new_u - proj_new_u
        u = proj_new_u

        # Reviewer load bounds
        new_u = (u + z_rev_load).copy()
        true_load = np.sum(new_u, axis=1)
        proj_new_u = new_u - (np.clip(true_load - loads, 0, None) / n).reshape((-1, 1))
        z_rev_load = new_u - proj_new_u
        u = proj_new_u

        # Let's say we've converged when none of the constraints is too far violated.
        # print("Violation of LB")
        # print(np.abs(np.sum(u[u < 0])))
        # print("Violation of UB")
        # print(np.sum(np.clip(u - np.ones(u.shape), 0, None)))
        # print("Violation of paper cov")
        # print(np.sum(np.abs(np.sum(u, axis=0) - covs)))
        # print("Violation of review loads")
        # print(np.sum(np.clip(np.sum(u, axis=1) - loads, 0, None)), flush=True)
        if np.abs(np.sum(u[u < 0])) < .1 and \
                np.sum(np.clip(u - np.ones(u.shape), 0, None)) < .1 and \
                np.sum(np.abs(np.sum(u, axis=0) - covs)) < .1 and \
                np.sum(np.clip(np.sum(u, axis=1) - loads, 0, None)) < .1:
            converged = True

    print("Ran %d iterations of projection algorithm" % t)
    print("Violation of LB")
    print(np.abs(np.sum(u[u < 0])))
    print("Violation of UB")
    print(np.sum(np.clip(u - np.ones(u.shape), 0, None)))
    print("Violation of paper cov")
    print(np.sum(np.abs(np.sum(u, axis=0) - covs)))
    print("Violation of review loads")
    print(np.sum(np.clip(np.sum(u, axis=1) - loads, 0, None)), flush=True)

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
def solve_max_min(tpms, covs, loads, std_devs, caching=False, dykstra=False, noise_model="ball"):
    assert noise_model in ["ball", "ellipse"]

    st = time.time()
    print("Solving for initial max USW alloc", flush=True)
    _, alloc = solve_usw_gurobi(tpms, covs, loads)
    # alloc = np.random.randn(tpms.shape[0], tpms.shape[1])
    # alloc = np.clip(alloc, 0, 1)
    # alloc = project_to_feasible(alloc, covs, loads)

    global_opt_obj = 0.0
    global_opt_alloc = alloc.copy()

    print("Solving max min: %s elapsed" % (time.time() - st), flush=True)

    converged = False
    max_iter = 30

    # Init params for grad asc

    # For adagrad
    # cache = np.zeros(tpms.shape)
    # lr = .01
    # eps = 1e-4

    # For vanilla
    t = 0
    lr = 1
    steps_no_imp = 0

    adv_times = []
    proj_times = []

    u = cp.Variable(tpms.shape)
    soc_constraint = [
        cp.SOC(np.sqrt(chi2.ppf(.95, alloc.size)), cp.reshape(cp.multiply(u - tpms, 1 / std_devs), (tpms.shape[0]*tpms.shape[1])))]
    alloc_param = cp.Parameter(alloc.shape)
    # adv_prob = cp.Problem(cp.Minimize(alloc_param.T @ u),
    #                       soc_constraint + [u >= np.zeros(u.shape), u <= np.ones(u.shape)])
    adv_prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(alloc_param, u))),
                          soc_constraint + [u >= np.zeros(u.shape), u <= np.ones(u.shape)])
    print("adv_prob is DPP? ", adv_prob.is_dcp(dpp=True))
    print("adv_prob is DCP? ", adv_prob.is_dcp(dpp=False))

    x = cp.Variable(tpms.shape)
    m, n = tpms.shape
    cost = cp.sum_squares(x - alloc_param)
    n_vec = np.ones((n, 1))
    m_vec = np.ones((m, 1))
    constraints = [x @ n_vec <= loads.reshape((m, 1)),
                   x.T @ m_vec == covs.reshape((n, 1)),
                   x >= np.zeros(x.shape),
                   x <= np.ones(x.shape)]
    proj_prob = cp.Problem(cp.Minimize(cost), constraints)

    print("proj_prob is DPP? ", proj_prob.is_dcp(dpp=True))
    print("proj_prob is DCP? ", proj_prob.is_dcp(dpp=False))

    while not converged and t < max_iter:
        # Compute the worst-case S matrix using second order cone programming
        print("Computing worst case S matrix")
        print("%s elapsed" % (time.time() - st), flush=True)

        # worst case depends on noise model.
        # worst_s = get_worst_case(alloc, tpms, std_devs, noise_model=noise_model)

        st = time.time()
        if caching:
            # alloc_param.value = alloc.ravel()
            alloc_param.value = alloc
            adv_prob.solve(warm_start=True)
            worst_s = u.value
            # worst_s = u.value.reshape(tpms.shape)
        else:
            worst_s = get_worst_case(alloc, tpms, std_devs, noise_model=noise_model)
        adv_times.append(time.time() - st)

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
        st = time.time()
        if dykstra:
            alloc = project_to_feasible(alloc, covs, loads, max_iter=1500)
        else:
            if caching:
                # alloc_param.value = alloc.ravel()
                alloc_param.value = alloc
                proj_prob.solve(warm_start=True)
                # alloc = x.value.reshape(tpms.shape)
                alloc = x.value
            else:
                alloc = project_to_feasible_exact(alloc, covs, loads)
        proj_times.append(time.time() - st)

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

    st = time.time()
    if caching:
        # alloc_param.value = global_opt_alloc.ravel()
        alloc_param.value = global_opt_alloc
        proj_prob.solve(warm_start=True)
        # final_alloc = x.value.reshape(tpms.shape)
        final_alloc = x.value
    else:
        final_alloc = project_to_feasible_exact(global_opt_alloc, covs, loads)
    proj_times.append(time.time() - st)

    return final_alloc


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




def compute_objective(optimal_x, c):
    val = np.dot(optimal_x, c)
    return val


def check_ellipsoid(Sigma, mu, x, rsquared):
    temp = (x - mu).reshape(-1, 1)
    temp1 = np.matmul(temp.transpose(), Sigma)
    temp2 = np.matmul(temp1.reshape(1, -1), temp)

    if temp2.flatten()[0] <= rsquared:
        return True
    else:
        return False


def softtime(model, where):
    softlimit = 5
    gaplimit = 0.05
    if where == gp.GRB.Callback.MIP:
        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
        objbst = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        gap = abs((objbst - objbnd) / objbst)

        if runtime > softlimit and gap < gaplimit:
            model.terminate()


def solve_max_min_alt(tpms, covs, loads, std_devs, integer=True, rsquared=None, check=False):
    """
    :param tpms: 2d matrix of size #reviewers x #papers representing means
    :param covs: 1d numpy array with length # papers representing no of reviews required per paper
    :param loads: 1d numpy array with length # reviewers representing maximum no of papers per reviewer
    :param std_devs: 2d matrix of size #reviewers x #papers representing std devs
    :param noise_model: "ball" or "ellipse"
    :param integer:
    :return: 2d matrix #reviewers x #papers representing allocation and another 2d matrix #reviewers x #papers
        representing affinite scores
    """
    tpms = np.array(tpms)
    std_devs = np.array(std_devs)
    covs = np.array(covs)
    loads = np.array(loads)
    n_reviewers = tpms.shape[0]
    n_papers = tpms.shape[1]

    assert (np.all(covs <= n_reviewers))

    if rsquared is None:
        rsquared = chi2.ppf(.95, tpms.size)

    num = int(n_reviewers * n_papers)
    m = gp.Model()
    mu = tpms.flatten()
    var = (std_devs.flatten()) ** 2
    diag = var
    idiag = 1.0 / var

    lamda = m.addVar(0.0, gp.GRB.INFINITY, 0.0, gp.GRB.CONTINUOUS, "lamda")

    beta = m.addMVar(num, lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta")

    zeta = m.addVar(0.0, gp.GRB.INFINITY, 0.0, gp.GRB.CONTINUOUS, "zeta")

    lrs = m.addVar(0.0, gp.GRB.INFINITY, 0.0, gp.GRB.CONTINUOUS, "lrs")

    if integer == True:
        alloc = m.addMVar(num, lb=0, ub=1, vtype=gp.GRB.INTEGER, name="alloc")
    else:
        alloc = m.addMVar(num, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="alloc")

    temp1 = m.addMVar(num, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, ub=gp.GRB.INFINITY, name="temp1")
    temp2 = m.addMVar(num, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, ub=gp.GRB.INFINITY, name="temp2")

    temp3 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="temp3")
    temp4 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="temp4")

    temp6 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="temp6")

    temp5 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="temp5")

    temp8 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="temp8")

    for idx in range(num):
        m.addConstr(temp1[idx] == mu[idx], name='c1')

    for idx in range(num):
        m.addConstr(temp2[idx] == (alloc[idx] - beta[idx]), name="c2" + str(idx))

    for idx in range(num):
        m.addConstr(temp3[idx] == temp2[idx] * diag[idx], name='c3' + str(idx))

    for idx in range(num):
        m.addConstr(temp4[idx] == temp2[idx] * temp3[idx], name='c4' + str(idx))

    for idx in range(num):
        m.addConstr(temp5[idx] == temp4[idx] * zeta, name='c5' + str(idx))

    for idx in range(num):
        m.addConstr(temp6[idx] == temp1[idx] * temp2[idx], name='c6' + str(idx))

    for idx in range(num):
        m.addConstr(temp8[idx] == temp6[idx] - temp5[idx], name='c7' + str(idx))

    for idx in range(num):
        m.addConstr(beta[idx] >= 0, name='c8' + str(idx))

    for idx in range(n_reviewers):
        m.addConstr(gp.quicksum(alloc[idx * n_papers + jdx] for jdx in range(n_papers)) <= loads[idx],
                    name="c9" + str(idx))

    for idx in range(n_papers):
        m.addConstr(
            gp.quicksum(alloc[jdx * n_papers + idx] for jdx in range(n_reviewers)) == covs[idx],
            name="c10" + str(idx))

    m.addConstr(lamda * zeta * 4 == 1, name='c11')

    m.addConstr(lrs == lamda * rsquared, name='c12')

    m.addConstr(lamda >= 0.0, name="c13")
    m.addConstr(zeta >= 0, name="c14")

    m.params.NonConvex = 2
    m.setObjective(gp.quicksum(temp8[idx] for idx in range(num)) - lrs, gp.GRB.MAXIMIZE)
    m.setParam('OutputFlag', 1)

    m.optimize(softtime)
    alloc_v = alloc.X

    lamda_v = lamda.X
    beta_v = beta.X
    diff = (alloc_v - beta_v)
    affinity = mu - (diff * diag) / (2 * lamda_v)
    if check == True:
        sigma = np.eye(num) * var
        print(check_ellipsoid(sigma, mu, affinity, rsquared))
    m.dispose()

    del m
    return affinity, alloc_v


def get_worst_case_alt(alloc, tpms, std_devs, noise_model="ball", rsquared=None, check=False):
    """
    :param tpms: 2d matrix of size #reviewers x #papers representing means
    :param covs: 1d numpy array with length # papers representing no of reviews required per paper
    :param loads: 1d numpy array with length # reviewers representing maximum no of papers per reviewer
    :param std_devs: 2d matrix of size #reviewers x #papers representing std devs
    :param noise_model: "ball" or "ellipse"
    :return: 2d matrix #reviewers x #papers representing allocation and another 2d matrix #reviewers x #papers
    representing affinite scores
    """

    tpms = np.array(tpms)
    std_devs = np.array(std_devs)
    n_reviewers = tpms.shape[0]
    n_papers = tpms.shape[1]

    # assert (np.all(covs <= n_reviewers))

    if rsquared is None:
        rsquared = chi2.ppf(.95)

    num = int(n_reviewers * n_papers)
    m = gp.Model()
    mu = tpms.flatten()
    var = (std_devs.flatten()) ** 2
    ivar = 1 / var
    diag = var
    idiag = ivar

    m = gp.Model()
    x = m.addMVar(num, lb=0, ub=gp.GRB.INFINITY, name="affinity")
    temp1 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="temp1")
    temp2 = m.addMVar(num, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="temp2")
    temp3 = m.addMVar(num, lb=0, ub=gp.GRB.INFINITY, name="temp2")

    for idx in range(num):
        m.addConstr(temp1[idx] == (x[idx] - mu[idx]), name="c2" + str(idx))

    for idx in range(num):
        m.addConstr(temp2[idx] == temp1[idx] * idiag[idx], "c3" + str(idx))

    for idx in range(num):
        m.addConstr(temp3[idx] == temp2[idx] * temp1[idx], "c4" + str(idx))

    m.addConstr(gp.quicksum(temp3[idx] for idx in range(num)) <= rsquared, name="c5")

    m.params.NonConvex = 2
    m.setObjective(gp.quicksum(alloc[idx] * x[idx] for idx in range(num)), gp.GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)

    m.optimize(softtime)

    affinity = np.round(np.array(x.X), 5)
    if check == True:
        isigma = np.eye(num) * ivar
        print(check_ellipsoid(isigma, mu, affinity, rsquared))
    m.dispose()
    del m
    return affinity
#
# if __name__=='__main__':
#     n = np.random.randint(10, 100)
#
#     n_reviewers = 5
#     n_papers = 10
#     n = n_reviewers * n_papers
#     c = np.random.uniform(0.1, 1, n)
#     k = np.random.uniform(0.1, 1, n)
#     ksquared = k * k
#     sigma = np.eye(n) * ksquared
#     mu = np.random.uniform(0.1, 1, n)
#     p = np.random.rand()
#     df = np.random.randint(1, 10)
#     rsquared = chi2.ppf(p, df=df)
#     loads = np.ones(n_reviewers) * n_papers
#     covs = np.random.randint(1, n_reviewers, n_papers)
#
#     std_devs = np.sqrt(np.diag(sigma))
#     mu = mu.reshape((n_reviewers, n_papers))
#
#     affinity, alloc_s = solve_max_min(mu, covs, loads, std_devs, rsquared=rsquared)
#     affinity1 = get_worst_case(alloc_s, mu, std_devs, rsquared=rsquared)
#     sol = compute_objective(affinity, alloc_s)
#     sol1 = compute_objective(affinity1, alloc_s)
#     print("solution1 ", sol)
#     print("solution2", sol1)