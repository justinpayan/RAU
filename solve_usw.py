import numpy as np

# from docplex.mp.model import Model
from gurobipy import *
import gurobipy as gp

def solve_usw(affinity_scores, covs, loads):
    opt_mod = Model(name="Max USW")
    m, n = affinity_scores.shape

    x = opt_mod.continuous_var_matrix(keys1=range(m), keys2=range(n), name='x', lb=0.0, ub=1.0)

    for p in range(n):
        num_revs_assigned = opt_mod.scal_prod_f(x, lambda keys: int(keys[1] == p))
        opt_mod.add_constraint(num_revs_assigned == covs[p])

    for r in range(m):
        num_papers_assigned = opt_mod.scal_prod_f(x, lambda keys: int(keys[0] == r))
        opt_mod.add_constraint(num_papers_assigned <= loads[r])

    obj_fn = opt_mod.scal_prod_f(x, lambda keys: affinity_scores[keys[0], keys[1]])

    opt_mod.maximize(obj_fn)
    print(opt_mod.print_information())
    print("solving")
    cplex_soln = opt_mod.solve()
    print("done")
    # print(opt_mod.print_solution())
    np_soln = np.zeros((m, n))
    for r in range(m):
        for p in range(n):
            np_soln[r, p] = cplex_soln[x[(r, p)]]

    return opt_mod.objective_value, np_soln


def create_multidict(pra):
    d = {}
    for rev in range(pra.shape[0]):
        for paper in range(pra.shape[1]):
            d[(paper, rev)] = pra[rev, paper]
    return multidict(d)


def add_vars_to_model(m, paper_rev_pairs):
    x = m.addVars(paper_rev_pairs, name="assign", vtype=GRB.BINARY)  # The binary assignment variables
    return x


def add_constrs_to_model(m, x, covs, loads):
    papers = range(covs.shape[0])
    revs = range(loads.shape[0])
    m.addConstrs((x.sum(paper, '*') == covs[paper] for paper in papers), 'covs')  # Paper coverage constraints
    total_demand = np.sum(covs)
    print("Total demand: %d" % total_demand)
    print("Number of revs: %d" % loads.shape[0])
    # max_num_papers_per_rev = math.ceil(total_demand/loads.shape[0])
    # min_num_papers_per_rev = math.floor(total_demand/loads.shape[0])
    # print("Max num papers per rev: %d" % max_num_papers_per_rev)
    # print("Min num papers per rev: %d" % min_num_papers_per_rev)
    m.addConstrs((x.sum('*', rev) <= loads[rev] for rev in revs), 'loads_ub')  # Reviewer load constraints
    # m.addConstrs((x.sum('*', rev) >= min_num_papers_per_rev for rev in revs), 'loads_lb')  # Reviewer load constraints


def convert_to_mat(m, num_papers, num_revs):
    alloc = np.zeros((num_revs, num_papers))
    for var in m.getVars():
        if var.varName.startswith("assign") and var.x > .1:
            s = re.findall("(\d+)", var.varName)
            p = int(s[0])
            r = int(s[1])
            alloc[r, p] = 1
    return alloc


def solve_usw_gurobi(affinity_scores, covs, loads):
    env = gp.Env(empty=True)
    env.setParam('WLSAccessID', 'a874a18f-8070-4ce8-b5c3-206c3625d8a6')
    env.setParam('WLSSecret', 'e685b441-d07f-43f4-aaeb-5e6385a6bc07')
    env.setParam('LicenseID', 937238)
    env.start()

    paper_rev_pairs, pras = create_multidict(affinity_scores)

    m = Model("TPMS")

    x = add_vars_to_model(m, paper_rev_pairs)
    add_constrs_to_model(m, x, covs, loads)

    m.setObjective(x.prod(pras), GRB.MAXIMIZE)

    # m.write("TPMS.lp")

    m.optimize()

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_mat(m, covs.shape[0], loads.shape[0])

    return m.objVal, alloc
