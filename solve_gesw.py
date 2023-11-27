import numpy as np

# from docplex.mp.model import Model
from gurobipy import *


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
    # num_groups = int(np.max(groups)) + 1
    # for g in range(num_groups):
    #     group_welfare =
    #     m.addConstr(c <= group_welfare, 'c%d' % g)


def convert_to_mat(m, num_papers, num_revs):
    alloc = np.zeros((num_revs, num_papers))
    for var in m.getVars():
        if var.varName.startswith("assign") and var.x > .1:
            s = re.findall("(\d+)", var.varName)
            p = int(s[0])
            r = int(s[1])
            alloc[r, p] = 1
    return alloc


# Groups is a 1D numpy array with n entries, holds the group label for each paper.
def solve_gesw_gurobi(affinity_scores, covs, loads, groups):
    # paper_rev_pairs, pras = create_multidict(affinity_scores)
    #
    # m = Model("TPMS")
    #
    # x, c = add_vars_to_model(m, paper_rev_pairs)
    # add_constrs_to_model(m, x, c, covs, loads, groups)
    #
    # m.setObjective(c, GRB.MAXIMIZE)

    # m.write("TPMS.lp")

    m = Model("TPMS")
    x = m.addMVar(shape=affinity_scores.shape, vtype=GRB.BINARY, name='asst')
    c = m.addVar(0.0, 1000.0, obj=1.0, name='gesw')

    # Add general constraints
    # add_constrs_to_model(m, x, covs, loads)
    rev_ones = np.ones(affinity_scores.shape[0])
    pap_ones = np.ones(affinity_scores.shape[1])

    # m.addConstrs((x.sum(paper, '*') == covs[paper] for paper in papers), 'covs')  # Paper coverage constraints
    # m.addConstrs((x.sum('*', rev) <= loads[rev] for rev in revs), 'loads_ub')  # Reviewer load constraints
    # m.addConstr(x @ pap_ones <= loads, 'loads_ub')  # Reviewer load constraints
    # m.addConstr(x.T @ rev_ones == covs, 'covs')  # Paper coverage constraints
    m.addConstr(x.sum(axis=0) == covs, name='covs')
    m.addConstr(x.sum(axis=1) <= loads, name='loads')

    # Add constraint to make c = group esw (c <= group usw for all groups)
    num_groups = int(np.max(groups)) + 1
    realized_scores = affinity_scores * x
    for g in range(num_groups):
        group_size = np.where(groups == g)[0].shape[0]
        group_indicator = (groups == g).astype(int)
        group_welfare = (realized_scores @ group_indicator) @ rev_ones
        group_welfare /= group_size
        m.addConstr(group_welfare >= c, name=('c%d' % g))

    # Set the objective
    m.setObjective(c, GRB.MAXIMIZE)

    m.optimize()

    # Convert to the format we were using, and then print it out and run print_stats
    alloc = convert_to_mat(m, covs.shape[0], loads.shape[0])

    return m.objVal, alloc
