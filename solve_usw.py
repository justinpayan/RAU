import numpy as np

from docplex.mp.model import Model


def solve_usw(affinity_scores, covs, loads):
    opt_mod = Model(name="Max USW")
    m, n = affinity_scores.shape

    x = opt_mod.continuous_var_matrix(keys1=range(m), keys2=range(n), name='x', lb=0.0, ub=1.0)
    # vars = []
    # for r in range(m):
    #     rev_vars = []
    #     for p in range(n):
    #         rev_vars.append(
    #             opt_mod.continuous_var(lb=0.0, ub=1.0, name="%d_%d" % (r, p))
    #         )
    #     vars.append(rev_vars)

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

