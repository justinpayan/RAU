import numpy as np
import os
import cvxpy as cp
import time

from scipy.stats.distributions import chi2

from solve_usw import solve_usw_gurobi


# Reconstruct the full bid matrix given the samples.
# fixed_estimators is a list of fixed m x n matrices with estimates (for example, tpms scores, keyword matching scores)
# hierarchical_information is a list of tuples of (rev_mixture, pap_mixture). Each tuple is generated
# by a soft clustering of reviewers and papers, which has been done ahead of time and fixed. We will use these mixtures
# to estimate a "proclivity matrix" which indicates how likely each category of reviewers is to bid on each category
# of papers.
# sampled_bids is a matrix that has the sampled bids where they've been sampled and nan elsewhere
def reconstruct_bids(sampled_bids, fixed_estimators, hierarchical_information):
    m, n = fixed_estimators[0].shape

    # bid_mask is 1 where we have sampled bids, 0 elsewhere
    bid_mask = 1 - sampled_bids.isnan(sampled_bids)
    b = np.sum(bid_mask)[0]

    # Need a lambda value for each estimator
    lambdas = [1] * len(fixed_estimators)
    lambdas.extend([1] * len(hierarchical_information))
    lambdas.append(1)  # One last estimator, which is just the direct estimator

    # TODO: Lambda values are learned!!!



    # These things are m x c and n x c, so we should take c x c sized variables for the proclivity matrices
    hierarchical_estimators = [cp.Variable((h[0].shape[1], h[0].shape[1])) for h in hierarchical_information]
    S_hat_direct = cp.Variable((m, n))

    # Compute S_hat, which is basically just the linear combination of all the estimators.
    S_hat = lambdas[-1] * S_hat_direct

    # TODO: Add scale parameters for the fixed estimators, on top of the learned lambda values.
    # TODO: we'll set them so that the Rade avg of all the fixed and learned estimators are equal.

    for i in range(len(fixed_estimators)):
        S_hat += lambdas[i]*fixed_estimators[i]

    for i in range(len(hierarchical_information)):
        S_hat += lambdas[len(fixed_estimators) + i] * ((hierarchical_information[i][0] @ hierarchical_estimators[i]) @
                                                   hierarchical_information[i][1].T)

    t = cp.Variable(1)

    # Now construct the constraints.
    # The first constraint allows us to express the quadratic loss function as
    # min t s.t. [I_b, P S_hat], [(S_hat P)^T, t - r^T S_hat] is PSD.
    # P has b rows and mn columns, with a 1 in the rp entry iff the b'th bid is for reviewer r and paper p
    # r is a length mn vector with -2 * true bid for the sampled bid idxs and 0 everywhere else
    # The rest of the constraints are there to enforce the max-norm constraints on the estimator matrices.
    raveled_bid_mask = bid_mask.ravel()
    raveled_S_hat = cp.reshape(S_hat, (m * n, 1), order='C')

    P = np.zeros((b, m*n))
    P[list(range(b)), np.where(raveled_bid_mask)[0]] = 1
    r = -2*sampled_bids
    r[r.isnan()] = 0
    r = np.reshape(r, (1, m*n))

    l_side = cp.vstack([np.eye(b), raveled_S_hat.T @ P.T])
    r_side = cp.vstack([P @ raveled_S_hat, t-(r @ raveled_S_hat)])

    # This matrix needs to be positive semidefinite.
    constraints = [cp.hstack([l_side, r_side]) >> 0]

    # include the constraints that correspond to the max norm constraints.
    W1_direct = cp.Variable((m, m))
    W2_direct = cp.Variable((n, n))
    l_side_direct = cp.vstack([W1_direct, S_hat_direct.T])
    r_side_direct = cp.vstack([S_hat_direct, W2_direct])
    constraints.append(cp.hstack([l_side_direct, r_side_direct]) >> 0)

    # TODO: Set these scale parameters, the max-norm constraints, such that every estimator has equal Rade avg
    R = 1
    constraints.append(cp.diag(W1_direct) <= np.ones(m)*R)
    constraints.append(cp.diag(W2_direct) <= np.ones(n)*R)
    constraints.append(S_hat_direct <= np.ones(S_hat_direct.shape))

    # Do a constraint for each of the hierarchical levels
    for i in range(len(hierarchical_information)):
        he = hierarchical_estimators[i]
        c = he.shape[0]
        W1 = cp.Variable(he.shape)
        W2 = cp.Variable(he.shape)
        l_side = cp.vstack([W1, he.T])
        r_side = cp.vstack([he, W2])
        constraints.append(cp.hstack([l_side, r_side]) >> 0)

        # TODO: Set these scale parameters, the max-norm constraints, such that every estimator has equal Rade avg
        R = 1
        constraints.append(cp.diag(W1) <= np.ones(c) * R)
        constraints.append(cp.diag(W2) <= np.ones(c) * R)
        constraints.append(he <= np.ones(he.shape))

    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve()

    reconstructed_bid_matrix = S_hat.value

    # Calculate the empirical error
    empirical_error = np.mean((sampled_bids - reconstructed_bid_matrix)[bid_mask]**2)

    # TODO: Calculate the generalization error (maybe with a Monte Carlo Rademacher average, probably just
    # TODO: using the general upper bound we give in the paper.
    generalization_error = 0

    return reconstructed_bid_matrix, empirical_error, generalization_error

