import random

from copy import deepcopy
from collections import defaultdict
import numpy as np

from utils import spfa


class QueryModel(object):
    def __init__(self, tpms):
        self.v_tilde = tpms.copy()
        self.already_queried = defaultdict(set)
        self.m, self.n = tpms.shape

    def get_query(self, reviewer):
        pass

    def update(self, r, query, response):
        self.v_tilde[r, query] = response
        self.already_queried[r].add(query)


class RandomQueryModel(QueryModel):
    def get_query(self, reviewer):
        return random.choice(list(set(range(self.n)) - self.already_queried[reviewer]))


class TpmsQueryModel(QueryModel):
    def get_query(self, reviewer):
        top_papers = np.argsort(self.v_tilde[reviewer, :]).tolist()
        for p in top_papers:
            if p not in self.already_queried[reviewer]:
                return p
        return None


# TODO: This won't work for ESW. Need to change the allocation update model and the way I estimate the variance given
# TODO: some allocation.
class VarianceReductionQueryModel(QueryModel):
    def __init__(self, tpms, covs, loads, solver):
        super().__init__(tpms)
        self.solver = solver
        self.covs = covs
        self.loads = loads
        self.curr_expected_value, self.curr_alloc = self.solver(self.v_tilde, self.covs, self.loads)

        # Bipartite graph, with reviewers on left side, papers on right. There is a dummy paper which we will
        # assign to all reviewers with remaining review load.
        # We need to have edges with positive v_tilde from paper j to reviewer i when j is assigned to i.
        # Any unassigned papers have edges from reviewer i to paper j with negative edge weight.
        # We draw an edge TO the dummy paper when a reviewer has been assigned at least one paper.
        # We draw an edge FROM the dummy paper when a reviewer still has extra capacity.
        # We will search for negative weight cycles in this thing.
        # TODO: once this whole thing is implemented, I should also make sure that the suggested updates are valid.
        self.residual_fwd_neighbors = {r: dict() for r in range(self.m)} | \
                                      {(p + self.m): dict() for p in range(self.n + 1)}

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                self.residual_fwd_neighbors[reviewer][self.n + self.m] = 0
            if num_papers < self.loads[reviewer]:
                self.residual_fwd_neighbors[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    self.residual_fwd_neighbors[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    self.residual_fwd_neighbors[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]

        self.n_samples = 5
        self.curr_variance = self._calculate_variance(self.curr_alloc, self.v_tilde)

    def get_query(self, reviewer):
        best_value = -np.inf
        best_q = None
        for q in set(range(self.n)) - self.already_queried[reviewer]:
            print("Checking value of query %d for reviewer %d" % (q, reviewer))
            # Compute the value of this paper. Return whichever has the best value.
            # If the paper is not in the current alloc to reviewer, then the alloc won't change if the reviewer bids no
            # Likewise, if the paper IS in the current alloc, the alloc won't change if the reviewer bids yes. The
            # variance will actually change slightly in this case though.

            # Estimate the reduction in variance for both answers
            if q in np.where(self.curr_alloc[reviewer, :])[0].tolist():
                _, updated_alloc_if_no = self._update_alloc(reviewer, q, 0)
                v_tilde_no = self.v_tilde.copy()
                v_tilde_no[reviewer, q] = 0
                var_if_no = self._calculate_variance(updated_alloc_if_no, v_tilde_no)
                var_red_if_no = self.curr_variance - var_if_no
            else:
                var_red_if_no = 0

            _, updated_alloc_if_yes = self._update_alloc(reviewer, q, 1)
            v_tilde_yes = self.v_tilde.copy()
            v_tilde_yes[reviewer, q] = 1
            var_if_yes = self._calculate_variance(updated_alloc_if_yes, v_tilde_yes)
            var_red_if_yes = self.curr_variance - var_if_yes

            expected_variance_reduction = self.v_tilde[reviewer, q] * var_red_if_yes + \
                                          (1 - self.v_tilde[reviewer, q]) * var_red_if_no
            print("Expected variance reduction of query %d for reviewer %d is %.4f" % (q, reviewer, expected_variance_reduction))
            if expected_variance_reduction > best_value:
                best_q = q
                best_value = expected_variance_reduction
        return best_q

    def update(self, r, query, response):
        super().update(r, query, response)
        self.curr_expected_value, self.curr_alloc, self.residual_fwd_neighbors = self._update_alloc(r, query, response)
        self.curr_variance = self._calculate_variance(self.curr_alloc, self.v_tilde)

    def _calculate_variance(self, alloc, v_tilde):
        # Take a few samples to determine the variance in the solution objective given this allocation and
        # this estimate v_tilde of the true valuations.
        # expected_objs = []
        # for i in range(self.n_samples):
        #     sampled_valns = np.random.uniform(size=v_tilde.shape) < v_tilde
        #     expected_objs.append(np.sum(alloc * sampled_valns))
        # return np.var(expected_objs)

        # If we assume that the v_tilde entries are uncorrelated, then variance is linear
        # Variance of a Bernoulli is p(1-p)
        return np.sum(alloc * v_tilde * (1 - v_tilde))

    def _update_alloc(self, r, query, response):
        # We know that if the queried paper is not currently assigned, and its value is 0, the allocation won't change.
        if self.curr_alloc[r, query] < .1 and response == 0:
            return self.curr_expected_value, self.curr_alloc

        # Otherwise, we need to repeatedly check for augmenting paths in the residual graph
        # Honestly, I should probably maintain the residual graph at all times
        # Also, I should first check for augmenting paths coming into/out of the edge we just
        # queried (oh... or I can just relax basically nm times until I find a negative weight cycle).

        # The residual graph can be represented as a matrix (based on the allocation matrix)
        # And then I will find an augmenting path by keeping an array that says what the length of the shortest path
        # is, and an array with the parents for each node.

        # Bipartite graph, with reviewers on left side, papers on right. There is a dummy paper which we will
        # assign to all reviewers with remaining review load.
        # We need to have edges with negative v_tilde from paper j to reviewer i when j is assigned to i.
        # Any unassigned papers have edges from reviewer i to paper j with positive edge weight.
        # We draw an edge TO the dummy paper when a reviewer has been assigned at least one paper.
        # We draw an edge FROM the dummy paper when a reviewer still has extra capacity.
        # TODO: once this whole thing is implemented, I should also make sure that the suggested updates are valid.

        # Use the shortest path faster algorithm to find negative weight cycles, until there aren't any.
        # https://konaeakira.github.io/posts/using-the-shortest-path-faster-algorithm-to-find-negative-cycles.html
        updated_alloc = self.curr_alloc.copy()
        res_copy = deepcopy(self.residual_fwd_neighbors)

        if self.curr_alloc[r, query] > .5:
            # update the weight of the edge from query to r (should be positive).
            res_copy[query + self.m][r] = response
        else:
            # update the weight of the edge from r to query (should be negative).
            res_copy[r][query + self.m] = -response

        cycle = True
        while cycle:
            cycle = spfa(res_copy)
            print(self.residual_fwd_neighbors)
            print(cycle)
            if cycle is not None:
                # update the allocation and residual graph using the cycle

                # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
                # reviewer, and unassign the i+1'th paper.
                ctr = 0 if cycle[0] < self.m else 1
                while ctr < len(cycle):
                    paper_to_assign = cycle[(ctr - 1) % len(cycle)] - self.m
                    paper_to_drop = cycle[(ctr + 1) % len(cycle)] - self.m
                    curr_rev = cycle[ctr]

                    print("Remove paper %d from reviewer %d, and add paper %d" % (paper_to_drop, curr_rev, paper_to_assign))

                    if paper_to_assign < self.n:
                        # We are assigning a non-dummy paper to the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_assign] = 1
                        # Reverse the edge and negate its weight
                        res_copy[paper_to_assign + self.m, curr_rev] = -res_copy[curr_rev, paper_to_assign + self.m]
                        del res_copy[curr_rev, paper_to_assign + self.m]
                    else:
                        # Assign dummy paper (meaning, we've dropped a paper without being assigned one)
                        # Do not update the allocation for this reviewer here
                        # Update the residual graph only if we have dropped the last paper, I think?
                        # We need to make it so that curr_rev can't receive the dummy paper anymore.
                        num_papers = np.sum(updated_alloc[curr_rev, :])
                        if num_papers < 0.1:
                            del res_copy[curr_rev][self.n + self.m]

                    if paper_to_drop < self.n:
                        # We are dropping a non-dummy paper from the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_drop] = 0
                        # Reverse the edge and negate its weight
                        res_copy[curr_rev, paper_to_drop + self.m] = -res_copy[paper_to_drop + self.m, curr_rev]
                        del res_copy[paper_to_drop + self.m, curr_rev]
                    else:
                        # Drop dummy paper (meaning, we've been assigned a paper without having to drop a paper)
                        # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
                        # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
                        num_papers = np.sum(updated_alloc[curr_rev, :])
                        if num_papers > self.loads[curr_rev] - .1:
                            del res_copy[self.n + self.m][curr_rev]

                    # Move to the next REVIEWER... not the next vertex in the cycle
                    ctr += 2

        # Ok, so now this should be the best allocation. Check the new value of the expected USW, and make sure it
        # exceeds the value from applying the previous allocation with the new v_tilde.
        updated_expected_value = np.sum(updated_alloc * self.v_tilde) - \
            self.v_tilde[r, query] * updated_alloc[r, query] + \
            response * updated_alloc[r, query]

        updated_expected_value_if_using_old_alloc = np.sum(self.curr_alloc * self.v_tilde) - \
            self.v_tilde[r, query] * self.curr_alloc[r, query] + \
            response * self.curr_alloc[r, query]

        assert updated_expected_value_if_using_old_alloc <= updated_expected_value
        # if updated_expected_value_if_using_old_alloc < updated_expected_value:
        #     print("Improved expected value")

        # TODO: This method just returns what the allocation would be. But I need to make sure to change the update()
        # TODO: method so that the residual graph gets updated when the object's allocation is updated.

        # TODO: Compute the updated expected value only at the very end, so we save that computation at least...
        # TODO: Anyway, I think sometimes it isn't even necessary and can be safely omitted.
        return updated_expected_value, updated_alloc, res_copy
