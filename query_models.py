import random

from copy import deepcopy
from collections import defaultdict
import math
import numpy as np
import os

from utils import spfa, spfa_simple, super_algorithm


class QueryModel(object):
    def __init__(self, tpms, dset_name):
        self.dset_name = dset_name
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


class SuperStarQueryModel(QueryModel):
    def __init__(self, tpms, dset_name):
        super().__init__(tpms, dset_name)
        self.tpms_orig = tpms
        self.bids = np.zeros(tpms.shape)

    def get_query(self, reviewer):
        def g_r(s, pi=None):
            if pi is None:
                return (2 ** s - 1)
            else:
                return (2 ** s - 1) / np.log2(pi + 1)

        def f(s, pi=None):
            if pi is None:
                return s
            else:
                return s / np.log2(pi + 1)
        # g_p = lambda bids: np.sqrt(bids)
        g_p = lambda bids: np.clip(bids, a_min=0, a_max=6)
        # g_r = lambda s, pi: (2 ** s - 1) / np.log2(pi + 1)
        # f = lambda s, pi: s/np.log2(pi + 1)
        s = self.tpms_orig[reviewer, :]
        bids = self.bids[reviewer, :]
        h = np.zeros(bids.shape)
        trade_param = .5
        pi_t = super_algorithm(g_p, g_r, f, s, bids, h, trade_param, special=True)

        top_papers = np.argsort(pi_t)
        for p in top_papers:
            if p not in self.already_queried[reviewer]:
                return p
        return None

    def update(self, r, query, response):
        self.bids[r, query] = response
        self.v_tilde[r, query] = response
        self.already_queried[r].add(query)
        # TODO: update the heuristic? Just try it with the 0 heuristic first, because it seemed alright.


# TODO: This won't work for ESW. Need to change the allocation update model and the way I estimate the variance given
# TODO: some allocation.
class VarianceReductionQueryModel(QueryModel):
    def __init__(self, tpms, covs, loads, solver, dset_name):
        super().__init__(tpms, dset_name)
        self.solver = solver
        self.covs = covs
        self.loads = loads

        print("Loading/computing optimal initial solution")
        try:
            self.curr_expected_value = np.load(os.path.join("saved_init_expected_usw", dset_name + ".npy"))
            self.curr_alloc = np.load(os.path.join("saved_init_max_usw_soln", dset_name + ".npy"))
        except FileNotFoundError:
            print("Recomputing")
            os.makedirs("saved_init_expected_usw", exist_ok=True)
            os.makedirs("saved_init_max_usw_soln", exist_ok=True)

            self.curr_expected_value, self.curr_alloc = self.solver(self.v_tilde, self.covs, self.loads)
            np.save(os.path.join("saved_init_expected_usw", dset_name), self.curr_expected_value)
            np.save(os.path.join("saved_init_max_usw_soln", dset_name), self.curr_alloc)


        # Bipartite graph, with reviewers on left side, papers on right. There is a dummy paper which we will
        # assign to all reviewers with remaining review load.
        # We need to have edges with positive v_tilde from paper j to reviewer i when j is assigned to i.
        # Any unassigned papers have edges from reviewer i to paper j with negative edge weight.
        # We draw an edge TO the dummy paper when a reviewer has been assigned at least one paper.
        # We draw an edge FROM the dummy paper when a reviewer still has extra capacity.
        # We will search for negative weight cycles in this thing.
        # TODO: once this whole thing is implemented, I should also make sure that the suggested updates are valid.
        print("Setting up residual graph")
        self.residual_fwd_neighbors = {r: dict() for r in range(self.m)} | \
                                      {(p + self.m): dict() for p in range(self.n + 1)}

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                self.residual_fwd_neighbors[reviewer][self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                self.residual_fwd_neighbors[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    self.residual_fwd_neighbors[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    self.residual_fwd_neighbors[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]

        self.curr_variance = self._calculate_variance(self.curr_alloc, self.v_tilde)

    def get_query(self, reviewer):
        best_value = -np.inf
        best_q = None
        for q in set(range(self.n)) - self.already_queried[reviewer]:
            print("Determine value of %d to %d" % (q,reviewer))
            # Compute the value of this paper. Return whichever has the best value.
            # If the paper is not in the current alloc to reviewer, then the alloc won't change if the reviewer bids no
            # Likewise, if the paper IS in the current alloc, the alloc won't change if the reviewer bids yes. The
            # variance will actually change slightly in this case though.

            # Estimate the reduction in variance for both answers
            if q in np.where(self.curr_alloc[reviewer, :])[0].tolist():
                # print("Update if no")
                _, updated_alloc_if_no, _ = self._update_alloc(reviewer, q, 0)
                v_tilde_no = self.v_tilde.copy()
                v_tilde_no[reviewer, q] = 0
                var_if_no = self._calculate_variance(updated_alloc_if_no, v_tilde_no)
                var_red_if_no = self.curr_variance - var_if_no
            else:
                var_red_if_no = 0

            # print("Update if yes")
            _, updated_alloc_if_yes, _ = self._update_alloc(reviewer, q, 1)
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
            return self.curr_expected_value, self.curr_alloc, self.residual_fwd_neighbors

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

        sum_of_gains = 0

        # print("curr_alloc")
        # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
        #     print("%d: %s" % (rev, np.where(self.curr_alloc[rev, :])[0].tolist()))

        cycle = True
        while cycle:
            # print("SPFA start")
            cycle = spfa(res_copy)
            # print(cycle)

            if cycle is not None:
                # for i in cycle:
                    # print(i)
                    # print(res_copy[i])
                # update the allocation and residual graph using the cycle

                # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
                # reviewer, and unassign the i+1'th paper.
                total_gain = 0
                ctr = 0 if cycle[0] < self.m else 1
                while ctr < len(cycle):
                    paper_to_assign = cycle[(ctr - 1) % len(cycle)] - self.m
                    paper_to_drop = cycle[(ctr + 1) % len(cycle)] - self.m
                    curr_rev = cycle[ctr]

                    total_gain += res_copy[curr_rev][paper_to_assign + self.m]
                    total_gain += res_copy[paper_to_drop + self.m][curr_rev]

                    # print("Remove paper %d from reviewer %d, and add paper %d" % (paper_to_drop, curr_rev, paper_to_assign))
                    # print("Gain: %.2f, Loss: %.2f" % (res_copy[curr_rev][paper_to_assign + self.m], res_copy[paper_to_drop + self.m][curr_rev]))

                    if paper_to_assign < self.n:
                        # We are assigning a non-dummy paper to the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_assign] = 1
                        # Reverse the edge and negate its weight
                        res_copy[paper_to_assign + self.m][curr_rev] = -res_copy[curr_rev][paper_to_assign + self.m]
                        del res_copy[curr_rev][paper_to_assign + self.m]

                    if paper_to_drop < self.n:
                        # We are dropping a non-dummy paper from the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_drop] = 0
                        # Reverse the edge and negate its weight
                        res_copy[curr_rev][paper_to_drop + self.m] = -res_copy[paper_to_drop + self.m][curr_rev]
                        del res_copy[paper_to_drop + self.m][curr_rev]



                    # Update the residual graph if we have dropped the last paper
                    # We need to make it so that curr_rev can't receive the dummy paper anymore.
                    num_papers = np.sum(updated_alloc[curr_rev, :])
                    if num_papers < 0.1:
                        try:
                            del res_copy[curr_rev][self.n + self.m]
                        except KeyError:
                            pass
                    # If we have a paper assigned, we can ASSIGN the dummy
                    else:
                        res_copy[curr_rev][self.n + self.m] = 0

                    # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
                    # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
                    if num_papers > self.loads[curr_rev] - .1:
                        try:
                            del res_copy[self.n + self.m][curr_rev]
                        except KeyError:
                            pass
                    else:
                        # They can still give away the dummy
                        res_copy[self.n + self.m][curr_rev] = 0

                    # Move to the next REVIEWER... not the next vertex in the cycle
                    ctr += 2

                # print("Overall gain: ", total_gain)
                sum_of_gains += total_gain

        # Ok, so now this should be the best allocation. Check the new value of the expected USW, and make sure it
        # exceeds the value from applying the previous allocation with the new v_tilde.
        updated_expected_value = np.sum(updated_alloc * self.v_tilde) - \
            self.v_tilde[r, query] * updated_alloc[r, query] + \
            response * updated_alloc[r, query]

        updated_expected_value_if_using_old_alloc = np.sum(self.curr_alloc * self.v_tilde) - \
            self.v_tilde[r, query] * self.curr_alloc[r, query] + \
            response * self.curr_alloc[r, query]

        # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
        #     print("%d: %s" % (rev, np.where(updated_alloc[rev, :])[0].tolist()))
        #
        # print("We should expected new EV (%s) to be equal to old EV (%s) plus negative total gain (%s)" % (updated_expected_value, updated_expected_value_if_using_old_alloc, sum_of_gains))
        # print("new - (old - gain) = %s" % (updated_expected_value - (updated_expected_value_if_using_old_alloc - sum_of_gains)))

        if updated_expected_value_if_using_old_alloc > updated_expected_value:
            print("ERROR")
            print("PROBABLY AN ERROR, THIS SHOULDNT BE HAPPENING")
            print(updated_expected_value_if_using_old_alloc)
            print(updated_expected_value)
            print(np.isclose(updated_expected_value_if_using_old_alloc, updated_expected_value))
        # if updated_expected_value_if_using_old_alloc < updated_expected_value:
        #     print("Improved expected value")

        # TODO: This method just returns what the allocation would be. But I need to make sure to change the update()
        # TODO: method so that the residual graph gets updated when the object's allocation is updated.

        # TODO: Compute the updated expected value only at the very end, so we save that computation at least...
        # TODO: Anyway, I think sometimes it isn't even necessary and can be safely omitted.
        return updated_expected_value, updated_alloc, res_copy


# TODO: This won't work for ESW. Need to change the allocation update model and the way I estimate the variance given
# TODO: some allocation.
class GreedyMaxQueryModel(QueryModel):
    def __init__(self, tpms, covs, loads, solver, dset_name, data_dir, num_procs):
        super().__init__(tpms, dset_name)
        self.solver = solver
        self.covs = covs
        self.loads = loads
        self.num_procs = num_procs

        print("Loading/computing optimal initial solution")
        try:
            self.curr_expected_value = np.load(os.path.join(data_dir, "saved_init_expected_usw", dset_name + ".npy"))
            self.curr_alloc = np.load(os.path.join(data_dir, "saved_init_max_usw_soln", dset_name + ".npy"))
        except FileNotFoundError:
            print("Recomputing")
            os.makedirs(os.path.join(data_dir, "saved_init_expected_usw"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "saved_init_max_usw_soln"), exist_ok=True)

            self.curr_expected_value, self.curr_alloc = self.solver(self.v_tilde, self.covs, self.loads)
            np.save(os.path.join("saved_init_expected_usw", dset_name), self.curr_expected_value)
            np.save(os.path.join("saved_init_max_usw_soln", dset_name), self.curr_alloc)


        # Bipartite graph, with reviewers on left side, papers on right. There is a dummy paper which we will
        # assign to all reviewers with remaining review load.
        # We need to have edges with positive v_tilde from paper j to reviewer i when j is assigned to i.
        # Any unassigned papers have edges from reviewer i to paper j with negative edge weight.
        # We draw an edge TO the dummy paper when a reviewer has been assigned at least one paper.
        # We draw an edge FROM the dummy paper when a reviewer still has extra capacity.
        # We will search for negative weight cycles in this thing.
        # TODO: once this whole thing is implemented, I should also make sure that the suggested updates are valid.
        print("Setting up residual graph")
        self.residual_fwd_neighbors = np.zeros
        self.residual_fwd_neighbors = {r: dict() for r in range(self.m)} | \
                                      {(p + self.m): dict() for p in range(self.n + 1)}

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                self.residual_fwd_neighbors[reviewer][self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                self.residual_fwd_neighbors[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    self.residual_fwd_neighbors[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    self.residual_fwd_neighbors[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]

    def get_query(self, reviewer):
        qry_values = {}

        for q in set(range(self.n)) - self.already_queried[reviewer]:
            print("Determine value of %d to %d" % (q, reviewer), flush=True)
            # Compute the value of this paper. Return whichever has the best value.
            # If the paper is not in the current alloc to reviewer, then the alloc won't change if the reviewer bids no
            # Likewise, if the paper IS in the current alloc, the alloc won't change if the reviewer bids yes.

            # print("Reviewer %d is currently assigned %s" % (reviewer, np.where(self.curr_alloc[reviewer, :])))
            # Estimate the improvement in expected value for both answers
            if q in np.where(self.curr_alloc[reviewer, :])[0].tolist():
                # print("Update if no")
                updated_expected_value_if_no, _ = self._update_alloc(reviewer, q, 0)
            else:
                updated_expected_value_if_no = self.curr_expected_value

            improvement_ub = self.v_tilde[reviewer, q] * (1 - self.v_tilde[reviewer, q]) + self.curr_expected_value
            max_query_val = max(qry_values.values()) if qry_values else 0

            if qry_values and improvement_ub < max_query_val or math.isclose(improvement_ub, max_query_val):
                qry_values[q] = self.curr_expected_value
            else:
                updated_expected_value_if_yes, _ = self._update_alloc(reviewer, q, 1)

                expected_expected_value = self.v_tilde[reviewer, q] * updated_expected_value_if_yes + \
                                          (1 - self.v_tilde[reviewer, q]) * updated_expected_value_if_no
                print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
                qry_values[q] = expected_expected_value

        # print(sorted(qry_values.items(), key=lambda x: -x[1])[:5], sorted(qry_values.items(), key=lambda x: -x[1])[-5:])
        best_q = [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])][0]
        return best_q


    @staticmethod
    def check_expected_value(args):
        q, reviewer, max_query_val, query_model_object = args

        if q in np.where(query_model_object.curr_alloc[reviewer, :])[0].tolist():
            # print("Update if no")
            updated_expected_value_if_no = GreedyMaxQueryModel._update_alloc_static(reviewer, q, 0, query_model_object)
        else:
            updated_expected_value_if_no = query_model_object.curr_expected_value

        improvement_ub = query_model_object.v_tilde[reviewer, q] * (1 - query_model_object.v_tilde[reviewer, q]) + query_model_object.curr_expected_value

        if improvement_ub < max_query_val or math.isclose(improvement_ub, max_query_val):
            return query_model_object.curr_expected_value
        else:
            updated_expected_value_if_yes = GreedyMaxQueryModel._update_alloc_static(reviewer, q, 1, query_model_object)

            expected_expected_value = query_model_object.v_tilde[reviewer, q] * updated_expected_value_if_yes + \
                                      (1 - query_model_object.v_tilde[reviewer, q]) * updated_expected_value_if_no
            # print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
            return expected_expected_value

    def get_query_parallel(self, reviewer, pool):
        papers_to_check = set(range(self.n)) - self.already_queried[reviewer]

        qry_values = {}

        while len(papers_to_check):
            next_to_check = list(papers_to_check)[:min(self.num_procs, len(papers_to_check))]
            for p in next_to_check:
                papers_to_check.remove(p)

            max_query_val = max(qry_values.values()) if qry_values else 0

            list_of_copied_args = [next_to_check]
            for argument in [reviewer, max_query_val, self]:
                list_of_copied_args.append(len(next_to_check) * [argument])

            expected_expected_values = pool.map(GreedyMaxQueryModel.check_expected_value, zip(*list_of_copied_args))
            for q, eev in zip(next_to_check, expected_expected_values):
                qry_values[q] = eev

        best_q = [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])][0]
        return best_q

    # def get_queries(self, reviewer):
    #     qry_values = {}
    #     # to_process =
    #     for q in set(range(self.n)) - self.already_queried[reviewer]:
    #         print("Determine value of %d to %d" % (q, reviewer))
    #         # Compute the value of this paper. Return whichever has the best value.
    #         # If the paper is not in the current alloc to reviewer, then the alloc won't change if the reviewer bids no
    #         # Likewise, if the paper IS in the current alloc, the alloc won't change if the reviewer bids yes.
    #
    #         # Estimate the improvement in expected value for both answers
    #         if q in np.where(self.curr_alloc[reviewer, :])[0].tolist():
    #             # print("Update if no")
    #             updated_expected_value_if_no, _ = self._update_alloc(reviewer, q, 0)
    #         else:
    #             updated_expected_value_if_no = self.curr_expected_value
    #
    #         # print("Update if yes")
    #         # Assume that no paper gives an improvement better than bumping it up to 1 (i.e. 1-s_ij)
    #         # and then the probability of yes is s_ij. So if s_ij * (1 - s_ij) is worse than the best improvement so far,
    #         # we can skip.
    #         improvement_ub = self.v_tilde[reviewer, q] * (1 - self.v_tilde[reviewer, q]) + self.curr_expected_value
    #         max_query_val = max(qry_values.values()) if qry_values else 0
    #
    #         if qry_values and improvement_ub < max_query_val or math.isclose(improvement_ub, max_query_val):
    #             qry_values[q] = self.curr_expected_value
    #         else:
    #
    #             updated_expected_value_if_yes, _ = self._update_alloc(reviewer, q, 1)
    #
    #             expected_expected_value = self.v_tilde[reviewer, q] * updated_expected_value_if_yes + \
    #                                           (1 - self.v_tilde[reviewer, q]) * updated_expected_value_if_no
    #             # print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
    #             qry_values[q] = expected_expected_value
    #
    #     return [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])]
    #
    # def get_queries_parallel(self, reviewer):
    #     qry_values = {}
    #     to_process = []
    #     for q in set(range(self.n)) - self.already_queried[reviewer]:
    #         # print("Determine value of %d to %d" % (q, reviewer))
    #         # Compute the value of this paper. Return whichever has the best value.
    #         # If the paper is not in the current alloc to reviewer, then the alloc won't change if the reviewer bids no
    #         # Likewise, if the paper IS in the current alloc, the alloc won't change if the reviewer bids yes.
    #
    #         # Estimate the improvement in expected value for both answers
    #         if q in np.where(self.curr_alloc[reviewer, :])[0].tolist():
    #             # print("Update if no")
    #             updated_expected_value_if_no, _ = self._update_alloc(reviewer, q, 0)
    #         else:
    #             updated_expected_value_if_no = self.curr_expected_value
    #
    #         # print("Update if yes")
    #         # Assume that no paper gives an improvement better than bumping it up to 1 (i.e. 1-s_ij)
    #         # and then the probability of yes is s_ij. So if s_ij * (1 - s_ij) is worse than the best improvement so far,
    #         # we can skip.
    #         improvement_ub = self.v_tilde[reviewer, q] * (1 - self.v_tilde[reviewer, q]) + self.curr_expected_value
    #         max_query_val = max(qry_values.values()) if qry_values else 0
    #
    #         if qry_values and improvement_ub < max_query_val or math.isclose(improvement_ub, max_query_val):
    #             qry_values[q] = self.curr_expected_value
    #         else:
    #             to_process.append(q)
    #
    #         if len(to_process) >= self.num_procs:
    #             pass
    #             # self.execute_updates()
    #             # updated_expected_value_if_yes, _ = self._update_alloc(reviewer, q, 1)
    #             #
    #             # expected_expected_value = self.v_tilde[reviewer, q] * updated_expected_value_if_yes + \
    #             #                               (1 - self.v_tilde[reviewer, q]) * updated_expected_value_if_no
    #             # # print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
    #             # qry_values[q] = expected_expected_value
    #
    #     return [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])]

    def update(self, r, query, response):
        super().update(r, query, response)
        self.curr_expected_value, self.curr_alloc = self._update_alloc(r, query, response)

        # print("Setting up residual graph")
        self.residual_fwd_neighbors = {r: dict() for r in range(self.m)} | \
                                      {(p + self.m): dict() for p in range(self.n + 1)}

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                self.residual_fwd_neighbors[reviewer][self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                self.residual_fwd_neighbors[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    self.residual_fwd_neighbors[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    self.residual_fwd_neighbors[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]

    def _update_alloc(self, r, query, response):
        # We know that if the queried paper is not currently assigned, and its value is 0, the allocation won't change.
        # print("check value if paper %d for rev %d is %d" % (query, r, response))

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


        # print("curr_alloc")
        # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
        #     print("%d: %s" % (rev, np.where(self.curr_alloc[rev, :])[0].tolist()))

        cycle = True
        while cycle:
            # print("SPFA start")
            cycle = spfa(res_copy)
            # print(cycle)

            if cycle is not None:
                # for i in cycle:
                #     print(i)
                #     print(res_copy[i])
                # update the allocation and residual graph using the cycle

                # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
                # reviewer, and unassign the i+1'th paper.
                ctr = 0 if cycle[0] < self.m else 1
                while ctr < len(cycle):
                    paper_to_assign = cycle[(ctr - 1) % len(cycle)] - self.m
                    paper_to_drop = cycle[(ctr + 1) % len(cycle)] - self.m
                    curr_rev = cycle[ctr]

                    # print("Remove paper %d from reviewer %d, and add paper %d" % (paper_to_drop, curr_rev, paper_to_assign))
                    # print("Gain: %.2f, Loss: %.2f" % (res_copy[curr_rev][paper_to_assign + self.m], res_copy[paper_to_drop + self.m][curr_rev]))

                    if paper_to_assign < self.n:
                        # We are assigning a non-dummy paper to the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_assign] = 1
                        # Reverse the edge and negate its weight
                        res_copy[paper_to_assign + self.m][curr_rev] = -res_copy[curr_rev][paper_to_assign + self.m]
                        del res_copy[curr_rev][paper_to_assign + self.m]

                    if paper_to_drop < self.n:
                        # We are dropping a non-dummy paper from the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_drop] = 0
                        # Reverse the edge and negate its weight
                        res_copy[curr_rev][paper_to_drop + self.m] = -res_copy[paper_to_drop + self.m][curr_rev]
                        del res_copy[paper_to_drop + self.m][curr_rev]



                    # Update the residual graph if we have dropped the last paper
                    # We need to make it so that curr_rev can't receive the dummy paper anymore.
                    num_papers = np.sum(updated_alloc[curr_rev, :])
                    if num_papers < 0.1:
                        try:
                            del res_copy[curr_rev][self.n + self.m]
                        except KeyError:
                            pass
                    # If we have a paper assigned, we can ASSIGN the dummy
                    else:
                        res_copy[curr_rev][self.n + self.m] = 0

                    # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
                    # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
                    if num_papers > self.loads[curr_rev] - .1:
                        try:
                            del res_copy[self.n + self.m][curr_rev]
                        except KeyError:
                            pass
                    else:
                        # They can still give away the dummy
                        res_copy[self.n + self.m][curr_rev] = 0

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
        # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
        #     print("%d: %s" % (rev, np.where(updated_alloc[rev, :])[0].tolist()))
        #
        # print("We should expected new EV (%s) to be equal to old EV (%s) plus negative total gain (%s)" % (updated_expected_value, updated_expected_value_if_using_old_alloc, sum_of_gains))
        # print("new - (old - gain) = %s" % (updated_expected_value - (updated_expected_value_if_using_old_alloc - sum_of_gains)))

        if updated_expected_value_if_using_old_alloc > updated_expected_value:
            print("ERROR")
            print("PROBABLY AN ERROR, THIS SHOULDNT BE HAPPENING")
            print(updated_expected_value_if_using_old_alloc)
            print(updated_expected_value)
            print(np.isclose(updated_expected_value_if_using_old_alloc, updated_expected_value))
        # if updated_expected_value_if_using_old_alloc < updated_expected_value:
        #     print("Improved expected value")

        # TODO: Compute the updated expected value only at the very end, so we save that computation at least...
        # TODO: Anyway, I think sometimes it isn't even necessary and can be safely omitted.

        return updated_expected_value, updated_alloc

    @staticmethod
    def _update_alloc_static(r, query, response, query_model_object):
        # We know that if the queried paper is not currently assigned, and its value is 0, the allocation won't change.
        # print("check value if paper %d for rev %d is %d" % (query, r, response))

        if query_model_object.curr_alloc[r, query] < .1 and response == 0:
            return query_model_object.curr_expected_value, query_model_object.curr_alloc

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

        updated_alloc = query_model_object.curr_alloc
        res_copy = deepcopy(query_model_object.residual_fwd_neighbors)

        if query_model_object.curr_alloc[r, query] > .5:
            # update the weight of the edge from query to r (should be positive).
            res_copy[query + query_model_object.m][r] = response
        else:
            # update the weight of the edge from r to query (should be negative).
            res_copy[r][query + query_model_object.m] = -response

        cycle = True
        while cycle:
            cycle = spfa(res_copy)

            if cycle is not None:

                # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
                # reviewer, and unassign the i+1'th paper.
                ctr = 0 if cycle[0] < query_model_object.m else 1
                while ctr < len(cycle):
                    paper_to_assign = cycle[(ctr - 1) % len(cycle)] - query_model_object.m
                    paper_to_drop = cycle[(ctr + 1) % len(cycle)] - query_model_object.m
                    curr_rev = cycle[ctr]

                    if paper_to_assign < query_model_object.n:
                        # We are assigning a non-dummy paper to the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_assign] = 1
                        # Reverse the edge and negate its weight
                        res_copy[paper_to_assign + query_model_object.m][curr_rev] = -res_copy[curr_rev][paper_to_assign + query_model_object.m]
                        del res_copy[curr_rev][paper_to_assign + query_model_object.m]

                    if paper_to_drop < query_model_object.n:
                        # We are dropping a non-dummy paper from the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_drop] = 0
                        # Reverse the edge and negate its weight
                        res_copy[curr_rev][paper_to_drop + query_model_object.m] = -res_copy[paper_to_drop + query_model_object.m][curr_rev]
                        del res_copy[paper_to_drop + query_model_object.m][curr_rev]

                    # Update the residual graph if we have dropped the last paper
                    # We need to make it so that curr_rev can't receive the dummy paper anymore.
                    num_papers = np.sum(updated_alloc[curr_rev, :])
                    if num_papers < 0.1:
                        try:
                            del res_copy[curr_rev][query_model_object.n + query_model_object.m]
                        except KeyError:
                            pass
                    # If we have a paper assigned, we can ASSIGN the dummy
                    else:
                        res_copy[curr_rev][query_model_object.n + query_model_object.m] = 0

                    # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
                    # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
                    if num_papers > query_model_object.loads[curr_rev] - .1:
                        try:
                            del res_copy[query_model_object.n + query_model_object.m][curr_rev]
                        except KeyError:
                            pass
                    else:
                        # They can still give away the dummy
                        res_copy[query_model_object.n + query_model_object.m][curr_rev] = 0

                    # Move to the next REVIEWER... not the next vertex in the cycle
                    ctr += 2

        # Ok, so now this should be the best allocation. Check the new value of the expected USW, and make sure it
        # exceeds the value from applying the previous allocation with the new v_tilde.
        updated_expected_value = np.sum(updated_alloc * query_model_object.v_tilde) - \
            query_model_object.v_tilde[r, query] * updated_alloc[r, query] + \
            response * updated_alloc[r, query]

        return updated_expected_value
