import random

from copy import deepcopy
from collections import defaultdict
import math
from multiprocessing import Value, Manager, RawArray, Pool
import functools
import numpy as np
import os
import time

from utils import spfa, spfa_simple, super_algorithm, cycle_beam, spfa_adj_matrix


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

    def __str__(self):
        return "random"


class TpmsQueryModel(QueryModel):
    def get_query(self, reviewer):
        top_papers = np.argsort(self.v_tilde[reviewer, :]).tolist()
        for p in top_papers:
            if p not in self.already_queried[reviewer]:
                return p
        return None

    def __str__(self):
        return "tpms"


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

    def __str__(self):
        return "superstar"


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
            print("Determine value of %d to %d" % (q, reviewer))
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
            print("Expected variance reduction of query %d for reviewer %d is %.4f" % (
            q, reviewer, expected_variance_reduction))
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

    def __str__(self):
        return "var"


#
# proc_manager = Manager()
# shared_max_query_value = proc_manager.Value('d', 0.0)
# buffers = {}
#
# def init_worker(m, n, raw_curr_alloc, raw_v_tilde, raw_loads, raw_adj_matrix):
#     # curr_alloc_buffer = raw_curr_alloc
#     # v_tilde_buffer = raw_v_tilde
#     # loads_buffer = raw_loads
#     st = time.time()
#     buffers['curr_alloc'] = raw_curr_alloc
#     buffers['v_tilde'] = raw_v_tilde
#     buffers['loads'] = raw_loads
#     buffers['adj_matrix'] = raw_adj_matrix
#
#     local_curr_alloc = np.frombuffer(raw_curr_alloc).reshape((m, n), order='C')
#     local_v_tilde = np.frombuffer(raw_v_tilde).reshape((m, n), order='C')
#     local_loads = np.frombuffer(raw_loads)
#     local_adj_matrix = np.frombuffer(raw_adj_matrix).reshape((m+n+1, m+n+1), order='C')
#     print("Time taken in creating buffers per worker: %s s" % (time.time() - st))


# TODO: This won't work for ESW. Need to change the allocation update model and the way I estimate the variance given
# TODO: some allocation.
# class GreedyMaxQueryModelParallel(QueryModel):
#     def __init__(self, tpms, covs, loads, solver, dset_name, data_dir, num_procs):
#         super().__init__(tpms, dset_name)
#         self.solver = solver
#         self.covs = covs
#         self.loads = loads
#         self.num_procs = num_procs
#
#         print("Loading/computing optimal initial solution")
#         try:
#             self.curr_expected_value = np.load(os.path.join(data_dir, "saved_init_expected_usw", dset_name + ".npy"))
#             self.curr_alloc = np.load(os.path.join(data_dir, "saved_init_max_usw_soln", dset_name + ".npy"))
#         except FileNotFoundError:
#             print("Recomputing")
#             os.makedirs(os.path.join(data_dir, "saved_init_expected_usw"), exist_ok=True)
#             os.makedirs(os.path.join(data_dir, "saved_init_max_usw_soln"), exist_ok=True)
#
#             self.curr_expected_value, self.curr_alloc = self.solver(self.v_tilde, self.covs, self.loads)
#             np.save(os.path.join("saved_init_expected_usw", dset_name), self.curr_expected_value)
#             np.save(os.path.join("saved_init_max_usw_soln", dset_name), self.curr_alloc)
#
#     def get_query(self, reviewer):
#         qry_values = {}
#
#         for q in set(range(self.n)) - self.already_queried[reviewer]:
#             print("Determine value of %d to %d" % (q, reviewer), flush=True)
#             # Compute the value of this paper. Return whichever has the best value.
#             # If the paper is not in the current alloc to reviewer, then the alloc won't change if the reviewer bids no
#             # Likewise, if the paper IS in the current alloc, the alloc won't change if the reviewer bids yes.
#
#             # print("Reviewer %d is currently assigned %s" % (reviewer, np.where(self.curr_alloc[reviewer, :])))
#             # Estimate the improvement in expected value for both answers
#             if q in np.where(self.curr_alloc[reviewer, :])[0].tolist():
#                 # print("Update if no")
#                 updated_expected_value_if_no, _ = self._update_alloc(reviewer, q, 0)
#             else:
#                 updated_expected_value_if_no = self.curr_expected_value
#
#             improvement_ub = self.v_tilde[reviewer, q] * (1 - self.v_tilde[reviewer, q]) + self.curr_expected_value
#             max_query_val = max(qry_values.values()) if qry_values else 0
#
#             if qry_values and improvement_ub < max_query_val or math.isclose(improvement_ub, max_query_val):
#                 qry_values[q] = self.curr_expected_value
#             else:
#                 updated_expected_value_if_yes, _ = self._update_alloc(reviewer, q, 1)
#
#                 expected_expected_value = self.v_tilde[reviewer, q] * updated_expected_value_if_yes + \
#                                           (1 - self.v_tilde[reviewer, q]) * updated_expected_value_if_no
#                 print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
#                 qry_values[q] = expected_expected_value
#
#         # print(sorted(qry_values.items(), key=lambda x: -x[1])[:5], sorted(qry_values.items(), key=lambda x: -x[1])[-5:])
#         best_q = [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])][0]
#         return best_q
#
#
#     @staticmethod
#     def check_expected_value(args, mqv):
#         start_time = time.time()
#         q, reviewer, curr_expected_value, m, n = args
#
#         # print(q)
#         # print(flush=True)
#         # print(mqv.value)
#         # print(flush=True)
#
#         # local_curr_alloc = np.frombuffer(curr_alloc).reshape(m, n)
#         # local_v_tilde = np.frombuffer(v_tilde).reshape(m, n)
#         # local_loads = np.frombuffer(loads)
#         local_curr_alloc = np.frombuffer(buffers['curr_alloc']).reshape((m, n), order='C')
#         local_v_tilde = np.frombuffer(buffers['v_tilde']).reshape((m, n), order='C')
#
#         total_time_spent_searching = 0.0
#
#         if q in np.where(local_curr_alloc[reviewer, :])[0].tolist():
#             # print("Update if no")
#             updated_expected_value_if_no, time_spent_searching = GreedyMaxQueryModelParallel._update_alloc_static(reviewer, q, 0, curr_expected_value, m, n)
#             total_time_spent_searching += time_spent_searching
#         else:
#             updated_expected_value_if_no = curr_expected_value
#
#         improvement_ub = local_v_tilde[reviewer, q] * (1 - local_v_tilde[reviewer, q]) + curr_expected_value
#
#         if improvement_ub < mqv.value or math.isclose(improvement_ub, mqv.value):
#             # print("check_expected_values: %s" % (time.time() - start_time), flush=True)
#             return (curr_expected_value, time.time() - start_time, total_time_spent_searching)
#         else:
#             updated_expected_value_if_yes, time_spent_searching = GreedyMaxQueryModelParallel._update_alloc_static(reviewer, q, 1, curr_expected_value, m, n)
#             total_time_spent_searching += time_spent_searching
#
#             expected_expected_value = local_v_tilde[reviewer, q] * updated_expected_value_if_yes + \
#                                       (1 - local_v_tilde[reviewer, q]) * updated_expected_value_if_no
#             # print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
#             if expected_expected_value > mqv.value:
#                 mqv.value = expected_expected_value
#             # print("check_expected_values: %s" % (time.time() - start_time), flush=True)
#             return (expected_expected_value, time.time() - start_time, total_time_spent_searching)
#
#     def get_query_parallel(self, reviewer):
#         papers_to_check = sorted(list(set(range(self.n)) - self.already_queried[reviewer]), key=lambda x: random.random())
#
#         start_time = time.time()
#         l = len(papers_to_check)
#         list_of_copied_args = [papers_to_check,
#                                l * [reviewer],
#                                l * [self.curr_expected_value],
#                                l * [self.m],
#                                l * [self.n]]
#
#         shared_max_query_value.value = 0.0
#
#         raw_curr_alloc = RawArray('d', self.curr_alloc.shape[0] * self.curr_alloc.shape[1])
#         curr_alloc_np = np.frombuffer(raw_curr_alloc).reshape(self.curr_alloc.shape, order='C')
#         np.copyto(curr_alloc_np, self.curr_alloc)
#
#         raw_v_tilde = RawArray('d', self.v_tilde.shape[0] * self.v_tilde.shape[1])
#         v_tilde_np = np.frombuffer(raw_v_tilde).reshape(self.v_tilde.shape, order='C')
#         np.copyto(v_tilde_np, self.v_tilde)
#
#         raw_loads = RawArray('d', self.loads.shape[0])
#         loads_np = np.frombuffer(raw_loads).reshape(self.loads.shape)
#         np.copyto(loads_np, self.loads)
#
#         adj_matrix = np.ones((self.m + self.n + 1, self.m + self.n + 1)) * np.inf
#
#         for reviewer in range(self.m):
#             num_papers = np.sum(self.curr_alloc[reviewer, :])
#             if num_papers > 0.1:
#                 adj_matrix[reviewer, self.n + self.m] = 0
#             if num_papers < self.loads[reviewer] - .1:
#                 adj_matrix[self.n + self.m][reviewer] = 0
#             for paper in range(self.n):
#                 if self.curr_alloc[reviewer, paper] > .5:
#                     adj_matrix[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
#                 else:
#                     adj_matrix[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]
#
#         raw_adj_matrix = RawArray('d', adj_matrix.shape[0] * adj_matrix.shape[1])
#         adj_matrix_np = np.frombuffer(raw_adj_matrix).reshape(adj_matrix.shape, order='C')
#         np.copyto(adj_matrix_np, adj_matrix)
#
#         # local_curr_alloc = curr_alloc
#         # local_max_query_value = max_query_value
#         # local_v_tilde = v_tilde
#         # local_residual_fwd_neighbors = residual_fwd_neighbors
#         # local_loads = loads
#
#         with Pool(processes=self.num_procs, initializer=init_worker, initargs=(self.m, self.n, raw_curr_alloc, raw_v_tilde, raw_loads, raw_adj_matrix)) as pool:
#             expected_expected_values_and_times = pool.map(functools.partial(GreedyMaxQueryModelParallel.check_expected_value,
#                                                 mqv=shared_max_query_value),
#                                                 zip(*list_of_copied_args), 100)
#             expected_expected_values = [x[0] for x in expected_expected_values_and_times]
#             times = [x[1] for x in expected_expected_values_and_times]
#             timesearching = [x[2] for x in expected_expected_values_and_times]
#             print("Total time spent inside check_expected_values: %s" % np.sum(times))
#             print("Total time spent finding cycles: %s" % np.sum(timesearching))
#
#         # print("Average check_expected_value time: %s" % np.mean(times))
#         print("Total time in starting and running check_expected_values: %s" % (time.time() - start_time))
#         indices = np.argsort(expected_expected_values)[::-1].tolist()
#         return [papers_to_check[i] for i in indices]
#
#     def update(self, r, query, response):
#         super().update(r, query, response)
#         old_expected_value = self.curr_expected_value
#         self.curr_expected_value, self.curr_alloc = self._update_alloc(r, query, response)
#         updated = math.isclose(old_expected_value, self.curr_expected_value)
#
#         return updated
#
#     def _update_alloc(self, r, query, response):
#         # We know that if the queried paper is not currently assigned, and its value is 0, the allocation won't change.
#         # print("check value if paper %d for rev %d is %d" % (query, r, response))
#
#         if self.curr_alloc[r, query] < .1 and response == 0:
#             return self.curr_expected_value, self.curr_alloc
#
#         # Otherwise, we need to repeatedly check for augmenting paths in the residual graph
#         # Honestly, I should probably maintain the residual graph at all times
#         # Also, I should first check for augmenting paths coming into/out of the edge we just
#         # queried (oh... or I can just relax basically nm times until I find a negative weight cycle).
#
#         # The residual graph can be represented as a matrix (based on the allocation matrix)
#         # And then I will find an augmenting path by keeping an array that says what the length of the shortest path
#         # is, and an array with the parents for each node.
#
#         # Bipartite graph, with reviewers on left side, papers on right. There is a dummy paper which we will
#         # assign to all reviewers with remaining review load.
#         # We need to have edges with negative v_tilde from paper j to reviewer i when j is assigned to i.
#         # Any unassigned papers have edges from reviewer i to paper j with positive edge weight.
#         # We draw an edge TO the dummy paper when a reviewer has been assigned at least one paper.
#         # We draw an edge FROM the dummy paper when a reviewer still has extra capacity.
#         # TODO: once this whole thing is implemented, I should also make sure that the suggested updates are valid.
#
#         # Use the shortest path faster algorithm to find negative weight cycles, until there aren't any.
#         # https://konaeakira.github.io/posts/using-the-shortest-path-faster-algorithm-to-find-negative-cycles.html
#
#         updated_alloc = self.curr_alloc.copy()
#         m = self.m
#         n = self.n
#
#         rfn = dict()
#         for r in range(m):
#             rfn[r] = dict()
#         for p in range(n + 1):
#             rfn[p + m] = dict()
#
#         for reviewer in range(m):
#             num_papers = np.sum(self.curr_alloc[reviewer, :])
#             if num_papers > 0.1:
#                 rfn[reviewer][n + m] = 0
#             if num_papers < self.loads[reviewer] - .1:
#                 rfn[n + m][reviewer] = 0
#             for paper in range(n):
#                 if self.curr_alloc[reviewer, paper] > .5:
#                     rfn[paper + m][reviewer] = self.v_tilde[reviewer, paper]
#                 else:
#                     rfn[reviewer][paper + m] = -self.v_tilde[reviewer, paper]
#         res_copy = rfn
#
#         if self.curr_alloc[r, query] > .5:
#             # update the weight of the edge from query to r (should be positive).
#             res_copy[query + self.m][r] = response
#         else:
#             # update the weight of the edge from r to query (should be negative).
#             res_copy[r][query + self.m] = -response
#
#
#         # print("curr_alloc")
#         # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
#         #     print("%d: %s" % (rev, np.where(self.curr_alloc[rev, :])[0].tolist()))
#
#         cycle = True
#         src_set = {r, query}
#
#         while cycle:
#             # print("SPFA start")
#             # cycle = spfa(res_copy)
#             cycle = spfa_simple(res_copy, src_set)
#             # print(cycle)
#
#             if cycle is not None:
#                 src_set |= set(cycle)
#                 # for i in cycle:
#                 #     print(i)
#                 #     print(res_copy[i])
#                 # update the allocation and residual graph using the cycle
#
#                 # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
#                 # reviewer, and unassign the i+1'th paper.
#                 ctr = 0 if cycle[0] < self.m else 1
#                 while ctr < len(cycle):
#                     paper_to_assign = cycle[(ctr - 1) % len(cycle)] - self.m
#                     paper_to_drop = cycle[(ctr + 1) % len(cycle)] - self.m
#                     curr_rev = cycle[ctr]
#
#                     # print("Remove paper %d from reviewer %d, and add paper %d" % (paper_to_drop, curr_rev, paper_to_assign))
#                     # print("Gain: %.2f, Loss: %.2f" % (res_copy[curr_rev][paper_to_assign + self.m], res_copy[paper_to_drop + self.m][curr_rev]))
#
#                     if paper_to_assign < self.n:
#                         # We are assigning a non-dummy paper to the reviewer curr_rev
#                         updated_alloc[curr_rev, paper_to_assign] = 1
#                         # Reverse the edge and negate its weight
#                         res_copy[paper_to_assign + self.m][curr_rev] = -res_copy[curr_rev][paper_to_assign + self.m]
#                         del res_copy[curr_rev][paper_to_assign + self.m]
#
#                     if paper_to_drop < self.n:
#                         # We are dropping a non-dummy paper from the reviewer curr_rev
#                         updated_alloc[curr_rev, paper_to_drop] = 0
#                         # Reverse the edge and negate its weight
#                         res_copy[curr_rev][paper_to_drop + self.m] = -res_copy[paper_to_drop + self.m][curr_rev]
#                         del res_copy[paper_to_drop + self.m][curr_rev]
#
#                     # Update the residual graph if we have dropped the last paper
#                     # We need to make it so that curr_rev can't receive the dummy paper anymore.
#                     num_papers = np.sum(updated_alloc[curr_rev, :])
#                     if num_papers < 0.1:
#                         try:
#                             del res_copy[curr_rev][self.n + self.m]
#                         except KeyError:
#                             pass
#                     # If we have a paper assigned, we can ASSIGN the dummy
#                     else:
#                         res_copy[curr_rev][self.n + self.m] = 0
#
#                     # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
#                     # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
#                     if num_papers > self.loads[curr_rev] - .1:
#                         try:
#                             del res_copy[self.n + self.m][curr_rev]
#                         except KeyError:
#                             pass
#                     else:
#                         # They can still give away the dummy
#                         res_copy[self.n + self.m][curr_rev] = 0
#
#                     # Move to the next REVIEWER... not the next vertex in the cycle
#                     ctr += 2
#
#         # Ok, so now this should be the best allocation. Check the new value of the expected USW, and make sure it
#         # exceeds the value from applying the previous allocation with the new v_tilde.
#         updated_expected_value = np.sum(updated_alloc * self.v_tilde) - \
#             self.v_tilde[r, query] * updated_alloc[r, query] + \
#             response * updated_alloc[r, query]
#
#         updated_expected_value_if_using_old_alloc = np.sum(self.curr_alloc * self.v_tilde) - \
#             self.v_tilde[r, query] * self.curr_alloc[r, query] + \
#             response * self.curr_alloc[r, query]
#         # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
#         #     print("%d: %s" % (rev, np.where(updated_alloc[rev, :])[0].tolist()))
#         #
#         # print("We should expected new EV (%s) to be equal to old EV (%s) plus negative total gain (%s)" % (updated_expected_value, updated_expected_value_if_using_old_alloc, sum_of_gains))
#         # print("new - (old - gain) = %s" % (updated_expected_value - (updated_expected_value_if_using_old_alloc - sum_of_gains)))
#
#         if updated_expected_value_if_using_old_alloc > updated_expected_value:
#             print("ERROR")
#             print("PROBABLY AN ERROR, THIS SHOULDNT BE HAPPENING")
#             print(updated_expected_value_if_using_old_alloc)
#             print(updated_expected_value)
#             print(np.isclose(updated_expected_value_if_using_old_alloc, updated_expected_value))
#         # if updated_expected_value_if_using_old_alloc < updated_expected_value:
#         #     print("Improved expected value")
#
#         # TODO: Compute the updated expected value only at the very end, so we save that computation at least...
#         # TODO: Anyway, I think sometimes it isn't even necessary and can be safely omitted.
#
#         return updated_expected_value, updated_alloc
#
#     @staticmethod
#     def _update_alloc_static(r, query, response, curr_expected_value, m, n):
#         # We know that if the queried paper is not currently assigned, and its value is 0, the allocation won't change.
#         # print("check value if paper %d for rev %d is %d" % (query, r, response))
#
#         curr_alloc = np.frombuffer(buffers['curr_alloc']).reshape((m, n), order='C')
#         v_tilde = np.frombuffer(buffers['v_tilde']).reshape((m, n), order='C')
#         loads = np.frombuffer(buffers['loads'])
#         adj_matrix = np.frombuffer(buffers['adj_matrix']).reshape((m+n+1, m+n+1), order='C').copy()
#
#         if curr_alloc[r, query] < .1 and response == 0:
#             return curr_expected_value
#
#         touched_nodes = set()
#
#         # Otherwise, we need to repeatedly check for augmenting paths in the residual graph
#         # Honestly, I should probably maintain the residual graph at all times
#         # Also, I should first check for augmenting paths coming into/out of the edge we just
#         # queried (oh... or I can just relax basically nm times until I find a negative weight cycle).
#
#         # The residual graph can be represented as a matrix (based on the allocation matrix)
#         # And then I will find an augmenting path by keeping an array that says what the length of the shortest path
#         # is, and an array with the parents for each node.
#
#         # Bipartite graph, with reviewers on left side, papers on right. There is a dummy paper which we will
#         # assign to all reviewers with remaining review load.
#         # We need to have edges with negative v_tilde from paper j to reviewer i when j is assigned to i.
#         # Any unassigned papers have edges from reviewer i to paper j with positive edge weight.
#         # We draw an edge TO the dummy paper when a reviewer has been assigned at least one paper.
#         # We draw an edge FROM the dummy paper when a reviewer still has extra capacity.
#         # TODO: once this whole thing is implemented, I should also make sure that the suggested updates are valid.
#
#         # Use the shortest path faster algorithm to find negative weight cycles, until there aren't any.
#         # https://konaeakira.github.io/posts/using-the-shortest-path-faster-algorithm-to-find-negative-cycles.html
#
#         updated_alloc = curr_alloc.copy()
#
#         if curr_alloc[r, query] > .5:
#             # update the weight of the edge from query to r (should be positive).
#             adj_matrix[query + m][r] = response
#             touched_nodes.add(query+m)
#         else:
#             # update the weight of the edge from r to query (should be negative).
#             adj_matrix[r][query + m] = -response
#             touched_nodes.add(r)
#
#         cycle = True
#         num_iters = 0
#         iter_bd = 15
#         # src_set = {r, query}
#
#         time_spent_searching = 0.0
#
#         while cycle and num_iters < iter_bd:
#             num_iters += 1
#
#             st = time.time()
#             # depth_limit = 10
#             # cycle = spfa_simple(res_copy, src_set, depth_limit)
#             b = 3
#             d = 10
#             beamwidth = 10
#
#             # We have to start searching from the paper when response is 0,
#             # and start from the reviewer when response is 1
#             if response == 0:
#                 cycle = cycle_beam(adj_matrix, query, b, d, beamwidth)
#             elif response == 1:
#                 cycle = cycle_beam(adj_matrix, r, b, d, beamwidth)
#             if cycle is not None:
#                 cycle = cycle[::-1]
#
#             time_spent_searching += time.time() - st
#
#             if cycle is not None:
#                 cycle_set = set(cycle)
#                 touched_nodes |= cycle_set
#
#                 # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
#                 # reviewer, and unassign the i+1'th paper.
#                 ctr = 0 if cycle[0] < m else 1
#                 while ctr < len(cycle):
#                     paper_to_assign = cycle[(ctr - 1) % len(cycle)] - m
#                     paper_to_drop = cycle[(ctr + 1) % len(cycle)] - m
#                     curr_rev = cycle[ctr]
#
#                     if paper_to_assign < n:
#                         # We are assigning a non-dummy paper to the reviewer curr_rev
#                         updated_alloc[curr_rev, paper_to_assign] = 1
#                         # Reverse the edge and negate its weight
#                         adj_matrix[paper_to_assign + m][curr_rev] = -adj_matrix[curr_rev][paper_to_assign + m]
#                         adj_matrix[curr_rev][paper_to_assign + m] = np.inf
#
#                     if paper_to_drop < n:
#                         # We are dropping a non-dummy paper from the reviewer curr_rev
#                         updated_alloc[curr_rev, paper_to_drop] = 0
#                         # Reverse the edge and negate its weight
#                         adj_matrix[curr_rev][paper_to_drop + m] = -adj_matrix[paper_to_drop + m][curr_rev]
#                         adj_matrix[paper_to_drop + m][curr_rev] = np.inf
#
#                     # Update the residual graph if we have dropped the last paper
#                     # We need to make it so that curr_rev can't receive the dummy paper anymore.
#                     num_papers = np.sum(updated_alloc[curr_rev, :])
#                     if num_papers < 0.1:
#                         adj_matrix[curr_rev][n + m] = np.inf
#
#                     # If we have a paper assigned, we can ASSIGN the dummy
#                     else:
#                         adj_matrix[curr_rev][n + m] = 0
#
#                     # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
#                     # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
#                     if num_papers > loads[curr_rev] - .1:
#                         adj_matrix[n + m][curr_rev] = np.inf
#                     else:
#                         # They can still give away the dummy
#                         adj_matrix[n + m][curr_rev] = 0
#
#                     # Move to the next REVIEWER... not the next vertex in the cycle
#                     ctr += 2
#
#         # total_time = time.time() - start_time
#         # print("%d iters, %s s" % (num_iters, total_time))
#
#         # Ok, so now this should be the best allocation. Check the new value of the expected USW, and make sure it
#         # exceeds the value from applying the previous allocation with the new v_tilde.
#         updated_expected_value = np.sum(updated_alloc * v_tilde) - \
#             v_tilde[r, query] * updated_alloc[r, query] + \
#             response * updated_alloc[r, query]
#
#         # Fix the residual forward neighbors so it is like it was before.
#         # for node in touched_nodes:
#         #     if node < m:
#         #         # It's a rev
#         #         adj_matrix[node, :] = np.inf
#         #         num_papers = np.sum(curr_alloc[node, :])
#         #         if num_papers > 0.1:
#         #             adj_matrix[node][n + m] = 0
#         #         elif adj_matrix[node][n+m] < np.inf:
#         #             adj_matrix[node][n + m] = np.inf
#         #         if num_papers < loads[node] - .1:
#         #             adj_matrix[n + m][node] = 0
#         #         elif adj_matrix[n + m][node] < np.inf:
#         #             adj_matrix[n + m][node] = np.inf
#         #
#         #         for paper in range(n):
#         #             if curr_alloc[node, paper] > .5:
#         #                 adj_matrix[paper + m][node] = v_tilde[node, paper]
#         #                 if adj_matrix[node][paper+m] < np.inf:
#         #                     adj_matrix[node][paper+m] = np.inf
#         #             else:
#         #                 adj_matrix[node][paper + m] = -v_tilde[node, paper]
#         #                 if adj_matrix[paper+m][node] < np.inf:
#         #                     adj_matrix[paper+m][node] = np.inf
#
#         return updated_expected_value, time_spent_searching


class GreedyMaxQueryModel(QueryModel):
    def __init__(self, tpms, covs, loads, solver, dset_name):
        super().__init__(tpms, dset_name)
        self.solver = solver
        self.covs = covs
        self.loads = loads
        self.num_procs = 20
        # A pool for running the updates in multiple threads

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
        adj_matrix = np.ones((self.m + self.n + 1, self.m + self.n + 1)) * np.inf

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                adj_matrix[reviewer, self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                adj_matrix[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    adj_matrix[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    adj_matrix[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]
        self.adj_matrix = adj_matrix

    def get_query(self, reviewer):
        qry_values = {}

        for q in set(range(self.n)) - self.already_queried[reviewer]:
            # print("Determine value of %d to %d" % (q, reviewer))
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
                # print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
                qry_values[q] = expected_expected_value

        # print(sorted(qry_values.items(), key=lambda x: -x[1])[:5], sorted(qry_values.items(), key=lambda x: -x[1])[-5:])
        best_q = [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])][0]
        return best_q

    def update(self, r, query, response):
        super().update(r, query, response)
        self.curr_expected_value, self.curr_alloc = self._update_alloc(r, query, response)

        # print("Setting up residual graph")
        adj_matrix = np.ones((self.m + self.n + 1, self.m + self.n + 1)) * np.inf

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                adj_matrix[reviewer, self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                adj_matrix[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    adj_matrix[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    adj_matrix[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]
        self.adj_matrix = adj_matrix

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
        adj_matrix = self.adj_matrix.copy()

        if self.curr_alloc[r, query] > .5:
            # update the weight of the edge from query to r (should be positive).
            adj_matrix[query + self.m][r] = response
        else:
            # update the weight of the edge from r to query (should be negative).
            adj_matrix[r][query + self.m] = -response

        # print("curr_alloc")
        # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
        #     print("%d: %s" % (rev, np.where(self.curr_alloc[rev, :])[0].tolist()))

        cycle = True
        while cycle:
            # print("SPFA start")
            # cycle = spfa_adj_matrix(adj_matrix)
            if response == 0:
                cycle = cycle_beam(adj_matrix, query, 3, 10, 10)
            else:
                cycle = cycle_beam(adj_matrix, r, 3, 10, 10)
            if cycle is not None:
                cycle = cycle[::-1]
            # print(cycle)

            if cycle is not None:
                # for i in range(len(cycle)):
                #     print(cycle[i])
                #     print(adj_matrix[cycle[(i+1)%len(cycle)], cycle[i]])

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
                        adj_matrix[paper_to_assign + self.m][curr_rev] = -adj_matrix[curr_rev][paper_to_assign + self.m]
                        adj_matrix[curr_rev][paper_to_assign + self.m] = np.inf

                    if paper_to_drop < self.n:
                        # We are dropping a non-dummy paper from the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_drop] = 0
                        # Reverse the edge and negate its weight
                        adj_matrix[curr_rev][paper_to_drop + self.m] = -adj_matrix[paper_to_drop + self.m][curr_rev]
                        adj_matrix[paper_to_drop + self.m][curr_rev] = np.inf

                    # Update the residual graph if we have dropped the last paper
                    # We need to make it so that curr_rev can't receive the dummy paper anymore.
                    num_papers = np.sum(updated_alloc[curr_rev, :])
                    if num_papers < 0.1:
                        adj_matrix[curr_rev][self.n + self.m] = np.inf
                    # If we have a paper assigned, we can ASSIGN the dummy
                    else:
                        adj_matrix[curr_rev][self.n + self.m] = 0

                    # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
                    # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
                    if num_papers > self.loads[curr_rev] - .1:
                        adj_matrix[self.n + self.m][curr_rev] = np.inf
                    else:
                        # They can still give away the dummy
                        adj_matrix[self.n + self.m][curr_rev] = 0

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

    def __str__(self):
        return "greedymax"


class SuperStarGreedyMaxQueryModel(QueryModel):
    def __init__(self, tpms, covs, loads, solver, dset_name, k):
        super().__init__(tpms, dset_name)
        self.solver = solver
        self.covs = covs
        self.loads = loads
        self.tpms_orig = tpms.copy()
        self.bids = np.zeros(tpms.shape)
        self.k = k

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
        adj_matrix = np.ones((self.m + self.n + 1, self.m + self.n + 1)) * np.inf

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                adj_matrix[reviewer, self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                adj_matrix[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    adj_matrix[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    adj_matrix[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]
        self.adj_matrix = adj_matrix

    def get_query(self, reviewer):
        qry_values = {}

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
        to_search = []
        for p in top_papers:
            if p not in self.already_queried[reviewer]:
                to_search.append(p)

        for q in to_search[:self.k]:
            # print("Determine value of %d to %d" % (q, reviewer))
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
                # print("Expected expected value of query %d for reviewer %d is %.4f" % (q, reviewer, expected_expected_value))
                qry_values[q] = expected_expected_value

        # print(sorted(qry_values.items(), key=lambda x: -x[1])[:5], sorted(qry_values.items(), key=lambda x: -x[1])[-5:])
        best_q = [x[0] for x in sorted(qry_values.items(), key=lambda x: -x[1])][0]
        return best_q

    def update(self, r, query, response):
        super().update(r, query, response)
        self.curr_expected_value, self.curr_alloc = self._update_alloc(r, query, response)

        self.bids[r, query] = response

        # print("Setting up residual graph")
        adj_matrix = np.ones((self.m + self.n + 1, self.m + self.n + 1)) * np.inf

        for reviewer in range(self.m):
            num_papers = np.sum(self.curr_alloc[reviewer, :])
            if num_papers > 0.1:
                adj_matrix[reviewer, self.n + self.m] = 0
            if num_papers < self.loads[reviewer] - .1:
                adj_matrix[self.n + self.m][reviewer] = 0
            for paper in range(self.n):
                if self.curr_alloc[reviewer, paper] > .5:
                    adj_matrix[paper + self.m][reviewer] = self.v_tilde[reviewer, paper]
                else:
                    adj_matrix[reviewer][paper + self.m] = -self.v_tilde[reviewer, paper]
        self.adj_matrix = adj_matrix

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
        adj_matrix = self.adj_matrix.copy()

        if self.curr_alloc[r, query] > .5:
            # update the weight of the edge from query to r (should be positive).
            adj_matrix[query + self.m][r] = response
        else:
            # update the weight of the edge from r to query (should be negative).
            adj_matrix[r][query + self.m] = -response

        # print("curr_alloc")
        # for rev in sorted([22, 50, 37, 109, 127, 108, 146, 152, 19]):
        #     print("%d: %s" % (rev, np.where(self.curr_alloc[rev, :])[0].tolist()))

        cycle = True
        while cycle:
            # print("SPFA start")
            # cycle = spfa_adj_matrix(adj_matrix)
            if response == 0:
                cycle = cycle_beam(adj_matrix, query, 3, 10, 10)
            else:
                cycle = cycle_beam(adj_matrix, r, 3, 10, 10)
            if cycle is not None:
                cycle = cycle[::-1]
            # print(cycle)

            if cycle is not None:
                # for i in range(len(cycle)):
                #     print(cycle[i])
                #     print(adj_matrix[cycle[(i+1)%len(cycle)], cycle[i]])

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
                        adj_matrix[paper_to_assign + self.m][curr_rev] = -adj_matrix[curr_rev][
                            paper_to_assign + self.m]
                        adj_matrix[curr_rev][paper_to_assign + self.m] = np.inf

                    if paper_to_drop < self.n:
                        # We are dropping a non-dummy paper from the reviewer curr_rev
                        updated_alloc[curr_rev, paper_to_drop] = 0
                        # Reverse the edge and negate its weight
                        adj_matrix[curr_rev][paper_to_drop + self.m] = -adj_matrix[paper_to_drop + self.m][curr_rev]
                        adj_matrix[paper_to_drop + self.m][curr_rev] = np.inf

                    # Update the residual graph if we have dropped the last paper
                    # We need to make it so that curr_rev can't receive the dummy paper anymore.
                    num_papers = np.sum(updated_alloc[curr_rev, :])
                    if num_papers < 0.1:
                        adj_matrix[curr_rev][self.n + self.m] = np.inf
                    # If we have a paper assigned, we can ASSIGN the dummy
                    else:
                        adj_matrix[curr_rev][self.n + self.m] = 0

                    # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
                    # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
                    if num_papers > self.loads[curr_rev] - .1:
                        adj_matrix[self.n + self.m][curr_rev] = np.inf
                    else:
                        # They can still give away the dummy
                        adj_matrix[self.n + self.m][curr_rev] = 0

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

    def __str__(self):
        return "supergreedymax"
