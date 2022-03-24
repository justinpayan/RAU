import lap
import math
import numpy as np
import os

from queue import Queue


def load_dset(dname, data_dir="."):
    tpms = np.load(os.path.join(data_dir, "data", dname, "scores.npy"))
    covs = np.load(os.path.join(data_dir, "data", dname, "covs.npy"))
    loads = np.load(os.path.join(data_dir, "data", dname, "loads.npy"))

    np.random.seed(31415)

    # Sample the "true" bids that would occur if reviewers bid on all papers.
    noisy_tpms = tpms + np.random.randn(*tpms.shape) * 0.1
    noisy_tpms = np.clip(noisy_tpms, 0, 1)
    true_bids = np.random.uniform(size=tpms.shape) < noisy_tpms

    return tpms, true_bids, covs, loads


# Use the shortest path faster algorithm to find a negative weight cycle if it exists.
# https://konaeakira.github.io/posts/using-the-shortest-path-faster-algorithm-to-find-negative-cycles.html
def spfa(fwd_adj_list):
    # print(fwd_adj_list[50])
    # print(fwd_adj_list[177])
    # print(fwd_adj_list[22])
    # print(fwd_adj_list[225])
    dis = [0] * len(fwd_adj_list)
    pre = [-1] * len(fwd_adj_list)
    vertex_queue = Queue()
    vertex_queue_set = set()
    for v in range(len(fwd_adj_list)):
        vertex_queue.put(v)
        vertex_queue_set.add(v)
    i = 0
    while vertex_queue.qsize():
        u = vertex_queue.get()
        vertex_queue_set.remove(u)
        for v in fwd_adj_list[u]:
            if dis[u] + fwd_adj_list[u][v] < dis[v] and not math.isclose(dis[u] + fwd_adj_list[u][v], dis[v]):
                pre[v] = u
                dis[v] = dis[u] + fwd_adj_list[u][v]
                i += 1
                if i == len(fwd_adj_list):
                    i = 0
                    cyc = detect_cycle(pre)
                    if cyc:
                        return cyc
                if v not in vertex_queue_set:
                    vertex_queue.put(v)
                    vertex_queue_set.add(v)
    cyc = detect_cycle(pre)
    if cyc:
        return cyc
    return None


def spfa_simple(fwd_adj_list, src_set):
    dis = [0] * len(fwd_adj_list)
    pre = [-1] * len(fwd_adj_list)
    vertex_queue = Queue()
    vertex_queue_set = set()
    # for v in range(len(fwd_adj_list)):
    #     vertex_queue.put(v)
    #     vertex_queue_set.add(v)
    for v in src_set:
        vertex_queue.put(v)
        vertex_queue_set.add(v)

    while vertex_queue.qsize():
        u = vertex_queue.get()
        vertex_queue_set.remove(u)
        for v in fwd_adj_list[u]:
            if dis[u] + fwd_adj_list[u][v] < dis[v] and not math.isclose(dis[u] + fwd_adj_list[u][v], dis[v]):
                pre[v] = u
                dis[v] = dis[u] + fwd_adj_list[u][v]
                if v in src_set:
                    cyc = detect_cycle(pre)
                    if cyc:
                        return cyc
                # if i == len(fwd_adj_list):
                #     i = 0
                #     cyc = detect_cycle(pre)
                #     if cyc:
                #         return cyc
                if v not in vertex_queue_set:
                    vertex_queue.put(v)
                    vertex_queue_set.add(v)
    if v in src_set:
        cyc = detect_cycle(pre)
        if cyc:
            return cyc

    return None


def spfa_adj_matrix(adj_matrix):
    dis = [0] * adj_matrix.shape[0]
    pre = [-1] * adj_matrix.shape[0]
    vertex_queue = Queue()
    vertex_queue_set = set()
    for v in range(adj_matrix.shape[0]):
        vertex_queue.put(v)
        vertex_queue_set.add(v)
    i = 0
    while vertex_queue.qsize():
        u = vertex_queue.get()
        vertex_queue_set.remove(u)
        for v in np.where(adj_matrix[u, :] < np.inf)[0].tolist():
            if dis[u] + adj_matrix[u][v] < dis[v] and not math.isclose(dis[u] + adj_matrix[u][v], dis[v]):
                pre[v] = u
                dis[v] = dis[u] + adj_matrix[u][v]
                i += 1
                if i == adj_matrix.shape[0]:
                    i = 0
                    cyc = detect_cycle(pre)
                    if cyc:
                        return cyc
                if v not in vertex_queue_set:
                    vertex_queue.put(v)
                    vertex_queue_set.add(v)
    cyc = detect_cycle(pre)
    if cyc:
        return cyc
    return None


# Detect a cycle in the graph using the precursor list. How is this done? I think via some kind of DFS? Oh, right,
# you keep a list of nodes you have visited, and you traverse the precursor edges. If one of these traversals ends up
# back where it started, you have a cycle. If you get to the end of pre and you haven't revisited anything,
# you're good.
def detect_cycle(pre):
    # print("\n\ndetecting a cycle")
    visited_overall = set()
    for u, pre_u in enumerate(pre):
        # print("u, pre_u", u, pre_u)
        if pre_u != -1 and u not in visited_overall:
            # Begin a DFS, backward using the precursor list
            visited_overall.add(u)
            visited_this_pass = {u}
            while pre_u != -1:
                # print("u, pre_u", u, pre_u)
                u, pre_u = pre_u, pre[pre_u]
                # print("NEW u, pre_u", u, pre_u)

                visited_overall.add(u)
                visited_this_pass.add(u)
                if pre_u != -1 and pre_u in visited_this_pass:
                    # We found a cycle. We need to traverse the cycle again to print it out though
                    cycle_set = set()
                    cycle = []
                    while pre_u not in cycle_set:
                        cycle.append(pre_u)
                        cycle_set.add(pre_u)
                        u, pre_u = pre_u, pre[pre_u]
                    # print("cycle detected: ", cycle)
                    return cycle
    return None


def cycle_beam(adj_matrix, start_node, b, d, beamwidth):
    # adj_matrix is [m + (n+1)] x [m + (n+1)]. It has the cost of an edge when an edge exists, otherwise np.inf

    # Go ahead and argsort the adj_matrix, so we know the best edges for each node
    best_nbrs = np.argsort(adj_matrix, axis=1)

    def expand_path(path, b, best_nbrs, adj_matrix, start_node):
        final_node = path[-1]
        next_nodes = []
        costs = []
        nbr_idx = 0
        found_start = []

        while len(next_nodes) < b and \
            nbr_idx < best_nbrs.shape[1] and \
            adj_matrix[final_node, best_nbrs[final_node, nbr_idx]] < np.inf:

            v = best_nbrs[final_node, nbr_idx]
            if v not in path:
                next_nodes.append(v)
                costs.append(adj_matrix[final_node, v])
                found_start.append(False)
            elif v == start_node:
                next_nodes.append(v)
                costs.append(adj_matrix[final_node, start_node])
                found_start.append(True)

            nbr_idx += 1
        return next_nodes, costs, found_start

    # Search forward and backward (basically, forward through the transposed matrix) from the start_node
    # The beam holds 1) the path, 2) the cost of the path, and 3) whether or not the path has reached the start_node
    # At the end of a full iteration, we will check all the paths in the beam. If any path has reached the start
    # and has a negative cost, we can return such a cycle with lowest cost. If not, then we truncate the beam
    # and move to the next iteration.
    beam = [[[start_node], 0, {start_node}]]
    for _ in range(d):
        new_beam = []
        check_for_term = set()
        for path_idx in range(len(beam)):
            # print(beam[path_idx])
            # print(beam[path_idx][0])
            next_nodes, costs, found_start = \
                expand_path(beam[path_idx][0], b, best_nbrs, adj_matrix, start_node)

            # Append all the new paths to the new_beam.
            curr_cost = beam[path_idx][1]
            for n_idx in range(len(next_nodes)):
                if found_start[n_idx]:
                    check_for_term.add(len(new_beam))
                new_beam.append([beam[path_idx][0] + [next_nodes[n_idx]], curr_cost + costs[n_idx]])
            # print("new_beam: ", new_beam)


        # If any of the paths got back to the start, we should check which of them have negative costs (if any) and
        # return the best one.
        if check_for_term:
            cycs = [new_beam[i] for i in check_for_term]
            lowest_cost = np.inf
            best_cyc = -1
            for c_idx, c in enumerate(cycs):
                if c[1] < lowest_cost:
                    lowest_cost = c[1]
                    best_cyc = c_idx
            if lowest_cost < 0:
                return cycs[best_cyc][0][:-1]
            # If none of the cycles have negative cost, then we can just remove all of them from the beam.
            else:
                tmp = []
                for idx, path in enumerate(new_beam):
                    if idx not in check_for_term:
                        tmp.append(path)
                new_beam = tmp

        # Sort the new beam and truncate.
        beam = sorted(new_beam, key=lambda x: x[1])[:beamwidth]

    return None


def super_algorithm(g_p, g_r, f, s, bids, h, trade_param, special=False):
    """
    Copied from https://github.com/fiezt/Peer-Review-Bidding/blob/master/CODE/SUPER_Algorithm.ipynb.

    See the paper A SUPER* Algorithm to Optimize Paper Bidding in Peer Review

    Solve for a paper ordering for a reviewer using SUPER* procedure.

    This procedure requires numpy and the lap package https://github.com/gatagat/lap
    to solve the linear assignment problem.

    :param g_p (function): Paper-side gain function mapping bid counts to a score.
    The score function should be non-decreasing in the number of bids.
    The function should handle the bid count input as an array containing
    the number of bids for each paper ordered by the paper index or the
    bid count input as a number for a fixed paper.

    Ex/ g_p = lambda bids: np.sqrt(bids)

    :param g_r (function): Reviewer-side gain function mapping similarity score and paper position to a score.
    The score function should be non-decreasing in the similarity score and and non-increasing in the paper position.
    The function should handle the similarity score input and the paper position input as arrays containing the
    similarity scores and paper positions for each paper ordered by the paper index or the similarity score and
    paper position for a fixed paper.

    Ex/ g_r = lambda s, pi: (2**s - 1)/np.log2(pi + 1)

    :param f (function): Bidding function mapping similarity score and paper position to a bid probability.
    The function should be non-decreasing in the similarity score and non-increasing in the paper position.
    The function should handle the similarity score imput and the paper position input as arrays containing
    the similarity scores and paper positions for each paper ordered by the paper index or the similarity score
    and paper position for a fixed paper.

    Ex/ f = lambda s, pi: s/np.log2(pi + 1)

    :param s (array): Similarity scores for each paper ordered by paper index.

    :param bids (array): Number of bids for each paper ordered by the paper index prior to the arrival of the reviewer.

    :param h (array): Heuristic values estimating the number of bids for each paper in the future ordered by the paper index.

    :param trade_param (float): Parameter dictating the weight given to the reviewer-side gain function.

    :param special (bool): indicator for the ability to use computationally effiicent solution.
    If the reviewer-side gain function is multiplicatively separable into the form g_r(s, pi) = g_r_s(s)f_p(pi)
    where g_r_s is a non-decreasing function of the similarity score and f_p is the non-increasing bidding function
    of the position a paper is shown and similarly the bidding function can be decomposed into the form
    f(s, pi) = f_s(s)f_p(pi), then a simple sorting routine can be used instead of the linear program. To run the sorting
    procedure, the functions g_r and f should take in the paper ordering as an optional argument.

    Ex/ If g_r(s, pi) = (2**s - 1)/np.log2(pi + 1) and f(s, pi) = s/np.log2(pi + 1), then special can be set and
    define g_r(s, pi=None) = if pi is None: (2**s - 1) else: (2**s - 1)/np.log2(pi + 1) and similarly define
    f(s, pi=None) = if pi is None: s else: s/np.log2(pi + 1).

    return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example,
    pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1.
    """

    d = len(s)

    if not special:
        # Solve linear assignment problem to get ordering to present.
        w_p = lambda j, k: f(s[j], k) * (g_p(bids[j] + h[j] + 1) - g_p(bids[j] + h[j]))
        w_r = lambda j, k: trade_param * g_r(s[j], k)
        w = np.array([w_p(j, np.arange(1, d + 1)) + w_r(j, np.arange(1, d + 1)) for j in range(d)])
        pi_t = lap.lapjv(-w)[1]
        pi_t += 1
    else:
        # Rank papers from maximum to minimum for alpha breaking ties by the similarity score.
        alpha = f(s) * (g_p(bids + h + 1) - g_p(bids + h)) + (trade_param * g_r(s))
        alpha_pairs = np.array(list(zip(alpha, np.arange(1, d + 1))), dtype=[('alpha', float), ('index', float)])
        pi_t = np.argsort(np.lexsort((alpha_pairs['index'], -alpha_pairs['alpha']))) + 1

    return pi_t


if __name__ == '__main__':
    # print(detect_cycle([-1, 0, 1]))
    # print(detect_cycle([3, 0, 1, 2]))

    # print(-1 % 10)

    m = 3
    n = 2

    residual_fwd_neighbors = {
        # Reviewers
        0: {4: -.6, 5: 0},
        1: {3: 0, 5: 0},
        2: {3: -.7, 4: -.7},

        # Papers
        3: {0: .5},
        4: {1: 1},
        5: {2: 0}
    }

    # I think that spfa should find the cycle 0, 5, 2, 3
    print(spfa(residual_fwd_neighbors))