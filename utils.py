import lap
import math
import numpy as np
import os

from queue import Queue


def load_dset(dname):
    tpms = np.load(os.path.join("data", dname, "scores.npy"))
    covs = np.load(os.path.join("data", dname, "covs.npy"))
    loads = np.load(os.path.join("data", dname, "loads.npy"))

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


def spfa_simple(fwd_adj_list, src_rev):
    dis = [0] * len(fwd_adj_list)
    pre = [-1] * len(fwd_adj_list)
    vertex_queue = Queue()
    vertex_queue_set = set()
    # for v in range(len(fwd_adj_list)):
    #     vertex_queue.put(v)
    #     vertex_queue_set.add(v)
    vertex_queue.put(src_rev)
    vertex_queue_set.add(src_rev)

    while vertex_queue.qsize():
        u = vertex_queue.get()
        vertex_queue_set.remove(u)
        for v in fwd_adj_list[u]:
            if dis[u] + fwd_adj_list[u][v] < dis[v] and not math.isclose(dis[u] + fwd_adj_list[u][v], dis[v]):
                pre[v] = u
                dis[v] = dis[u] + fwd_adj_list[u][v]
                if v == src_rev:
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
    if v == src_rev:
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