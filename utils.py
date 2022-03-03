import numpy as np
import os

from queue import Queue


def load_dset(dname):
    print(os.getcwd())
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
            if dis[u] + fwd_adj_list[u][v] < dis[v]:
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


# Detect a cycle in the graph using the precursor list. How is this done? I think via some kind of DFS? Oh, right,
# you keep a list of nodes you have visited, and you traverse the precursor edges. If one of these traversals ends up
# back where it started, you have a cycle. If you get to the end of pre and you haven't revisited anything,
# you're good.
def detect_cycle(pre):
    visited_overall = set()
    for u, pre_u in enumerate(pre):
        if pre_u != -1 and u not in visited_overall:
            # Begin a DFS, backward using the precursor list
            visited_overall.add(u)
            visited_this_pass = {u}
            visited_list = [u]
            while pre_u != -1:
                u, pre_u = pre_u, pre[pre_u]
                visited_overall.add(u)
                visited_this_pass.add(u)
                visited_list.append(u)
                if pre_u != -1 and pre_u in visited_this_pass:
                    return visited_list
    return None


if __name__ == '__main__':
    print(detect_cycle([-1, 0, 1]))
    print(detect_cycle([3, 0, 1, 2]))

    print(-1 % 10)