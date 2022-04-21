import lap
import math
import numpy as np
import os

from itertools import product
from queue import Queue
from sortedcontainers import SortedList
import networkx as nx
# from floyd_warshall import floyd_warshall_single_core


import time


def load_dset(dname, seed, data_dir="."):
    tpms = np.load(os.path.join(data_dir, "data", dname, "scores.npy"))
    covs = np.load(os.path.join(data_dir, "data", dname, "covs.npy"))
    loads = np.load(os.path.join(data_dir, "data", dname, "loads.npy"))

    rng = np.random.default_rng(seed)

    tpms = np.clip(tpms, 0, np.inf)
    tpms /= np.max(tpms)

    # Sample the "true" bids that would occur if reviewers bid on all papers.
    noisy_tpms = tpms + rng.normal(-0.1, 0.1, tpms.shape)
    noisy_tpms = np.clip(noisy_tpms, 0, 1)
    true_bids = rng.uniform(0, 1, size=tpms.shape) < noisy_tpms

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


def spfa_adj_matrix(adj_matrix, src_set):
    # print(src_set)
    # print("\n\nStart spfa_adj_matrix")
    dis = [0] * adj_matrix.shape[0]
    pre = [-1] * adj_matrix.shape[0]
    vertex_queue = SortedList()
    vertex_queue_set = set()
    for v in src_set:
        vertex_queue.add((0, v))
        vertex_queue_set.add(v)

    while len(vertex_queue):
        # print("vertex_queue: ", vertex_queue)
        u = vertex_queue.pop()[1]
        vertex_queue_set.remove(u)
        # print(u)
        for v in np.where(adj_matrix[u, :] < np.inf)[0].tolist():
            if dis[u] + adj_matrix[u][v] < dis[v] and not math.isclose(dis[u] + adj_matrix[u][v], dis[v]):
                pre[v] = u
                dis[v] = dis[u] + adj_matrix[u][v]

                if v in src_set:
                    cyc = detect_cycle(pre)
                    if cyc:
                        return cyc
                if v not in vertex_queue_set:
                    vertex_queue.add((-dis[v], v))
                    vertex_queue_set.add(v)

    # print("starting to check solution")
    #
    # for u in range(adj_matrix.shape[0]):
    #     if u % 10 == 0:
    #         print(u/adj_matrix.shape[0])
    #     for v in np.where(adj_matrix[u, :] < np.inf)[0].tolist():
    #         if dis[u] + adj_matrix[u][v] < dis[v] and not math.isclose(dis[u] + adj_matrix[u][v], dis[v]):
    #             print("error, some edge did not get relaxed when it should have")

    for v in src_set:
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


def reconstruct_path(source, target, predecessors):
    if source == target:
        return []
    prev = predecessors[source, :]
    curr = prev[target]
    path = [target, curr]
    while curr != source:
        curr = prev[curr]
        path.append(curr)
    return list(reversed(path))


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


def decycle(p):
    final_path = []
    seen = set()
    for i in p:
        final_path.append(i)
        if i in seen:
            # back track
            final_path.pop()
            while final_path[-1] != i:
                seen.remove(final_path.pop())
        seen.add(i)
    return final_path


# Rotate the papers along the cycle. Update the allocation and the shortest paths and the adjacency matrix
# formulation of the graph.
# The adjacency matrix is probably not necessary anymore though.
# We only need to update shortest paths for the node pairs whose shortest paths went through the region.
# We should maintain the inverted index showing, for each node, which node-pairs have that node on
# the shortest path.
def apply_cycle(cycle, adj_matrix, updated_alloc, lb, ub):
    st = time.time()
    # update the allocation and residual graph using the cycle
    m, n = updated_alloc.shape

    # for i in cycle:
    #     if i < m:
    #         print("Prior alloc, %d: %s" % (i, np.where(updated_alloc[i, :])))
    #     print("Prior adj_matrix, %d: %s" % (i, adj_matrix[i, :]))

    # print(cycle)

    # The cycle goes backward in the residual graph. Thus, we need to assign the i-1'th paper to the i'th
    # reviewer, and unassign the i+1'th paper.
    ctr = 0 if cycle[0] < m else 1
    while ctr < len(cycle):
        paper_to_assign = cycle[(ctr - 1) % len(cycle)] - m
        paper_to_drop = cycle[(ctr + 1) % len(cycle)] - m
        curr_rev = cycle[ctr]

        # print("curr_rev, paper_to_assign, paper_to_drop: ", curr_rev, paper_to_assign, paper_to_drop)

        if paper_to_assign < n:
            # We are assigning a non-dummy paper to the reviewer curr_rev
            updated_alloc[curr_rev, paper_to_assign] = 1
            # Reverse the edge and negate its weight
            adj_matrix[paper_to_assign + m][curr_rev] = -adj_matrix[curr_rev][
                paper_to_assign + m]
            adj_matrix[curr_rev][paper_to_assign + m] = np.inf

        if paper_to_drop < n:
            # We are dropping a non-dummy paper from the reviewer curr_rev
            updated_alloc[curr_rev, paper_to_drop] = 0
            # Reverse the edge and negate its weight
            adj_matrix[curr_rev][paper_to_drop + m] = -adj_matrix[paper_to_drop + m][curr_rev]
            adj_matrix[paper_to_drop + m][curr_rev] = np.inf

        # Update the residual graph if we have dropped the last paper
        # We need to make it so that curr_rev can't receive the dummy paper anymore.
        num_papers = np.sum(updated_alloc[curr_rev, :])
        if num_papers < lb + .1:
            adj_matrix[curr_rev][n + m] = np.inf
        # If we have a paper assigned (over the lb), we can ASSIGN the dummy
        else:
            adj_matrix[curr_rev][n + m] = 0

        # We drop the edge to the dummy paper here if we have assigned the reviewer up to their max.
        # So we make it so they can't give away the dummy paper (and thus receive a new assignment).
        if num_papers > ub - .1:
            adj_matrix[n + m][curr_rev] = np.inf
        else:
            # They can still give away the dummy
            adj_matrix[n + m][curr_rev] = 0

        # Move to the next REVIEWER... not the next vertex in the cycle
        ctr += 2

    # for i in cycle:
    #     if i < m:
    #         print("New alloc, %d: %s" % (i, np.where(updated_alloc[i, :])))
    #     print("New adj_matrix, %d: %s" % (i, adj_matrix[i, :]))
    return updated_alloc, adj_matrix

# def update_shortest_paths(adj_matrix, dists, preds, region):
#     # updated_alloc and adj_matrix have been updated. Now we need to update the
#     # dists, preds, inverted_idx
#     # Any shortest path which passes through region will need to be updated
#     # print("updated the alloc and adj_matrix. Updating the dists, preds, inverted_idx")
#     # print("region: %s" % region)
#     # update_pairs = set()
#     # for u, v in product(region, region):
#     #     update_pairs |= inverted_idx[(u, v)]
#
#
#     # Figure out which paths need to be updated by following edges out of the region
#     update_pairs = set()
#     for v in region:
#         all_preds_v = preds[:, v]
#         # For each idx where the pred is in the region, follow the path forward to get all the pairs we need to update
#         for i in region:
#             pred_is_i = np.where(all_preds_v == i)[0].tolist()
#             # update_pairs |= {(s, v) for s in pred_is_i}
#             for s in pred_is_i:
#                 dfs_queue = [v]
#                 while dfs_queue:
#                     end = dfs_queue.pop()
#                     update_pairs.add((s, end))
#                     # Get the new nodes
#                     pred_is_end = np.where(preds[s, :] == end)[0].tolist()
#                     dfs_queue.extend(pred_is_end)
#
#
#     # print(nx.reconstruct_path(38, 9, preds))
#     # print("update_pairs: %s" % update_pairs)
#     # cycle_edges = set()
#     # for i in range(len(cycle)-1):
#     #     cycle_edges.add((cycle[::-1][i], cycle[::-1][i+1]))
#     # ct = 0
#     # for x, y in update_pairs:
#     #     xy_path = nx.reconstruct_path(x, y, preds)
#     #     # print("%d-%d path: %s" % (x, y, xy_path))
#     #     overlap = False
#     #     for i in range(len(xy_path)-1):
#     #         if (xy_path[i], xy_path[i+1]) in cycle_edges:
#     #             overlap = True
#     #     if overlap:
#     #         ct += 1
#     # print(ct)
#     # for x, y in update_pairs:
#     #     if x in region and y in region:
#     #         print("%d-%d are both in region" % (x, y))
#
#     # We will update these shortest paths by setting all edges within the region to infinity,
#     # and set all u -> v distances to infinity for u and v which previously had a shortest path through the region.
#     # Then you can say that the shortest u -> v path is the min over all v' of u -> v' -> v. When you change the u -> v
#     # distance in this manner, then we will go through and update preds likewise. When we make a full pass without
#     # changing any pairs, we'll be done.
#     for i, j in product(region, region):
#         if i != j:
#             update_pairs.add((i, j))
#
#     adj_matrix_mask = np.zeros(adj_matrix.shape)
#     for i, j in product(region, region):
#         adj_matrix_mask[i, j] = np.inf
#
#     for i, j in update_pairs:
#         dists[i, j] = adj_matrix[i, j] + adj_matrix_mask[i, j]
#         preds[i, j] = i
#
#     # print("Time spent until updating shortest paths: %s s" % (time.time()-st))
#     st = time.time()
#
#     # print("Number of nodes in region: %d" % len(region))
#     # print("Number of pairs of nodes using edges in region in s.p.: %d" % len(update_pairs))
#
#     change = True
#     while change:
#         change = False
#         # print("restart")
#         for i, j in update_pairs:
#             print(i, j)
#             print("dist %d to %d, before update: %s" % (i, j, dists[i, j]))
#             if dists[i,j] < np.inf:
#                 "path before update: "
#                 curr = j
#                 while curr != i:
#                     print(curr)
#                     curr = preds[i][curr]
#                 # print("%d-%d path, before update: %s" % (i, j, nx.reconstruct_path(i, j, preds)))
#
#             intermed_node = np.argmin(dists[i, :] + dists[:, j])
#             # print("intermed: %d" % intermed_node)
#             new_dist = dists[i, intermed_node] + dists[intermed_node, j]
#             # print(new_dist < dists[i, j])
#             if new_dist < dists[i, j]:
#                 # if cycle[0] == 28 and cycle[-1] == 37:
#                 #     seen = set()
#                 #     print(i)
#                 #     print(j)
#                 #     print(intermed_node)
#                 #     print("path to intermed: ")
#                 #     curr = intermed_node
#                 #     while curr != i:
#                 #         if curr in seen:
#                 #             print(curr)
#                 #             sys.exit(0)
#                 #         seen.add(curr)
#                 #         print(curr)
#                 #         curr = preds[i][curr]
#                 #     print("path from intermed: ")
#                 #     curr = j
#                 #     seen = set()
#                 #     while curr != intermed_node:
#                 #         print(curr)
#                 #         if curr in seen:
#                 #             sys.exit(0)
#                 #         seen.add(curr)
#                 #         curr = preds[intermed_node][curr]
#                 # time.sleep(0.00001)
#                 path_to_intermed = reconstruct_path(i, intermed_node, preds)
#                 path_from_intermed = reconstruct_path(intermed_node, j, preds)
#                 # print("%d-%d path: %s" % (i, intermed_node, path_to_intermed))
#                 # print("%d-%d path: %s" % (intermed_node, j, path_from_intermed))
#
#                 change = True
#                 # Update dists
#                 dists[i, j] = new_dist
#                 # print("dist %d to %d is now %s" % (i, j, dists[i, j]))
#
#                 # If the paths intersect, we need to decycle them. They may intersect in multiple places.
#                 # There are no negative cycles outside the region, but we must just not have updated the distance
#                 # to that intersecting node yet, so we think that it's faster to go to the other node intermed_node,
#                 # which has the intersecting node on the way to intermed_node and on the way from it.
#
#                 actual_path = path_to_intermed[:-1] + path_from_intermed
#                 actual_path = decycle(actual_path)
#
#                 # path_from_intermed_set = set(path_from_intermed)
#                 # if set(path_to_intermed[:-1]) & path_from_intermed_set:
#                 #     actual_path = []
#                 #     curr_idx = 0
#                 #     while path_to_intermed[curr_idx] not in path_from_intermed_set:
#                 #         actual_path.append(path_to_intermed[curr_idx])
#                 #         curr_idx += 1
#                 #     curr_idx = 0
#                 #     while path_from_intermed[curr_idx]
#                 #     actual_path.extend(path_from_intermed)
#                 # else:
#                 #     actual_path = path_to_intermed[:-1] + path_from_intermed
#
#                 # Update preds:
#
#                 for i_idx in range(len(actual_path)-1, 0, -1):
#                     preds[i, actual_path[i_idx]] = actual_path[i_idx-1]
#
#                 # if actual_path != nx.reconstruct_path(i, j, preds):
#                 #     print(actual_path, nx.reconstruct_path(i, j, preds))
#                 #     sys.exit(0)
#
#                 # pred = j
#                 # while pred != intermed_node:
#                 #     print(pred)
#                 #     preds[i][pred] = preds[intermed_node][pred]
#                 #     pred = preds[i][pred]
#                 #     print(pred == intermed_node)
#                 # print("%d-%d path: %s" % (i, intermed_node, nx.reconstruct_path(i, intermed_node, preds)))
#                 # print("%d-%d path: %s" % (intermed_node, j, nx.reconstruct_path(intermed_node, j, preds)))
#                 # print("new %d-%d path: %s" % (i, j, nx.reconstruct_path(i, j, preds)))
#         # print("end")
#
#     # print("updated sp's in %s s" % (time.time() - st))
#
#     return dists, preds


# def update_shortest_paths(adj_matrix, dists, preds, region):
#     adj_matrix_mask = np.zeros(adj_matrix.shape)
#     for i, j in product(region, region):
#         if i != j:
#             adj_matrix_mask[i, j] = np.inf
#
#     dists, preds = floyd_warshall_single_core(adj_matrix + adj_matrix_mask)
#     preds = preds.astype(np.int32)
#
#     return dists, preds




def update_shortest_paths_keeping_region_edges(adj_matrix, dists, preds, region):
    print("Starting update sp's")
    print(adj_matrix)
    print(dists)
    print(preds)
    print(region)

    # Figure out which paths need to be updated by following edges out of the region
    update_pairs = set()
    for v in region:
        all_preds_v = preds[:, v]
        # For each idx where the pred is in the region, follow the path forward to get all the pairs we need to update
        for i in region:
            pred_is_i = np.where(all_preds_v == i)[0].tolist()
            # update_pairs |= {(s, v) for s in pred_is_i}
            for s in pred_is_i:
                dfs_queue = [v]
                while dfs_queue:
                    end = dfs_queue.pop()
                    update_pairs.add((s, end))
                    # Get the new nodes
                    pred_is_end = np.where(preds[s, :] == end)[0].tolist()
                    dfs_queue.extend(pred_is_end)

    print("Update pairs: ", sorted(update_pairs))

    # We will update these shortest paths by setting all edges within the region to infinity,
    # and set all u -> v distances to infinity for u and v which previously had a shortest path through the region.
    # Then you can say that the shortest u -> v path is the min over all v' of u -> v' -> v. When you change the u -> v
    # distance in this manner, then we will go through and update preds likewise. When we make a full pass without
    # changing any pairs, we'll be done.
    for i, j in product(region, region):
        if i != j:
            update_pairs.add((i, j))

    # adj_matrix_mask = np.zeros(adj_matrix.shape)
    # for i, j in product(region, region):
    #     adj_matrix_mask[i, j] = np.inf

    for i, j in update_pairs:
        dists[i, j] = adj_matrix[i, j]
        if adj_matrix[i, j] < np.inf:
            preds[i, j] = i
        else:
            preds[i, j] = -9999

    # print("Time spent until updating shortest paths: %s s" % (time.time()-st))
    st = time.time()

    # print("Number of nodes in region: %d" % len(region))
    # print("Number of pairs of nodes using edges in region in s.p.: %d" % len(update_pairs))

    change = True
    while change:
        change = False
        # print("restart")
        for i, j in update_pairs:
            print(i, j)
            print("dist %d to %d, before update: %s" % (i, j, dists[i, j]))
            if dists[i, j] < np.inf:
                "path before update: "
                curr = j
                while curr != i:
                    print(curr)
                    curr = preds[i][curr]
            # print("%d-%d path, before update: %s" % (i, j, nx.reconstruct_path(i, j, preds)))

            intermed_node = np.argmin(dists[i, :] + dists[:, j])
            print("intermed: %d" % intermed_node)
            new_dist = dists[i, intermed_node] + dists[intermed_node, j]
            print(new_dist < dists[i, j])
            if new_dist < dists[i, j]:
                path_to_intermed = reconstruct_path(i, intermed_node, preds)
                path_from_intermed = reconstruct_path(intermed_node, j, preds)
                print("%d-%d path: %s" % (i, intermed_node, path_to_intermed))
                print("%d-%d path: %s" % (intermed_node, j, path_from_intermed))

                change = True
                # Update dists
                dists[i, j] = new_dist
                print("dist %d to %d is now %s" % (i, j, dists[i, j]))

                # If the paths intersect, we need to decycle them. They may intersect in multiple places.
                # There are no negative cycles outside the region, but we must just not have updated the distance
                # to that intersecting node yet, so we think that it's faster to go to the other node intermed_node,
                # which has the intersecting node on the way to intermed_node and on the way from it.

                actual_path = path_to_intermed[:-1] + path_from_intermed
                actual_path = decycle(actual_path)
                print("actual path: ", actual_path)

                # path_from_intermed_set = set(path_from_intermed)
                # if set(path_to_intermed[:-1]) & path_from_intermed_set:
                #     actual_path = []
                #     curr_idx = 0
                #     while path_to_intermed[curr_idx] not in path_from_intermed_set:
                #         actual_path.append(path_to_intermed[curr_idx])
                #         curr_idx += 1
                #     curr_idx = 0
                #     while path_from_intermed[curr_idx]
                #     actual_path.extend(path_from_intermed)
                # else:
                #     actual_path = path_to_intermed[:-1] + path_from_intermed

                # Update preds:

                for i_idx in range(len(actual_path)-1, 0, -1):
                    preds[i, actual_path[i_idx]] = actual_path[i_idx-1]

                # if actual_path != nx.reconstruct_path(i, j, preds):
                #     print(actual_path, nx.reconstruct_path(i, j, preds))
                #     sys.exit(0)

                # pred = j
                # while pred != intermed_node:
                #     print(pred)
                #     preds[i][pred] = preds[intermed_node][pred]
                #     pred = preds[i][pred]
                #     print(pred == intermed_node)
                # print("%d-%d path: %s" % (i, intermed_node, nx.reconstruct_path(i, intermed_node, preds)))
                # print("%d-%d path: %s" % (intermed_node, j, nx.reconstruct_path(intermed_node, j, preds)))
                # print("new %d-%d path: %s" % (i, j, nx.reconstruct_path(i, j, preds)))
        # print("end")

    # print("updated sp's in %s s" % (time.time() - st))

    return dists, preds


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