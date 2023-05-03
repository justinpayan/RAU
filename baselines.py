# Code obtained from Ivan Stelmakh's website (https://www.cs.cmu.edu/~istelmak/), and slightly modified by
# Justin Payan.

from gurobipy import Model, GRB
from itertools import product
import numpy as np
import time
from collections import defaultdict
from ortools.graph import pywrapgraph

import uuid


class auto_assigner:
    # tolerance for integrality check
    _EPS = 1e-3

    # initialize the parameters
    # demand - requested number of reviewers per paper
    # ability - the maximum number of papers reviewer can review
    # function - transformation function of similarities
    # iter_limit - maximum number of iterations of Steps 2 to 7
    # time_limit - time limit in seconds. The algorithm performs iterations of Steps 2 to 7 until the time limit is exceeded
    def __init__(self, simmatrix, demand=3, ability=None, function=lambda x: x, iter_limit=np.inf, time_limit=np.inf):
        self.simmatrix = simmatrix
        self.numrev = simmatrix.shape[0]
        self.numpapers = simmatrix.shape[1]
        self.ability = ability
        self.demand = demand
        self.function = function
        if iter_limit < 1:
            raise ValueError('Maximum number of iterations must be at least 1')
        self.iter_limit = iter_limit
        self.time_limit = time_limit

    # initialize the flow network in the subroutine
    def _initialize_model(self):

        problem = Model()
        problem.setParam('OutputFlag', False)

        # edges from source to reviewers, capacity controls maximum reviewer load
        self._source_vars = problem.addVars(self.numrev, vtype=GRB.CONTINUOUS, lb=0.0,
                                            ub=self.ability, name='reviewers')

        # edges from papers to sink, capacity controls a number of reviewers per paper
        self._sink_vars = problem.addVars(self.numpapers, vtype=GRB.CONTINUOUS, lb=0.0,
                                          ub=self.demand, name='papers')

        # edges between reviewers and papers. Initially capacities are set to 0 (no edge is added in the network)
        self._mix_vars = problem.addVars(self.numrev, self.numpapers, vtype=GRB.CONTINUOUS,
                                         lb=0.0, ub=0.0, name='assignment')
        problem.update()

        # flow balance equations for reviewers' nodes
        self._balance_reviewers = problem.addConstrs((self._source_vars[i] == self._mix_vars.sum(i, '*')
                                                      for i in range(self.numrev)))

        # flow balance equations for papers' nodes
        self._balance_papers = problem.addConstrs((self._sink_vars[i] == self._mix_vars.sum('*', i)
                                                   for i in range(self.numpapers)))
        problem.update()

        self._problem = problem

    # compute the order in which subroutine adds edges to the network
    def _ranking_of_pairs(self, simmatrix):
        pairs = [[reviewer, paper] for (reviewer, paper) in product(range(self.numrev), range(self.numpapers))]
        sorted_pairs = sorted(pairs, key=lambda x: simmatrix[x[0], x[1]], reverse=True)
        return sorted_pairs

    # subroutine
    # simmatrix - similarity matrix, updated for previously assigned papers
    # kappa - requested number of reviewers per paper
    # abilities - current constraints on reviewers' loads
    # not_assigned - a set of papers to be assigned
    # lower_bound - internal variable to start the binary search from
    def _subroutine(self, simmatrix, kappa, abilities, not_assigned, lower_bound, *args):

        # set up the max flow objective
        self._problem.setObjective(sum([self._source_vars[i] for i in range(self.numrev)]), GRB.MAXIMIZE)

        # if paper is not fixed in the final output yet, assign it with kappa reviewers
        for paper in not_assigned:
            self._sink_vars[paper].ub = kappa
            self._sink_vars[paper].lb = 0

        # adjust reviewers' loads (in the network) for paper that are already fixed in the final assignment
        for reviewer in range(self.numrev):
            self._source_vars[reviewer].ub = abilities[reviewer]

        sorted_pairs = self._ranking_of_pairs(simmatrix)

        # upper_bound - internal variable to start the binary search from
        if args != ():
            upper_bound = args[0]
        else:
            upper_bound = len(sorted_pairs)

        current_solution = 0

        # if upper_bound == lower_bound, do one iteration to add corresponding edges to the flow network
        one_iteration_done = False

        # binary search to find the minimum number of edges
        # with largest similarity that should be added to the network
        # to achieve the requested max flow

        while lower_bound < upper_bound or not one_iteration_done:
            one_iteration_done = True
            prev_solution = current_solution
            current_solution = int(lower_bound + (upper_bound - lower_bound) / 2)

            # the next condition is to control the case when upper_bound - lower_bound = 1
            # then it must be the case that max flow is less then required
            if current_solution == prev_solution:
                if maxflow < len(not_assigned) * kappa and current_solution == lower_bound:
                    current_solution += 1
                    lower_bound += 1
                else:
                    raise ValueError('An error occured1')

            # if binary choice increased the current estimate, add corresponding edges to the network
            if current_solution > prev_solution:
                for cur_pair in sorted_pairs[prev_solution: current_solution]:
                    self._mix_vars[cur_pair[0], cur_pair[1]].ub = 1
            # otherwise remove the corresponding edges
            else:
                for cur_pair in sorted_pairs[current_solution: prev_solution]:
                    self._mix_vars[cur_pair[0], cur_pair[1]].ub = 0

            # check maxflow in the current estimate
            self._problem.optimize()
            maxflow = self._problem.objVal

            # if maxflow equals to the required flow, decrease the upper bound on the solution
            if maxflow == len(not_assigned) * kappa:
                upper_bound = current_solution
            # otherwise increase the lower bound
            elif maxflow < len(not_assigned) * kappa:
                lower_bound = current_solution
            else:
                raise ValueError('An error occured2')

        # check if binary search succesfully converged
        if maxflow != len(not_assigned) * kappa or lower_bound != current_solution:
            # shouldn't enter here
            print
            maxflow, len(not_assigned), lower_bound, current_solution
            raise ValueError('An error occured3')

        # prepare for max-cost max-flow -- we enforce each paper to be reviewed by kappa reviewers
        for paper in not_assigned:
            self._sink_vars[paper].lb = kappa

        # max cost max flow objective
        self._problem.setObjective(sum([sum([simmatrix[reviewer, paper] * self._mix_vars[reviewer, paper]
                                             for paper in not_assigned])
                                        for reviewer in range(self.numrev)]), GRB.MAXIMIZE)
        self._problem.optimize()

        # return assignment
        assignment = {}
        for paper in not_assigned:
            assignment[paper] = []
        for reviewer in range(self.numrev):
            for paper in not_assigned:
                if self._mix_vars[reviewer, paper].X == 1:
                    assignment[paper] += [reviewer]
                if np.abs(self._mix_vars[reviewer, paper].X - int(self._mix_vars[reviewer, paper].X)) > self._EPS:
                    raise ValueError('Error with rounding -- please check that demand and ability are integal')
                self._mix_vars[reviewer, paper].ub = 0
        self._problem.update()

        return assignment, current_solution

    # Join two assignments
    @staticmethod
    def _join_assignment(assignment1, assignment2):
        assignment = {}
        for paper in assignment1:
            assignment[paper] = assignment1[paper] + assignment2[paper]
        return assignment

    # Compute fairness
    def quality(self, assignment, *args):
        qual = np.inf
        if args != ():
            paper = args[0]
            return np.sum([self.function(self.simmatrix[reviewer, paper]) for reviewer in assignment[paper]])
        else:
            for paper in assignment:
                if qual > sum([self.function(self.simmatrix[reviewer, paper]) for reviewer in assignment[paper]]):
                    qual = np.sum([self.function(self.simmatrix[reviewer, paper]) for reviewer in assignment[paper]])
        return qual

    # Full algorithm
    def _fair_assignment(self):

        # Counter for number of performed iterations
        iter_counter = 0
        # Start time
        start_time = time.time()

        current_best = None
        current_best_score = 0
        local_simmatrix = self.simmatrix.copy()
        # local_abilities = self.ability * np.ones(self.numrev)
        local_abilities = self.ability
        not_assigned = set(range(self.numpapers))
        final_assignment = {}

        # One iteration of Steps 2 to 7 of the algorithm
        while not_assigned != set() and iter_counter < self.iter_limit and (
                time.time() < start_time + self.time_limit or iter_counter == 0):
            print(iter_counter)
            iter_counter += 1

            lower_bound = 0
            upper_bound = len(not_assigned) * self.numrev

            # Step 2
            for kappa in range(1, self.demand + 1):

                # Step 2(a)
                tmp_abilities = local_abilities.copy()
                tmp_simmatrix = local_simmatrix.copy()

                # Step 2(b)
                assignment1, lower_bound = self._subroutine(tmp_simmatrix, kappa, tmp_abilities, not_assigned,
                                                            lower_bound, upper_bound)

                # Step 2(c)
                for paper in assignment1:
                    for reviewer in assignment1[paper]:
                        tmp_simmatrix[reviewer, paper] = -1
                        tmp_abilities[reviewer] -= 1

                # Step 2(d)
                assignment2 = self._subroutine(tmp_simmatrix, self.demand - kappa, tmp_abilities, not_assigned,
                                               lower_bound, upper_bound)[0]

                # Step 2(e)
                assignment = self._join_assignment(assignment1, assignment2)

                # Keep track of the best candidate assignment (including the one from the prev. iteration)
                if self.quality(assignment) > current_best_score or current_best_score == 0:
                    current_best = assignment
                    current_best_score = self.quality(assignment)

            # Steps 4 to 6
            for paper in not_assigned.copy():
                # For every paper not yet fixed in the final assignment we update the assignment
                final_assignment[paper] = current_best[paper]
                # Find the most worst-off paper
                if self.quality(current_best, paper) == current_best_score:
                    # Delete it from current candidate assignment and from the set of papers which are
                    # not yet fixed in the final output
                    del current_best[paper]
                    not_assigned.discard(paper)
                    # This paper is now fixed in the final assignment

                    # Update abilities of reviewers
                    for reviewer in range(self.numrev):
                        # edges adjunct to the vertex of the most worst-off papers
                        # will not be used in the flow network any more
                        local_simmatrix[reviewer, paper] = -1
                        self._mix_vars[reviewer, paper].ub = 0
                        self._mix_vars[reviewer, paper].lb = 0
                        if reviewer in final_assignment[paper]:
                            local_abilities[reviewer] -= 1

            current_best_score = self.quality(current_best)
            self._problem.update()

        self.fa = final_assignment
        self.best_quality = self.quality(final_assignment)

    def fair_assignment(self):
        self._initialize_model()
        self._fair_assignment()

class FairFlow(object):
    """Approximate makespan matching via flow network (with lower bounds).
    Approximately solve the reviewer assignment problem with makespan
    constraint. Based on the algorithm introduced in Gairing et. al 2004 and
    Gairing et. al. 2007.  Our adaptation works as follows.  After we have a
    matching, construct three groups of papers.  The first group are all papers
    with scores > makespan value, the second group are all papers whose
    papers scores are between the makespan and the makespan - maxaffinity, the
    final group are the papers whose paper scores are less than makespan -
    maxaffinity.  For each paper in the last group, we'll unassign the
    reviewer with the lowest score. Then, we'll construct a new flow network
    from the papers in the first group as sources through reviewers assigned to
    those paper and terminating in the papers in the last group. Each sink will
    accept a single new assignment.  Once this assignment is made.  We'll
    construct another flow network of all available reviewers to the papers that
    do not have enough reviewers and solve the flow problem again.  Then we'll
    have a feasible solution. We can continue to iterate this process until
    either: there are no papers in the first group, there are no papers in the
    third group, or running the procedure does not change the sum total score of
    the matching.
    """
    def __init__(self, loads, loads_lb, coverages, affs, sol=None):
        """Initialize a makespan flow matcher
        Args:
            loads - a list of integers specifying the maximum number of papers
                  for each reviewer.
            loads_lb - list of integers specifying min number of papers per rev.
            coverages - a list of integers specifying the number of reviews per
                 paper.
            weights - the affinity matrix (np.array) of papers to reviewers.
                   Rows correspond to reviewers and columns correspond to
                   papers.
            solution - a matrix of assignments (same shape as weights).
        Returns:
            initialized makespan matcher.
        """
        self.n_rev = np.size(affs, axis=0)
        self.n_pap = np.size(affs, axis=1)
        self.loads = loads
        self.loads_lb = loads_lb
        self.coverages = coverages
        # make sure that all weights are positive:
        self.orig_affs = affs.copy()
        self.affs = affs.copy()
        min_aff = np.min(affs)
        if min_aff < 0:
            self.affs -= min_aff
        self.id = uuid.uuid4()
        self.makespan = 0.0     # the minimum allowable paper score.
        self.solution = sol if sol else np.zeros((self.n_rev, self.n_pap))
        self.valid = True if sol else False
        assert(self.affs.shape == self.solution.shape)
        self.maxaff = np.max(self.affs)
        self.big_c = 10000
        self.bigger_c = self.big_c ** 2

        self.min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        self.start_inds = []
        self.end_inds = []
        self.caps = []
        self.costs = []
        self.source = self.n_rev + self.n_pap
        self.sink = self.n_rev + self.n_pap + 1

    def objective_val(self):
        """Get the objective value of the RAP."""
        return np.sum(self.sol_as_mat() * self.orig_affs)

    def _refresh_internal_vars(self):
        """Set start, end, caps, costs to be empty."""
        self.min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        self.start_inds = []
        self.end_inds = []
        self.caps = []
        self.costs = []

    def _grp_paps_by_ms(self):
        """Group papers by makespan.
        Divide papers into 3 groups based on their paper scores. A paper score
        is the sum affinities among all reviewers assigned to review that paper.
        The first group will contain papers with paper scores greater than or
        equal to the makespan.  The second group will contain papers with paper
        scores less than the makespan but greater than makespan - maxaffinity.
        The third group will contain papers with papers scores less than
        makespan - maxaffinity.
        Args:
            None
        Returns:
            A 3-tuple of paper ids.
        """
        paper_scores = np.sum(self.solution * self.affs, axis=0)
        g1 = np.where(paper_scores >= self.makespan)[0]
        g2 = np.intersect1d(
            np.where(self.makespan > paper_scores),
            np.where(paper_scores >= self.makespan - self.maxaff))
        g3 = np.where(self.makespan - self.maxaff > paper_scores)[0]
        assert(np.size(g1) + np.size(g2) + np.size(g3) == self.n_pap)
        return g1, g2, g3

    def _worst_reviewer(self, papers):
        """Get the worst reviewer from each paper in the input.
        Args:
            papers - numpy array of paper indices.
        Returns:
            A tuple of rows and columns of the
        """
        mask = (self.solution - 1.0) * -self.big_c
        tmp = (mask + self.affs).astype('float')
        worst_revs = np.argmin(tmp, axis=0)
        return worst_revs[papers], papers

    def _construct_and_solve_validifier_network(self):
        """Construct a network to make an invalid solution valid.
        To do this we need to ensure that:
            1) each load upper bound is satisfied
            2) each paper coverage constraint is satisfied.
        Returns:
            None -- modifies the internal min_cost_flow network.
        """
        # First solve flow with lower bounds as caps.
        # Construct edges between the source and each reviewer that must review.
        if self.loads_lb is not None:
            rev_caps = np.maximum(self.loads_lb - np.sum(self.solution, axis=1),
                                  0)
            assert (np.size(rev_caps) == self.n_rev)
            flow = np.sum(rev_caps)
            pap_caps = np.maximum(self.coverages - np.sum(
                self.solution, axis=0), 0)
            self._construct_graph_and_solve(self.n_rev, self.n_pap, rev_caps,
                                            pap_caps, self.affs, flow)

        # Now compute the residual flow that must be routed so that each paper
        # is sufficiently reviewed. Also compute residual loads and coverages.
        rev_caps = self.loads - np.sum(self.solution, axis=1)
        assert (np.size(rev_caps) == self.n_rev)
        pap_caps = np.maximum(self.coverages - np.sum(self.solution, axis=0), 0)
        flow = np.sum(pap_caps)
        self._construct_graph_and_solve(self.n_rev, self.n_pap, rev_caps,
                                        pap_caps, self.affs, flow)
        # Finally, return.
        assert (np.all(np.sum(self.solution, axis=0) == self.coverages))
        assert (np.all(np.sum(self.solution, axis=1) <= self.loads))
        if self.loads_lb is not None:
            assert (np.all(np.sum(self.solution, axis=1) >= self.loads_lb))
        self.valid = True
        return self.solution

    def _construct_ms_improvement_network(self, g1, g2, g3):
        """Construct the network the reassigns reviewers to improve makespan.
        We allow for each paper in G1 to have 1 reviewer removed. This
        guarantees that papers in G1 can only fall to G2. Then, we may assign
        each unassigned reviewer to a paper in G2 or G3. Papers in G2 **may**
        have their reviewers unassigned **only if** their score, s, satisfies
        s - r(g2)_max + r(g1)_min > T - max, so that they remain in G2. Then,
        allow all reviewers who were unassigned to be assigned to the available
        papers in G3.
        Args:
            g1 - numpy array of paper ids in group 1 (best).
            g2 - numpy array of paper ids in group 2.
            g3 - numpy array of paper ids in group 3 (worst).
        Returns:
            None -- modifies the internal min_cost_flow network.
        """
        # Must convert to python ints first.
        g1 = [int(x) for x in g1]
        g2 = [int(x) for x in g2]
        g3 = [int(x) for x in g3]

        pap_scores = np.sum(self.solution * self.affs, axis=0)

        # First construct edges between the source and each pap in g1.
        self._refresh_internal_vars()
        for i in range(np.size(g1)):
            self.start_inds.append(self.source)
            self.end_inds.append(self.n_rev + g1[i])
            self.caps.append(1)
            self.costs.append(0)

        # Next construct the sink node and edges to each paper in g3.
        for i in range(np.size(g3)):
            self.start_inds.append(self.n_rev + g3[i])
            self.end_inds.append(self.sink)
            self.caps.append(1)
            self.costs.append(0)

        # For each paper in g2, create a dummy node the restricts the flow to
        # that paper to 1.
        for pap2 in g2:
            self.start_inds.append(self.n_rev + self.n_pap + 2 + pap2)
            self.end_inds.append(self.n_rev + pap2)
            self.caps.append(1)
            self.costs.append(0)

        # For each assignment in the g1 group, reverse the flow.
        revs, paps1 = np.nonzero(self.solution[:, g1])
        assignment_to_give = set()
        added = set()
        pg2_to_minaff = defaultdict(lambda: np.inf) # min incoming affinity.
        for i in range(np.size(revs)):
            rev = int(revs[i])
            pap = g1[paps1[i]]
            assert(self.solution[rev, pap] == 1.0)
            self.start_inds.append(self.n_rev + pap)
            self.end_inds.append(rev)
            self.caps.append(1)
            self.costs.append(0)
            assignment_to_give.add(rev)

            # and now connect this reviewer to each dummy paper associate with
            # a paper in g2 if that rev not already been assigned to that paper.
            if rev not in added:
                for pap2 in g2:
                    if self.solution[rev, pap2] == 0.0:
                        rp_aff = self.affs[rev, pap2]
                        self.start_inds.append(rev)
                        self.end_inds.append(self.n_rev + self.n_pap + 2 + pap2)
                        pg2_to_minaff[pap2] = min(pg2_to_minaff[pap2], rp_aff)

                        self.caps.append(1)
                        self.costs.append(0)
                added.add(rev)
        # For each paper in g2, reverse the flow to assigned revs only if the
        # reversal, plus the min edge coming in from G1 wouldn't violate ms.
        revs, paps2 = np.nonzero(self.solution[:, g2])
        for i in range(np.size(revs)):
            rev = int(revs[i])
            pap = g2[paps2[i]]
            pap_score = pap_scores[pap]
            assert(self.solution[rev, pap] == 1.0)
            min_in = pg2_to_minaff[pap]
            rp_aff = self.affs[rev, pap]
            # lower bound on new paper score.
            lower_bound = (pap_score + min_in - rp_aff)
            ms_satisfied = (self.makespan - self.maxaff) <= lower_bound
            if min_in < np.inf and ms_satisfied:
                self.start_inds.append(self.n_rev + pap)
                self.end_inds.append(rev)
                self.caps.append(1)
                self.costs.append(0)
                assignment_to_give.add(rev)

        # For each reviewer, connect them to a paper in g3 if not assigned.
        for rev in assignment_to_give:
            for pap3 in g3:
                if self.solution[rev, pap3] == 0.0:
                    self.start_inds.append(rev)
                    self.end_inds.append(self.n_rev + pap3)
                    self.caps.append(1)
                    lb = self.makespan - self.maxaff
                    pap_score = pap_scores[pap3]
                    rp_aff = self.affs[rev, pap3]
                    # give a bigger reward if assignment would improve group.
                    if rp_aff + pap_score >= lb:
                        self.costs.append(int(-1.0 - self.bigger_c * rp_aff))
                    else:
                        self.costs.append(int(-1.0 - self.big_c * rp_aff))

        flow = int(min(np.size(g3), np.size(g1)))
        self.supplies = np.zeros(self.n_rev + self.n_pap + 2)
        self.supplies[self.source] = flow
        self.supplies[self.sink] = -flow

        for i in range(len(self.start_inds)):
            self.min_cost_flow.AddArcWithCapacityAndUnitCost(
                self.start_inds[i], self.end_inds[i], self.caps[i],
                self.costs[i])
        for i in range(len(self.supplies)):
            self.min_cost_flow.SetNodeSupply(i, int(self.supplies[i]))

    def solve_ms_improvement(self):
        """Reassign reviewers to improve the makespan.
        After solving min-cost-flow in the improvement network, record the
        corresponding solution. In particular, if we have flow leaving a paper
        and entering a reviewer, unassign the reviewer from that paper. If we
        have flow leaving a reviewer and entering a paper, assign the reviewer
        to that paper.
        """
        if self.min_cost_flow.Solve() == self.min_cost_flow.OPTIMAL:
            num_un = 0
            for arc in range(self.min_cost_flow.NumArcs()):
                # Can ignore arcs leading out of source or into sink.
                if self.min_cost_flow.Tail(arc) != self.source and \
                                self.min_cost_flow.Head(arc) != self.sink:
                    if self.min_cost_flow.Flow(arc) > 0:
                        # flow goes from tail to head
                        head = self.min_cost_flow.Head(arc)
                        tail = self.min_cost_flow.Tail(arc)
                        if head >= self.n_rev + self.n_pap + 2:
                            # this is an edge that restricts flow to a paper
                            pap = head - (self.n_rev + self.n_pap + 2)
                            assert(tail <= self.n_rev)
                            rev = tail
                            assert(self.solution[rev, pap] == 0.0)
                            self.solution[rev, pap] = 1.0
                        elif tail >= self.n_rev + self.n_pap + 2:
                            continue
                        elif head >= self.n_rev:
                            pap = head - self.n_rev
                            rev = tail
                            assert(self.solution[rev, pap] == 0.0)
                            self.solution[rev, pap] = 1.0
                            num_un += 1
                        else:
                            rev = head
                            pap = tail - self.n_rev
                            assert(self.solution[rev, pap] == 1.0)
                            self.solution[rev, pap] = 0.0
            self.valid = False
        else:
            raise Exception('There was an issue with the min cost flow input.')

    def solve_validifier(self):
        """Reassign reviewers to make the matching valid."""
        if self.min_cost_flow.Solve() == self.min_cost_flow.OPTIMAL:
            for arc in range(self.min_cost_flow.NumArcs()):
                # Can ignore arcs leading out of source or into sink.
                if self.min_cost_flow.Tail(arc) != self.source and \
                                self.min_cost_flow.Head(arc) != self.sink:
                    if self.min_cost_flow.Flow(arc) > 0:
                        rev = self.min_cost_flow.Tail(arc)
                        pap = self.min_cost_flow.Head(arc) - self.n_rev
                        assert(self.solution[rev, pap] == 0.0)
                        assert(np.sum(self.solution[:, pap], axis=0) ==
                               self.coverages[pap] - 1)
                        self.solution[rev, pap] = 1.0
            assert np.all(np.sum(self.solution, axis=1) <= self.loads)
            assert (np.sum(self.solution) == np.sum(self.coverages))
            self.valid = True
        else:
            raise Exception('There was an issue with the min cost flow input.')

    def sol_as_mat(self):
        if self.valid:
            return self.solution
        else:
            raise Exception(
                'You must have solved the model optimally or suboptimally '
                'before calling this function.')

    def try_improve_ms(self):
        """Try to improve the minimum paper score.
        Construct the refinement network (that routes assignments from the
        group of papers with high paper score to low paper scores) and solve the
        corresponding min cost flow problem. Then, remove the worst reviewer
        from each paper with more than the required number of reviewers.
        Finally, construct the validifier network to route available reviewers
        to papers missing a reviewer.
        Args:
            None
        Returns:
            A tuple of the size of the top group (papers with highest paper
            scores) and the size of the bottom group (papers with the lowest
            paper scores).
        """
        self._refresh_internal_vars()
        if np.sum(self.solution) != np.sum(self.coverages):
            self._construct_and_solve_validifier_network()
        assert(np.sum(self.solution) == np.sum(self.coverages))
        g1, g2, g3 = self._grp_paps_by_ms()
        old_g1, old_g2, old_g3 = set(g1), set(g2), set(g3)
        if np.size(g1) > 0 and np.size(g3) > 0:
            self._refresh_internal_vars()
            # Unassign the worst reviewer from each paper in g3.
            w_revs, w_paps = self._worst_reviewer(g3)
            assert (np.sum(self.solution) == np.sum(self.coverages))
            assert(len(set(w_paps)) == len(w_paps))
            self.solution[w_revs, w_paps] = 0.0

            # Try to route reviewers from the top group to the bottom.
            self._construct_ms_improvement_network(g1, g2, g3)
            self.solve_ms_improvement()

            # Construct a valid solution.
            self._construct_and_solve_validifier_network()

            # Checks: the bottom group should never grow in size.
            g1, g2, g3 = self._grp_paps_by_ms()
            assert(len(g3) <= len(old_g3))
            return np.size(g1), np.size(g3)
        else:
            return np.size(g1), np.size(g3)

    def _construct_graph_and_solve(self, n_rev, n_pap, _caps, _covs, ws, flow):
        """Solve min-cost-flow.
        Args:
            n_rev - (int) number of reviewers (sources)
            n_pap - (int) number of papers (sinks)
            _caps - (array of ints) capacities for each reviewer
            _covs - (array of ints) coverages for each paper
            ws - (matrix) affinities between reviewers and papers.
            flow - (int) total flow from revs to paps (some of coverages)
        Returns:
            None -- but sets self.solution to be a binary matrix containing the
            assignment of reviewers to papers.
        """
        start_inds = []
        end_inds = []
        caps = []
        costs = []
        source = n_rev + n_pap
        sink = n_rev + n_pap + 1

        # edges from source to revs.
        for i in range(n_rev):
            start_inds.append(source)
            end_inds.append(i)
            caps.append(int(_caps[i]))
            costs.append(0)

        # edges from rev to pap.
        for i in range(n_rev):
            for j in range(n_pap):
                start_inds.append(i)
                end_inds.append(n_rev + j)
                if self.solution[i, j] == 1:
                    caps.append(0)
                else:
                    caps.append(1)
                # Costs must be integers. Also, we have affinities so make
                # the "costs" negative affinities.
                costs.append(int(-1.0 - self.big_c * ws[i, j]))

        # edges from pap to sink.
        for j in range(n_pap):
            start_inds.append(n_rev + j)
            end_inds.append(sink)
            caps.append(int(_covs[j]))
            costs.append(0)

        supplies = np.zeros(n_rev + n_pap + 2)
        supplies[source] = int(flow)
        supplies[sink] = int(-flow)

        # Add arcs.
        mcf = pywrapgraph.SimpleMinCostFlow()
        for i in range(len(start_inds)):
            mcf.AddArcWithCapacityAndUnitCost(
                start_inds[i], end_inds[i], caps[i],
                costs[i])
        for i in range(len(supplies)):
            mcf.SetNodeSupply(i, int(supplies[i]))

        # Solve.
        if mcf.Solve() == mcf.OPTIMAL:
            for arc in range(mcf.NumArcs()):
                # Can ignore arcs leading out of source or into sink.
                if mcf.Tail(arc) != source and mcf.Head(arc) != sink:
                    if mcf.Flow(arc) > 0:
                        rev = mcf.Tail(arc)
                        pap = mcf.Head(arc) - n_rev
                        assert(self.solution[rev, pap] == 0.0)
                        self.solution[rev, pap] = 1.0
            self.solved = True
        else:
            raise Exception('There was an issue with the min cost flow input.')

    def find_ms(self):
        """Find an the highest possible makespan.
        Perform a binary search on the makespan value. Solve the RAP with each
        makespan value and return the solution corresponding to the makespan
        which achieves the largest minimum paper score.
        Args:
            None
        Return:
            Highest feasible makespan value found.
        """
        mn = 0.0
        mx = np.max(self.affs) * np.max(self.coverages)
        ms = (mx - mn) / 2.0
        self.makespan = ms
        best = None
        best_worst_pap_score = 0.0

        for i in range(10):
            print('#info FairFlow:ITERATION %s ms %s' % (i, ms))
            s1, s3 = self.try_improve_ms()
            can_improve = s3 > 0
            prev_s1, prev_s3 = -1, -1
            while can_improve and prev_s3 != s3:
                prev_s1, prev_s3 = s1, s3
                start = time.time()
                s1, s3 = self.try_improve_ms()
                can_improve = s3 > 0
                print('#info FairFlow:try_improve takes: %s s' % (
                        time.time() - start))

            worst_pap_score = np.min(np.sum(self.solution * self.affs, axis=0))
            print('#info FairFlow:best worst paper score %s worst score %s' % (
                best_worst_pap_score, worst_pap_score))

            success = s3 == 0
            print('#info FairFlow:success = %s' % success)
            if success and worst_pap_score >= best_worst_pap_score:
                best = ms
                best_worst_pap_score = worst_pap_score
                mn = ms
                ms += (mx - ms) / 2.0
            else:
                assert (not success or worst_pap_score < best_worst_pap_score)
                mx = ms
                ms -= (ms - mn) / 2.0
            self.makespan = ms
        print('#info FairFlow:Best found %s' % best)
        print('#info FairFlow:Best Worst Paper Score found %s' %
              best_worst_pap_score)
        if best is None:
            return 0.0
        else:
            return best

    def solve(self):
        """Find a makespan and solve flow.
        Run a binary search to find best makespan and return the corresponding
        solution.
        Args:
            mn - the minimum feasible makespan (optional).
            mx - the maximum possible makespan( optional).
            itr - the number of iterations of binary search for the makespan.
        Returns:
            The solution as a matrix.
        """
        ms = self.find_ms()
        self.makespan = ms
        s1, s3 = self.try_improve_ms()
        can_improve = s3 > 0
        prev_s1, prev_s3 = -1, -1
        while can_improve and (prev_s1 != s1 or prev_s3 != s3):
            prev_s1, prev_s3 = s1, s3
            s1, s3 = self.try_improve_ms()
            can_improve = s3 > 0

        return self.sol_as_mat()


def get_valuation(paper, reviewer_set, paper_reviewer_affinities):
    val = 0
    for r in reviewer_set:
        val += paper_reviewer_affinities[r, paper]
    return val


def greedy(scores, loads, covs, best_revs):
    m, n = scores.shape

    available_agents = set(range(n))
    ordering = []
    alloc = {p: list() for p in available_agents}

    round_num = 0

    max_mg = np.max(scores)

    matrix_alloc = np.zeros((scores.shape), dtype=np.bool)

    loads_copy = loads.copy()

    best_revs_map = {}
    for a in range(n):
        best_revs_map[a] = best_revs[:, a].tolist()

    # Each paper has a set of papers they need to check with. If they take a reviewer that is worth more to
    # some paper than the smallest value reviewer that other paper has taken, they have to check with that paper from
    # then on.

    # Maintain the invariant that no paper can take a reviewer that is worth more to some other paper
    # than what that paper has chosen in previous rounds. So basically, each round we will construct
    # the vector of the realized values for all papers. Then when you try to select a reviewer, you check
    # if np.any(scores[r, :] > previous_attained_scores). If so, you move on. Actually, it doesn't need to be
    # per round. Suppose that i comes before j in round t. Then suppose i picks something, and my non-per-round
    # update rules out what j was going to pick (because it is worth more to i). If j is allowed to pick this thing, i
    # would have been ok to pick it too. But it didn't.

    previous_attained_scores = np.ones(n) * 1000

    # max_mg_per_agent = np.ones(n) * 1000

    def is_valid_assignment(previous_attained_scores, r, a, alloc, scores):
        papers_to_check_against = set()
        for rev in alloc[a] + [r]:
            papers_to_check_against |= set(np.where(previous_attained_scores < scores[rev, :])[0].tolist())

        for p in papers_to_check_against:
            other = get_valuation(p, alloc[a] + [r], scores) - np.max(scores[alloc[a] + [r], [p]*len(alloc[a] + [r])])
            curr = get_valuation(p, alloc[p], scores)
            if other > curr:
                return False

        return True

    while len(ordering) < np.sum(covs):
        next_agent = None
        next_mg = -10000
        for a in sorted(available_agents):
            if next_mg == max_mg:
                break
            removal_set = []
            for r in best_revs_map[a]:
                if loads_copy[r] <= 0 or r in alloc[a]:
                    removal_set.append(r)
                elif scores[r, a] > next_mg:
                    # This agent might be the greedy choice.
                    # Check if this is a valid assignment, then make it the greedy choice if so.
                    # If not a valid assignment, go to the next reviewer for this agent.
                    if is_valid_assignment(previous_attained_scores, r, a, alloc, scores):
                        next_agent = a
                        next_mg = scores[r, a]
                        break
                else:
                    # This agent cannot be the greedy choice
                    break
            for r in removal_set:
                best_revs_map[a].remove(r)

        new_assn = False
        for r in best_revs_map[next_agent]:
            if loads_copy[r] > 0 and r not in alloc[next_agent]:
                loads_copy[r] -= 1
                alloc[next_agent].append(r)
                matrix_alloc[r, next_agent] = 1
                previous_attained_scores[next_agent] = min(scores[r, next_agent], previous_attained_scores[next_agent])
                new_assn = True
                break

        if not new_assn:
            print("no new assn")
            return alloc, loads_copy, matrix_alloc

        ordering.append(next_agent)
        available_agents.remove(next_agent)

        if len(available_agents) == 0:
            round_num += 1
            available_agents = set(range(n))

    return alloc, ordering


def fairseq(scores, covs, loads):
    best_revs = np.argsort(-1 * scores, axis=0)
    alloc, _ = greedy(scores, loads, covs, best_revs)

    numpy_alloc = np.zeros(scores.shape)

    for p in alloc:
        numpy_alloc[alloc[p], p] = 1

    return numpy_alloc


def fairflow(scores, covs, loads):
    x = FairFlow(loads, np.zeros(loads.shape), covs, scores)
    return x.solve()


def pr4a(pra, covs, loads, iter_limit=1):
    # Normalize the affinities so they're between 0 and 1
    pra[np.where(pra < 0)] = 0
    pra /= np.max(pra)

    pr4a_instance = auto_assigner(pra, demand=int(covs[0]), ability=loads, iter_limit=iter_limit)
    pr4a_instance.fair_assignment()

    alloc = pr4a_instance.fa

    numpy_alloc = np.zeros(pra.shape)

    for p in alloc:
        numpy_alloc[alloc[p], p] = 1

    return numpy_alloc
