"""Reference solutions — Module 6, state-space search."""

import heapq
import itertools
from collections import deque

from csai import data, search


def reconstruct(parents, goal):
    if goal not in parents:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    return path[::-1]


def bfs(problem):
    start = problem.initial()
    parents = {start: None}
    expanded = 0
    if problem.is_goal(start):
        return [start], expanded
    frontier = deque([start])
    while frontier:
        state = frontier.popleft()
        expanded += 1
        for action in problem.actions(state):
            nxt = problem.result(state, action)
            if nxt in parents:
                continue
            parents[nxt] = state
            if problem.is_goal(nxt):
                return reconstruct(parents, nxt), expanded
            frontier.append(nxt)
    return None, expanded


def manhattan(state):
    total = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(v - 1, 3)
        total += abs(r1 - r2) + abs(c1 - c2)
    return total


def astar(problem):
    start = problem.initial()
    tie = itertools.count()
    frontier = [(problem.heuristic(start), next(tie), start)]
    best = {start: 0.0}
    parents = {start: None}
    expanded = 0
    while frontier:
        _, _, state = heapq.heappop(frontier)
        if problem.is_goal(state):
            return reconstruct(parents, state), best[state], expanded
        expanded += 1
        for action in problem.actions(state):
            nxt = problem.result(state, action)
            cost = best[state] + problem.step_cost(state, action, nxt)
            if cost < best.get(nxt, float("inf")):
                best[nxt] = cost
                parents[nxt] = state
                heapq.heappush(
                    frontier, (cost + problem.heuristic(nxt), next(tie), nxt))
    return None, float("inf"), expanded


def overestimates(problem_for, states):
    bad = []
    for state in states:
        problem = problem_for(state)
        true_cost = search.uniform_cost(problem).cost
        if problem.heuristic(state) > true_cost:
            bad.append(state)
    return bad


def reachable(problem, start):
    distance = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for action in problem.actions(state):
            nxt = problem.result(state, action)
            if nxt not in distance:
                distance[nxt] = distance[state] + 1
                queue.append(nxt)
    return distance


# --- project ---------------------------------------------------------------

def plan(start_state, goal_state):
    problem = CubeProblem(as_tuple(start_state), as_tuple(goal_state))
    result = search.breadth_first(problem)
    return result.actions if result.found else None


def redundancy_report(examples):
    stated, optimal = [], []
    by_length = {}
    for ex in examples:
        s = data.num_rotations(ex)
        route = plan(data.initial_state(ex), data.gold_states(ex)[-1])
        o = len(route)
        stated.append(s)
        optimal.append(o)
        by_length.setdefault(s, []).append(o)
    n = len(examples)
    return {
        "n": n,
        "mean_stated": sum(stated) / n,
        "mean_optimal": sum(optimal) / n,
        "max_optimal": max(optimal),
        "redundant_fraction": sum(o < s for o, s in zip(optimal, stated)) / n,
        "by_length": {k: sum(v) / len(v) for k, v in sorted(by_length.items())},
    }
