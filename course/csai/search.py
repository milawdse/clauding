"""State-space search: the classical algorithms, instrumented.

The canonical version of what you build in Module 6. Every algorithm here is
the same loop over a frontier — the only thing that changes is *what comes
out of the frontier next*:

| algorithm | frontier order |
|---|---|
| breadth-first | FIFO (shallowest first) |
| depth-first | LIFO (deepest first) |
| uniform cost | lowest `g` — cost so far |
| greedy best-first | lowest `h` — estimated cost remaining |
| A* | lowest `g + h` |

Define a problem by subclassing `Problem`, and every algorithm applies.
Each returns a `SearchResult` carrying the path, the cost, and the counters
that let you compare strategies honestly.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

State = Any
Action = Any


class Problem:
    """A search problem. Subclass and override.

    States must be **hashable** — tuples, strings, frozensets — because every
    algorithm here keeps a set of states already reached.
    """

    def initial(self) -> State:
        raise NotImplementedError

    def is_goal(self, state: State) -> bool:
        raise NotImplementedError

    def actions(self, state: State) -> Iterable[Action]:
        raise NotImplementedError

    def result(self, state: State, action: Action) -> State:
        raise NotImplementedError

    def step_cost(self, state: State, action: Action, nxt: State) -> float:
        return 1.0

    def heuristic(self, state: State) -> float:
        """Estimated cost from `state` to the nearest goal. 0 = uninformed."""
        return 0.0


@dataclass
class SearchResult:
    """What a search found, and what it cost to find it."""

    path: list[State] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    cost: float = 0.0
    expanded: int = 0          # nodes taken off the frontier and expanded
    generated: int = 0         # successor states created
    max_frontier: int = 0
    found: bool = False

    @property
    def depth(self) -> int:
        return len(self.actions)

    def __bool__(self) -> bool:
        return self.found

    def render(self) -> str:
        if not self.found:
            return (f"no solution   expanded={self.expanded} "
                    f"generated={self.generated}")
        return (f"solved in {self.depth} steps, cost {self.cost:g}   "
                f"expanded={self.expanded} generated={self.generated} "
                f"max frontier={self.max_frontier}")

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class Node:
    state: State
    parent: "Node | None" = None
    action: Action = None
    cost: float = 0.0
    depth: int = 0


def path_of(node: Node) -> tuple[list[State], list[Action]]:
    """Walk parent pointers back to the root."""
    states, actions = [], []
    while node is not None:
        states.append(node.state)
        if node.action is not None:
            actions.append(node.action)
        node = node.parent
    return states[::-1], actions[::-1]


def _result(node: Node, expanded: int, generated: int, max_frontier: int
            ) -> SearchResult:
    states, actions = path_of(node)
    return SearchResult(states, actions, node.cost, expanded, generated,
                        max_frontier, True)


def breadth_first(problem: Problem, *, max_expansions: int = 1_000_000
                  ) -> SearchResult:
    """Shallowest first. Optimal when every step costs the same."""
    start = Node(problem.initial())
    if problem.is_goal(start.state):
        return _result(start, 0, 0, 1)
    frontier = deque([start])
    reached = {start.state}
    expanded = generated = 0
    peak = 1
    while frontier:
        node = frontier.popleft()
        expanded += 1
        if expanded > max_expansions:
            break
        for action in problem.actions(node.state):
            nxt = problem.result(node.state, action)
            generated += 1
            if nxt in reached:
                continue
            child = Node(nxt, node, action,
                         node.cost + problem.step_cost(node.state, action, nxt),
                         node.depth + 1)
            if problem.is_goal(nxt):
                return _result(child, expanded, generated, peak)
            reached.add(nxt)
            frontier.append(child)
        peak = max(peak, len(frontier))
    return SearchResult(expanded=expanded, generated=generated, max_frontier=peak)


def best_first(problem: Problem, priority, *,
               max_expansions: int = 1_000_000) -> SearchResult:
    """Generic best-first search. `priority(node)` decides what comes next.

    `priority = lambda n: n.cost` gives uniform cost; `n.cost + h(n.state)`
    gives A*; `h(n.state)` alone gives greedy best-first.
    """
    start = Node(problem.initial())
    tie = itertools.count()
    frontier = [(priority(start), next(tie), start)]
    best_cost = {start.state: 0.0}
    expanded = generated = 0
    peak = 1
    while frontier:
        _, _, node = heapq.heappop(frontier)
        if problem.is_goal(node.state):
            return _result(node, expanded, generated, peak)
        if node.cost > best_cost.get(node.state, float("inf")):
            continue                       # a cheaper route was already found
        expanded += 1
        if expanded > max_expansions:
            break
        for action in problem.actions(node.state):
            nxt = problem.result(node.state, action)
            generated += 1
            cost = node.cost + problem.step_cost(node.state, action, nxt)
            if cost < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = cost
                child = Node(nxt, node, action, cost, node.depth + 1)
                heapq.heappush(frontier, (priority(child), next(tie), child))
        peak = max(peak, len(frontier))
    return SearchResult(expanded=expanded, generated=generated, max_frontier=peak)


def uniform_cost(problem: Problem, **kw) -> SearchResult:
    """Dijkstra: cheapest path first. Optimal for any non-negative costs."""
    return best_first(problem, lambda n: n.cost, **kw)


def greedy_best_first(problem: Problem, **kw) -> SearchResult:
    """Follow the heuristic alone. Fast, and not optimal."""
    return best_first(problem, lambda n: problem.heuristic(n.state), **kw)


def astar(problem: Problem, **kw) -> SearchResult:
    """f = g + h. Optimal whenever the heuristic never overestimates."""
    return best_first(problem, lambda n: n.cost + problem.heuristic(n.state), **kw)


def depth_limited(problem: Problem, limit: int, *,
                  max_expansions: int = 1_000_000) -> SearchResult:
    """Depth-first to a fixed depth. Cheap memory, no optimality."""
    expanded = generated = 0

    def recurse(node: Node, depth: int, on_path: frozenset) -> Node | None:
        nonlocal expanded, generated
        if problem.is_goal(node.state):
            return node
        if depth == 0 or expanded > max_expansions:
            return None
        expanded += 1
        for action in problem.actions(node.state):
            nxt = problem.result(node.state, action)
            generated += 1
            if nxt in on_path:
                continue                    # don't cycle within this branch
            child = Node(nxt, node, action,
                         node.cost + problem.step_cost(node.state, action, nxt),
                         node.depth + 1)
            found = recurse(child, depth - 1, on_path | {nxt})
            if found is not None:
                return found
        return None

    start = Node(problem.initial())
    found = recurse(start, limit, frozenset({start.state}))
    if found is None:
        return SearchResult(expanded=expanded, generated=generated)
    return _result(found, expanded, generated, limit + 1)


def iterative_deepening(problem: Problem, *, max_depth: int = 50,
                        **kw) -> SearchResult:
    """Depth-limited search at 0, 1, 2, … — BFS's optimality, DFS's memory."""
    expanded = generated = 0
    for limit in range(max_depth + 1):
        result = depth_limited(problem, limit, **kw)
        expanded += result.expanded
        generated += result.generated
        if result.found:
            result.expanded, result.generated = expanded, generated
            return result
    return SearchResult(expanded=expanded, generated=generated)


ALGORITHMS = {
    "breadth-first": breadth_first,
    "uniform cost": uniform_cost,
    "iterative deepening": iterative_deepening,
    "greedy best-first": greedy_best_first,
    "A*": astar,
}


def effective_branching_factor(expanded: int, depth: int) -> float:
    """The b* satisfying  N ≈ b*^1 + … + b*^d.  Lower is a better heuristic."""
    if depth <= 0 or expanded <= 0:
        return float("nan")
    lo, hi = 1.0000001, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        total = sum(mid ** i for i in range(1, depth + 1))
        if total < expanded:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
