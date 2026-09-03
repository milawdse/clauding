"""Constraint satisfaction: backtracking with inference.

The canonical version of what you build in Module 8. A CSP is variables,
domains, and binary constraints. Solving one is backtracking search with as
much *inference* as you can afford between assignments — which is where all
the leverage is.

Everything is switchable so the notebook can measure each technique:

    solve(csp)                                    # everything on
    solve(csp, inference="none", select="first")  # plain backtracking
    solve(csp, inference="forward")               # forward checking
    solve(csp, inference="ac3")                   # full arc consistency
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable, Sequence

Variable = Any
Value = Any


class CSP:
    """Variables with finite domains and binary constraints.

    `constraint(A, a, B, b)` returns True when `A = a` is compatible with
    `B = b`. It is only ever called for neighbouring pairs.
    """

    def __init__(self, variables: Sequence[Variable],
                 domains: dict[Variable, list[Value]],
                 neighbours: dict[Variable, set],
                 constraint: Callable[[Variable, Value, Variable, Value], bool]):
        self.variables = list(variables)
        self.domains = {v: list(d) for v, d in domains.items()}
        self.neighbours = {v: set(n) for v, n in neighbours.items()}
        self.constraint = constraint

    def consistent(self, var: Variable, value: Value,
                   assignment: dict) -> bool:
        """Does `var = value` violate any constraint with what is assigned?"""
        return all(
            self.constraint(var, value, other, assignment[other])
            for other in self.neighbours[var] if other in assignment
        )

    def is_solution(self, assignment: dict) -> bool:
        if set(assignment) != set(self.variables):
            return False
        return all(
            self.constraint(a, assignment[a], b, assignment[b])
            for a in self.variables for b in self.neighbours[a]
        )


def from_constraints(variables, domains, pairs):
    """Build a CSP from explicit `(A, B, predicate)` triples.

    Convenient when constraints are heterogeneous — as in a logic puzzle,
    where nearly every clue is its own relation.
    """
    table: dict[tuple, list] = {}
    neighbours = {v: set() for v in variables}
    for a, b, pred in pairs:
        table.setdefault((a, b), []).append(pred)
        table.setdefault((b, a), []).append(lambda y, x, p=pred: p(x, y))
        neighbours[a].add(b)
        neighbours[b].add(a)

    def constraint(a, va, b, vb):
        return all(p(va, vb) for p in table.get((a, b), ()))

    return CSP(variables, domains, neighbours, constraint)


def all_different(variables) -> list:
    """Pairwise `!=` constraints — the most common global constraint, decomposed."""
    pairs = []
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            pairs.append((a, b, lambda x, y: x != y))
    return pairs


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def forward_check(csp: CSP, var: Variable, value: Value,
                  domains: dict) -> dict | None:
    """Remove values from unassigned neighbours that clash with `var = value`.

    Returns the pruned domains, or None if a domain is emptied — which means
    this assignment is already doomed, discovered one step earlier than plain
    backtracking would have found out.
    """
    pruned = {v: list(d) for v, d in domains.items()}
    pruned[var] = [value]
    for other in csp.neighbours[var]:
        if len(pruned[other]) == 1 and pruned[other][0] == value:
            pass
        kept = [x for x in pruned[other] if csp.constraint(var, value, other, x)]
        if not kept:
            return None
        pruned[other] = kept
    return pruned


def revise(csp: CSP, xi: Variable, xj: Variable, domains: dict) -> bool:
    """Make `xi` arc-consistent with respect to `xj`. True if anything changed.

    A value of `xi` survives only if *some* value of `xj` is compatible with
    it. Anything with no support is impossible and can go.
    """
    kept = [
        x for x in domains[xi]
        if any(csp.constraint(xi, x, xj, y) for y in domains[xj])
    ]
    if len(kept) == len(domains[xi]):
        return False
    domains[xi] = kept
    return True


def ac3(csp: CSP, domains: dict, stats: dict | None = None) -> bool:
    """Enforce arc consistency over the whole problem. False if a domain empties.

    Every arc goes in a queue. Revising one arc can destroy the support of
    another, so whenever `xi` shrinks, every arc pointing *into* `xi` goes
    back in the queue. Runs in O(e·d³).
    """
    queue = deque((xi, xj) for xi in csp.variables for xj in csp.neighbours[xi])
    while queue:
        xi, xj = queue.popleft()
        if stats is not None:
            stats["revisions"] = stats.get("revisions", 0) + 1
        if revise(csp, xi, xj, domains):
            if not domains[xi]:
                return False
            for xk in csp.neighbours[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True


# --------------------------------------------------------------------------
# Variable and value ordering
# --------------------------------------------------------------------------

def select_first(csp: CSP, assignment: dict, domains: dict) -> Variable:
    return next(v for v in csp.variables if v not in assignment)


def select_mrv(csp: CSP, assignment: dict, domains: dict) -> Variable:
    """Minimum remaining values: fail as early as possible.

    Tie-broken by degree — the variable involved in the most constraints with
    still-unassigned variables.
    """
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: (
        len(domains[v]),
        -sum(1 for n in csp.neighbours[v] if n not in assignment),
    ))


def order_values(csp: CSP, var: Variable, assignment: dict, domains: dict,
                 least_constraining: bool = True) -> list:
    """Least constraining value first: rule out as few neighbours as possible."""
    values = list(domains[var])
    if not least_constraining:
        return values

    def conflicts(value):
        return sum(
            1
            for other in csp.neighbours[var] if other not in assignment
            for x in domains[other]
            if not csp.constraint(var, value, other, x)
        )

    return sorted(values, key=conflicts)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def solve(csp: CSP, *, inference: str = "ac3", select: str = "mrv",
          least_constraining: bool = True, stats: dict | None = None,
          all_solutions: bool = False, max_nodes: int = 2_000_000):
    """Backtracking search. Returns an assignment, or None.

    With `all_solutions=True`, returns a list of every solution instead —
    useful for asking whether a puzzle is uniquely determined.
    """
    stats = {} if stats is None else stats
    for key in ("nodes", "backtracks", "revisions"):
        stats.setdefault(key, 0)

    select_var = {"first": select_first, "mrv": select_mrv}[select]
    solutions: list[dict] = []

    domains = {v: list(d) for v, d in csp.domains.items()}
    if inference == "ac3" and not ac3(csp, domains, stats):
        return [] if all_solutions else None

    def backtrack(assignment, domains):
        if len(assignment) == len(csp.variables):
            solutions.append(dict(assignment))
            return not all_solutions
        if stats["nodes"] > max_nodes:
            return True
        var = select_var(csp, assignment, domains)
        for value in order_values(csp, var, assignment, domains,
                                  least_constraining):
            stats["nodes"] += 1
            if not csp.consistent(var, value, assignment):
                continue
            assignment[var] = value
            if inference == "none":
                new_domains = dict(domains)
                new_domains[var] = [value]
                ok = True
            elif inference == "forward":
                new_domains = forward_check(csp, var, value, domains)
                ok = new_domains is not None
            else:
                new_domains = forward_check(csp, var, value, domains)
                ok = new_domains is not None and ac3(csp, new_domains, stats)
            if ok and backtrack(assignment, new_domains):
                return True
            del assignment[var]
            stats["backtracks"] += 1
        return False

    backtrack({}, domains)
    if all_solutions:
        return solutions
    return solutions[0] if solutions else None
