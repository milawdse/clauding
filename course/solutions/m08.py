"""Reference solutions — Module 8, constraint satisfaction."""

from collections import deque


def consistent(csp, var, value, assignment):
    return all(
        csp.constraint(var, value, other, assignment[other])
        for other in csp.neighbours[var] if other in assignment
    )


def backtracking(csp):
    def recurse(assignment):
        if len(assignment) == len(csp.variables):
            return dict(assignment)
        var = next(v for v in csp.variables if v not in assignment)
        for value in csp.domains[var]:
            if not consistent(csp, var, value, assignment):
                continue
            assignment[var] = value
            found = recurse(assignment)
            if found is not None:
                return found
            del assignment[var]
        return None

    return recurse({})


def mrv(csp, assignment, domains):
    unassigned = [v for v in csp.variables if v not in assignment]
    if not unassigned:
        return None
    return min(unassigned, key=lambda v: (
        len(domains[v]),
        -sum(1 for n in csp.neighbours[v] if n not in assignment),
    ))


def forward(csp, var, value, domains):
    pruned = {v: list(d) for v, d in domains.items()}
    pruned[var] = [value]
    for other in csp.neighbours[var]:
        kept = [x for x in pruned[other]
                if csp.constraint(var, value, other, x)]
        if not kept:
            return None
        pruned[other] = kept
    return pruned


def revise(csp, xi, xj, domains):
    kept = [x for x in domains[xi]
            if any(csp.constraint(xi, x, xj, y) for y in domains[xj])]
    if len(kept) == len(domains[xi]):
        return False
    domains[xi] = kept
    return True


def arc_consistency(csp, domains):
    queue = deque((xi, xj) for xi in csp.variables for xj in csp.neighbours[xi])
    while queue:
        xi, xj = queue.popleft()
        if revise(csp, xi, xj, domains):
            if not domains[xi]:
                return False
            for xk in csp.neighbours[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True


# --- project ---------------------------------------------------------------

def solve_csp(csp, *, inference="ac3", use_mrv=True, stats=None):
    stats = {} if stats is None else stats
    stats.setdefault("nodes", 0)
    stats.setdefault("backtracks", 0)

    def recurse(assignment, domains):
        if len(assignment) == len(csp.variables):
            return dict(assignment)
        if use_mrv:
            var = mrv(csp, assignment, domains)
        else:
            var = next(v for v in csp.variables if v not in assignment)
        for value in domains[var]:
            stats["nodes"] += 1
            if not consistent(csp, var, value, assignment):
                continue
            assignment[var] = value
            if inference == "none":
                new_domains = dict(domains)
                new_domains[var] = [value]
            elif inference == "forward":
                new_domains = forward(csp, var, value, domains)
            else:
                new_domains = forward(csp, var, value, domains)
                if new_domains is not None and not arc_consistency(csp, new_domains):
                    new_domains = None
            if new_domains is not None:
                found = recurse(assignment, new_domains)
                if found is not None:
                    return found
            del assignment[var]
            stats["backtracks"] += 1
        return None

    domains = {v: list(d) for v, d in csp.domains.items()}
    if inference == "ac3" and not arc_consistency(csp, domains):
        return None
    return recurse({}, domains)


def zebra_answers(assignment):
    nations = CATEGORIES["nationality"]
    return {
        "water": next(n for n in nations
                      if assignment[n] == assignment["water"]),
        "zebra": next(n for n in nations
                      if assignment[n] == assignment["zebra"]),
    }


def compare(csp, configurations):
    results = {}
    for label, kwargs in configurations:
        stats = {}
        found = solve_csp(csp, stats=stats, **kwargs)
        results[label] = {
            "nodes": stats["nodes"],
            "backtracks": stats["backtracks"],
            "solved": found is not None and csp.is_solution(found),
        }
    return results
