"""Reference solutions — Module 9, STRIPS planning."""

import itertools

from csai import fol, planning as pl, search
from csai.planning import Action


def applicable(action, state):
    return action.preconditions <= state


def apply_action(action, state):
    return frozenset((set(state) - set(action.delete)) | set(action.add))


def ground(schema, objects):
    actions = []
    for combo in itertools.product(objects, repeat=len(schema.parameters)):
        if len(set(combo)) != len(combo):
            continue
        subst = dict(zip(schema.parameters, combo))
        actions.append(Action(
            schema.name, combo,
            frozenset(fol.substitute(p, subst) for p in schema.preconditions),
            frozenset(fol.substitute(p, subst) for p in schema.add),
            frozenset(fol.substitute(p, subst) for p in schema.delete),
        ))
    return actions


def check_plan(problem, plan):
    state = problem.initial
    for i, action in enumerate(plan, start=1):
        if not applicable(action, state):
            return False, i, frozenset(action.preconditions - state)
        state = apply_action(action, state)
    if not problem.goal <= state:
        return False, len(plan), frozenset(problem.goal - state)
    return True, None, frozenset()


def relaxed_reach(problem, state):
    reached = set(state)
    changed = True
    while changed:
        changed = False
        for action in problem.actions:
            if action.preconditions <= reached and not action.add <= reached:
                reached |= action.add
                changed = True
    return frozenset(reached)


def possibly_solvable(problem, state):
    return problem.goal <= relaxed_reach(problem, state)


# --- project ---------------------------------------------------------------

def find_plan(problem, heuristic="h_ff", max_expansions=40000):
    h = pl.HEURISTICS[heuristic]

    class Adapter(search.Problem):
        def initial(self):
            return problem.initial

        def is_goal(self, state):
            return problem.achieved(state)

        def actions(self, state):
            return problem.applicable(state)

        def result(self, state, action):
            return action.apply(state)

        def heuristic(self, state):
            return h(problem, state)

    result = search.astar(Adapter(), max_expansions=max_expansions)
    return {
        "plan": result.actions if result.found else None,
        "length": result.depth if result.found else None,
        "expanded": result.expanded,
        "found": result.found,
    }


def grade(problem, candidates):
    by_key = {(a.name, a.args): a for a in problem.actions}
    reports = {}
    for name, steps in candidates.items():
        state = problem.initial
        report = {"valid": True, "failed_step": None, "reason": "ok",
                  "missing": frozenset(), "length": len(steps)}
        for i, step in enumerate(steps, start=1):
            action = by_key.get((step[0], tuple(step[1:])))
            if action is None:
                report.update(valid=False, failed_step=i,
                              reason="no such action", missing=frozenset())
                break
            if not applicable(action, state):
                report.update(valid=False, failed_step=i,
                              reason="precondition",
                              missing=frozenset(action.preconditions - state))
                break
            state = apply_action(action, state)
        else:
            if not problem.goal <= state:
                report.update(valid=False, failed_step=len(steps) or None,
                              reason="goal not reached",
                              missing=frozenset(problem.goal - state))
        reports[name] = report
    return reports
