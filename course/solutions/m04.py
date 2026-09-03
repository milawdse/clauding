"""Reference solutions — Module 4, first-order logic and mini-Prolog."""

import itertools

from csai import fol
from csai.fol import Rule, Var, is_compound, is_var, parse


def apply_subst(term, subst):
    if is_var(term):
        return apply_subst(subst[term], subst) if term in subst else term
    if is_compound(term):
        return (term[0],) + tuple(apply_subst(a, subst) for a in term[1:])
    return term


def occurs_in(var, term, subst):
    term = apply_subst(term, subst)
    if var == term:
        return True
    if is_compound(term):
        return any(occurs_in(var, a, subst) for a in term[1:])
    return False


def mgu(x, y, subst=None):
    subst = {} if subst is None else dict(subst)

    def unify_var(v, t, s):
        if v in s:
            return mgu(s[v], t, s)
        if is_var(t) and t in s:
            return mgu(v, s[t], s)
        if v == t:
            return s
        if occurs_in(v, t, s):
            return None
        s = dict(s)
        s[v] = t
        return s

    if is_var(x):
        return unify_var(x, y, subst)
    if is_var(y):
        return unify_var(y, x, subst)
    if is_compound(x) and is_compound(y):
        if x[0] != y[0] or len(x) != len(y):
            return None
        for a, b in zip(x[1:], y[1:]):
            subst = mgu(a, b, subst)
            if subst is None:
                return None
        return subst
    return subst if x == y else None


def rename_rule(rule, n):
    terms = (rule.head,) + tuple(rule.body)
    mapping = {}
    for t in terms:
        for v in fol.variables_in(t):
            mapping.setdefault(v, Var(f"{v.name}_{n}"))
    return Rule(apply_subst(rule.head, mapping),
                tuple(apply_subst(g, mapping) for g in rule.body))


def ground(term):
    if is_var(term):
        return False
    if is_compound(term):
        return all(ground(a) for a in term[1:])
    return True


def unify_all(pairs, subst=None):
    subst = {} if subst is None else dict(subst)
    for a, b in pairs:
        subst = mgu(a, b, subst)
        if subst is None:
            return None
    return subst


# --- project ---------------------------------------------------------------

_fresh = itertools.count(1)


def solve(rules, goals, subst=None, depth=20):
    subst = {} if subst is None else subst
    if not goals:
        yield subst
        return
    if depth <= 0:
        return
    goal = apply_subst(goals[0], subst)
    rest = tuple(goals[1:])
    for rule in rules:
        fresh = rename_rule(rule, next(_fresh))
        unified = mgu(fresh.head, goal, subst)
        if unified is None:
            continue
        yield from solve(rules, tuple(fresh.body) + rest, unified, depth - 1)


def ask(rules, goal_text, depth=20):
    goal = parse(goal_text) if isinstance(goal_text, str) else goal_text
    wanted = fol.variables_in(goal)
    return [
        {v: apply_subst(v, s) for v in wanted}
        for s in solve(rules, [goal], {}, depth)
    ]
