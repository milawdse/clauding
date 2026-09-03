"""Reference solutions — Module 3, SAT: CNF, resolution and DPLL."""

from csai import logic
from csai.logic import And, Not, Or, literal_symbol, negate_literal


def to_nnf(formula):
    def strip(f):
        """Rewrite iff/implies away, bottom up."""
        if not isinstance(f, tuple):
            return f
        op, *args = f
        args = [strip(a) for a in args]
        if op == "implies":
            return Or(Not(args[0]), args[1])
        if op == "iff":
            return And(Or(Not(args[0]), args[1]), Or(args[0], Not(args[1])))
        return (op,) + tuple(args)

    def push(f):
        if not isinstance(f, tuple):
            return f
        op, *args = f
        if op != "not":
            return (op,) + tuple(push(a) for a in args)
        inner = args[0]
        if not isinstance(inner, tuple):
            return Not(inner)
        iop, *iargs = inner
        if iop == "not":
            return push(iargs[0])
        if iop == "and":
            return Or(*[push(Not(a)) for a in iargs])
        if iop == "or":
            return And(*[push(Not(a)) for a in iargs])
        raise ValueError(f"unexpected connective under ¬: {iop!r}")

    return push(strip(formula))


def tseitin_and(z, a, b):
    return [
        frozenset({"-" + z, a}),
        frozenset({"-" + z, b}),
        frozenset({z, "-" + a, "-" + b}),
    ]


def resolve(c1, c2, literal):
    if literal not in c1 or negate_literal(literal) not in c2:
        return None
    return frozenset((c1 - {literal}) | (c2 - {negate_literal(literal)}))


def unit_propagate(clauses, assignment):
    clauses = list(clauses)
    assignment = dict(assignment)
    while True:
        if any(len(c) == 0 for c in clauses):
            return None
        unit = next((next(iter(c)) for c in clauses if len(c) == 1), None)
        if unit is None:
            return clauses, assignment
        assignment[literal_symbol(unit)] = not unit.startswith("-")
        clauses = simplify(clauses, unit)


def pure_literals(clauses):
    lits = {l for c in clauses for l in c}
    return sorted(l for l in lits if negate_literal(l) not in lits)


def proves(kb, query):
    refutation = logic.to_clauses(And(logic.as_conjunction(kb), Not(query)))
    return logic.dpll(refutation) is None


# --- project ---------------------------------------------------------------

def solve(clauses, *, propagate=True, pure=True, stats=None):
    if stats is None:
        stats = {}
    for key in ("decisions", "propagations", "conflicts"):
        stats.setdefault(key, 0)

    all_vars = variables(clauses)

    def search(cs, assign):
        if propagate:
            while True:
                if any(len(c) == 0 for c in cs):
                    stats["conflicts"] += 1
                    return None
                unit = next((next(iter(c)) for c in cs if len(c) == 1), None)
                if unit is None:
                    break
                stats["propagations"] += 1
                assign = dict(assign)
                assign[literal_symbol(unit)] = not unit.startswith("-")
                cs = simplify(cs, unit)

        if any(len(c) == 0 for c in cs):
            stats["conflicts"] += 1
            return None
        if not cs:
            return assign

        if pure:
            pures = pure_literals(cs)
            if pures:
                assign = dict(assign)
                for lit in pures:
                    assign[literal_symbol(lit)] = not lit.startswith("-")
                    cs = simplify(cs, lit)
                return search(cs, assign)

        lit = next(iter(next(iter(cs))))
        stats["decisions"] += 1
        for choice in (lit, negate_literal(lit)):
            branch = dict(assign)
            branch[literal_symbol(choice)] = not choice.startswith("-")
            found = search(simplify(cs, choice), branch)
            if found is not None:
                return found
        return None

    model = search([set(c) for c in clauses], {})
    if model is None:
        return None
    return {v: model.get(v, False) for v in all_vars}
