"""Propositional logic: syntax, semantics, entailment, CNF, DPLL.

The canonical version of what you build in Modules 2 and 3. Later modules
import from here rather than redefining it.

**Representation.** A formula is either

* a `str` — a propositional symbol, e.g. `"rain"`;
* a `bool` — the constants `True` / `False`;
* a tuple `(connective, *args)` where connective is one of
  `"not"` (1 arg), `"and"` / `"or"` (any number), `"implies"` (2),
  `"iff"` (2).

Plain tuples, not classes, so formulas are hashable, comparable, printable
and pattern-matchable with nothing but the standard library::

    >>> f = Implies("rain", And("wet", Not("dry")))
    >>> evaluate(f, {"rain": True, "wet": True, "dry": False})
    True
    >>> entails([Implies("rain", "wet"), "rain"], "wet")
    True

**A model** is a dict from symbol to bool. `evaluate` requires every symbol
in the formula to be assigned; use `all_models` to enumerate.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Iterator, Sequence

Formula = Any  # str | bool | tuple

CONNECTIVES = ("not", "and", "or", "implies", "iff")


# --------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------

def Not(x: Formula) -> Formula:
    return ("not", x)


def And(*args: Formula) -> Formula:
    return ("and",) + tuple(args)


def Or(*args: Formula) -> Formula:
    return ("or",) + tuple(args)


def Implies(a: Formula, b: Formula) -> Formula:
    return ("implies", a, b)


def Iff(a: Formula, b: Formula) -> Formula:
    return ("iff", a, b)


# --------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------

def evaluate(formula: Formula, model: dict[str, bool]) -> bool:
    """Truth value of `formula` under `model`. Raises on an unassigned symbol."""
    if isinstance(formula, bool):
        return formula
    if isinstance(formula, str):
        try:
            return bool(model[formula])
        except KeyError:
            raise KeyError(f"symbol {formula!r} is not assigned in the model") from None
    op, *args = formula
    if op == "not":
        return not evaluate(args[0], model)
    if op == "and":
        return all(evaluate(a, model) for a in args)
    if op == "or":
        return any(evaluate(a, model) for a in args)
    if op == "implies":
        return (not evaluate(args[0], model)) or evaluate(args[1], model)
    if op == "iff":
        return evaluate(args[0], model) == evaluate(args[1], model)
    raise ValueError(f"unknown connective {op!r}")


def symbols(formula: Formula) -> list[str]:
    """Every propositional symbol in `formula`, in sorted order."""
    found: set[str] = set()

    def walk(f: Formula) -> None:
        if isinstance(f, str):
            found.add(f)
        elif isinstance(f, tuple):
            for a in f[1:]:
                walk(a)

    walk(formula)
    return sorted(found)


def all_models(syms: Sequence[str]) -> Iterator[dict[str, bool]]:
    """Every one of the 2**n assignments over `syms`."""
    syms = list(syms)
    for values in itertools.product([False, True], repeat=len(syms)):
        yield dict(zip(syms, values))


def as_conjunction(kb: Formula | Iterable[Formula]) -> Formula:
    """Accept a formula or a list of formulas; return one formula."""
    if isinstance(kb, (str, bool, tuple)):
        return kb
    items = list(kb)
    return And(*items) if items else True


def satisfying_models(kb: Formula | Iterable[Formula]) -> list[dict[str, bool]]:
    """All models of `kb` (which may be a list of premises)."""
    f = as_conjunction(kb)
    return [m for m in all_models(symbols(f)) if evaluate(f, m)]


def is_satisfiable(kb: Formula | Iterable[Formula]) -> bool:
    f = as_conjunction(kb)
    return any(evaluate(f, m) for m in all_models(symbols(f)))


def is_valid(formula: Formula) -> bool:
    """True when the formula holds in every model (a tautology)."""
    return all(evaluate(formula, m) for m in all_models(symbols(formula)))


def entails(kb: Formula | Iterable[Formula], query: Formula) -> bool:
    """KB ⊨ query: every model of KB is a model of query."""
    f = And(as_conjunction(kb), True)
    syms = sorted(set(symbols(f)) | set(symbols(query)))
    return all(
        evaluate(query, m) for m in all_models(syms) if evaluate(f, m)
    )


def counterexample(kb: Formula | Iterable[Formula],
                   query: Formula) -> dict[str, bool] | None:
    """A model satisfying KB but falsifying `query`, or None if KB ⊨ query."""
    f = as_conjunction(kb)
    syms = sorted(set(symbols(f)) | set(symbols(query)))
    for m in all_models(syms):
        if evaluate(f, m) and not evaluate(query, m):
            return m
    return None


# --------------------------------------------------------------------------
# Conjunctive normal form
# --------------------------------------------------------------------------

def eliminate_iff_implies(f: Formula) -> Formula:
    """Rewrite `iff` and `implies` in terms of and/or/not."""
    if not isinstance(f, tuple):
        return f
    op, *args = f
    args = [eliminate_iff_implies(a) for a in args]
    if op == "implies":
        return Or(Not(args[0]), args[1])
    if op == "iff":
        return And(Or(Not(args[0]), args[1]), Or(args[0], Not(args[1])))
    return (op,) + tuple(args)


def push_negations(f: Formula) -> Formula:
    """Move negations inward (De Morgan) until they sit on symbols only."""
    if not isinstance(f, tuple):
        return f
    op, *args = f
    if op != "not":
        return (op,) + tuple(push_negations(a) for a in args)
    inner = args[0]
    if isinstance(inner, bool):
        return not inner
    if isinstance(inner, str):
        return Not(inner)
    iop, *iargs = inner
    if iop == "not":
        return push_negations(iargs[0])
    if iop == "and":
        return Or(*[push_negations(Not(a)) for a in iargs])
    if iop == "or":
        return And(*[push_negations(Not(a)) for a in iargs])
    return push_negations(Not(eliminate_iff_implies(inner)))


def distribute_or(f: Formula) -> Formula:
    """Distribute `or` over `and`, the step that can blow up exponentially."""
    if not isinstance(f, tuple):
        return f
    op, *args = f
    args = [distribute_or(a) for a in args]
    if op == "and":
        return And(*_flatten("and", args))
    if op != "or":
        return (op,) + tuple(args)
    args = _flatten("or", args)
    for i, a in enumerate(args):
        if isinstance(a, tuple) and a[0] == "and":
            rest = args[:i] + args[i + 1:]
            return distribute_or(And(*[Or(c, *rest) for c in a[1:]]))
    return Or(*args)


def _flatten(op: str, args: list[Formula]) -> list[Formula]:
    out: list[Formula] = []
    for a in args:
        if isinstance(a, tuple) and a[0] == op:
            out.extend(a[1:])
        else:
            out.append(a)
    return out


def to_cnf(f: Formula) -> Formula:
    """Equivalent formula as a conjunction of disjunctions of literals."""
    return distribute_or(push_negations(eliminate_iff_implies(f)))


def to_clauses(f: Formula) -> list[frozenset[str]]:
    """CNF as a clause set: each clause a frozenset of literals.

    A literal is a symbol `"p"` or its negation `"-p"`. Clauses containing
    both `p` and `-p` are tautologies and are dropped.
    """
    cnf = to_cnf(f)
    conjuncts = cnf[1:] if isinstance(cnf, tuple) and cnf[0] == "and" else [cnf]
    clauses = []
    for c in conjuncts:
        if isinstance(c, bool):
            if c:
                continue
            clauses.append(frozenset())
            continue
        lits = c[1:] if isinstance(c, tuple) and c[0] == "or" else [c]
        clause = set()
        skip = False
        for lit in lits:
            if isinstance(lit, bool):
                if lit:
                    skip = True
                continue
            clause.add(literal_name(lit))
        if not skip:
            if any(negate_literal(l) in clause for l in clause):
                continue
            clauses.append(frozenset(clause))
    return clauses


def literal_name(lit: Formula) -> str:
    """`"p"` for a symbol, `"-p"` for its negation."""
    if isinstance(lit, str):
        return lit
    if isinstance(lit, tuple) and lit[0] == "not" and isinstance(lit[1], str):
        return "-" + lit[1]
    raise ValueError(f"not a literal: {lit!r}")


def negate_literal(lit: str) -> str:
    return lit[1:] if lit.startswith("-") else "-" + lit


def literal_symbol(lit: str) -> str:
    return lit[1:] if lit.startswith("-") else lit


# --------------------------------------------------------------------------
# DPLL
# --------------------------------------------------------------------------

def dpll(clauses: Sequence[frozenset[str]],
         assignment: dict[str, bool] | None = None,
         stats: dict[str, int] | None = None) -> dict[str, bool] | None:
    """Davis-Putnam-Logemann-Loveland satisfiability search.

    Returns a satisfying assignment, or None if the clause set is
    unsatisfiable. `stats` (if given) accumulates counts of decisions, unit
    propagations and conflicts.
    """
    assignment = dict(assignment or {})
    clauses = [set(c) for c in clauses]
    if stats is None:
        stats = {}
    stats.setdefault("decisions", 0)
    stats.setdefault("propagations", 0)
    stats.setdefault("conflicts", 0)

    def simplify(cs, lit):
        out = []
        for c in cs:
            if lit in c:
                continue                      # clause satisfied
            neg = negate_literal(lit)
            out.append(c - {neg} if neg in c else c)
        return out

    def search(cs, assign):
        while True:
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
        lit = next(iter(next(iter(cs))))
        stats["decisions"] += 1
        for choice in (lit, negate_literal(lit)):
            assign2 = dict(assign)
            assign2[literal_symbol(choice)] = not choice.startswith("-")
            result = search(simplify(cs, choice), assign2)
            if result is not None:
                return result
        return None

    return search(clauses, assignment)


def sat(formula: Formula | Iterable[Formula]) -> dict[str, bool] | None:
    """Satisfying model for a formula (or list of premises) via CNF + DPLL."""
    f = as_conjunction(formula)
    model = dpll(to_clauses(f))
    if model is None:
        return None
    for s in symbols(f):
        model.setdefault(s, False)
    return model


def entails_by_refutation(kb: Formula | Iterable[Formula],
                          query: Formula) -> bool:
    """KB ⊨ query iff KB ∧ ¬query is unsatisfiable."""
    return sat(And(as_conjunction(kb), Not(query))) is None


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

SYMBOLS_PRETTY = {"not": "¬", "and": " ∧ ", "or": " ∨ ",
                  "implies": " → ", "iff": " ↔ "}


def to_str(f: Formula) -> str:
    """Readable infix rendering."""
    if isinstance(f, bool):
        return "⊤" if f else "⊥"
    if isinstance(f, str):
        return f
    op, *args = f
    if op == "not":
        inner = to_str(args[0])
        return f"¬{inner}" if _atomic(args[0]) else f"¬({inner})"
    if not args:
        return "⊤" if op == "and" else "⊥"   # empty conjunction / disjunction
    joined = SYMBOLS_PRETTY[op].join(
        to_str(a) if _atomic(a) else f"({to_str(a)})" for a in args
    )
    return joined


def _atomic(f: Formula) -> bool:
    return isinstance(f, (str, bool)) or (isinstance(f, tuple) and f[0] == "not")
