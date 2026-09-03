"""Reference solutions — Module 2, propositional logic."""

import itertools

from csai import logic
from csai.logic import And, Iff, Implies, Not, Or


def truth_value(formula, model):
    if isinstance(formula, bool):
        return formula
    if isinstance(formula, str):
        return bool(model[formula])
    op, *args = formula
    if op == "not":
        return not truth_value(args[0], model)
    if op == "and":
        return all(truth_value(a, model) for a in args)
    if op == "or":
        return any(truth_value(a, model) for a in args)
    if op == "implies":
        return (not truth_value(args[0], model)) or truth_value(args[1], model)
    if op == "iff":
        return truth_value(args[0], model) == truth_value(args[1], model)
    raise ValueError(f"unknown connective {op!r}")


def atoms(formula):
    found = set()

    def walk(f):
        if isinstance(f, str):
            found.add(f)
        elif isinstance(f, tuple):
            for a in f[1:]:
                walk(a)

    walk(formula)
    return sorted(found)


def models_of(symbols):
    symbols = list(symbols)
    return [
        dict(zip(symbols, values))
        for values in itertools.product([False, True], repeat=len(symbols))
    ]


def classify(formula):
    ms = models_of(atoms(formula))
    n_true = sum(truth_value(formula, m) for m in ms)
    if n_true == 0:
        return "unsatisfiable"
    if n_true == len(ms):
        return "valid"
    return "contingent"


def _kb_and_query_symbols(kb, query):
    syms = set(atoms(query))
    for f in kb:
        syms |= set(atoms(f))
    return sorted(syms)


def follows(kb, query):
    for m in models_of(_kb_and_query_symbols(kb, query)):
        if all(truth_value(f, m) for f in kb) and not truth_value(query, m):
            return False
    return True


def why_not(kb, query):
    for m in models_of(_kb_and_query_symbols(kb, query)):
        if all(truth_value(f, m) for f in kb) and not truth_value(query, m):
            return m
    return None


# --- project ---------------------------------------------------------------

def encode(puzzle):
    return [Iff(speaker, statement) for speaker, statement in puzzle.items()]


def _names(puzzle):
    syms = set()
    for speaker, statement in puzzle.items():
        syms.add(speaker)
        syms |= set(atoms(statement))
    return sorted(syms)


def solutions(puzzle):
    formulas = encode(puzzle)
    models = [
        m for m in models_of(_names(puzzle))
        if all(truth_value(f, m) for f in formulas)
    ]
    return sorted(models, key=lambda m: sorted(m.items()))


def identify(puzzle):
    if not solutions(puzzle):
        return {}
    formulas = encode(puzzle)
    verdicts = {}
    for name in _names(puzzle):
        if follows(formulas, name):
            verdicts[name] = "knight"
        elif follows(formulas, Not(name)):
            verdicts[name] = "knave"
        else:
            verdicts[name] = "unknown"
    return verdicts
