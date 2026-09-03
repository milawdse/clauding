"""First-order terms, unification, and a backward-chaining engine.

The canonical version of what you build in Module 4.

**Representation.**

* A **variable** is `Var("X")` — a frozen dataclass, so it is hashable and
  can never be confused with a constant.
* A **constant** is a plain `str` or `int`: `"socrates"`, `3`.
* A **compound term** is a tuple `(functor, *args)` with a `str` functor:
  `("parent", "bob", Var("X"))`.
* A **substitution** is a dict from `Var` to term.
* A **rule** is `Rule(head, body)`; a fact is a rule with an empty body.

There is a small parser so notebooks can write `parse("parent(bob, X)")`
and `parse_rule("ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).")` instead of
nesting tuples by hand.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

Term = Any  # Var | str | int | tuple


@dataclass(frozen=True)
class Var:
    """A logic variable. Frozen so it can key a substitution."""

    name: str

    def __repr__(self) -> str:
        return self.name


def is_var(x: Term) -> bool:
    return isinstance(x, Var)


def is_compound(x: Term) -> bool:
    return isinstance(x, tuple) and x and isinstance(x[0], str)


# --------------------------------------------------------------------------
# Substitution
# --------------------------------------------------------------------------

def substitute(term: Term, subst: dict[Var, Term]) -> Term:
    """Apply `subst` to `term`, following chains of bindings to the end."""
    if is_var(term):
        if term in subst:
            return substitute(subst[term], subst)
        return term
    if is_compound(term):
        return (term[0],) + tuple(substitute(a, subst) for a in term[1:])
    return term


def variables_in(term: Term) -> list[Var]:
    """Distinct variables in `term`, in first-appearance order."""
    out: list[Var] = []

    def walk(t: Term) -> None:
        if is_var(t):
            if t not in out:
                out.append(t)
        elif is_compound(t):
            for a in t[1:]:
                walk(a)

    walk(term)
    return out


def is_ground(term: Term) -> bool:
    """True when `term` contains no variables."""
    return not variables_in(term)


# --------------------------------------------------------------------------
# Unification
# --------------------------------------------------------------------------

def occurs(var: Var, term: Term, subst: dict[Var, Term]) -> bool:
    """Does `var` occur inside `term` (after substitution)? The occurs check."""
    term = substitute(term, subst)
    if var == term:
        return True
    if is_compound(term):
        return any(occurs(var, a, subst) for a in term[1:])
    return False


def unify(x: Term, y: Term,
          subst: dict[Var, Term] | None = None,
          *, occurs_check: bool = True) -> dict[Var, Term] | None:
    """Most general unifier of `x` and `y`, extending `subst`, or None.

    The MGU is the *least committal* substitution making the two terms
    identical: it binds a variable only when forced to, so any other unifier
    is an instance of it.
    """
    if subst is None:
        subst = {}
    else:
        subst = dict(subst)

    if is_var(x):
        return _unify_var(x, y, subst, occurs_check)
    if is_var(y):
        return _unify_var(y, x, subst, occurs_check)
    if is_compound(x) and is_compound(y):
        if x[0] != y[0] or len(x) != len(y):
            return None
        for a, b in zip(x[1:], y[1:]):
            result = unify(a, b, subst, occurs_check=occurs_check)
            if result is None:
                return None
            subst = result
        return subst
    return subst if x == y else None


def _unify_var(var: Var, other: Term, subst: dict[Var, Term],
               occurs_check: bool) -> dict[Var, Term] | None:
    if var in subst:
        return unify(subst[var], other, subst, occurs_check=occurs_check)
    if is_var(other) and other in subst:
        return unify(var, subst[other], subst, occurs_check=occurs_check)
    if var == other:
        return subst
    if occurs_check and occurs(var, other, subst):
        return None                      # X = f(X) has no finite solution
    subst = dict(subst)
    subst[var] = other
    return subst


# --------------------------------------------------------------------------
# Rules and the knowledge base
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """`head :- body`. A fact is a rule with an empty body."""

    head: Term
    body: tuple[Term, ...] = ()

    def __repr__(self) -> str:
        if not self.body:
            return f"{term_str(self.head)}."
        return f"{term_str(self.head)} :- {', '.join(map(term_str, self.body))}."


_counter = itertools.count(1)


def rename(rule: Rule, suffix: int | None = None) -> Rule:
    """Fresh copy of `rule` with every variable renamed.

    Without this, using the same rule twice in one proof would accidentally
    identify its variables across uses — the bug every hand-written prover
    hits first.
    """
    if suffix is None:
        suffix = next(_counter)
    mapping = {v: Var(f"{v.name}_{suffix}")
               for v in variables_in((("r",) + (rule.head,) + rule.body))}
    return Rule(substitute(rule.head, mapping),
                tuple(substitute(g, mapping) for g in rule.body))


class KnowledgeBase:
    """A set of definite clauses, queried by backward chaining (SLD)."""

    def __init__(self, rules: Sequence[Rule] = ()):
        self.rules: list[Rule] = list(rules)

    def tell(self, rule: Rule | str) -> "KnowledgeBase":
        self.rules.append(parse_rule(rule) if isinstance(rule, str) else rule)
        return self

    def ask(self, goal: Term | str, *, max_depth: int = 30
            ) -> Iterator[dict[Var, Term]]:
        """Yield one substitution per proof of `goal`."""
        if isinstance(goal, str):
            goal = parse(goal)
        for subst, _ in self.prove([goal], {}, max_depth):
            yield {v: substitute(v, subst) for v in variables_in(goal)}

    def ask_one(self, goal: Term | str, **kw) -> dict[Var, Term] | None:
        return next(self.ask(goal, **kw), None)

    def holds(self, goal: Term | str, **kw) -> bool:
        return self.ask_one(goal, **kw) is not None

    def prove(self, goals: Sequence[Term], subst: dict[Var, Term],
              depth: int, trace: tuple = ()
              ) -> Iterator[tuple[dict[Var, Term], tuple]]:
        """SLD resolution: yield (substitution, trace) per proof of `goals`."""
        if not goals:
            yield subst, trace
            return
        if depth <= 0:
            return
        goal, rest = substitute(goals[0], subst), goals[1:]
        for rule in self.rules:
            fresh = rename(rule)
            unified = unify(fresh.head, goal, subst)
            if unified is None:
                continue
            step = (len(trace), term_str(goal), repr(rule))
            yield from self.prove(tuple(fresh.body) + tuple(rest), unified,
                                  depth - 1, trace + (step,))


# --------------------------------------------------------------------------
# Parsing and printing
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|\d+|:-|[(),.])")


def _tokenize(text: str) -> list[str]:
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                break
            raise SyntaxError(f"cannot tokenize at {text[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def _parse_term(tokens: list[str], i: int) -> tuple[Term, int]:
    tok = tokens[i]
    i += 1
    if tok.isdigit():
        return int(tok), i
    if tok[0].isupper() or tok[0] == "_":
        return Var(tok), i
    if i < len(tokens) and tokens[i] == "(":
        i += 1
        args = []
        while True:
            arg, i = _parse_term(tokens, i)
            args.append(arg)
            if tokens[i] == ",":
                i += 1
                continue
            if tokens[i] == ")":
                return (tok,) + tuple(args), i + 1
            raise SyntaxError(f"expected , or ) near {tokens[i]!r}")
    return tok, i


def parse(text: str) -> Term:
    """`"parent(bob, X)"` -> `("parent", "bob", Var("X"))`."""
    tokens = _tokenize(text.rstrip(" ."))
    term, i = _parse_term(tokens, 0)
    if i != len(tokens):
        raise SyntaxError(f"trailing input: {tokens[i:]}")
    return term


def parse_rule(text: str) -> Rule:
    """`"a(X) :- b(X), c(X)."` -> a Rule. A bare term becomes a fact."""
    text = text.strip().rstrip(".")
    if ":-" not in text:
        return Rule(parse(text))
    head, body = text.split(":-", 1)
    return Rule(parse(head), tuple(parse(g) for g in _split_goals(body)))


def _split_goals(text: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    out, depth, current = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        out.append(current)
    return [g.strip() for g in out if g.strip()]


def parse_program(text: str) -> list[Rule]:
    """Parse a whole program: one clause per `.`-terminated statement."""
    body = re.sub(r"%[^\n]*", "", text)          # strip % comments
    return [parse_rule(c) for c in body.split(".") if c.strip()]


def term_str(term: Term) -> str:
    if is_var(term):
        return term.name
    if is_compound(term):
        return f"{term[0]}({', '.join(term_str(a) for a in term[1:])})"
    return str(term)


def subst_str(subst: dict[Var, Term]) -> str:
    if not subst:
        return "{}"
    return "{" + ", ".join(f"{v.name} = {term_str(t)}"
                           for v, t in sorted(subst.items(),
                                              key=lambda kv: kv[0].name)) + "}"
