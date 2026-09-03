"""Forward chaining over definite clauses, with provenance.

The canonical version of what you build in Module 5. Where `csai.fol` chases
a goal *backwards*, this runs rules *forwards* from what is known until
nothing new can be derived — the fixpoint. It also records, for every derived
fact, which rule produced it and from which premises, which is what lets an
expert system answer "how do you know that?".

Two optimisations are switchable so their value can be measured:

* **indexing** — group facts by predicate, and by their first argument when
  it is known, so matching a goal does not scan the whole fact base. This is
  the essential idea behind RETE's alpha memories.
* **semi-naive evaluation** — each round, require every match to use at
  least one fact derived in the previous round. Any match that doesn't was
  already found earlier, so re-finding it is wasted work.

Terms, rules and unification come from `csai.fol`; only the control strategy
is different.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from csai import fol
from csai.fol import Rule, Term


@dataclass(frozen=True)
class Justification:
    """Why a fact is believed: the rule that fired and what it fired on."""

    fact: Term
    rule: Rule | None            # None for a fact that was given, not derived
    premises: tuple[Term, ...] = ()

    @property
    def given(self) -> bool:
        return self.rule is None


@dataclass
class Derivation:
    """The result of a forward-chaining run."""

    facts: list[Term]
    justifications: dict[Term, Justification] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def __contains__(self, fact: Term) -> bool:
        return fact in self.justifications

    def __len__(self) -> int:
        return len(self.facts)


# --------------------------------------------------------------------------
# Fact stores
# --------------------------------------------------------------------------

class LinearScan:
    """The obvious fact store: every goal is compared against every fact."""

    def __init__(self, facts: Sequence[Term] = ()):
        self.facts: list[Term] = list(facts)

    def add(self, fact: Term) -> None:
        self.facts.append(fact)

    def candidates(self, goal: Term) -> Sequence[Term]:
        return self.facts


class FactIndex(LinearScan):
    """Facts grouped by predicate, and by first argument where it is known.

    A goal `carnivore(X)` with `X` already bound to `stripes` then looks up a
    single bucket instead of scanning. On a fact base with many individuals
    this is the difference between quadratic and nearly linear.
    """

    def __init__(self, facts: Sequence[Term] = ()):
        super().__init__()
        self.by_predicate: dict[tuple, list[Term]] = {}
        self.by_first_arg: dict[tuple, list[Term]] = {}
        for f in facts:
            self.add(f)

    def add(self, fact: Term) -> None:
        super().add(fact)
        if not fol.is_compound(fact):
            return
        key = (fact[0], len(fact))
        self.by_predicate.setdefault(key, []).append(fact)
        if len(fact) > 1:
            self.by_first_arg.setdefault(key + (fact[1],), []).append(fact)

    def candidates(self, goal: Term) -> Sequence[Term]:
        if not fol.is_compound(goal):
            return self.facts
        key = (goal[0], len(goal))
        if len(goal) > 1 and not fol.is_var(goal[1]) and fol.is_ground(goal[1]):
            return self.by_first_arg.get(key + (goal[1],), ())
        return self.by_predicate.get(key, ())


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def match_goals(goals: Sequence[Term], facts: Sequence[Term],
                subst: dict | None = None,
                stats: dict[str, int] | None = None
                ) -> Iterator[tuple[dict, tuple[Term, ...]]]:
    """Every way to satisfy all `goals` against `facts`.

    Yields `(substitution, premises)` — the premises being the specific facts
    used, which is exactly what a justification needs to record. This is the
    simple, list-based version; `forward_chain` uses the store-based one below.
    """
    return _match(goals, LinearScan(facts), subst, stats, None)


def _match(goals: Sequence[Term], store: LinearScan,
           subst: dict | None = None,
           stats: dict[str, int] | None = None,
           sources: Sequence[LinearScan] | None = None
           ) -> Iterator[tuple[dict, tuple[Term, ...]]]:
    subst = {} if subst is None else subst
    if not goals:
        yield subst, ()
        return
    head, rest = goals[0], goals[1:]
    bound = fol.substitute(head, subst)
    source = store if sources is None else sources[0]
    rest_sources = None if sources is None else sources[1:]
    for fact in list(source.candidates(bound)):
        if stats is not None:
            stats["match_attempts"] = stats.get("match_attempts", 0) + 1
        unified = fol.unify(bound, fact, subst)
        if unified is None:
            continue
        for sub, premises in _match(rest, store, unified, stats, rest_sources):
            yield sub, (fact,) + premises


# --------------------------------------------------------------------------
# The fixpoint
# --------------------------------------------------------------------------

def forward_chain(rules: Sequence[Rule], facts: Sequence[Term],
                  *, indexed: bool = True, semi_naive: bool = True,
                  max_rounds: int = 100,
                  stats: dict[str, int] | None = None) -> Derivation:
    """Derive every consequence of `rules` from `facts`.

    The answer does not depend on `indexed` or `semi_naive` — only the amount
    of work does, which is the point of being able to switch them off.
    """
    stats = {} if stats is None else stats
    for key in ("rounds", "match_attempts", "derived"):
        stats.setdefault(key, 0)

    make_store = FactIndex if indexed else LinearScan
    store = make_store()
    known: list[Term] = []
    justifications: dict[Term, Justification] = {}
    for f in facts:
        if f not in justifications:
            known.append(f)
            store.add(f)
            justifications[f] = Justification(f, None, ())

    frontier_store = store
    first_round = True
    for _ in range(max_rounds):
        stats["rounds"] += 1
        new: list[Term] = []
        for rule in rules:
            delta = semi_naive and not first_round
            for subst, premises in _rule_matches(rule, store, frontier_store,
                                                 stats, delta):
                derived = fol.substitute(rule.head, subst)
                if not fol.is_ground(derived) or derived in justifications:
                    continue
                justifications[derived] = Justification(derived, rule, premises)
                new.append(derived)
                stats["derived"] += 1
        if not new:
            break
        for f in new:
            known.append(f)
            store.add(f)
        frontier_store = make_store(new)
        first_round = False
    return Derivation(known, justifications, stats)


def _rule_matches(rule: Rule, store: LinearScan, frontier: LinearScan,
                  stats: dict[str, int], delta: bool):
    """Matches for one rule, optionally requiring a newly derived premise.

    The semi-naive version pins each body goal in turn to the facts derived
    last round, letting the others range over everything known.
    """
    if not delta:
        yield from _match(rule.body, store, None, stats, None)
        return
    for i in range(len(rule.body)):
        sources = [store] * len(rule.body)
        sources[i] = frontier
        yield from _match(rule.body, store, None, stats, sources)


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------

def how(fact: Term, justifications: dict[Term, Justification],
        indent: int = 0) -> str:
    """A derivation tree for `fact`, as indented text."""
    pad = "  " * indent
    j = justifications.get(fact)
    if j is None:
        return f"{pad}{fol.term_str(fact)}   (not derived)"
    if j.given:
        return f"{pad}{fol.term_str(fact)}   (given)"
    lines = [f"{pad}{fol.term_str(fact)}   by {j.rule}"]
    for p in j.premises:
        lines.append(how(p, justifications, indent + 1))
    return "\n".join(lines)


def consequences(fact: Term,
                 justifications: dict[Term, Justification]) -> list[Term]:
    """Everything derived using `fact`, directly or indirectly."""
    out: list[Term] = []
    frontier = [fact]
    while frontier:
        current = frontier.pop()
        for other, j in justifications.items():
            if current in j.premises and other not in out:
                out.append(other)
                frontier.append(other)
    return out
