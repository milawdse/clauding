"""Discrete probability, Bayesian networks, exact and approximate inference.

The canonical version of what you build in Module 10.

A **Bayesian network** is a directed acyclic graph of variables, each with a
conditional probability table given its parents. It encodes a full joint
distribution in space exponential in the largest *family* rather than in the
number of variables — which is the whole reason it is usable.

    net = BayesNet([
        Variable("burglary", [], {(): {True: 0.001, False: 0.999}}),
        Variable("alarm", ["burglary"], {
            (True,):  {True: 0.94, False: 0.06},
            (False,): {True: 0.001, False: 0.999},
        }),
    ])
    enumeration_ask(net, "burglary", {"alarm": True})
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

Value = Any
Assignment = dict


@dataclass
class Variable:
    """A node: its name, its parents, and P(this | parents) as a table."""

    name: str
    parents: tuple
    cpt: dict            # {(parent values...): {value: probability}}

    def __post_init__(self):
        self.parents = tuple(self.parents)
        self.cpt = {tuple(k) if isinstance(k, tuple) else (k,): dict(v)
                    for k, v in self.cpt.items()}

    @property
    def domain(self) -> tuple:
        return tuple(next(iter(self.cpt.values())))

    def probability(self, value: Value, assignment: Assignment) -> float:
        """P(this = value | the parents' values in `assignment`)."""
        key = tuple(assignment[p] for p in self.parents)
        return self.cpt[key][value]

    def sample(self, assignment: Assignment, rng: random.Random) -> Value:
        row = self.cpt[tuple(assignment[p] for p in self.parents)]
        r = rng.random()
        total = 0.0
        for value, p in row.items():
            total += p
            if r <= total:
                return value
        return list(row)[-1]


class BayesNet:
    """A DAG of `Variable`s, held in topological order."""

    def __init__(self, variables: Sequence[Variable]):
        self.variables = list(variables)
        self.by_name = {v.name: v for v in self.variables}
        self._check_topological()

    def _check_topological(self):
        seen: set = set()
        for v in self.variables:
            missing = [p for p in v.parents if p not in seen]
            if missing:
                raise ValueError(
                    f"{v.name} is listed before its parent(s) {missing}; "
                    "variables must be in topological order")
            seen.add(v.name)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def domain(self, name: str) -> tuple:
        return self.by_name[name].domain

    def probability(self, name: str, value: Value,
                    assignment: Assignment) -> float:
        return self.by_name[name].probability(value, assignment)

    def joint(self, assignment: Assignment) -> float:
        """P(assignment) by the chain rule — one factor per variable."""
        p = 1.0
        for v in self.variables:
            p *= v.probability(assignment[v.name], assignment)
        return p

    def markov_blanket_size(self) -> int:
        return max(len(v.parents) for v in self.variables)


def normalize(distribution: dict) -> dict:
    """Scale a dict of non-negative weights to sum to 1."""
    total = sum(distribution.values())
    if total == 0:
        n = len(distribution)
        return {k: 1.0 / n for k in distribution}
    return {k: v / total for k, v in distribution.items()}


# --------------------------------------------------------------------------
# Exact inference by enumeration
# --------------------------------------------------------------------------

def enumeration_ask(net: BayesNet, query: str, evidence: Assignment,
                    stats: dict | None = None) -> dict:
    """P(query | evidence), by summing over every completion of the evidence.

    Correct, and exponential in the number of unobserved variables.
    """
    distribution = {}
    for value in net.domain(query):
        extended = dict(evidence)
        extended[query] = value
        distribution[value] = _enumerate_all(net, net.names, extended, stats)
    return normalize(distribution)


def _enumerate_all(net: BayesNet, names: Sequence[str],
                   assignment: Assignment, stats: dict | None) -> float:
    if not names:
        return 1.0
    if stats is not None:
        stats["terms"] = stats.get("terms", 0) + 1
    first, rest = names[0], names[1:]
    if first in assignment:
        return (net.probability(first, assignment[first], assignment)
                * _enumerate_all(net, rest, assignment, stats))
    total = 0.0
    for value in net.domain(first):
        extended = dict(assignment)
        extended[first] = value
        total += (net.probability(first, value, extended)
                  * _enumerate_all(net, rest, extended, stats))
    return total


# --------------------------------------------------------------------------
# Exact inference by variable elimination
# --------------------------------------------------------------------------

@dataclass
class Factor:
    """A function from assignments of `variables` to non-negative numbers."""

    variables: tuple
    table: dict = field(default_factory=dict)   # {(values...): number}

    def value(self, assignment: Assignment) -> float:
        return self.table[tuple(assignment[v] for v in self.variables)]

    def __repr__(self) -> str:
        return f"Factor({', '.join(self.variables)})"


def make_factor(net: BayesNet, name: str, evidence: Assignment) -> Factor:
    """The CPT of `name`, restricted to what the evidence fixes."""
    variable = net.by_name[name]
    scope = tuple(v for v in (name,) + variable.parents if v not in evidence)
    table = {}
    domains = [net.domain(v) for v in scope]
    for combo in itertools.product(*domains):
        assignment = dict(evidence)
        assignment.update(dict(zip(scope, combo)))
        table[combo] = variable.probability(assignment[name], assignment)
    return Factor(scope, table)


def pointwise_product(f1: Factor, f2: Factor, net: BayesNet) -> Factor:
    """Multiply two factors, joining on the variables they share."""
    scope = tuple(f1.variables) + tuple(v for v in f2.variables
                                        if v not in f1.variables)
    table = {}
    for combo in itertools.product(*[net.domain(v) for v in scope]):
        assignment = dict(zip(scope, combo))
        table[combo] = f1.value(assignment) * f2.value(assignment)
    return Factor(scope, table)


def sum_out(factor: Factor, name: str, net: BayesNet) -> Factor:
    """Marginalise `name` out of a factor."""
    if name not in factor.variables:
        return factor
    scope = tuple(v for v in factor.variables if v != name)
    table: dict = {}
    for combo in itertools.product(*[net.domain(v) for v in scope]):
        assignment = dict(zip(scope, combo))
        total = 0.0
        for value in net.domain(name):
            assignment[name] = value
            total += factor.value(assignment)
        table[combo] = total
    return Factor(scope, table)


def variable_elimination(net: BayesNet, query: str, evidence: Assignment,
                         stats: dict | None = None) -> dict:
    """P(query | evidence), eliminating hidden variables one at a time.

    Same answer as enumeration, but intermediate results are reused rather
    than recomputed — the difference between exponential and, on a
    well-shaped network, linear.
    """
    factors = [make_factor(net, name, evidence) for name in net.names]
    hidden = [n for n in reversed(net.names)
              if n != query and n not in evidence]
    for name in hidden:
        relevant = [f for f in factors if name in f.variables]
        if not relevant:
            continue
        factors = [f for f in factors if name not in f.variables]
        product = relevant[0]
        for f in relevant[1:]:
            product = pointwise_product(product, f, net)
            if stats is not None:
                stats["products"] = stats.get("products", 0) + 1
        factors.append(sum_out(product, name, net))
    product = factors[0]
    for f in factors[1:]:
        product = pointwise_product(product, f, net)
    return normalize({v: product.table[(v,)] for v in net.domain(query)})


# --------------------------------------------------------------------------
# Approximate inference
# --------------------------------------------------------------------------

def prior_sample(net: BayesNet, rng: random.Random) -> Assignment:
    """One sample from the joint, drawn parents-first."""
    assignment: Assignment = {}
    for variable in net.variables:
        assignment[variable.name] = variable.sample(assignment, rng)
    return assignment


def rejection_sampling(net: BayesNet, query: str, evidence: Assignment,
                       samples: int = 10_000,
                       rng: random.Random | None = None,
                       stats: dict | None = None) -> dict:
    """Sample from the prior; throw away anything inconsistent with evidence.

    Simple, unbiased, and wasteful: with unlikely evidence almost every
    sample is discarded.
    """
    rng = rng or random.Random(0)
    counts = {v: 0 for v in net.domain(query)}
    kept = 0
    for _ in range(samples):
        sample = prior_sample(net, rng)
        if all(sample[k] == v for k, v in evidence.items()):
            counts[sample[query]] += 1
            kept += 1
    if stats is not None:
        stats["kept"] = kept
        stats["rejected"] = samples - kept
    return normalize(counts)


def likelihood_weighting(net: BayesNet, query: str, evidence: Assignment,
                         samples: int = 10_000,
                         rng: random.Random | None = None,
                         stats: dict | None = None) -> dict:
    """Fix the evidence, sample the rest, and weight by the evidence's likelihood.

    Every sample counts, so no work is thrown away. The weights get very
    skewed when the evidence is unlikely, which is this method's own version
    of the same problem.
    """
    rng = rng or random.Random(0)
    weights = {v: 0.0 for v in net.domain(query)}
    for _ in range(samples):
        assignment = dict(evidence)
        weight = 1.0
        for variable in net.variables:
            if variable.name in evidence:
                weight *= variable.probability(evidence[variable.name], assignment)
            else:
                assignment[variable.name] = variable.sample(assignment, rng)
        weights[assignment[query]] += weight
    if stats is not None:
        stats["total_weight"] = sum(weights.values())
    return normalize(weights)


def max_error(a: dict, b: dict) -> float:
    """Largest absolute difference between two distributions."""
    return max(abs(a[k] - b.get(k, 0.0)) for k in a)


# --------------------------------------------------------------------------
# Example networks
# --------------------------------------------------------------------------

def burglary_net() -> BayesNet:
    """Pearl's alarm network — the standard worked example."""
    return BayesNet([
        Variable("burglary", (), {(): {True: 0.001, False: 0.999}}),
        Variable("earthquake", (), {(): {True: 0.002, False: 0.998}}),
        Variable("alarm", ("burglary", "earthquake"), {
            (True, True): {True: 0.95, False: 0.05},
            (True, False): {True: 0.94, False: 0.06},
            (False, True): {True: 0.29, False: 0.71},
            (False, False): {True: 0.001, False: 0.999},
        }),
        Variable("john_calls", ("alarm",), {
            (True,): {True: 0.90, False: 0.10},
            (False,): {True: 0.05, False: 0.95},
        }),
        Variable("mary_calls", ("alarm",), {
            (True,): {True: 0.70, False: 0.30},
            (False,): {True: 0.01, False: 0.99},
        }),
    ])
