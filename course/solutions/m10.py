"""Reference solutions — Module 10, probabilistic reasoning."""

import itertools
import random

from csai import probability as prob
from csai.probability import BayesNet, Factor, Variable


def normalise(distribution):
    total = sum(distribution.values())
    if total == 0:
        n = len(distribution)
        return {k: 1.0 / n for k in distribution}
    return {k: v / total for k, v in distribution.items()}


def joint_probability(net, assignment):
    p = 1.0
    for variable in net.variables:
        p *= variable.probability(assignment[variable.name], assignment)
    return p


def query(net, variable, evidence):
    hidden = [n for n in net.names if n != variable and n not in evidence]
    distribution = {}
    for value in net.domain(variable):
        total = 0.0
        for combo in itertools.product(*[net.domain(h) for h in hidden]):
            world = dict(evidence)
            world[variable] = value
            world.update(dict(zip(hidden, combo)))
            total += joint_probability(net, world)
        distribution[value] = total
    return normalise(distribution)


def marginalise(factor, name, net):
    if name not in factor.variables:
        return factor
    scope = tuple(v for v in factor.variables if v != name)
    table = {}
    for combo in itertools.product(*[net.domain(v) for v in scope]):
        assignment = dict(zip(scope, combo))
        total = 0.0
        for value in net.domain(name):
            assignment[name] = value
            total += factor.value(assignment)
        table[combo] = total
    return Factor(scope, table)


def sample_once(net, rng):
    assignment = {}
    for variable in net.variables:
        assignment[variable.name] = variable.sample(assignment, rng)
    return assignment


def weighted_sample(net, evidence, rng):
    assignment = dict(evidence)
    weight = 1.0
    for variable in net.variables:
        if variable.name in evidence:
            weight *= variable.probability(evidence[variable.name], assignment)
        else:
            assignment[variable.name] = variable.sample(assignment, rng)
    return assignment, weight


# --- project ---------------------------------------------------------------

def build_ci_net():
    return BayesNet([
        Variable("bug", (), {(): {True: 0.20, False: 0.80}}),
        Variable("flaky_infra", (), {(): {True: 0.10, False: 0.90}}),
        Variable("ci_red", ("bug", "flaky_infra"), {
            (True, True): {True: 0.99, False: 0.01},
            (True, False): {True: 0.95, False: 0.05},
            (False, True): {True: 0.80, False: 0.20},
            (False, False): {True: 0.02, False: 0.98},
        }),
        Variable("local_pass", ("bug",), {
            (True,): {True: 0.30, False: 0.70},
            (False,): {True: 0.97, False: 0.03},
        }),
        Variable("review_comment", ("bug",), {
            (True,): {True: 0.60, False: 0.40},
            (False,): {True: 0.15, False: 0.85},
        }),
    ])


def infer(net, variable, evidence, method="exact", samples=5000, rng=None):
    if method == "exact":
        return prob.enumeration_ask(net, variable, evidence)
    if method == "elimination":
        return prob.variable_elimination(net, variable, evidence)
    rng = rng or random.Random(0)
    if method == "rejection":
        return prob.rejection_sampling(net, variable, evidence, samples, rng)
    if method == "weighting":
        return prob.likelihood_weighting(net, variable, evidence, samples, rng)
    raise ValueError(f"unknown method {method!r}")


def convergence(net, variable, evidence, sizes, rng_seed=0):
    truth = prob.enumeration_ask(net, variable, evidence)
    out = {}
    for method in ("rejection", "weighting"):
        curve = {}
        for n in sizes:
            estimate = infer(net, variable, evidence, method, samples=n,
                             rng=random.Random(rng_seed))
            curve[n] = prob.max_error(truth, estimate)
        out[method] = curve
    return out
