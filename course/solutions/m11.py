"""Reference solutions — Module 11, decisions and metareasoning."""

import random
import statistics


def expected_value(outcomes, utility):
    return sum(p * utility(o) for o, p in outcomes.items())


def choose(actions, utility):
    scored = {a: expected_value(o, utility) for a, o in actions.items()}
    best = max(scored, key=scored.get)
    return best, scored[best]


def bellman(mdp, state, values):
    if mdp.is_terminal(state):
        return mdp.reward(state)
    return mdp.reward(state) + mdp.discount * max(
        sum(p * values[nxt] for nxt, p in mdp.transitions(state, a).items())
        for a in mdp.actions(state)
    )


def value_iterate(mdp, epsilon=1e-6, max_sweeps=1000):
    values = {s: 0.0 for s in mdp.states()}
    for _ in range(max_sweeps):
        updated = {s: bellman(mdp, s, values) for s in mdp.states()}
        delta = max(abs(updated[s] - values[s]) for s in values)
        values = updated
        if delta < epsilon:
            break
    return values


def greedy_policy(mdp, values):
    policy = {}
    for s in mdp.states():
        if mdp.is_terminal(s):
            policy[s] = None
            continue
        policy[s] = max(
            mdp.actions(s),
            key=lambda a: sum(p * values[nxt]
                              for nxt, p in mdp.transitions(s, a).items()),
        )
    return policy


def stop_at(quality_fn, difficulty, cost, max_effort):
    best_effort, best_value = 0, quality_fn(difficulty, 0)
    for effort in range(1, max_effort + 1):
        value = quality_fn(difficulty, effort) - cost * effort
        if value > best_value:
            best_effort, best_value = effort, value
    return best_effort


# --- project ---------------------------------------------------------------

def allocate(quality_fn, difficulties, budget, cap=25):
    allocation = [0] * len(difficulties)
    for _ in range(budget):
        gains = [
            (quality_fn(d, allocation[i] + 1) - quality_fn(d, allocation[i]))
            if allocation[i] < cap else float("-inf")
            for i, d in enumerate(difficulties)
        ]
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0:
            break
        allocation[best] += 1
    return allocation


def simulate(quality_fn, difficulties, allocation, rng):
    solved = sum(rng.random() < quality_fn(d, e)
                 for d, e in zip(difficulties, allocation))
    return solved / len(difficulties)


def simulate_verified(quality_fn, difficulties, budget, rng, cap=25):
    n = len(difficulties)
    solved = [False] * n
    effort = [0] * n
    spent = 0
    while spent < budget:
        live = [i for i in range(n) if not solved[i] and effort[i] < cap]
        if not live:
            break
        i = max(live, key=lambda j: quality_fn(difficulties[j], effort[j] + 1)
                - quality_fn(difficulties[j], effort[j]))
        effort[i] += 1
        spent += 1
        if rng.random() < quality_fn(difficulties[i], effort[i]):
            solved[i] = True
    return {"solved": sum(solved) / n, "spent": spent, "wasted": 0}


def compare(quality_fn, difficulties, budgets, trials=100):
    results = {}
    for budget in budgets:
        uniform_alloc = _uniform(difficulties, budget)
        adaptive_alloc = allocate(quality_fn, difficulties, budget)
        results[budget] = {
            "uniform": statistics.mean(
                simulate(quality_fn, difficulties, uniform_alloc,
                         random.Random(s)) for s in range(trials)),
            "adaptive": statistics.mean(
                simulate(quality_fn, difficulties, adaptive_alloc,
                         random.Random(s)) for s in range(trials)),
            "verified": statistics.mean(
                simulate_verified(quality_fn, difficulties, budget,
                                  random.Random(s))["solved"]
                for s in range(trials)),
        }
    return results


def _uniform(difficulties, budget):
    n = len(difficulties)
    base, extra = divmod(budget, n)
    return [base + (1 if i < extra else 0) for i in range(n)]
