"""Decisions under uncertainty: expected utility, MDPs, value of information.

The canonical version of what you build in Module 11.

Three layers, each answering a bigger question than the last:

* **Expected utility** — one decision, uncertain outcome. Which action?
* **Markov decision processes** — a sequence of decisions in a stochastic
  world. Which *policy*?
* **Value of information / metareasoning** — deciding what to find out, and
  how long to deliberate, before acting. Which *computation*?

The third layer is what makes this module about reasoning rather than about
control: thinking is an action, it costs something, and it should be chosen
by the same criterion as any other action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

State = Any
Action = Any


# --------------------------------------------------------------------------
# One-shot decisions
# --------------------------------------------------------------------------

def expected_utility(outcomes: dict, utility: Callable[[Any], float]) -> float:
    """Σ P(outcome) · U(outcome), for `outcomes` a {outcome: probability} dict."""
    return sum(p * utility(o) for o, p in outcomes.items())


def best_action(actions: dict, utility: Callable[[Any], float]
                ) -> tuple[Action, float]:
    """The maximum-expected-utility action, given {action: {outcome: prob}}."""
    scored = {a: expected_utility(o, utility) for a, o in actions.items()}
    best = max(scored, key=scored.get)
    return best, scored[best]


def value_of_perfect_information(actions: dict, utility: Callable[[Any], float],
                                 world: dict,
                                 outcome_given: Callable[[Action, Any], dict]
                                 ) -> float:
    """How much is it worth learning the true state *before* choosing?

    `world` is a distribution over hidden states; `outcome_given(action,
    state)` gives the outcome distribution if that state is the true one.

    The result is never negative — information cannot hurt a rational agent —
    and is **zero whenever the information cannot change your choice**, which
    is the most useful thing about it.
    """
    without = max(
        sum(w * expected_utility(outcome_given(a, s), utility)
            for s, w in world.items())
        for a in actions
    )
    with_info = sum(
        w * max(expected_utility(outcome_given(a, s), utility) for a in actions)
        for s, w in world.items()
    )
    return with_info - without


# --------------------------------------------------------------------------
# Markov decision processes
# --------------------------------------------------------------------------

class MDP:
    """States, actions, a stochastic transition model, and rewards.

    `transitions(state, action)` returns `{next_state: probability}`.
    """

    discount = 0.9

    def states(self) -> Iterable[State]:
        raise NotImplementedError

    def actions(self, state: State) -> Sequence[Action]:
        raise NotImplementedError

    def transitions(self, state: State, action: Action) -> dict:
        raise NotImplementedError

    def reward(self, state: State) -> float:
        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:
        return False


def q_value(mdp: MDP, state: State, action: Action, values: dict) -> float:
    """Expected value of taking `action` in `state`, then following `values`."""
    return sum(p * values[nxt]
               for nxt, p in mdp.transitions(state, action).items())


def bellman_backup(mdp: MDP, state: State, values: dict) -> float:
    """R(s) + γ · max over actions of the expected next value.

    The Bellman equation is the whole of dynamic programming in one line: the
    value of a state is its immediate reward plus the discounted value of
    acting optimally from wherever you land.
    """
    if mdp.is_terminal(state):
        return mdp.reward(state)
    return mdp.reward(state) + mdp.discount * max(
        q_value(mdp, state, a, values) for a in mdp.actions(state)
    )


def value_iteration(mdp: MDP, epsilon: float = 1e-6,
                    max_sweeps: int = 10_000,
                    stats: dict | None = None) -> dict:
    """Apply the Bellman backup everywhere until the values stop moving.

    Converges because the backup is a contraction with factor γ: each sweep
    shrinks the error by at least that much, so the fixpoint is unique and
    reachable regardless of where you start.
    """
    values = {s: 0.0 for s in mdp.states()}
    for sweep in range(max_sweeps):
        updated = {s: bellman_backup(mdp, s, values) for s in mdp.states()}
        delta = max(abs(updated[s] - values[s]) for s in values)
        values = updated
        if stats is not None:
            stats["sweeps"] = sweep + 1
            stats["delta"] = delta
        if delta < epsilon * (1 - mdp.discount) / mdp.discount:
            break
    return values


def extract_policy(mdp: MDP, values: dict) -> dict:
    """The greedy policy with respect to a value function."""
    policy = {}
    for s in mdp.states():
        if mdp.is_terminal(s):
            policy[s] = None
            continue
        policy[s] = max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, values))
    return policy


def policy_evaluation(mdp: MDP, policy: dict, sweeps: int = 100) -> dict:
    """Value of following a fixed policy — no max, so it is a linear system."""
    values = {s: 0.0 for s in mdp.states()}
    for _ in range(sweeps):
        values = {
            s: (mdp.reward(s) if mdp.is_terminal(s)
                else mdp.reward(s) + mdp.discount * q_value(mdp, s, policy[s], values))
            for s in mdp.states()
        }
    return values


def policy_iteration(mdp: MDP, max_rounds: int = 100,
                     stats: dict | None = None) -> tuple[dict, dict]:
    """Alternate evaluating a policy and improving it. Returns (policy, values).

    Usually converges in far fewer rounds than value iteration takes sweeps,
    because a policy stops changing long before the numbers stop moving.
    """
    policy = {s: (None if mdp.is_terminal(s) else mdp.actions(s)[0])
              for s in mdp.states()}
    values = {s: 0.0 for s in mdp.states()}
    for round_ in range(max_rounds):
        values = policy_evaluation(mdp, policy, sweeps=50)
        improved = extract_policy(mdp, values)
        if stats is not None:
            stats["rounds"] = round_ + 1
        if improved == policy:
            break
        policy = improved
    return policy, values


# --------------------------------------------------------------------------
# The 4x3 gridworld
# --------------------------------------------------------------------------

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
RIGHT_ANGLES = {"up": ("left", "right"), "down": ("right", "left"),
                "left": ("down", "up"), "right": ("up", "down")}


class GridWorld(MDP):
    """The standard 4x3 world: slippery movement, a goal, a pit, a wall.

    Intended movement succeeds 80% of the time; the rest of the mass goes to
    the two right angles. Bumping into a wall leaves you where you are.
    """

    def __init__(self, step_reward: float = -0.04, discount: float = 0.9,
                 rows: int = 3, cols: int = 4,
                 walls: Sequence[tuple] = ((1, 1),),
                 terminals: dict | None = None,
                 noise: float = 0.2):
        self.step_reward = step_reward
        self.discount = discount
        self.rows, self.cols = rows, cols
        self.walls = set(walls)
        self.terminals = dict(terminals or {(0, 3): +1.0, (1, 3): -1.0})
        self.noise = noise

    def states(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.walls]

    def actions(self, state):
        return list(MOVES)

    def is_terminal(self, state):
        return state in self.terminals

    def reward(self, state):
        return self.terminals.get(state, self.step_reward)

    def _move(self, state, action):
        dr, dc = MOVES[action]
        nxt = (state[0] + dr, state[1] + dc)
        if (nxt in self.walls or not (0 <= nxt[0] < self.rows)
                or not (0 <= nxt[1] < self.cols)):
            return state
        return nxt

    def transitions(self, state, action):
        if self.is_terminal(state):
            return {state: 1.0}
        side = self.noise / 2
        result: dict = {}
        for act, p in [(action, 1 - self.noise)] + [
                (a, side) for a in RIGHT_ANGLES[action]]:
            nxt = self._move(state, act)
            result[nxt] = result.get(nxt, 0.0) + p
        return result


ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">", None: " "}


def show_policy(mdp: GridWorld, policy: dict) -> list[list[str]]:
    """Grid of arrows, for `csai.render.grid`."""
    out = []
    for r in range(mdp.rows):
        row = []
        for c in range(mdp.cols):
            if (r, c) in mdp.walls:
                row.append("#")
            elif (r, c) in mdp.terminals:
                row.append("+" if mdp.terminals[(r, c)] > 0 else "-")
            else:
                row.append(ARROWS[policy[(r, c)]])
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Metareasoning: choosing how much to compute
# --------------------------------------------------------------------------

@dataclass
class PerformanceProfile:
    """How good an anytime algorithm's answer is after `effort` units of work.

    Russell and Wefald's framing (1991): computation is an action with a cost
    and an expected benefit, so the decision of *whether to keep thinking* is
    itself a decision-theoretic problem — one you can be rational about.
    """

    quality: Callable[[Any, int], float]     # (problem descriptor, effort) -> [0,1]
    cost_per_unit: float = 0.0

    def net_value(self, descriptor: Any, effort: int) -> float:
        return self.quality(descriptor, effort) - self.cost_per_unit * effort

    def stopping_point(self, descriptor: Any, max_effort: int) -> int:
        """The effort maximising quality minus cost — where to stop thinking."""
        return max(range(max_effort + 1),
                   key=lambda e: self.net_value(descriptor, e))


def allocate_greedy(profile: PerformanceProfile, descriptors: Sequence[Any],
                    budget: int, max_each: int | None = None) -> list[int]:
    """Spend a fixed total budget where each unit buys the most.

    Hand each unit of effort to whichever problem gains most from it. When the
    quality curves are concave this greedy allocation is optimal, and it is a
    good heuristic when they are not.
    """
    max_each = budget if max_each is None else max_each
    allocation = [0] * len(descriptors)
    for _ in range(budget):
        gains = [
            (profile.quality(d, allocation[i] + 1)
             - profile.quality(d, allocation[i]))
            if allocation[i] < max_each else float("-inf")
            for i, d in enumerate(descriptors)
        ]
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0:
            break
        allocation[best] += 1
    return allocation


def allocate_uniform(descriptors: Sequence[Any], budget: int) -> list[int]:
    """Spread the budget evenly — the baseline any allocator must beat."""
    n = len(descriptors)
    base, extra = divmod(budget, n)
    return [base + (1 if i < extra else 0) for i in range(n)]
