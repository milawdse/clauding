"""Game-tree search and Monte-Carlo tree search.

The canonical version of what you build in Module 7. Two families:

* **Minimax and alpha-beta** — exact, for two-player zero-sum games where you
  can afford to look at the whole tree (or a depth-limited slice of it with a
  static evaluation function).
* **Monte-Carlo tree search** — anytime and statistical, for trees too large
  or too poorly understood to evaluate. It needs no evaluation function: it
  plays the position out at random and averages.

MCTS here is written for **single-agent maximisation** — find the highest
reward reachable — because that is the shape a reasoning tree has, and the
shape Tree-of-Thoughts uses.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

State = Any
Action = Any
INF = float("inf")


# --------------------------------------------------------------------------
# Two-player zero-sum games
# --------------------------------------------------------------------------

class Game:
    """A two-player zero-sum game. Utilities are from `player`'s point of view."""

    players = ("MAX", "MIN")

    def initial(self) -> State:
        raise NotImplementedError

    def player(self, state: State) -> str:
        """Whose turn it is."""
        raise NotImplementedError

    def actions(self, state: State) -> Iterable[Action]:
        raise NotImplementedError

    def result(self, state: State, action: Action) -> State:
        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:
        raise NotImplementedError

    def utility(self, state: State) -> float:
        """Terminal value, positive when MAX has won."""
        raise NotImplementedError

    def evaluate(self, state: State) -> float:
        """Static estimate for a non-terminal state. Used by depth limits."""
        return 0.0


def minimax(game: Game, state: State, stats: dict | None = None
            ) -> tuple[float, Action | None]:
    """Exact value of `state`, and the best action. Explores the whole tree."""
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1
    if game.is_terminal(state):
        return game.utility(state), None
    maximising = game.player(state) == game.players[0]
    best_value = -INF if maximising else INF
    best_action = None
    for action in game.actions(state):
        value, _ = minimax(game, game.result(state, action), stats)
        if (value > best_value) if maximising else (value < best_value):
            best_value, best_action = value, action
    return best_value, best_action


def alphabeta(game: Game, state: State, alpha: float = -INF, beta: float = INF,
              depth: float = INF, stats: dict | None = None
              ) -> tuple[float, Action | None]:
    """Minimax with alpha-beta pruning, and an optional depth limit.

    `alpha` is the best value MAX can already guarantee, `beta` the best MIN
    can. When they cross, the remaining branches cannot affect the result and
    are never looked at — the same answer for a fraction of the nodes.
    """
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1
    if game.is_terminal(state):
        return game.utility(state), None
    if depth <= 0:
        return game.evaluate(state), None

    maximising = game.player(state) == game.players[0]
    best_action = None
    if maximising:
        value = -INF
        for action in game.actions(state):
            child, _ = alphabeta(game, game.result(state, action),
                                 alpha, beta, depth - 1, stats)
            if child > value:
                value, best_action = child, action
            alpha = max(alpha, value)
            if alpha >= beta:
                if stats is not None:
                    stats["cutoffs"] = stats.get("cutoffs", 0) + 1
                break
        return value, best_action

    value = INF
    for action in game.actions(state):
        child, _ = alphabeta(game, game.result(state, action),
                             alpha, beta, depth - 1, stats)
        if child < value:
            value, best_action = child, action
        beta = min(beta, value)
        if alpha >= beta:
            if stats is not None:
                stats["cutoffs"] = stats.get("cutoffs", 0) + 1
            break
    return value, best_action


# --------------------------------------------------------------------------
# Monte-Carlo tree search (single-agent maximisation)
# --------------------------------------------------------------------------

class RewardProblem:
    """A search problem scored by a reward in [0, 1] at terminal states."""

    def initial(self) -> State:
        raise NotImplementedError

    def actions(self, state: State) -> Sequence[Action]:
        raise NotImplementedError

    def result(self, state: State, action: Action) -> State:
        raise NotImplementedError

    def is_terminal(self, state: State) -> bool:
        raise NotImplementedError

    def reward(self, state: State) -> float:
        """Value of a terminal state, in [0, 1]."""
        raise NotImplementedError


@dataclass
class MCTSNode:
    state: State
    parent: "MCTSNode | None" = None
    action: Action = None
    children: dict = field(default_factory=dict)
    untried: list = field(default_factory=list)
    visits: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return not self.untried


def uct(node: MCTSNode, parent_visits: int, c: float) -> float:
    """Upper confidence bound for trees: exploit + explore.

    `mean` rewards what has worked; the second term rewards what has been
    tried least, and shrinks as the node is visited. `c` sets the balance —
    0 is pure greed, large is nearly uniform.
    """
    if node.visits == 0:
        return INF
    return node.mean + c * math.sqrt(math.log(parent_visits) / node.visits)


@dataclass
class MCTSResult:
    best_state: State = None
    best_reward: float = 0.0
    root: MCTSNode | None = None
    iterations: int = 0
    rollouts: int = 0

    def __bool__(self) -> bool:
        return self.best_reward > 0


def mcts(problem: RewardProblem, *, iterations: int = 500, c: float = 1.4,
         rng: random.Random | None = None,
         max_rollout_depth: int = 50,
         evaluate: Callable[[State], float] | None = None) -> MCTSResult:
    """Monte-Carlo tree search, returning the best terminal state found.

    Four phases per iteration:
    1. **select** — walk down the tree by UCT until a node with untried
       actions, or a terminal state;
    2. **expand** — add one child for an untried action;
    3. **simulate** — play randomly to a terminal state and score it;
    4. **backpropagate** — add that score to every node on the path.

    Pass `evaluate` to score a node directly instead of rolling out — the
    AlphaZero move, and the one that turns a value model (a process reward
    model, in LLM terms) into search guidance. With no evaluator, the random
    rollout is the value estimate, and it is only as informative as the
    domain allows.

    Anytime by construction: stop whenever you like and the statistics are
    valid for however many samples you took.
    """
    rng = rng or random.Random(0)
    root = MCTSNode(problem.initial())
    root.untried = list(problem.actions(root.state))
    result = MCTSResult(root=root)

    for i in range(iterations):
        node = root
        # 1. select
        while node.expanded and node.children and not problem.is_terminal(node.state):
            node = max(node.children.values(),
                       key=lambda ch: uct(ch, node.visits, c))
        # 2. expand
        if node.untried and not problem.is_terminal(node.state):
            action = node.untried.pop(rng.randrange(len(node.untried)))
            child_state = problem.result(node.state, action)
            child = MCTSNode(child_state, node, action)
            child.untried = list(problem.actions(child_state))
            node.children[action] = child
            node = child
        # 3. simulate — randomly, or guided by the evaluator
        state = node.state
        for _ in range(max_rollout_depth):
            if problem.is_terminal(state):
                break
            options = list(problem.actions(state))
            if not options:
                break
            if evaluate is None:
                state = problem.result(state, rng.choice(options))
            else:
                # Follow the value estimate, breaking ties at random. This is
                # what verifier-guided decoding does with a reward model.
                scored = [(evaluate(problem.result(state, a)), rng.random(), a)
                          for a in options]
                state = problem.result(state, max(scored)[2])
        reward = problem.reward(state) if problem.is_terminal(state) else 0.0
        result.rollouts += 1
        if problem.is_terminal(state) and problem.reward(state) > result.best_reward:
            result.best_reward = problem.reward(state)
            result.best_state = state
        # 4. backpropagate
        while node is not None:
            node.visits += 1
            node.total += reward
            node = node.parent
        result.iterations = i + 1
        if result.best_reward >= 1.0:
            break
    return result


def exhaustive(problem: RewardProblem, *, max_nodes: int = 200_000
               ) -> MCTSResult:
    """Depth-first search of the whole tree — the exact baseline."""
    result = MCTSResult()
    stack = [problem.initial()]
    seen = set()
    while stack and result.iterations < max_nodes:
        state = stack.pop()
        result.iterations += 1
        if problem.is_terminal(state):
            reward = problem.reward(state)
            if reward > result.best_reward:
                result.best_reward, result.best_state = reward, state
            if reward >= 1.0:
                break
            continue
        for action in problem.actions(state):
            nxt = problem.result(state, action)
            key = repr(nxt)
            if key in seen:
                continue
            seen.add(key)
            stack.append(nxt)
    return result


def random_sampling(problem: RewardProblem, *, samples: int = 500,
                    rng: random.Random | None = None,
                    max_depth: int = 50) -> MCTSResult:
    """Independent random playouts, no tree. The self-consistency baseline."""
    rng = rng or random.Random(0)
    result = MCTSResult()
    for i in range(samples):
        state = problem.initial()
        for _ in range(max_depth):
            if problem.is_terminal(state):
                break
            options = list(problem.actions(state))
            if not options:
                break
            state = problem.result(state, rng.choice(options))
        reward = problem.reward(state) if problem.is_terminal(state) else 0.0
        result.rollouts += 1
        result.iterations = i + 1
        if reward > result.best_reward:
            result.best_reward, result.best_state = reward, state
        if reward >= 1.0:
            break
    return result
