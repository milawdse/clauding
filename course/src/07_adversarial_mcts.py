# %% [markdown]
# # Module 7 — Adversarial and Anytime Search: Minimax, Alpha-Beta, MCTS
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Compute the exact value of a game position with minimax, and explain what
#    that value assumes.
# 2. Implement alpha-beta pruning and measure the ~30× it buys on a real tree.
# 3. Explain the depth limit, static evaluation, the horizon effect, and what
#    makes an algorithm **anytime**.
# 4. Implement Monte-Carlo tree search — selection, expansion, simulation,
#    backpropagation — and the UCT rule that balances them.
# 5. Demonstrate, by measurement, that **on a reasoning tree the search
#    algorithm is not the bottleneck — the value function is.** This is the
#    single most useful thing in the module.
#
# **Prerequisites:** Module 6 (state spaces, frontiers, heuristics).
#
# **Time:** ~80 minutes plus exercises.

# %% [markdown]
# ## 1. An opponent changes the question
#
# Module 6's A* assumed the world sits still while you plan. Add an agent that
# wants you to fail and the value of a state stops being "distance to the
# goal" and becomes "what happens if we both play well".
#
# The classical formulation (von Neumann's minimax theorem, 1928; Shannon's
# chess paper, 1950) is a **game tree** with two players who alternate. One
# maximises the utility, the other minimises it. The value of a position is
# defined by mutual recursion:
#
# > `value(s)` = `utility(s)` if `s` is terminal
# > = `max` over moves of `value(child)` if it is MAX's turn
# > = `min` over moves of `value(child)` if it is MIN's turn
#
# Read the assumption out loud, because it is strong: **the opponent plays
# optimally.** Minimax computes the value against a perfect adversary. Against
# a weak one it is needlessly cautious — it will not set a trap that a perfect
# opponent would see through, even when the actual opponent would fall for it.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import functools
import itertools
import math
import random
import time
from dataclasses import dataclass, field
from fractions import Fraction

from csai import adversarial as adv
from csai.check import checker
from csai.render import bar_chart, grid, table

print("ready")

# %% [markdown]
# ## 2. Minimax on tic-tac-toe

# %%
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
         (0, 3, 6), (1, 4, 7), (2, 5, 8),      # columns
         (0, 4, 8), (2, 4, 6)]                 # diagonals


class TicTacToe(adv.Game):
    """X is MAX, O is MIN. A state is a 9-tuple of "X", "O" or ""."""

    def initial(self):
        return ("",) * 9

    def player(self, state):
        return "MAX" if sum(1 for c in state if c) % 2 == 0 else "MIN"

    def actions(self, state):
        return [i for i, c in enumerate(state) if not c]

    def result(self, state, action):
        cells = list(state)
        cells[action] = "X" if self.player(state) == "MAX" else "O"
        return tuple(cells)

    def winner(self, state):
        for a, b, c in LINES:
            if state[a] and state[a] == state[b] == state[c]:
                return state[a]
        return None

    def is_terminal(self, state):
        return self.winner(state) is not None or all(state)

    def utility(self, state):
        w = self.winner(state)
        return 1.0 if w == "X" else -1.0 if w == "O" else 0.0

    def show(self, state):
        cells = [c or "." for c in state]
        return grid([cells[0:3], cells[3:6], cells[6:9]], cell_width=1)


game = TicTacToe()
stats = {}
t0 = time.perf_counter()
value, move = adv.minimax(game, game.initial(), stats)
print(f"value of the empty board: {value}   best opening: cell {move}")
print(f"nodes visited: {stats['nodes']:,}   time: {time.perf_counter() - t0:.2f}s")
print("\nValue 0 means: with perfect play by both sides, tic-tac-toe is a draw.")
print("That is a *proof*, obtained by looking at every position that matters.")

# %% [markdown]
# ## 3. Alpha-beta pruning
#
# Half a million nodes for a game that fits on a napkin. Most of them never
# mattered, and alpha-beta is the observation that says which.
#
# Carry two bounds down the tree:
#
# * **α** — the best value MAX can already guarantee somewhere above;
# * **β** — the best value MIN can already guarantee somewhere above.
#
# If at any node `α ≥ β`, the player to move above here already has a better
# option elsewhere, so *nothing in this subtree can change the answer*. Stop
# and return. Not an approximation — the same value, with fewer nodes.
#
# With perfect move ordering, alpha-beta examines `O(b^(d/2))` nodes instead of
# `O(b^d)`: **the same time buys twice the depth**. That is why every serious
# game program in the twentieth century was built on it.

# %%
stats_ab = {}
t0 = time.perf_counter()
value_ab, move_ab = adv.alphabeta(game, game.initial(), stats=stats_ab)
print(f"same value: {value_ab}   same move: {move_ab}")
print(table([("minimax", f"{stats['nodes']:,}", 0),
             ("alpha-beta", f"{stats_ab['nodes']:,}", stats_ab["cutoffs"])],
            ["algorithm", "nodes", "cutoffs"], align="lrr"))
print(f"\n{stats['nodes'] / stats_ab['nodes']:.0f}x fewer nodes, "
      f"identical answer, in {time.perf_counter() - t0:.2f}s.")

# %% [markdown]
# ### When the tree does not fit
#
# Chess has around 10⁴⁰ legal positions; Go has vastly more. The classical
# response has three parts:
#
# 1. **Cut off at a depth limit** rather than at terminal states.
# 2. Replace the missing subtree with a **static evaluation function** — a
#    hand-written estimate of who is winning (material, mobility, king safety).
# 3. **Iterate the depth**: search to depth 1, then 2, then 3, keeping the
#    best move found so far. This makes the algorithm **anytime** — interrupt
#    it whenever and it has an answer, better the longer you waited — and the
#    shallow passes also order the moves for the deeper ones, which makes
#    alpha-beta prune harder.
#
# The cost is the **horizon effect**: a disaster one move past the cutoff is
# invisible, and a program will happily make pointless checks to push it over
# the horizon. Every depth-limited search has some version of this pathology,
# including, as we will see, a language model that stops reasoning when the
# text feels finished.

# %% [markdown]
# ## 4. Monte-Carlo tree search
#
# Alpha-beta needs a good evaluation function, and for some domains nobody
# could write one — Go being the famous case, where a thirty-year effort
# produced programs at the level of a weak amateur.
#
# MCTS (Coulom, 2006; Kocsis & Szepesvári, 2006) sidesteps the problem:
# **evaluate a position by playing it out at random and averaging the
# results.** No domain knowledge required. Four phases per iteration:
#
# 1. **Select** — from the root, repeatedly descend to the most promising
#    child until reaching a node with unexplored actions.
# 2. **Expand** — add one child for an untried action.
# 3. **Simulate** — play out to the end (randomly, or guided) and score it.
# 4. **Backpropagate** — add that score to every node on the path.
#
# The whole algorithm turns on step 1, and the rule is **UCT**:
#
# > score(child) = mean reward + c · √( ln(parent visits) / child visits )
#
# The first term exploits — go where results have been good. The second
# explores — go where you have looked least, with the bonus shrinking as a node
# is visited. This is the UCB1 bandit algorithm applied to each node, and it
# comes with a regret bound. `c` sets the balance; 0 is pure greed.
#
# MCTS is **anytime** and **asymmetric**: it spends its budget on the branches
# that look worth it, growing a lopsided tree, which is exactly what you want
# when the branching factor is large.

# %%
print("UCT scores for three sibling nodes after 100 visits to their parent:\n")
rows = []
for label, wins, visits in [("looks great, well tested", 45, 50),
                            ("looks poor, well tested", 5, 40),
                            ("barely tried", 3, 4),
                            ("never tried", 0, 0)]:
    for c in (0.0, 1.4):
        score = (float("inf") if visits == 0
                 else wins / visits + c * math.sqrt(math.log(100) / visits))
        rows.append((label, wins, visits, c, f"{score:.3f}"))
print(table(rows, ["node", "total reward", "visits", "c", "UCT score"],
            align="lrrrr"))
print("\nAt c = 0 the well-tested winner dominates. At c = 1.4 the barely-tried")
print("node is competitive, and an untried one is always taken first.")

# %% [markdown]
# ## 5. A reasoning tree: the Game of 24
#
# Now the version that matters for this course. Take four or five numbers and
# combine them two at a time with `+ − × ÷` until one is left; hit 24 exactly.
# It is a puzzle, but the shape is a **reasoning tree**: each action is a step
# of work, states are partial progress, and only complete solutions can be
# scored.
#
# This is not an analogy. The Tree-of-Thoughts paper (Yao et al., 2023) used
# the Game of 24 as its headline benchmark, with a language model proposing
# the arithmetic steps and rating the partial states.

# %%
class Game24(adv.RewardProblem):
    """Combine numbers two at a time to reach the target exactly."""

    def __init__(self, numbers, target=24):
        self.numbers = tuple(sorted(Fraction(n) for n in numbers))
        self.target = Fraction(target)

    def initial(self):
        return self.numbers

    def is_terminal(self, state):
        return len(state) == 1

    def reward(self, state):
        return 1.0 if len(state) == 1 and state[0] == self.target else 0.0

    def actions(self, state):
        return [(i, j, op)
                for i, j in itertools.combinations(range(len(state)), 2)
                for op in "+-*/"
                if not (op == "/" and state[j] == 0)]

    def result(self, state, action):
        i, j, op = action
        x, y = state[i], state[j]
        value = (x + y if op == "+" else x - y if op == "-"
                 else x * y if op == "*" else x / y)
        rest = [state[k] for k in range(len(state)) if k not in (i, j)]
        return tuple(sorted(rest + [value]))


puzzle = Game24([4, 7, 8, 8])
result = adv.exhaustive(puzzle)
print(f"exhaustive search: solved={bool(result)} after {result.iterations} nodes")

# %% [markdown]
# Four numbers is small enough to enumerate. Five is where it gets interesting,
# and where we can compare strategies under an equal budget.
#
# Three contenders, all given the same number of playouts:
#
# * **random sampling** — independent random playouts, no tree, keep the best.
#   This is the self-consistency baseline: sample `k` chains, take what works.
# * **MCTS** — the same playouts, but organised into a tree with UCT.
# * **MCTS + value** — the same tree, but the playout is *guided* by a value
#   function instead of being random. Ours is an oracle: it can tell whether a
#   partial state can still reach 24. Think of it as a perfect process reward
#   model.

# %%
@functools.lru_cache(maxsize=None)
def can_reach(state, target):
    """Oracle: is `target` still reachable from this multiset of numbers?"""
    if len(state) == 1:
        return state[0] == target
    for i, j in itertools.combinations(range(len(state)), 2):
        for op in "+-*/":
            if op == "/" and state[j] == 0:
                continue
            x, y = state[i], state[j]
            value = (x + y if op == "+" else x - y if op == "-"
                     else x * y if op == "*" else x / y)
            rest = tuple(sorted([state[k] for k in range(len(state))
                                 if k not in (i, j)] + [value]))
            if can_reach(rest, target):
                return True
    return False


rng = random.Random(7)
INSTANCES = []
while len(INSTANCES) < 30:
    candidate = [rng.randint(1, 13) for _ in range(5)]
    if can_reach(tuple(sorted(Fraction(n) for n in candidate)), Fraction(24)):
        INSTANCES.append(candidate)

print(f"{len(INSTANCES)} five-number instances, every one of them solvable.")
print("first few:", INSTANCES[:4])

# %%
def oracle_value(problem):
    return lambda state: 1.0 if can_reach(state, problem.target) else 0.0


rows = []
for budget in (5, 20, 50, 100, 300):
    counts = {}
    for name in ("random sampling", "MCTS", "MCTS + value"):
        solved = 0
        for numbers in INSTANCES:
            problem = Game24(numbers)
            if name == "random sampling":
                r = adv.random_sampling(problem, samples=budget,
                                        rng=random.Random(3))
            elif name == "MCTS":
                r = adv.mcts(problem, iterations=budget, rng=random.Random(3))
            else:
                r = adv.mcts(problem, iterations=budget, rng=random.Random(3),
                             evaluate=oracle_value(problem))
            solved += r.best_reward >= 1.0
        counts[name] = solved
    rows.append((budget, counts["random sampling"], counts["MCTS"],
                 counts["MCTS + value"]))
print(table(rows, ["playout budget", "random sampling", "MCTS",
                   "MCTS + value"], align="rrrr"))
print(f"\n(out of {len(INSTANCES)} instances, all of which are solvable)")

# %% [markdown]
# **Read that table carefully, because it is the point of the module.**
#
# Plain MCTS is not meaningfully better than independent random sampling here,
# and at small budgets it is worse. That is not a bug in the implementation. It
# is what happens when the reward is **binary and sparse**: until a playout
# happens to land on 24, every node has mean reward 0, UCT has nothing to
# exploit, and the tree is a more expensive way of sampling at random.
#
# Add a value function and the same algorithm, the same tree, the same budget
# solves nearly everything at **five playouts**.
#
# > The search algorithm was never the bottleneck. The value function was.
#
# This is the single most transferable fact in the module, and it explains a
# great deal of what is happening in LLM reasoning research right now.

# %% [markdown]
# ## 6. Bridge to language models
#
# Line the methods up against their modern counterparts:
#
# | classical | LLM reasoning |
# |---|---|
# | a single greedy path | plain chain of thought |
# | independent random playouts, keep the best | self-consistency / best-of-n sampling |
# | beam search | beam search over reasoning steps |
# | MCTS with random rollouts | tree-of-thoughts with a weak or absent scorer |
# | MCTS with a value network | tree search guided by a process reward model |
# | alpha-beta's static evaluation | a learned verifier scoring partial reasoning |
# | anytime iterative deepening | test-time compute scaling: think longer, do better |
#
# Four consequences follow directly from the table you just produced:
#
# 1. **Sampling more chains has sharply diminishing returns.** It is the
#    weakest possible use of a budget — no structure is shared between
#    samples. It works at all only because a verifiable answer lets you keep
#    the good one.
# 2. **Tree search without a scorer buys little.** If a partial chain of
#    thought cannot be evaluated, the tree has nothing to be selective about.
#    Reports that tree-of-thoughts "didn't help on our task" are usually this.
# 3. **Therefore the value model is the research frontier**, not the search.
#    Process reward models, step-level verifiers, self-evaluation prompts —
#    all of them are attempts to supply the column that turned 4/30 into 30/30
#    above.
# 4. **AlphaZero is this table, closed into a loop.** Search improves on the
#    current value estimate; the improved results retrain the value model;
#    better values make the search sharper. Whether that loop closes for
#    open-ended reasoning — where, unlike Go, there is no cheap ground truth —
#    is one of the genuinely open questions in the field.
#
# And keep the horizon effect from §3 in mind. A model that stops reasoning
# when the text *feels* complete has a horizon exactly like a depth-limited
# search, and the same pathology: the failure just past the cutoff is
# invisible, and the confident summary hides it.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — minimax
#
# Write `minimax_value(game, state)` returning the exact value of `state` under
# optimal play by both sides. Recursion; do not prune.

# %%
def minimax_value(game, state):
    """Exact game-theoretic value of `state`."""
    # TODO: terminal -> utility; else max or min over children by whose turn
    return None


# %%
@checker("Exercise 7.1 — minimax_value")
def check_ex1():
    g = TicTacToe()
    x_wins = ("X", "X", "X", "O", "O", "", "", "", "")
    yield "a won terminal position", minimax_value(g, x_wins), 1.0
    o_wins = ("O", "O", "O", "X", "X", "", "", "X", "")
    yield "a lost terminal position", minimax_value(g, o_wins), -1.0
    full_draw = ("X", "O", "X", "X", "O", "O", "O", "X", "X")
    yield "a drawn terminal position", minimax_value(g, full_draw), 0.0
    forced_win = ("X", "X", "", "O", "O", "", "", "", "")
    yield "X to move with two in a row wins", minimax_value(g, forced_win), 1.0
    forced_block = ("X", "X", "", "O", "", "", "", "", "")
    yield ("O to move: blocking is forced, and still not enough",
           minimax_value(g, forced_block), 1.0)
    midgame = ("X", "", "", "", "O", "", "", "", "")
    yield ("a sensible opening is still drawn",
           minimax_value(g, midgame), adv.minimax(g, midgame)[0])


check_ex1()

# %% [markdown]
# ### Exercise 2 — pick the move
#
# Write `best_move(game, state)` returning the action with the best value for
# whoever is to move. Break ties by choosing the **smallest** action, so the
# result is deterministic.

# %%
def best_move(game, state):
    """The optimal action for the player to move, ties broken by lowest index."""
    # TODO: score every action with minimax_value and take the best
    return None


# %%
@checker("Exercise 7.2 — best_move")
def check_ex2():
    g = TicTacToe()
    win_now = ("X", "X", "", "O", "O", "", "", "", "")
    yield "take the win", best_move(g, win_now), 2
    must_block = ("X", "X", "", "O", "", "", "", "", "")
    yield "block the threat", best_move(g, must_block), 2
    fork = ("X", "", "", "", "O", "", "", "", "X")
    yield ("play a non-losing move",
           minimax_value(g, g.result(fork, best_move(g, fork))),
           minimax_value(g, fork))
    yield "no legal moves -> None", best_move(
        g, ("X", "O", "X", "X", "O", "O", "O", "X", "X")), None
    yield "deterministic", best_move(g, fork), best_move(g, fork)


check_ex2()

# %% [markdown]
# ### Exercise 3 — alpha-beta
#
# Write `alphabeta_value(game, state, alpha=-inf, beta=inf, counter=None)`
# returning the same value as `minimax_value` while visiting far fewer nodes.
# When `counter` is a dict, increment `counter["nodes"]` once per call.
#
# <details><summary>Hint</summary>
#
# At a MAX node: start `value = -inf`; for each child, `value = max(value,
# recurse(child, alpha, beta))`, then `alpha = max(alpha, value)`, and
# `break` as soon as `alpha >= beta`. MIN is the mirror image with `beta`.
# The bounds must be passed *down* to children and the updates kept local.
# </details>

# %%
def alphabeta_value(game, state, alpha=-math.inf, beta=math.inf, counter=None):
    """Minimax value with alpha-beta pruning."""
    # TODO: as minimax, but carry alpha/beta and cut off when they cross
    return None


# %%
@checker("Exercise 7.3 — alphabeta_value")
def check_ex3():
    g = TicTacToe()
    positions = [
        ("", "", "", "", "", "", "", "", ""),
        ("X", "", "", "", "O", "", "", "", ""),
        ("X", "X", "", "O", "O", "", "", "", ""),
        ("X", "O", "X", "", "O", "", "", "", ""),
    ]
    for i, s in enumerate(positions):
        yield (f"same value as minimax (position {i})",
               alphabeta_value(g, s), adv.minimax(g, s)[0])

    counter = {"nodes": 0}
    alphabeta_value(g, positions[1], counter=counter)
    plain = {}
    adv.minimax(g, positions[1], plain)
    yield "counts the nodes it visits", counter["nodes"] > 0, True
    yield ("…and visits far fewer than minimax",
           counter["nodes"] < plain["nodes"] / 3, True)


check_ex3()

# %% [markdown]
# ### Exercise 4 — the UCT rule
#
# Write `uct_score(total_reward, visits, parent_visits, c=1.4)`: the mean
# reward plus `c · sqrt(ln(parent_visits) / visits)`. An unvisited node scores
# `math.inf`, so it is always tried before any visited one.

# %%
def uct_score(total_reward, visits, parent_visits, c=1.4):
    """Upper confidence bound for a child node."""
    # TODO: infinity when unvisited; otherwise exploit + explore
    return None


# %%
@checker("Exercise 7.4 — uct_score")
def check_ex4():
    yield "unvisited nodes come first", uct_score(0, 0, 10), math.inf
    yield "c = 0 is pure exploitation", uct_score(5, 10, 100, c=0), 0.5
    yield ("the exploration term is added",
           round(uct_score(5, 10, 100, c=1.4), 6),
           round(0.5 + 1.4 * math.sqrt(math.log(100) / 10), 6))
    yield ("a less-visited node scores higher, all else equal",
           uct_score(2, 4, 100) > uct_score(20, 40, 100), True)
    yield ("…and a better node scores higher at equal visits",
           uct_score(9, 10, 100) > uct_score(1, 10, 100), True)
    yield ("the bonus shrinks as visits grow",
           uct_score(5, 10, 100) - 0.5 > uct_score(50, 100, 100) - 0.5, True)


check_ex4()

# %% [markdown]
# ### Exercise 5 — a random playout
#
# Write `rollout(problem, state, rng, max_depth=50)`: take random actions until
# the state is terminal or the depth runs out; return `problem.reward(state)`
# for a terminal state and `0.0` otherwise.

# %%
def rollout(problem, state, rng, max_depth=50):
    """Play randomly to the end; return the reward reached."""
    # TODO: loop taking rng.choice of the available actions
    return None


# %%
@checker("Exercise 7.5 — rollout")
def check_ex5():
    problem = Game24([4, 7, 8, 8])
    terminal = (Fraction(24),)
    yield "a solved terminal state scores 1", rollout(
        problem, terminal, random.Random(0)), 1.0
    yield "a wrong terminal state scores 0", rollout(
        problem, (Fraction(23),), random.Random(0)), 0.0
    rewards = [rollout(problem, problem.initial(), random.Random(s))
               for s in range(200)]
    yield "every playout returns 0 or 1", set(rewards) <= {0.0, 1.0}, True
    yield ("random play essentially never solves this instance",
           sum(rewards) / len(rewards) < 0.05, True)
    near = [rollout(problem, (Fraction(3), Fraction(8)), random.Random(s))
            for s in range(200)]
    yield ("…though one step from the answer it lands about a fifth of the time",
           0.05 < sum(near) / len(near) < 0.5, True)
    yield ("a depth of zero cannot finish",
           rollout(problem, problem.initial(), random.Random(0), max_depth=0), 0.0)


check_ex5()

# %% [markdown]
# ### Exercise 6 — backpropagation
#
# Write `backpropagate(node, reward)`: walk from `node` up through `.parent`
# adding 1 to `.visits` and `reward` to `.total` at every node, and return the
# number of nodes updated.

# %%
@dataclass
class Node:
    """A node in the MCTS tree."""
    state: object
    parent: object = None
    action: object = None
    children: dict = field(default_factory=dict)
    untried: list = field(default_factory=list)
    visits: int = 0
    total: float = 0.0

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0.0


def backpropagate(node, reward):
    """Add `reward` to every node from `node` up to the root. Returns the count."""
    # TODO: walk .parent to the top
    return None


# %%
@checker("Exercise 7.6 — backpropagate")
def check_ex6():
    root = Node("root")
    child = Node("child", parent=root)
    grandchild = Node("grandchild", parent=child)

    yield "updates the whole path", backpropagate(grandchild, 1.0), 3
    yield "the leaf is visited", grandchild.visits, 1
    yield "…and so is the root", root.visits, 1
    yield "rewards accumulate", root.total, 1.0

    backpropagate(grandchild, 0.0)
    yield "a second visit is counted", root.visits, 2
    yield "…but adds no reward", root.total, 1.0
    yield "means are computed from the path", root.mean, 0.5
    yield "a root on its own", backpropagate(Node("solo"), 1.0), 1


check_ex6()

# %% [markdown]
# ---
# ## Project — MCTS from scratch, and what a value function is worth
#
# Write the whole algorithm, then use it to reproduce §5's finding yourself.
#
# ```python
# tree_search(problem, iterations, c=1.4, rng=None, evaluate=None) -> dict
# ```
#
# returning a dict with keys `"best_reward"`, `"best_state"`, `"iterations"`
# (how many you actually ran), `"root"` (the root `Node`), and `"solved"`.
#
# Each iteration:
#
# 1. **select** — from the root, while the current node has no untried actions
#    and does have children, descend to the child with the highest
#    `uct_score`;
# 2. **expand** — if the node has untried actions and is not terminal, pop one
#    (use `rng` so runs are reproducible), create the child, and move to it;
# 3. **simulate** — `rollout` from there. When `evaluate` is given, do not
#    choose randomly: at each step take the action whose result scores highest
#    under `evaluate`, breaking ties with `rng`;
# 4. **backpropagate** the reward up the path.
#
# Record the best terminal state seen, and stop early once a reward of 1.0 is
# found.
#
# **Write-up questions:**
#
# 1. Reproduce the budget sweep. At what budget does unguided MCTS overtake
#    random sampling, if it does? Explain the shape of both curves in terms of
#    what UCT has to work with.
# 2. The oracle evaluator is perfect. Degrade it — make it wrong 20% of the
#    time — and re-run. How gracefully does the guided search fail? What does
#    that predict about an imperfect process reward model?
# 3. Set `c = 0`. What happens, and why is the answer different with and
#    without the evaluator?

# %%
def tree_search(problem, iterations, c=1.4, rng=None, evaluate=None):
    """Monte-Carlo tree search. Returns the report described above."""
    # TODO: select / expand / simulate / backpropagate, `iterations` times
    return None


# %%
@checker("Project 7 — MCTS")
def check_project():
    easy = Game24([4, 7, 8, 8])
    report = tree_search(easy, 400, rng=random.Random(0))
    yield "returns the required keys", sorted(report or {}), [
        "best_reward", "best_state", "iterations", "root", "solved"]
    yield "solves a four-number instance", (report or {}).get("solved"), True
    yield "…recording the winning state", (report or {}).get("best_state"), (
        Fraction(24),)
    yield ("…and stopping early once solved",
           (report or {}).get("iterations", 999) <= 400, True)

    root = (report or {}).get("root")
    yield "the root is a Node", isinstance(root, Node), True
    yield ("…visited once per iteration",
           root.visits if root else None, (report or {}).get("iterations"))
    yield "…with children", len(root.children) > 0 if root else None, True
    yield ("…whose visits do not exceed the root's",
           all(ch.visits <= root.visits for ch in root.children.values())
           if root else None, True)

    impossible = Game24([1, 1, 1, 1])
    dead = tree_search(impossible, 60, rng=random.Random(0))
    yield "an unsolvable instance is not solved", (dead or {}).get("solved"), False
    yield "…and runs the full budget", (dead or {}).get("iterations"), 60
    yield "…with no best state", (dead or {}).get("best_state"), None

    # The headline comparison, on a subset so the checker stays quick.
    subset = INSTANCES[:12]
    unguided = sum(tree_search(Game24(n), 30, rng=random.Random(3))["solved"]
                   for n in subset)
    guided = sum(tree_search(Game24(n), 30, rng=random.Random(3),
                             evaluate=oracle_value(Game24(n)))["solved"]
                 for n in subset)
    yield ("a value function solves all of them at a budget of 30",
           guided, len(subset))
    yield ("…which unguided search does not come close to",
           unguided < guided, True)

    yield ("c = 0 still runs (pure exploitation)",
           isinstance(tree_search(easy, 20, c=0.0, rng=random.Random(1)), dict),
           True)


check_project()

# %%
# Your own version of the module's headline table.
if tree_search(Game24([4, 7, 8, 8]), 5, rng=random.Random(0)) is not None:
    rows = []
    for budget in (5, 20, 50, 100):
        plain = sum(tree_search(Game24(n), budget, rng=random.Random(3))["solved"]
                    for n in INSTANCES)
        guided = sum(tree_search(Game24(n), budget, rng=random.Random(3),
                                 evaluate=oracle_value(Game24(n)))["solved"]
                     for n in INSTANCES)
        rows.append((budget, plain, guided))
    print(table(rows, ["budget", "MCTS", "MCTS + value"], align="rrr"))
    print(f"(out of {len(INSTANCES)} solvable instances)")

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * C. Shannon, "Programming a Computer for Playing Chess" (1950) — depth
#   limits and evaluation functions, before there were computers to run them.
# * D. Knuth & R. Moore, "An Analysis of Alpha-Beta Pruning" (1975).
# * R. Coulom, "Efficient Selectivity and Backup Operators in Monte-Carlo Tree
#   Search" (2006); L. Kocsis & C. Szepesvári, "Bandit Based Monte-Carlo
#   Planning" (2006) — MCTS and UCT.
# * C. Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012).
# * D. Silver et al., "Mastering the Game of Go without Human Knowledge"
#   (2017) — search and value learning in a loop.
# * S. Yao et al., "Tree of Thoughts" (2023) — the Game of 24, with a language
#   model as successor function and scorer.
# * C. Snell et al., "Scaling LLM Test-Time Compute Optimally" (2024) — how to
#   spend a thinking budget, which is the anytime question of §3.
#
# **Next:** Module 8 goes back to exactness. Constraint satisfaction —
# backtracking with propagation, where inference happens *inside* the search
# rather than around it.
