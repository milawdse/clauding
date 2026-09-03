# %% [markdown]
# # Module 6 — Reasoning as Search: BFS, Uniform Cost, A*
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Formulate a problem as a state space — states, actions, transitions, goal
#    test, costs — and recognise that this formulation *is* the modelling work.
# 2. Implement breadth-first, depth-limited, iterative-deepening, uniform-cost
#    and A* search from one shared loop.
# 3. State what **admissible** and **consistent** mean, and why A* is optimal
#    when they hold.
# 4. Compare heuristics by nodes expanded and by effective branching factor,
#    rather than by intuition.
# 5. Map the cube task's entire state space, and use it to say something exact
#    about what `test_extrapolate` does and does not test.
#
# **Prerequisites:** Module 1 (the cube task). Modules 2–5 are not needed.
#
# **Time:** ~75 minutes plus exercises.

# %% [markdown]
# ## 1. Everything is a graph
#
# Newell and Simon's framing from *Human Problem Solving* (1972): deliberate
# problem solving is **search through a space of states**. Five components,
# and once you have them the algorithms come for free:
#
# | component | 8-puzzle | cube rotation |
# |---|---|---|
# | **state** | tile arrangement | which colour is on which face |
# | **initial state** | the scramble | the cube as described |
# | **actions** | slide a tile into the blank | bring a named face to the top |
# | **transition** | the resulting arrangement | the rotated cube |
# | **goal test** | tiles in order | a target configuration |
# | **step cost** | 1 per slide | 1 per rotation |
#
# The hard, creative part is choosing the state representation. Everything
# after it is mechanical. Choose badly — states that are not hashable, or that
# distinguish things that don't matter — and the search space explodes for no
# reason. Choose well and a hard problem becomes small, as §5 will show rather
# dramatically.
#
# One decision deserves naming: **graph search versus tree search.** Keep a set
# of states already reached and never revisit one, or don't. Not keeping it is
# simpler and re-explores the same state along every path that reaches it —
# usually catastrophic. Every algorithm in `csai.search` keeps the set.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import random
from collections import Counter, deque

from csai import data, search
from csai.check import checker
from csai.render import bar_chart, grid, table
from csai.search import Problem

print("ready")

# %% [markdown]
# ## 2. The classical benchmark: the 8-puzzle
#
# Eight numbered tiles and a blank in a 3×3 frame; slide tiles into the blank
# until they are in order. 181,440 reachable states (half of `9!` — the other
# half is unreachable, which is a nice piece of group theory in its own
# right). Small enough to solve exactly, big enough that the choice of
# algorithm shows up immediately.

# %%
GOAL_8 = (1, 2, 3, 4, 5, 6, 7, 8, 0)
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class EightPuzzle(Problem):
    """Slide tiles into the blank (0). Actions move *the blank*."""

    def __init__(self, start, heuristic="manhattan"):
        self.start = start
        self.h_name = heuristic

    def initial(self):
        return self.start

    def is_goal(self, state):
        return state == GOAL_8

    def actions(self, state):
        row, col = divmod(state.index(0), 3)
        for name, (dr, dc) in DELTAS.items():
            if 0 <= row + dr < 3 and 0 <= col + dc < 3:
                yield name

    def result(self, state, action):
        blank = state.index(0)
        row, col = divmod(blank, 3)
        dr, dc = DELTAS[action]
        target = (row + dr) * 3 + (col + dc)
        tiles = list(state)
        tiles[blank], tiles[target] = tiles[target], tiles[blank]
        return tuple(tiles)

    def heuristic(self, state):
        if self.h_name == "none":
            return 0
        if self.h_name == "misplaced":
            return sum(1 for i, v in enumerate(state) if v and v != GOAL_8[i])
        total = 0
        for i, v in enumerate(state):
            if v == 0:
                continue
            r1, c1 = divmod(i, 3)
            r2, c2 = divmod(v - 1, 3)
            total += abs(r1 - r2) + abs(c1 - c2)
        return total


def scramble(moves, seed=0):
    """A solvable puzzle: random-walk backwards from the goal."""
    rng = random.Random(seed)
    problem, state = EightPuzzle(GOAL_8), GOAL_8
    for _ in range(moves):
        state = problem.result(state, rng.choice(list(problem.actions(state))))
    return state


puzzle = scramble(30, seed=13)
print(grid([puzzle[0:3], puzzle[3:6], puzzle[6:9]], cell_width=1))
print("misplaced tiles:", EightPuzzle(puzzle, "misplaced").heuristic(puzzle))
print("manhattan      :", EightPuzzle(puzzle, "manhattan").heuristic(puzzle))

# %% [markdown]
# ## 3. Uninformed search
#
# All of these are the same loop over a **frontier** of nodes to expand. The
# only difference is what comes out of the frontier next.
#
# | strategy | frontier | complete? | optimal? | memory |
# |---|---|---|---|---|
# | breadth-first | FIFO | yes | yes, if all steps cost the same | O(b^d) — the killer |
# | depth-first | LIFO | no (can loop) | no | O(bd) |
# | depth-limited | LIFO to depth ℓ | only if ℓ ≥ d | no | O(bℓ) |
# | iterative deepening | ℓ = 0, 1, 2, … | yes | yes, unit costs | O(bd) |
# | uniform cost (Dijkstra) | lowest `g` | yes | yes, any costs ≥ 0 | O(b^d) |
#
# Iterative deepening looks wasteful — it re-expands the shallow levels on
# every pass — and isn't, because in a tree with branching factor `b` the
# bottom level contains most of the nodes. Redoing everything above it costs a
# constant factor of about `b/(b-1)`. Breadth-first's optimality with
# depth-first's memory, for a few per cent overhead.

# %%
easy = EightPuzzle(scramble(12, seed=1))
rows = []
for name in ("breadth-first", "uniform cost", "iterative deepening"):
    result = search.ALGORITHMS[name](easy)
    rows.append((name, result.depth, result.expanded, result.generated,
                 result.max_frontier))
print(table(rows, ["algorithm", "solution length", "expanded", "generated",
                   "peak frontier"], align="lrrrr"))

# %% [markdown]
# Same answer, wildly different memory. Note the peak frontier column: that,
# not time, is what stops breadth-first search on real problems.

# %% [markdown]
# ## 4. Heuristics and A*
#
# A **heuristic** `h(s)` estimates the remaining cost from `s` to a goal.
# Two properties matter:
#
# * **Admissible** — `h(s)` never *overestimates* the true remaining cost. An
#   optimistic heuristic can be corrected by search; a pessimistic one causes
#   A* to discard the optimal path before examining it.
# * **Consistent** (monotone) — `h(s) ≤ cost(s → s′) + h(s′)` for every
#   transition. A triangle inequality. Consistency implies admissibility, and
#   guarantees that the first time A* pops a state it already has the cheapest
#   route to it — so it never has to reopen a closed state.
#
# A* orders the frontier by `f(n) = g(n) + h(n)`: cost so far plus estimated
# cost remaining. With an admissible heuristic it returns an optimal solution,
# and with a consistent one it is *optimally efficient* — no algorithm using
# the same heuristic can expand fewer nodes and still guarantee optimality
# (Hart, Nilsson & Raphael, 1968; Dechter & Pearl, 1985).
#
# Set `h = 0` and A* degenerates to uniform cost. That is the right way to see
# it: **A* is Dijkstra plus a hint about where to look.**
#
# Two heuristics for the 8-puzzle:
#
# * **misplaced tiles** — count tiles not in place. Admissible: each needs at
#   least one move.
# * **Manhattan distance** — sum each tile's row plus column distance from
#   home. Also admissible, and it *dominates*: it is never smaller, because a
#   misplaced tile is at least one step from home. A dominating admissible
#   heuristic never expands more nodes.

# %%
rows = []
for h in ("none", "misplaced", "manhattan"):
    problem = EightPuzzle(puzzle, h)
    result = search.astar(problem)
    rows.append((f"A* with h = {h}", result.depth, result.expanded,
                 result.generated,
                 f"{search.effective_branching_factor(result.expanded, result.depth):.3f}"))
greedy = search.greedy_best_first(EightPuzzle(puzzle, "manhattan"))
rows.append(("greedy (manhattan, no g)", greedy.depth, greedy.expanded,
             greedy.generated, "—"))
print(table(rows, ["search", "solution length", "expanded", "generated",
                   "effective branching factor b*"], align="lrrrr"))

# %% [markdown]
# Three things to take from that table.
#
# 1. **`h = none` is uniform cost.** The heuristic is the entire difference.
# 2. **Manhattan dominates misplaced** and expands a fraction of the nodes,
#    for the same optimal answer. Better heuristics do not merely speed things
#    up — they change what is feasible at all.
# 3. **Greedy is fast and wrong.** It ignores `g`, so it dives at whatever
#    looks closest and returns a longer solution. Dropping the cost-so-far
#    term is what makes a search *greedy*, and it is precisely what a
#    left-to-right chain of thought does.
#
# The effective branching factor `b*` is the standard way to compare
# heuristics across problem sizes: the branching factor a uniform tree would
# need to contain as many nodes as the search expanded. Closer to 1 is better.

# %% [markdown]
# ## 5. The cube task as a search problem
#
# Now Module 1's cube, framed as search: given a starting cube and a target
# configuration, what is the **shortest** rotation sequence between them?
#
# States are cubes, so first they must be made hashable — a tuple of six
# colours in a fixed face order. That representation choice is the modelling
# work; everything else follows.

# %%
SIDES = data.SIDES               # ("top", "right", "front", "left", "back", "bottom")
ROTATIONS = ["front", "back", "left", "right", "bottom"]


def as_tuple(state_dict):
    return tuple(state_dict[s] for s in SIDES)


def as_dict(state_tuple):
    return dict(zip(SIDES, state_tuple))


def turn(state_tuple, side):
    """One rotation, via the repo's verified simulator."""
    return as_tuple(data.simulate(as_dict(state_tuple), [side])[0])


class CubeProblem(Problem):
    """Shortest rotation sequence from one cube configuration to another."""

    def __init__(self, start, goal):
        self.start, self.goal = start, goal

    def initial(self):
        return self.start

    def is_goal(self, state):
        return state == self.goal

    def actions(self, state):
        return ROTATIONS

    def result(self, state, action):
        return turn(state, action)

    def heuristic(self, state):
        # Every rotation moves exactly four faces, so with k faces out of
        # place at least ceil(k/4) rotations remain. Admissible, and just
        # about the weakest useful heuristic there is.
        wrong = sum(1 for a, b in zip(state, self.goal) if a != b)
        return -(-wrong // 4)


example = data.load_split("test_extrapolate", limit=1)[0]
start = as_tuple(data.initial_state(example))
finish = as_tuple(data.gold_states(example)[-1])

print("the dataset's own sequence:", example["metadata"]["rotations"])
result = search.astar(CubeProblem(start, finish))
print("A* shortest sequence      :", result.actions)
print(result)

# %% [markdown]
# ### How big is this space, actually?
#
# Before optimising a search, measure the thing you are searching. Breadth-first
# from any cube, with no goal, enumerates everything reachable.

# %%
def explore(start_state):
    """Every reachable configuration, with its distance from `start_state`."""
    distance = {start_state: 0}
    queue = deque([start_state])
    while queue:
        state = queue.popleft()
        for side in ROTATIONS:
            nxt = turn(state, side)
            if nxt not in distance:
                distance[nxt] = distance[state] + 1
                queue.append(nxt)
    return distance


distances = explore(start)
print(f"reachable configurations: {len(distances)}")
print(f"eccentricity (furthest any cube can get): {max(distances.values())}")
print(bar_chart(sorted(Counter(distances.values()).items()), width=30,
                title="configurations at each distance", value_fmt="{:.0f}"))

# %% [markdown]
# **Twenty-four states, and nothing is more than three rotations from anything
# else.** That is not a coincidence: the five rotations generate the rotation
# group of the cube, which has exactly 24 elements — one per way of orienting a
# cube in space. The colours never mix; the whole cube just turns.
#
# So this "search problem" has a state space you could draw on a napkin. Which
# raises an uncomfortable question about the dataset.

# %%
def optimal_distance(example):
    """Shortest rotation count between an example's endpoints."""
    a = as_tuple(data.initial_state(example))
    b = as_tuple(data.gold_states(example)[-1])
    return search.breadth_first(CubeProblem(a, b)).depth


rows = []
for split in ("test_seen", "test_extrapolate"):
    examples = data.load_split(split, limit=200)
    stated = [data.num_rotations(e) for e in examples]
    optimal = [optimal_distance(e) for e in examples]
    redundant = sum(o < s for o, s in zip(optimal, stated))
    rows.append((split,
                 f"{sum(stated) / len(stated):.2f}",
                 f"{sum(optimal) / len(optimal):.2f}",
                 max(optimal),
                 f"{redundant / len(examples):.0%}"))
print(table(rows, ["split", "mean rotations stated", "mean actually needed",
                   "max needed", "longer than necessary"], align="lrrrr"))

# %% [markdown]
# Every single `test_extrapolate` problem states a longer sequence than its
# endpoints require, and none of them is more than three rotations apart. The
# 4-to-6-rotation "extrapolation" split does not pose deeper *problems*; it
# poses longer *descriptions* of problems that were never more than three
# moves wide.
#
# Be careful about what this does and does not mean, because both halves
# matter:
#
# * It **does not** make the split useless. A solver reading the question has
#   to track the cube through each rotation as stated; it cannot know the
#   sequence was redundant without doing the work. Six steps of bookkeeping
#   is genuinely harder than two, and that is exactly the failure mode
#   Module 1 measured.
# * It **does** mean "chain length" and "problem depth" are different axes,
#   and this dataset varies only the first. A model could in principle learn
#   the shortcut — recognise that `bottom` twice is the identity, and collapse
#   the sequence — and would then look like it was extrapolating when it had
#   found a simplification. The step-level faithfulness metric from Module 1
#   is what would tell the two apart.
#
# The general lesson is worth more than the specific finding: **map your state
# space before you design experiments in it.** Twenty lines of breadth-first
# search told us something about this benchmark that no amount of accuracy
# measurement would have.

# %% [markdown]
# ## 6. Bridge to language models
#
# Line up the two pictures.
#
# > **Chain of thought is a single path through a search space, chosen
# > greedily, with no backtracking and no goal test.**
#
# Every clause of that sentence is a missing piece:
#
# | search has | a linear chain of thought has |
# |---|---|
# | a frontier of alternatives | one current state |
# | `g`, the cost so far | nothing — no notion of how far it has come |
# | `h`, an estimate of what remains | at best, an implicit sense of progress |
# | backtracking on failure | tokens already emitted, and they stay emitted |
# | a goal test | the model's own judgement that it is finished |
# | a proof of optimality under admissibility | no guarantee at all |
#
# The correspondences to current practice write themselves:
#
# * **Sampling `k` chains and voting** (self-consistency) is a beam of width
#   `k` with no shared structure between beams — the crudest possible frontier.
# * **Tree-of-thoughts** is literally best-first search where the successor
#   function is "ask the model for next steps" and the heuristic is "ask the
#   model how promising this is". Module 7 builds it.
# * **Process reward models** are learned heuristics `h`. Whether they are
#   admissible is not usually asked, which is exactly why verifier-guided
#   decoding sometimes prunes the right answer.
# * **Backtracking in reasoning models** — "wait, that's wrong, let me
#   reconsider" — is the frontier being reconstructed inside the context
#   window, expensively and unreliably, because the architecture has nowhere
#   else to put it.
#
# And the piece with no neural analogue at all is the **goal test**. A*
# terminates because it can *recognise* a solution. For puzzles, code and
# proofs a goal test exists and is cheap, which is precisely why search-based
# LLM systems work best there. For open-ended reasoning nobody has one, and
# that gap — not the search algorithm — is what Modules 9, 11 and 12 keep
# coming back to.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — reconstruct a path
#
# Search finds a goal *node*; you need the route. Write
# `reconstruct(parents, goal)` where `parents` maps each state to the state it
# was reached from (the start maps to `None`). Return the path from start to
# goal inclusive; return `[]` if `goal` is not in `parents`.

# %%
def reconstruct(parents, goal):
    """Path from the start state to `goal`, inclusive."""
    # TODO: walk parent pointers back to the state whose parent is None
    return None


# %%
@checker("Exercise 6.1 — reconstruct")
def check_ex1():
    parents = {"a": None, "b": "a", "c": "b", "d": "a"}
    yield "the start alone", reconstruct(parents, "a"), ["a"]
    yield "one step", reconstruct(parents, "d"), ["a", "d"]
    yield "a longer chain", reconstruct(parents, "c"), ["a", "b", "c"]
    yield "an unreached state", reconstruct(parents, "z"), []
    yield "empty map", reconstruct({}, "a"), []


check_ex1()

# %% [markdown]
# ### Exercise 2 — breadth-first search
#
# Write `bfs(problem)` returning `(path, expanded)`: the list of states from
# initial to goal, and the number of nodes you took off the frontier and
# expanded. Return `(None, expanded)` if there is no solution.
#
# Keep a `reached` set, and test for the goal **when you generate** a node
# rather than when you pop it — that saves a whole level of expansion.

# %%
def bfs(problem):
    """(path, nodes expanded) via breadth-first graph search."""
    # TODO: a deque frontier, a reached set, and a parents dict for the path
    return None


# %%
@checker("Exercise 6.2 — bfs")
def check_ex2():
    small = EightPuzzle(scramble(6, seed=1))
    path, expanded = bfs(small) or (None, None)
    yield "finds a path", path is not None, True
    yield "…starting at the initial state", (path or [None])[0], small.initial()
    yield "…ending at the goal", (path or [None])[-1], GOAL_8
    yield ("…of optimal length",
           len(path) - 1 if path else None, search.breadth_first(small).depth)
    yield ("…that is a legal sequence of moves",
           all(b in [small.result(a, act) for act in small.actions(a)]
               for a, b in zip(path, path[1:])) if path else None, True)
    yield "reports expansions", isinstance(expanded, int), True

    cube = CubeProblem(start, finish)
    path, _ = bfs(cube) or (None, None)
    yield ("works on the cube too",
           len(path) - 1 if path else None, search.breadth_first(cube).depth)

    unreachable = CubeProblem(start, tuple(["nosuchcolour"] * 6))
    yield "no solution -> None", (bfs(unreachable) or (0,))[0], None


check_ex2()

# %% [markdown]
# ### Exercise 3 — the Manhattan heuristic
#
# Write `manhattan(state)` for the 8-puzzle: for each tile (ignore the blank),
# the row distance plus the column distance from where it belongs in `GOAL_8`.

# %%
def manhattan(state):
    """Sum of each tile's row + column distance from its goal position."""
    # TODO: divmod(i, 3) gives the current cell; divmod(v - 1, 3) the target
    return None


# %%
@checker("Exercise 6.3 — manhattan")
def check_ex3():
    yield "the goal scores zero", manhattan(GOAL_8), 0
    yield ("one tile one step away",
           manhattan((1, 2, 3, 4, 5, 6, 7, 0, 8)), 1)
    yield ("the blank is not a tile",
           manhattan((0, 1, 2, 3, 4, 5, 6, 7, 8)), 12)
    for seed in (1, 5, 9):
        s = scramble(20, seed=seed)
        yield (f"agrees with the lecture version (seed {seed})",
               manhattan(s), EightPuzzle(s, "manhattan").heuristic(s))
    for seed in (2, 6):
        s = scramble(14, seed=seed)
        yield (f"never overestimates (seed {seed})",
               manhattan(s) <= search.breadth_first(EightPuzzle(s)).depth, True)


check_ex3()

# %% [markdown]
# ### Exercise 4 — A*
#
# Write `astar(problem)` returning `(path, cost, expanded)`, using
# `problem.heuristic`. A `heapq` of `(f, tiebreak, state)` and dicts for the
# best-known cost and the parents is enough.
#
# <details><summary>Hint</summary>
#
# Push `(h(start), 0, start)`. Pop the lowest `f`; if it is the goal, stop.
# Otherwise for each action compute `g_new = g[state] + step_cost`; if that
# beats the best `g` recorded for the successor, record it, set its parent,
# and push it with `f = g_new + h`. Use `itertools.count()` for the tiebreak
# so equal-`f` states never compare the states themselves.
# </details>

# %%
def astar(problem):
    """(path, cost, nodes expanded) via A* with problem.heuristic."""
    # TODO: heapq frontier ordered by g + h; keep best-known g per state
    return None


# %%
@checker("Exercise 6.4 — astar")
def check_ex4():
    problem = EightPuzzle(scramble(18, seed=3), "manhattan")
    path, cost, expanded = astar(problem) or (None, None, None)
    optimal = search.astar(problem)
    yield "finds a path", path is not None, True
    yield "…ending at the goal", (path or [None])[-1], GOAL_8
    yield "…of optimal cost", cost, optimal.cost
    yield "…and optimal length", len(path) - 1 if path else None, optimal.depth
    yield ("expands no more than uniform cost does",
           expanded <= search.uniform_cost(problem).expanded
           if expanded is not None else None, True)
    yield ("a better heuristic expands fewer nodes",
           astar(EightPuzzle(problem.initial(), "manhattan"))[2] <
           astar(EightPuzzle(problem.initial(), "misplaced"))[2], True)
    yield ("with h = 0 it is still optimal",
           astar(EightPuzzle(problem.initial(), "none"))[1], optimal.cost)
    cube = CubeProblem(start, finish)
    yield ("works on the cube", astar(cube)[1], search.astar(cube).cost)


check_ex4()

# %% [markdown]
# ### Exercise 5 — is the heuristic admissible?
#
# Admissibility is a *claim*, and claims can be tested. Write
# `overestimates(problem_for, states)`: for each state, compare the heuristic
# against the true optimal cost (found by uniform-cost search) and return the
# list of states where `h` was too large. An admissible heuristic returns `[]`.
#
# `problem_for(state)` builds a problem instance starting from `state`.

# %%
def overestimates(problem_for, states):
    """States where the heuristic exceeds the true optimal cost."""
    # TODO: for each state, compare problem.heuristic(state) with the cost
    # that uniform-cost search actually finds from it
    return None


# %%
@checker("Exercise 6.5 — overestimates")
def check_ex5():
    sample = [scramble(n, seed=n) for n in range(4, 14)]
    yield ("manhattan is admissible",
           overestimates(lambda s: EightPuzzle(s, "manhattan"), sample), [])
    yield ("so is misplaced-tiles",
           overestimates(lambda s: EightPuzzle(s, "misplaced"), sample), [])
    yield ("and so is h = 0",
           overestimates(lambda s: EightPuzzle(s, "none"), sample), [])

    class TooOptimistic(EightPuzzle):
        def heuristic(self, state):
            return 3 * manhattan(state)      # triple it: no longer admissible

    caught = overestimates(lambda s: TooOptimistic(s), sample)
    yield "an inflated heuristic is caught", len(caught or []) > 0, True
    yield "…and the caught states are from the sample", set(caught or []) <= set(
        sample), True
    yield "no states -> nothing to report", overestimates(
        lambda s: EightPuzzle(s), []), []


check_ex5()

# %% [markdown]
# ### Exercise 6 — map a state space
#
# Write `reachable(problem, start)` returning `{state: distance from start}`
# for everything reachable, ignoring the goal test. This is the tool that
# produced §5's finding, and it is worth having in your hands.

# %%
def reachable(problem, start):
    """{state: shortest number of actions from `start`}."""
    # TODO: breadth-first, recording depth, with no goal test
    return None


# %%
@checker("Exercise 6.6 — reachable")
def check_ex6():
    cube = CubeProblem(start, finish)
    got = reachable(cube, start)
    yield "the cube space has 24 configurations", len(got or {}), 24
    yield "the start is at distance 0", (got or {}).get(start), 0
    yield "nothing is further than 3 rotations", max((got or {1: 1}).values()), 3
    yield "agrees with the lecture's explore()", got, explore(start)
    yield ("every distance is realisable",
           sorted(set((got or {}).values())), [0, 1, 2, 3])

    puzzle_states = reachable(EightPuzzle(GOAL_8), GOAL_8)
    yield ("…and it works on other problems",
           isinstance(puzzle_states, dict) and len(puzzle_states) > 1000, True)


check_ex6()

# %% [markdown]
# ---
# ## Project — a rotation planner, and what it reveals
#
# Two deliverables: a planner, and an analysis that uses it.
#
# **Part 1 — the planner.**
#
# ```python
# plan(start_state, goal_state) -> list[str] | None
# ```
#
# taking and returning plain `{face: colour}` dicts (not tuples — the caller
# should not have to know your representation), and returning the **shortest**
# rotation sequence, `[]` if the cube is already there, or `None` if the goal
# is unreachable.
#
# **Part 2 — the analysis.**
#
# ```python
# redundancy_report(examples) -> dict
# ```
#
# with keys `"n"`, `"mean_stated"`, `"mean_optimal"`, `"max_optimal"`,
# `"redundant_fraction"` (stated longer than optimal), and `"by_length"`
# mapping each stated chain length to the mean optimal distance for problems
# of that length.
#
# **Write-up questions:**
#
# 1. Plot `by_length` for `test_extrapolate`. Does the optimal distance grow
#    with the stated chain length? What should it converge to, and why?
# 2. Module 1 showed accuracy falling with stated chain length. Given Part 2,
#    what exactly is getting harder? Name the resource that runs out.
# 3. Suppose you wanted a split where problem *depth* genuinely grows. What
#    would you have to change about the task — not the sampling? (There is
#    more than one answer; the cheapest one does not need a bigger cube.)

# %%
def plan(start_state, goal_state):
    """Shortest rotation sequence between two {face: colour} dicts."""
    # TODO: convert to hashable states, search, convert the actions back
    return None


def redundancy_report(examples):
    """Compare stated chain length against the distance actually required."""
    # TODO: for each example, the stated rotation count and the optimal
    # distance between its endpoints; aggregate as described
    return None


# %%
@checker("Project 6 — rotation planner")
def check_project():
    s0 = data.initial_state(example)
    s1 = data.gold_states(example)[-1]

    got = plan(s0, s1)
    yield "returns a list of rotation names", isinstance(got, list), True
    yield ("…all of them legal",
           set(got or []) <= set(ROTATIONS) if got is not None else None, True)
    yield ("…that actually reach the goal",
           data.simulate(s0, got)[-1] if got else None, s1)
    yield ("…and are as short as possible",
           len(got) if got is not None else None,
           search.breadth_first(CubeProblem(as_tuple(s0), as_tuple(s1))).depth)
    yield ("…never longer than the diameter of the space",
           len(got) <= 3 if got is not None else None, True)
    yield "an identical cube needs no rotations", plan(s0, s0), []
    yield ("an impossible goal returns None",
           plan(s0, {s: "nosuchcolour" for s in SIDES}), None)

    single = data.simulate(s0, ["front"])[-1]
    yield "a one-rotation goal", plan(s0, single), ["front"]

    sample = data.load_split("test_extrapolate", limit=60)
    report = redundancy_report(sample)
    yield "report has the required keys", sorted(report or {}), [
        "by_length", "max_optimal", "mean_optimal", "mean_stated", "n",
        "redundant_fraction"]
    yield "n counts the examples", (report or {}).get("n"), len(sample)
    yield ("nothing needs more than three rotations",
           (report or {}).get("max_optimal"), 3)
    yield ("every extrapolation problem is stated longer than needed",
           (report or {}).get("redundant_fraction"), 1.0)
    yield ("stated length exceeds optimal on average",
           (report or {}).get("mean_stated", 0) > (report or {}).get("mean_optimal", 9),
           True)
    yield ("by_length covers the split's chain lengths",
           sorted((report or {}).get("by_length") or {}), [4, 5, 6])

    seen_report = redundancy_report(data.load_split("test_seen", limit=60))
    yield ("shallow problems are much less redundant",
           (seen_report or {}).get("redundant_fraction", 1.0) < 0.6, True)


check_project()

# %%
# The report your write-up discusses.
if plan(data.initial_state(example), data.initial_state(example)) is not None:
    for split in ("test_seen", "test_extrapolate"):
        report = redundancy_report(data.load_split(split, limit=200))
        print(f"{split}: {report['n']} problems, "
              f"{report['redundant_fraction']:.0%} stated longer than needed")
        print(bar_chart(report["by_length"].items(), width=30, maximum=6.0,
                        title="  mean rotations actually needed, "
                              "by stated chain length",
                        value_fmt="{:.2f}"))
        print()

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * P. Hart, N. Nilsson & B. Raphael, "A Formal Basis for the Heuristic
#   Determination of Minimum Cost Paths" (1968) — A*.
# * R. Dechter & J. Pearl, "Generalized Best-First Search Strategies and the
#   Optimality of A*" (1985) — optimal efficiency.
# * R. Korf, "Depth-First Iterative-Deepening: An Optimal Admissible Tree
#   Search" (1985), and his 1997 solution of Rubik's cube with pattern
#   databases — where better heuristics come from.
# * S. Russell & P. Norvig, *AIMA* ch. 3–4.
# * S. Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large
#   Language Models" (2023) — best-first search with a model as the successor
#   function. Module 7 builds the classical core of it.
#
# **Next:** Module 7 adds an opponent and a clock — minimax, alpha-beta, and
# Monte-Carlo tree search, which is Tree-of-Thoughts with the model removed.
