# %% [markdown]
# # Module 9 — Planning: STRIPS, Relaxations, and the Validator
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Represent a domain in STRIPS — preconditions, add lists, delete lists —
#    and explain how that representation dodges the frame problem.
# 2. Ground action schemas, and see why grounding is where planning problems
#    get big.
# 3. Solve the Sussman anomaly and say why it embarrassed an entire generation
#    of planners.
# 4. Derive **domain-independent heuristics** from the delete relaxation
#    (`h_max`, `h_add`, `h_FF`) and measure what they are worth.
# 5. Write a **plan validator**, and use it to localise the first broken step
#    in a plausible-looking but wrong plan — which is the single most useful
#    thing to build around a language model that proposes plans.
#
# **Prerequisites:** Module 6 (A*, admissibility). Module 5's forward chaining
# is a useful echo — the delete relaxation is a fixpoint computation.
#
# **Time:** ~75 minutes plus exercises.

# %% [markdown]
# ## 1. Why planning is not just search
#
# Formally, a planning problem *is* a search problem: states, actions, a goal.
# Module 6's A* would solve it. So why a separate field?
#
# Because of what the representation makes visible. In Module 6 a state was an
# opaque tuple and `actions(state)` was a black box, so a heuristic had to be
# written by hand, per domain, by someone who understood it. In planning the
# state is a **set of facts** and each action says exactly **which facts it
# needs and which it changes**. That structure can be analysed automatically —
# and out of it come heuristics that work on domains their author never saw.
#
# > Classical search: *you* supply the heuristic.
# > Classical planning: the **representation** supplies the heuristic.
#
# That shift, which happened in the late 1990s, is what took planning from toy
# problems to hundreds of actions, and it is the main event of this module.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import time

from csai import planning as pl, search
from csai.check import checker
from csai.fol import Var
from csai.planning import (Action, ActionSchema, PlanningProblem, blocks_world,
                           show_blocks)
from csai.render import table

print("ready")

# %% [markdown]
# ## 2. STRIPS
#
# Fikes and Nilsson, 1971, written for Shakey the robot at SRI. An action is
# three sets of facts:
#
# | part | meaning |
# |---|---|
# | **preconditions** | facts that must hold for the action to apply |
# | **add list** | facts it makes true |
# | **delete list** | facts it makes false |
#
# Applying an action to a state: `(state − delete) ∪ add`.
#
# ### The frame problem, and the trick that dodges it
#
# Write this in logic and you immediately face the **frame problem** (McCarthy
# & Hayes, 1969): to conclude anything after an action, you must also state
# everything the action did *not* change. Move a block and the colour of the
# walls is unchanged, and the day of the week, and every other block's
# position, and so on for every fact in the world. The axioms are quadratic in
# the size of the domain, and writing them is hopeless.
#
# STRIPS dodges it by moving the assumption out of the logic and into the
# execution semantics: **whatever is not mentioned does not change.** Combined
# with the closed-world assumption — whatever is not in the state is false —
# this makes actions local and states small. It is a representational
# commitment, not a theorem, and everything else in the field is built on it.

# %%
B, FROM, TO = Var("B"), Var("From"), Var("To")

move = ActionSchema(
    name="move",
    parameters=(B, FROM, TO),
    preconditions=(("on", B, FROM), ("clear", B), ("clear", TO), ("block", TO)),
    add=(("on", B, TO), ("clear", FROM)),
    delete=(("on", B, FROM), ("clear", TO)),
)

print("schema:", move.name, move.parameters)
print("  needs :", move.preconditions)
print("  adds  :", move.add)
print("  deletes:", move.delete)

# %% [markdown]
# Read the delete list carefully. Moving `B` from `From` onto `To` deletes
# `clear(To)` — because `B` is now sitting on it — and adds `clear(From)`. The
# fact that `B` is still a block, that the table is still the table, that every
# other tower is untouched: none of it is mentioned, and none of it changes.

# %% [markdown]
# ## 3. Grounding
#
# A schema is a template. **Grounding** substitutes objects for parameters,
# producing the concrete actions a search can apply. With `p` parameters and
# `n` objects that is up to `n^p` actions, and this is where planning problems
# get big: a modest domain with 4-parameter actions and 30 objects grounds to
# nearly a million.

# %%
sussman = blocks_world(
    blocks=["a", "b", "c"],
    initial_on={"c": "a", "a": "table", "b": "table"},
    goal_on={"a": "b", "b": "c"},
)

print("initial state:")
print(show_blocks(sussman.initial))
print("\ngoal:", sorted(sussman.goal))
print(f"\n{len(sussman.objects)} objects -> {len(sussman.actions)} ground actions")
print(f"of which {len(sussman.applicable(sussman.initial))} are applicable now:")
for a in sussman.applicable(sussman.initial):
    print("   ", a)

# %% [markdown]
# ## 4. The Sussman anomaly
#
# The goal above is two facts: `on(a, b)` and `on(b, c)`. It looks like two
# independent subgoals, and early planners (including STRIPS itself) tried to
# achieve subgoals one at a time and splice the results.
#
# It does not work here.
#
# * Achieve `on(a, b)` first: unstack `c`, put `a` on `b`. Now to get `b` onto
#   `c` you must move `a` off again. You undid your own work.
# * Achieve `on(b, c)` first: put `b` on `c`. Now `a` is under `c`… and `c` is
#   on `b`, so getting `a` out means dismantling the tower.
#
# Neither order works, and no *interleaving-free* strategy solves it. Gerald
# Sussman noticed this in 1973 and it became the standard counterexample to
# "linear planning". The lesson is permanent:
#
# > **Subgoals interact.** You cannot in general solve a conjunctive goal by
# > solving the conjuncts separately and concatenating.
#
# Anyone who has watched a language model confidently decompose a task into
# steps that individually make sense and jointly do not has seen the Sussman
# anomaly.

# %%
result = pl.plan(sussman, "h_ff")
print("plan:")
state = sussman.initial
for i, action in enumerate(result.actions, 1):
    state = action.apply(state)
    print(f"\nstep {i}: {action}")
    print(show_blocks(state))
print(f"\ngoal achieved: {sussman.achieved(state)}")
print("\nNote the first move: unstacking c achieves *neither* goal fact. Any")
print("strategy that only ever works directly on a subgoal cannot find it.")

# %% [markdown]
# ## 5. Heuristics from a relaxation
#
# Now the domain-independent heuristics. The idea that unlocked them is
# beautifully simple:
#
# > **Delete the delete lists.**
#
# Without deletes, facts only ever accumulate. The relaxed problem is monotone,
# so its reachable set is a fixpoint you can compute in polynomial time — and
# because the relaxation only ever makes things easier, the relaxed cost is a
# lower bound on the real one.
#
# Three heuristics come out of it:
#
# | heuristic | definition | admissible? |
# |---|---|---|
# | `h_max` | relaxed cost of the *hardest* single goal fact | yes |
# | `h_add` | relaxed cost of the goal facts, *summed* | no — double-counts shared work |
# | `h_FF` | length of an actual relaxed *plan*, extracted from the planning graph | no, but rarely by much |
#
# `h_add` over-counts because two goals often share most of their work, and
# summing charges for it twice. `h_FF` (Hoffmann & Nebel, 2001) fixes that by
# extracting a relaxed plan and counting its *actions*, so shared work is
# charged once. FF won the 2000 planning competition on the strength of it,
# and the delete relaxation still underpins the state of the art.
#
# The relaxation also gives you something for free: if the goal is not
# reachable *even with deletes ignored*, it is not reachable at all. A cheap
# and complete dead-end detector.

# %%
relaxed = pl.relaxed_reachable(sussman, sussman.initial)
print(f"facts true initially:            {len(sussman.initial)}")
print(f"facts reachable ignoring deletes: {len(relaxed)}")
print(f"goal relaxed-reachable:           {sussman.goal <= relaxed}")

unreachable = blocks_world(["a", "b"], {"a": "table", "b": "table"},
                           {"a": "b", "b": "a"})
print("\nan impossible goal (a on b AND b on a):")
print("  relaxed-reachable:",
      unreachable.goal <= pl.relaxed_reachable(unreachable, unreachable.initial))
print("  — and here the relaxation is fooled: ignoring deletes, a can be on b")
print("    and b on a at the same time. Relaxation gives a lower bound, not a")
print("    decision procedure.")
print("  actually solvable:", bool(pl.plan(unreachable, "h_ff",
                                           max_expansions=3000).found))

# %%
# A harder instance: eight blocks, two configurations with nothing in common.
tall_order = blocks_world(
    blocks=list("abcdefgh"),
    initial_on={"h": "table", "c": "table", "f": "table", "g": "c", "a": "g",
                "d": "f", "b": "table", "e": "table"},
    goal_on={"h": "table", "f": "h", "d": "f", "e": "d", "c": "e", "b": "c",
             "g": "table", "a": "b"},
)
print("initial:"); print(show_blocks(tall_order.initial)); print()
print(f"{len(tall_order.actions)} ground actions\n")

print("first, all five heuristics on the tiny Sussman problem:")
rows = []
for name in ("none", "goals unmet", "h_max", "h_add", "h_ff"):
    t0 = time.perf_counter()
    r = pl.plan(sussman, name)
    rows.append((name, r.depth, r.expanded, f"{time.perf_counter() - t0:.3f}"))
print(table(rows, ["heuristic", "plan length", "nodes expanded", "seconds"],
            align="lrrr"))

print("\nand now on eight blocks, where the differences bite:")
rows = []
for name, budget in [("none", 15000), ("goals unmet", 40000),
                     ("h_add", 40000), ("h_ff", 40000)]:
    t0 = time.perf_counter()
    r = pl.plan(tall_order, name, max_expansions=budget)
    rows.append((name, r.depth if r.found else f"none within {budget}",
                 r.expanded, f"{time.perf_counter() - t0:.2f}"))
print(table(rows, ["heuristic", "plan length", "nodes expanded", "seconds"],
            align="lrrr"))
print("\n(h_max is omitted from the second table: it is admissible but weak,")
print(" and recomputing relaxed costs over 576 ground actions at every node")
print(" costs more than the nodes it saves. Admissibility is not free.)")

# %% [markdown]
# `h_FF` expands a few dozen nodes where counting unmet goals expands
# thousands, and uninformed search does not finish at all — and none of these
# heuristics knows anything about blocks. All of them were derived from the
# action definitions automatically. That is the payoff of the structured
# representation, and it is why "just use search" is not the whole story.
#
# The seconds column is the other half of the lesson. `h_add` expands the
# fewest nodes and takes the longest, because each node costs a full relaxed
# cost computation. `h_FF` expands a few more and finishes fastest. Nodes are
# not the currency; time is.
#
# Note also that `h_add` and `h_FF` may return a *slightly longer* plan. They
# are inadmissible, so A* loses its optimality guarantee. In planning that is
# usually the right trade: a plan a step longer, found a hundred times faster,
# beats an optimal plan you never see.

# %% [markdown]
# ## 6. The validator
#
# Here is the asymmetry that this whole course keeps returning to, in its
# sharpest form:
#
# > **Finding a plan is PSPACE-hard. Checking one is linear.**
#
# A plan is a *claim*: that this sequence of actions, from this state, achieves
# this goal. Verifying the claim means executing it and testing each
# precondition — one pass, no search, no cleverness.
#
# The planning community institutionalised this. Every International Planning
# Competition entry is checked by a separate program, VAL (Howey, Long &
# Fox, 2004), and plans that do not validate simply do not count. No plausible
# argument gets to substitute for execution.
#
# A validator should not just say no. It should say **where**.

# %%
good = pl.plan(sussman, "h_ff").actions
print("checking the planner's own plan:")
print(pl.validate(sussman, good), "\n")


def action_named(problem, name, *args):
    """Look up a ground action by name and arguments."""
    return next(a for a in problem.actions
                if a.name == name and a.args == args)


# A plausible-looking plan that skips a step: put a on b before clearing a.
plausible = [action_named(sussman, "move", "a", "table", "b"),
             action_named(sussman, "move", "b", "table", "c")]
print("checking a plan that reads perfectly well:")
print(pl.validate(sussman, plausible))

# %% [markdown]
# Step 1, and the reason is precise: `clear(a)` does not hold, because `c` is
# sitting on it. Not "this plan seems wrong" — *this fact, at this step*.
#
# That is the same measurement as Module 1's **first divergent step**, and the
# same thing a step-level verifier does to a chain of thought. It is the most
# valuable output a checker can produce, because it is actionable: replan from
# step 1, or repair that one precondition, rather than throwing the plan away.

# %% [markdown]
# ## 7. Bridge to language models
#
# Planning is where LLM reasoning claims have been tested most sharply, and
# the results are unusually clear.
#
# Valmeekam et al.'s **PlanBench** (2022–2024) put language models on exactly
# the blocks world above. The findings, repeatedly reproduced:
#
# * Models produce plans that read fluently, use the right vocabulary, and are
#   frequently **invalid** — a precondition unmet three steps in, an effect
#   assumed that the action does not have.
# * Accuracy falls sharply with plan length, which is Module 1's curve again.
# * Obfuscating the names ("block a" → "object qxr") degrades performance
#   markedly, which suggests a good deal of what looks like planning is
#   retrieval of familiar patterns.
# * Reasoning-trained models do substantially better and still fall off with
#   length.
#
# The productive architecture that came out of this is called **LLM-Modulo**
# (Kambhampati et al., 2024), and you have now built both halves of it:
#
# > The model **proposes**. A sound verifier **disposes**. The verifier's
# > complaint — *this precondition, at this step* — goes back to the model as
# > the next prompt. Loop.
#
# Every part of that is load-bearing:
#
# * The model is genuinely good at the part that is hard to automate: reading
#   an English goal, inventing plausible decompositions, retrieving relevant
#   domain knowledge.
# * The validator is cheap, sound, and *complete for what it checks* — it
#   never accepts a broken plan, and it does not need to be smart.
# * The **error message** is what makes the loop converge. "Wrong, try again"
#   wastes a round; "step 1's precondition `clear(a)` fails" localises the fix.
#
# This is also the right way to read every agent framework you will meet.
# ReAct, tool-use loops, self-refine: all of them are proposal-plus-feedback,
# and their quality is set almost entirely by how good the feedback is. A
# compiler error, a failing test, a type checker, a precondition check — these
# are the verifiers that make the loop work. Where no such checker exists, the
# loop degenerates into the model grading its own homework, and the literature
# on self-correction without external feedback is accordingly discouraging.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — is the action applicable?
#
# Write `applicable(action, state)`: are all of `action.preconditions` in
# `state`? Both are frozensets.

# %%
def applicable(action, state):
    """True when every precondition holds in `state`."""
    # TODO: subset test
    return None


# %%
@checker("Exercise 9.1 — applicable")
def check_ex1():
    s0 = sussman.initial
    clear_c = action_named(sussman, "to_table", "c", "a")
    yield "an available action", applicable(clear_c, s0), True
    blocked = action_named(sussman, "move", "a", "table", "b")
    yield "a blocked action", applicable(blocked, s0), False
    yield ("…becomes available once c is moved",
           applicable(blocked, clear_c.apply(s0)), True)
    yield ("an action with no preconditions always applies",
           applicable(Action("noop"), frozenset()), True)
    yield ("agrees with csai.planning", applicable(blocked, s0),
           blocked.applicable(s0))


check_ex1()

# %% [markdown]
# ### Exercise 2 — apply it
#
# Write `apply_action(action, state)` returning `(state − delete) ∪ add`.
# Return a frozenset; do not modify `state`.

# %%
def apply_action(action, state):
    """The state after `action`."""
    # TODO: remove the delete list, add the add list
    return None


# %%
@checker("Exercise 9.2 — apply_action")
def check_ex2():
    s0 = sussman.initial
    unstack = action_named(sussman, "to_table", "c", "a")
    after = apply_action(unstack, s0)
    yield "returns a frozenset", isinstance(after, frozenset), True
    yield "the add list is present", ("on", "c", "table") in (after or ()), True
    yield "…including the freed block", ("clear", "a") in (after or ()), True
    yield "the delete list is gone", ("on", "c", "a") in (after or ()), False
    yield "untouched facts survive", ("block", "b") in (after or ()), True
    yield "the original state is unchanged", ("on", "c", "a") in s0, True
    yield "agrees with csai.planning", after, unstack.apply(s0)


check_ex2()

# %% [markdown]
# ### Exercise 3 — grounding
#
# Write `ground(schema, objects)` returning the list of ground `Action`s: one
# per assignment of distinct objects to parameters, in
# `itertools.product` order. Use `csai.fol.substitute` on each fluent.

# %%
def ground(schema, objects):
    """Every ground instance of `schema` over `objects`, parameters distinct."""
    # TODO: itertools.product over the parameters, skip repeats, substitute
    return None


# %%
@checker("Exercise 9.3 — ground")
def check_ex3():
    objects = ("a", "b", "table")
    got = ground(pl.MOVE, objects)
    yield "3 objects, 3 parameters, all distinct -> 6 actions", len(got or []), 6
    yield "all are Actions", all(isinstance(a, Action) for a in (got or [])), True
    yield ("no parameter repeats",
           all(len(set(a.args)) == 3 for a in (got or [])), True)

    one = next((a for a in (got or []) if a.args == ("a", "table", "b")), None)
    yield "arguments recorded", one.args if one else None, ("a", "table", "b")
    yield ("preconditions substituted",
           one.preconditions if one else None,
           frozenset({("on", "a", "table"), ("clear", "a"), ("clear", "b"),
                      ("block", "b")}))
    yield ("effects substituted",
           (one.add, one.delete) if one else None,
           (frozenset({("on", "a", "b"), ("clear", "table")}),
            frozenset({("on", "a", "table"), ("clear", "b")})))
    yield ("agrees with csai.planning",
           sorted(map(repr, got or [])), sorted(map(repr, pl.MOVE.ground(objects))))


check_ex3()

# %% [markdown]
# ### Exercise 4 — validate a plan
#
# Write `check_plan(problem, plan)` returning
# `(valid, failed_step, missing)`:
#
# * `valid` — did every action apply and the goal hold at the end?
# * `failed_step` — the **1-based** index of the first inapplicable action,
#   or `None` if all applied;
# * `missing` — the frozenset of facts that were needed and absent: the
#   failing action's unmet preconditions, or the unachieved goal facts if the
#   plan ran to the end without reaching the goal.
#
# This is the module's core deliverable. Take it seriously.

# %%
def check_plan(problem, plan):
    """(valid, first failing step or None, the facts that were missing)."""
    # TODO: execute step by step from problem.initial
    return None


# %%
@checker("Exercise 9.4 — check_plan")
def check_ex4():
    good_plan = pl.plan(sussman, "h_ff").actions
    yield "a correct plan validates", check_plan(sussman, good_plan), (
        True, None, frozenset())

    skipping = [action_named(sussman, "move", "a", "table", "b"),
                action_named(sussman, "move", "b", "table", "c")]
    yield ("a plan whose first step is blocked",
           check_plan(sussman, skipping),
           (False, 1, frozenset({("clear", "a")})))

    late = [action_named(sussman, "to_table", "c", "a"),
            action_named(sussman, "move", "a", "table", "b")]
    yield ("a plan that runs out before the goal",
           check_plan(sussman, late)[:2], (False, 2))
    yield ("…and says which goal facts are unmet",
           check_plan(sussman, late)[2], frozenset({("on", "b", "c")}))

    yield ("the empty plan does not achieve the goal",
           check_plan(sussman, [])[0], False)
    yield ("a failure mid-plan is localised",
           check_plan(sussman, good_plan[:1] + skipping)[1], 3)

    already = blocks_world(["a", "b"], {"a": "b", "b": "table"}, {"a": "b"})
    yield ("the empty plan is valid when the goal already holds",
           check_plan(already, []), (True, None, frozenset()))


check_ex4()

# %% [markdown]
# ### Exercise 5 — the delete relaxation
#
# Write `relaxed_reach(problem, state)`: the frozenset of every fact reachable
# when actions add but never delete. Iterate to a fixpoint.

# %%
def relaxed_reach(problem, state):
    """Every fact reachable if delete lists are ignored."""
    # TODO: repeat until nothing new: any applicable action contributes its adds
    return None


# %%
@checker("Exercise 9.5 — relaxed_reach")
def check_ex5():
    got = relaxed_reach(sussman, sussman.initial)
    yield "a frozenset", isinstance(got, frozenset), True
    yield "contains the initial state", sussman.initial <= (got or frozenset()), True
    yield "and the goal", sussman.goal <= (got or frozenset()), True
    yield ("agrees with csai.planning",
           got, pl.relaxed_reachable(sussman, sussman.initial))
    yield ("it is a fixpoint",
           relaxed_reach(sussman, got), got)
    yield ("relaxation is optimistic: it 'reaches' an impossible goal",
           unreachable.goal <= relaxed_reach(unreachable, unreachable.initial),
           True)


check_ex5()

# %% [markdown]
# ### Exercise 6 — a dead-end detector
#
# Write `possibly_solvable(problem, state)`: `False` when the goal is not
# reachable even in the relaxed problem — in which case it is certainly
# unreachable in the real one. `True` otherwise, which promises nothing.
#
# One-sided tests like this are everywhere in search: cheap, sound in one
# direction, and worth a great deal because they prune whole subtrees.

# %%
def possibly_solvable(problem, state):
    """False proves unsolvable; True proves nothing."""
    # TODO: is the goal inside the relaxed reachable set?
    return None


# %%
@checker("Exercise 9.6 — possibly_solvable")
def check_ex6():
    yield "the Sussman anomaly might be solvable", possibly_solvable(
        sussman, sussman.initial), True
    yield "…and in fact is", bool(pl.plan(sussman, "h_ff").found), True

    stuck = PlanningProblem(
        objects=("a", "b", "table"),
        schemas=(pl.MOVE, pl.MOVE_TO_TABLE),
        initial=frozenset({("block", "a"), ("block", "b"),
                           ("on", "a", "table"), ("on", "b", "table"),
                           ("clear", "a"), ("clear", "b")}),
        goal=frozenset({("on", "a", "moon")}),
    )
    yield ("a goal mentioning an object that does not exist is refuted",
           possibly_solvable(stuck, stuck.initial), False)
    yield ("the relaxation cannot refute a and b on each other",
           possibly_solvable(unreachable, unreachable.initial), True)
    yield ("a goal already achieved is trivially possible",
           possibly_solvable(sussman, sussman.initial | sussman.goal), True)


check_ex6()

# %% [markdown]
# ---
# ## Project — a planner, and a grader for other people's plans
#
# Two halves, and the second is the point.
#
# **Part 1 — the planner.**
#
# ```python
# find_plan(problem, heuristic="h_ff", max_expansions=40000) -> dict
# ```
#
# returning `{"plan": [Action] | None, "length": int | None,
# "expanded": int, "found": bool}`. Use A* over the state space with the named
# heuristic from `csai.planning.HEURISTICS`. Reuse `csai.search` or write your
# own loop — your choice.
#
# **Part 2 — the grader.**
#
# ```python
# grade(problem, candidates) -> {name: report}
# ```
#
# where `candidates` is `{name: [(action_name, args…), …]}` — plans written as
# plain tuples, the way a language model would hand them to you. Each report
# has:
#
# | key | meaning |
# |---|---|
# | `"valid"` | did it work? |
# | `"failed_step"` | 1-based index of the first problem, or `None` |
# | `"reason"` | `"ok"`, `"no such action"`, `"precondition"`, or `"goal not reached"` |
# | `"missing"` | the frozenset of facts that were needed and absent |
# | `"length"` | number of steps |
#
# An action tuple that names no legal ground action must be reported, not
# crash on — a model will produce those, and "move block a onto block a" needs
# an error message rather than a traceback.
#
# **Write-up questions:**
#
# 1. For each flawed candidate, state the smallest repair. Which are one-step
#    fixes and which need replanning from scratch?
# 2. `"reason"` distinguishes three kinds of failure. If you were feeding this
#    back to a model that proposed the plan, would you phrase all three the
#    same way? What would you say for each?
# 3. Compare the cost of `find_plan` against the cost of `grade` on the same
#    problem. Then argue for or against: *"if you have a validator, the
#    proposer does not need to be reliable."* Give the condition under which
#    the argument fails.

# %%
CANDIDATE_PLANS = {
    # The plan a careful person writes.
    "correct": [("to_table", "c", "a"),
                ("move", "b", "table", "c"),
                ("move", "a", "table", "b")],
    # Reads perfectly; a is not clear.
    "forgot to unstack": [("move", "a", "table", "b"),
                          ("move", "b", "table", "c")],
    # Right moves, wrong order — the Sussman trap.
    "subgoals in the wrong order": [("to_table", "c", "a"),
                                    ("move", "a", "table", "b"),
                                    ("move", "b", "table", "c")],
    # Stops one step early.
    "gives up early": [("to_table", "c", "a"),
                       ("move", "b", "table", "c")],
    # Invents an action that does not exist.
    "hallucinated action": [("teleport", "a", "b")],
    # Correct, but with a pointless detour.
    "wasteful but valid": [("to_table", "c", "a"),
                           ("move", "a", "table", "c"),
                           ("to_table", "a", "c"),
                           ("move", "b", "table", "c"),
                           ("move", "a", "table", "b")],
}


def find_plan(problem, heuristic="h_ff", max_expansions=40000):
    """A* planning. Returns the report described above."""
    # TODO: search over states; successors are problem.applicable(state)
    return None


def grade(problem, candidates):
    """Validate plans written as (action_name, *args) tuples."""
    # TODO: resolve each tuple to a ground action, then execute step by step
    return None


# %%
@checker("Project 9 — planner and grader")
def check_project():
    report = find_plan(sussman)
    yield "returns the required keys", sorted(report or {}), [
        "expanded", "found", "length", "plan"]
    yield "solves the Sussman anomaly", (report or {}).get("found"), True
    yield "…in three steps", (report or {}).get("length"), 3
    yield ("…with a plan that validates",
           bool(pl.validate(sussman, (report or {}).get("plan") or [])), True)
    yield ("…and its first move unstacks c",
           repr(((report or {}).get("plan") or [None])[0]), "to_table(c, a)")

    strong = find_plan(tall_order, "h_ff")
    weak = find_plan(tall_order, "goals unmet")
    yield "solves the eight-block problem", strong["found"], True
    yield ("…and h_ff expands far fewer nodes than counting unmet goals",
           strong["expanded"] < weak["expanded"] / 10, True)

    reports = grade(sussman, CANDIDATE_PLANS)
    yield "grades every candidate", sorted(reports or {}), sorted(CANDIDATE_PLANS)

    r = (reports or {})
    yield "the correct plan passes", r["correct"]["valid"], True
    yield "…with no failing step", r["correct"]["failed_step"], None
    yield "…and reason 'ok'", r["correct"]["reason"], "ok"

    yield "the unstacking mistake fails", r["forgot to unstack"]["valid"], False
    yield "…at step 1", r["forgot to unstack"]["failed_step"], 1
    yield "…on a precondition", r["forgot to unstack"]["reason"], "precondition"
    yield ("…naming the missing fact",
           r["forgot to unstack"]["missing"], frozenset({("clear", "a")}))

    yield ("the wrong order fails later, not at the start",
           r["subgoals in the wrong order"]["failed_step"], 3)

    yield ("stopping early is a goal failure, not a precondition failure",
           r["gives up early"]["reason"], "goal not reached")
    yield ("…and says which goal fact is unmet",
           r["gives up early"]["missing"], frozenset({("on", "a", "b")}))

    yield ("an invented action is reported, not crashed on",
           r["hallucinated action"]["reason"], "no such action")
    yield "…at its step", r["hallucinated action"]["failed_step"], 1

    yield ("a wasteful plan is still valid",
           r["wasteful but valid"]["valid"], True)
    yield ("…and is longer than the planner's",
           r["wasteful but valid"]["length"] > (report or {})["length"], True)


check_project()

# %%
# The grading report your write-up discusses.
if find_plan(sussman) is not None:
    rows = []
    for name, report in grade(sussman, CANDIDATE_PLANS).items():
        rows.append((name, report["length"], "yes" if report["valid"] else "no",
                     report["failed_step"] or "-", report["reason"],
                     ", ".join(sorted(map(str, report["missing"]))) or "-"))
    print(table(rows, ["candidate plan", "steps", "valid", "failed at",
                       "reason", "missing facts"], align="lrccll"))

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * R. Fikes & N. Nilsson, "STRIPS: A New Approach to the Application of
#   Theorem Proving to Problem Solving" (1971).
# * J. McCarthy & P. Hayes, "Some Philosophical Problems from the Standpoint
#   of Artificial Intelligence" (1969) — the frame problem.
# * A. Blum & M. Furst, "Fast Planning Through Planning Graph Analysis" (1997)
#   — GRAPHPLAN, where the relaxed graph comes from.
# * J. Hoffmann & B. Nebel, "The FF Planning System" (2001).
# * R. Howey, D. Long & M. Fox, "VAL: Automatic Plan Validation" (2004).
# * K. Valmeekam et al., "PlanBench" (2022–2024) — language models on the
#   blocks world.
# * S. Kambhampati et al., "LLMs Can't Plan, But Can Help Planning in
#   LLM-Modulo Frameworks" (2024) — propose-and-verify, argued in full.
#
# **Next:** Module 10 drops the assumption that has been silently in force
# since Module 2 — that you know which facts are true. Probability, Bayesian
# networks, and inference under uncertainty.
