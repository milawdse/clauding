# %% [markdown]
# # Module 11 — Decisions, Metareasoning, and How Long to Think
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Choose actions by maximum expected utility, and say why utility is not
#    money.
# 2. Solve a Markov decision process with value iteration and policy
#    iteration, and read the policy off the value function.
# 3. Compute the **value of information**, and explain the case where it is
#    exactly zero.
# 4. State the metareasoning problem — *computation is an action* — and
#    implement a cost-aware stopping rule.
# 5. Build a controller that spends a fixed thinking budget across a set of
#    problems, and measure that **a verifier is worth more than a better
#    allocation policy**.
#
# **Prerequisites:** Module 10 (probability). Module 7's value-function
# finding and Module 9's validator both pay off here.
#
# **Time:** ~80 minutes plus exercises.

# %% [markdown]
# ## 1. From belief to action
#
# Module 10 produced beliefs. Beliefs are not decisions. The bridge is
# **utility**: a number on each outcome saying how much you want it, and the
# rule that you should choose the action maximising its expectation.
#
# > **MEU:** choose `argmaxₐ Σ P(outcome | a) · U(outcome)`
#
# Like probability, this is not one option among many. Von Neumann and
# Morgenstern (1944) showed that any preference ordering satisfying a handful
# of consistency axioms — transitivity, and a few others that look
# unobjectionable — *is* the maximisation of expected utility for some utility
# function. Violate MEU systematically and someone can construct a series of
# bets you will accept and reliably lose on.
#
# **Utility is not money.** Bernoulli's St Petersburg paradox (1738): a coin is
# tossed until it comes up heads, paying `2ⁿ` for `n` tosses. The expected
# payoff is `½·2 + ¼·4 + ⅛·8 + … = ∞`, and nobody will pay much to play.
# Bernoulli's resolution — utility is concave in wealth, so a doubling of money
# is less than a doubling of value — is also the definition of risk aversion,
# and it means the numbers you maximise are not the numbers on the banknotes.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import math
import random
import statistics

from csai import decision as dec
from csai.check import checker
from csai.decision import GridWorld, PerformanceProfile, show_policy
from csai.render import bar_chart, grid, table

print("ready")

# %%
# A single decision: ship the change now, test it first, or leave it alone?
# The hidden state is whether the change is defective.
UTILITY = {
    "shipped_ok": 10.0,          # shipped a working change
    "outage": -40.0,             # shipped a broken one
    "tested_then_shipped": 8.0,  # same, minus the two units the test cost
    "caught": -2.0,              # the test found the bug: no value, cost paid
    "outage_after_test": -42.0,  # the test missed it, and we paid anyway
    "no_value": 0.0,
}


def utility(outcome):
    return UTILITY[outcome]


def outcome_given(action, defective):
    """What happens, given the action and whether the change is defective."""
    if action == "do nothing":
        return {"no_value": 1.0}
    if action == "ship now":
        return {"outage": 1.0} if defective else {"shipped_ok": 1.0}
    if defective:                                 # testing catches most bugs
        return {"caught": 0.9, "outage_after_test": 0.1}
    return {"tested_then_shipped": 1.0}


def marginal(action, world):
    """The outcome distribution, averaging over the unknown state."""
    out = {}
    for state, w in world.items():
        for o, p in outcome_given(action, state).items():
            out[o] = out.get(o, 0.0) + w * p
    return out


WORLD = {True: 0.25, False: 0.75}          # a 25% chance the change is broken
ACTIONS = {a: marginal(a, WORLD)
           for a in ("ship now", "test first", "do nothing")}

rows = []
for name, outcomes in ACTIONS.items():
    rows.append((name, ", ".join(f"{o} {p:.0%}" for o, p in outcomes.items()),
                 f"{dec.expected_utility(outcomes, utility):+.2f}"))
print(table(rows, ["action", "outcomes", "expected utility"], align="llr"))
print("\nTesting costs two units and is still clearly right, because the")
print("downside is large. Halve the cost of an outage and the answer flips.")
print("The decision lives in the ratio of the stakes, not the probabilities.")

# %% [markdown]
# ## 2. Sequential decisions: MDPs
#
# One decision is easy. A *sequence* of them, in a world that responds
# stochastically, is a **Markov decision process**: states, actions, a
# transition model `P(s′ | s, a)`, and a reward per state. The answer is not an
# action but a **policy** — what to do in every state.
#
# The standard example is a 4×3 grid. You intend to move in a direction and
# succeed 80% of the time; 20% of the time you slip to one side. There is a
# goal (+1), a pit (−1), a wall, and a small cost per step.
#
# The **Bellman equation** defines the value of a state:
#
# > `V(s) = R(s) + γ · maxₐ Σ P(s′ | s, a) · V(s′)`
#
# Reward here, plus the discounted value of acting optimally from wherever you
# land. `γ < 1` discounts the future, which keeps infinite-horizon sums finite
# and expresses a preference for reward sooner.
#
# **Value iteration** applies that equation as an assignment, over and over,
# until nothing moves. It converges because the backup is a contraction by
# `γ`: every sweep shrinks the error, so the fixpoint is unique and you reach
# it from any starting guess.

# %%
world = GridWorld(step_reward=-0.04, discount=0.9)
stats = {}
values = dec.value_iteration(world, stats=stats)
policy = dec.extract_policy(world, values)

print(f"converged in {stats['sweeps']} sweeps\n")
print("values:")
print(grid([[f"{values[(r, c)]:+.3f}" if (r, c) not in world.walls else "wall"
             for c in range(world.cols)] for r in range(world.rows)]))
print("\noptimal policy  (+ goal, - pit, # wall):")
print(grid(show_policy(world, policy), cell_width=1))

# %% [markdown]
# Look at the bottom-right cell, next to the pit. The policy points **left,
# away from the goal**. Going up would be the direct route, and a 20% slip
# would drop you in the pit. The long way round is worth the extra steps.
#
# That is a genuinely non-obvious plan, and nobody wrote it. It fell out of the
# reward structure and the noise model. Change the cost of a step and the whole
# character of the policy changes.

# %%
for step_reward, comment in [(-0.04, "a mild cost: take the safe route"),
                             (-2.0, "steps are expensive: risk the shortcut"),
                             (0.0, "steps are free: never risk anything")]:
    w = GridWorld(step_reward=step_reward, discount=0.9)
    p = dec.extract_policy(w, dec.value_iteration(w))
    print(f"step reward {step_reward:+.2f} — {comment}")
    print(grid(show_policy(w, p), cell_width=1))
    print()

# %% [markdown]
# ### Policy iteration
#
# Value iteration refines numbers. **Policy iteration** alternates two steps:
# evaluate the current policy exactly (no `max`, so it is a linear system),
# then improve it greedily. It usually converges in a handful of rounds,
# because the policy stops changing long before the numbers stop moving —
# which is worth remembering whenever you are tempted to run an optimiser to
# high precision for a decision that only needs a ranking.

# %%
pi_stats = {}
pi_policy, pi_values = dec.policy_iteration(world, stats=pi_stats)
print(f"value iteration:  {stats['sweeps']} sweeps")
print(f"policy iteration: {pi_stats['rounds']} rounds")
print(f"same policy:      {pi_policy == policy}")

# %% [markdown]
# ## 3. The value of information
#
# Now a question that leads directly to metareasoning: **what is it worth to
# find something out before deciding?**
#
# The value of perfect information about a variable is
#
# > (expected utility if you knew it, then chose) − (expected utility choosing
# > now)
#
# Two properties, both important:
#
# * **It is never negative.** More information cannot hurt a rational agent —
#   you can always ignore it. (Note what this does *not* say about agents with
#   limited computation, who can be swamped by irrelevant data.)
# * **It is zero whenever the information cannot change your action.** If you
#   would ship either way, testing tells you nothing worth paying for. This is
#   the practical form of the rule, and it is the one people forget.

# %%
rows = []
for prior in (0.0, 0.02, 0.1, 0.25, 0.5, 0.9, 1.0):
    belief = {True: prior, False: 1 - prior}
    without = dec.best_action({a: marginal(a, belief) for a in ACTIONS}, utility)
    vpi = dec.value_of_perfect_information(ACTIONS, utility, belief, outcome_given)
    rows.append((f"{prior:.2f}", without[0], f"{without[1]:+.2f}", f"{vpi:+.3f}"))
print(table(rows, ["P(defective)", "best action without knowing",
                   "its value", "value of knowing"], align="rlrr"))
print("\nThe end points are zero, and that is the whole lesson. If you are")
print("certain the change is fine you ship; if you are certain it is broken")
print("you do not; in neither case does being told change anything. Between")
print("them the decision is genuinely poised and information is worth most.")
print("Information is valuable exactly in proportion to how often it flips")
print("a decision — never for its own sake.")

# %% [markdown]
# ## 4. Metareasoning: thinking is an action
#
# Here is the step that makes this module about reasoning.
#
# Deliberation costs time and buys accuracy. So the question *"should I think
# for longer?"* is a decision like any other, and can be made by the same
# criterion. This is **metareasoning** — reasoning about reasoning — and the
# classical treatment is Russell and Wefald's *Do the Right Thing* (1991).
#
# The framework needs two ingredients:
#
# * a **performance profile**: how good the answer is as a function of effort;
# * a **cost of time**: what a unit of deliberation is worth elsewhere.
#
# Then keep computing while the *marginal* value of another unit exceeds its
# cost, and stop when it does not. Note the word marginal — the total value of
# thinking may be enormous while the value of the next second is negative.
#
# Russell's distinction is useful vocabulary: **type-I rationality** is picking
# the best action, which is usually intractable. **Type-II rationality** is
# picking the best action *given what your deliberation costs* — the best you
# can actually do, and a coherent standard rather than a resigned one.

# %%
def quality(difficulty, effort):
    """Probability of a correct answer at this effort, for this difficulty.

    One error probability per reasoning step, falling with effort; the answer
    is right only if every step is. So harder problems need more thinking to
    reach the same accuracy — the compounding Module 1 measured.
    """
    per_step_error = 0.5 * (0.6 ** effort)
    return (1 - per_step_error) ** difficulty


print(table([[d] + [f"{quality(d, e):.3f}" for e in (0, 1, 2, 4, 8, 16)]
             for d in (1, 2, 3, 4, 6)],
            ["difficulty", "e=0", "e=1", "e=2", "e=4", "e=8", "e=16"],
            align="rrrrrrr"))

# %%
for cost in (0.0, 0.005, 0.02, 0.10):
    profile = PerformanceProfile(quality, cost_per_unit=cost)
    stops = {d: profile.stopping_point(d, 25) for d in (1, 2, 3, 4, 6)}
    print(f"cost per unit {cost:.3f} -> stop at effort {stops}")
print("\nFree thinking means think forever. As the cost rises the optimal")
print("effort falls — and falls fastest for the easy problems, which reach")
print("diminishing returns first. That is the whole content of 'think harder")
print("about harder things', derived rather than asserted.")

# %% [markdown]
# ## 5. Spending a budget across many problems
#
# Now the practical version. You have `N` problems and a **fixed total budget**
# of thinking. How do you spend it?
#
# Three strategies, in increasing order of how much they know:
#
# 1. **Uniform** — everyone gets `B/N`. The baseline.
# 2. **Adaptive** — give each unit to whichever problem gains most from it
#    (greedy on marginal quality, which is optimal when the curves are
#    concave).
# 3. **Verifier-driven** — the same greedy allocation, but after each unit you
#    *check whether the answer is already right*, and if it is you stop
#    spending on that problem and move the budget elsewhere.
#
# The third strategy needs something the other two do not have: a way to tell
# a correct answer from an incorrect one. That is Module 9's validator, or
# Module 3's model check, or Module 12's interpreter. Watch what it is worth.

# %%
DIFFICULTIES = [1, 1, 2, 2, 3, 3, 4, 5, 6, 6] * 4     # 40 problems
profile = PerformanceProfile(quality)


def run_allocation(allocation, rng):
    """Solve each problem once at its allotted effort. Returns the fraction right."""
    return sum(rng.random() < quality(d, e)
               for d, e in zip(DIFFICULTIES, allocation)) / len(DIFFICULTIES)


def run_verified(budget, rng, cap=25):
    """Spend one unit at a time, checking after each and stopping when right."""
    solved = [False] * len(DIFFICULTIES)
    effort = [0] * len(DIFFICULTIES)
    spent = 0
    while spent < budget:
        live = [i for i in range(len(DIFFICULTIES))
                if not solved[i] and effort[i] < cap]
        if not live:
            break
        i = max(live, key=lambda j: quality(DIFFICULTIES[j], effort[j] + 1)
                - quality(DIFFICULTIES[j], effort[j]))
        effort[i] += 1
        spent += 1
        if rng.random() < quality(DIFFICULTIES[i], effort[i]):
            solved[i] = True                       # the verifier confirms it
    return sum(solved) / len(DIFFICULTIES)


rows = []
for budget in (40, 80, 160, 320):
    uniform = statistics.mean(
        run_allocation(dec.allocate_uniform(DIFFICULTIES, budget),
                       random.Random(s)) for s in range(200))
    adaptive = statistics.mean(
        run_allocation(dec.allocate_greedy(profile, DIFFICULTIES, budget,
                                           max_each=25),
                       random.Random(s)) for s in range(200))
    verified = statistics.mean(run_verified(budget, random.Random(s))
                               for s in range(200))
    rows.append((budget, f"{uniform:.3f}", f"{adaptive:.3f}", f"{verified:.3f}"))
print(table(rows, ["budget", "uniform", "adaptive", "verifier-driven"],
            align="rrrr"))

# %% [markdown]
# **Read the gaps.** Adaptive allocation beats uniform by one or two points —
# real, and modest. Adding a verifier takes the same budget from 55% to 94%.
#
# > The allocation policy is worth a couple of points. **The verifier is worth
# > forty.**
#
# The reason is not subtle once stated. Without a checker you must guess how
# much effort a problem needs and spend it all up front, and any effort past
# the point where the answer became correct is wasted — invisibly. With a
# checker you stop the moment you are right and the surplus goes to a problem
# that still needs it. Most of a fixed budget, spent blindly, is spent on
# problems that were already solved.
#
# This is the same conclusion Module 7 reached from tree search and Module 9
# reached from planning, arrived at a third time from decision theory. It is
# the most consistent empirical finding in this course.

# %%
# Look at where the adaptive allocator actually puts the effort.
print("effort allocated by difficulty:\n")
rows = []
for budget in (40, 80, 160, 320):
    allocation = dec.allocate_greedy(profile, DIFFICULTIES, budget, max_each=25)
    by_difficulty = {}
    for d, e in zip(DIFFICULTIES, allocation):
        by_difficulty.setdefault(d, []).append(e)
    rows.append([budget] + [round(statistics.mean(by_difficulty[d]), 1)
                            for d in sorted(by_difficulty)])
print(table(rows, ["budget"] + [f"difficulty {d}" for d in sorted(set(DIFFICULTIES))],
            align="r" * 7))
print("\nAt a tight budget the allocator *abandons* the hardest problems and")
print("spends on the easy ones — triage. As the budget grows it reverses and")
print("pours effort into the hard ones. Neither behaviour was programmed;")
print("both fall out of where the marginal gain is largest.")

# %% [markdown]
# ## 6. Bridge to language models
#
# This module is the theory of **test-time compute**, written down thirty
# years early.
#
# | classical | LLM reasoning |
# |---|---|
# | performance profile | accuracy against tokens of reasoning |
# | anytime algorithm | a model that can be interrupted for its current best answer |
# | cost of time | latency and price per token |
# | optimal stopping | deciding when to stop generating |
# | value of information | is another tool call, or another sample, worth it? |
# | value function `V(s)` | a process reward model scoring a partial chain |
# | type-II rationality | the best answer *for the budget*, which is the only kind on offer |
#
# Four things follow, and they are practical rather than philosophical.
#
# 1. **Uniform thinking budgets are wasteful.** Most benchmarks and most
#    deployments give every question the same treatment. §5 says that is close
#    to the worst allocation available, and the fix — route by predicted
#    difficulty — is exactly what "easy/hard routing" and adaptive reasoning
#    modes are doing.
# 2. **A verifier is worth more than a smarter policy.** Where a cheap correct
#    checker exists — a compiler, a test suite, a unit conversion, an
#    interpreter running generated code — build the loop around it. This is
#    the strongest recommendation in the course, and §5 is the third
#    independent measurement of it.
# 3. **Knowing when to stop is a skill, and a separate one.** A model that
#    reasons for four thousand tokens about "what is 2 + 2" is failing at
#    metareasoning, not at arithmetic. So is one that answers a hard question
#    instantly. Both are the stopping rule of §4, gone wrong in opposite
#    directions.
# 4. **Marginal value is what matters, not total value.** "More thinking helps"
#    is true and useless. The question is whether the *next* unit helps more
#    than it costs, and the answer depends on the problem, the budget, and
#    what else the compute could have done.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — expected utility
#
# Write `expected_value(outcomes, utility)` where `outcomes` is
# `{outcome: probability}` and `utility` is a function.

# %%
def expected_value(outcomes, utility):
    """Σ P(outcome) · U(outcome)."""
    # TODO: one line
    return None


# %%
@checker("Exercise 11.1 — expected_value")
def check_ex1():
    yield "a certainty", expected_value({"a": 1.0}, lambda o: 5.0), 5.0
    yield ("a fair coin",
           expected_value({"h": 0.5, "t": 0.5}, {"h": 10, "t": 0}.get), 5.0)
    yield ("a skewed bet",
           expected_value({"win": 0.1, "lose": 0.9}, {"win": 100, "lose": -5}.get),
           5.5)
    yield ("negatives are handled",
           expected_value({"a": 0.5, "b": 0.5}, {"a": -10, "b": -20}.get), -15.0)
    yield ("agrees with the lecture's shipping decision",
           expected_value(ACTIONS["ship now"], utility),
           dec.expected_utility(ACTIONS["ship now"], utility))
    yield "no outcomes -> zero", expected_value({}, lambda o: 1.0), 0


check_ex1()

# %% [markdown]
# ### Exercise 2 — maximum expected utility
#
# Write `choose(actions, utility)` returning `(best_action, its_value)` for
# `actions` a `{action: {outcome: probability}}` dict. Break ties by taking
# the first in iteration order.

# %%
def choose(actions, utility):
    """The MEU action and its expected utility."""
    # TODO: score every action, return the best
    return None


# %%
@checker("Exercise 11.2 — choose")
def check_ex2():
    simple = {"safe": {"ok": 1.0}, "risky": {"great": 0.5, "awful": 0.5}}
    u = {"ok": 5.0, "great": 20.0, "awful": -20.0}.get
    yield "prefers the certain payoff here", choose(simple, u)[0], "safe"
    yield "…and reports its value", choose(simple, u)[1], 5.0

    better = {"safe": {"ok": 1.0}, "risky": {"great": 0.5, "awful": 0.5}}
    u2 = {"ok": 5.0, "great": 20.0, "awful": -5.0}.get
    yield ("prefers the gamble when it pays",
           choose(better, u2)[0], "risky")

    yield ("agrees with csai.decision",
           choose(ACTIONS, utility), dec.best_action(ACTIONS, utility))
    yield ("a single action is chosen by default",
           choose({"only": {"x": 1.0}}, lambda o: 3.0), ("only", 3.0))


check_ex2()

# %% [markdown]
# ### Exercise 3 — the Bellman backup
#
# Write `bellman(mdp, state, values)`: for a terminal state return its reward;
# otherwise `R(s) + γ · maxₐ Σ P(s′|s,a)·V(s′)`. Use `mdp.transitions`,
# `mdp.actions`, `mdp.reward` and `mdp.discount`.

# %%
def bellman(mdp, state, values):
    """One Bellman backup at `state`."""
    # TODO: reward, plus the discounted best expected next value
    return None


# %%
@checker("Exercise 11.3 — bellman")
def check_ex3():
    zeros = {s: 0.0 for s in world.states()}
    yield ("with zero values it is just the reward",
           round(bellman(world, (2, 0), zeros), 6), round(world.reward((2, 0)), 6))
    yield ("a terminal state is its reward",
           bellman(world, (0, 3), zeros), 1.0)
    yield ("…including the pit", bellman(world, (1, 3), zeros), -1.0)
    yield ("next to the goal, the backup sees it",
           bellman(world, (0, 2), {**zeros, (0, 3): 1.0}) >
           world.reward((0, 2)), True)
    for s in [(0, 0), (2, 2), (1, 2)]:
        yield (f"agrees with csai.decision at {s}",
               round(bellman(world, s, values), 9),
               round(dec.bellman_backup(world, s, values), 9))
    yield ("the converged values are a fixpoint",
           all(abs(bellman(world, s, values) - values[s]) < 1e-4
               for s in world.states()), True)


check_ex3()

# %% [markdown]
# ### Exercise 4 — value iteration
#
# Write `value_iterate(mdp, epsilon=1e-6, max_sweeps=1000)`: start every state
# at 0, apply `bellman` everywhere in each sweep, and stop when the largest
# change falls below `epsilon`. Return the value dict.
#
# Update **synchronously** — compute all the new values from the old ones,
# then swap.

# %%
def value_iterate(mdp, epsilon=1e-6, max_sweeps=1000):
    """Iterate the Bellman backup to convergence."""
    # TODO: sweep until the largest change is below epsilon
    return None


# %%
@checker("Exercise 11.4 — value_iterate")
def check_ex4():
    got = value_iterate(world)
    yield "covers every state", sorted(got or {}), sorted(world.states())
    yield ("matches csai.decision",
           all(abs(got[s] - values[s]) < 1e-3 for s in world.states())
           if got else None, True)
    yield ("the goal is worth 1", round((got or {}).get((0, 3), 0), 6), 1.0)
    yield ("the pit is worth -1", round((got or {}).get((1, 3), 0), 6), -1.0)
    yield ("states nearer the goal are worth more",
           (got or {})[(0, 2)] > (got or {})[(2, 0)], True)

    free = GridWorld(step_reward=0.0, discount=0.9)
    yield ("with free steps every state is still positive",
           all(v > 0 for s, v in value_iterate(free).items()
               if s != (1, 3)), True)


check_ex4()

# %% [markdown]
# ### Exercise 5 — extract the policy
#
# Write `greedy_policy(mdp, values)` returning `{state: action}`, with `None`
# for terminal states. For each state pick the action with the highest expected
# next value.

# %%
def greedy_policy(mdp, values):
    """The action in each state that maximises expected next value."""
    # TODO: argmax over actions of Σ P(s'|s,a) · V(s')
    return None


# %%
@checker("Exercise 11.5 — greedy_policy")
def check_ex5():
    got = greedy_policy(world, values)
    yield "terminal states have no action", (got or {}).get((0, 3)), None
    yield "…including the pit", (got or {}).get((1, 3)), None
    yield "matches csai.decision", got, policy
    yield ("from the start, head up or right",
           (got or {}).get((2, 0)) in {"up", "right"}, True)
    yield ("beside the pit, go the long way round",
           (got or {}).get((2, 3)), "left")

    risky = GridWorld(step_reward=-2.0, discount=0.9)
    risky_policy = greedy_policy(risky, dec.value_iteration(risky))
    yield ("when steps are expensive, take the shortcut past the pit",
           risky_policy[(2, 3)], "up")


check_ex5()

# %% [markdown]
# ### Exercise 6 — the stopping rule
#
# Write `stop_at(quality_fn, difficulty, cost, max_effort)` returning the
# effort in `0 … max_effort` maximising `quality − cost × effort`. Break ties
# by the **smallest** effort — never pay for thinking that buys nothing.

# %%
def stop_at(quality_fn, difficulty, cost, max_effort):
    """The effort maximising quality minus cost; ties go to less thinking."""
    # TODO: evaluate the net value at each effort and take the best
    return None


# %%
@checker("Exercise 11.6 — stop_at")
def check_ex6():
    yield ("free thinking runs to the limit",
           stop_at(quality, 3, 0.0, 20), 20)
    yield ("…and ties break toward less effort",
           stop_at(lambda d, e: 1.0, 3, 0.0, 20), 0)
    yield ("prohibitive cost means do not think at all",
           stop_at(quality, 3, 1.0, 20), 0)
    for difficulty in (1, 3, 6):
        yield (f"agrees with csai.decision (difficulty {difficulty})",
               stop_at(quality, difficulty, 0.01, 25),
               PerformanceProfile(quality, 0.01).stopping_point(difficulty, 25))
    yield ("harder problems justify more thinking",
           stop_at(quality, 6, 0.01, 25) >= stop_at(quality, 1, 0.01, 25), True)
    yield ("…and a higher cost justifies less",
           stop_at(quality, 3, 0.05, 25) <= stop_at(quality, 3, 0.005, 25), True)


check_ex6()

# %% [markdown]
# ---
# ## Project — a thinking-budget controller
#
# Reproduce §5's finding yourself, and then push on it.
#
# ```python
# allocate(quality_fn, difficulties, budget, cap=25) -> list[int]
# ```
#
# Greedy marginal allocation: hand each unit of budget to whichever problem
# gains the most from it, never exceeding `cap` on any one problem, and stop
# early if no unit would gain anything.
#
# ```python
# simulate(quality_fn, difficulties, allocation, rng) -> float
# ```
#
# Fraction solved: problem `i` succeeds with probability
# `quality_fn(difficulty, effort)`.
#
# ```python
# simulate_verified(quality_fn, difficulties, budget, rng, cap=25) -> dict
# ```
#
# Spend one unit at a time on the live problem with the best marginal gain;
# after each unit, a **perfect verifier** tells you whether that problem is now
# solved, and if so you stop spending on it. Return
# `{"solved": fraction, "spent": units used, "wasted": units spent on problems
# that were already solved}` — which is zero for this strategy, and is the
# point of comparison.
#
# ```python
# compare(quality_fn, difficulties, budgets, trials=100) -> {budget: {strategy: mean solved}}
# ```
#
# for strategies `"uniform"`, `"adaptive"` and `"verified"`.
#
# **Write-up questions:**
#
# 1. Make the verifier imperfect: it misses a correct answer 20% of the time
#    (so you keep spending), and it accepts a wrong answer 5% of the time.
#    Re-run. Which error hurts more, and why is the asymmetry what it is?
# 2. The adaptive allocator abandons hard problems at small budgets and
#    favours them at large ones. At which budget does the switch happen for
#    these difficulty curves, and what feature of `quality` decides it?
# 3. A verifier costs something to run. Extend `simulate_verified` to charge
#    `c` units per check. At what `c` does the verifier-driven strategy stop
#    beating plain adaptive allocation? Compare that number to how much a
#    single unit of thinking is worth.

# %%
def allocate(quality_fn, difficulties, budget, cap=25):
    """Greedy marginal allocation of `budget` units across the problems."""
    # TODO: repeatedly give one unit to the largest marginal gain
    return None


def simulate(quality_fn, difficulties, allocation, rng):
    """Fraction solved when each problem gets one attempt at its allotted effort."""
    # TODO: rng.random() < quality_fn(difficulty, effort), averaged
    return None


def simulate_verified(quality_fn, difficulties, budget, rng, cap=25):
    """Spend one unit at a time, checking after each and stopping when solved."""
    # TODO: track solved/effort per problem; pick the live problem with the
    # best marginal gain; stop spending on a problem once it verifies
    return None


def compare(quality_fn, difficulties, budgets, trials=100):
    """Mean fraction solved by each strategy, at each budget."""
    # TODO: average over `trials` seeded runs
    return None


# %%
@checker("Project 11 — thinking-budget controller")
def check_project():
    ds = [1, 2, 3, 6] * 5

    a = allocate(quality, ds, 40)
    yield "allocates a list, one entry per problem", len(a or []), len(ds)
    yield "…spending the whole budget", sum(a or []), 40
    yield "…never negative", all(e >= 0 for e in (a or [1])), True
    yield "…and respecting the cap", max(allocate(quality, ds, 400, cap=5) or [9]), 5
    yield ("a zero budget allocates nothing",
           sum(allocate(quality, ds, 0) or [1]), 0)
    yield ("agrees with csai.decision",
           allocate(quality, ds, 40),
           dec.allocate_greedy(PerformanceProfile(quality), ds, 40, max_each=25))

    yield ("zero effort solves few problems",
           simulate(quality, ds, [0] * len(ds), random.Random(0)) < 0.35, True)
    yield ("huge effort solves nearly all",
           simulate(quality, ds, [25] * len(ds), random.Random(0)) > 0.95, True)

    v = simulate_verified(quality, ds, 60, random.Random(0))
    yield "the verified run reports three keys", sorted(v or {}), [
        "solved", "spent", "wasted"]
    yield "…spending no more than the budget", (v or {})["spent"] <= 60, True
    yield "…wasting nothing, by construction", (v or {})["wasted"], 0
    yield "…and solving a decent fraction", 0.0 < (v or {})["solved"] <= 1.0, True

    results = compare(quality, ds, [40, 80], trials=60)
    yield "compare covers both budgets", sorted(results or {}), [40, 80]
    yield ("…and all three strategies",
           sorted((results or {}).get(40, {})), ["adaptive", "uniform", "verified"])
    yield ("more budget solves more, for every strategy",
           all(results[80][s] >= results[40][s] for s in results[40]), True)
    yield ("adaptive is at least as good as uniform",
           results[80]["adaptive"] >= results[80]["uniform"] - 0.02, True)
    yield ("and the verifier is worth far more than the allocation policy",
           results[80]["verified"] - results[80]["adaptive"] >
           3 * abs(results[80]["adaptive"] - results[80]["uniform"]), True)


check_project()

# %%
# Your own version of the module's headline table.
if allocate(quality, [1, 2], 4) is not None:
    results = compare(quality, DIFFICULTIES, [40, 80, 160, 320], trials=120)
    print(table([[b] + [f"{results[b][s]:.3f}"
                        for s in ("uniform", "adaptive", "verified")]
                 for b in sorted(results)],
                ["budget", "uniform", "adaptive", "verifier-driven"],
                align="rrrr"))

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * J. von Neumann & O. Morgenstern, *Theory of Games and Economic Behavior*
#   (1944) — the axioms behind expected utility.
# * R. Bellman, *Dynamic Programming* (1957).
# * S. Russell & E. Wefald, *Do the Right Thing: Studies in Limited
#   Rationality* (1991) — metareasoning, and still the clearest statement.
# * E. Horvitz, "Reasoning about Beliefs and Actions under Computational
#   Resource Constraints" (1987) — anytime algorithms and their profiles.
# * S. Zilberstein, "Using Anytime Algorithms in Intelligent Systems" (1996).
# * S. Russell, "Rationality and Intelligence" (1997) — type-I and type-II.
# * C. Snell et al., "Scaling LLM Test-Time Compute Optimally Can Be More
#   Effective Than Scaling Model Parameters" (2024) — this module, measured on
#   language models.
#
# **Next:** the capstone. Everything the course has built, turned on the task
# it opened with — three ways of solving a cube-rotation problem, graded on
# held-out chain lengths, with a verifier in the loop.
