# %% [markdown]
# # Module 10 — Reasoning Under Uncertainty: Bayesian Networks
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Say why logic alone cannot represent most of what an agent knows, and
#    what probability adds.
# 2. Build a Bayesian network and read its **conditional independence**
#    claims off the graph.
# 3. Compute exact posteriors by enumeration and by **variable elimination**,
#    and measure the difference (a million terms against eighteen).
# 4. Implement **rejection sampling** and **likelihood weighting**, and say
#    when each breaks.
# 5. Demonstrate **explaining away**, and connect it to what a language model
#    does when it commits to the first plausible cause.
#
# **Prerequisites:** Module 2 (models and semantics — the contrast is the
# point). No probability theory assumed beyond "probabilities sum to 1".
#
# **Time:** ~80 minutes plus exercises.

# %% [markdown]
# ## 1. Why logic is not enough
#
# Module 2's knowledge base could say `rain → wet`. Try to write down what you
# actually know about a car starting:
#
# > The car starts if the battery is charged **and** there is fuel **and** the
# > starter motor works **and** the fuel line is not blocked **and** the
# > immobiliser is not engaged **and** nobody has stolen the engine **and**…
#
# This is the **qualification problem** (McCarthy, 1977): a rule that is *true*
# needs an unbounded list of exceptions, and a rule short enough to write down
# is *false*. Logic offers no middle. A conclusion is entailed or it is not.
#
# Probability provides the missing middle, and it is not a retreat from rigour.
# The axioms are three lines; Cox's theorem (1946) shows that any calculus of
# belief satisfying a few reasonable-looking constraints — consistency,
# comparability, respecting the logic where it applies — *is* probability
# theory up to isomorphism. If you want to reason under uncertainty at all, you
# get to choose the numbers, not the algebra.
#
# The move that makes it tractable is the same one every module has made: find
# the **structure** and exploit it.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import itertools
import random
import time

from csai import probability as prob
from csai.check import checker
from csai.probability import BayesNet, Variable, burglary_net, normalize
from csai.render import bar_chart, table

print("ready")

# %% [markdown]
# ## 2. The joint distribution, and why you cannot have one
#
# Everything about `n` variables is in their **joint distribution**: a number
# for every combination. With `n` binary variables that is `2ⁿ − 1` numbers to
# specify, and the table below is why nobody writes one down.

# %%
print(table([(n, f"{2 ** n - 1:,}") for n in (5, 10, 20, 30, 50)],
            ["binary variables", "numbers in the joint distribution"],
            align="rr"))

# %% [markdown]
# The saving structure is **conditional independence**. Knowing the alarm went
# off, whether John calls tells you nothing more about whether Mary does: they
# are conditionally independent *given* the alarm. Each is caused by the alarm,
# and nothing else links them.
#
# A **Bayesian network** encodes exactly these claims:
#
# * nodes are variables, and an edge `A → B` means "A is one of B's direct
#   causes";
# * each node carries `P(node | its parents)`;
# * the graph asserts that **a node is conditionally independent of its
#   non-descendants given its parents**.
#
# Then the joint factorises into one term per node:
#
# > `P(x₁ … xₙ) = ∏ P(xᵢ | parents(xᵢ))`
#
# Space exponential in the largest *family* rather than in `n`. Pearl's alarm
# network is the standard example: burglaries and earthquakes both set off an
# alarm; two neighbours may or may not call when they hear it.

# %%
net = burglary_net()
print("burglary → alarm ← earthquake,  alarm → john_calls,  alarm → mary_calls\n")
print(table([(v.name, ", ".join(v.parents) or "—", len(v.cpt) * (len(v.domain) - 1))
             for v in net.variables],
            ["variable", "parents", "numbers needed"], align="llr"))
full_joint = 2 ** len(net.variables) - 1
specified = sum(len(v.cpt) * (len(v.domain) - 1) for v in net.variables)
print(f"\nfull joint: {full_joint} numbers.  network: {specified}.")
print("The saving is small here and grows exponentially with the network.")

# %%
# The chain rule in action: one factor per variable.
world = {"burglary": True, "earthquake": False, "alarm": True,
         "john_calls": True, "mary_calls": False}
terms = [f"P({v.name}={world[v.name]}"
         + (f" | {', '.join(f'{p}={world[p]}' for p in v.parents)})" if v.parents else ")")
         + f" = {v.probability(world[v.name], world):.4g}"
         for v in net.variables]
print("\n".join(terms))
print(f"\nproduct = {net.joint(world):.3g}")

# %% [markdown]
# ## 3. Exact inference
#
# The question is always the same: **P(query | evidence)**. Two exact methods.
#
# ### Enumeration
#
# Sum the joint over every assignment to the unobserved variables, then
# normalise. Correct, and exponential in the number of hidden variables.
#
# ### Variable elimination
#
# The same sum, rearranged. Push each summation as far right as it will go, and
# reuse intermediate results instead of recomputing them. Formally: build a
# *factor* per variable, then eliminate hidden variables one at a time by
# multiplying together the factors that mention them and summing the variable
# out.
#
# It is dynamic programming applied to a sum of products — the same idea as
# memoising a recursion — and on a well-shaped network it turns an exponential
# into a linear.

# %%
evidence = {"john_calls": True, "mary_calls": True}

s1, s2 = {}, {}
exact = prob.enumeration_ask(net, "burglary", evidence, s1)
ve = prob.variable_elimination(net, "burglary", evidence, s2)

print(f"P(burglary | both neighbours called) = {exact[True]:.4f}")
print(f"enumeration:          {s1['terms']} terms")
print(f"variable elimination: {s2['products']} factor products")
print(f"same answer: {abs(exact[True] - ve[True]) < 1e-12}")

print("\nWorth pausing on that number. Both neighbours called, and the")
print("probability of a burglary is still only 28% — because burglaries are")
print("rare (0.1%) and alarms have other causes. Base rates are not a")
print("technicality; they dominate.")

# %%
# Where the difference actually matters: a chain of variables.
def chain_net(n):
    """x0 → x1 → … → x(n-1): the friendliest possible shape for elimination."""
    variables = [Variable("x0", (), {(): {True: 0.5, False: 0.5}})]
    for i in range(1, n):
        variables.append(Variable(f"x{i}", (f"x{i - 1}",), {
            (True,): {True: 0.8, False: 0.2},
            (False,): {True: 0.3, False: 0.7},
        }))
    return BayesNet(variables)


rows = []
for n in (4, 8, 12, 16, 20):
    chain = chain_net(n)
    ev = {f"x{n - 1}": True}
    a, b = {}, {}
    t0 = time.perf_counter()
    p1 = prob.enumeration_ask(chain, "x0", ev, a)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    p2 = prob.variable_elimination(chain, "x0", ev, b)
    t2 = time.perf_counter() - t0
    rows.append((n, f"{a['terms']:,}", f"{t1:.3f}", b.get("products", 0),
                 f"{t2:.3f}", abs(p1[True] - p2[True]) < 1e-9))
print(table(rows, ["variables", "enumeration terms", "sec", "VE products",
                   "sec", "agree"], align="rrrrrc"))

# %% [markdown]
# A million terms against eighteen factor products, for an identical answer.
#
# The honest caveat: elimination's cost depends on the **elimination order**,
# and finding the best one is NP-hard. The relevant quantity is the network's
# *treewidth*; on a chain it is 1 and everything is easy, on a densely
# connected network it is large and exact inference is hopeless. This is the
# same story as Module 3's SAT solvers and Module 8's propagation — structure
# decides, and worst cases stay hard.

# %% [markdown]
# ## 4. Explaining away
#
# Now the phenomenon that makes Bayesian networks worth the trouble, and that
# no simpler formalism reproduces.
#
# `burglary` and `earthquake` are **independent** — knowing there was an
# earthquake tells you nothing about burglars. But they share a child, `alarm`.
# Observe the alarm, and they become **dependent**: each is now a competing
# explanation, and confirming one lowers the probability of the other.
#
# This is called **explaining away**, and it is the signature of a "collider"
# — a node with two parents, conditioned on. It is also a place where informal
# reasoning routinely goes wrong.

# %%
rows = []
for label, ev in [("nothing known", {}),
                  ("the alarm is going off", {"alarm": True}),
                  ("…and there was an earthquake",
                   {"alarm": True, "earthquake": True}),
                  ("…and there was no earthquake",
                   {"alarm": True, "earthquake": False})]:
    rows.append((label, f"{prob.enumeration_ask(net, 'burglary', ev)[True]:.4f}"))
print(table(rows, ["what you know", "P(burglary)"], align="lr"))
print("\nThe earthquake explains the alarm, so the burglary is no longer")
print("needed to explain it: 37% collapses to 0.3%. Nothing about the")
print("burglary itself changed — only the competition for an explanation.")

# %% [markdown]
# ## 5. Approximate inference
#
# When exact inference is out of reach, sample.
#
# **Rejection sampling.** Sample the whole network from the top; throw away
# any sample that contradicts the evidence; count what is left. Unbiased and
# embarrassingly wasteful — if the evidence has probability 0.001, you keep one
# sample in a thousand.
#
# **Likelihood weighting.** Fix the evidence variables to their observed
# values, sample only the rest, and weight each sample by how likely the
# evidence was given what you sampled. Nothing is thrown away. The catch moves
# rather than disappearing: with unlikely evidence the weights become extremely
# skewed and a handful of samples carry all the mass.

# %%
rows = []
for n in (200, 1000, 5000, 20000):
    rj_stats = {}
    rj = prob.rejection_sampling(net, "burglary", evidence, n,
                                 random.Random(1), rj_stats)
    lw = prob.likelihood_weighting(net, "burglary", evidence, n,
                                   random.Random(1))
    rows.append((n, f"{rj[True]:.4f}", rj_stats["kept"],
                 f"{prob.max_error(exact, rj):.4f}",
                 f"{lw[True]:.4f}", f"{prob.max_error(exact, lw):.4f}"))
print(f"exact answer: {exact[True]:.4f}\n")
print(table(rows, ["samples", "rejection est.", "samples kept", "error",
                   "weighted est.", "error"], align="rrrrrr"))

# %% [markdown]
# Look at the "samples kept" column. Both neighbours calling is a rare event,
# so rejection sampling discards well over 99% of its work and its estimate is
# built from a few dozen samples no matter how many you draw. Likelihood
# weighting uses every one.
#
# Hold that thought — the next section is about a technique that is rejection
# sampling wearing different clothes.

# %% [markdown]
# ## 6. Bridge to language models
#
# **A language model is a probability distribution.** Not a knowledge base,
# not a function — a conditional distribution over continuations. Everything
# in this module applies to it directly, and three connections matter.
#
# ### Self-consistency is Monte-Carlo marginalisation
#
# Sampling `k` chains of thought and taking the majority answer (Wang et al.,
# 2022) is estimating
#
# > `P(answer | question) = Σ over reasoning paths P(answer, path | question)`
#
# by sampling paths and marginalising the path away. It is the classic
# treatment for a nuisance variable you do not care about, and it comes with
# the classic diminishing returns: Monte-Carlo error falls as `1/√k`, so
# quadrupling the samples halves the error. That is the whole shape of the
# published self-consistency curves.
#
# It also inherits rejection sampling's weakness. If the correct reasoning path
# is rare under the model, sampling more paths finds it slowly, and a majority
# vote actively *suppresses* it. Nothing about scale changes that — it is a
# property of estimating a rare event by sampling. The fix is the same as here:
# stop sampling blindly and use the structure, which is what verifiers, tree
# search (Module 7) and constrained decoding (Module 8) are all doing.
#
# ### Calibration is a real, measurable property
#
# A model is **calibrated** when the things it says with 80% confidence are
# true 80% of the time. This is checkable without knowing anything about how
# the model works — bucket predictions by stated confidence and compare. Base
# models are often decently calibrated; heavy fine-tuning tends to damage it,
# usually toward overconfidence. Note that calibration and accuracy are
# independent: a model that always says "70%" and is right 70% of the time is
# perfectly calibrated and not very useful.
#
# ### Explaining away is what over-commitment looks like
#
# The alarm goes off; the model says "burglary" and writes three paragraphs
# about it. Then the earthquake is mentioned, and a Bayesian reasoner drops
# the burglary from 37% to 0.3% — while a left-to-right generator has already
# committed, and the text it has produced makes the burglary *more* likely to
# be repeated, not less.
#
# That is the deep version of the point Modules 6 and 8 made about control
# flow. Belief revision requires holding hypotheses in superposition and
# reweighting them as evidence arrives. Sequential generation collapses to one
# hypothesis early, and every token after that is conditioned on the collapse.
# Chain of thought helps a little, because writing the alternatives down keeps
# them in the context. Explicitly enumerating hypotheses and scoring each —
# which is what this module's machinery does — helps much more.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — normalise
#
# Write `normalise(distribution)`: scale a dict of non-negative weights so
# they sum to 1. If they are all zero, return a uniform distribution.

# %%
def normalise(distribution):
    """Scale non-negative weights to sum to 1; uniform if all are zero."""
    # TODO: divide by the total, or spread evenly when the total is zero
    return None


# %%
@checker("Exercise 10.1 — normalise")
def check_ex1():
    yield "already normalised", normalise({True: 0.25, False: 0.75}), {
        True: 0.25, False: 0.75}
    yield "counts become probabilities", normalise({"a": 3, "b": 1}), {
        "a": 0.75, "b": 0.25}
    yield "all zero -> uniform", normalise({"a": 0, "b": 0}), {"a": 0.5, "b": 0.5}
    yield "a single outcome", normalise({"only": 7}), {"only": 1.0}
    yield "sums to one", round(sum(normalise({"a": 1, "b": 2, "c": 5}).values()), 12), 1.0
    yield ("agrees with csai.probability",
           normalise({"a": 2, "b": 6}), normalize({"a": 2, "b": 6}))


check_ex1()

# %% [markdown]
# ### Exercise 2 — the chain rule
#
# Write `joint_probability(net, assignment)`: the probability of a **complete**
# assignment, as the product of `P(variable | its parents)` over every
# variable. Use `net.variables` and `variable.probability(value, assignment)`.

# %%
def joint_probability(net, assignment):
    """P(assignment) as a product of one conditional per variable."""
    # TODO: multiply variable.probability(assignment[name], assignment)
    return None


# %%
@checker("Exercise 10.2 — joint_probability")
def check_ex2():
    w = {"burglary": True, "earthquake": False, "alarm": True,
         "john_calls": True, "mary_calls": True}
    yield ("the textbook example",
           round(joint_probability(net, w), 12), round(net.joint(w), 12))
    quiet = {k: False for k in net.names}
    yield ("nothing happening is the likeliest world",
           joint_probability(net, quiet) > 0.9, True)
    yield ("…and everything happening at once is not",
           joint_probability(net, {k: True for k in net.names}) < 1e-5, True)
    total = sum(joint_probability(net, dict(zip(net.names, combo)))
                for combo in itertools.product([True, False], repeat=5))
    yield "the joint sums to 1 over all 32 worlds", round(total, 10), 1.0


check_ex2()

# %% [markdown]
# ### Exercise 3 — inference by enumeration
#
# Write `query(net, variable, evidence)` returning `P(variable | evidence)` as
# a normalised dict. Enumerate every assignment to the variables that are
# neither the query nor evidence, summing the joint.
#
# <details><summary>Hint</summary>
#
# The hidden variables are `net.names` minus the query minus the evidence
# keys. For each value of the query variable, sum `joint_probability` over
# every combination of hidden values, with the query and evidence fixed. Then
# normalise. `itertools.product(*[net.domain(h) for h in hidden])` gives the
# combinations.
# </details>

# %%
def query(net, variable, evidence):
    """P(variable | evidence) by summing the joint over hidden variables."""
    # TODO: for each value, sum the joint over all completions; normalise
    return None


# %%
@checker("Exercise 10.3 — query")
def check_ex3():
    got = query(net, "burglary", {"john_calls": True, "mary_calls": True})
    yield ("the classic answer",
           round((got or {})[True], 6), round(exact[True], 6))
    yield "sums to 1", round(sum((got or {}).values()), 10), 1.0

    prior = query(net, "burglary", {})
    yield "no evidence recovers the prior", round((prior or {})[True], 6), 0.001

    yield ("a single neighbour calling is weaker evidence than two",
           query(net, "burglary", {"john_calls": True})[True] <
           query(net, "burglary", {"john_calls": True, "mary_calls": True})[True],
           True)

    yield ("explaining away, measured",
           round(query(net, "burglary",
                       {"alarm": True, "earthquake": True})[True], 4),
           round(prob.enumeration_ask(
               net, "burglary", {"alarm": True, "earthquake": True})[True], 4))
    yield ("…is a large drop from the alarm alone",
           query(net, "burglary", {"alarm": True})[True] >
           20 * query(net, "burglary", {"alarm": True, "earthquake": True})[True],
           True)


check_ex3()

# %% [markdown]
# ### Exercise 4 — summing a variable out of a factor
#
# The heart of variable elimination. Write
# `marginalise(factor, name, net)`: return a new `Factor` over
# `factor.variables` without `name`, each entry the sum over that variable's
# values.

# %%
def marginalise(factor, name, net):
    """A new Factor with `name` summed out."""
    # TODO: build the reduced scope, then total over the removed variable
    return None


# %%
@checker("Exercise 10.4 — marginalise")
def check_ex4():
    f = prob.Factor(("burglary", "earthquake"), {
        (True, True): 0.1, (True, False): 0.2,
        (False, True): 0.3, (False, False): 0.4,
    })
    got = marginalise(f, "earthquake", net)
    yield "the scope shrinks", (got.variables if got else None), ("burglary",)
    yield "…summing over the removed variable", (
        round(got.table[(True,)], 10) if got else None), 0.3
    yield "…for every remaining value", (
        round(got.table[(False,)], 10) if got else None), 0.7

    both = marginalise(marginalise(f, "earthquake", net), "burglary", net)
    yield "removing everything leaves a number", (
        both.variables if both else None), ()
    yield "…which is the total", round(both.table[()], 10) if both else None, 1.0

    yield ("a variable that is not there changes nothing",
           (marginalise(f, "alarm", net).table if marginalise(f, "alarm", net)
            else None), f.table)
    yield ("agrees with csai.probability",
           marginalise(f, "burglary", net).table,
           prob.sum_out(f, "burglary", net).table)


check_ex4()

# %% [markdown]
# ### Exercise 5 — sample the network
#
# Write `sample_once(net, rng)` returning one complete assignment drawn from
# the joint. Go through `net.variables` in order — they are topologically
# sorted, so every parent is already assigned — and use
# `variable.sample(assignment, rng)`.

# %%
def sample_once(net, rng):
    """One assignment drawn from the network's joint distribution."""
    # TODO: parents first, then each variable conditioned on them
    return None


# %%
@checker("Exercise 10.5 — sample_once")
def check_ex5():
    rng = random.Random(0)
    one = sample_once(net, rng)
    yield "assigns every variable", sorted(one or {}), sorted(net.names)
    yield "with values from the domains", all(
        v in (True, False) for v in (one or {}).values()), True

    rng = random.Random(4)
    samples = [sample_once(net, rng) for _ in range(4000)]
    rate = sum(s["burglary"] for s in samples) / len(samples)
    yield "burglaries are rare, as specified", rate < 0.02, True
    alarms = [s for s in samples if s["alarm"]]
    yield ("alarms happen sometimes", 0 < len(alarms) < len(samples) / 2, True)
    if alarms:
        yield ("…and John usually calls when one does",
               sum(s["john_calls"] for s in alarms) / len(alarms) > 0.6, True)
    quiet = [s for s in samples if not s["alarm"]]
    yield ("…and rarely when one does not",
           sum(s["john_calls"] for s in quiet) / len(quiet) < 0.15, True)


check_ex5()

# %% [markdown]
# ### Exercise 6 — a weighted sample
#
# Write `weighted_sample(net, evidence, rng)` returning
# `(assignment, weight)`: evidence variables are **fixed, not sampled**, and
# the weight is the product of `P(evidence variable | its parents)` over the
# evidence.

# %%
def weighted_sample(net, evidence, rng):
    """(assignment, weight) with the evidence fixed rather than sampled."""
    # TODO: walk the variables in order; fix evidence and multiply its
    # probability into the weight; sample everything else
    return None


# %%
@checker("Exercise 10.6 — weighted_sample")
def check_ex6():
    rng = random.Random(0)
    assignment, weight = weighted_sample(net, {"john_calls": True}, rng)
    yield "the evidence is respected", assignment["john_calls"], True
    yield "everything is assigned", sorted(assignment), sorted(net.names)
    yield "the weight is a probability", 0.0 < weight <= 1.0, True
    yield ("…equal to P(john_calls=True | alarm) in that sample",
           round(weight, 10),
           round(net.by_name["john_calls"].probability(True, assignment), 10))

    yield ("no evidence means weight 1",
           weighted_sample(net, {}, random.Random(1))[1], 1.0)

    _, w2 = weighted_sample(net, {"john_calls": True, "mary_calls": True},
                            random.Random(2))
    yield "two pieces of evidence multiply", w2 < 1.0, True

    rng = random.Random(9)
    ev = {"john_calls": True, "mary_calls": True}
    total = {True: 0.0, False: 0.0}
    for _ in range(6000):
        a, w = weighted_sample(net, ev, rng)
        total[a["burglary"]] += w
    estimate = normalize(total)
    # A loose tolerance on purpose: with evidence this unlikely the weights
    # are badly skewed and the estimator has high variance, which is exactly
    # the weakness §5 measured. Averaging over seeds would tighten it.
    yield ("averaging weighted samples lands in the right region",
           abs(estimate[True] - exact[True]) < 0.15, True)


check_ex6()

# %% [markdown]
# ---
# ## Project — a diagnosis network, and which estimator to trust
#
# **Part 1 — build the network.** Write `build_ci_net()` returning a
# `BayesNet` for a question every developer asks: *the build is red — is it my
# code?*
#
# | variable | parents | table |
# |---|---|---|
# | `bug` | — | P(True) = 0.20 |
# | `flaky_infra` | — | P(True) = 0.10 |
# | `ci_red` | `bug`, `flaky_infra` | T,T → 0.99; T,F → 0.95; F,T → 0.80; F,F → 0.02 |
# | `local_pass` | `bug` | T → 0.30; F → 0.97 |
# | `review_comment` | `bug` | T → 0.60; F → 0.15 |
#
# Variables must be in topological order, and all are boolean.
#
# **Part 2 — one inference interface.**
#
# ```python
# infer(net, variable, evidence, method="exact", samples=5000, rng=None) -> dict
# ```
#
# with `method` one of `"exact"`, `"elimination"`, `"rejection"`,
# `"weighting"`.
#
# **Part 3 — measure the estimators.**
#
# ```python
# convergence(net, variable, evidence, sizes, rng_seed=0) -> {method: {n: error}}
# ```
#
# where the error is the largest absolute difference from the exact posterior,
# for `"rejection"` and `"weighting"` at each sample size.
#
# **Write-up questions:**
#
# 1. CI is red and the tests pass locally. What is P(bug)? Now add that infra
#    was flaky. How far does it drop, and which structural feature of the
#    network is responsible?
# 2. Take evidence rare enough that rejection sampling keeps under 5% of its
#    samples. Compare the two estimators' error curves. At what sample size
#    does rejection sampling reach the error likelihood weighting had at
#    n = 200?
# 3. Sampling error falls as `1/√n`. Given that, what does it cost to halve
#    the error of a self-consistency vote — and what would you do instead?

# %%
def build_ci_net():
    """The CI diagnosis network from the table above."""
    # TODO: five boolean Variables, parents before children
    return None


def infer(net, variable, evidence, method="exact", samples=5000, rng=None):
    """P(variable | evidence) by the named method."""
    # TODO: dispatch to enumeration / elimination / rejection / weighting
    return None


def convergence(net, variable, evidence, sizes, rng_seed=0):
    """{method: {n: max error against the exact posterior}} for the samplers."""
    # TODO: compute the exact posterior once, then measure each sampler
    return None


# %%
@checker("Project 10 — diagnosis network")
def check_project():
    ci = build_ci_net()
    yield "returns a BayesNet", isinstance(ci, BayesNet), True
    yield "with five variables", sorted(ci.names) if ci else None, [
        "bug", "ci_red", "flaky_infra", "local_pass", "review_comment"]
    yield ("in topological order",
           ci.names.index("ci_red") > ci.names.index("bug") if ci else None, True)

    yield ("the prior on a bug",
           round(infer(ci, "bug", {})[True], 6), 0.2)
    yield ("a red build raises it",
           round(infer(ci, "bug", {"ci_red": True})[True], 4),
           round(prob.enumeration_ask(ci, "bug", {"ci_red": True})[True], 4))
    yield ("…and a red build is much stronger evidence than none",
           infer(ci, "bug", {"ci_red": True})[True] > 0.5, True)
    yield ("passing locally pulls it back down",
           infer(ci, "bug", {"ci_red": True, "local_pass": True})[True] <
           infer(ci, "bug", {"ci_red": True})[True], True)
    yield ("and known-flaky infra explains the red build away",
           infer(ci, "bug", {"ci_red": True, "flaky_infra": True})[True] <
           infer(ci, "bug", {"ci_red": True})[True], True)

    yield ("elimination agrees with enumeration",
           round(infer(ci, "bug", {"ci_red": True}, "elimination")[True], 9),
           round(infer(ci, "bug", {"ci_red": True}, "exact")[True], 9))

    ev = {"ci_red": True, "local_pass": True}
    truth = infer(ci, "bug", ev)
    for method in ("rejection", "weighting"):
        est = infer(ci, "bug", ev, method, samples=20000, rng=random.Random(5))
        yield (f"{method} sampling lands near the exact answer",
               abs(est[True] - truth[True]) < 0.05, True)
        yield (f"…and {method} returns a distribution",
               round(sum(est.values()), 9), 1.0)

    curves = convergence(ci, "bug", ev, [200, 2000], rng_seed=1)
    yield "convergence covers both samplers", sorted(curves or {}), [
        "rejection", "weighting"]
    yield ("…at both sample sizes",
           sorted((curves or {}).get("weighting", {})), [200, 2000])
    yield ("…and more samples means less error, for weighting",
           curves["weighting"][2000] <= curves["weighting"][200] + 0.02, True)
    yield ("…with every error a non-negative number",
           all(e >= 0 for m in (curves or {}).values() for e in m.values()), True)


check_project()

# %%
# The report your write-up discusses.
if build_ci_net() is not None:
    ci = build_ci_net()
    print("P(bug | …)\n")
    rows = []
    for label, ev in [("nothing known", {}),
                      ("CI is red", {"ci_red": True}),
                      ("CI red, tests pass locally",
                       {"ci_red": True, "local_pass": True}),
                      ("CI red, and the infra is flaky",
                       {"ci_red": True, "flaky_infra": True}),
                      ("CI red, passes locally, infra flaky",
                       {"ci_red": True, "local_pass": True, "flaky_infra": True}),
                      ("CI red, and a reviewer flagged something",
                       {"ci_red": True, "review_comment": True})]:
        rows.append((label, f"{infer(ci, 'bug', ev)[True]:.3f}"))
    print(table(rows, ["evidence", "P(bug)"], align="lr"))

    ev = {"ci_red": True, "local_pass": True, "review_comment": False}
    curves = convergence(ci, "bug", ev, [100, 500, 2000, 10000], rng_seed=2)
    print("\nestimator error against the exact posterior:")
    for method, curve in curves.items():
        print(bar_chart(curve.items(), width=30, maximum=0.3,
                        title=f"  {method}", value_fmt="{:.4f}"))

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * J. Pearl, *Probabilistic Reasoning in Intelligent Systems* (1988) — the
#   book that created the field. The alarm network is from it.
# * R. T. Cox, "Probability, Frequency and Reasonable Expectation" (1946) —
#   why probability is the calculus of belief, not merely one option.
# * D. Koller & N. Friedman, *Probabilistic Graphical Models* (2009) — the
#   comprehensive reference.
# * S. Russell & P. Norvig, *AIMA* ch. 12–13.
# * X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"
#   (2022) — marginalising over reasoning paths, whether or not it says so.
# * S. Kadavath et al., "Language Models (Mostly) Know What They Know" (2022)
#   — calibration, measured.
#
# **Next:** Module 11 asks the question this module set up. If thinking costs
# something and buys something, how much should you do — expected utility,
# value of information, and a controller that decides how long to think.
