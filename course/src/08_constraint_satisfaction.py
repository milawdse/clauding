# %% [markdown]
# # Module 8 — Constraint Satisfaction: Inference Inside the Search
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Model a problem as variables, domains and constraints, and recognise when
#    that framing is the right one.
# 2. Implement backtracking search over a CSP.
# 3. Implement **forward checking** and **AC-3**, and explain what "arc
#    consistent" guarantees and what it does not.
# 4. Apply the MRV, degree and least-constraining-value heuristics, and
#    measure what each is worth.
# 5. Solve the Zebra puzzle from its clues, and explain why left-to-right
#    generation is the worst possible strategy for problems shaped like it.
#
# **Prerequisites:** Module 3 (clauses and propagation — AC-3 will feel
# familiar) and Module 6 (search).
#
# **Time:** ~75 minutes plus exercises.

# %% [markdown]
# ## 1. When the path does not matter
#
# Module 6 searched for a *route*: the sequence of actions was the answer. A
# constraint satisfaction problem is different — only the final assignment
# matters, and how you arrived at it is irrelevant.
#
# That sounds like a restriction. It is an opportunity, because it exposes
# structure a general search cannot see:
#
# * Every state has the **same shape** — a partial assignment — so you can ask
#   generic questions like "which variable is most constrained?" without
#   knowing anything about the domain.
# * Constraints let you **prune before you search**. If assigning `A = red`
#   makes `B` impossible, you can find that out immediately rather than
#   discovering it fifty levels down.
#
# The second point is the whole module. In Module 6, inference happened
# *around* the search — a heuristic scored states from outside. Here inference
# happens **inside** it: every assignment triggers propagation that shrinks the
# remaining problem, often to nothing.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import time
from collections import deque

from csai import csp as csp_lib
from csai.check import checker
from csai.csp import CSP, all_different, from_constraints
from csai.render import table

print("ready")

# %% [markdown]
# ## 2. The formalism
#
# A CSP is three things:
#
# * **Variables** `X₁ … Xₙ`
# * **Domains** `D₁ … Dₙ` — the values each variable may take
# * **Constraints** — relations restricting combinations of variables
#
# A **solution** assigns every variable a value from its domain, violating no
# constraint.
#
# Constraints come in arities. *Unary* ones ("this house is number 3") just
# shrink a domain. *Binary* ones relate two variables, and are the case the
# classical algorithms are written for. *Global* ones like `alldifferent`
# relate many, and although they decompose into pairwise `≠`, specialised
# propagators for them are much stronger — that is a large part of what a
# modern constraint solver sells.
#
# The classic first example is colouring the states of Australia so that no
# two neighbours share a colour.

# %%
AUSTRALIA_NEIGHBOURS = {
    "WA": {"NT", "SA"},
    "NT": {"WA", "SA", "Q"},
    "SA": {"WA", "NT", "Q", "NSW", "V"},
    "Q": {"NT", "SA", "NSW"},
    "NSW": {"Q", "SA", "V"},
    "V": {"SA", "NSW", "T"},
    "T": {"V"},
}
COLOURS = ["red", "green", "blue"]

australia = CSP(
    variables=list(AUSTRALIA_NEIGHBOURS),
    domains={s: list(COLOURS) for s in AUSTRALIA_NEIGHBOURS},
    neighbours=AUSTRALIA_NEIGHBOURS,
    constraint=lambda a, va, b, vb: va != vb,
)

solution = csp_lib.solve(australia)
print(table(sorted(solution.items()), ["state", "colour"]))
print("\nvalid:", australia.is_solution(solution))
print("number of solutions:", len(csp_lib.solve(australia, all_solutions=True)))

# %% [markdown]
# ## 3. Backtracking
#
# The naive approach — generate every complete assignment and test it — costs
# `d^n`. For Australia that is 3⁷ = 2,187, which is fine, and for anything real
# it is not.
#
# **Backtracking** is the fix, and it is barely more complicated: assign one
# variable at a time, check consistency with what is already assigned, and as
# soon as something conflicts, undo and try the next value. The saving is that
# an inconsistent *partial* assignment is abandoned immediately, taking every
# completion of it with it.
#
# Two decisions are left open, and both matter enormously:
#
# * **which variable to assign next** — any order is correct, and the orders
#   differ by orders of magnitude in cost;
# * **how much to infer after each assignment.**

# %%
def queens(n):
    """N-queens: one variable per column, its value the row of that queen."""
    columns = list(range(n))
    return CSP(
        variables=columns,
        domains={c: list(range(n)) for c in columns},
        neighbours={c: {o for o in columns if o != c} for c in columns},
        constraint=lambda a, ra, b, rb: ra != rb and abs(a - b) != abs(ra - rb),
    )


stats = {}
solution = csp_lib.solve(queens(8), inference="none", select="first",
                         least_constraining=False, stats=stats)
print("8 queens, plain backtracking:", stats)
print("rows by column:", [solution[c] for c in range(8)])

# %% [markdown]
# ## 4. Inference during search
#
# Now the interesting part. After assigning `X = v`, what can be deduced
# *before* choosing the next variable?
#
# ### Forward checking
#
# For each unassigned neighbour of `X`, delete the values now incompatible
# with `v`. If any domain empties, this assignment is already dead — backtrack
# now instead of discovering it several levels deeper.
#
# ### Arc consistency (AC-3)
#
# Forward checking only looks one step out. Arc consistency propagates
# *transitively*.
#
# An arc `X → Y` is consistent when **every** value of `X` has some compatible
# value in `Y`. A value with no support is impossible, so delete it. And
# deleting it may destroy the support of some value in a *third* variable, so
# every arc pointing into `X` goes back on the queue. Repeat until nothing
# changes.
#
# AC-3 (Mackworth, 1977) runs in `O(e·d³)`. What it guarantees is real but
# limited: it removes every value with no *pairwise* support. It does **not**
# guarantee a solution exists — three mutually-adjacent variables with two
# colours are arc consistent and unsatisfiable. Arc consistency prunes; it does
# not decide.
#
# Notice this is Module 3's unit propagation, generalised. A unit clause is a
# domain reduced to one value; propagating it prunes the neighbours. Same idea,
# richer domains — and in both cases it is propagation, not search, that does
# the work.

# %%
rows = []
for n in (8, 12, 16):
    for inference, select, lcv in [("none", "first", False),
                                   ("forward", "first", False),
                                   ("forward", "mrv", False),
                                   ("forward", "mrv", True),
                                   ("ac3", "mrv", True)]:
        st = {}
        t0 = time.perf_counter()
        found = csp_lib.solve(queens(n), inference=inference, select=select,
                              least_constraining=lcv, stats=st)
        label = f"{inference}, {select}" + (", LCV" if lcv else "")
        rows.append((n, label, st["nodes"], st["backtracks"],
                     f"{time.perf_counter() - t0:.3f}", "yes" if found else "no"))
print(table(rows, ["n", "configuration", "nodes", "backtracks", "seconds",
                   "solved"], align="rlrrrc"))

# %% [markdown]
# At `n = 16`, plain backtracking explores over 160,000 nodes and forward
# checking with MRV explores a few dozen. Three or four orders of magnitude,
# and the problem specification never changed.
#
# Two things in that table are worth more than the headline.
#
# **The techniques do not simply stack.** At `n = 16`, adding LCV to forward
# checking with MRV makes it *worse*, and so does adding AC-3. Each is a bet:
# spend effort now to prune later. When MRV has already collapsed the search
# to a near-straight line, there is nothing left for the extra machinery to
# prune and you have only paid for it. Anyone claiming a solver technique is
# universally good has not measured enough problems.
#
# **Compare nodes and seconds together.** AC-3 explores among the fewest nodes
# and is rarely the fastest, because each node costs a full propagation pass.
# That trade — cheaper nodes versus fewer of them — is the central engineering
# question in every solver, and the answer is problem-dependent.
#
# ### The ordering heuristics
#
# | heuristic | rule | intuition |
# |---|---|---|
# | **MRV** (minimum remaining values) | assign the variable with the fewest legal values | fail fast: if a variable is doomed, find out now, not after a thousand wasted assignments |
# | **degree** | tie-break on the most constraints with unassigned variables | reduce the branching factor for everything that follows |
# | **LCV** (least constraining value) | try the value that rules out fewest options for the neighbours | fail *last*: you only need one solution, so keep your options open |
#
# MRV and LCV pull in opposite directions on purpose, and both are right.
# Choosing variables, you want to hit the wall early — most of the search tree
# is doomed and the sooner you know, the less of it you visit. Choosing values,
# you want to stay flexible, because you only need one that works.

# %% [markdown]
# ## 5. The Zebra puzzle
#
# Five houses in a row, each with a nationality, colour, pet, drink and brand
# of cigarette. Fifteen clues. Who drinks water, and who owns the zebra?
#
# Published in *Life International* in 1962, and the archetype of a puzzle that
# is trivial for constraint propagation and miserable for anything that has to
# guess its way forward in a fixed order. Encoding it takes one idea:
#
# > **A variable for each attribute value, whose value is the house number.**
#
# So `red` is a variable ranging over 1–5, and `red = 3` means the red house is
# the third. Then `alldifferent` within each category, and every clue becomes a
# small binary relation.

# %%
CATEGORIES = {
    "nationality": ["englishman", "spaniard", "ukrainian", "norwegian", "japanese"],
    "colour": ["red", "green", "ivory", "yellow", "blue"],
    "pet": ["dog", "snails", "fox", "horse", "zebra"],
    "drink": ["coffee", "tea", "milk", "orange_juice", "water"],
    "cigarette": ["old_gold", "kools", "chesterfields", "lucky_strike", "parliaments"],
}

ZEBRA_VARIABLES = [v for group in CATEGORIES.values() for v in group]

same_house = lambda x, y: x == y
immediately_right = lambda x, y: x == y + 1
next_door = lambda x, y: abs(x - y) == 1

ZEBRA_CLUES = [
    ("englishman", "red", same_house),           # 2. lives in the red house
    ("spaniard", "dog", same_house),             # 3. owns the dog
    ("coffee", "green", same_house),             # 4. coffee in the green house
    ("ukrainian", "tea", same_house),            # 5. drinks tea
    ("green", "ivory", immediately_right),       # 6. green is right of ivory
    ("old_gold", "snails", same_house),          # 7. Old Gold owns snails
    ("kools", "yellow", same_house),             # 8. Kools in the yellow house
    ("chesterfields", "fox", next_door),         # 11. next to the fox
    ("kools", "horse", next_door),               # 12. next to the horse
    ("lucky_strike", "orange_juice", same_house),  # 13.
    ("japanese", "parliaments", same_house),     # 14.
    ("norwegian", "blue", next_door),            # 15.
]
# Clues 9 and 10 are unary — milk in the middle, the Norwegian first — so they
# are domain restrictions rather than constraints.
ZEBRA_DOMAINS = {v: [1, 2, 3, 4, 5] for v in ZEBRA_VARIABLES}
ZEBRA_DOMAINS["milk"] = [3]
ZEBRA_DOMAINS["norwegian"] = [1]

zebra = from_constraints(
    ZEBRA_VARIABLES, ZEBRA_DOMAINS,
    [c for group in CATEGORIES.values() for c in all_different(group)] + ZEBRA_CLUES,
)

st = {}
answer = csp_lib.solve(zebra, stats=st)
print(f"solved with {st['nodes']} nodes and {st['backtracks']} backtracks\n")
print(table([[h] + [next(v for v in group if answer[v] == h)
                    for group in CATEGORIES.values()]
             for h in range(1, 6)],
            ["house"] + list(CATEGORIES)))
print(f"\nWater is drunk by the "
      f"{next(n for n in CATEGORIES['nationality'] if answer[n] == answer['water'])}.")
print(f"The zebra belongs to the "
      f"{next(n for n in CATEGORIES['nationality'] if answer[n] == answer['zebra'])}.")
print("unique solution:",
      len(csp_lib.solve(zebra, all_solutions=True)) == 1)

# %% [markdown]
# ## 6. Bridge to language models
#
# The Zebra puzzle is a standing embarrassment for language models, and this
# module explains exactly why.
#
# **A puzzle like this has no left-to-right reading order.** Clue 15 constrains
# the same variable as clue 10; clue 6 only becomes useful once you know where
# the yellow house is; and the deduction that fixes house 1 depends on clues
# stated eight sentences apart. Solving it means holding all the domains
# simultaneously and shrinking them in whatever order the propagation demands.
#
# A model generating tokens left to right is doing the opposite. It commits to
# "house 1 is …" before it has consulted the clues that constrain house 5, and
# it cannot un-commit — the tokens are emitted. That is not a knowledge
# failure. It is a **control-flow** failure, and no amount of scale changes the
# shape of it. Chain of thought helps because it turns one forward pass into
# many, letting information from a late clue reach an early variable through
# the text. But it is still linear and still cannot backtrack cheaply.
#
# Three responses, all in use:
#
# 1. **Delegate.** Let the model translate the clues into variables and
#    constraints, then hand them to a solver. This is Module 4's Logic-LM
#    pattern, and it works well on exactly the tasks it targets, because
#    translating clue 6 into `green = ivory + 1` is a *local* reading task,
#    while solving the system is not.
# 2. **Constrained decoding.** Restrict the tokens the model may emit to those
#    consistent with a grammar or schema. This is arc consistency applied to
#    generation: prune the domain before sampling, rather than sampling and
#    then repairing. It guarantees well-formedness; it does not guarantee
#    correctness.
# 3. **Propagate in the prompt.** Ask the model to write out the domains and
#    cross them off explicitly — a tabular scratchpad rather than prose. This
#    moves the working memory into the context window where the model can
#    re-read it, and it measurably helps. It is also, exactly, asking the model
#    to simulate AC-3 by hand.
#
# The general principle is worth stating plainly, because it recurs:
#
# > **Match the control flow to the problem's structure.** Sequential
# > generation suits problems with a sequential structure. Constraint problems
# > have no such order, and forcing one on them is what makes them hard.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — consistency
#
# Write `consistent(csp, var, value, assignment)`: does `var = value` violate
# any constraint with an already-assigned neighbour? Only neighbours matter.

# %%
def consistent(csp, var, value, assignment):
    """True when var = value conflicts with nothing already assigned."""
    # TODO: check csp.constraint against each assigned neighbour
    return None


# %%
@checker("Exercise 8.1 — consistent")
def check_ex1():
    yield ("nothing assigned yet",
           consistent(australia, "WA", "red", {}), True)
    yield ("a neighbour with the same colour",
           consistent(australia, "WA", "red", {"NT": "red"}), False)
    yield ("…a different colour is fine",
           consistent(australia, "WA", "red", {"NT": "green"}), True)
    yield ("non-neighbours do not constrain",
           consistent(australia, "WA", "red", {"Q": "red"}), True)
    yield ("all neighbours are checked",
           consistent(australia, "SA", "red", {"WA": "green", "V": "red"}), False)
    yield ("agrees with csai.csp",
           consistent(australia, "SA", "blue", {"WA": "green", "V": "red"}),
           australia.consistent("SA", "blue", {"WA": "green", "V": "red"}))


check_ex1()

# %% [markdown]
# ### Exercise 2 — backtracking search
#
# Write `backtracking(csp)` returning a complete consistent assignment, or
# `None`. Assign variables in `csp.variables` order and values in domain order
# — no heuristics, no inference. This is the baseline everything else is
# measured against.

# %%
def backtracking(csp):
    """A solution by plain backtracking search, or None."""
    # TODO: recursive; pick the first unassigned variable, try each value
    return None


# %%
@checker("Exercise 8.2 — backtracking")
def check_ex2():
    got = backtracking(australia)
    yield "solves the map", australia.is_solution(got) if got else None, True
    yield ("…assigning every state",
           sorted(got or {}), sorted(AUSTRALIA_NEIGHBOURS))

    q6 = backtracking(queens(6))
    yield "solves 6 queens", queens(6).is_solution(q6) if q6 else None, True

    impossible = CSP(["a", "b", "c"], {v: ["x", "y"] for v in "abc"},
                     {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}},
                     lambda p, vp, q, vq: vp != vq)
    yield ("a triangle needs three colours, so two fail",
           backtracking(impossible), None)

    yield ("deterministic: first value in domain order",
           backtracking(australia)["WA"], "red")


check_ex2()

# %% [markdown]
# ### Exercise 3 — minimum remaining values
#
# Write `mrv(csp, assignment, domains)` returning the unassigned variable with
# the smallest domain, breaking ties by **degree** — the most neighbours that
# are still unassigned. Break any remaining tie by `csp.variables` order.

# %%
def mrv(csp, assignment, domains):
    """The unassigned variable with the fewest values, tie-broken by degree."""
    # TODO: min over unassigned by (len(domain), -unassigned neighbours)
    return None


# %%
@checker("Exercise 8.3 — mrv")
def check_ex3():
    domains = {s: list(COLOURS) for s in AUSTRALIA_NEIGHBOURS}
    yield ("all equal -> the highest degree wins (SA has five neighbours)",
           mrv(australia, {}, domains), "SA")

    tight = dict(domains)
    tight["T"] = ["red"]
    yield "a singleton domain wins", mrv(australia, {}, tight), "T"

    yield ("assigned variables are skipped",
           mrv(australia, {"T": "red"}, tight) != "T", True)

    two = dict(domains)
    two["Q"] = ["red", "green"]
    two["V"] = ["red", "green"]
    yield ("among equals, degree decides (Q has three, V has three)",
           mrv(australia, {}, two) in {"Q", "V"}, True)

    yield ("agrees with csai.csp",
           mrv(australia, {}, domains),
           csp_lib.select_mrv(australia, {}, domains))


check_ex3()

# %% [markdown]
# ### Exercise 4 — forward checking
#
# Write `forward(csp, var, value, domains)`: return new domains in which
# `var` is pinned to `[value]` and every unassigned neighbour has lost the
# values incompatible with it — or `None` if any domain becomes empty. Do not
# mutate the input.

# %%
def forward(csp, var, value, domains):
    """Domains after pruning neighbours against var = value, or None."""
    # TODO: copy, pin var, filter each neighbour's domain, detect a wipeout
    return None


# %%
@checker("Exercise 8.4 — forward")
def check_ex4():
    domains = {s: list(COLOURS) for s in AUSTRALIA_NEIGHBOURS}
    got = forward(australia, "WA", "red", domains)
    yield "the variable is pinned", (got or {}).get("WA"), ["red"]
    yield ("neighbours lose that colour",
           sorted((got or {}).get("NT", [])), ["blue", "green"])
    yield ("…and so does the other neighbour",
           sorted((got or {}).get("SA", [])), ["blue", "green"])
    yield ("non-neighbours are untouched",
           sorted((got or {}).get("Q", [])), sorted(COLOURS))
    yield "the input is not mutated", domains["NT"], COLOURS

    doomed = {s: list(COLOURS) for s in AUSTRALIA_NEIGHBOURS}
    doomed["T"] = ["red"]
    yield ("emptying a domain returns None",
           forward(australia, "V", "red", doomed), None)

    yield ("agrees with csai.csp",
           forward(australia, "SA", "green", domains),
           csp_lib.forward_check(australia, "SA", "green", domains))


check_ex4()

# %% [markdown]
# ### Exercise 5 — revise
#
# Write `revise(csp, xi, xj, domains)`: delete from `domains[xi]` every value
# with no compatible value in `domains[xj]`. Mutate `domains[xi]` in place and
# return `True` if anything was removed.

# %%
def revise(csp, xi, xj, domains):
    """Make xi arc-consistent with xj. Returns True if a value was removed."""
    # TODO: keep only values of xi supported by some value of xj
    return None


# %%
@checker("Exercise 8.5 — revise")
def check_ex5():
    domains = {"a": ["r", "g"], "b": ["r"]}
    tiny = CSP(["a", "b"], domains, {"a": {"b"}, "b": {"a"}},
               lambda p, vp, q, vq: vp != vq)
    d = {k: list(v) for k, v in domains.items()}
    yield "removes the unsupported value", revise(tiny, "a", "b", d), True
    yield "…leaving the rest", d["a"], ["g"]
    yield "a second pass changes nothing", revise(tiny, "a", "b", d), False
    yield "the other direction is already consistent", revise(
        tiny, "b", "a", {k: list(v) for k, v in domains.items()}), False

    wipe = {"a": ["r"], "b": ["r"]}
    tiny2 = CSP(["a", "b"], wipe, {"a": {"b"}, "b": {"a"}},
                lambda p, vp, q, vq: vp != vq)
    d2 = {k: list(v) for k, v in wipe.items()}
    revise(tiny2, "a", "b", d2)
    yield "an unsatisfiable pair empties the domain", d2["a"], []


check_ex5()

# %% [markdown]
# ### Exercise 6 — AC-3
#
# Write `arc_consistency(csp, domains)`: enforce arc consistency everywhere.
# Return `False` if a domain empties, `True` otherwise, mutating `domains`.
#
# <details><summary>Hint</summary>
#
# Start with a queue of every arc `(xi, xj)` where `xj` is a neighbour of
# `xi`. Pop one and `revise` it. If `domains[xi]` shrank, every arc *into*
# `xi` may have lost support, so push `(xk, xi)` for every neighbour `xk`
# other than `xj`. Stop when the queue is empty or a domain is empty.
# </details>

# %%
def arc_consistency(csp, domains):
    """Enforce arc consistency. False if a domain empties."""
    # TODO: a queue of arcs; re-enqueue neighbours whenever a domain shrinks
    return None


# %%
@checker("Exercise 8.6 — arc_consistency")
def check_ex6():
    d = {s: list(COLOURS) for s in AUSTRALIA_NEIGHBOURS}
    yield "three colours are enough for Australia", arc_consistency(australia, d), True
    yield ("…and with nothing assigned, nothing can be pruned",
           all(len(v) == 3 for v in d.values()), True)

    fixed = {s: list(COLOURS) for s in AUSTRALIA_NEIGHBOURS}
    fixed["WA"] = ["red"]
    fixed["NT"] = ["green"]
    arc_consistency(australia, fixed)
    yield ("pinning two states prunes a third",
           sorted(fixed["SA"]), ["blue"])

    triangle = CSP(["a", "b", "c"], {v: ["x", "y"] for v in "abc"},
                   {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}},
                   lambda p, vp, q, vq: vp != vq)
    d3 = {v: ["x", "y"] for v in "abc"}
    yield ("a 2-colour triangle is arc consistent...",
           arc_consistency(triangle, d3), True)
    yield ("...and still unsatisfiable — propagation prunes, it does not decide",
           csp_lib.solve(triangle), None)

    zd = {v: list(d) for v, d in ZEBRA_DOMAINS.items()}
    arc_consistency(zebra, zd)
    yield ("on the Zebra puzzle it prunes a great deal",
           sum(len(v) for v in zd.values()) < sum(
               len(v) for v in ZEBRA_DOMAINS.values()), True)
    yield ("…without emptying anything",
           all(zd.values()), True)


check_ex6()

# %% [markdown]
# ---
# ## Project — solve the Zebra puzzle, and measure what solved it
#
# ```python
# solve_csp(csp, *, inference="ac3", use_mrv=True, stats=None) -> dict | None
# ```
#
# Backtracking search with switchable inference (`"none"`, `"forward"`,
# `"ac3"`) and switchable MRV. `stats`, if given, records `"nodes"` (values
# tried) and `"backtracks"` (values undone).
#
# Then:
#
# ```python
# zebra_answers(assignment) -> {"water": nationality, "zebra": nationality}
# compare(csp, configurations) -> {label: {"nodes": …, "backtracks": …, "solved": bool}}
# ```
#
# where `configurations` is a list of `(label, kwargs)` pairs.
#
# **Write-up questions:**
#
# 1. Which single change buys the most on the Zebra puzzle — inference or
#    variable ordering? Look carefully at the MRV-without-inference row: it is
#    dramatically *worse* than plain backtracking. Explain why, in terms of
#    what MRV needs in order to be measuring anything. Is your answer the same
#    for 16-queens?
# 2. Run AC-3 once on the Zebra puzzle's initial domains, before any search.
#    How many values does it eliminate, and how many variables does it decide
#    outright? What does that say about how much of this puzzle is "search" at
#    all?
# 3. Add a `select="last"` ordering that always takes the *last* unassigned
#    variable, and compare. If you had to explain to someone why a
#    left-to-right reasoner struggles here, which number would you show them?

# %%
def solve_csp(csp, *, inference="ac3", use_mrv=True, stats=None):
    """Backtracking search with switchable inference and variable ordering."""
    # TODO: recursive backtracking. Choose the variable with mrv() or the
    # first unassigned one; for each value, check consistency, apply the
    # chosen inference, and recurse. Count nodes and backtracks in `stats`.
    return None


def zebra_answers(assignment):
    """Who drinks water and who owns the zebra."""
    # TODO: find the nationality sharing a house number with each
    return None


def compare(csp, configurations):
    """Run several configurations on one problem and collect their statistics."""
    # TODO: call solve_csp with each kwargs dict, capturing a fresh stats dict
    return None


# %%
@checker("Project 8 — CSP solver")
def check_project():
    got = solve_csp(australia, inference="none", use_mrv=False)
    yield "solves the map", australia.is_solution(got) if got else None, True

    q8 = solve_csp(queens(8))
    yield "solves 8 queens", queens(8).is_solution(q8) if q8 else None, True

    answer = solve_csp(zebra)
    yield "solves the Zebra puzzle", zebra.is_solution(answer) if answer else None, True
    yield ("…giving the classic answer",
           zebra_answers(answer),
           {"water": "norwegian", "zebra": "japanese"})
    yield ("…with the Norwegian in house 1 and milk in house 3",
           ((answer or {}).get("norwegian"), (answer or {}).get("milk")), (1, 3))
    yield ("…and the green house immediately right of the ivory one",
           (answer or {}).get("green") == (answer or {}).get("ivory", 0) + 1, True)

    impossible = CSP(["a", "b", "c"], {v: ["x", "y"] for v in "abc"},
                     {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}},
                     lambda p, vp, q, vq: vp != vq)
    yield "an unsatisfiable CSP returns None", solve_csp(impossible), None

    st = {}
    solve_csp(zebra, inference="none", use_mrv=False, stats=st)
    yield "stats has both counters", sorted(st), ["backtracks", "nodes"]
    yield "…and counted something", st["nodes"] > 0, True

    results = compare(zebra, [
        ("plain", {"inference": "none", "use_mrv": False}),
        ("forward", {"inference": "forward", "use_mrv": False}),
        ("forward + MRV", {"inference": "forward", "use_mrv": True}),
        ("AC-3 + MRV", {"inference": "ac3", "use_mrv": True}),
    ])
    yield "compare reports every configuration", sorted(results or {}), [
        "AC-3 + MRV", "forward", "forward + MRV", "plain"]
    yield "…all of which solve it", all(
        r["solved"] for r in (results or {}).values()), True
    yield ("…and inference cuts the node count sharply",
           (results or {})["forward"]["nodes"] < (results or {})["plain"]["nodes"] / 2,
           True)
    yield ("…with MRV cutting it further",
           (results or {})["forward + MRV"]["nodes"] <
           (results or {})["forward"]["nodes"], True)


check_project()

# %%
# The comparison your write-up discusses.
if solve_csp(australia) is not None:
    configurations = [
        ("plain backtracking", {"inference": "none", "use_mrv": False}),
        ("+ MRV", {"inference": "none", "use_mrv": True}),
        ("+ forward checking", {"inference": "forward", "use_mrv": False}),
        ("forward + MRV", {"inference": "forward", "use_mrv": True}),
        ("AC-3 + MRV", {"inference": "ac3", "use_mrv": True}),
    ]
    for name, problem in [("Zebra puzzle", zebra), ("16 queens", queens(16))]:
        results = compare(problem, configurations)
        print(name)
        print(table([(label, r["nodes"], r["backtracks"],
                      "yes" if r["solved"] else "no")
                     for label, r in results.items()],
                    ["configuration", "nodes", "backtracks", "solved"],
                    align="lrrc"))
        print()

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * A. Mackworth, "Consistency in Networks of Relations" (1977) — AC-3.
# * R. Haralick & G. Elliott, "Increasing Tree Search Efficiency for
#   Constraint Satisfaction Problems" (1980) — forward checking and MRV,
#   measured.
# * R. Dechter, *Constraint Processing* (2003) — the standard reference.
# * S. Russell & P. Norvig, *AIMA* ch. 6.
# * *Handbook of Constraint Programming* (Rossi, van Beek & Walsh, 2006) —
#   global constraints and their propagators.
# * B. Dziri et al., "Faith and Fate: Limits of Transformers on
#   Compositionality" (2023) — why problems needing non-sequential composition
#   are hard for sequential models.
#
# **Next:** Module 9 asks not "what assignment satisfies these constraints?"
# but "what sequence of actions reaches this goal?" — planning, and the
# validator that keeps it honest.
