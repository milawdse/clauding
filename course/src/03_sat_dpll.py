# %% [markdown]
# # Module 3 — Inference as Search: CNF, Resolution and DPLL
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Convert any formula to conjunctive normal form, and say why the
#    conversion can explode (and what Tseitin did about it).
# 2. Apply the **resolution** rule and use it to prove a theorem by refutation.
# 3. Implement **DPLL**: backtracking search plus unit propagation and the
#    pure-literal rule, with statistics you can actually interpret.
# 4. Encode a non-logical problem — graph colouring — as a clause set, and
#    solve it.
# 5. Explain, with measurements, why propagation rather than search is what
#    makes SAT solvers fast, and what the analogous move is for a language
#    model.
#
# **Prerequisites:** Module 2 (formulas, models, entailment, the deduction
# theorem).
#
# **Time:** ~75 minutes plus exercises.

# %% [markdown]
# ## 1. Stop enumerating. Start searching.
#
# Module 2 ended with a wall: `n` symbols, `2**n` models, and a table that ran
# out of universe somewhere around 60 variables. It also ended with the way
# through:
#
# > KB ⊨ α  iff  **KB ∧ ¬α is unsatisfiable.**
#
# Enumeration answers "unsatisfiable?" by checking every model. Search answers
# it by *hunting for a contradiction* — and a contradiction, once found, is a
# certificate. You don't have to look at the other models; you have a proof.
#
# That reframing is the entire content of this module. It is also the first
# time in this course where a System-2 method gets *dramatically* cheaper than
# brute force without giving up correctness, which is the pattern every later
# module repeats.
#
# Two ingredients:
#
# 1. A **uniform syntax** so that one inference rule suffices — conjunctive
#    normal form.
# 2. A **search strategy** that prunes — DPLL.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import itertools
import time

from csai import logic
from csai.check import checker
from csai.logic import (And, Iff, Implies, Not, Or, negate_literal,
                        literal_symbol, to_str)
from csai.render import table

print("ready")

# %% [markdown]
# ## 2. Conjunctive normal form
#
# A formula is in **CNF** when it is an AND of ORs of literals, where a
# *literal* is a symbol or its negation:
#
# > (p ∨ ¬q) ∧ (q ∨ r ∨ ¬s) ∧ (¬p)
#
# Every propositional formula has an equivalent CNF, reachable in three
# mechanical steps:
#
# | step | rewrite |
# |---|---|
# | 1. eliminate ↔ | `a ↔ b` ⟶ `(a → b) ∧ (b → a)` |
# | 2. eliminate → | `a → b` ⟶ `¬a ∨ b` |
# | 3. push ¬ inward (De Morgan) | `¬(a ∧ b)` ⟶ `¬a ∨ ¬b`;  `¬(a ∨ b)` ⟶ `¬a ∧ ¬b`;  `¬¬a` ⟶ `a` |
# | 4. distribute ∨ over ∧ | `a ∨ (b ∧ c)` ⟶ `(a ∨ b) ∧ (a ∨ c)` |
#
# Steps 1–3 are cheap and produce **negation normal form** (NNF). Step 4 is
# the dangerous one.

# %%
f = Iff("p", And("q", Not("r")))
print("original:  ", to_str(f))
print("no ↔ or →: ", to_str(logic.eliminate_iff_implies(f)))
print("NNF:       ", to_str(logic.push_negations(logic.eliminate_iff_implies(f))))
print("CNF:       ", to_str(logic.to_cnf(f)))
print("clauses:   ", sorted(map(sorted, logic.to_clauses(f))))

# %% [markdown]
# ### Why step 4 is dangerous

# %%
rows = []
for n in range(1, 7):
    # (a1 ∧ b1) ∨ (a2 ∧ b2) ∨ … — distribution multiplies out
    formula = Or(*[And(f"a{i}", f"b{i}") for i in range(n)])
    rows.append((n, 2 * n, len(logic.to_clauses(formula))))
print(table(rows, ["n", "variables", "clauses after distribution"], align="rrr"))

# %% [markdown]
# Exponential in the input. The fix, due to Tseitin (1968), is to stop
# demanding an *equivalent* formula and settle for an **equisatisfiable** one:
# invent a fresh variable for each subformula, define it with a handful of
# clauses, and assert the top variable. The result is linear in the input size
# and satisfiable exactly when the original was — which is all a solver needs.
# Exercise 2 has you build the key step.
#
# This is a habit worth internalising: when the faithful transformation is
# unaffordable, look for a *weaker relation that preserves the question you
# are actually asking*. Nearly every scalable reasoning method in this course
# is an instance of that move.

# %% [markdown]
# ## 3. Clauses as sets
#
# Once in CNF, drop the tree. A clause is a **set of literals**, a formula is
# a **list of clauses**, and a literal is a string: `"p"` or `"-p"`.

# %%
clauses = logic.to_clauses(And(Implies("rain", "wet"), "rain", Not("dry")))
for c in clauses:
    print(sorted(c))

print("\nthe two degenerate clauses:")
print("  empty clause  frozenset()  -- no literal can be true: a contradiction, ⊥")
print("  unit clause   {'rain'}     -- forces rain = True")

# %% [markdown]
# Those two carry the whole algorithm. Deriving the **empty clause** is a
# proof of unsatisfiability. A **unit clause** leaves no choice, and following
# forced choices is free.

# %% [markdown]
# ## 4. Resolution: one rule is enough
#
# Take two clauses containing complementary literals:
#
# ```
#     (a ∨ b ∨ ¬c)      (c ∨ d)
#     ------------------------------
#            (a ∨ b ∨ d)
# ```
#
# Cancel `¬c` against `c` and union the rest. Sound: in any model, `c` is
# either true — so `d` must hold — or false — so `a ∨ b` must. Either way the
# resolvent holds.
#
# The remarkable part is **refutation completeness**: if a clause set is
# unsatisfiable, repeatedly resolving will eventually derive the empty clause.
# One rule, mechanically applied, decides propositional logic (Robinson,
# 1965).

# %%
def resolvents(c1, c2):
    """Every clause obtainable by resolving c1 with c2 on one literal."""
    out = []
    for lit in c1:
        if negate_literal(lit) in c2:
            new = (c1 - {lit}) | (c2 - {negate_literal(lit)})
            if not any(negate_literal(x) in new for x in new):   # skip tautologies
                out.append(frozenset(new))
    return out


def resolution_unsat(clause_set, budget=3000):
    """Saturate by resolution, up to a budget of clauses held at once.

    Returns (verdict, clauses_held). The verdict is "unsat" when the empty
    clause is derived, "sat" when the set saturates without one, and
    "budget exhausted" when we gave up — resolution is complete, but only if
    you can afford to run it to completion.
    """
    known = set(clause_set)
    while True:
        new = set()
        for c1, c2 in itertools.combinations(known, 2):
            for r in resolvents(c1, c2):
                if not r:
                    return "unsat", len(known)
                if r not in known:
                    new.add(r)
        if not new:
            return "sat", len(known)
        known |= new
        if len(known) > budget:
            return "budget exhausted", len(known)


# "Socrates": all men are mortal; Socrates is a man; is Socrates mortal?
kb = [Implies("man", "mortal"), "man"]
goal = "mortal"
refutation = logic.to_clauses(And(logic.as_conjunction(kb), Not(goal)))
print("clauses of KB ∧ ¬goal:", sorted(map(sorted, refutation)))
print("resolution verdict    :", resolution_unsat(refutation)[0])
print("so KB ⊨ mortal        :", logic.entails(kb, goal))

# %% [markdown]
# ### Where resolution runs out of road
#
# The **pigeonhole principle** — `n+1` pigeons into `n` holes, no two in the
# same hole — is unsatisfiable and obvious to a human. Haken (1985) proved that
# every resolution refutation of it is exponentially long. Watch the clause
# count.

# %%
def pigeonhole(pigeons, holes):
    """Unsatisfiable clause set: `pigeons` pigeons into `holes` holes."""
    def v(p, h):
        return f"p{p}h{h}"

    cs = [frozenset(v(p, h) for h in range(holes)) for p in range(pigeons)]
    for h in range(holes):
        for p1, p2 in itertools.combinations(range(pigeons), 2):
            cs.append(frozenset({"-" + v(p1, h), "-" + v(p2, h)}))
    return cs


for pigeons in (3, 4):
    cs = pigeonhole(pigeons, pigeons - 1)
    t0 = time.perf_counter()
    verdict, generated = resolution_unsat(cs, budget=2000)
    print(f"{pigeons} pigeons, {pigeons - 1} holes: started from {len(cs):>2} clauses "
          f"-> {verdict:<17} holding {generated:>5}  "
          f"{time.perf_counter() - t0:.2f}s")

# %% [markdown]
# Three pigeons: refuted in a moment. Four: thousands of clauses generated and
# we call it off before it finishes — from an input of twenty-two clauses,
# describing a fact a child can state in one sentence.
#
# Resolution *works*, and it is the right way to think about what a proof is.
# It is not how you build a fast solver, because saturation generates
# enormous numbers of clauses nobody will ever need. The winning idea was to
# search over **assignments** rather than over derivations, and to let the
# clauses prune that search. Keep an eye on the decision counts below: DPLL
# settles the four-pigeon instance in single digits.

# %% [markdown]
# ## 5. DPLL
#
# Davis, Putnam, Logemann and Loveland (1962). Backtracking search over
# assignments, with two prunes that do nearly all the work:
#
# 1. **Unit propagation.** A clause down to one literal forces that literal.
#    Assign it, simplify, repeat. No branching, no guessing — and one
#    propagation often cascades into dozens.
# 2. **Pure literal.** A symbol that appears with only one polarity anywhere
#    can be set that way; it can only ever help.
# 3. Otherwise **decide**: pick a variable, try `True`, backtrack and try
#    `False`.
#
# `simplify(clauses, literal)` implements "assume this literal": drop the
# clauses it satisfies, and delete its negation from the rest. Deleting the
# negation is what shrinks clauses toward units, and the empty clause is how
# you learn you were wrong.

# %%
def simplify(clauses, literal):
    """Clause set after assuming `literal` is true."""
    out = []
    for c in clauses:
        if literal in c:
            continue                                   # satisfied, drop it
        out.append(c - {negate_literal(literal)})      # shrink
    return out


demo = [frozenset({"a", "b"}), frozenset({"-a", "c"}), frozenset({"-c", "d"})]
print("start        :", sorted(map(sorted, demo)))
print("assume a     :", sorted(map(sorted, simplify(demo, "a"))))
print("then assume c:", sorted(map(sorted, simplify(simplify(demo, "a"), "c"))))
print("assume -a    :", sorted(map(sorted, simplify(demo, "-a"))))

# %% [markdown]
# Now the search itself. `csai.logic.dpll` is the reference implementation;
# read it, because the project asks you to write your own with instrumentation.

# %%
formula = And(Or("a", "b"), Or(Not("a"), "c"), Or(Not("c"), "d"), Not("b"))
cs = logic.to_clauses(formula)
stats = {}
model = logic.dpll(cs, stats=stats)
print("clauses:", sorted(map(sorted, cs)))
print("model  :", model)
print("stats  :", stats)
print("check  :", logic.evaluate(formula, {s: model.get(s, False)
                                           for s in logic.symbols(formula)}))

print("\nand on the pigeonhole, which resolution struggled with:")
for pigeons in (4, 6, 8):
    cs = pigeonhole(pigeons, pigeons - 1)
    st = {}
    t0 = time.perf_counter()
    result = logic.dpll(cs, stats=st)
    print(f"  {pigeons} pigeons: {'UNSAT' if result is None else 'SAT'}  "
          f"decisions={st['decisions']:>6} propagations={st['propagations']:>6}  "
          f"{time.perf_counter() - t0:.3f}s")

# %% [markdown]
# ## 6. Encoding a problem you actually care about
#
# SAT is only useful because other problems reduce to it. The recipe never
# changes: **choose variables, then write down the constraints as clauses.**
#
# Graph `k`-colouring. Variables `x[node, colour]` meaning "this node has this
# colour". Three families of constraint:
#
# | constraint | clauses |
# |---|---|
# | every node gets a colour | `(x[n,c1] ∨ x[n,c2] ∨ … ∨ x[n,ck])` for each node |
# | no node gets two | `(¬x[n,ci] ∨ ¬x[n,cj])` for each node and each pair of colours |
# | adjacent nodes differ | `(¬x[u,c] ∨ ¬x[v,c])` for each edge and colour |
#
# "At least one" is a single clause; "at most one" is a quadratic pile of
# them. That asymmetry is a permanent fact of encoding, and a large literature
# exists purely to make "at most one" cheaper.

# %%
def colour_var(node, colour):
    return f"{node}#{colour}"


def colouring_clauses(nodes, edges, k):
    """Clause set that is satisfiable iff the graph is k-colourable."""
    colours = range(k)
    cs = []
    for n in nodes:
        cs.append(frozenset(colour_var(n, c) for c in colours))
        for c1, c2 in itertools.combinations(colours, 2):
            cs.append(frozenset({"-" + colour_var(n, c1), "-" + colour_var(n, c2)}))
    for u, v in edges:
        for c in colours:
            cs.append(frozenset({"-" + colour_var(u, c), "-" + colour_var(v, c)}))
    return cs


# The Petersen graph: 3-chromatic, and a classic counterexample generator.
PETERSEN_NODES = list(range(10))
PETERSEN_EDGES = (
    [(i, (i + 1) % 5) for i in range(5)]            # outer pentagon
    + [(i + 5, (i + 2) % 5 + 5) for i in range(5)]  # inner pentagram
    + [(i, i + 5) for i in range(5)]                # spokes
)

for k in (2, 3):
    cs = colouring_clauses(PETERSEN_NODES, PETERSEN_EDGES, k)
    st = {}
    model = logic.dpll(cs, stats=st)
    verdict = "no colouring exists" if model is None else "colourable"
    print(f"k={k}: {len(cs):>3} clauses -> {verdict:<20} "
          f"decisions={st['decisions']:>4} propagations={st['propagations']:>4}")
    if model:
        assigned = {n: next(c for c in range(k) if model.get(colour_var(n, c)))
                    for n in PETERSEN_NODES}
        bad = [(u, v) for u, v in PETERSEN_EDGES if assigned[u] == assigned[v]]
        print(f"     colouring {assigned}\n     conflicting edges: {bad}")

# %% [markdown]
# ## 7. Bridge to language models
#
# **Propagation, not search, is what made SAT solvers fast.** The 1962
# algorithm branches; the modern one (CDCL — conflict-driven clause learning,
# Marques-Silva & Sakallah 1996) adds *learning from failure*: on hitting a
# contradiction it analyses which decisions caused it and records a new clause
# forbidding that combination, so the same mistake is never made twice in any
# part of the search tree. That single idea took SAT from toy problems to
# hardware verification with millions of variables.
#
# Hold that next to how a language model reasons.
#
# | SAT solver | language model doing chain of thought |
# |---|---|
# | unit propagation: forced consequences are free | every consequence costs tokens and can be got wrong |
# | conflict analysis: failure localised to its cause | a contradiction late in a chain rarely revises the early step that caused it |
# | learned clauses: a mistake made once is excluded forever | the same wrong step recurs across samples |
# | a certificate: the empty clause, checkable in isolation | the answer's support is the prose, checkable only by rereading it |
#
# Three research directions live in that table, and all three are active:
#
# * **Neuro-symbolic delegation.** Let the model do what it is good at —
#   turning an English problem into variables and constraints — and hand the
#   clause set to a solver. Module 12's Program-of-Thoughts arm is exactly
#   this, with a Python interpreter in place of a SAT solver.
# * **Verification instead of trust.** A cheap checker over a generated
#   candidate converts "usually right" into "right, or flagged". Modules 9 and
#   11 build checkers.
# * **Backtracking as an explicit control structure.** A linear chain of
#   thought cannot undo a decision; tree search over reasoning states can, and
#   Module 7 builds that.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — negation normal form
#
# Write `to_nnf(formula)`: eliminate `iff` and `implies`, then push every
# negation down until it sits directly on a symbol. Do **not** distribute.
#
# <details><summary>Hint</summary>
#
# Handle `not` by looking at what is inside it: `¬¬a` ⟶ `a`, `¬(a ∧ b)` ⟶
# `¬a ∨ ¬b`, `¬(a ∨ b)` ⟶ `¬a ∧ ¬b`. Rewrite `implies`/`iff` *before* you
# reach a negation of one, and the `not` case only ever sees and/or/not/symbol.
# </details>

# %%
def to_nnf(formula):
    """Equivalent formula with ¬ only on symbols, and no → or ↔."""
    # TODO: rewrite iff/implies, then push negations inward
    return None


# %%
@checker("Exercise 3.1 — to_nnf")
def check_ex1():
    def is_nnf(f):
        if isinstance(f, (str, bool)):
            return True
        if f[0] == "not":
            return isinstance(f[1], (str, bool))
        return f[0] in ("and", "or") and all(is_nnf(a) for a in f[1:])

    cases = [Not(Not("p")),
             Implies("p", "q"),
             Not(And("p", "q")),
             Not(Or("p", "q")),
             Iff("p", "q"),
             Not(Implies("p", "q")),
             And("p", Not(Or("q", Not("r")))),
             Not(Iff("p", And("q", "r")))]
    for f in cases:
        got = to_nnf(f)
        yield f"{to_str(f)} is in NNF", is_nnf(got) if got is not None else None, True
        if got is not None:
            same = all(logic.evaluate(got, m) == logic.evaluate(f, m)
                       for m in logic.all_models(logic.symbols(f)))
            yield f"{to_str(f)} keeps its meaning", same, True


check_ex1()

# %% [markdown]
# ### Exercise 2 — Tseitin's trick, one gate at a time
#
# The exponential blow-up came from distribution. Tseitin's answer: name each
# subformula with a fresh variable and *define* it with clauses.
#
# For `z ↔ (a ∧ b)` the definition is three clauses:
#
# ```
# (¬z ∨ a)    (¬z ∨ b)    (z ∨ ¬a ∨ ¬b)
# ```
#
# Write `tseitin_and(z, a, b)` returning that clause set (as a list of
# frozensets of literal strings), for symbols `z`, `a`, `b`. Then convince
# yourself with the checker that it really does force `z ≡ a ∧ b`.

# %%
def tseitin_and(z, a, b):
    """Clauses asserting z ↔ (a ∧ b), for three symbol names."""
    # TODO: three clauses, using "-x" for a negated literal
    return None


# %%
@checker("Exercise 3.2 — tseitin_and")
def check_ex2():
    cs = tseitin_and("z", "a", "b")
    yield "three clauses", len(cs or []), 3
    yield "as frozensets of literals", all(
        isinstance(c, frozenset) for c in (cs or [])) if cs else None, True

    def holds(model):
        return all(any(model[literal_symbol(l)] != l.startswith("-") for l in c)
                   for c in cs)

    if cs:
        for va, vb, vz in itertools.product([False, True], repeat=3):
            m = {"a": va, "b": vb, "z": vz}
            yield (f"a={int(va)} b={int(vb)} z={int(vz)}",
                   holds(m), vz == (va and vb))


check_ex2()

# %% [markdown]
# ### Exercise 3 — the resolution rule
#
# Write `resolve(c1, c2, literal)`: resolve two clauses on `literal` (which
# appears positively in `c1` and negatively in `c2`), returning the resolvent
# as a frozenset. Return `None` if the literals are not complementary as
# described.

# %%
def resolve(c1, c2, literal):
    """Resolvent of c1 and c2 on `literal`, or None if it does not apply."""
    # TODO: check literal ∈ c1 and its negation ∈ c2, then union the remainders
    return None


# %%
@checker("Exercise 3.3 — resolve")
def check_ex3():
    a = frozenset({"a", "b", "-c"})
    b = frozenset({"c", "d"})
    yield "basic resolvent", resolve(a, b, "-c"), frozenset({"a", "b", "d"})
    yield "order of the arguments matters", resolve(b, a, "c"), frozenset(
        {"a", "b", "d"})
    yield "not complementary -> None", resolve(a, b, "a"), None
    yield "literal absent -> None", resolve(a, b, "z"), None
    yield ("two complementary units give the empty clause",
           resolve(frozenset({"p"}), frozenset({"-p"}), "p"), frozenset())
    yield ("shared literals are not duplicated",
           resolve(frozenset({"p", "q"}), frozenset({"-p", "q"}), "p"),
           frozenset({"q"}))


check_ex3()

# %% [markdown]
# ### Exercise 4 — unit propagation
#
# The workhorse. Write `unit_propagate(clauses, assignment)`:
#
# * repeatedly find a unit clause, record its literal in the assignment, and
#   simplify the clause set by it;
# * stop when no unit clause remains;
# * return `(clauses, assignment)` — or `None` if you ever produce the **empty
#   clause**, which means the current assignment is contradictory.
#
# `assignment` maps symbol to bool; return a new dict, don't mutate the input.

# %%
def unit_propagate(clauses, assignment):
    """Propagate all units. Returns (clauses, assignment) or None on conflict."""
    # TODO: loop while a length-1 clause exists; use simplify(); watch for
    # frozenset() appearing in the clause list
    return None


# %%
@checker("Exercise 3.4 — unit_propagate")
def check_ex4():
    cs = [frozenset({"a"}), frozenset({"-a", "b"}), frozenset({"-b", "c"})]
    got = unit_propagate(cs, {})
    yield "a cascade assigns everything", (got[1] if got else None), {
        "a": True, "b": True, "c": True}
    yield "…and empties the clause list", (list(got[0]) if got else None), []

    conflict = [frozenset({"a"}), frozenset({"-a"})]
    yield "a direct contradiction -> None", unit_propagate(conflict, {}), None

    nothing = [frozenset({"a", "b"}), frozenset({"-a", "c"})]
    got = unit_propagate(nothing, {})
    yield "no units -> unchanged", (got[1] if got else None), {}
    yield "…clauses preserved", (sorted(map(sorted, got[0])) if got else None), sorted(
        map(sorted, nothing))

    seeded = unit_propagate([frozenset({"-a", "b"})], {"z": False})
    yield "existing assignment is kept", (seeded[1] if seeded else None), {"z": False}
    original = [frozenset({"a"}), frozenset({"-a", "b"})]
    unit_propagate(original, {})
    yield "input clause list not mutated", len(original), 2


check_ex4()

# %% [markdown]
# ### Exercise 5 — pure literals
#
# Write `pure_literals(clauses)`: return the **sorted list** of literals whose
# symbol appears with only one polarity across the whole clause set. Assigning
# a pure literal true can never turn a satisfiable set unsatisfiable — it only
# removes clauses.

# %%
def pure_literals(clauses):
    """Sorted list of literals appearing with only one polarity."""
    # TODO: collect all literals, then keep those whose negation never appears
    return None


# %%
@checker("Exercise 3.5 — pure_literals")
def check_ex5():
    yield ("both polarities -> not pure",
           pure_literals([frozenset({"a", "b"}), frozenset({"-a", "c"})]),
           ["b", "c"])
    yield ("a negative pure literal",
           pure_literals([frozenset({"-a", "b"}), frozenset({"-a", "-b"})]), ["-a"])
    yield "no clauses -> nothing", pure_literals([]), []
    yield ("everything pure",
           pure_literals([frozenset({"a", "b"}), frozenset({"b", "c"})]),
           ["a", "b", "c"])
    yield ("sorted",
           pure_literals([frozenset({"z"}), frozenset({"a"})]), ["a", "z"])


check_ex5()

# %% [markdown]
# ### Exercise 6 — proof by refutation
#
# Put Module 2's deduction theorem to work with this module's machinery. Write
# `proves(kb, query)` that decides KB ⊨ query by converting `KB ∧ ¬query` to
# clauses and checking unsatisfiability with `logic.dpll`.

# %%
def proves(kb, query):
    """True when kb entails query, decided by refutation with DPLL."""
    # TODO: clauses of And(as_conjunction(kb), Not(query)); unsat means entailed
    return None


# %%
@checker("Exercise 3.6 — proves")
def check_ex6():
    kb1 = [Implies("rain", "wet"), "rain"]
    yield "modus ponens", proves(kb1, "wet"), True
    yield "affirming the consequent fails", proves(
        [Implies("rain", "wet"), "wet"], "rain"), False
    yield "modus tollens", proves(
        [Implies("rain", "wet"), Not("wet")], Not("rain")), True
    yield "hypothetical syllogism", proves(
        [Implies("a", "b"), Implies("b", "c")], Implies("a", "c")), True
    yield "a tautology needs no premises", proves([], Or("p", Not("p"))), True
    yield "a contradiction proves anything", proves(["p", Not("p")], "q"), True
    yield "agrees with model checking", proves(kb1, "wet"), logic.entails(kb1, "wet")
    yield ("…including where it says no",
           proves(kb1, "sprinkler"), logic.entails(kb1, "sprinkler"))


check_ex6()

# %% [markdown]
# ---
# ## Project — DPLL, instrumented
#
# Write your own solver and use it to measure where the speed comes from.
#
# ```python
# solve(clauses, *, propagate=True, pure=True, stats=None) -> dict | None
# ```
#
# * Returns a model assigning **every symbol occurring in `clauses`** (default
#   the unused ones to `False`), or `None` if unsatisfiable.
# * `propagate=False` disables unit propagation; `pure=False` disables the
#   pure-literal rule. Both off leaves plain backtracking search — which is
#   the point of the experiment.
# * `stats`, if given, is a dict you increment: `"decisions"` (branch points),
#   `"propagations"` (literals forced by unit clauses) and `"conflicts"`
#   (empty clauses produced).
#
# Then answer, in the write-up cell, using the sweep below:
#
# 1. How many decisions does plain backtracking need on Petersen 3-colouring,
#    and how many does unit propagation save?
# 2. The pure-literal rule is cheap to state and rarely used in modern
#    solvers. What do your numbers suggest about why?
# 3. Where does the time actually go on the pigeonhole instances, and what
#    does that say about the difference between "this is unsatisfiable" and
#    "here is a model"?

# %%
def variables(clauses):
    """Every symbol occurring in a clause set, sorted."""
    return sorted({literal_symbol(l) for c in clauses for l in c})


def satisfies(clauses, model):
    """Does `model` satisfy every clause?"""
    return all(
        any(model.get(literal_symbol(l), False) != l.startswith("-") for l in c)
        for c in clauses
    )


def solve(clauses, *, propagate=True, pure=True, stats=None):
    """DPLL. Returns a total model over variables(clauses), or None."""
    # TODO: recursive search. Propagate units (if enabled), assign pure
    # literals (if enabled), detect the empty clause as a conflict, then pick
    # an unassigned literal and branch on both polarities.
    return None


# %%
@checker("Project 3 — DPLL")
def check_project():
    sat_cs = logic.to_clauses(And(Or("a", "b"), Or(Not("a"), "c"),
                                  Or(Not("c"), "d"), Not("b")))
    m = solve(sat_cs)
    yield "solves a satisfiable set", isinstance(m, dict), True
    yield "…with a model that checks out", satisfies(sat_cs, m) if m else None, True
    yield ("…total over every variable",
           sorted(m) if m else None, variables(sat_cs))

    yield "detects a contradiction", solve([frozenset({"p"}), frozenset({"-p"})]), None
    yield "no clauses -> the empty model", solve([]), {}
    yield "the empty clause alone is unsat", solve([frozenset()]), None

    php = pigeonhole(4, 3)
    yield "pigeonhole 4->3 is unsatisfiable", solve(php), None
    yield ("…and still unsatisfiable without propagation",
           solve(php, propagate=False, pure=False), None)

    pet3 = colouring_clauses(PETERSEN_NODES, PETERSEN_EDGES, 3)
    m3 = solve(pet3)
    yield "Petersen is 3-colourable", isinstance(m3, dict), True
    yield "…legally", satisfies(pet3, m3) if m3 else None, True
    if m3:
        assigned = {n: [c for c in range(3) if m3[colour_var(n, c)]]
                    for n in PETERSEN_NODES}
        yield "…exactly one colour per node", all(
            len(v) == 1 for v in assigned.values()), True
        yield "…and no edge monochromatic", all(
            assigned[u] != assigned[v] for u, v in PETERSEN_EDGES), True
    yield ("Petersen is not 2-colourable",
           solve(colouring_clauses(PETERSEN_NODES, PETERSEN_EDGES, 2)), None)

    st = {}
    solve(pet3, stats=st)
    yield "stats has the three keys", sorted(st), [
        "conflicts", "decisions", "propagations"]
    yield "…and propagation happened", st.get("propagations", 0) > 0, True

    with_prop, without = {}, {}
    solve(pet3, stats=with_prop)
    solve(pet3, propagate=False, pure=False, stats=without)
    yield ("propagation cuts the number of decisions",
           with_prop["decisions"] < without["decisions"], True)

    every = solve(sat_cs, propagate=False, pure=False)
    yield ("plain backtracking finds a model too",
           satisfies(sat_cs, every) if every else None, True)


check_project()

# %%
# The sweep your write-up discusses.
if solve([frozenset({"a"})]) is not None:
    rows = []
    problems = [
        ("Petersen 3-col (SAT)", colouring_clauses(PETERSEN_NODES, PETERSEN_EDGES, 3)),
        ("Petersen 2-col (UNSAT)", colouring_clauses(PETERSEN_NODES, PETERSEN_EDGES, 2)),
        ("pigeonhole 5->4 (UNSAT)", pigeonhole(5, 4)),
        ("pigeonhole 6->5 (UNSAT)", pigeonhole(6, 5)),
    ]
    for name, cs in problems:
        for label, kw in [("full", {}),
                          ("no pure literal", {"pure": False}),
                          ("no propagation", {"propagate": False, "pure": False})]:
            st = {}
            t0 = time.perf_counter()
            result = solve(cs, stats=st, **kw)
            rows.append((name, label, "SAT" if result else "UNSAT",
                         st.get("decisions", 0), st.get("propagations", 0),
                         st.get("conflicts", 0), f"{time.perf_counter() - t0:.3f}"))
    print(table(rows, ["problem", "configuration", "result", "decisions",
                       "propagations", "conflicts", "seconds"],
                align="llrrrrr"))

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * M. Davis, G. Logemann & D. Loveland, "A Machine Program for
#   Theorem-Proving" (1962) — DPLL, five pages.
# * J. A. Robinson, "A Machine-Oriented Logic Based on the Resolution
#   Principle" (1965).
# * G. Tseitin, "On the Complexity of Derivation in Propositional Calculus"
#   (1968) — the linear-size encoding.
# * A. Haken, "The Intractability of Resolution" (1985) — the pigeonhole
#   lower bound.
# * J. Marques-Silva & K. Sakallah, "GRASP: A Search Algorithm for
#   Propositional Satisfiability" (1996) — conflict-driven clause learning.
# * *Handbook of Satisfiability* (Biere et al., 2nd ed. 2021) — the reference.
#
# **Next:** Module 4 gives logic objects, relations and variables — first-order
# logic, unification, and a working mini-Prolog.
