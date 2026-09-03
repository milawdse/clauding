# %% [markdown]
# # Module 4 — First-Order Logic, Unification, and a Mini-Prolog
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Say precisely what propositional logic cannot express, and why adding
#    objects and variables fixes it.
# 2. Implement **substitution** and **unification**, including the occurs
#    check, and explain what "most general" means.
# 3. Explain why rules must be **renamed apart** before use, by watching what
#    breaks when they aren't.
# 4. Implement **backward chaining** (SLD resolution) — a working Prolog — and
#    read the proof trace it produces.
# 5. Connect variable binding to the thing language models are least reliable
#    at, and know what the standard workaround is.
#
# **Prerequisites:** Modules 2 and 3. Python generators (`yield`, `yield from`)
# are used throughout; the project is much easier with them.
#
# **Time:** ~75 minutes plus exercises.

# %% [markdown]
# ## 1. Where propositional logic runs out
#
# "All men are mortal. Socrates is a man. Therefore Socrates is mortal." —
# the syllogism every logic course opens with, and propositional logic cannot
# state it.
#
# You can *approximate* it. Introduce a symbol `socrates_is_a_man` and another
# `socrates_is_mortal`, add `socrates_is_a_man → socrates_is_mortal`, and the
# inference goes through. But you have not represented "all men"; you have
# represented one instance of it. For a thousand people you need a thousand
# implications, all of them copies, none of them sharing structure. For an
# infinite domain — every natural number, every list — you need infinitely
# many, and you are stuck.
#
# What is missing is the ability to say something about **an unspecified
# object**:
#
# > ∀x. man(x) → mortal(x)
#
# Three new ingredients:
#
# | ingredient | example | what it buys |
# |---|---|---|
# | **objects** (terms) | `socrates`, `s(s(z))`, `bob` | things to talk about |
# | **relations** (predicates) | `man(socrates)`, `parent(bob, alice)` | structured claims, not opaque symbols |
# | **variables** | `X`, `Y` | one rule covering unboundedly many cases |
#
# Full first-order logic adds quantifiers (∀, ∃) and gets undecidable —
# semi-decidable, in fact: a proof will be found if one exists, and the search
# may never halt otherwise (Church and Turing, 1936). This module takes the
# fragment that stays practical and made a programming language: **definite
# clauses**, where every rule has one positive conclusion and all variables are
# implicitly universally quantified.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

from csai import fol
from csai.check import checker
from csai.fol import (KnowledgeBase, Rule, Var, is_compound, is_var, parse,
                      parse_program, parse_rule, subst_str, term_str)
from csai.render import table, tree

print("ready")

# %% [markdown]
# ## 2. Terms
#
# | thing | written | Python |
# |---|---|---|
# | constant | `socrates`, `3` | `"socrates"`, `3` |
# | variable | `X` (uppercase, by convention) | `Var("X")` |
# | compound | `parent(bob, X)` | `("parent", "bob", Var("X"))` |
#
# Compounds nest, which is how you build data: `s(s(z))` is the number two,
# `point(1, 2)` is a pair. There are no types and no evaluation — `plus(2, 3)`
# is a three-element structure, not `5`. Terms are *syntax*; meaning comes
# entirely from the rules you write about them.

# %%
for text in ["socrates", "X", "parent(bob, alice)", "add(s(s(z)), Y, R)"]:
    t = parse(text)
    print(f"{text:<24} {t!r:<52} vars={fol.variables_in(t)}")

# %% [markdown]
# ## 3. Substitution
#
# A **substitution** maps variables to terms; applying it replaces every
# occurrence. Bindings chain — if `X ↦ Y` and `Y ↦ bob` then `X` resolves all
# the way to `bob` — so `substitute` follows the chain rather than applying
# once.

# %%
theta = {Var("X"): "bob", Var("Y"): parse("f(X)")}
for text in ["p(X)", "p(X, Y)", "q(Y, g(Y, Z))"]:
    print(f"{text:<16} under {subst_str(theta):<24} -> "
          f"{term_str(fol.substitute(parse(text), theta))}")

# %% [markdown]
# ## 4. Unification
#
# The central operation. Given two terms, find a substitution making them
# **identical**:
#
# ```
# unify( parent(bob, X),  parent(Y, alice) )  ->  {X = alice, Y = bob}
# unify( f(X, X),         f(a, b)         )  ->  failure
# ```
#
# Unification is pattern matching in both directions at once: either side may
# contain variables, and each constrains the other. That two-way character is
# what lets a Prolog predicate run "forwards" and "backwards" from the same
# definition, as you will see in §7.
#
# The result is the **most general unifier** (MGU): it commits to as little as
# possible. `unify(p(X), p(Y))` returns `{X = Y}`, not `{X = bob, Y = bob}` —
# both make the terms identical, but only the first leaves every other
# possibility open. Every unifier is an instance of the MGU, which is why one
# proof step never needs to guess.

# %%
pairs = [
    ("parent(bob, X)", "parent(Y, alice)"),
    ("f(X, X)", "f(a, a)"),
    ("f(X, X)", "f(a, b)"),
    ("f(X, g(Y))", "f(a, g(h(Z)))"),
    ("p(X)", "p(Y)"),
    ("p(a)", "q(a)"),
    ("add(s(X), Y, s(Z))", "add(s(s(z)), s(z), R)"),
]
rows = []
for a, b in pairs:
    result = fol.unify(parse(a), parse(b))
    rows.append((a, b, "fail" if result is None else subst_str(result)))
print(table(rows, ["term 1", "term 2", "most general unifier"]))

# %% [markdown]
# ### The occurs check
#
# What unifies `X` with `f(X)`? A term that is a proper subterm of itself —
# `f(f(f(…)))` — which does not exist. The **occurs check** refuses the
# binding.
#
# It is also the standard example of a correctness/performance trade: checking
# costs time on every binding, and real Prolog systems leave it *off* by
# default, which makes them unsound on exactly this case. When your logic
# engine loops forever on a term that eats itself, this is why.

# %%
print("with occurs check   :", fol.unify(Var("X"), parse("f(X)")))
print("without occurs check:",
      subst_str(fol.unify(Var("X"), parse("f(X)"), occurs_check=False)))
print("…and now substituting into X diverges, which is the whole problem.")

# %% [markdown]
# ## 5. Rules, and why they must be renamed
#
# A **definite clause** is `head :- body₁, …, bodyₙ`, read "head holds if all
# the body goals hold", with every variable universally quantified. A **fact**
# has an empty body.
#
# The variables in a rule are *local to that rule*. Use the same rule twice in
# one proof without renaming, and the two uses share variables that have
# nothing to do with each other — the classic bug.

# %%
rule = parse_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")
print("as written:", rule)
print("renamed   :", fol.rename(rule, 1))
print("again     :", fol.rename(rule, 2))
print("\nEach use gets its own X, Y, Z, so nesting the rule inside itself")
print("cannot accidentally force the two levels to agree.")

# %% [markdown]
# ## 6. Backward chaining (SLD resolution)
#
# The proof procedure, in four lines:
#
# > To prove a list of goals:
# > * no goals left → success, return the accumulated substitution;
# > * otherwise take the first goal, and for **every** rule whose (renamed)
# >   head unifies with it, replace that goal by the rule's body and recurse.
#
# Depth-first over rules in order, backtracking on failure. That is Prolog.
# Everything else in a real system — indexing, cut, arithmetic, I/O — is
# performance or convenience on top of these four lines.

# %%
family = KnowledgeBase(parse_program("""
    parent(bob, alice).
    parent(alice, carol).
    parent(carol, dave).
    parent(bob, edward).
    male(bob).      male(dave).     male(edward).
    female(alice).  female(carol).

    ancestor(X, Y) :- parent(X, Y).
    ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
    sibling(X, Y)  :- parent(P, X), parent(P, Y), different(X, Y).
    different(alice, edward).  different(edward, alice).
"""))

for query in ["parent(bob, W)", "ancestor(bob, W)", "ancestor(W, dave)",
              "sibling(alice, W)", "ancestor(dave, W)"]:
    answers = [subst_str(s) for s in family.ask(query)]
    print(f"?- {query:<22} {answers if answers else 'no.'}")

# %% [markdown]
# `ancestor(dave, W)` correctly answers **no** — Dave has no descendants — and
# it does so by exhausting the search space, not by looking anything up. Note
# also that `ancestor(W, dave)` runs the relation *backwards*: the same two
# rules answer "who are Dave's ancestors" and "who are Bob's descendants",
# because unification does not care which side the variables are on.
#
# ### The proof trace
#
# Each answer comes with a derivation. This is a **certificate**: it can be
# checked step by step without redoing the search — the same property that
# made the SAT solver's empty clause valuable in Module 3, and exactly what a
# chain of thought promises but does not guarantee.

# %%
goal = parse("ancestor(bob, dave)")
subst, steps = next(family.prove([goal], {}, 30))
print(f"?- {term_str(goal)}\n")
for depth, resolved, used in steps:
    print(f"{'  ' * depth}goal {resolved}")
    print(f"{'  ' * depth}  by  {used}")
print("\nyes.")

# %% [markdown]
# ## 7. Computation as deduction
#
# Definite clauses are a programming language, and this is the demonstration
# that makes it obvious. Numbers as terms: `z` is zero, `s(X)` is X + 1.
# Addition as a *relation*, not a function:
#
# ```prolog
# add(z, Y, Y).                            % 0 + Y = Y
# add(s(X), Y, s(Z)) :- add(X, Y, Z).      % (X+1) + Y = (X+Y)+1
# ```
#
# Two clauses, no arithmetic, no assignment. And because it is a relation
# rather than a function, the same definition runs in every direction.

# %%
peano = KnowledgeBase(parse_program("""
    add(z, Y, Y).
    add(s(X), Y, s(Z)) :- add(X, Y, Z).
    leq(z, Y).
    leq(s(X), s(Y)) :- leq(X, Y).
"""))


def to_peano(n):
    return "z" if n == 0 else ("s", to_peano(n - 1))


def from_peano(t):
    return 0 if t == "z" else 1 + from_peano(t[1])


two, three = to_peano(2), to_peano(3)

forwards = peano.ask_one(("add", two, three, Var("R")))
print("forwards   2 + 3 = ?     ->", from_peano(forwards[Var("R")]))

backwards = peano.ask_one(("add", two, Var("R"), to_peano(5)))
print("backwards  2 + ? = 5     ->", from_peano(backwards[Var("R")]))

print("both       ? + ? = 4     ->", [
    (from_peano(s[Var("A")]), from_peano(s[Var("B")]))
    for s in peano.ask(("add", Var("A"), Var("B"), to_peano(4)))
])

print("checking   2 ≤ 3         ->", peano.holds(("leq", two, three)))
print("checking   3 ≤ 2         ->", peano.holds(("leq", three, two)))

# %% [markdown]
# "? + ? = 4" enumerating all five decompositions is the moment the idea lands.
# You wrote a *specification* of addition and got a solver, a checker and an
# enumerator from it. This is the same generate-and-test duality as Module 3's
# SAT encoding — state the constraints, let the machine search — and it is the
# core promise of declarative programming.
#
# It is also where the costs show up. Reorder the two `add` clauses, or query
# `add(A, B, C)` with everything unbound, and depth-first search happily runs
# forever down an infinite branch. Prolog is complete only in the sense that a
# proof will be found *if the search reaches it*; the search order is yours to
# get right. Module 6 is about search orders that do not have this problem.

# %% [markdown]
# ## 8. Bridge to language models
#
# **Variable binding is the specific thing this module adds, and it is the
# specific thing language models are worst at.**
#
# A transformer has no variables. It has attention over positions, which can
# *imitate* binding — copy this token to that slot — and does so well enough
# that models solve many problems that look like they need binding. But the
# failures are systematic and recognisable:
#
# * **Long-range coreference.** Bind `X` early, use it fifteen steps later,
#   and the binding drifts. This is the same failure Module 1 measured as
#   accuracy falling with depth.
# * **Consistent renaming.** "Let the first number be `a`" — and three
#   paragraphs later `a` quietly means something else. A prover renames apart
#   mechanically (§5); a model has to remember to.
# * **Genuine quantification.** "Everyone who owns a dog walks it" applied to
#   a list of forty people, where the answer requires *all* instances and not
#   the three most salient. Attention is a soft selection over positions; ∀ is
#   not.
#
# The productive response is not to make models better at binding. It is
# **delegation**: have the model translate the problem into a representation
# with real variables, and let an engine that has them do the work.
#
# | pipeline | who does what |
# |---|---|
# | LLM → Prolog/Datalog → engine | model writes the clauses, engine derives |
# | LLM → SQL → database | the same idea, with the world's most deployed logic engine |
# | LLM → Python → interpreter | Program-of-Thoughts, which Module 12 measures |
# | LLM → SMT/SAT → solver | for constraint and verification problems (Module 8) |
#
# Every row shares a shape: **the model handles ambiguity and world knowledge;
# the engine handles binding and exhaustive search.** Each is doing what the
# other is bad at. Keep the pattern in mind — the capstone is a direct
# measurement of what it buys.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — substitution
#
# Write `apply_subst(term, subst)`: replace variables by their bindings,
# following chains to the end. Recurse into compound terms.
#
# <details><summary>Hint</summary>
#
# Three cases. A `Var` bound in `subst`: substitute into *its value* too, so
# `{X: Y, Y: bob}` takes `X` all the way to `bob`. A compound
# `(functor, *args)`: rebuild it with each argument substituted. Anything
# else: return it unchanged.
# </details>

# %%
def apply_subst(term, subst):
    """Apply `subst` to `term`, following chains of bindings."""
    # TODO: Var / compound / constant
    return None


# %%
@checker("Exercise 4.1 — apply_subst")
def check_ex1():
    X, Y, Z = Var("X"), Var("Y"), Var("Z")
    yield "unbound variable is left alone", apply_subst(X, {}), X
    yield "bound variable", apply_subst(X, {X: "bob"}), "bob"
    yield "chained bindings", apply_subst(X, {X: Y, Y: "bob"}), "bob"
    yield "constants pass through", apply_subst("bob", {X: "alice"}), "bob"
    yield "numbers pass through", apply_subst(7, {X: "alice"}), 7
    yield ("inside a compound", apply_subst(parse("p(X, a)"), {X: "bob"}),
           parse("p(bob, a)"))
    yield ("nested compounds", apply_subst(parse("p(f(X), g(Y, X))"),
                                           {X: "a", Y: parse("h(Z)")}),
           parse("p(f(a), g(h(Z), a))"))
    yield ("agrees with csai.fol", apply_subst(parse("p(X, Y)"), {X: Y, Y: "c"}),
           fol.substitute(parse("p(X, Y)"), {X: Y, Y: "c"}))


check_ex1()

# %% [markdown]
# ### Exercise 2 — the occurs check
#
# Write `occurs_in(var, term, subst)`: does `var` appear anywhere inside
# `term` once `subst` has been applied?

# %%
def occurs_in(var, term, subst):
    """True when `var` occurs inside `term` after substitution."""
    # TODO: substitute first, then look for the variable in the result
    return None


# %%
@checker("Exercise 4.2 — occurs_in")
def check_ex2():
    X, Y = Var("X"), Var("Y")
    yield "a variable occurs in itself", occurs_in(X, X, {}), True
    yield "…but not in another", occurs_in(X, Y, {}), False
    yield "not in a constant", occurs_in(X, "bob", {}), False
    yield "directly inside", occurs_in(X, parse("f(X)"), {}), True
    yield "deeply inside", occurs_in(X, parse("f(g(h(X)), a)"), {}), True
    yield "absent", occurs_in(X, parse("f(g(h(Y)), a)"), {}), False
    yield ("only via the substitution", occurs_in(X, Y, {Y: parse("f(X)")}), True)


check_ex2()

# %% [markdown]
# ### Exercise 3 — unification
#
# The centrepiece. Write `mgu(x, y, subst=None)` returning the most general
# unifier extending `subst`, or `None` if the terms cannot be unified. Include
# the occurs check. Do not mutate `subst`.
#
# <details><summary>Hint</summary>
#
# Recursive, five cases:
# 1. `x` is a variable → `unify_var(x, y)`;
# 2. `y` is a variable → `unify_var(y, x)`;
# 3. both compound → functors and arities must match, then fold over the
#    argument pairs, threading the substitution through;
# 4. otherwise → the substitution unchanged if `x == y`, else `None`.
#
# `unify_var(v, t)`: if `v` is already bound, unify its value with `t`
# instead; if `t` is a bound variable, unify `v` with *its* value; if the
# occurs check fires, fail; otherwise add `v ↦ t`.
# </details>

# %%
def mgu(x, y, subst=None):
    """Most general unifier of x and y extending subst, or None."""
    # TODO: recurse; handle variables on either side, then compounds
    return None


# %%
@checker("Exercise 4.3 — mgu")
def check_ex3():
    X, Y, Z, R = Var("X"), Var("Y"), Var("Z"), Var("R")
    yield "identical constants", mgu("a", "a"), {}
    yield "different constants fail", mgu("a", "b"), None
    yield "variable binds a constant", mgu(X, "a"), {X: "a"}
    yield "…in either direction", mgu("a", X), {X: "a"}
    yield "a variable with itself", mgu(X, X), {}
    yield ("two arguments at once", mgu(parse("parent(bob, X)"),
                                        parse("parent(Y, alice)")),
           {Y: "bob", X: "alice"})
    yield "different functors fail", mgu(parse("p(a)"), parse("q(a)")), None
    yield "different arities fail", mgu(parse("p(a)"), parse("p(a, b)")), None
    yield ("a repeated variable constrains both",
           mgu(parse("f(X, X)"), parse("f(a, b)")), None)
    yield ("…and succeeds when consistent",
           mgu(parse("f(X, X)"), parse("f(a, a)")), {X: "a"})
    yield "the occurs check fires", mgu(X, parse("f(X)")), None
    yield "nested structure", mgu(parse("f(X, g(Y))"),
                                  parse("f(a, g(h(Z)))")), {X: "a",
                                                            Y: parse("h(Z)")}

    got = mgu(parse("add(s(X), Y, s(Z))"), parse("add(s(s(z)), s(z), R)"))
    want = fol.unify(parse("add(s(X), Y, s(Z))"), parse("add(s(s(z)), s(z), R)"))
    yield "agrees with csai.fol on a Peano goal", got, want

    seed = {Y: "bob"}
    mgu(X, Y, seed)
    yield "does not mutate the input substitution", seed, {Y: "bob"}

    # Most general: unifying two variables must not over-commit.
    result = mgu(parse("p(X)"), parse("p(Y)"))
    yield ("p(X) with p(Y) binds one variable, not two",
           len(result) if result is not None else None, 1)


check_ex3()

# %% [markdown]
# ### Exercise 4 — rename apart
#
# Write `rename_rule(rule, n)` returning a copy of `rule` with every variable
# `V` renamed to `V_n`. Head and body must be renamed **consistently**.

# %%
def rename_rule(rule, n):
    """Copy of `rule` with each variable V renamed to Var(f"{V.name}_{n}")."""
    # TODO: collect the variables of head and body, build one mapping, apply it
    return None


# %%
@checker("Exercise 4.4 — rename_rule")
def check_ex4():
    r = parse_rule("ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).")
    got = rename_rule(r, 7)
    yield "still a Rule", isinstance(got, Rule), True
    yield ("head renamed", got.head if got else None,
           parse("ancestor(X_7, Y_7)"))
    yield ("body renamed consistently", tuple(got.body) if got else None,
           (parse("parent(X_7, Z_7)"), parse("ancestor(Z_7, Y_7)")))
    yield ("a fact has an empty body",
           rename_rule(parse_rule("parent(bob, alice)."), 1),
           Rule(parse("parent(bob, alice)")))
    yield ("constants are untouched",
           rename_rule(parse_rule("p(X, bob)."), 2), Rule(parse("p(X_2, bob)")))
    yield ("two renamings share no variables",
           set(fol.variables_in(rename_rule(r, 1).head)) &
           set(fol.variables_in(rename_rule(r, 2).head)), set())


check_ex4()

# %% [markdown]
# ### Exercise 5 — ground terms
#
# Write `ground(term)`: `True` when `term` contains no variables. Ground terms
# are the ones that denote a definite object — the answers a query is trying
# to produce.

# %%
def ground(term):
    """True when `term` contains no variables."""
    # TODO: recurse into compounds; a Var makes it non-ground
    return None


# %%
@checker("Exercise 4.5 — ground")
def check_ex5():
    yield "a constant", ground("bob"), True
    yield "a number", ground(3), True
    yield "a variable", ground(Var("X")), False
    yield "a ground compound", ground(parse("parent(bob, alice)")), True
    yield "a compound with a variable", ground(parse("parent(bob, X)")), False
    yield "deeply buried variable", ground(parse("f(g(h(a, X)), b)")), False
    yield "deeply ground", ground(parse("f(g(h(a, c)), b)")), True
    yield "Peano numerals are ground", ground(to_peano(3)), True


check_ex5()

# %% [markdown]
# ### Exercise 6 — unify a list of pairs
#
# Write `unify_all(pairs, subst=None)`: unify each `(a, b)` in turn, threading
# the substitution through, and return the final substitution or `None` if any
# pair fails. This is exactly what matching a rule's whole body requires, and
# the threading is where hand-written provers go wrong.

# %%
def unify_all(pairs, subst=None):
    """Unify every pair in sequence, threading the substitution. None on failure."""
    # TODO: fold mgu over the pairs, stopping at the first failure
    return None


# %%
@checker("Exercise 4.6 — unify_all")
def check_ex6():
    X, Y, Z = Var("X"), Var("Y"), Var("Z")
    yield "no pairs", unify_all([]), {}
    yield "one pair", unify_all([(X, "a")]), {X: "a"}
    yield ("later pairs see earlier bindings",
           unify_all([(X, Y), (Y, "bob")]), {X: Y, Y: "bob"})
    yield ("…and the result substitutes correctly",
           fol.substitute(X, unify_all([(X, Y), (Y, "bob")]) or {}), "bob")
    yield ("an inconsistency anywhere fails",
           unify_all([(X, "a"), (X, "b")]), None)
    yield ("failure is detected even at the end",
           unify_all([(X, "a"), (Y, "b"), (parse("p(c)"), parse("q(c)"))]), None)
    yield ("structures across pairs",
           unify_all([(parse("p(X)"), parse("p(f(Y))")), (Y, "a")]),
           {X: parse("f(Y)"), Y: "a"})


check_ex6()

# %% [markdown]
# ---
# ## Project — a mini-Prolog
#
# Put it together into a working engine.
#
# ```python
# solve(rules, goals, subst=None, depth=20) -> iterator of substitutions
# ```
#
# * `rules` is a list of `Rule`; `goals` a list of terms to prove together.
# * Yield one substitution per successful proof — a **generator**, so callers
#   can stop after the first answer instead of paying for all of them.
# * No goals left → yield the current substitution.
# * Otherwise: substitute into the first goal, then for **every** rule whose
#   freshly-renamed head unifies with it, recurse on `rule.body + rest`.
# * `depth` bounds the recursion so a runaway program terminates. Return
#   nothing when it is exhausted.
#
# Then write:
#
# ```python
# ask(rules, goal_text) -> list of {Var: term} for the goal's own variables
# ```
#
# which parses `goal_text`, runs `solve`, and reports bindings for the
# variables *in the query* — not the internal renamed ones, which no caller
# should ever see.
#
# **Write-up questions.** Answer these in the final cell:
#
# 1. Remove the renaming (use `rule` instead of `rename_rule(rule, …)`) and
#    run the `ancestor` queries. What breaks, and precisely why?
# 2. Swap the order of the two `add` clauses in `PEANO_SOURCE` and re-run the
#    "? + ? = 4" query. What changes, and what does that tell you about
#    depth-first search as a proof strategy?
# 3. `solve` yields answers lazily. Which of the queries below would be
#    expensive or non-terminating if it returned a list instead?

# %%
FAMILY_SOURCE = """
    parent(bob, alice).
    parent(alice, carol).
    parent(carol, dave).
    parent(bob, edward).
    ancestor(X, Y) :- parent(X, Y).
    ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
"""

PEANO_SOURCE = """
    add(z, Y, Y).
    add(s(X), Y, s(Z)) :- add(X, Y, Z).
"""

_fresh = iter(range(1, 10 ** 9))


def solve(rules, goals, subst=None, depth=20):
    """Yield one substitution per proof of all `goals`."""
    # TODO: base case (no goals) -> yield subst. Otherwise substitute into
    # goals[0], and for each rule whose renamed head unifies, recurse on
    # rule.body + goals[1:] with the extended substitution and depth - 1.
    return
    yield  # keeps this a generator while it is still a stub


def ask(rules, goal_text, depth=20):
    """Parse a query, prove it, and report bindings for its own variables."""
    # TODO: parse, run solve, and project each answer onto the query's
    # variables using fol.substitute
    return None


# %%
@checker("Project 4 — mini-Prolog")
def check_project():
    family_rules = parse_program(FAMILY_SOURCE)
    peano_rules = parse_program(PEANO_SOURCE)
    W, A, B, R = Var("W"), Var("A"), Var("B"), Var("R")

    yield ("a ground fact holds",
           ask(family_rules, "parent(bob, alice)"), [{}])
    yield ("a false fact does not",
           ask(family_rules, "parent(alice, bob)"), [])
    yield ("direct children",
           sorted(a[W] for a in ask(family_rules, "parent(bob, W)")),
           ["alice", "edward"])
    yield ("ancestors, via the recursive rule",
           sorted(a[W] for a in ask(family_rules, "ancestor(bob, W)")),
           ["alice", "carol", "dave", "edward"])
    yield ("…and backwards",
           sorted(a[W] for a in ask(family_rules, "ancestor(W, dave)")),
           ["alice", "bob", "carol"])
    yield ("a leaf has no descendants",
           ask(family_rules, "ancestor(dave, W)"), [])

    yield ("Peano addition forwards",
           from_peano(ask(peano_rules, "add(s(s(z)), s(z), R)")[0][R]), 3)
    yield ("…and backwards",
           from_peano(ask(peano_rules, "add(s(s(z)), R, s(s(s(s(z)))))")[0][R]), 2)
    yield ("…and in both directions at once",
           sorted((from_peano(a[A]), from_peano(a[B]))
                  for a in ask(peano_rules, "add(A, B, s(s(s(s(z)))))")),
           [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)])

    yield ("solve is a generator, not a list",
           hasattr(solve(family_rules, [parse("parent(bob, W)")]), "__next__"),
           True)
    first = next(solve(family_rules, [parse("ancestor(bob, W)")]), None)
    yield "…and its first answer arrives without computing the rest", (
        first is not None), True

    yield ("the depth bound stops runaway recursion",
           ask(parse_program("loop(X) :- loop(X)."), "loop(a)", depth=8), [])

    yield ("answers mention only the query's variables",
           all(set(a) <= {W} for a in ask(family_rules, "ancestor(bob, W)")), True)
    yield ("…and bind them to ground terms",
           all(fol.is_ground(v) for a in ask(family_rules, "ancestor(bob, W)")
               for v in a.values()), True)


check_project()

# %%
# A query report, once the project works.
if ask(parse_program(FAMILY_SOURCE), "parent(bob, alice)") is not None:
    rows = []
    for source, query in [(FAMILY_SOURCE, "ancestor(bob, W)"),
                          (FAMILY_SOURCE, "ancestor(W, dave)"),
                          (FAMILY_SOURCE, "ancestor(dave, W)"),
                          (PEANO_SOURCE, "add(A, B, s(s(s(z))))")]:
        answers = ask(parse_program(source), query)
        shown = "; ".join(subst_str(a) for a in answers[:6]) or "no."
        rows.append((query, len(answers), shown))
    print(table(rows, ["query", "answers", "bindings"], align="llr"))

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * J. A. Robinson, "A Machine-Oriented Logic Based on the Resolution
#   Principle" (1965) — unification and resolution in one paper.
# * A. Colmerauer & P. Roussel, "The Birth of Prolog" (1993) — how the
#   language happened.
# * L. Sterling & E. Shapiro, *The Art of Prolog* (1994) — the book to read
#   next if this module appealed to you.
# * S. Russell & P. Norvig, *AIMA* ch. 8–9 — first-order logic and inference.
# * A. Church (1936), A. Turing (1936) — first-order validity is undecidable.
# * L. Pan et al., "Logic-LM: Empowering Large Language Models with Symbolic
#   Solvers" (2023) — the LLM-writes-the-clauses pipeline, measured.
#
# **Next:** Module 5 turns the arrow around. Instead of chasing a goal
# backwards, run the rules forwards from what you know — production systems,
# and the expert systems that first had to explain themselves.
