# %% [markdown]
# # Module 5 — Production Systems, Forward Chaining, and Explanation
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Choose between forward and backward chaining for a given problem, and say
#    why.
# 2. Implement forward chaining to a fixpoint, including refractoriness.
# 3. Name the conflict-resolution strategies a production system needs, and
#    show that the answer depends on them when rules are not monotone.
# 4. Make matching fast with indexing and semi-naive evaluation, and *measure*
#    what each is worth.
# 5. Build a system that explains itself — "how do you know?", "what did that
#    let you conclude?", "which of my inputs actually mattered?" — and say what
#    separates a real justification from a plausible story.
#
# **Prerequisites:** Module 4 (terms, unification, definite clauses).
#
# **Time:** ~70 minutes plus exercises.

# %% [markdown]
# ## 1. Turning the arrow around
#
# Module 4 chased goals backwards: *to prove this, what would suffice?*
# Forward chaining goes the other way: *given what I know, what follows?*
#
# Same rules, same unification, opposite control strategy — and the choice
# matters enormously in practice:
#
# | | backward chaining | forward chaining |
# |---|---|---|
# | driven by | the question | the data |
# | derives | only what the goal needs | everything derivable |
# | good when | one specific query, huge fact base | facts arrive over time, many queries |
# | bad when | the same subgoal is re-proved constantly | most consequences are irrelevant |
# | natural home | Prolog, theorem provers | expert systems, monitoring, databases, business rules |
#
# The forward direction is the one that got deployed. From the mid-1970s to
# the late 1980s, "AI" in industry very largely *meant* a forward-chaining
# production system: MYCIN diagnosing bacterial infections (Shortliffe, 1976),
# R1/XCON configuring VAX computers for DEC — reportedly saving tens of
# millions of dollars a year, on around 10,000 rules. The architecture is
# still everywhere; it is just called a business rules engine now.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import time

from csai import fol, rules as rules_lib
from csai.check import checker
from csai.fol import Rule, Var, parse, parse_program, parse_rule, term_str
from csai.render import table

print("ready")

# %% [markdown]
# ## 2. The recognise–act cycle
#
# A production system has three parts:
#
# 1. **Working memory** — the facts currently believed.
# 2. **Production rules** — `IF conditions THEN conclusion`.
# 3. **The interpreter**, looping forever over three steps:
#
# > **match** — find every rule whose conditions are satisfied by working
# > memory, with the bindings that satisfy them (the *conflict set*);
# > **resolve** — choose one;
# > **act** — add its conclusion to working memory.
#
# Newell and Simon proposed this in *Human Problem Solving* (1972) as a model
# of cognition, not as software architecture — the claim being that human
# deliberate thought *is* a recognise–act loop over long-term rules and a small
# working memory. Whether or not that holds up as psychology, it turned out to
# be an excellent way to build systems.
#
# Our rules are the definite clauses of Module 4, read left to right.

# %%
ANIMALS = parse_program("""
    % --- classification ---
    mammal(X)    :- has_hair(X).
    mammal(X)    :- gives_milk(X).
    bird(X)      :- has_feathers(X).
    bird(X)      :- flies(X), lays_eggs(X).

    % --- diet and body plan ---
    carnivore(X) :- mammal(X), eats_meat(X).
    carnivore(X) :- mammal(X), has_pointed_teeth(X), has_claws(X).
    ungulate(X)  :- mammal(X), has_hooves(X).

    % --- species ---
    cheetah(X)   :- carnivore(X), tawny(X), has_dark_spots(X).
    tiger(X)     :- carnivore(X), tawny(X), has_black_stripes(X).
    giraffe(X)   :- ungulate(X), has_long_neck(X), has_dark_spots(X).
    zebra(X)     :- ungulate(X), has_black_stripes(X).
    penguin(X)   :- bird(X), does_not_fly(X), swims(X).
    albatross(X) :- bird(X), flies_well(X).
""")

OBSERVATIONS = [parse(f) for f in [
    "has_hair(stripes)", "eats_meat(stripes)", "tawny(stripes)",
    "has_black_stripes(stripes)",
    "gives_milk(spot)", "has_pointed_teeth(spot)", "has_claws(spot)",
    "tawny(spot)", "has_dark_spots(spot)",
    "has_hair(longlegs)", "has_hooves(longlegs)", "has_long_neck(longlegs)",
    "has_dark_spots(longlegs)",
    "has_feathers(pengo)", "does_not_fly(pengo)", "swims(pengo)",
]]

print(f"{len(ANIMALS)} rules, {len(OBSERVATIONS)} observations")

# %% [markdown]
# ## 3. Running it to a fixpoint
#
# Match every rule against everything known; add whatever is new; repeat until
# a round adds nothing. Termination is guaranteed here because no rule invents
# new terms — the set of derivable facts is finite (this is the *Datalog*
# restriction, and it is what separates a decidable rule language from full
# Prolog).

# %%
stats = {}
derivation = rules_lib.forward_chain(ANIMALS, OBSERVATIONS, stats=stats)

given = {f for f in OBSERVATIONS}
derived = [f for f in derivation.facts if f not in given]
print(f"{len(given)} given -> {len(derived)} derived in "
      f"{stats['rounds']} rounds\n")
for f in derived:
    print(" ", term_str(f))

# %% [markdown]
# Note what did *not* happen: nothing was asked. The system was handed
# observations and produced a full classification of every animal at once. If
# a seventeenth observation arrives, one more round extends the conclusions
# incrementally. That is why this architecture suits monitoring, alerting and
# configuration — settings where the data comes to you.
#
# ### Refractoriness
#
# One detail matters enormously: a rule instantiation that has already fired
# must not fire again. Without that check the loop never terminates, because
# every round rederives everything. Our implementation gets this from
# recording a justification per *fact* and skipping facts already known — the
# same effect, at fact granularity.

# %% [markdown]
# ## 4. Conflict resolution
#
# When several rules match at once, which fires first? For pure definite
# clauses it does not matter — the fixpoint is unique, so the order affects
# only the path. The moment a system can *retract* facts, or attaches
# certainty factors, or has rules with side effects, order decides the answer.
# Classic strategies:
#
# | strategy | rule |
# |---|---|
# | **refractoriness** | never fire the same instantiation twice (mandatory, not optional) |
# | **recency** | prefer rules matching the most recently added facts — keeps a train of thought going |
# | **specificity** | prefer the rule with more conditions — the special case beats the general one |
# | **priority / salience** | an explicit number the knowledge engineer sets |
# | **textual order** | first rule in the file wins; simple and shockingly common |
#
# Specificity is the interesting one, because it is how a rule base encodes
# exceptions: `bird(X) → flies(X)`, overridden by
# `bird(X), penguin(X) → ¬flies(X)`. That is **non-monotonic** reasoning —
# learning more makes you conclude less — and it is exactly what classical
# logic forbids. Every real expert system needed it, and the theory that grew
# up to justify it (default logic, circumscription, the closed-world
# assumption) is one of the great sprawling literatures of classical AI.

# %%
# Specificity in action, done by hand: the general rule, then the exception.
general = parse_rule("flies(X) :- bird(X).")
exception = parse_rule("grounded(X) :- bird(X), penguin(X).")

birds = [parse("bird(pengo)"), parse("penguin(pengo)"), parse("bird(al)")]
d = rules_lib.forward_chain([general, exception], birds)
print("derived:", [term_str(f) for f in d.facts if f not in birds])
print("\nBoth flies(pengo) and grounded(pengo) are derived. Monotone logic")
print("cannot let the second retract the first — which is precisely why")
print("production systems added conflict resolution outside the logic.")

# %% [markdown]
# ## 5. Making the match fast
#
# Forgy's observation, building OPS5 in the late 1970s: production systems
# spend around **90% of their time in matching**, not in firing. Each round
# re-tests every rule against a working memory that has barely changed.
#
# Two fixes, both switchable in `csai.rules.forward_chain` so you can measure
# them:
#
# * **Indexing.** Bucket facts by predicate, and by their first argument when
#   it is known. A goal `carnivore(X)` with `X` already bound to `stripes`
#   then looks in one bucket instead of scanning working memory. This is the
#   idea behind RETE's alpha memories (Forgy, 1982).
# * **Semi-naive evaluation.** Require every match to use at least one fact
#   derived in the *previous* round. Any match that doesn't was already found
#   earlier, so looking again is pure waste. This is the standard Datalog
#   optimisation, and it is what makes recursive rules affordable.
#
# The benchmark below is transitive closure over a chain — `path/2` defined
# recursively, which is where naive evaluation hurts most.

# %%
CLOSURE = parse_program("""
    path(X, Y) :- edge(X, Y).
    path(X, Z) :- edge(X, Y), path(Y, Z).
""")

rows = []
for n in (10, 20, 30):
    chain = [parse(f"edge(n{i}, n{i + 1})") for i in range(n)]
    for indexed, semi in [(True, True), (True, False), (False, True), (False, False)]:
        st = {}
        t0 = time.perf_counter()
        d = rules_lib.forward_chain(CLOSURE, chain, indexed=indexed,
                                    semi_naive=semi, stats=st)
        rows.append((n, "yes" if indexed else "no", "yes" if semi else "no",
                     len(d.facts), st["match_attempts"],
                     f"{time.perf_counter() - t0:.3f}"))
print(table(rows, ["chain", "indexed", "semi-naive", "facts", "match attempts",
                   "seconds"], align="rccrrr"))

# %% [markdown]
# Same answer every time — `facts` is constant down each block — for two
# orders of magnitude less work. That is the shape of nearly every optimisation
# in this course: the *result* is fixed by the specification, and engineering
# decides only what it costs. Keep the two apart in your head, and you can
# optimise fearlessly.

# %% [markdown]
# ## 6. Explanation
#
# MYCIN's most influential feature was not its diagnoses. It was that a
# physician could ask **WHY** — why are you asking me this? — and **HOW** —
# how did you conclude that? — and get an answer built from the rules that
# actually fired. Doctors would not act on a recommendation they could not
# interrogate, and the system's designers understood that early.
#
# In a forward-chaining system, explanation is nearly free if you record, for
# each derived fact, the rule that produced it and the premises it used.
# `csai.rules` calls that a `Justification`.

# %%
tiger = parse("tiger(stripes)")
print(rules_lib.how(tiger, derivation.justifications))

print("\nAnd downstream — what did one observation let us conclude?")
for f in rules_lib.consequences(parse("has_hair(stripes)"),
                                derivation.justifications):
    print("  ", term_str(f))

# %% [markdown]
# Three different questions, all answerable from the same record:
#
# * **How do you know `tiger(stripes)`?** Walk the justification tree down to
#   the given facts. This is a *proof*.
# * **What did `has_hair(stripes)` buy us?** Walk it upwards.
# * **Which observations actually mattered?** Harder, and the most useful:
#   remove one given fact, re-run, and see whether the conclusion survives.
#   That is a **counterfactual** explanation, and the project builds it.
#
# The distinction to hold on to: a justification is a *causal* record — those
# rules, on those premises, are what produced the fact, and re-running the
# system reproduces it. It is not a story reconstructed afterwards to sound
# convincing. Keep that distinction in mind for the next section, because it
# is the whole point.

# %% [markdown]
# ## 7. Bridge to language models
#
# Expert systems collapsed in the late 1980s, and the cause has a name: the
# **knowledge acquisition bottleneck**. Every rule had to be elicited from a
# human expert, hand-encoded, and maintained. A thousand-rule base was a
# multi-year project, brittle at the edges of its competence, and it could not
# learn from data. Machine learning won because it removed exactly that cost.
#
# But look at what was traded away, because it is being rebuilt right now:
#
# | expert system | large language model |
# |---|---|
# | knowledge is explicit, inspectable, editable | knowledge is distributed in weights |
# | one rule can be fixed in isolation | fixing one behaviour perturbs others |
# | conclusions come with a derivation, by construction | conclusions come with prose that may or may not describe the process |
# | fails loudly outside its rule base | fails fluently outside its competence |
# | cannot learn from data | learns from nothing else |
#
# **The crucial row is the third one.** A chain of thought looks like the
# derivation tree you printed above. It is not one — not automatically. The
# justification tree is *causal*: those premises and that rule produced that
# fact, and you can re-run the system to confirm it. A model's stated
# reasoning is generated text; it correlates with the computation that
# produced the answer, sometimes strongly, sometimes not at all. Module 1
# measured exactly this gap — a buggy solver answered nearly 200 problems
# correctly with traces that had already gone wrong.
#
# Which points at the productive combination, and it is the one this course
# keeps arriving at from different directions:
#
# * **Ground the claims.** Retrieval-augmented generation with citations is a
#   justification structure: this claim, from that source. Weaker than a proof
#   — the model still writes the sentence — but checkable, which nothing about
#   the raw generation is.
# * **Rules as guardrails.** A small hand-written rule base over a model's
#   output catches the cases you can specify, at expert-system reliability,
#   without needing to specify everything.
# * **Let the model write the rules.** The knowledge acquisition bottleneck
#   was a *transcription* cost. A model that drafts candidate rules from
#   documents, for a human to approve and an engine to run, attacks precisely
#   the cost that killed the field — and keeps the derivation.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — match one goal
#
# Write `match_one(goal, facts)` returning the list of substitutions under
# which `goal` unifies with some fact — one per matching fact, in fact order.
# Use `fol.unify`.

# %%
def match_one(goal, facts):
    """List of substitutions unifying `goal` with each fact it matches."""
    # TODO: unify the goal against every fact, keeping the successes
    return None


# %%
@checker("Exercise 5.1 — match_one")
def check_ex1():
    facts = [parse("p(a)"), parse("p(b)"), parse("q(a)"), parse("p(a, b)")]
    X = Var("X")
    yield "a variable goal matches each p/1", match_one(parse("p(X)"), facts), [
        {X: "a"}, {X: "b"}]
    yield "a ground goal matches exactly", match_one(parse("p(a)"), facts), [{}]
    yield "no match", match_one(parse("r(X)"), facts), []
    yield "arity is respected", match_one(parse("p(X, Y)"), facts), [
        {X: "a", Var("Y"): "b"}]
    yield "empty fact base", match_one(parse("p(X)"), []), []


check_ex1()

# %% [markdown]
# ### Exercise 2 — match a whole rule body
#
# Write `match_body(goals, facts, subst=None)`: every substitution satisfying
# **all** the goals simultaneously, as a list. Bindings made by an earlier goal
# must constrain later ones — that threading is the whole exercise.
#
# <details><summary>Hint</summary>
#
# Recursion. No goals left → `[subst]`. Otherwise substitute the current
# bindings into the first goal, find each fact it unifies with, and recurse on
# the rest with the extended substitution, collecting all results.
# </details>

# %%
def match_body(goals, facts, subst=None):
    """All substitutions satisfying every goal in `goals` against `facts`."""
    # TODO: recurse over the goals, threading the substitution
    return None


# %%
@checker("Exercise 5.2 — match_body")
def check_ex2():
    facts = [parse("p(a)"), parse("p(b)"), parse("q(b)"), parse("q(c)")]
    X, Y = Var("X"), Var("Y")
    yield "no goals -> the empty substitution", match_body([], facts), [{}]
    yield "one goal", match_body([parse("q(X)")], facts), [{X: "b"}, {X: "c"}]
    yield ("a shared variable joins the goals",
           match_body([parse("p(X)"), parse("q(X)")], facts), [{X: "b"}])
    yield ("independent variables give the cross product",
           len(match_body([parse("p(X)"), parse("q(Y)")], facts) or []), 4)
    yield ("an unsatisfiable goal kills the whole body",
           match_body([parse("p(X)"), parse("r(X)")], facts), [])
    yield ("agrees with csai.rules",
           [s for s, _ in rules_lib.match_goals(
               (parse("p(X)"), parse("q(X)")), facts)],
           match_body([parse("p(X)"), parse("q(X)")], facts))


check_ex2()

# %% [markdown]
# ### Exercise 3 — one round of firing
#
# Write `fire_once(rules, facts)` returning the list of **new ground facts**
# derivable in a single pass — no duplicates, none already in `facts`, in the
# order they were derived.

# %%
def fire_once(rules, facts):
    """New ground facts derivable in one pass over the rules."""
    # TODO: for each rule, for each body match, instantiate the head; keep the
    # ground ones that are genuinely new
    return None


# %%
@checker("Exercise 5.3 — fire_once")
def check_ex3():
    rs = parse_program("mammal(X) :- has_hair(X).  pet(X) :- mammal(X), tame(X).")
    facts = [parse("has_hair(rex)"), parse("tame(rex)")]
    yield ("one round derives only what one step allows",
           fire_once(rs, facts), [parse("mammal(rex)")])
    yield ("the next round reaches the rest",
           fire_once(rs, facts + [parse("mammal(rex)")]), [parse("pet(rex)")])
    yield ("a fixpoint derives nothing",
           fire_once(rs, facts + [parse("mammal(rex)"), parse("pet(rex)")]), [])
    yield "no rules, no derivations", fire_once([], facts), []
    yield ("no duplicates within a round",
           fire_once(parse_program("a(X) :- b(X).  a(X) :- c(X)."),
                     [parse("b(k)"), parse("c(k)")]), [parse("a(k)")])


check_ex3()

# %% [markdown]
# ### Exercise 4 — run to a fixpoint
#
# Write `saturate(rules, facts, max_rounds=50)` returning **all** facts, given
# and derived, with the given ones first and derived ones in derivation order.

# %%
def saturate(rules, facts, max_rounds=50):
    """Everything derivable, given first, then derived in order."""
    # TODO: repeat fire_once until a round adds nothing
    return None


# %%
@checker("Exercise 5.4 — saturate")
def check_ex4():
    rs = parse_program("mammal(X) :- has_hair(X).  pet(X) :- mammal(X), tame(X).")
    facts = [parse("has_hair(rex)"), parse("tame(rex)")]
    got = saturate(rs, facts)
    yield "given facts come first", (got or [])[:2], facts
    yield "…then the derived ones", (got or [])[2:], [
        parse("mammal(rex)"), parse("pet(rex)")]
    yield ("agrees with csai.rules on the animals",
           set(saturate(ANIMALS, OBSERVATIONS) or []),
           set(rules_lib.forward_chain(ANIMALS, OBSERVATIONS).facts))
    yield ("terminates on recursion",
           len(saturate(CLOSURE, [parse(f"edge(n{i}, n{i + 1})")
                                  for i in range(4)]) or []), 4 + 10)
    yield "no rules -> just the facts", saturate([], facts), facts


check_ex4()

# %% [markdown]
# ### Exercise 5 — record provenance
#
# Write `derive_with_reasons(rules, facts)` returning a dict mapping each fact
# to `(rule, premises)` — with `(None, ())` for a given fact, and `premises` a
# **tuple** of the facts the rule matched. Where a fact is derivable more than
# one way, keep the first derivation found.

# %%
def derive_with_reasons(rules, facts):
    """{fact: (rule, premises)}; given facts map to (None, ())."""
    # TODO: saturate, but record the rule and the matched premises each time
    return None


# %%
@checker("Exercise 5.5 — derive_with_reasons")
def check_ex5():
    rs = parse_program("mammal(X) :- has_hair(X).  pet(X) :- mammal(X), tame(X).")
    facts = [parse("has_hair(rex)"), parse("tame(rex)")]
    got = derive_with_reasons(rs, facts) or {}
    yield "given facts are marked as given", got.get(parse("has_hair(rex)")), (None, ())
    yield ("a derived fact names its rule",
           (got.get(parse("mammal(rex)")) or (None,))[0], rs[0])
    yield ("…and its premises",
           (got.get(parse("mammal(rex)")) or (None, None))[1],
           (parse("has_hair(rex)"),))
    yield ("a two-premise rule records both",
           (got.get(parse("pet(rex)")) or (None, None))[1],
           (parse("mammal(rex)"), parse("tame(rex)")))
    yield "every fact is accounted for", set(got), set(saturate(rs, facts) or [])


check_ex5()

# %% [markdown]
# ### Exercise 6 — what does it rest on?
#
# Write `supports(fact, reasons)` returning the **set** of *given* facts that
# `fact` ultimately depends on, by walking the provenance to its leaves.
# A given fact supports itself.

# %%
def supports(fact, reasons):
    """Set of given facts that `fact` ultimately rests on."""
    # TODO: recurse through premises down to the facts whose rule is None
    return None


# %%
@checker("Exercise 5.6 — supports")
def check_ex6():
    reasons = derive_with_reasons(ANIMALS, OBSERVATIONS) or {}
    yield ("a given fact supports itself",
           supports(parse("has_hair(stripes)"), reasons),
           {parse("has_hair(stripes)")})
    yield ("a one-step conclusion",
           supports(parse("mammal(stripes)"), reasons),
           {parse("has_hair(stripes)")})
    yield ("a deep conclusion reaches only observations",
           supports(parse("tiger(stripes)"), reasons),
           {parse("has_hair(stripes)"), parse("eats_meat(stripes)"),
            parse("tawny(stripes)"), parse("has_black_stripes(stripes)")})
    yield ("…all of them given",
           all(reasons[f][0] is None
               for f in (supports(parse("tiger(stripes)"), reasons) or set())), True)
    yield ("an unrelated animal contributes nothing",
           any("spot" in term_str(f)
               for f in (supports(parse("tiger(stripes)"), reasons) or set())), False)
    yield "an unknown fact rests on nothing", supports(parse("dragon(x)"), reasons), set()


check_ex6()

# %% [markdown]
# ---
# ## Project — an expert system shell that explains itself
#
# Wrap the machinery in a class a domain expert could actually use.
#
# ```python
# class ExpertSystem:
#     def __init__(self, rules)
#     def run(self, facts)          -> list of all facts, given first
#     def how(self, fact)           -> str: indented derivation tree
#     def consequences(self, fact)  -> set of facts derived using it
#     def essential(self, fact)     -> set of given facts the conclusion needs
#     def diagnose(self, facts, predicates) -> {individual: conclusion}
# ```
#
# * `run` saturates and stores the provenance; the other methods work off the
#   last run.
# * `how` prints the fact, then its rule, then each premise indented below it,
#   recursively; a given fact is annotated `(given)`.
# * `essential(fact)` is the counterfactual one, and the reason this project
#   exists: a given fact is **essential** if removing it from the inputs and
#   re-running loses `fact` entirely. Note this is *not* the same as
#   `supports` — a conclusion reachable two independent ways may rest on
#   several facts of which none is individually essential. Your checker will
#   catch you if you conflate them.
# * `diagnose(facts, predicates)` runs the system and reports, for each
#   individual mentioned, which of the listed predicates it satisfies.
#
# **Write-up questions:**
#
# 1. Give an input for the animal rules where `supports` and `essential`
#    differ, and explain what that difference means to someone asking "why did
#    you say that?".
# 2. `essential` costs one full re-run per given fact. When is that
#    affordable, and what would you do instead if it weren't?
# 3. What could this system say if asked about an animal it cannot classify,
#    and what would it take to say something useful? (Look at what is in
#    working memory when no species rule fires.)

# %%
class ExpertSystem:
    """A forward-chaining rule engine that can justify its conclusions."""

    def __init__(self, rules):
        self.rules = list(rules)
        self.facts = []
        self.reasons = {}

    def run(self, facts):
        """Saturate from `facts`, recording provenance. Returns all facts."""
        # TODO: store self.facts and self.reasons; return the fact list
        return None

    def how(self, fact, indent=0):
        """Indented derivation tree for `fact`."""
        # TODO: "fact   (given)" for a given fact; otherwise the rule and,
        # indented one level, the how() of each premise
        return None

    def consequences(self, fact):
        """Set of facts derived using `fact`, directly or indirectly."""
        # TODO: walk the provenance upwards
        return None

    def essential(self, fact):
        """Given facts without which `fact` would not be derivable."""
        # TODO: for each given fact, re-run without it and check
        return None

    def diagnose(self, facts, predicates):
        """{individual: predicate} for each individual a listed rule classifies."""
        # TODO: run, then collect facts whose functor is in `predicates`
        return None


# %%
@checker("Project 5 — expert system shell")
def check_project():
    es = ExpertSystem(ANIMALS)
    all_facts = es.run(OBSERVATIONS)

    yield "run returns every fact", (
        set(all_facts) if all_facts else None), set(
        rules_lib.forward_chain(ANIMALS, OBSERVATIONS).facts)
    yield "…given ones first", (all_facts or [])[:len(OBSERVATIONS)], OBSERVATIONS

    yield ("it identifies the tiger", parse("tiger(stripes)") in (all_facts or []), True)
    yield ("and the cheetah", parse("cheetah(spot)") in (all_facts or []), True)
    yield ("and the giraffe", parse("giraffe(longlegs)") in (all_facts or []), True)
    yield ("and the penguin", parse("penguin(pengo)") in (all_facts or []), True)
    yield ("without inventing a zebra",
           parse("zebra(longlegs)") in (all_facts or []), False)

    explanation = es.how(parse("tiger(stripes)")) or ""
    yield "how() names the fact", "tiger(stripes)" in explanation, True
    yield "…and the intermediate step", "carnivore(stripes)" in explanation, True
    yield "…and reaches a given fact", "(given)" in explanation, True
    yield "…and indents the premises", any(
        line.startswith("  ") for line in explanation.splitlines()), True
    yield ("a given fact explains itself in one line",
           len((es.how(parse("has_hair(stripes)")) or "").splitlines()), 1)

    yield ("consequences of an observation",
           es.consequences(parse("has_hair(stripes)")),
           {parse("mammal(stripes)"), parse("carnivore(stripes)"),
            parse("tiger(stripes)")})
    yield ("a conclusion has no consequences here",
           es.consequences(parse("tiger(stripes)")), set())

    yield ("every observation about the tiger is essential to it",
           es.essential(parse("tiger(stripes)")),
           {parse("has_hair(stripes)"), parse("eats_meat(stripes)"),
            parse("tawny(stripes)"), parse("has_black_stripes(stripes)")})

    # Two independent routes to mammal(dual): neither premise is essential,
    # though both support it. This is the distinction the write-up asks about.
    dual = [parse("has_hair(dual)"), parse("gives_milk(dual)")]
    es2 = ExpertSystem(ANIMALS)
    es2.run(dual)
    yield ("with two routes to a conclusion, nothing is essential",
           es2.essential(parse("mammal(dual)")), set())
    yield ("…even though supports() lists a premise",
           len(supports(parse("mammal(dual)"), es2.reasons) or set()) > 0, True)

    yield ("diagnose reports one species per animal",
           es.diagnose(OBSERVATIONS, {"tiger", "cheetah", "giraffe", "zebra",
                                      "penguin", "albatross"}),
           {"stripes": "tiger", "spot": "cheetah", "longlegs": "giraffe",
            "pengo": "penguin"})


check_project()

# %%
# The full report, once the project works.
es = ExpertSystem(ANIMALS)
if es.run(OBSERVATIONS) is not None:
    species = {"tiger", "cheetah", "giraffe", "zebra", "penguin", "albatross"}
    rows = []
    for who, what in sorted(es.diagnose(OBSERVATIONS, species).items()):
        essential = es.essential(parse(f"{what}({who})"))
        rows.append((who, what, len(es.consequences(parse(f"has_hair({who})"))),
                     ", ".join(sorted(term_str(f) for f in essential))))
    print(table(rows, ["individual", "conclusion", "consequences of has_hair",
                       "essential evidence"]))
    print()
    print(es.how(parse("giraffe(longlegs)")))

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's three questions.

# %% [markdown]
# ---
# ## Further reading
#
# * A. Newell & H. Simon, *Human Problem Solving* (1972) — production systems
#   as a theory of cognition.
# * E. Shortliffe, *Computer-Based Medical Consultations: MYCIN* (1976), and
#   B. Buchanan & E. Shortliffe, *Rule-Based Expert Systems* (1984) — including
#   the chapters on explanation, which read as strikingly current.
# * C. Forgy, "Rete: A Fast Algorithm for the Many Pattern/Many Object Pattern
#   Match Problem" (1982).
# * J. McDermott, "R1: A Rule-Based Configurer of Computer Systems" (1982) —
#   the commercial success story.
# * R. Reiter, "A Logic for Default Reasoning" (1980) — making non-monotonic
#   reasoning respectable.
# * M. Turpin et al., "Language Models Don't Always Say What They Think"
#   (2023) — the modern version of the justification-versus-story problem.
#
# **Next:** Module 6 leaves logic behind for a while. Reasoning as *search*
# through a space of states — BFS, uniform cost, A*, and heuristics — applied
# to the cube task from Module 1.
