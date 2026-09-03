# %% [markdown]
# # Module 2 — Propositional Logic: What "Follows From" Means
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Write formulas in a machine-readable syntax and evaluate them in a model.
# 2. Distinguish **valid**, **satisfiable** and **unsatisfiable** formulas, and
#    say which question you are actually asking in each case.
# 3. Implement **entailment** by model checking, and produce a counterexample
#    when entailment fails.
# 4. State the deduction theorem and explain why it turns "does this follow?"
#    into "is this unsatisfiable?" — the hinge the next module swings on.
# 5. Explain why "the model said something plausible" and "the conclusion
#    follows" are different claims, and build a checker for the second.
#
# **Prerequisites:** Module 1. Python tuples and recursion.
#
# **Time:** ~60 minutes plus exercises.

# %% [markdown]
# ## 1. The oldest idea in the course
#
# Module 1 gave us a way to *measure* reasoning: does accuracy hold up as
# problems get deeper? It said nothing about what makes an inference
# **correct**. For that we need the idea Aristotle started with around 350 BC
# and Boole (1854) and Frege (1879) made mechanical:
#
# > A conclusion **follows from** premises when there is no possible situation
# > in which the premises hold and the conclusion fails.
#
# Notice what this definition does *not* mention: how you found the
# conclusion, how confident you feel, or how many times you have seen
# something similar. Correctness is a property of the *relationship between
# premises and conclusion*, checkable independently of the process that
# produced it.
#
# That independence is the whole reason logic matters to this course. It gives
# you a **verifier**: something cheap that can check an answer without being
# able to produce it. Every later module leans on it — the SAT solver in
# Module 3, the constraint propagator in Module 8, the plan validator in
# Module 9, and the answer checker in the Module 12 capstone.
#
# We start with the simplest useful logic: propositions that are true or
# false, combined with *and*, *or*, *not*, *implies*, *if and only if*. No
# objects, no quantifiers, no time — those come in Module 4.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import itertools

from csai import logic
from csai.check import checker
from csai.logic import And, Iff, Implies, Not, Or, to_str
from csai.render import table

print("ready")

# %% [markdown]
# ## 2. Syntax
#
# A formula is one of:
#
# | form | example | meaning |
# |---|---|---|
# | a string | `"rain"` | a propositional symbol |
# | a bool | `True` | the constant ⊤ (or ⊥) |
# | `("not", f)` | `Not("rain")` | ¬rain |
# | `("and", *fs)` | `And("rain", "cold")` | rain ∧ cold |
# | `("or", *fs)` | `Or("rain", "snow")` | rain ∨ snow |
# | `("implies", a, b)` | `Implies("rain", "wet")` | rain → wet |
# | `("iff", a, b)` | `Iff("rain", "wet")` | rain ↔ wet |
#
# Tuples rather than a class hierarchy: formulas are then hashable,
# comparable, and printable with no machinery at all, which matters when you
# start putting them in sets (Module 3 does exactly that).

# %%
rain_makes_it_wet = Implies("rain", "wet")
sprinkler_story = And(Implies("sprinkler", "wet"), Not("rain"), "wet")

print(rain_makes_it_wet)
print(to_str(rain_makes_it_wet))
print()
print(sprinkler_story)
print(to_str(sprinkler_story))
print()
print("symbols:", logic.symbols(sprinkler_story))

# %% [markdown]
# ## 3. Semantics
#
# Syntax is inert until you say what the symbols *mean*. In propositional
# logic that is refreshingly cheap: a **model** is an assignment of `True` or
# `False` to every symbol. Nothing else. `"rain"` does not mean rain; it means
# whatever the model says.
#
# `evaluate(formula, model)` then walks the formula, and the connectives have
# exactly the truth tables you would write down — with one that catches
# everybody out.

# %%
model = {"rain": True, "wet": True, "sprinkler": False}
print("under", model)
print("  rain → wet :", logic.evaluate(rain_makes_it_wet, model))
print("  ¬rain ∧ wet:", logic.evaluate(And(Not("rain"), "wet"), model))

print("\ntruth table for p → q:")
rows = [(p, q, logic.evaluate(Implies("p", "q"), {"p": p, "q": q}))
        for p, q in itertools.product([False, True], repeat=2)]
print(table(rows, ["p", "q", "p → q"]))

# %% [markdown]
# The third row is the famous one: **`False → True` is `True`**, and so is
# `False → False`. "If the moon is cheese then 2 + 2 = 5" is a true statement
# of propositional logic.
#
# This is not a bug, and it is not deep. `p → q` is *defined* as `¬p ∨ q`; it
# claims only that you never get `p` true with `q` false. A promise with a
# condition that never fires is not broken. Material implication is a
# deliberately weak notion, and every attempt to make it match the English
# "if… then…" (relevance logics, counterfactuals, conditional probability)
# ends up somewhere much more complicated. Take the weak version; it composes.

# %% [markdown]
# ## 4. Three questions about a formula
#
# With models defined, there are exactly three interesting questions:
#
# | question | meaning | example |
# |---|---|---|
# | **valid** (a tautology) | true in *every* model | `p ∨ ¬p` |
# | **satisfiable** | true in *at least one* model | `p ∧ q` |
# | **unsatisfiable** (a contradiction) | true in *no* model | `p ∧ ¬p` |
#
# They are duals: `f` is valid exactly when `¬f` is unsatisfiable. That single
# observation is why an unsatisfiability checker — a SAT solver — is enough to
# answer all three, and it is what Module 3 builds.

# %%
for f in [Or("p", Not("p")),
          And("p", "q"),
          And("p", Not("p")),
          Implies(And("p", Implies("p", "q")), "q")]:
    kind = ("valid" if logic.is_valid(f)
            else "satisfiable" if logic.is_satisfiable(f)
            else "unsatisfiable")
    print(f"{to_str(f):<28} {kind}")

# %% [markdown]
# The last one is *modus ponens* written as a single formula, and it is valid.
# That is what "modus ponens is a sound rule" actually means: not that people
# find it convincing, but that no model makes its premises true and its
# conclusion false.

# %% [markdown]
# ## 5. Entailment, and how to check it
#
# Now the definition we started with, made precise:
#
# > **KB ⊨ α** ("KB entails α") when every model that satisfies KB also
# > satisfies α.
#
# The algorithm falls straight out of the definition: enumerate all models,
# keep the ones where KB is true, check α in each. This is **model checking**,
# and it is sound and complete — it always terminates with the right answer.

# %%
kb = [
    Implies("rain", "wet"),
    Implies("sprinkler", "wet"),
    Implies("wet", "slippery"),
    "rain",
]

for query in ["wet", "slippery", "sprinkler", Not("sprinkler")]:
    ok = logic.entails(kb, query)
    line = f"KB ⊨ {to_str(query):<12} {ok}"
    if not ok:
        witness = logic.counterexample(kb, query)
        true_syms = sorted(s for s, v in witness.items() if v)
        line += f"    counterexample: everything false except {true_syms}"
    print(line)

# %% [markdown]
# Look at the last two lines. The KB does not entail `sprinkler`, and it does
# not entail `¬sprinkler` either. Both are consistent with what is known — the
# grass is wet, and rain alone explains it, so the sprinkler is simply
# undetermined. **"I don't know" is a real answer**, and a logical system can
# say it precisely, which is exactly what a next-token predictor trained to
# always produce fluent text finds hardest.
#
# When entailment fails, the counterexample model *is* the explanation: here
# is a world where everything you told me holds and your conclusion doesn't.
# That is a far better error message than "no".
#
# ### The deduction theorem
#
# One rearrangement, and it is the hinge for the whole next module:
#
# > KB ⊨ α  if and only if  **KB ∧ ¬α is unsatisfiable**.
#
# If there is no world where the premises hold and the conclusion fails, then
# asserting the premises *and* the negated conclusion is contradictory. So a
# machine that only knows how to answer "is this set of constraints
# satisfiable?" can answer every question logic can ask. Proof by
# contradiction, mechanised.

# %%
alpha = "slippery"
print("by model checking :", logic.entails(kb, alpha))
print("by refutation     :", not logic.is_satisfiable(list(kb) + [Not(alpha)]))
print("…and for a non-consequence:")
print("by model checking :", logic.entails(kb, "sprinkler"))
print("by refutation     :", not logic.is_satisfiable(list(kb) + [Not("sprinkler")]))

# %% [markdown]
# ## 6. The wall
#
# Model checking is correct and hopeless. `n` symbols means `2**n` models, and
# the table below is the reason the rest of this course is largely about
# *avoiding* enumeration.

# %%
rows = []
for n in (10, 20, 30, 40, 60, 100):
    models = 2 ** n
    seconds = models / 1e9  # optimistically, a billion models a second
    if seconds < 60:
        cost = f"{seconds:.3g} s"
    elif seconds < 3.15e7:
        cost = f"{seconds / 3600:.3g} hours"
    else:
        cost = f"{seconds / 3.15e7:.3g} years"
    rows.append((n, f"{models:.3g}", cost))
print(table(rows, ["symbols", "models", "at 1e9 models/sec"], align="rrr"))

# %% [markdown]
# Satisfiability was the first problem proved NP-complete (Cook, 1971), so
# nobody expects to remove that exponent in the worst case. What *did* happen
# is one of the great engineering stories in computer science: modern SAT
# solvers routinely dispatch industrial problems with millions of variables,
# because real problems are not worst cases and propagation prunes ferociously.
# Module 3 builds the core of one.

# %% [markdown]
# ## 7. Bridge to language models
#
# **Plausible is not the same as entailed.** A language model produces the
# continuation its training distribution favours. Entailment asks whether
# *every* consistent world agrees. Those come apart in both directions, and
# each direction is a known failure mode:
#
# * *Plausible but not entailed* — the model answers `sprinkler` because
#   sprinklers usually explain wet grass. Fluent, useful, and not supported by
#   the premises. This is one honest description of a hallucination.
# * *Entailed but not plausible* — a valid conclusion phrased in a way that is
#   rare in text, which the model deprecates for exactly that reason. Long
#   chains where each step is individually unremarkable but the conclusion is
#   surprising are where models drop the thread.
#
# Three practical consequences run through the rest of the course:
#
# 1. **Verification is cheaper than generation, and that asymmetry is
#    exploitable.** Checking a candidate answer against a KB is linear; finding
#    it is exponential. Generate-and-check architectures — the LLM proposes, a
#    solver disposes — are the direct descendant of this observation.
# 2. **"I don't know" is representable.** Entailment distinguishes *false*
#    from *undetermined*. Calibration work is, in effect, trying to recover
#    that distinction statistically.
# 3. **Consistency is checkable.** Ask a model the same question two ways, get
#    `p` and `¬p`, and you need no ground truth to know something is wrong.
#    Self-consistency checks (Wang et al., 2022) and formal-verification
#    pipelines are built on this.

# %% [markdown]
# ---
# ## Exercises
#
# Reimplement the core of `csai.logic` yourself. The checkers compare your
# version against it, so you get an oracle for free. Solutions in
# `course/solutions/m02.py`.

# %% [markdown]
# ### Exercise 1 — evaluate
#
# Write `truth_value(formula, model)`. Handle `bool`, `str`, and the five
# connectives. Recursion; about a dozen lines.
#
# <details><summary>Hint</summary>
#
# `op, *args = formula` unpacks the tuple. `implies` is `not a or b`; `iff` is
# `a == b`; `and`/`or` take any number of arguments, so use `all(...)` and
# `any(...)` over a generator.
# </details>

# %%
def truth_value(formula, model):
    """Truth value of `formula` under `model` (a {symbol: bool} dict)."""
    # TODO: recurse over bool / str / ("not"|"and"|"or"|"implies"|"iff", ...)
    return None


# %%
@checker("Exercise 2.1 — truth_value")
def check_ex1():
    m = {"p": True, "q": False, "r": True}
    for f in [True, False, "p", "q",
              Not("p"), Not("q"),
              And("p", "r"), And("p", "q"), And(),
              Or("q", "r"), Or("q", Not("r")), Or(),
              Implies("p", "q"), Implies("q", "p"), Implies("q", "q"),
              Iff("p", "r"), Iff("p", "q"),
              And(Or("p", "q"), Not(And("q", "r"))),
              Implies(And("p", Implies("p", "q")), "q")]:
        yield to_str(f), truth_value(f, m), logic.evaluate(f, m)


check_ex1()

# %% [markdown]
# ### Exercise 2 — collect the symbols
#
# Write `atoms(formula)` returning a **sorted list** of the distinct symbols.
# `True`/`False` are not symbols.

# %%
def atoms(formula):
    """Sorted list of distinct propositional symbols in `formula`."""
    # TODO: walk the tree collecting strings into a set, then sort
    return None


# %%
@checker("Exercise 2.2 — atoms")
def check_ex2():
    yield "a symbol", atoms("p"), ["p"]
    yield "a constant has none", atoms(True), []
    yield "duplicates collapse", atoms(And("p", "p", "q")), ["p", "q"]
    yield "nested", atoms(Implies(Not("b"), Or("a", And("c", True)))), ["a", "b", "c"]
    yield "sorted", atoms(Or("z", "a", "m")), ["a", "m", "z"]
    big = And(Implies("rain", "wet"), Iff("sprinkler", Not("dry")))
    yield "agrees with csai.logic", atoms(big), logic.symbols(big)


check_ex2()

# %% [markdown]
# ### Exercise 3 — enumerate the models
#
# Write `models_of(symbols)` returning a **list** of all `2**n` assignments,
# each a dict. Order does not matter to the checker, but every assignment must
# appear exactly once.

# %%
def models_of(symbols):
    """All 2**n assignments of True/False to `symbols`, as a list of dicts."""
    # TODO: itertools.product([False, True], repeat=n) then zip with symbols
    return None


# %%
@checker("Exercise 2.3 — models_of")
def check_ex3():
    yield "no symbols -> one empty model", models_of([]), [{}]
    got = models_of(["p"])
    yield "one symbol -> two models", sorted((got or []), key=lambda m: m["p"]), [
        {"p": False}, {"p": True}]
    three = models_of(["a", "b", "c"])
    yield "three symbols -> eight models", len(three or []), 8
    yield "all distinct", len({tuple(sorted(m.items())) for m in (three or [])}), 8
    yield "every model assigns every symbol", all(
        set(m) == {"a", "b", "c"} for m in (three or [])), True
    yield "values are bools", all(
        isinstance(v, bool) for m in (three or []) for v in m.values()), True


check_ex3()

# %% [markdown]
# ### Exercise 4 — classify a formula
#
# Write `classify(formula)` returning one of the strings `"valid"`,
# `"contingent"` (satisfiable but not valid) or `"unsatisfiable"`. Use your own
# `truth_value`, `atoms` and `models_of`.

# %%
def classify(formula):
    """Return "valid", "contingent" or "unsatisfiable"."""
    # TODO: count how many models satisfy it, compare against how many exist
    return None


# %%
@checker("Exercise 2.4 — classify")
def check_ex4():
    yield "excluded middle", classify(Or("p", Not("p"))), "valid"
    yield "contradiction", classify(And("p", Not("p"))), "unsatisfiable"
    yield "a plain symbol", classify("p"), "contingent"
    yield "modus ponens", classify(
        Implies(And("p", Implies("p", "q")), "q")), "valid"
    yield "constant true", classify(True), "valid"
    yield "constant false", classify(False), "unsatisfiable"
    yield "de Morgan", classify(
        Iff(Not(And("p", "q")), Or(Not("p"), Not("q")))), "valid"
    yield "affirming the consequent is not valid", classify(
        Implies(And("q", Implies("p", "q")), "p")), "contingent"


check_ex4()

# %% [markdown]
# ### Exercise 5 — entailment
#
# Write `follows(kb, query)` where `kb` is a **list** of formulas: return
# `True` when every model satisfying all of `kb` also satisfies `query`.
#
# Careful with the symbol set: the query may mention symbols the KB never
# does. Enumerate over the union.

# %%
def follows(kb, query):
    """True when every model of all the formulas in `kb` satisfies `query`."""
    # TODO: enumerate models over the union of symbols; check the definition
    return None


# %%
@checker("Exercise 2.5 — follows")
def check_ex5():
    kb1 = [Implies("rain", "wet"), "rain"]
    yield "modus ponens", follows(kb1, "wet"), True
    yield "not the converse", follows([Implies("rain", "wet"), "wet"], "rain"), False
    yield "modus tollens", follows(
        [Implies("rain", "wet"), Not("wet")], Not("rain")), True
    yield "chaining", follows(
        [Implies("a", "b"), Implies("b", "c"), "a"], "c"), True
    yield "an empty KB entails only tautologies", follows([], Or("p", Not("p"))), True
    yield "…and not contingent claims", follows([], "p"), False
    yield "unseen symbols are undetermined", follows(kb1, "cold"), False
    yield "a contradictory KB entails everything", follows(
        ["p", Not("p")], "anything_at_all"), True
    yield "agrees with csai.logic", follows(kb1, "wet"), logic.entails(kb1, "wet")


check_ex5()

# %% [markdown]
# ### Exercise 6 — explain the failure
#
# Write `why_not(kb, query)`: return a model satisfying every formula in `kb`
# but falsifying `query`, or `None` if the query does follow. This is the error
# message a reasoning system owes its user.

# %%
def why_not(kb, query):
    """A model of `kb` that falsifies `query`, or None if kb entails query."""
    # TODO: the first model where the KB holds and the query doesn't
    return None


# %%
@checker("Exercise 2.6 — why_not")
def check_ex6():
    kb1 = [Implies("rain", "wet"), "rain"]
    yield "entailed -> None", why_not(kb1, "wet"), None
    w = why_not(kb1, "sprinkler")
    yield "not entailed -> a model", isinstance(w, dict), True
    yield "…that satisfies the KB", all(
        logic.evaluate(f, w) for f in kb1) if isinstance(w, dict) else False, True
    yield "…and falsifies the query", (
        logic.evaluate("sprinkler", w) is False) if isinstance(w, dict) else False, True
    yield "…assigning every relevant symbol", (
        set(w) >= {"rain", "wet", "sprinkler"}) if isinstance(w, dict) else False, True
    yield "contradictory KB -> None", why_not(["p", Not("p")], "q"), None


check_ex6()

# %% [markdown]
# ---
# ## Project — a Knights and Knaves solver
#
# Raymond Smullyan's island puzzles are propositional logic with a single,
# beautiful encoding trick. Every inhabitant is either a **knight**, who only
# says true things, or a **knave**, who only says false ones. You are told what
# people said; work out who is what.
#
# The trick: let symbol `"A"` mean *A is a knight*. Then A saying `S` is
# **exactly** the formula
#
# > `Iff("A", S)`
#
# — if A is a knight, `S` holds; if A is a knave, `S` fails. One line, and the
# puzzle becomes a satisfiability question.
#
# A puzzle is a dict mapping a speaker's name to the formula they assert:
#
# ```python
# {"A": Not("A")}                 # A says "I am a knave"
# {"A": And(Not("A"), Not("B"))}  # A says "we are both knaves"
# ```
#
# Implement three functions.
#
# | function | returns |
# |---|---|
# | `encode(puzzle)` | list of `Iff(speaker, statement)`, one per speaker, in the puzzle's key order |
# | `solutions(puzzle)` | every model consistent with the encoding, as a list of dicts over the puzzle's names, sorted by `sorted(model.items())` |
# | `identify(puzzle)` | `{name: "knight" \| "knave" \| "unknown"}`, or `{}` when the puzzle has no solutions |
#
# `identify` must use **entailment**, not "read it off the single model": a
# puzzle with two solutions that agree about C should still report C
# confidently. Return `{}` for a paradox — a puzzle with no models entails
# everything, and reporting that everybody is simultaneously a knight and a
# knave would be true but useless.

# %%
PUZZLES = {
    # A: "I am a knave."  -- the liar paradox; no consistent assignment.
    "paradox": {"A": Not("A")},
    # A: "We are both knaves."
    "both_knaves": {"A": And(Not("A"), Not("B"))},
    # A: "B is a knight."   B: "A and I are of opposite kinds."
    "opposites": {"A": "B", "B": Not(Iff("A", "B"))},
    # A: "B is a knight."  -- consistent either way; nothing is determined.
    "undetermined": {"A": "B"},
    # A: "At least one of us is a knave."  B says nothing.
    "at_least_one": {"A": Or(Not("A"), Not("B"))},
}


def encode(puzzle):
    """Turn {speaker: statement} into a list of propositional formulas."""
    # TODO: one Iff(speaker, statement) per entry, in the puzzle's key order
    return None


def solutions(puzzle):
    """All models of the encoding, over exactly the puzzle's names."""
    # TODO: enumerate models over the names; keep those satisfying every
    # formula from encode(); return them sorted by sorted(model.items())
    return None


def identify(puzzle):
    """{name: "knight"|"knave"|"unknown"}, or {} if the puzzle is a paradox."""
    # TODO: no solutions -> {}. Otherwise decide each name by entailment
    # against the encoding, and say "unknown" when neither direction follows.
    return None


# %%
@checker("Project 2 — Knights and Knaves")
def check_project():
    enc = encode(PUZZLES["both_knaves"])
    yield "encode returns one formula per speaker", len(enc or []), 1
    yield "…of the Iff(speaker, statement) shape", (enc or [None])[0], Iff(
        "A", And(Not("A"), Not("B")))

    yield "a paradox has no solutions", solutions(PUZZLES["paradox"]), []
    yield "…and identify says so", identify(PUZZLES["paradox"]), {}

    yield ("'we are both knaves' has one solution",
           solutions(PUZZLES["both_knaves"]), [{"A": False, "B": True}])
    yield ("…A is a knave and B a knight",
           identify(PUZZLES["both_knaves"]), {"A": "knave", "B": "knight"})

    yield ("'opposite kinds' makes them both knaves",
           identify(PUZZLES["opposites"]), {"A": "knave", "B": "knave"})

    und = solutions(PUZZLES["undetermined"])
    yield "an undetermined puzzle has two solutions", len(und or []), 2
    yield ("…and identify refuses to guess",
           identify(PUZZLES["undetermined"]), {"A": "unknown", "B": "unknown"})

    yield ("'at least one of us is a knave' pins both down",
           identify(PUZZLES["at_least_one"]), {"A": "knight", "B": "knave"})

    yield ("solutions are sorted deterministically",
           solutions(PUZZLES["undetermined"]),
           sorted(solutions(PUZZLES["undetermined"]) or [],
                  key=lambda m: sorted(m.items())))

    # Every reported answer must actually be entailed by the encoding.
    for name, verdict in (identify(PUZZLES["opposites"]) or {}).items():
        if verdict != "unknown":
            want = name if verdict == "knight" else Not(name)
            yield (f"'{name} is a {verdict}' is entailed, not guessed",
                   logic.entails(encode(PUZZLES["opposites"]), want), True)


check_project()

# %%
# A report over every puzzle, once the project works.
if identify(PUZZLES["both_knaves"]) is not None:
    rows = []
    for name, puzzle in PUZZLES.items():
        said = "; ".join(f"{who}: “{to_str(what)}”" for who, what in puzzle.items())
        verdicts = identify(puzzle)
        answer = ("paradox — no consistent assignment" if not verdicts
                  else ", ".join(
                      f"{k}: {v}" for k, v in sorted(verdicts.items())))
        rows.append((name, said, len(solutions(puzzle)), answer))
    print(table(rows, ["puzzle", "statements", "models", "conclusion"]))

# %% [markdown]
# ### Extension (optional, no checker)
#
# Add a third kind of islander: a **spy**, who may say anything at all. Can
# your encoding still express the puzzles? What happens to `identify` — and
# what does that tell you about the cost of admitting an agent whose
# statements carry no information?

# %% [markdown]
# ---
# ## Further reading
#
# * S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*,
#   ch. 7 — the standard treatment; this module follows its notation.
# * G. Boole, *An Investigation of the Laws of Thought* (1854) — the book that
#   made logic algebra.
# * R. Smullyan, *What Is the Name of This Book?* (1978) — the knights and
#   knaves puzzles, and a great deal more.
# * S. Cook, "The Complexity of Theorem-Proving Procedures" (1971) — SAT is
#   NP-complete.
# * X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning"
#   (2022) — consistency as a signal without ground truth.
#
# **Next:** Module 3 stops enumerating models and starts searching for a
# contradiction instead — CNF, resolution and DPLL.
