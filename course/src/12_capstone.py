# %% [markdown]
# # Module 12 — Capstone: Verification, Program-of-Thoughts, and the Bridge
#
# *Reasoning & System 2: from classical methods to language models*
#
# ---
#
# **You will be able to:**
#
# 1. Assemble the course's pieces into one harness that compares reasoning
#    strategies on a held-out benchmark.
# 2. Reproduce the analyses this repo's `PLAN.md` §5 specifies — accuracy by
#    chain length, step-level faithfulness, first-wrong-step — without a GPU.
# 3. Generate **Program-of-Thoughts** solutions and grade them by *executing*
#    them in the repo's real sandboxed interpreter.
# 4. Show what a step-level verifier is worth, in accuracy and in calls.
# 5. Say precisely what this experiment does and does not establish.
#
# **Prerequisites:** Module 1 (traces, depth curves), Module 6 (the cube state
# space), Module 9 (validation), Module 11 (compute budgets).
#
# **Time:** ~90 minutes.
#
# ---
#
# ### Read this before you read the results
#
# The reasoning "models" in this notebook are **simulations**: small stochastic
# processes with error rates I chose, standing in for a language model. The
# numbers below are therefore *not evidence about any real model*, and no
# claim here should be cited as if they were.
#
# What is real:
#
# * the problems, from `data/*.jsonl`;
# * the ground-truth cube simulator every trace is checked against;
# * the Program-of-Thoughts arm's **interpreter** — generated programs are
#   really executed by `data_gen/pot_executor.py`, in a subprocess, and graded
#   on what they print;
# * the measurement harness, which is the actual deliverable.
#
# Swapping a real model in is one function: replace a `Strategy`'s
# `solve(example, rng)` with a call to your model. Everything downstream —
# the depth curves, the step-faithfulness analysis, the compute accounting —
# works unchanged. That separation is the point of building it this way.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
_course = next(p for p in [_here, *_here.parents] if (p / "csai").is_dir())
if str(_course) not in sys.path:
    sys.path.insert(0, str(_course))

import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field

from csai import data, search
from csai.check import checker
from csai.render import bar_chart, table
from csai.trace import Trace, diff_traces

SIDES = data.SIDES
executor = data.pot_executor()

print("splits:", data.available_splits())
print("PoT interpreter:", executor.__file__)

# %% [markdown]
# ## 1. The course, in one table
#
# | module | idea | where it appears below |
# |---|---|---|
# | 1 | System 2 = computation that scales with difficulty | the whole design; the depth curves |
# | 1 | traces, step accuracy, first divergence | how every arm is graded |
# | 2 | entailment vs. plausibility | why a fluent trace is not evidence |
# | 3 | a certificate is checkable in isolation | the executed program's output |
# | 4 | delegation to an engine with real variables | the PoT arm |
# | 5 | provenance vs. a story told afterwards | step checking rather than answer checking |
# | 6 | the cube's state space is 24 states, diameter 3 | why "chain length" ≠ "problem depth" |
# | 7 | search is cheap; the value function is not | why the verifier arm wins |
# | 8 | propagation beats guessing in order | why left-to-right commits early |
# | 9 | finding is hard, checking is linear | the step verifier |
# | 10 | sampling estimates, and converges slowly | why more samples is the weak lever |
# | 11 | thinking is an action with a cost | the calls-per-problem column |

# %% [markdown]
# ## 2. The setup
#
# A **strategy** takes a problem and returns an answer, a trace (possibly
# empty), and what it cost in model calls.

# %%
@dataclass
class Attempt:
    """One strategy's attempt at one problem."""

    answer: str | None
    trace: Trace | None = None
    calls: int = 1
    note: str = ""


@dataclass
class Strategy:
    """A named way of solving a problem."""

    name: str
    solve: object                      # (example, rng) -> Attempt
    description: str = ""


def gold_answer(example):
    return example["answer"]


def true_step(state, side):
    """The correct cube state after one rotation — the ground-truth simulator."""
    return data.simulate(state, [side])[-1]


def perturb(state, rng):
    """A plausible slip: two faces swapped."""
    a, b = rng.sample(list(state), 2)
    slipped = dict(state)
    slipped[a], slipped[b] = slipped[b], slipped[a]
    return slipped


# %% [markdown]
# ## 3. Five strategies
#
# | strategy | what it stands for | error model |
# |---|---|---|
# | **classical solver** | the algorithm, executed exactly | none — this is the ceiling |
# | **answer only** | one forward pass, no working | success falls geometrically with chain length |
# | **chain of thought** | writing the state out at each step | one chance of a state-update slip per step |
# | **checked chain of thought** | the same, with a step verifier and retries | same slips, caught and redone |
# | **program of thought** | writing a program and running it | one chance of a *transcription* slip per line |
#
# The difference between the last two error models is the substantive claim
# built into this simulation, and it is worth stating explicitly: **writing
# `cube.rotate_to_top("back")` is an easier act than working out what the cube
# looks like afterwards.** Transcription is local and checkable against the
# sentence in front of you; state update is a computation you must carry. So
# the per-line error rate for a program is set well below the per-step error
# rate for a natural-language trace. Whether that gap is real, and how large it
# is, is exactly what the study in `PLAN.md` exists to measure.

# %%
STEP_ERROR = 0.12        # chance of botching one state update in prose
LINE_ERROR = 0.02        # chance of botching one line of transcribed code
ANSWER_SKILL = 0.55      # per-rotation odds of guessing the whole thing right


def classical(example, rng):
    """Module 1's simulator. The ceiling, and it is not a model."""
    state = data.initial_state(example)
    tr = Trace(initial=state)
    for side in example["metadata"]["rotations"]:
        state = true_step(state, side)
        tr.step(f"rotate {side} to top", state)
    return Attempt(state[example["metadata"]["target_side"]], tr.finish(
        state[example["metadata"]["target_side"]]))


def answer_only(example, rng):
    """No trace, no working: one shot at the answer."""
    depth = data.num_rotations(example)
    if rng.random() < ANSWER_SKILL ** depth:
        return Attempt(example["answer"])
    return Attempt(rng.choice(list(data.initial_state(example).values())))


def chain_of_thought(example, rng, step_error=STEP_ERROR):
    """Write the cube state after each rotation, slipping occasionally."""
    state = data.initial_state(example)
    tr = Trace(initial=state)
    for side in example["metadata"]["rotations"]:
        state = true_step(state, side)
        if rng.random() < step_error:
            state = perturb(state, rng)
        tr.step(f"rotate {side} to top", state)
    answer = state[example["metadata"]["target_side"]]
    return Attempt(answer, tr.finish(answer))


def checked_chain_of_thought(example, rng, step_error=STEP_ERROR,
                             max_retries=4):
    """The same, but a verifier checks each step and makes it try again.

    The verifier is **local**: it asks only whether the stated new state is
    the correct rotation of the *previous* state. It never looks at the answer
    key, so this is a checker you could actually build — and because the
    initial state is given, every step being locally right makes the whole
    trace right. That is the same property Module 9's plan validator had.
    """
    state = data.initial_state(example)
    tr = Trace(initial=state)
    calls = 1
    for side in example["metadata"]["rotations"]:
        correct_next = true_step(state, side)
        candidate = correct_next
        for attempt in range(max_retries + 1):
            candidate = correct_next
            if rng.random() < step_error:
                candidate = perturb(candidate, rng)
            if candidate == correct_next:
                break                       # the verifier accepts it
            if attempt < max_retries:
                calls += 1                  # pay for another go
        state = candidate
        tr.step(f"rotate {side} to top", state)
    answer = state[example["metadata"]["target_side"]]
    return Attempt(answer, tr.finish(answer), calls)


def pot_program(example, wrong_line=None, rng=None):
    """The Program-of-Thoughts solution, as source text.

    Targets the `Cube` API in `data_gen/pot_library.py` — the same one the
    repo's PoT training data uses.
    """
    state = data.initial_state(example)
    meta = example["metadata"]
    args = ", ".join(f'{side}="{state[side]}"' for side in SIDES)
    lines = [f"cube = Cube({args})"]
    for i, side in enumerate(meta["rotations"]):
        used = side
        if i == wrong_line:
            used = rng.choice([s for s in SIDES if s not in (side, "top")])
        lines.append(f'cube.rotate_to_top("{used}")')
    lines.append(f'print(cube.{meta["target_side"]})')
    return "\n".join(lines)


def program_of_thought(example, rng, line_error=LINE_ERROR):
    """Write a program, then *run it*. The interpreter does the reasoning."""
    wrong = None
    for i in range(data.num_rotations(example)):
        if rng.random() < line_error:
            wrong = i
            break
    program = pot_program(example, wrong, rng)
    answer = executor.run_program(program)      # a real subprocess, really run
    return Attempt(answer, note="execution failed" if answer is None else "")


STRATEGIES = [
    Strategy("classical solver", classical, "the algorithm, exactly"),
    Strategy("answer only", answer_only, "one shot, no working"),
    Strategy("chain of thought", chain_of_thought, "state written out per step"),
    Strategy("checked chain of thought", checked_chain_of_thought,
             "…with a step verifier and retries"),
    Strategy("program of thought", program_of_thought,
             "generate code, execute it"),
]

# A single problem, five ways.
example = data.load_split("test_extrapolate", limit=1)[0]
print(example["question"][:120].replace("\n", " ") + " …")
print(f"\ngold answer: {example['answer']}   "
      f"({data.num_rotations(example)} rotations)\n")
for strategy in STRATEGIES:
    attempt = strategy.solve(example, random.Random(0))
    mark = "✔" if attempt.answer == example["answer"] else "✘"
    print(f"{mark} {strategy.name:<26} -> {str(attempt.answer):<10} "
          f"({attempt.calls} call{'s' if attempt.calls != 1 else ''})")

print("\nthe program that was actually executed:\n")
print(pot_program(example))

# %% [markdown]
# ## 4. The experiment
#
# Now run every strategy over both held-out splits and report what `PLAN.md`
# §5 asks for.

# %%
SAMPLE = 120       # per split; raise for tighter estimates, at a cost in seconds


def run(strategy, examples, seed=0):
    """Evaluate one strategy, recording everything the analysis needs."""
    correct = lucky = calls = failures = 0
    by_depth: dict = {}
    step_scores: list = []
    first_wrong: Counter = Counter()

    for ex in examples:
        rng = random.Random(f"{seed}:{strategy.name}:{ex['id']}")
        attempt = strategy.solve(ex, rng)
        depth = data.num_rotations(ex)
        gold = data.gold_states(ex)
        ok = attempt.answer == ex["answer"]

        correct += ok
        calls += attempt.calls
        failures += attempt.answer is None
        by_depth.setdefault(depth, [0, 0])
        by_depth[depth][0] += ok
        by_depth[depth][1] += 1

        if attempt.trace is not None:
            d = diff_traces(attempt.trace, gold)
            step_scores.append(d.step_accuracy)
            first_wrong[d.first_divergence or 0] += 1
            lucky += ok and d.first_divergence is not None

    n = len(examples)
    return {
        "n": n,
        "accuracy": correct / n,
        "by_depth": {k: c / t for k, (c, t) in sorted(by_depth.items())},
        "calls_per_problem": calls / n,
        "step_accuracy": statistics.mean(step_scores) if step_scores else None,
        "first_wrong": dict(sorted(first_wrong.items())),
        "lucky": lucky,
        "execution_failures": failures,
    }


results = {}
for split in ("test_seen", "test_extrapolate"):
    examples = data.load_split(split, limit=SAMPLE)
    depths = sorted({data.num_rotations(e) for e in examples})
    rows = []
    t0 = time.perf_counter()
    for strategy in STRATEGIES:
        report = run(strategy, examples)
        results[(split, strategy.name)] = report
        rows.append([strategy.name]
                    + [f"{report['by_depth'][d]:.0%}" for d in depths]
                    + [f"{report['accuracy']:.0%}",
                       f"{report['calls_per_problem']:.2f}"])
    print(f"{split}  ({len(examples)} problems, "
          f"{time.perf_counter() - t0:.1f}s)")
    print(table(rows, ["strategy"] + [f"k={d}" for d in depths]
                + ["overall", "calls/problem"], align="l" + "r" * (len(depths) + 2)))
    print()

# %% [markdown]
# Read the two tables side by side, because the comparison is the result.
#
# * **The classical solver is flat at 100%** on chain lengths it was never
#   tuned for. That is what having an algorithm looks like, and it is the only
#   row here that is not a simulation.
# * **Answer-only collapses with depth.** Its accuracy at six rotations is
#   near the rate you get by naming a colour you can see. A fixed amount of
#   computation cannot track a growing amount of state — Module 1's prediction,
#   holding.
# * **Chain of thought decays, more slowly.** Writing the state out means the
#   computation grows with the problem, which is the entire mechanism. It still
#   decays, because per-step errors compound: `(1 − ε)ᵏ` falls however small
#   `ε` is.
# * **Program of thought decays more slowly still** — and note *why*, because
#   it is easy to get wrong. Delegating to an interpreter does not remove the
#   length dependence; it *lowers the per-line error rate*, since transcribing
#   a sentence into an API call is easier than computing a state update. Same
#   curve shape, gentler slope. Anyone claiming code execution "solves" length
#   generalisation should be asked for the depth curve.
# * **Checked chain of thought is flat at 100%**, at well under twice the
#   calls. The verifier turns a noisy reasoner into a reliable one.
#
# That last row is idealised — a perfect step verifier, available because this
# dataset ships the intermediate states. Take it as an upper bound on what
# step-level verification can buy, not as a forecast. The shape of the result,
# though, is the same one Module 7 found in tree search and Module 11 found in
# budget allocation, and it is the most consistent finding in the course.

# %% [markdown]
# ### Step-level faithfulness
#
# Final-answer accuracy is one bit about a process that took several steps.
# `PLAN.md` §5 asks for more: how much of the *reasoning* was right, and where
# it first went wrong.

# %%
rows = []
for strategy in STRATEGIES:
    report = results[("test_extrapolate", strategy.name)]
    if report["step_accuracy"] is None:
        rows.append((strategy.name, "—", "—", "—"))
        continue
    diverged = {k: v for k, v in report["first_wrong"].items() if k}
    modal = max(diverged, key=diverged.get) if diverged else "—"
    rows.append((strategy.name, f"{report['step_accuracy']:.0%}",
                 report["lucky"], modal))
print("test_extrapolate, 4-6 rotations:\n")
print(table(rows, ["strategy", "step accuracy", "right for the wrong reasons",
                   "modal first wrong step"], align="lrrr"))

report = results[("test_extrapolate", "chain of thought")]
print("\nwhere the chain of thought first goes wrong:")
print(bar_chart(
    [("never" if k == 0 else f"step {k}", v)
     for k, v in sorted(report["first_wrong"].items())],
    width=32, value_fmt="{:.0f}"))

# %% [markdown]
# The "right for the wrong reasons" column is the one worth staring at. Those
# are problems answered correctly by a trace that had already diverged from
# the truth — the trace is a story, not a derivation. Grade only the final
# answer and you cannot see them; grade the trace and they are obvious.
#
# The first-divergence histogram says something else again. Errors are spread
# across the steps rather than concentrated at the start, which is the
# signature of *accumulating* failure rather than misunderstanding the setup.
# A model that failed at step 1 every time would need a different fix
# entirely — and telling those two apart is exactly what this measurement is
# for.

# %% [markdown]
# ## 5. What is a call worth?
#
# Module 11's question, applied here. Checked chain of thought buys a large
# accuracy gain for extra calls. Is that a good trade — and where should extra
# calls go?

# %%
def sweep_retries(examples, retries_list):
    rows = []
    for retries in retries_list:
        strategy = Strategy(
            f"checked (≤{retries} retries)",
            lambda ex, rng, r=retries: checked_chain_of_thought(
                ex, rng, max_retries=r))
        report = run(strategy, examples)
        rows.append((retries, f"{report['accuracy']:.0%}",
                     f"{report['step_accuracy']:.0%}",
                     f"{report['calls_per_problem']:.2f}"))
    return rows


deep = data.load_split("test_extrapolate", limit=SAMPLE)
print("checked chain of thought, varying the retry budget:\n")
print(table(sweep_retries(deep, [0, 1, 2, 4, 8]),
            ["max retries per step", "accuracy", "step accuracy",
             "calls/problem"], align="rrrr"))
print("\nZero retries is plain chain of thought — the verifier runs but cannot")
print("act, so it costs one call and buys nothing. The first retry buys most")
print("of the remaining gain, and later retries buy almost nothing because a")
print("step has to slip several times in a row to survive them. Module 11's")
print("marginal-value rule, with real numbers attached.")

# %% [markdown]
# ## 6. What the course adds up to
#
# Twelve modules, and they keep arriving at the same three things.
#
# ### 1. Deliberation is computation that scales with the problem
#
# Module 1 defined it, and every module since has been an instance. Search
# expands more nodes for harder instances. Propagation runs more rounds. Value
# iteration sweeps until convergence. Chain of thought emits more tokens. The
# common failure is always the same: a fixed amount of computation meeting a
# growing amount of state, and the depth curve bending downwards.
#
# ### 2. Verification is cheap and generation is expensive, and that asymmetry
# is the most exploitable fact in the field
#
# It appeared, independently, in:
#
# * Module 2 — checking a model against a KB is linear; finding one is not;
# * Module 3 — the empty clause is a certificate anyone can check;
# * Module 7 — a value function beat a better search algorithm, 30/30 against
#   4/30;
# * Module 9 — plans are PSPACE-hard to find and linear to validate;
# * Module 11 — the verifier was worth forty points and the allocation policy
#   two;
# * Module 12 — the checked arm above.
#
# Six measurements, one conclusion: **where a cheap correct checker exists,
# build the system around it.** Where none exists, the first engineering
# question is whether one can be manufactured — a test suite, a type checker,
# an interpreter, a simulator, a second cheaper model. That question is worth
# more than a better prompt.
#
# ### 3. Match the control flow to the problem's structure
#
# Sequential generation suits sequential problems. Constraint problems have no
# reading order (Module 8). Subgoals interact, so you cannot solve them
# separately and concatenate (Module 9). Belief must be revised when a
# competing explanation arrives, which requires not having collapsed onto one
# hypothesis already (Module 10). Backtracking needs a frontier, and a linear
# chain of text has nowhere to keep one (Modules 6 and 7).
#
# None of these is a knowledge problem, and none is fixed by knowing more.
# They are control-flow problems, and classical AI spent forty years working
# out the control flows. That is the reason to learn this material now: not
# nostalgia, but that the field is rediscovering it, and it is much cheaper to
# recognise an idea than to reinvent it.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 1 — write the program
#
# Write `build_program(example)` returning the Program-of-Thoughts source for
# an example: a `Cube(...)` constructor with all six faces as keyword
# arguments in `data.SIDES` order, one `cube.rotate_to_top("side")` per
# rotation, and a final `print(cube.<target>)`.
#
# The checker **executes** what you produce.

# %%
def build_program(example):
    """PoT source text solving `example` against the Cube API."""
    # TODO: constructor line, one rotate per rotation, then print the target
    return None


# %%
@checker("Exercise 12.1 — build_program")
def check_ex1():
    src = build_program(example)
    yield "returns source text", isinstance(src, str), True
    yield "constructs a cube", "Cube(" in (src or ""), True
    yield ("one rotation call per rotation",
           (src or "").count("rotate_to_top"), data.num_rotations(example))
    yield "and prints something", "print(" in (src or ""), True
    yield ("…and running it gives the right answer",
           executor.run_program(src) if src else None, example["answer"])

    for ex in data.load_split("test_seen", limit=4):
        yield (f"correct on problem {ex['id']}",
               executor.run_program(build_program(ex)), ex["answer"])
    deep_one = next(e for e in data.load_split("test_extrapolate", limit=40)
                    if data.num_rotations(e) == 6)
    yield ("…and at six rotations",
           executor.run_program(build_program(deep_one)), deep_one["answer"])


check_ex1()

# %% [markdown]
# ### Exercise 2 — grade by execution
#
# Write `grade(program, expected)` returning `1.0` if running `program` prints
# `expected`, `0.01` if it prints something else, and `0.0` if it fails to run
# at all. That is reasoning-gym's own scoring convention, and
# `data.pot_executor().score_program` implements it — call it, and satisfy
# yourself the three cases really are distinguished.

# %%
def grade(program, expected):
    """1.0 correct, 0.01 wrong, 0.0 failed to run."""
    # TODO: use executor.score_program
    return None


# %%
@checker("Exercise 12.2 — grade")
def check_ex2():
    yield ("a correct program scores 1",
           grade(build_program(example), example["answer"]), 1.0)
    yield ("a wrong answer scores 0.01",
           grade('print("magenta")', "not_magenta"), 0.01)
    yield ("a crash scores 0", grade("raise ValueError()", "anything"), 0.0)
    yield ("so does a program that prints nothing", grade("x = 1", "anything"), 0.0)
    yield ("…and one that never terminates is not fatal either",
           grade("while True:\n    pass", "anything"), 0.0)
    yield ("the API really is available to the program",
           grade('cube = Cube(top="a", right="b", front="c", left="d", '
                 'back="e", bottom="f")\ncube.rotate_to_top("front")\n'
                 'print(cube.top)', "c"), 1.0)


check_ex2()

# %% [markdown]
# ### Exercise 3 — a noisy chain of thought
#
# Write `noisy_trace(example, rng, step_error)` returning a `Trace` of the
# simulation in which each step has probability `step_error` of being
# perturbed (use `perturb`), with the error carried forward. Finish the trace
# with the answer read off the final state.

# %%
def noisy_trace(example, rng, step_error):
    """A Trace of the simulation, slipping with probability `step_error` per step."""
    # TODO: true_step, then perturb with probability step_error; carry it forward
    return None


# %%
@checker("Exercise 12.3 — noisy_trace")
def check_ex3():
    clean = noisy_trace(example, random.Random(0), 0.0)
    yield "returns a Trace", isinstance(clean, Trace), True
    yield "one step per rotation", len(clean or []), data.num_rotations(example)
    yield ("with no noise it is the ground truth",
           (clean.states if clean else None), data.gold_states(example))
    yield ("…and the right answer", (clean.result if clean else None),
           example["answer"])

    always = noisy_trace(example, random.Random(1), 1.0)
    yield ("with certain noise, step 1 already diverges",
           diff_traces(always, data.gold_states(example)).first_divergence, 1)

    rates = []
    for seed in range(60):
        t = noisy_trace(example, random.Random(seed), 0.3)
        rates.append(diff_traces(t, data.gold_states(example)).identical)
    yield ("with 30% noise, some traces survive intact and some do not",
           0 < sum(rates) < len(rates), True)


check_ex3()

# %% [markdown]
# ### Exercise 4 — verify a trace
#
# Write `first_bad_step(trace, example)`: the 1-based index of the first step
# whose state differs from the ground truth, or `None`. This is Module 9's
# validator, applied to reasoning instead of to a plan.

# %%
def first_bad_step(trace, example):
    """1-based index of the first divergent step, or None if the trace is right."""
    # TODO: compare trace.states against data.gold_states(example)
    return None


# %%
@checker("Exercise 12.4 — first_bad_step")
def check_ex4():
    yield ("a correct trace has no bad step",
           first_bad_step(classical(example, None).trace, example), None)
    yield ("a trace that is wrong from the start",
           first_bad_step(noisy_trace(example, random.Random(1), 1.0), example), 1)
    yield ("agrees with csai.trace",
           first_bad_step(noisy_trace(example, random.Random(4), 0.4), example),
           diff_traces(noisy_trace(example, random.Random(4), 0.4),
                       data.gold_states(example)).first_divergence)

    # Hand-built: correct until step 2.
    gold = data.gold_states(example)
    tr = Trace(initial=data.initial_state(example))
    for i, state in enumerate(gold):
        tr.step("x", perturb(state, random.Random(0)) if i == 1 else state)
    yield "localises a mid-trace error", first_bad_step(tr, example), 2

    short = Trace(initial=data.initial_state(example))
    short.step("x", gold[0])
    yield ("a truncated trace diverges where it stops",
           first_bad_step(short, example), 2)


check_ex4()

# %% [markdown]
# ### Exercise 5 — the depth curve
#
# Write `depth_curve(strategy, examples, seed=0)` returning
# `{chain length: accuracy}`, keys ascending. This is the plot the whole
# course has been building toward.

# %%
def depth_curve(strategy, examples, seed=0):
    """{number of rotations: fraction correct}."""
    # TODO: bucket by data.num_rotations; run the strategy with a seeded rng
    return None


# %%
@checker("Exercise 12.5 — depth_curve")
def check_ex5():
    sample = data.load_split("test_seen", limit=180) + \
        data.load_split("test_extrapolate", limit=180)
    exact = depth_curve(STRATEGIES[0], sample)
    yield "covers every depth", sorted(exact or {}), [1, 2, 3, 4, 5, 6]
    yield "the classical solver is perfect everywhere", set(
        (exact or {}).values()), {1.0}

    shallow = depth_curve(STRATEGIES[1], sample)
    yield ("answer-only is much better at depth 1 than depth 6",
           (shallow or {})[1] > (shallow or {})[6] + 0.2, True)

    cot = depth_curve(STRATEGIES[2], sample)
    yield ("chain of thought beats answer-only at depth 6",
           (cot or {})[6] > (shallow or {})[6], True)
    yield ("…and is between 0 and 1 everywhere",
           all(0.0 <= v <= 1.0 for v in (cot or {}).values()), True)
    yield ("running it twice with the same seed gives the same curve",
           depth_curve(STRATEGIES[2], sample), cot)


check_ex5()

# %% [markdown]
# ---
# ## Project — the full comparison, and a claim you can defend
#
# ```python
# experiment(strategies, split, limit=120, seed=0) -> {name: report}
# ```
#
# Each report carries `"accuracy"`, `"by_depth"`, `"calls_per_problem"`,
# `"step_accuracy"` (or `None` for strategies that produce no trace),
# `"first_wrong"`, `"lucky"` and `"execution_failures"`.
#
# ```python
# summarise(results, depths) -> str   # the comparison table, as text
# ```
#
# Then answer these in the write-up. They are the questions the study in
# `PLAN.md` is actually asking, and you now have a harness that can address
# them.
#
# 1. **Which arm degrades least with depth, and is that because of the error
#    *rate* or the error *structure*?** Set `LINE_ERROR` equal to `STEP_ERROR`
#    and re-run. What survives of the program-of-thoughts advantage, and what
#    does that tell you about why it helps?
# 2. **What is a verifier worth here, in accuracy per extra call?** Compute
#    it, then compare against buying the same number of calls as independent
#    samples with a majority vote (Module 10's self-consistency). Which is the
#    better use of the budget, and does the answer depend on chain length?
# 3. **Module 6 showed every `test_extrapolate` problem is at most three
#    rotations from its start.** Design one additional split that would
#    separate "tracks a long description" from "executes a deep computation",
#    and say which arms you would expect to come apart on it.
# 4. **What would you have to change to run this against a real model?** Be
#    specific: name the function, the inputs it needs, and which parts of the
#    analysis would still work unchanged. Then name one number in your tables
#    that you would *not* expect to survive contact with a real model, and say
#    why.

# %%
def experiment(strategies, split, limit=120, seed=0):
    """Run every strategy over a split and collect the full report."""
    # TODO: load the split, run each strategy, and gather the eight fields
    return None


def summarise(results, depths):
    """The comparison table, as a string."""
    # TODO: one row per strategy: name, accuracy at each depth, overall, calls
    return None


# %%
@checker("Project 12 — the capstone experiment")
def check_project():
    got = experiment(STRATEGIES, "test_extrapolate", limit=60)
    yield "reports every strategy", sorted(got or {}), sorted(
        s.name for s in STRATEGIES)

    keys = sorted((got or {}).get("classical solver", {}))
    yield "with the required fields", keys, [
        "accuracy", "by_depth", "calls_per_problem", "execution_failures",
        "first_wrong", "lucky", "n", "step_accuracy"]

    exact = got["classical solver"]
    yield "the classical solver is exact", exact["accuracy"], 1.0
    yield "…at every depth", set(exact["by_depth"].values()), {1.0}
    yield "…with a perfect trace", exact["step_accuracy"], 1.0
    yield "…that never diverges", set(exact["first_wrong"]), {0}
    yield "…and is never merely lucky", exact["lucky"], 0

    shallow = got["answer only"]
    yield "answer-only does badly on deep problems", shallow["accuracy"] < 0.5, True
    yield "…and produces no trace to grade", shallow["step_accuracy"], None

    cot = got["chain of thought"]
    yield "chain of thought beats answer-only", cot["accuracy"] > shallow[
        "accuracy"], True
    yield "…and its trace is mostly right", cot["step_accuracy"] > 0.5, True
    yield ("…but sometimes right for the wrong reasons",
           cot["lucky"] > 0, True)

    checked = got["checked chain of thought"]
    yield "the verifier makes it reliable", checked["accuracy"] > 0.95, True
    yield "…for more calls", checked["calls_per_problem"] > cot[
        "calls_per_problem"], True
    yield "…but fewer than double", checked["calls_per_problem"] < 2.0, True

    pot = got["program of thought"]
    yield ("program of thought beats plain chain of thought at these depths",
           pot["accuracy"] > cot["accuracy"], True)
    yield "…and every program ran", pot["execution_failures"], 0
    yield "…in one call each", pot["calls_per_problem"], 1.0

    text = summarise(got, [4, 5, 6])
    yield "summarise returns text", isinstance(text, str), True
    yield "…naming every strategy", all(
        s.name in (text or "") for s in STRATEGIES), True
    yield "…with a row per strategy", len(
        [ln for ln in (text or "").splitlines() if ln.strip()]) >= len(STRATEGIES),\
        True


check_project()

# %%
# Your own version of the headline result.
if experiment(STRATEGIES, "test_seen", limit=10) is not None:
    for split, depths in [("test_seen", [1, 2, 3]),
                          ("test_extrapolate", [4, 5, 6])]:
        print(split)
        print(summarise(experiment(STRATEGIES, split, limit=SAMPLE), depths))
        print()

# %% [markdown]
# ### Write-up
#
# Replace this cell with your answers to the project's four questions.

# %% [markdown]
# ---
# ## Where to go next
#
# **In this repo.** `PLAN.md` describes the study this harness was built to
# support: LoRA fine-tuning of Qwen2.5 at three sizes on the CoT, answer-only
# and PoT splits, then exactly the analyses you just ran, with a real model in
# place of the simulations. The measurement code is done. Note the two findings
# the course turned up along the way — the seed overlap between `train` and the
# "held-out" splits (Module 1), and the fact that `test_extrapolate` varies
# description length rather than problem depth (Module 6) — both of which
# affect how those results should be read.
#
# **The classical material.** Russell & Norvig, *Artificial Intelligence: A
# Modern Approach* covers every module here properly and is the natural next
# book. For depth in one direction: Sterling & Shapiro on Prolog, the
# *Handbook of Satisfiability*, Dechter on constraints, Ghallab, Nau & Traverso
# on planning, Koller & Friedman on graphical models, Sutton & Barto on
# reinforcement learning.
#
# **The bridge.** The papers cited at the end of each module, read in the order
# the modules introduce them, make a reasonable syllabus of their own: Nye on
# scratchpads, Wei on chain of thought, Wang on self-consistency, Yao on
# tree-of-thoughts, Lightman on process supervision, Valmeekam on planning,
# Kambhampati on LLM-Modulo, Snell on test-time compute.
#
# **Your own version of this.** The most useful thing you can do with this
# harness is point it at a task you care about, with a real model, and ask the
# three questions the course keeps returning to. Does accuracy hold up as the
# problem gets deeper? Is there a cheap checker, and what does it buy? Does the
# control flow match the problem's structure? Those three questions are the
# course.
