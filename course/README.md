# Reasoning & System 2 — from Classical Methods to Language Models

A twelve-module Jupyter course on how machines deliberate: propositional and
first-order logic, SAT, production systems, search, constraint satisfaction,
planning, probabilistic inference, decision theory — and, at every step, the
line from each classical method to the LLM reasoning technique that
rediscovered it.

Every module ends with **five or six exercises** and **one project**. Every
exercise self-tests in the notebook. Nothing outside the Python standard
library is required, anywhere.

## Why this course sits in this repo

The parent repo is a study (see [`../PLAN.md`](../PLAN.md)) of whether
supervising a small language model on *intermediate reasoning steps* changes
how it acquires an algorithm — the `color_cube_rotation` task, with paired
chain-of-thought, answer-only and Program-of-Thoughts training targets. The
datasets are here; the ideas behind them are what this course teaches.

So the course uses the repo's own material as its running example. Module 1
measures memorisation against algorithm on `data/*.jsonl`; Module 6 runs A*
over `data_gen/cube.py`; Module 12 grades generated programs with
`data_gen/pot_executor.py` and reproduces the analyses `PLAN.md` §5 describes
— all on a laptop, with no GPU and no model download.

## Getting started

```bash
pip install jupyterlab      # the only dependency, and only to *view* notebooks
cd course
make lab                    # or: jupyter lab notebooks
```

Open `notebooks/01_system_2.ipynb` and Run All. A fresh notebook runs top to
bottom without errors: the exercise stubs return `None`, and their checkers
report what is still missing rather than raising.

No Jupyter? The notebooks are generated from plain Python sources, so
`python3 src/01_system_2.py` runs the same lecture in a terminal.

## How a module works

1. **Lecture** — motivation, history, formalism, then the method built up in
   runnable cells with worked examples.
2. **Bridge to language models** — what this classical idea became, and which
   modern paper it turns into.
3. **Exercises** — five or six stubs. Fill one in, re-run the checker cell
   under it, get a line-by-line report:
   ```
   ✔ Exercise 3.4 — unit_propagate  --  7/7 checks passed
   ```
4. **Project** — one larger build per module, with acceptance tests and a
   write-up cell. Projects accumulate: Module 1's evaluation harness is what
   Module 12's capstone reports through.

Reference solutions live in [`solutions/`](solutions), one file per module.
They are also what `make verify` runs, so every exercise is guaranteed
solvable and every checker is guaranteed to accept a correct answer.

## Syllabus

| # | Module | Project |
|---|---|---|
| 1 | **What System 2 buys you** — deliberation as computation that scales with difficulty; accuracy-vs-depth as the measurement that separates algorithm from lookup | A trace-aware evaluation harness: per-depth accuracy, step accuracy, first-wrong-step, and a count of answers that were right for the wrong reasons |
| 2 | **Propositional logic** — models, entailment, validity; why "plausible" and "follows from" are different claims | A Knights-and-Knaves solver that answers by entailment and refuses to guess when the puzzle is undetermined |
| 3 | **SAT: CNF, resolution, DPLL** — inference as search for a contradiction; why propagation, not search, is what made solvers fast | An instrumented DPLL, a graph-colouring encoder, and a measurement of what each prune is worth |
| 4 | **First-order logic** — objects, relations, variables, substitution, unification | A mini-Prolog: most general unifier, backtracking resolution, and a readable proof trace |
| 5 | **Production systems** — forward chaining, conflict resolution, and the expert systems that first had to explain themselves | A rule engine that answers "why?" and "how?" from its own derivation |
| 6 | **State-space search** — BFS, uniform cost, iterative deepening, A*, admissible heuristics | A* over the cube-rotation task: find a rotation sequence reaching a target configuration |
| 7 | **Adversarial and anytime search** — minimax, alpha-beta, Monte-Carlo tree search | MCTS over a tree of partial solutions — Tree-of-Thoughts with the LLM removed |
| 8 | **Constraint satisfaction** — backtracking, forward checking, AC-3, variable and value ordering | A CSP solver strong enough for the Zebra puzzle, reporting what propagation saved |
| 9 | **Planning** — STRIPS, progression search, relaxed-plan heuristics | A blocks-world planner and, just as important, a plan validator |
| 10 | **Probabilistic reasoning** — Bayes, conditional independence, Bayesian networks, variable elimination, sampling | An inference engine, exact and approximate, on a diagnosis network |
| 11 | **Decisions and metareasoning** — expected utility, value iteration, value of information, anytime algorithms, verifiers | A gridworld MDP solver and a controller that decides *how long to think* per problem |
| 12 | **Capstone: verification and Program-of-Thoughts** — the classical/neural bridge | A hybrid solver for `color_cube_rotation`: answer-only vs. checked chain-of-thought vs. generated-and-executed program, evaluated on held-out chain lengths |

Modules 1–3 stand alone. Module 4 assumes 2–3; 6 assumes 1; 7 assumes 6;
8 assumes 3; 11 assumes 10; 12 assumes 1, 6 and 11.

## Layout

```
course/
  notebooks/   the course — 01_*.ipynb … 12_*.ipynb          <- open these
  csai/        shared helpers, standard library only
    check.py     the exercise self-test harness
    trace.py     Trace/Step recording and trace comparison
    logic.py     propositional logic, CNF, DPLL
    render.py    text tables, bar charts, trees, grids
    data.py      loaders for ../data and bridges to ../data_gen
  solutions/   reference solutions, m01.py … m12.py
  src/         percent-format sources the notebooks are generated from
  tools/       build.py, verify.py, run_notebook.py
```

## Working on the course

```bash
make build     # src/*.py  ->  notebooks/*.ipynb
make verify    # run every notebook twice: as a learner sees it, and solved
make check     # both
```

`make verify` executes each notebook's code cells in a fresh interpreter,
first with the stubs (nothing may raise) and then with `solutions/mNN.py`
injected (every check must pass). It needs no Jupyter — which is also why the
notebooks are generated from `src/` rather than edited directly. Edit
`src/NN_name.py`, run `make build`, commit both.

## A note on the data

Module 1 has you check the parent repo's splits for leakage, and on the data
currently checked in it finds some: `data/val.jsonl` and
`data/test_seen.jsonl` are drawn from seed ranges that overlap `train`'s, so
every one of their problems also appears in the training set. The course
works either way — the exercise reports what it finds — and
`data/test_extrapolate.jsonl` is unaffected, which is why the capstone relies
on it. If the splits are regenerated with disjoint seeds, Module 1's
leakage cell will simply report `CLEAN`.
