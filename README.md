# Qwen2.5 Reasoning Capacity on `color_cube_rotation`

Data-generation stage for a study of how model capacity (Qwen2.5 0.5B / 3B /
7B, fine-tuned with LoRA on a single RTX 4090) affects acquisition of a
spatial-reasoning algorithm, using reasoning-gym's
[`color_cube_rotation`](https://github.com/open-thought/reasoning-gym) task.

See **[PLAN.md](PLAN.md)** for the full research plan, LoRA configs, and
evaluation design, and **[course/](course/)** for a twelve-module Jupyter
course on classical deliberative reasoning — logic, SAT, search, constraints,
planning, probabilistic inference, decision theory — and the line from each
classical method to the LLM reasoning technique that rediscovered it. Its
capstone runs against the datasets below. Training hasn't started yet — this repo currently holds
only the dataset generation pipeline, which is complete and self-verified.

## The task

A cube has six colored faces. A random sequence of rotations is applied
("rotate so the side that was at X is now on top"), and the question asks
for one named face's color afterward. `data_gen/cube.py` ports the rotation
mechanics verbatim from reasoning-gym (Apache-2.0) so every generated
question/answer pair is reproducible against upstream's own dataset for the
same seed.

Three assistant-target variants are generated per problem, so all three are
directly comparable (same seeds, same underlying scrambles):

| Variant | Assistant target | File suffix |
|---|---|---|
| **CoT** | Natural-language step-by-step trace of the cube's face colors after every rotation | `*.jsonl` (`messages_cot`) |
| **No-CoT** | Just the final color, no trace | `*.jsonl` (`messages_answer_only`) |
| **PoT** | A short Python program against a fixed `Cube` API; the interpreter executes it and the printed output is the answer — no natural-language reasoning at all | `*_pot.jsonl` (`messages_pot`) |

## Repo layout

```
data_gen/
  cube.py                  cube rotation mechanics (ported from reasoning-gym)
  generate_dataset.py      builds CoT + no-CoT splits
  pot_library.py           the Cube API exposed to PoT programs (system prompt + executable preamble)
  pot_executor.py          sandboxed executor/grader for PoT programs
  generate_pot_dataset.py  builds PoT splits (same seeds as generate_dataset.py)
data/
  train.jsonl / val.jsonl / test_seen.jsonl / test_extrapolate.jsonl        CoT + no-CoT
  train_pot.jsonl / val_pot.jsonl / test_seen_pot.jsonl / test_extrapolate_pot.jsonl   PoT
PLAN.md                    full research plan
```

Splits: `train` (8000 examples, 1–3 rotations), `val` (1000, 1–3),
`test_seen` (1000, 1–3, disjoint seed), `test_extrapolate` (1000, 4–6
rotations — never seen in training, for testing generalization beyond
trained chain length).

## Regenerating the data

```bash
cd data_gen
python3 generate_dataset.py --out-dir ../data      # CoT + no-CoT
python3 generate_pot_dataset.py --out-dir ../data  # PoT (executes and verifies every program)
```

`generate_pot_dataset.py` runs every generated program through the sandboxed
executor at generation time and raises immediately if its output doesn't
match the recorded ground-truth answer — so the checked-in `*_pot.jsonl`
files are already execution-verified.

## Running/grading a PoT program yourself

```python
from pot_executor import run_program, score_program

program = '''
cube = Cube(top="pink", right="gray", front="orange", left="purple", back="indigo", bottom="cyan")
cube.rotate_to_top("bottom")
print(cube.back)
'''
run_program(program)            # -> "orange"
score_program(program, "orange")  # -> 1.0
```

`run_program` executes in an isolated `python3 -I` subprocess with a 5s
timeout — a lightweight sandbox suitable for local research use, not a
hardened one.

## Example: same problem, three ways

**Question** (identical across all three variants):
```
A cube has:
- a indigo top side
- a red right side
- a brown front side
- a magenta left side
- a gray back side
- a yellow bottom side

The cube is rotated so that the side which was before at the back is now at the top.
Now the cube is rotated to place its front side at the top.

What is now the color of the bottom side of the cube?
Provide only the color as your final answer.
```

**CoT target:**
```
Initial state: top=indigo, right=red, front=brown, left=magenta, back=gray, bottom=yellow
Step 1: rotate so the back side becomes the top. New state: top=gray, right=red, front=indigo, left=magenta, back=yellow, bottom=brown.
Step 2: rotate so the front side becomes the top. New state: top=indigo, right=red, front=brown, left=magenta, back=gray, bottom=yellow.
The bottom side is now yellow.
```

**No-CoT target:** `yellow`

**PoT target:**
```python
cube = Cube(top="indigo", right="red", front="brown", left="magenta", back="gray", bottom="yellow")
cube.rotate_to_top("back")
cube.rotate_to_top("front")
print(cube.bottom)
```
