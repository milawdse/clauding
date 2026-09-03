"""Reference solutions — Module 12, the capstone."""

import random
import statistics
from collections import Counter

from csai import data
from csai.render import table
from csai.trace import Trace, diff_traces


def build_program(example):
    state = data.initial_state(example)
    meta = example["metadata"]
    args = ", ".join(f'{side}="{state[side]}"' for side in data.SIDES)
    lines = [f"cube = Cube({args})"]
    for side in meta["rotations"]:
        lines.append(f'cube.rotate_to_top("{side}")')
    lines.append(f'print(cube.{meta["target_side"]})')
    return "\n".join(lines)


def grade(program, expected):
    return executor.score_program(program, expected)


def noisy_trace(example, rng, step_error):
    state = data.initial_state(example)
    tr = Trace(initial=state)
    for side in example["metadata"]["rotations"]:
        state = true_step(state, side)
        if rng.random() < step_error:
            state = perturb(state, rng)
        tr.step(f"rotate {side} to top", state)
    return tr.finish(state[example["metadata"]["target_side"]])


def first_bad_step(trace, example):
    return diff_traces(trace, data.gold_states(example)).first_divergence


def depth_curve(strategy, examples, seed=0):
    buckets = {}
    for ex in examples:
        rng = random.Random(f"{seed}:{strategy.name}:{ex['id']}")
        attempt = strategy.solve(ex, rng)
        depth = data.num_rotations(ex)
        buckets.setdefault(depth, [0, 0])
        buckets[depth][0] += attempt.answer == ex["answer"]
        buckets[depth][1] += 1
    return {k: c / t for k, (c, t) in sorted(buckets.items())}


# --- project ---------------------------------------------------------------

def experiment(strategies, split, limit=120, seed=0):
    examples = data.load_split(split, limit=limit)
    results = {}
    for strategy in strategies:
        correct = lucky = calls = failures = 0
        by_depth = {}
        step_scores = []
        first_wrong = Counter()
        for ex in examples:
            rng = random.Random(f"{seed}:{strategy.name}:{ex['id']}")
            attempt = strategy.solve(ex, rng)
            depth = data.num_rotations(ex)
            ok = attempt.answer == ex["answer"]
            correct += ok
            calls += attempt.calls
            failures += attempt.answer is None
            by_depth.setdefault(depth, [0, 0])
            by_depth[depth][0] += ok
            by_depth[depth][1] += 1
            if attempt.trace is not None:
                d = diff_traces(attempt.trace, data.gold_states(ex))
                step_scores.append(d.step_accuracy)
                first_wrong[d.first_divergence or 0] += 1
                lucky += ok and d.first_divergence is not None
        n = len(examples)
        results[strategy.name] = {
            "n": n,
            "accuracy": correct / n,
            "by_depth": {k: c / t for k, (c, t) in sorted(by_depth.items())},
            "calls_per_problem": calls / n,
            "step_accuracy": statistics.mean(step_scores) if step_scores else None,
            "first_wrong": dict(sorted(first_wrong.items())),
            "lucky": lucky,
            "execution_failures": failures,
        }
    return results


def summarise(results, depths):
    rows = []
    for name, report in results.items():
        rows.append([name]
                    + [f"{report['by_depth'].get(d, float('nan')):.0%}"
                       for d in depths]
                    + [f"{report['accuracy']:.0%}",
                       f"{report['calls_per_problem']:.2f}",
                       "—" if report["step_accuracy"] is None
                       else f"{report['step_accuracy']:.0%}"])
    return table(rows,
                 ["strategy"] + [f"k={d}" for d in depths]
                 + ["overall", "calls/problem", "step accuracy"],
                 align="l" + "r" * (len(depths) + 3))
