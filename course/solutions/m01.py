"""Reference solutions — Module 1, "What System 2 Buys You".

Exec'd into the notebook's namespace by `tools/verify.py`, so these may use
anything the lecture defined above the exercises (`data`, `Trace`,
`traced_simulation`, …).
"""

from collections import Counter, defaultdict

from csai import data
from csai.trace import Trace

# Each rotation is a 4-cycle over faces: the colour on cycle[i] moves to
# cycle[i+1]. The two faces not mentioned are the axis and stay put.
CYCLES = {
    "front":  ("front", "top", "back", "bottom"),
    "back":   ("back", "top", "front", "bottom"),
    "right":  ("right", "top", "left", "bottom"),
    "left":   ("left", "top", "right", "bottom"),
    # `bottom` is a half turn: two independent 2-cycles.
    "bottom": (("bottom", "top"), ("front", "back")),
}


def rotate(state, side):
    new = dict(state)
    spec = CYCLES[side]
    cycles = spec if isinstance(spec[0], tuple) else (spec,)
    for cycle in cycles:
        for i, face in enumerate(cycle):
            new[cycle[(i + 1) % len(cycle)]] = state[face]
    return new


def solve_by_simulation(example):
    state = data.initial_state(example)
    for side in example["metadata"]["rotations"]:
        state = rotate(state, side)
    return state[example["metadata"]["target_side"]]


def last_rotation_only(example):
    rotations = example["metadata"]["rotations"]
    state = rotate(data.initial_state(example), rotations[-1])
    return state[example["metadata"]["target_side"]]


def accuracy_by_length(solver, examples):
    buckets = defaultdict(list)
    for ex in examples:
        buckets[data.num_rotations(ex)].append(ex)
    return {
        k: sum(solver(ex) == ex["answer"] for ex in group) / len(group)
        for k, group in sorted(buckets.items())
    }


def trace_for(example):
    state = data.initial_state(example)
    tr = Trace(name="cube rotation", initial=state)
    for side in example["metadata"]["rotations"]:
        state = rotate(state, side)
        tr.step(f"rotate {side} to top", state)
    return tr.finish(state[example["metadata"]["target_side"]])


def first_divergence(predicted, gold):
    for i, want in enumerate(gold):
        if i >= len(predicted) or predicted[i] != want:
            return i + 1
    return None


def evaluate(traced_solver, examples):
    correct = 0
    lucky = 0
    step_scores = []
    first_wrong = Counter()
    by_length = defaultdict(lambda: [0, 0])

    for ex in examples:
        tr = traced_solver(ex)
        gold = data.gold_states(ex)
        predicted = tr.states

        matched = sum(
            1 for i, want in enumerate(gold)
            if i < len(predicted) and predicted[i] == want
        )
        step_scores.append(matched / len(gold) if gold else 1.0)

        diverged_at = first_divergence(predicted, gold)
        first_wrong[diverged_at or 0] += 1

        answer_ok = tr.result == ex["answer"]
        correct += answer_ok
        lucky += answer_ok and diverged_at is not None

        depth = data.num_rotations(ex)
        by_length[depth][0] += answer_ok
        by_length[depth][1] += 1

    n = len(examples)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "by_length": {k: c / t for k, (c, t) in sorted(by_length.items())},
        "step_accuracy": sum(step_scores) / n if n else 0.0,
        "first_wrong": dict(sorted(first_wrong.items())),
        "lucky": lucky,
    }
