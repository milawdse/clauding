"""Generate SFT datasets for Qwen2.5 LoRA training on reasoning-gym's
color_cube_rotation task, with algorithmically-derived middle-step reasoning
traces (a "scratchpad" of the cube's face colors after every rotation).

Each example is generated with the *same* RNG scheme as upstream
(`random.Random(seed + idx)`, same rotation-selection and story-generation
logic), so it reproduces the identical question/answer upstream's own
ColorCubeRotationDataset(seed=..., size=...) would produce for that seed+idx.
We only add the middle-step trace on top, by tracking cube state ourselves
as we apply the exact same rotation calls.

Usage:
    python generate_dataset.py --out-dir ../data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from cube import ROTATION_TEMPLATES, Side, generate_cube, rotate_to_top

SYSTEM_PROMPT = (
    "You are given a cube with six colored sides. Track how the colors move "
    "as the cube is rotated, then answer with only the requested color."
)


def build_example(seed: int, idx: int, min_rotations: int, max_rotations: int) -> dict[str, Any]:
    rng = random.Random(seed + idx)

    cube = generate_cube(rng)
    initial_state = cube.colors.copy()

    num_rotations = rng.randint(min_rotations, max_rotations)
    rotations: list[Side] = []
    available_sides = [s for s in Side if s != Side.TOP]

    # Mirrors upstream: pick rotations, apply immediately, but here we also
    # snapshot the resulting state after each one for the reasoning trace.
    step_states: list[dict[str, str]] = []
    while len(rotations) < num_rotations:
        from_side = rng.choice(available_sides)
        rotations.append(from_side)
        rotate_to_top(cube, from_side)
        step_states.append(cube.state())

    target_side = rng.choice(list(Side))
    answer = cube.colors[target_side].value

    # --- question text (byte-identical construction to upstream) ---
    story_parts = ["A cube has:"]
    for side in Side:
        story_parts.append(f"- a {initial_state[side].value} {side.value} side")
    for i, from_side in enumerate(rotations):
        template = ROTATION_TEMPLATES[0] if i == 0 else rng.choice(ROTATION_TEMPLATES[1:])
        story_parts.append(f"\n{template.format(side=from_side.value)}")
    story_parts.append(f"\nWhat is now the color of the {target_side.value} side of the cube?")
    story_parts.append("Provide only the color as your final answer.")
    question = "\n".join(story_parts)

    # --- middle-step reasoning trace (our addition) ---
    def fmt_state(state: dict[str, str]) -> str:
        order = ["top", "right", "front", "left", "back", "bottom"]
        return ", ".join(f"{s}={state[s]}" for s in order)

    trace_lines = [f"Initial state: {fmt_state({k.value: v.value for k, v in initial_state.items()})}"]
    for i, (from_side, state) in enumerate(zip(rotations, step_states), start=1):
        trace_lines.append(
            f"Step {i}: rotate so the {from_side.value} side becomes the top. "
            f"New state: {fmt_state(state)}."
        )
    trace_lines.append(f"The {target_side.value} side is now {answer}.")
    cot_trace = "\n".join(trace_lines)

    answer_text = answer  # bare color, matches score_answer's exact-match format

    return {
        "id": idx,
        "question": question,
        "answer": answer,
        "cot_trace": cot_trace,
        "metadata": {
            "initial_state": {k.value: v.value for k, v in initial_state.items()},
            "rotations": [r.value for r in rotations],
            "step_states": step_states,
            "target_side": target_side.value,
            "num_rotations": num_rotations,
        },
        "messages_cot": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": cot_trace},
        ],
        "messages_answer_only": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_text},
        ],
    }


def write_split(path: Path, seed: int, size: int, min_rotations: int, max_rotations: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for idx in range(size):
            example = build_example(seed, idx, min_rotations, max_rotations)
            f.write(json.dumps(example) + "\n")
    print(f"wrote {size:>5} examples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("../data"))
    args = parser.parse_args()

    # (split name, seed, size, min_rotations, max_rotations)
    splits = [
        ("train", 1000, 8000, 1, 3),
        ("val", 2000, 1000, 1, 3),
        ("test_seen", 3000, 1000, 1, 3),
        ("test_extrapolate", 4000, 1000, 4, 6),
    ]

    for name, seed, size, min_r, max_r in splits:
        write_split(args.out_dir / f"{name}.jsonl", seed, size, min_r, max_r)


if __name__ == "__main__":
    main()
