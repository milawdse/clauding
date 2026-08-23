"""Generate PoT (Program of Thoughts) SFT datasets for the same
color_cube_rotation problems as generate_dataset.py -- same seeds, same
questions, same answers -- but with NO natural-language reasoning trace.
Instead, the assistant target is a short Python program against the fixed
`Cube` API in pot_library.py; solving the task means the *interpreter*
executes the program and its printed output is graded, not the model's text.

This makes CoT and PoT directly comparable: both are trained/evaluated on
the identical set of problems, differing only in how the model is asked to
externalize (or not externalize) its reasoning.

Every generated program is executed here at generation time and checked
against the ground-truth answer, so a bug in program synthesis can't
silently produce a mislabeled training example.

Usage:
    python generate_pot_dataset.py --out-dir ../data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_dataset import SPLITS, build_example
from pot_executor import run_program
from pot_library import POT_LIBRARY_SOURCE

SYSTEM_PROMPT = (
    "You solve cube-rotation problems by writing a short Python program, "
    "not by explaining your reasoning in words.\n\n"
    "The following class is already defined and available to your program:\n\n"
    f"{POT_LIBRARY_SOURCE}\n"
    "Write ONLY a Python program (no prose, no comments, no markdown "
    "fences) that:\n"
    "1. Constructs a Cube from the colors given in the problem.\n"
    "2. Calls .rotate_to_top(side) once per rotation described, in order.\n"
    "3. Ends with a single `print(...)` of the requested face's color.\n"
    "Do not compute or state the answer yourself -- the program's printed "
    "output is the answer."
)


def build_program(initial_state: dict[str, str], rotations: list[str], target_side: str) -> str:
    kwargs = ", ".join(f'{side}="{color}"' for side, color in initial_state.items())
    lines = [f"cube = Cube({kwargs})"]
    for side in rotations:
        lines.append(f'cube.rotate_to_top("{side}")')
    lines.append(f"print(cube.{target_side})")
    return "\n".join(lines)


def to_pot_example(example: dict[str, Any]) -> dict[str, Any]:
    meta = example["metadata"]
    program = build_program(meta["initial_state"], meta["rotations"], meta["target_side"])

    executed = run_program(program)
    if executed != example["answer"].strip().lower():
        raise RuntimeError(
            f"PoT program disagrees with ground truth for id={example['id']}: "
            f"got {executed!r}, expected {example['answer']!r}\nprogram:\n{program}"
        )

    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "program": program,
        "metadata": meta,
        "messages_pot": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": program},
        ],
    }


def write_split(path: Path, seed: int, size: int, min_rotations: int, max_rotations: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for idx in range(size):
            example = build_example(seed, idx, min_rotations, max_rotations)
            pot_example = to_pot_example(example)
            f.write(json.dumps(pot_example) + "\n")
    print(f"wrote {size:>5} examples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("../data"))
    args = parser.parse_args()

    for name, seed, size, min_r, max_r in SPLITS:
        write_split(args.out_dir / f"{name}_pot.jsonl", seed, size, min_r, max_r)


if __name__ == "__main__":
    main()
