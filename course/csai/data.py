"""Access to this repo's `color_cube_rotation` datasets and generators.

The course sits inside a research repo whose `data_gen/` already contains a
verified cube simulator, a Program-of-Thoughts runtime and a sandboxed
executor. Rather than reimplement any of that, the notebooks import it
through here. Nothing in `course/` copies upstream code.

Record shape in `data/*.jsonl` (see `data_gen/generate_dataset.py`)::

    {
      "id": 0,
      "question": "A cube has:\\n- a indigo top side\\n...",
      "answer": "yellow",
      "cot_trace": "Initial state: ...\\nStep 1: ...\\nThe bottom side is now yellow.",
      "metadata": {
         "initial_state": {"top": "indigo", ...},   # side -> colour
         "rotations": ["back", "front"],            # sides brought to the top
         "step_states": [{...}, {...}],             # state after each rotation
         "target_side": "bottom",
         "num_rotations": 2
      },
      "messages_cot": [...], "messages_answer_only": [...]
    }

`train`/`val`/`test_seen` use 1-3 rotations; `test_extrapolate` uses 4-6 and
appears in no training split. That split is the course's recurring test of
whether something has learned an *algorithm* or a *table*.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

#: repo root: course/csai/data.py -> course/csai -> course -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATA_GEN_DIR = REPO_ROOT / "data_gen"

SPLITS = ("train", "val", "test_seen", "test_extrapolate")
SIDES = ("top", "right", "front", "left", "back", "bottom")


class DataNotFound(FileNotFoundError):
    """Raised with a repair hint when a split file is missing."""


def _ensure_data_gen_on_path() -> None:
    path = str(DATA_GEN_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def split_path(name: str, *, pot: bool = False) -> Path:
    return DATA_DIR / f"{name}{'_pot' if pot else ''}.jsonl"


def available_splits() -> list[str]:
    return [s for s in SPLITS if split_path(s).exists()]


def iter_split(name: str, *, pot: bool = False,
               limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream one split, one decoded record at a time."""
    path = split_path(name, pot=pot)
    if not path.exists():
        raise DataNotFound(
            f"{path} not found. Regenerate it with:\n"
            f"    cd {DATA_GEN_DIR} && python3 generate_dataset.py --out-dir ../data"
        )
    with path.open() as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def load_split(name: str, *, pot: bool = False,
               limit: int | None = None) -> list[dict[str, Any]]:
    """Load one split into a list. `limit` keeps notebooks fast."""
    return list(iter_split(name, pot=pot, limit=limit))


def num_rotations(example: dict[str, Any]) -> int:
    """Chain length of an example — the difficulty axis used throughout."""
    meta = example.get("metadata") or {}
    if "num_rotations" in meta:
        return int(meta["num_rotations"])
    return len(meta.get("rotations", []))


def gold_states(example: dict[str, Any]) -> list[dict[str, str]]:
    """Ground-truth cube state after each rotation (the gold trace)."""
    return list((example.get("metadata") or {}).get("step_states", []))


def initial_state(example: dict[str, Any]) -> dict[str, str]:
    return dict((example.get("metadata") or {})["initial_state"])


def group_by_length(examples: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Bucket examples by rotation count, for accuracy-vs-depth curves."""
    out: dict[int, list[dict[str, Any]]] = {}
    for ex in examples:
        out.setdefault(num_rotations(ex), []).append(ex)
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------
# Bridges to data_gen/
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def cube_module() -> ModuleType:
    """`data_gen/cube.py` — Colour/Side enums and the rotation methods."""
    _ensure_data_gen_on_path()
    import cube  # type: ignore

    return cube


@lru_cache(maxsize=None)
def pot_library() -> ModuleType:
    """`data_gen/pot_library.py` — the `Cube` API shown to PoT programs."""
    _ensure_data_gen_on_path()
    import pot_library  # type: ignore

    return pot_library


@lru_cache(maxsize=None)
def pot_executor() -> ModuleType:
    """`data_gen/pot_executor.py` — sandboxed `run_program`/`score_program`."""
    _ensure_data_gen_on_path()
    import pot_executor  # type: ignore

    return pot_executor


def make_cube(state: dict[str, str]):
    """Build a `data_gen.cube.Cube` from a plain side->colour dict."""
    cube = cube_module()
    return cube.Cube(colors={cube.Side(s): cube.Color(c) for s, c in state.items()})


def rotate_to_top(cube_obj, side: str) -> None:
    """Apply one 'bring `side` to the top' rotation, upstream's way."""
    cube = cube_module()
    cube.rotate_to_top(cube_obj, cube.Side(side))


def simulate(state: dict[str, str], rotations: list[str]) -> list[dict[str, str]]:
    """Reference simulator: the cube state after each rotation in turn."""
    cube_obj = make_cube(state)
    states = []
    for side in rotations:
        rotate_to_top(cube_obj, side)
        states.append(cube_obj.state())
    return states
