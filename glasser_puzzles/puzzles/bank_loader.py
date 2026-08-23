"""Load and cache the frozen puzzle bank."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from ..needs.constructs import Need
from .contrasts import canonical
from .schema import Option, Puzzle

BANK_DIR = Path(__file__).parent / "bank"


@functools.lru_cache(maxsize=1)
def load_bank(bank_dir: Path | None = None) -> tuple[Puzzle, ...]:
    directory = bank_dir or BANK_DIR
    puzzles: list[Puzzle] = []
    for path in sorted(directory.glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text()) or []:
            puzzles.append(_parse(raw, path))
    ids = [p.id for p in puzzles]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate puzzle ids in bank: {sorted(duplicates)}")
    return tuple(puzzles)


def _parse(raw: dict, path: Path) -> Puzzle:
    try:
        options = tuple(
            Option(id=o["id"], text=o["text"].strip(), need=Need(o["need"]))
            for o in raw["options"]
        )
        contrast = canonical((Need(raw["contrast"][0]), Need(raw["contrast"][1])))
        return Puzzle(
            id=raw["id"],
            scenario=" ".join(raw["scenario"].split()),
            options=options,
            contrast=contrast,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path.name}: bad puzzle {raw.get('id', '?')}: {exc}") from exc


def by_contrast() -> dict[tuple[Need, Need], list[Puzzle]]:
    out: dict[tuple[Need, Need], list[Puzzle]] = {}
    for puzzle in load_bank():
        out.setdefault(puzzle.contrast, []).append(puzzle)
    return out
