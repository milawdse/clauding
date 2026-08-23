"""Choose which puzzle to serve next.

This is where tailoring to a person actually happens.  The puzzles themselves
are fixed and profile-blind; what adapts is *which contrast we put in front of
them*.  Keeping it that way matters: a generator that saw the profile would
write the profile-matching option to be the most attractive one, and the
predictor would then be predicting its own bias rather than the person.

Selection is active-learning shaped.  A puzzle is worth serving when:

* its contrast has not been served much already -- otherwise five puzzles in a
  row measure one axis very well and everything else not at all;
* the needs it contests are ones we are currently unsure about;
* the outcome is genuinely uncertain -- a puzzle we can already call at 95%
  teaches us nothing whichever way it goes.
"""

from __future__ import annotations

import numpy as np

from ..needs.constructs import need_index
from ..predict.contest import ContestParams, DEFAULT_PARAMS, probabilities
from ..update.belief import Belief
from .bank_loader import load_bank
from .schema import Puzzle


def _entropy(probs: np.ndarray) -> float:
    safe = np.clip(probs, 1e-12, 1.0)
    return float(-(safe * np.log(safe)).sum())


def information_score(
    puzzle: Puzzle, belief: Belief, seen_contrasts: dict, params: ContestParams
) -> float:
    probs = probabilities(belief.mean, puzzle, params)
    uncertainty = sum(belief.variance[need_index(n)] for n in puzzle.contrast)
    repetition = 1.0 + seen_contrasts.get(puzzle.contrast, 0)
    return _entropy(probs) * uncertainty / repetition


def next_puzzle(
    belief: Belief,
    seen_puzzle_ids: set[str],
    params: ContestParams = DEFAULT_PARAMS,
    bank: tuple[Puzzle, ...] | None = None,
) -> Puzzle | None:
    puzzles = [p for p in (bank or load_bank()) if p.id not in seen_puzzle_ids]
    if not puzzles:
        return None

    seen_contrasts: dict = {}
    for puzzle in bank or load_bank():
        if puzzle.id in seen_puzzle_ids:
            seen_contrasts[puzzle.contrast] = seen_contrasts.get(puzzle.contrast, 0) + 1

    return max(puzzles, key=lambda p: information_score(p, belief, seen_contrasts, params))
