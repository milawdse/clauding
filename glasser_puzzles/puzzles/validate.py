"""Vet the frozen puzzle bank.

**What this checks and what it does not.**  The balance check runs neutral
personas through the contest model, which reads a puzzle's *need composition*
and nothing else.  It catches option sets where one need dominates for everyone
regardless of profile -- a real failure mode, since a puzzle with an obvious
answer carries no information.  It cannot judge the writing: whether one option
is worded more attractively than the others is a question for a human reader,
and at V0 there is no model that could stand in for one.  Treat a pass here as
"the structure is sound", not "the puzzle is good".
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from ..needs.constructs import NEEDS
from ..predict.contest import ContestParams, DEFAULT_PARAMS, probabilities
from .bank_loader import load_bank
from .contrasts import CONTRAST_PAIRS, label
from .schema import Puzzle

#: An option chosen this often by profile-neutral respondents is dominating.
DOMINANCE_LIMIT = 0.60
#: Longest option may not exceed the shortest by more than this factor --
#: length is itself a choice cue.
LENGTH_RATIO_LIMIT = 2.5

#: Matched on a word boundary: "fund" must not trip the check for "fun".
_NEED_WORD_RE = re.compile(
    r"\b(" + "|".join(list({n.value for n in NEEDS} | {"belong"})) + r")\b",
    re.IGNORECASE,
)


def check(puzzle: Puzzle, params: ContestParams = DEFAULT_PARAMS) -> list[str]:
    problems: list[str] = []

    needs = [o.need for o in puzzle.options]
    if len(set(needs)) != len(needs):
        problems.append("two options serve the same need")

    for option in puzzle.options:
        for hit in set(_NEED_WORD_RE.findall(option.text)):
            problems.append(f"option {option.id} leaks the need word {hit.lower()!r}")

    lengths = [len(o.text) for o in puzzle.options]
    if max(lengths) / min(lengths) > LENGTH_RATIO_LIMIT:
        problems.append(
            f"option lengths vary too much ({min(lengths)}..{max(lengths)} chars)"
        )

    # Balance: a profile-neutral respondent should not have an obvious pick.
    probs = probabilities(np.zeros(len(NEEDS)), puzzle, params)
    if probs.max() > DOMINANCE_LIMIT:
        winner = puzzle.options[int(probs.argmax())]
        problems.append(
            f"option {winner.id} dominates at {probs.max():.0%} for a neutral profile"
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit nonzero on problems")
    args = parser.parse_args()

    bank = load_bank()
    total_problems = 0
    for puzzle in bank:
        problems = check(puzzle)
        total_problems += len(problems)
        if problems:
            print(f"{puzzle.id}:")
            for problem in problems:
                print(f"    - {problem}")

    counts = {pair: 0 for pair in CONTRAST_PAIRS}
    for puzzle in bank:
        counts[puzzle.contrast] += 1
    missing = [label(p) for p, c in counts.items() if c == 0]

    print(f"\n{len(bank)} puzzles, {total_problems} problems")
    print(f"contrast coverage: {sum(1 for c in counts.values() if c)}/{len(CONTRAST_PAIRS)} pairs")
    if missing:
        print("uncovered contrasts: " + ", ".join(missing))
    if args.strict and (total_problems or missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
