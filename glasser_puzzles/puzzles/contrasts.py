"""The ten unordered pairs of basic needs.

Each puzzle in the bank is authored to set one pair against each other.  Serving
puzzles across all ten pairs is what lets the system separate a person's needs:
a run of puzzles that all pit Freedom against Belonging measures one axis very
well and everything else not at all.
"""

from __future__ import annotations

from itertools import combinations

from ..needs.constructs import NEEDS, Need

CONTRAST_PAIRS: tuple[tuple[Need, Need], ...] = tuple(combinations(NEEDS, 2))

assert len(CONTRAST_PAIRS) == 10


def canonical(pair: tuple[Need, Need]) -> tuple[Need, Need]:
    """Order a pair the same way every time, so it can be used as a dict key."""
    a, b = pair
    return (a, b) if NEEDS.index(a) <= NEEDS.index(b) else (b, a)


def label(pair: tuple[Need, Need]) -> str:
    a, b = canonical(pair)
    return f"{a.label} vs {b.label}"
