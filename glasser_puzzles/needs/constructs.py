"""Core Choice Theory constructs.

Grounded in William Glasser's *Choice Theory*: every person is driven by the
same five basic needs, and people differ in the *relative* strength of those
needs rather than in which needs they have.  Everything downstream reasons over
a centred (ipsative) need vector for exactly that reason -- see
``needs.scoring`` for why absolute Likert sums are not usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Need(str, Enum):
    """The five basic needs."""

    SURVIVAL = "survival"
    BELONGING = "belonging"
    POWER = "power"
    FREEDOM = "freedom"
    FUN = "fun"

    @property
    def label(self) -> str:
        return {
            Need.SURVIVAL: "Survival",
            Need.BELONGING: "Love & Belonging",
            Need.POWER: "Power",
            Need.FREEDOM: "Freedom",
            Need.FUN: "Fun",
        }[self]


#: Canonical ordering.  Every need vector in this codebase is a length-5 numpy
#: array indexed by this order; use :func:`need_index` rather than hardcoding.
NEEDS: tuple[Need, ...] = (
    Need.SURVIVAL,
    Need.BELONGING,
    Need.POWER,
    Need.FREEDOM,
    Need.FUN,
)

N_NEEDS = len(NEEDS)


def need_index(need: Need) -> int:
    return NEEDS.index(need)


def vector_to_dict(vec: np.ndarray) -> dict[Need, float]:
    return {need: float(vec[i]) for i, need in enumerate(NEEDS)}


def dict_to_vector(mapping: dict[Need, float]) -> np.ndarray:
    return np.array([mapping[need] for need in NEEDS], dtype=float)


@dataclass(frozen=True)
class NeedProfile:
    """The output of scoring a questionnaire.

    ``strengths`` is *centred*: it is the person's relative need ordering, not
    an absolute score.  A value of 0.0 means "this need sits at this person's
    own average"; positive means it outranks their other needs.
    """

    strengths: np.ndarray  # shape (5,), centred, roughly -1..1
    confidence: np.ndarray  # shape (5,), 0..1, from within-need item agreement
    control_orientation: float  # -1 external control .. +1 internal control
    free_text: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.strengths.shape != (N_NEEDS,):
            raise ValueError(f"strengths must have shape ({N_NEEDS},)")
        if self.confidence.shape != (N_NEEDS,):
            raise ValueError(f"confidence must have shape ({N_NEEDS},)")

    @property
    def dominant(self) -> Need | None:
        """Highest-ranked need, or ``None`` if the profile is flat.

        A respondent who gives identical answers to every item has no
        meaningful ordering; reporting an arbitrary winner there would be
        fabrication, so we report nothing.
        """
        if float(np.ptp(self.strengths)) < 1e-9:
            return None
        return NEEDS[int(np.argmax(self.strengths))]

    @property
    def secondary(self) -> Need | None:
        if self.dominant is None:
            return None
        order = np.argsort(-self.strengths)
        return NEEDS[int(order[1])]

    def as_dict(self) -> dict[Need, float]:
        return vector_to_dict(self.strengths)

    def ranked(self) -> list[tuple[Need, float]]:
        order = np.argsort(-self.strengths)
        return [(NEEDS[int(i)], float(self.strengths[i])) for i in order]
