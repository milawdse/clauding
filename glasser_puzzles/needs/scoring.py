"""Deterministic questionnaire scoring.  No model, no randomness, no network.

The one decision that matters here is **ipsatization**.  Raw Likert sums are
dominated by acquiescence bias: a respondent who agrees with everything scores
high on all five needs, and their "profile" is really a measure of how
agreeable they are.  Glasser's model is about *relative* need strength anyway,
so we subtract each respondent's own mean before normalising.  What survives is
the person's need *ordering*, which is the thing the rest of the system uses.
"""

from __future__ import annotations

import numpy as np

from .constructs import NEEDS, N_NEEDS, Need, NeedProfile
from .items import (
    CONTROL_ITEMS,
    ITEMS_BY_ID,
    LIKERT_MAX,
    LIKERT_MIN,
    NEED_ITEMS,
    items_for,
)

NEUTRAL = (LIKERT_MIN + LIKERT_MAX) / 2.0  # 3.0

#: Standard deviation (in Likert points) at which we call a need's three items
#: fully inconsistent and drop confidence to zero.
_MAX_DISAGREEMENT = 2.0


def _oriented(item_id: str, raw: float) -> float:
    """Flip reverse-keyed responses onto the common direction."""
    item = ITEMS_BY_ID[item_id]
    return (LIKERT_MIN + LIKERT_MAX) - raw if item.reverse else raw


def score(
    responses: dict[str, float],
    free_text: tuple[str, ...] = (),
) -> NeedProfile:
    """Score a completed questionnaire.

    Parameters
    ----------
    responses:
        Item id -> Likert response in ``[1, 5]``.  Missing items are imputed to
        neutral and cost confidence on the need they belong to, so a partially
        completed questionnaire degrades gracefully instead of failing.
    """
    for item_id, value in responses.items():
        if item_id not in ITEMS_BY_ID:
            raise KeyError(f"unknown item id: {item_id!r}")
        if not LIKERT_MIN <= value <= LIKERT_MAX:
            raise ValueError(
                f"response for {item_id} is {value}, outside [{LIKERT_MIN}, {LIKERT_MAX}]"
            )

    need_means = np.zeros(N_NEEDS)
    confidence = np.zeros(N_NEEDS)

    for idx, need in enumerate(NEEDS):
        values = []
        n_missing = 0
        for item in items_for(need):
            if item.id in responses:
                values.append(_oriented(item.id, responses[item.id]))
            else:
                values.append(NEUTRAL)
                n_missing += 1
        arr = np.asarray(values, dtype=float)
        need_means[idx] = arr.mean()

        # Confidence: do this need's items agree with each other?  Three items
        # that disagree wildly mean we do not really know this need's strength.
        disagreement = float(arr.std(ddof=0))
        agreement = 1.0 - min(1.0, disagreement / _MAX_DISAGREEMENT)
        # Imputed items carry no information, so scale confidence by how much
        # of the need was actually answered.
        answered_fraction = 1.0 - n_missing / len(items_for(need))
        confidence[idx] = agreement * answered_fraction

    # Ipsatize, then scale so a full Likert point of deviation is 0.5.
    centred = need_means - need_means.mean()
    strengths = np.clip(centred / 2.0, -1.0, 1.0)

    return NeedProfile(
        strengths=strengths,
        confidence=confidence,
        control_orientation=_score_control(responses),
        free_text=free_text,
    )


def _score_control(responses: dict[str, float]) -> float:
    """-1 (external control psychology) .. +1 (internal control psychology)."""
    values = [
        _oriented(item.id, responses.get(item.id, NEUTRAL)) for item in CONTROL_ITEMS
    ]
    return float((np.mean(values) - NEUTRAL) / (LIKERT_MAX - NEUTRAL))


def neutral_responses() -> dict[str, float]:
    """Every item answered at the midpoint -- used in tests and balance checks."""
    return {item.id: NEUTRAL for item in ITEMS_BY_ID.values()}


def responses_from_vector(strengths: dict[Need, float]) -> dict[str, float]:
    """Inverse-ish of :func:`score`, for building fixtures by hand.

    Not used in the pipeline; handy for demos and tests where you want a
    respondent with a known ordering without inventing 18 numbers.
    """
    out: dict[str, float] = {}
    for need, value in strengths.items():
        target = NEUTRAL + value * 2.0
        for item in items_for(need):
            raw = np.clip(target, LIKERT_MIN, LIKERT_MAX)
            out[item.id] = float(
                (LIKERT_MIN + LIKERT_MAX) - raw if item.reverse else raw
            )
    for item in CONTROL_ITEMS:
        out.setdefault(item.id, NEUTRAL)
    return out
