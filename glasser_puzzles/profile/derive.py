"""Build a ProfileCard from a scored questionnaire.  Rules only, no model."""

from __future__ import annotations

from ..needs.constructs import NeedProfile, need_index, vector_to_dict
from .heuristics import CONTRAST_HEURISTICS, CONTROL_HEURISTICS
from .schema import ProfileCard

#: How far apart two needs must sit before we claim the contrast means anything.
#: Below this the ordering is inside the instrument's own noise, and asserting a
#: heuristic would be inventing a personality rather than measuring one.
GAP_THRESHOLD = 0.25
MAX_HEURISTICS = 3


def derive(profile: NeedProfile) -> ProfileCard:
    gaps = []
    for (high, low), text in CONTRAST_HEURISTICS.items():
        gap = float(profile.strengths[need_index(high)] - profile.strengths[need_index(low)])
        if gap >= GAP_THRESHOLD:
            gaps.append((gap, text))
    gaps.sort(key=lambda g: -g[0])
    heuristics = [text for _, text in gaps[:MAX_HEURISTICS]]

    for low, high, text in CONTROL_HEURISTICS:
        if low <= profile.control_orientation < high:
            heuristics.append(text)
            break

    return ProfileCard(
        need_strengths=vector_to_dict(profile.strengths),
        dominant=profile.dominant,
        secondary=profile.secondary,
        control_orientation=profile.control_orientation,
        decision_heuristics=tuple(heuristics),
        confidence=vector_to_dict(profile.confidence),
    )
