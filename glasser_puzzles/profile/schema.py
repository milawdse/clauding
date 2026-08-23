"""The profile card shown to the user and consumed by the predictor."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..needs.constructs import Need, NeedProfile


@dataclass(frozen=True)
class ProfileCard:
    need_strengths: dict[Need, float]
    dominant: Need | None
    secondary: Need | None
    control_orientation: float
    decision_heuristics: tuple[str, ...]
    confidence: dict[Need, float]
    #: Empty at V0: there is no language model, and keyword matching on the
    #: free-text answers would add noise rather than signal.
    quality_world_themes: tuple[str, ...] = field(default=())

    def ranked(self) -> list[tuple[Need, float]]:
        return sorted(self.need_strengths.items(), key=lambda kv: -kv[1])
