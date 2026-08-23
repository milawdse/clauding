"""Puzzle data model.

A puzzle is a short scenario with several *defensible* solutions.  There is no
correct answer -- each option serves a different basic need, so which one a
person reaches for is information about their need ordering.  That is the whole
mechanism: a puzzle with an objectively best answer would tell us nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..needs.constructs import Need


@dataclass(frozen=True)
class Option:
    id: str
    text: str
    need: Need  # the need this solution serves; never shown to the user


@dataclass(frozen=True)
class Puzzle:
    id: str
    scenario: str
    options: tuple[Option, ...]
    contrast: tuple[Need, Need]  # the two needs this puzzle deliberately opposes

    def __post_init__(self) -> None:
        if not 3 <= len(self.options) <= 5:
            raise ValueError(f"{self.id}: expected 3-5 options, got {len(self.options)}")
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.id}: duplicate option ids {ids}")
        needs = {o.need for o in self.options}
        for need in self.contrast:
            if need not in needs:
                raise ValueError(
                    f"{self.id}: contrast names {need.value} but no option serves it"
                )

    def option(self, option_id: str) -> Option:
        for opt in self.options:
            if opt.id == option_id:
                return opt
        raise KeyError(f"{self.id}: no option {option_id!r}")

    @property
    def need_indices(self) -> list[int]:
        from ..needs.constructs import need_index

        return [need_index(o.need) for o in self.options]
