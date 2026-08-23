"""Closed vocabulary of decision heuristics.

Each entry is keyed by a *directed* need contrast: ``(high, low)`` means "this
person ranks `high` above `low`".  Phrasing is deliberately behavioural -- what
the person does when the two needs collide -- rather than trait adjectives.
"Values their independence" is not usable; "pays a premium to keep their options
open" describes something you could watch someone do, and it is what the
prediction actually rests on.

The vocabulary is fixed rather than generated so that V0 needs no model, and so
that V1's language model has a well-formed target to select from rather than a
blank page.
"""

from __future__ import annotations

from ..needs.constructs import Need

CONTRAST_HEURISTICS: dict[tuple[Need, Need], str] = {
    (Need.SURVIVAL, Need.BELONGING): "secures their own footing before extending help to others",
    (Need.BELONGING, Need.SURVIVAL): "absorbs a personal cost rather than let someone down",
    (Need.SURVIVAL, Need.POWER): "takes the steady option over the visible one",
    (Need.POWER, Need.SURVIVAL): "accepts real exposure for work that carries their name",
    (Need.SURVIVAL, Need.FREEDOM): "trades room to manoeuvre for knowing where they stand",
    (Need.FREEDOM, Need.SURVIVAL): "pays a premium, in money or security, to keep their options open",
    (Need.SURVIVAL, Need.FUN): "settles the ground before spending anything on curiosity",
    (Need.FUN, Need.SURVIVAL): "will spend a reserve on something interesting rather than sit on it",
    (Need.BELONGING, Need.POWER): "lets credit go rather than put a working relationship under strain",
    (Need.POWER, Need.BELONGING): "wants their specific contribution identified, even at a social cost",
    (Need.BELONGING, Need.FREEDOM): "keeps standing commitments to people even when they chafe",
    (Need.FREEDOM, Need.BELONGING): "protects their own time even when it disappoints someone close",
    (Need.BELONGING, Need.FUN): "chooses familiar company over the novel experience",
    (Need.FUN, Need.BELONGING): "follows something interesting away from the people they know",
    (Need.POWER, Need.FREEDOM): "accepts someone else's process when it buys real influence",
    (Need.FREEDOM, Need.POWER): "turns down scope that arrives with somebody else's schedule attached",
    (Need.POWER, Need.FUN): "channels curiosity into things that build a visible record",
    (Need.FUN, Need.POWER): "follows what is interesting even when it means starting at the bottom",
    (Need.FREEDOM, Need.FUN): "guards unstructured time against even enjoyable commitments",
    (Need.FUN, Need.FREEDOM): "signs up to a fixed commitment when the subject pulls hard enough",
}

#: Keyed by (lower_bound, upper_bound) on control_orientation.
CONTROL_HEURISTICS: tuple[tuple[float, float, str], ...] = (
    (-1.01, -0.45, "expects things to improve once other people take their advice"),
    (-0.45, -0.15, "tends to locate the problem in what the other person did"),
    (0.15, 0.45, "looks at their own part before assigning fault"),
    (0.45, 1.01, "starts from what they can change in themselves when something goes wrong"),
)

assert len(CONTRAST_HEURISTICS) == 20
