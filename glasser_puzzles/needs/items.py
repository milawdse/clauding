"""The questionnaire.

These items are **original work** written from the constructs described in
Glasser's *Choice Theory*.  They are deliberately not a reproduction of the
Glasser Institute's "Basic Needs Profile", which is a proprietary instrument.
No validity claims are made for them; see the README.

Design notes
------------
* Five needs x four items.  Four items per need is still thin, which is why
  :mod:`glasser_puzzles.needs.scoring` emits a per-need confidence derived from
  how much the three items agree.
* Each need is **balanced two forward, two reverse**.  Without reverse keying a
  respondent who agrees with everything scores high on all five needs and the
  profile is pure acquiescence bias; balancing the direction within each scale
  also keeps that bias from leaking unevenly between needs.
* Three further items measure internal vs external control psychology (the
  Seven Caring Habits vs the Seven Deadly Habits).
"""

from __future__ import annotations

from dataclasses import dataclass

from .constructs import Need

LIKERT_MIN = 1
LIKERT_MAX = 5
LIKERT_LABELS = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neither",
    4: "Agree",
    5: "Strongly agree",
}


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    need: Need | None  # None => control-orientation item
    reverse: bool = False


ITEMS: tuple[Item, ...] = (
    # --- Survival: security, stability, health, resources -----------------
    Item("S1", "I keep a financial cushion so an unexpected bill would not disrupt my life.", Need.SURVIVAL),
    Item("S2", "Before committing to something new, I want to know exactly what could go wrong.", Need.SURVIVAL),
    Item("S3", "I am comfortable leaving my future arrangements loose and unplanned.", Need.SURVIVAL, reverse=True),
    Item("S4", "I would rather gamble on something uncertain than settle for something dependable.", Need.SURVIVAL, reverse=True),
    # --- Love & Belonging: connection, closeness, being needed ------------
    Item("B1", "A day feels incomplete if I have not had a real conversation with someone I care about.", Need.BELONGING),
    Item("B2", "I will change my own plans to keep someone close to me from feeling let down.", Need.BELONGING),
    Item("B3", "I am content spending long stretches of time without contacting anyone.", Need.BELONGING, reverse=True),
    Item("B4", "I make the decisions that matter to me without needing to talk them through with anyone.", Need.BELONGING, reverse=True),
    # --- Power: achievement, competence, significance ---------------------
    Item("P1", "I want the work I do to be visibly better than what was expected of me.", Need.POWER),
    Item("P2", "Being recognised as good at something matters to me more than I usually admit.", Need.POWER),
    Item("P3", "I am content to contribute without anyone knowing which part was mine.", Need.POWER, reverse=True),
    Item("P4", "I have no particular wish to be the best at the things I do.", Need.POWER, reverse=True),
    # --- Freedom: autonomy, independence, room to choose ------------------
    Item("F1", "Being told exactly how to do something makes me want to do it my own way instead.", Need.FREEDOM),
    Item("F2", "I would accept a smaller reward if it meant nobody could tell me how to spend my time.", Need.FREEDOM),
    Item("F3", "I would rather someone else set the rules so that I do not have to decide.", Need.FREEDOM, reverse=True),
    Item("F4", "Committing myself to a long arrangement does not feel like giving anything up.", Need.FREEDOM, reverse=True),
    # --- Fun: learning, play, novelty -------------------------------------
    Item("U1", "I go out of my way to learn things that have no practical use to me.", Need.FUN),
    Item("U2", "I lose track of time when something turns out to be more interesting than I expected.", Need.FUN),
    Item("U3", "I prefer activities whose outcome I can already predict.", Need.FUN, reverse=True),
    Item("U4", "I would rather repeat something I know I enjoy than take a chance on something new.", Need.FUN, reverse=True),
    # --- Internal vs external control psychology --------------------------
    Item("C1", "When someone close to me behaves badly, I focus on what I can change in how I respond.", None),
    Item("C2", "If people would just take my advice, most of my frustrations would disappear.", None, reverse=True),
    Item("C3", "When something goes wrong between me and someone else, it is usually because they would not listen.", None, reverse=True),
)

ITEMS_BY_ID: dict[str, Item] = {item.id: item for item in ITEMS}

NEED_ITEMS: tuple[Item, ...] = tuple(i for i in ITEMS if i.need is not None)
CONTROL_ITEMS: tuple[Item, ...] = tuple(i for i in ITEMS if i.need is None)

#: Open-ended Quality World prompts.  Stored at V0 but not parsed -- there is no
#: language model in V0 and keyword matching would add noise, not signal.
FREE_TEXT_PROMPTS: tuple[str, ...] = (
    "Describe a moment from the past year you would choose to live again.",
    "Describe a situation in which you felt most like yourself.",
)


def items_for(need: Need) -> tuple[Item, ...]:
    return tuple(i for i in ITEMS if i.need == need)
