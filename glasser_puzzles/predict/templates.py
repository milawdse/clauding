"""Turn a contest-model prediction into the reasoning shown to the user.

Every number in the output is read off the actual computation: the person's own
centred need scores, the fitted beta, and the resulting probabilities.  Nothing
here is written after the answer is known, and nothing is decorative -- if the
model changes its mind, this text changes with it.

That is the whole reason V0 uses an interpretable model.  A stronger black-box
predictor could not produce these sentences, and the sentences are the feature.
"""

from __future__ import annotations

import textwrap

from ..needs.constructs import Need, NeedProfile, need_index
from ..puzzles.schema import Puzzle
from .contest import ContestParams, Prediction, contested_needs

#: Below this margin we say the call is close rather than assert it.
_CLOSE_MARGIN = 0.12


def _phrase_score(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.1:
        return "right at your own average"
    if magnitude < 0.3:
        return "slightly above your average" if value > 0 else "slightly below your average"
    if magnitude < 0.6:
        return "well above your average" if value > 0 else "well below your average"
    return "at the top of your ordering" if value > 0 else "at the bottom of your ordering"


def explain(
    profile: NeedProfile,
    puzzle: Puzzle,
    prediction: Prediction,
    params: ContestParams,
) -> list[str]:
    high, low = contested_needs(puzzle, profile.strengths)
    high_score = float(profile.strengths[need_index(high)])
    low_score = float(profile.strengths[need_index(low)])

    steps: list[str] = []

    # 1. What the questionnaire said.
    if abs(high_score - low_score) < 0.05:
        steps.append(
            f"Your answers put {high.label} and {low.label} at almost the same "
            f"level ({high_score:+.2f} against {low_score:+.2f} on your own "
            "scale), so there is not much in this one."
        )
    else:
        steps.append(
            f"Your answers put {high.label} {_phrase_score(high_score)} "
            f"({high_score:+.2f}) and {low.label} {_phrase_score(low_score)} "
            f"({low_score:+.2f}), measured against your own average rather than "
            "anyone else's."
        )

    # 2. What this puzzle actually puts in contest, in the puzzle's own terms.
    high_option = next(o for o in puzzle.options if o.need == high)
    low_option = next(o for o in puzzle.options if o.need == low)
    steps.append(
        f"This scenario sets those two against each other: option "
        f"{high_option.id} is the {high.label.lower()} answer, option "
        f"{low_option.id} is the {low.label.lower()} one."
    )
    if set(puzzle.contrast) != {high, low}:
        authored = " and ".join(sorted(n.label for n in puzzle.contrast))
        steps.append(
            f"It was written to test {authored}, but given your ordering the "
            f"real contest here is {high.label} against {low.label}."
        )

    # 3. The arithmetic, stated as arithmetic.
    top = prediction.probabilities[prediction.predicted_option_id]
    runner = prediction.probabilities[prediction.runner_up_id]
    steps.append(
        f"Weighting need strength by how strongly it drove choices in the data "
        f"(beta = {params.beta:.2f}), that comes out at "
        f"{top:.0%} for {prediction.predicted_option_id} against "
        f"{runner:.0%} for {prediction.runner_up_id}."
    )

    # 4. The call, hedged honestly.
    if prediction.margin < _CLOSE_MARGIN:
        steps.append(
            f"So my guess is {prediction.predicted_option_id} — but the gap is "
            f"{prediction.margin:.0%}, which is close enough that "
            f"{prediction.runner_up_id} would not surprise me."
        )
    else:
        steps.append(f"So my guess is {prediction.predicted_option_id}.")

    return steps


def render_reveal(
    prediction: Prediction,
    steps: list[str],
    actual_option_id: str,
    puzzle: Puzzle,
    width: int = 76,
) -> str:
    def wrap(text: str, first: str = "", rest: str = "") -> str:
        return textwrap.fill(text, width, initial_indent=first, subsequent_indent=rest)

    hit = prediction.predicted_option_id == actual_option_id
    lines = [
        wrap(
            f"You chose {actual_option_id}.  I had guessed "
            f"{prediction.predicted_option_id} at {prediction.confidence:.0%} "
            f"confidence — {'hit' if hit else 'miss'}."
        ),
        "",
        "How I got there:",
    ]
    lines += [wrap(s, f"  {i}. ", "     ") for i, s in enumerate(steps, 1)]
    if not hit:
        actual_need = puzzle.option(actual_option_id).need
        lines += [
            "",
            wrap(
                f"You went for the {actual_need.label} answer instead, which I "
                f"had at {prediction.probabilities[actual_option_id]:.0%}. That "
                f"moves {actual_need.label} up in your profile.",
                "  ",
                "  ",
            ),
        ]
    return "\n".join(lines)
