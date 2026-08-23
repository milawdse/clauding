"""Terminal front end for V0.

    python -m glasser_puzzles.cli play          # take the quiz, solve puzzles
    python -m glasser_puzzles.cli demo          # same loop, auto-answered
    python -m glasser_puzzles.cli verify-log    # audit the sealed predictions
"""

from __future__ import annotations

import argparse
import textwrap
import uuid
from pathlib import Path

import numpy as np

from .needs.constructs import NeedProfile
from .needs.items import FREE_TEXT_PROMPTS, ITEMS, LIKERT_LABELS, LIKERT_MAX, LIKERT_MIN
from .needs.scoring import score
from .predict.contest import predict
from .predict.fit_default import load_params
from .predict.seal import DEFAULT_LOG, record_answer, seal, verify_log
from .predict.templates import explain, render_reveal
from .profile.derive import derive
from .profile.schema import ProfileCard
from .puzzles.select import next_puzzle
from .update.belief import Belief, update

WIDTH = 76


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent)


# --------------------------------------------------------------------------
# Questionnaire
# --------------------------------------------------------------------------


def ask_questionnaire() -> tuple[dict[str, float], tuple[str, ...]]:
    print(_rule("="))
    print("A few questions about how you tend to decide things.")
    print(f"Answer 1-{LIKERT_MAX}:  " + ",  ".join(f"{k} {v.lower()}" for k, v in LIKERT_LABELS.items()))
    print(_rule("="))

    responses: dict[str, float] = {}
    for n, item in enumerate(ITEMS, 1):
        print(f"\n{n}/{len(ITEMS)}")
        print(_wrap(item.text))
        while True:
            raw = input("  > ").strip()
            try:
                value = float(raw)
            except ValueError:
                print(f"  Please enter a number from {LIKERT_MIN} to {LIKERT_MAX}.")
                continue
            if LIKERT_MIN <= value <= LIKERT_MAX:
                responses[item.id] = value
                break
            print(f"  Please enter a number from {LIKERT_MIN} to {LIKERT_MAX}.")

    free_text = []
    for prompt in FREE_TEXT_PROMPTS:
        print(f"\n{_wrap(prompt)}")
        print("  (optional — press enter to skip)")
        free_text.append(input("  > ").strip())
    return responses, tuple(free_text)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def print_card(card: ProfileCard) -> None:
    print("\n" + _rule("="))
    print("YOUR PROFILE")
    print(_rule("="))
    print("\nRelative need strength (measured against your own average, not other people's):\n")
    for need, value in card.ranked():
        filled = int(round((value + 1) * 15))
        bar = "#" * filled + "." * (30 - filled)
        conf = card.confidence[need]
        print(f"  {need.label:<18} {bar} {value:+.2f}   (confidence {conf:.0%})")

    orientation = card.control_orientation
    where = "internal control" if orientation > 0 else "external control"
    print(f"\n  Control orientation: {orientation:+.2f} — leaning toward {where}.")

    if card.decision_heuristics:
        print("\nWhat that predicts about how you decide:\n")
        for heuristic in card.decision_heuristics:
            print(_wrap(f"- {heuristic}", indent="  "))
    else:
        print("\n  Your needs came out close together, so no single pattern stands out.")


def print_puzzle(puzzle, number: int, total: int) -> None:
    print("\n" + _rule("="))
    print(f"PUZZLE {number}/{total}")
    print(_rule("="))
    print("\n" + _wrap(puzzle.scenario))
    print()
    for option in puzzle.options:
        print(
            textwrap.fill(
                f"{option.id}. {option.text}",
                WIDTH,
                initial_indent="  ",
                subsequent_indent="     ",
            )
        )
    print()


# --------------------------------------------------------------------------
# The session loop
# --------------------------------------------------------------------------


def run_session(
    profile: NeedProfile,
    n_puzzles: int,
    log_path: Path,
    answer_fn,
    session_id: str | None = None,
) -> dict:
    """Shared by ``play`` and ``demo``.

    ``answer_fn(puzzle) -> option_id`` is the only difference between a human
    session and an automated one, which keeps the demo an honest exercise of the
    same code path rather than a separate script that might drift.
    """
    session_id = session_id or uuid.uuid4().hex[:12]
    params = load_params()
    belief = Belief.from_profile(profile)
    seen: set[str] = set()
    hits = 0
    played = 0

    for index in range(1, n_puzzles + 1):
        puzzle = next_puzzle(belief, seen, params)
        if puzzle is None:
            break
        seen.add(puzzle.id)

        # --- predict and SEAL, before the puzzle is displayed --------------
        working = belief.as_profile(profile.control_orientation)
        prediction = predict(working, puzzle, params)
        steps = explain(working, puzzle, prediction, params)
        sealed = seal(session_id, prediction, steps, params, log_path)

        # --- only now does the user see anything --------------------------
        print_puzzle(puzzle, index, n_puzzles)
        chosen = answer_fn(puzzle)
        record_answer(session_id, sealed, chosen, log_path)

        played += 1
        if prediction.predicted_option_id == chosen:
            hits += 1

        print("\n" + _rule())
        print(render_reveal(prediction, list(sealed.steps), chosen, puzzle))
        print(_rule())
        print(f"  Running: {hits}/{played} correct.")

        belief = update(belief, puzzle, chosen, params)

    return {"session_id": session_id, "hits": hits, "played": played, "belief": belief}


def print_summary(result: dict, profile: NeedProfile, log_path: Path) -> None:
    print("\n" + _rule("="))
    print("SUMMARY")
    print(_rule("="))
    played, hits = result["played"], result["hits"]
    if played:
        print(f"\n  I called {hits} of {played} correctly ({hits / played:.0%}).")
        print(f"  Guessing at random on four options would be about 25%.")

    print("\n  How your profile moved as you played:\n")
    before = profile.strengths
    after = result["belief"].mean
    from .needs.constructs import NEEDS

    for i, need in enumerate(NEEDS):
        delta = after[i] - before[i]
        arrow = "  " if abs(delta) < 0.01 else (" up" if delta > 0 else " dn")
        print(f"    {need.label:<18} {before[i]:+.2f} -> {after[i]:+.2f}  {arrow}")

    print(f"\n  Sealed predictions logged to {log_path}")
    print("  Audit them with: python -m glasser_puzzles.cli verify-log")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _human_answer(puzzle):
    valid = [o.id for o in puzzle.options]
    while True:
        chosen = input(f"  Your choice ({'/'.join(valid)}) > ").strip().upper()
        if chosen in valid:
            return chosen
        print(f"  Please enter one of {', '.join(valid)}.")


def cmd_play(args) -> None:
    responses, free_text = ask_questionnaire()
    profile = score(responses, free_text)
    print_card(derive(profile))
    result = run_session(profile, args.puzzles, args.log, _human_answer)
    print_summary(result, profile, args.log)


def cmd_demo(args) -> None:
    """Same loop, answered by a synthetic persona -- for verifying end to end."""
    from .synth.irt import make_item_bank, simulate_responses
    from .synth.personas import sample_personas
    from .synth.simulate import SimulatorSpec, answer as sim_answer, persona_offsets

    rng = np.random.default_rng(args.seed)
    spec = SimulatorSpec()
    persona = sample_personas(1, rng)[0]
    jitter, is_random = persona_offsets([persona], spec, rng)
    profile = score(simulate_responses(persona, make_item_bank(rng), rng))

    print(_rule("="))
    print("DEMO — a synthetic person answers the questionnaire and the puzzles.")
    print(_rule("="))
    from .needs.constructs import NEEDS

    print("\n  Their true need vector (which the system never sees):")
    for i, need in enumerate(NEEDS):
        print(f"    {need.label:<18} {persona.latent[i]:+.2f}")

    print_card(derive(profile))

    def answer_fn(puzzle):
        chosen = sim_answer(persona, puzzle, spec, rng, jitter[0], bool(is_random[0]))
        print(f"  Their choice > {chosen}")
        return chosen

    result = run_session(profile, args.puzzles, args.log, answer_fn)
    print_summary(result, profile, args.log)


def cmd_verify_log(args) -> None:
    checked, problems = verify_log(args.log)
    print(f"Checked {checked} sealed predictions in {args.log}")
    if not checked:
        print("  Nothing logged yet.")
        return
    if problems:
        print(f"\n  {len(problems)} PROBLEMS:")
        for problem in problems:
            print(f"    - {problem}")
        raise SystemExit(1)
    print("  All predictions were sealed before their answers were recorded.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="glasser_puzzles", description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="take the questionnaire and solve puzzles")
    play.add_argument("--puzzles", type=int, default=5)
    play.set_defaults(func=cmd_play)

    demo = sub.add_parser("demo", help="run the same loop with a synthetic person")
    demo.add_argument("--puzzles", type=int, default=5)
    demo.add_argument("--seed", type=int, default=3)
    demo.set_defaults(func=cmd_demo)

    verify = sub.add_parser("verify-log", help="audit the sealed prediction log")
    verify.set_defaults(func=cmd_verify_log)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
