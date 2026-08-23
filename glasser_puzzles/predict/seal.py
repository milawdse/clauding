"""Commit a prediction before the puzzle is shown.

The prediction and its reasoning are written to an append-only log and hashed
*before* the user ever sees the scenario.  The hash goes on the session record;
the reveal reads the sealed entry back rather than recomputing it.

This ordering is the entire credibility of the feature.  A reasoning trace
produced after the answer is known is rationalisation, not prediction, and a
system that shows one is lying to its user in a way that is very hard to detect
from the outside.  Sealing makes the claim auditable: every entry's hash is
recorded before its answer's timestamp, and the log is append-only.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .contest import ContestParams, Prediction

DEFAULT_LOG = Path.home() / ".glasser_puzzles" / "predictions.jsonl"


@dataclass(frozen=True)
class SealedPrediction:
    digest: str
    sealed_at: float
    prediction: Prediction
    steps: tuple[str, ...]


#: Exactly the fields covered by the digest.  Kept explicit so that adding a
#: field to the log cannot silently drop it out of the hash.
SEALED_FIELDS = (
    "session_id",
    "puzzle_id",
    "predicted_option_id",
    "runner_up_id",
    "confidence",
    "probabilities",
    "reasoning_steps",
    "params",
)


def _digest(payload: dict) -> str:
    body = json.dumps(
        {k: payload[k] for k in SEALED_FIELDS}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(body.encode()).hexdigest()


def _payload(
    session_id: str, prediction: Prediction, steps: list[str], params: ContestParams
) -> dict:
    return {
        "session_id": session_id,
        "puzzle_id": prediction.puzzle_id,
        "predicted_option_id": prediction.predicted_option_id,
        "runner_up_id": prediction.runner_up_id,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "reasoning_steps": steps,
        "params": params.as_dict(),
    }


def seal(
    session_id: str,
    prediction: Prediction,
    steps: list[str],
    params: ContestParams,
    log_path: Path | None = None,
) -> SealedPrediction:
    path = log_path or DEFAULT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = _payload(session_id, prediction, steps, params)
    digest = _digest(payload)
    sealed_at = time.time()

    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {"kind": "sealed", "digest": digest, "sealed_at": sealed_at, **payload},
                sort_keys=True,
            )
            + "\n"
        )
    return SealedPrediction(
        digest=digest, sealed_at=sealed_at, prediction=prediction, steps=tuple(steps)
    )


def record_answer(
    session_id: str,
    sealed: SealedPrediction,
    actual_option_id: str,
    log_path: Path | None = None,
) -> None:
    """Append the observed answer, referencing the sealed digest."""
    path = log_path or DEFAULT_LOG
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "answer",
                    "session_id": session_id,
                    "digest": sealed.digest,
                    "puzzle_id": sealed.prediction.puzzle_id,
                    "actual_option_id": actual_option_id,
                    "hit": sealed.prediction.predicted_option_id == actual_option_id,
                    "answered_at": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )


def verify_log(log_path: Path | None = None) -> tuple[int, list[str]]:
    """Re-hash every sealed entry and check each was sealed before it was answered.

    Returns ``(entries_checked, problems)``.  An empty problem list is the
    evidence that no reasoning was written after the fact.
    """
    path = log_path or DEFAULT_LOG
    if not path.exists():
        return 0, []

    sealed: dict[str, dict] = {}
    problems: list[str] = []
    checked = 0

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["kind"] == "sealed":
            checked += 1
            if _digest(entry) != entry["digest"]:
                problems.append(f"line {line_no}: digest does not match its payload")
            sealed[entry["digest"]] = entry
        elif entry["kind"] == "answer":
            origin = sealed.get(entry["digest"])
            if origin is None:
                problems.append(f"line {line_no}: answer references an unsealed prediction")
            elif entry["answered_at"] < origin["sealed_at"]:
                problems.append(
                    f"line {line_no}: answered before it was sealed — "
                    "the prediction cannot be trusted"
                )
    return checked, problems
