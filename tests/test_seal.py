import json

import numpy as np
import pytest

from glasser_puzzles.predict.contest import ContestParams, Prediction
from glasser_puzzles.predict.seal import record_answer, seal, verify_log

PARAMS = ContestParams(beta=2.0, alpha=np.zeros(5))
PREDICTION = Prediction(
    puzzle_id="BF1",
    probabilities={"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1},
    predicted_option_id="A",
    runner_up_id="B",
    confidence=0.6,
)


def _seal_one(path, steps=("because",)):
    sealed = seal("sess", PREDICTION, list(steps), PARAMS, path)
    record_answer("sess", sealed, "A", path)
    return sealed


def test_clean_log_verifies(tmp_path):
    path = tmp_path / "log.jsonl"
    _seal_one(path)
    checked, problems = verify_log(path)
    assert checked == 1
    assert problems == []


def test_rewriting_the_reasoning_afterwards_is_caught(tmp_path):
    """The property the whole feature rests on."""
    path = tmp_path / "log.jsonl"
    _seal_one(path)

    lines = path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["reasoning_steps"] = ["I always knew you would pick that."]
    lines[0] = json.dumps(entry, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    _, problems = verify_log(path)
    assert any("digest does not match" in p for p in problems)


def test_answer_before_seal_is_caught(tmp_path):
    path = tmp_path / "log.jsonl"
    sealed = _seal_one(path)

    lines = path.read_text().splitlines()
    answer = json.loads(lines[1])
    answer["answered_at"] = sealed.sealed_at - 10
    lines[1] = json.dumps(answer, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    _, problems = verify_log(path)
    assert any("answered before it was sealed" in p for p in problems)


def test_orphan_answer_is_caught(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "answer",
                "session_id": "s",
                "digest": "deadbeef",
                "puzzle_id": "BF1",
                "actual_option_id": "A",
                "hit": True,
                "answered_at": 1.0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    _, problems = verify_log(path)
    assert any("unsealed prediction" in p for p in problems)


def test_missing_log_is_not_an_error(tmp_path):
    assert verify_log(tmp_path / "nope.jsonl") == (0, [])


def test_digest_is_stable_across_calls(tmp_path):
    a = seal("s", PREDICTION, ["x"], PARAMS, tmp_path / "a.jsonl")
    b = seal("s", PREDICTION, ["x"], PARAMS, tmp_path / "b.jsonl")
    assert a.digest == b.digest
