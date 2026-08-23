import numpy as np
import pytest

from glasser_puzzles.needs.constructs import NeedProfile, Need, need_index
from glasser_puzzles.predict.contest import (
    ContestParams,
    DEFAULT_PARAMS,
    build_dataset,
    fit,
    predict,
    probabilities,
    contested_needs,
)
from glasser_puzzles.puzzles.bank_loader import load_bank

BANK = load_bank()


def _profile(vec):
    return NeedProfile(np.asarray(vec, float), np.ones(5), 0.0)


def test_probabilities_sum_to_one():
    for puzzle in BANK:
        probs = probabilities(np.array([0.3, -0.2, 0.5, -0.4, 0.1]), puzzle)
        assert probs.shape == (len(puzzle.options),)
        assert probs.sum() == pytest.approx(1.0)


def test_flat_profile_with_flat_alpha_is_uniform():
    params = ContestParams(beta=2.0, alpha=np.zeros(5))
    probs = probabilities(np.zeros(5), BANK[0], params)
    assert probs == pytest.approx(np.full(len(BANK[0].options), 1 / len(BANK[0].options)))


def test_dominant_need_wins_its_own_option():
    puzzle = next(p for p in BANK if p.id == "BF1")
    strengths = np.zeros(5)
    strengths[need_index(Need.FREEDOM)] = 1.0
    params = ContestParams(beta=3.0, alpha=np.zeros(5))
    prediction = predict(_profile(strengths), puzzle, params)
    assert puzzle.option(prediction.predicted_option_id).need is Need.FREEDOM


def test_beta_recovery_on_synthetic_data():
    """Generate choices with a known beta; the fit must find it back."""
    rng = np.random.default_rng(0)
    true = ContestParams(beta=2.0, alpha=np.array([0.0, 0.3, -0.2, 0.1, 0.0]))
    observations = []
    for _ in range(4000):
        strengths = rng.normal(0, 0.5, size=5)
        strengths -= strengths.mean()
        puzzle = BANK[rng.integers(len(BANK))]
        probs = probabilities(strengths, puzzle, true)
        chosen = puzzle.options[int(rng.choice(len(probs), p=probs))].id
        observations.append((strengths, puzzle, chosen))
    fitted = fit(observations)
    assert fitted.beta == pytest.approx(true.beta, abs=0.25)
    assert np.allclose(fitted.alpha, true.alpha, atol=0.25)


def test_fingerprint_distinguishes_parameters():
    a = ContestParams(beta=1.0, alpha=np.zeros(5))
    b = ContestParams(beta=1.0, alpha=np.zeros(5))
    c = ContestParams(beta=1.1, alpha=np.zeros(5))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_contested_needs_reflects_the_person_not_the_author():
    """A distractor option can be the real contest for a given person."""
    puzzle = next(p for p in BANK if p.id == "BF1")  # authored belonging vs freedom
    strengths = np.zeros(5)
    strengths[need_index(Need.POWER)] = 1.0
    strengths[need_index(Need.FUN)] = 0.5
    high, low = contested_needs(puzzle, strengths)
    assert {high, low} == {Need.POWER, Need.FUN}


def test_build_dataset_rejects_empty():
    with pytest.raises(ValueError):
        build_dataset([])


def test_prediction_margin_is_consistent():
    prediction = predict(_profile([0.5, -0.3, 0.1, 0.4, -0.7]), BANK[0], DEFAULT_PARAMS)
    assert prediction.confidence >= prediction.probabilities[prediction.runner_up_id]
    assert prediction.margin >= 0
