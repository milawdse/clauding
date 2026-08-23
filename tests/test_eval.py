"""Tests for the evaluation harness itself.

These matter more than they look: an evaluation that quietly measures the wrong
thing is worse than no evaluation, because it produces a number you believe.
"""

import numpy as np
import pytest

from glasser_puzzles.eval.prediction import RiggedEvaluationError, run as run_prediction
from glasser_puzzles.eval.recovery import run as run_recovery
from glasser_puzzles.synth.simulate import SimulatorSpec


@pytest.fixture(scope="module")
def prediction_run():
    return run_prediction(n=400, seed=2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_instrument_recovers_the_latent_vector(seed):
    """Checked across seeds: item quality varies, the instrument must not."""
    result = run_recovery(n=800, seed=seed)
    assert result.passed(), result.per_need_r
    assert result.dominant_accuracy > 0.5


def test_correctly_specified_simulator_is_refused():
    """The single easiest way to fool yourself on this project."""
    rigged = SimulatorSpec(
        persona_alpha_sd=0.0, logit_noise_sd=0.0, random_responder_fraction=0.0
    )
    assert not rigged.is_misspecified
    with pytest.raises(RiggedEvaluationError):
        run_prediction(n=100, spec=rigged)


def test_gate_0_real_profile_beats_shuffled_profile(prediction_run):
    """If this fails, the personality conditioning is decoration."""
    out = prediction_run
    by_name = {s.name: s for s in out["results"]}
    real = by_name["contest model, real profile (V0)"].accuracy
    shuffled = by_name["contest model, SHUFFLED profile"].accuracy
    assert real - shuffled > 0.10, f"real {real:.3f} vs shuffled {shuffled:.3f}"


def test_gate_1_beats_profile_free_baselines(prediction_run):
    out = prediction_run
    by_name = {s.name: s for s in out["results"]}
    real = by_name["contest model, real profile (V0)"].accuracy
    assert real > by_name["modal option (no profile)"].accuracy + 0.02
    assert real > by_name["uniform random"].accuracy + 0.02


def test_gate_1_beta_is_meaningfully_nonzero(prediction_run):
    out = prediction_run
    low, _ = out["beta_ci"]
    assert low > 0.0


def test_shuffled_profile_is_confidently_wrong(prediction_run):
    """Its calibration should be visibly bad -- that is the signature."""
    out = prediction_run
    by_name = {s.name: s for s in out["results"]}
    assert by_name["contest model, SHUFFLED profile"].ece > (
        by_name["contest model, real profile (V0)"].ece
    )


def test_oracle_bounds_the_real_model(prediction_run):
    """Nothing may beat the ceiling; if it does, the eval is leaking."""
    out = prediction_run
    by_name = {s.name: s for s in out["results"]}
    assert (
        by_name["oracle (true latent + true params)"].accuracy
        >= by_name["contest model, real profile (V0)"].accuracy
    )
