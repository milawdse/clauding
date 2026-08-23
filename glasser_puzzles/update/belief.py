"""Sharpen the profile as the person plays.

The questionnaire gives a prior over need strengths; every puzzle answered is
one more observation.  We keep a diagonal Gaussian belief and update it by
Laplace approximation: find the MAP under the contest likelihood plus the
Gaussian prior, then take curvature at that point as the new precision.

Per-need questionnaire confidence sets the prior width, so a need measured by
four items that disagreed with each other moves quickly under evidence, while
one measured cleanly holds its ground.  That is the behaviour you want: the
system should be most willing to revise what it was least sure of.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..needs.constructs import N_NEEDS, NeedProfile
from ..predict.contest import ContestParams, DEFAULT_PARAMS
from ..puzzles.schema import Puzzle

#: Prior variance at full questionnaire confidence.
_BASE_VARIANCE = 0.15
_MIN_CONFIDENCE = 0.15


@dataclass
class Belief:
    mean: np.ndarray  # (5,)
    variance: np.ndarray  # (5,)

    @classmethod
    def from_profile(cls, profile: NeedProfile) -> "Belief":
        confidence = np.maximum(profile.confidence, _MIN_CONFIDENCE)
        return cls(
            mean=profile.strengths.copy(),
            variance=_BASE_VARIANCE / confidence,
        )

    def as_profile(self, control_orientation: float = 0.0) -> NeedProfile:
        """Back to a NeedProfile so the rest of the pipeline is unchanged."""
        confidence = np.clip(_BASE_VARIANCE / self.variance, 0.0, 1.0)
        return NeedProfile(
            strengths=np.clip(self.mean, -1.0, 1.0),
            confidence=confidence,
            control_orientation=control_orientation,
        )


def _log_likelihood(s: np.ndarray, puzzle: Puzzle, chosen_id: str, params: ContestParams) -> float:
    needs = puzzle.need_indices
    logits = params.beta * s[needs] + params.alpha[needs]
    logits = logits - logits.max()
    chosen = [o.id for o in puzzle.options].index(chosen_id)
    return float(logits[chosen] - np.log(np.exp(logits).sum()))


def update(
    belief: Belief,
    puzzle: Puzzle,
    chosen_id: str,
    params: ContestParams = DEFAULT_PARAMS,
) -> Belief:
    """One observation -> a tighter, shifted belief."""

    def objective(s: np.ndarray) -> float:
        prior = -0.5 * np.sum((s - belief.mean) ** 2 / belief.variance)
        return -(_log_likelihood(s, puzzle, chosen_id, params) + prior)

    result = minimize(objective, belief.mean.copy(), method="L-BFGS-B")
    new_mean = result.x

    # Laplace precision: prior precision plus the likelihood's curvature.
    needs = puzzle.need_indices
    logits = params.beta * new_mean[needs] + params.alpha[needs]
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    per_need = np.zeros(N_NEEDS)
    for idx, p in zip(needs, probs):
        per_need[idx] += p
    curvature = params.beta**2 * per_need * (1.0 - per_need)

    new_variance = 1.0 / (1.0 / belief.variance + curvature)
    return Belief(mean=new_mean, variance=new_variance)
