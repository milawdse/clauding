"""Simulate how a persona answers puzzles.

**The trap this module exists to avoid.** If the same contest model both
generates the synthetic answers and predicts them, accuracy approaches 100% and
measures nothing except that softmax equals softmax.  It is the easiest way to
fool yourself on this project, so the simulator here is deliberately *wrong in
form*, not merely in parameter values:

* per-persona intercept jitter -- unmodelled heterogeneity, people differ in
  which options appeal beyond their need ordering;
* extra logit noise -- choice is noisier than any tidy model of it;
* a fraction of near-random responders -- some people do not engage with the
  scenario at all.

None of those three are in the predictor's functional form, so it cannot fit
them away no matter how much data it sees.  Separately, the simulator answers
from the persona's **true** latent vector while the predictor only ever sees the
**scored** vector recovered from a noisy questionnaire, so the evaluation
measures the whole pipeline rather than just the choice model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..needs.constructs import N_NEEDS
from ..puzzles.schema import Puzzle
from .personas import Persona


@dataclass(frozen=True)
class SimulatorSpec:
    """Ground-truth answer process.  Not the predictor's model."""

    beta: float = 2.2
    alpha: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.35, -0.15, 0.20, 0.10])
    )
    #: SD of the per-persona intercept jitter the predictor cannot represent.
    persona_alpha_sd: float = 0.55
    #: SD of extra Gumbel-scale noise on every option's logit.
    logit_noise_sd: float = 0.45
    #: Fraction of personas who answer close to uniformly at random.
    random_responder_fraction: float = 0.10

    def fingerprint(self) -> tuple:
        return (round(self.beta, 9), tuple(np.round(self.alpha, 9).tolist()))

    @property
    def is_misspecified(self) -> bool:
        """True when the predictor's form genuinely cannot reproduce this."""
        return (
            self.persona_alpha_sd > 0
            or self.logit_noise_sd > 0
            or self.random_responder_fraction > 0
        )


def persona_offsets(
    personas: list[Persona], spec: SimulatorSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Per-persona intercept jitter and random-responder flags."""
    jitter = rng.normal(0.0, spec.persona_alpha_sd, size=(len(personas), N_NEEDS))
    is_random = rng.random(len(personas)) < spec.random_responder_fraction
    return jitter, is_random


def answer(
    persona: Persona,
    puzzle: Puzzle,
    spec: SimulatorSpec,
    rng: np.random.Generator,
    alpha_jitter: np.ndarray,
    is_random_responder: bool,
) -> str:
    """Return the option id this persona picks."""
    needs = puzzle.need_indices
    if is_random_responder:
        return puzzle.options[int(rng.integers(len(puzzle.options)))].id

    logits = (
        spec.beta * persona.latent[needs]
        + spec.alpha[needs]
        + alpha_jitter[needs]
        + rng.normal(0.0, spec.logit_noise_sd, size=len(needs))
    )
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return puzzle.options[int(rng.choice(len(needs), p=probs))].id
