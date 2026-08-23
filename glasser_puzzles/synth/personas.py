"""Sample synthetic people.

A persona is a latent need vector plus two response-style nuisance parameters.
The nuisance parameters exist so that scoring has something real to defend
against: if every simulated respondent used the Likert scale identically,
ipsatization would look unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..needs.constructs import N_NEEDS

#: Mild, theory-plausible correlations between need strengths.  Freedom pulls
#: against Belonging and Survival; Fun pulls with Freedom and against Survival.
#: These are priors for generating data, not empirical findings.
NEED_CORRELATION = np.array(
    [
        [1.00, 0.05, 0.15, -0.30, -0.25],
        [0.05, 1.00, 0.00, -0.30, 0.10],
        [0.15, 0.00, 1.00, 0.05, 0.00],
        [-0.30, -0.30, 0.05, 1.00, 0.25],
        [-0.25, 0.10, 0.00, 0.25, 1.00],
    ]
)


@dataclass(frozen=True)
class Persona:
    latent: np.ndarray  # (5,) true need strengths, centred
    acquiescence: float  # pushes every raw response toward "agree"
    extremity: float  # >1 uses the ends of the scale more

    @property
    def dominant_index(self) -> int:
        return int(np.argmax(self.latent))


def sample_personas(n: int, rng: np.random.Generator) -> list[Persona]:
    chol = np.linalg.cholesky(_nearest_psd(NEED_CORRELATION))
    raw = rng.standard_normal((n, N_NEEDS)) @ chol.T
    # Centre each persona: only the ordering is meaningful, matching what the
    # scorer can actually recover.
    latent = raw - raw.mean(axis=1, keepdims=True)
    acquiescence = rng.normal(0.0, 0.45, size=n)
    extremity = np.exp(rng.normal(0.0, 0.25, size=n))
    return [
        Persona(latent=latent[i], acquiescence=float(acquiescence[i]), extremity=float(extremity[i]))
        for i in range(n)
    ]


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if eigenvalues.min() > 1e-8:
        return matrix
    clipped = np.clip(eigenvalues, 1e-8, None)
    rebuilt = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(scale, scale)
