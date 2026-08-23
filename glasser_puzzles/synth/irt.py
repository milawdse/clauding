"""Simulate questionnaire responses with a graded response model.

For an item with discrimination ``a`` and ordered thresholds ``b``::

    P(X >= k) = sigmoid(a * (theta_item - b_k))
    P(X  = k) = P(X >= k) - P(X >= k+1)

``theta_item`` is the persona's strength on the need the item measures, negated
for reverse-keyed items, plus their acquiescence.  Acquiescence is applied in
the *raw* direction on purpose -- it is a habit about the printed statement, not
about the construct -- which is precisely the bias ipsatization removes.
"""

from __future__ import annotations

import numpy as np

from ..needs.constructs import need_index
from ..needs.items import ITEMS, LIKERT_MAX, LIKERT_MIN
from .personas import Persona

N_CATEGORIES = LIKERT_MAX - LIKERT_MIN + 1

#: Item thresholds on the latent scale.  Symmetric around zero so a persona at
#: the population mean answers near the midpoint.
_THRESHOLDS = np.array([-1.6, -0.5, 0.5, 1.6])


def _item_parameters(rng: np.random.Generator) -> dict[str, tuple[float, np.ndarray]]:
    """Discrimination and thresholds per item, fixed for a whole experiment."""
    params: dict[str, tuple[float, np.ndarray]] = {}
    for item in ITEMS:
        discrimination = float(rng.uniform(1.1, 2.0))
        jitter = rng.normal(0.0, 0.15, size=len(_THRESHOLDS))
        params[item.id] = (discrimination, np.sort(_THRESHOLDS + jitter))
    return params


def make_item_bank(rng: np.random.Generator) -> dict[str, tuple[float, np.ndarray]]:
    return _item_parameters(rng)


def _category_probabilities(a: float, theta: float, thresholds: np.ndarray) -> np.ndarray:
    cumulative = 1.0 / (1.0 + np.exp(-a * (theta - thresholds)))  # P(X >= 2..5)
    padded = np.concatenate([[1.0], cumulative, [0.0]])
    return padded[:-1] - padded[1:]


def simulate_responses(
    persona: Persona,
    item_bank: dict[str, tuple[float, np.ndarray]],
    rng: np.random.Generator,
) -> dict[str, float]:
    """One completed questionnaire."""
    responses: dict[str, float] = {}
    for item in ITEMS:
        a, thresholds = item_bank[item.id]
        if item.need is None:
            # Control-orientation items are not part of the need vector; give
            # them a persona-stable latent of their own so they are not noise.
            theta = float(persona.latent.mean() + persona.acquiescence)
        else:
            theta = float(persona.latent[need_index(item.need)])
            if item.reverse:
                theta = -theta
            theta += persona.acquiescence
        probs = _category_probabilities(a * persona.extremity, theta, thresholds)
        responses[item.id] = float(rng.choice(np.arange(LIKERT_MIN, LIKERT_MAX + 1), p=probs))
    return responses
