"""The V0 predictor: a logistic need-contest model.

For a puzzle whose option *i* serves need ``n_i``, the probability that a person
with centred need vector ``s`` picks option *i* is::

    P(i | s) = softmax_i( beta * s[n_i] + alpha[n_i] )

``beta``
    A single scalar: how strongly relative need strength drives choice.  **This
    is the parameter that decides whether the whole idea works.**  A beta near
    zero means personality does not predict choice and no amount of model will
    fix that.

``alpha``
    Per-need option-attractiveness intercepts.  Some needs are simply more
    appealing to choose regardless of who is answering.  Without these, beta
    absorbs the base rate and the model looks predictive when it is only
    reproducing what everybody picks.  ``alpha[0]`` is pinned to zero because
    softmax is invariant to a shift.

Why logistic rather than something stronger: the reveal has to explain itself.
A gradient-boosted model would predict a little better and be unable to say
why, which would kill the feature this project exists for.  Interpretability is
a hard requirement here, not a preference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..needs.constructs import N_NEEDS, Need, NeedProfile, need_index
from ..puzzles.schema import Puzzle

_NEG_INF = -1.0e9


@dataclass(frozen=True)
class ContestParams:
    beta: float
    alpha: np.ndarray  # shape (5,), alpha[0] == 0

    def fingerprint(self) -> tuple:
        """Identity of these parameters, used to detect a rigged evaluation.

        If the simulator that generates synthetic answers and the predictor
        being scored share a fingerprint, accuracy is near-perfect and means
        nothing at all.  ``eval.prediction`` refuses to report in that case.
        """
        return (round(self.beta, 9), tuple(np.round(self.alpha, 9).tolist()))

    def as_dict(self) -> dict:
        from ..needs.constructs import NEEDS

        return {
            "beta": self.beta,
            "alpha": {n.value: float(self.alpha[i]) for i, n in enumerate(NEEDS)},
        }


#: Sensible starting point before any data exists.  beta=1.5 says need strength
#: matters but does not fully determine choice; flat alpha says we assume no
#: option is inherently more attractive until the data says otherwise.
DEFAULT_PARAMS = ContestParams(beta=1.5, alpha=np.zeros(N_NEEDS))


@dataclass(frozen=True)
class Prediction:
    puzzle_id: str
    probabilities: dict[str, float]  # option id -> probability
    predicted_option_id: str
    runner_up_id: str
    confidence: float

    @property
    def margin(self) -> float:
        return self.confidence - self.probabilities[self.runner_up_id]


def option_logits(
    strengths: np.ndarray, needs: list[int], params: ContestParams
) -> np.ndarray:
    return params.beta * strengths[needs] + params.alpha[needs]


def probabilities(
    strengths: np.ndarray, puzzle: Puzzle, params: ContestParams = DEFAULT_PARAMS
) -> np.ndarray:
    logits = option_logits(strengths, puzzle.need_indices, params)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def predict(
    profile: NeedProfile, puzzle: Puzzle, params: ContestParams = DEFAULT_PARAMS
) -> Prediction:
    probs = probabilities(profile.strengths, puzzle, params)
    order = np.argsort(-probs)
    return Prediction(
        puzzle_id=puzzle.id,
        probabilities={o.id: float(p) for o, p in zip(puzzle.options, probs)},
        predicted_option_id=puzzle.options[int(order[0])].id,
        runner_up_id=puzzle.options[int(order[1])].id,
        confidence=float(probs[order[0]]),
    )


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    """Padded, vectorised observations.

    ``needs`` is padded with 0 and masked rather than ragged, so the whole
    likelihood is a couple of numpy operations instead of a Python loop.
    """

    strengths: np.ndarray  # (N, 5)
    needs: np.ndarray  # (N, K) int
    mask: np.ndarray  # (N, K) bool
    chosen: np.ndarray  # (N,) int, column index into K

    def __len__(self) -> int:
        return len(self.chosen)


def build_dataset(
    observations: list[tuple[np.ndarray, Puzzle, str]],
) -> Dataset:
    """From ``(strengths, puzzle, chosen_option_id)`` triples."""
    if not observations:
        raise ValueError("no observations")
    k = max(len(p.options) for _, p, _ in observations)
    n = len(observations)
    strengths = np.zeros((n, N_NEEDS))
    needs = np.zeros((n, k), dtype=int)
    mask = np.zeros((n, k), dtype=bool)
    chosen = np.zeros(n, dtype=int)
    for row, (s, puzzle, chosen_id) in enumerate(observations):
        idx = puzzle.need_indices
        strengths[row] = s
        needs[row, : len(idx)] = idx
        mask[row, : len(idx)] = True
        chosen[row] = [o.id for o in puzzle.options].index(chosen_id)
    return Dataset(strengths, needs, mask, chosen)


def _unpack(theta: np.ndarray) -> ContestParams:
    alpha = np.concatenate([[0.0], theta[1:]])
    return ContestParams(beta=float(theta[0]), alpha=alpha)


def _neg_log_likelihood(theta: np.ndarray, data: Dataset, l2: float) -> float:
    params = _unpack(theta)
    # (N, K) logits, gathering each row's own need strengths.
    s = np.take_along_axis(data.strengths, data.needs, axis=1)
    logits = params.beta * s + params.alpha[data.needs]
    logits = np.where(data.mask, logits, _NEG_INF)
    logits = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(logits).sum(axis=1))
    chosen_logits = np.take_along_axis(
        logits, data.chosen[:, None], axis=1
    ).squeeze(1)
    nll = float(-(chosen_logits - log_norm).mean())
    return nll + l2 * float(np.sum(theta[1:] ** 2))


def fit(
    observations: list[tuple[np.ndarray, Puzzle, str]],
    l2: float = 1e-3,
) -> ContestParams:
    """Maximum-likelihood fit.  Five free parameters -- effectively instant."""
    data = build_dataset(observations)
    theta0 = np.concatenate([[1.0], np.zeros(N_NEEDS - 1)])
    result = minimize(
        _neg_log_likelihood, theta0, args=(data, l2), method="L-BFGS-B"
    )
    if not result.success:  # pragma: no cover - L-BFGS on 5 params is reliable
        raise RuntimeError(f"contest model failed to converge: {result.message}")
    return _unpack(result.x)


def log_likelihood_per_obs(
    observations: list[tuple[np.ndarray, Puzzle, str]], params: ContestParams
) -> float:
    data = build_dataset(observations)
    theta = np.concatenate([[params.beta], params.alpha[1:]])
    return -_neg_log_likelihood(theta, data, l2=0.0)


def contested_needs(puzzle: Puzzle, strengths: np.ndarray) -> tuple[Need, Need]:
    """The two needs actually deciding this puzzle for this person.

    Usually the authored contrast, but if a distractor option outranks one of
    them for this particular person, that is the real contest and the
    explanation should say so.
    """
    from ..needs.constructs import NEEDS

    scores = {o.need: strengths[need_index(o.need)] for o in puzzle.options}
    ranked = sorted(scores, key=lambda n: -scores[n])
    return ranked[0], ranked[1]
