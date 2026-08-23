"""Fit the shipped contest parameters on synthetic data and write them to disk.

Re-run this whenever the item bank, the puzzle bank, or the simulator changes::

    python -m glasser_puzzles.predict.fit_default

The resulting parameters are a *starting point*, not a finding about people.
They come from simulated personas, so they encode the prior in
``synth/personas.py`` and nothing more.  Refit on real sessions as soon as there
are any -- that is what turns this from an assumption into a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..needs.scoring import score
from ..puzzles.bank_loader import load_bank
from ..synth.irt import make_item_bank, simulate_responses
from ..synth.personas import sample_personas
from ..synth.simulate import SimulatorSpec, answer, persona_offsets
from .contest import ContestParams, fit

PARAMS_PATH = Path(__file__).parent / "params.json"


def fit_from_synthetic(n: int = 1500, seed: int = 7) -> ContestParams:
    rng = np.random.default_rng(seed)
    spec = SimulatorSpec()
    item_bank = make_item_bank(rng)
    personas = sample_personas(n, rng)
    jitter, is_random = persona_offsets(personas, spec, rng)
    puzzles = load_bank()

    observations = []
    for i, persona in enumerate(personas):
        profile = score(simulate_responses(persona, item_bank, rng))
        for puzzle in puzzles:
            chosen = answer(persona, puzzle, spec, rng, jitter[i], bool(is_random[i]))
            observations.append((profile.strengths, puzzle, chosen))
    return fit(observations)


def load_params() -> ContestParams:
    if not PARAMS_PATH.exists():
        from .contest import DEFAULT_PARAMS

        return DEFAULT_PARAMS
    raw = json.loads(PARAMS_PATH.read_text())
    from ..needs.constructs import NEEDS

    return ContestParams(
        beta=float(raw["beta"]),
        alpha=np.array([raw["alpha"][n.value] for n in NEEDS]),
    )


def main() -> None:
    params = fit_from_synthetic()
    PARAMS_PATH.write_text(json.dumps(params.as_dict(), indent=2) + "\n")
    print(f"wrote {PARAMS_PATH}")
    print(json.dumps(params.as_dict(), indent=2))


if __name__ == "__main__":
    main()
