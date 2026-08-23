"""Does the profile actually predict the choice?

This is the go/no-go for the whole project, and it costs nothing to run.

Gate 0 (decisive)
    Hand the predictor *somebody else's* profile.  If accuracy barely drops,
    personality conditioning is decorative, the reveal is theatre, and the
    instrument or the puzzle bank needs rethinking before a single GPU hour is
    spent.  This is the ablation that would otherwise wait until a 32B model was
    running; getting it at V0 is the main reason V0 exists.

Gate 1
    The contest model must beat random and modal choice by a clear margin on
    *mis-specified* simulated personas, with a beta that is meaningfully
    non-zero.

Everything is evaluated on held-out personas: parameters are fitted on a train
split and never see the test set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ..needs.scoring import score
from ..predict.contest import ContestParams, fit, predict
from ..puzzles.bank_loader import load_bank
from ..synth.irt import make_item_bank, simulate_responses
from ..synth.personas import sample_personas
from ..synth.simulate import SimulatorSpec, answer, persona_offsets


class RiggedEvaluationError(RuntimeError):
    """Raised when the simulator and the predictor are the same object."""


@dataclass
class Scored:
    name: str
    accuracy: float
    ece: float
    mean_confidence: float

    @property
    def line(self) -> str:
        return (
            f"  {self.name:<34} acc {self.accuracy:6.1%}   "
            f"ECE {self.ece:5.3f}   mean conf {self.mean_confidence:6.1%}"
        )


def _ece(confidences: np.ndarray, hits: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: does 70% confidence mean 70% right?"""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        total += in_bin.mean() * abs(hits[in_bin].mean() - confidences[in_bin].mean())
    return float(total)


def _build(n: int, seed: int, spec: SimulatorSpec):
    """Simulate the full pipeline: persona -> questionnaire -> scored -> choice."""
    rng = np.random.default_rng(seed)
    item_bank = make_item_bank(rng)
    personas = sample_personas(n, rng)
    jitter, is_random = persona_offsets(personas, spec, rng)
    puzzles = load_bank()

    scored_profiles = []
    for persona in personas:
        scored_profiles.append(score(simulate_responses(persona, item_bank, rng)))

    rows = []  # (persona_idx, puzzle, chosen_id)
    for i, persona in enumerate(personas):
        for puzzle in puzzles:
            chosen = answer(persona, puzzle, spec, rng, jitter[i], bool(is_random[i]))
            rows.append((i, puzzle, chosen))
    return scored_profiles, rows, rng, personas


def run(
    n: int = 1500,
    seed: int = 0,
    spec: SimulatorSpec | None = None,
    train_fraction: float = 0.6,
) -> dict:
    spec = spec or SimulatorSpec()
    if not spec.is_misspecified:
        raise RiggedEvaluationError(
            "The simulator has no unmodelled structure (no persona jitter, no "
            "logit noise, no random responders). The predictor's functional form "
            "would then be exactly correct and the reported accuracy would be "
            "meaningless. Refusing to report a number."
        )

    profiles, rows, rng, personas = _build(n, seed, spec)
    oracle_params = ContestParams(beta=spec.beta, alpha=spec.alpha)
    n_train = int(n * train_fraction)
    train = [r for r in rows if r[0] < n_train]
    test = [r for r in rows if r[0] >= n_train]

    params = fit([(profiles[i].strengths, p, c) for i, p, c in train])

    if params.fingerprint() == spec.fingerprint():
        raise RiggedEvaluationError(
            "The fitted predictor is parameter-identical to the simulator that "
            "generated the answers. Refusing to report a number."
        )

    # --- baselines ---------------------------------------------------------
    modal: dict[str, str] = {}
    for _, puzzle, chosen in train:
        modal.setdefault(puzzle.id, {})
        modal[puzzle.id][chosen] = modal[puzzle.id].get(chosen, 0) + 1  # type: ignore[index]
    modal_choice = {
        pid: max(counts, key=counts.get) for pid, counts in modal.items()  # type: ignore[arg-type]
    }
    modal_conf = {
        pid: max(counts.values()) / sum(counts.values())  # type: ignore[union-attr]
        for pid, counts in modal.items()
    }

    # A shuffled profile: real, well-formed, and belonging to somebody else.
    test_ids = sorted({i for i, _, _ in test})
    shuffled_ids = list(test_ids)
    rng.shuffle(shuffled_ids)
    remap = dict(zip(test_ids, shuffled_ids))

    results: list[Scored] = []

    def collect(name: str, preds: list[tuple[float, bool]]) -> None:
        conf = np.array([c for c, _ in preds])
        hits = np.array([h for _, h in preds], dtype=float)
        results.append(Scored(name, float(hits.mean()), _ece(conf, hits), float(conf.mean())))

    collect(
        "uniform random",
        [(1.0 / len(p.options), rng.integers(len(p.options)) == [o.id for o in p.options].index(c))
         for _, p, c in test],
    )
    collect(
        "modal option (no profile)",
        [(modal_conf[p.id], modal_choice[p.id] == c) for _, p, c in test],
    )
    collect(
        "contest model, SHUFFLED profile",
        [
            (lambda pr: (pr.confidence, pr.predicted_option_id == c))(
                predict(profiles[remap[i]], p, params)
            )
            for i, p, c in test
        ],
    )
    # Oracle: true latent vector, true simulator parameters, no questionnaire
    # noise.  Nothing can beat this, so it says how much headroom is actually
    # left for a language model to win -- and therefore whether V1/V2 are worth
    # building at all.
    collect(
        "oracle (true latent + true params)",
        [
            (lambda pr: (pr.confidence, pr.predicted_option_id == c))(
                predict(_oracle_profile(personas[i]), p, oracle_params)
            )
            for i, p, c in test
        ],
    )
    collect(
        "contest model, real profile (V0)",
        [
            (lambda pr: (pr.confidence, pr.predicted_option_id == c))(
                predict(profiles[i], p, params)
            )
            for i, p, c in test
        ],
    )

    beta_ci = _bootstrap_beta(train, profiles, rng)
    return {
        "results": results,
        "params": params,
        "beta_ci": beta_ci,
        "n_train": n_train,
        "n_test": n - n_train,
        "n_observations": len(test),
        "spec": spec,
    }


def _oracle_profile(persona):
    """A profile carrying the persona's true latent vector, for the ceiling."""
    from ..needs.constructs import NeedProfile

    return NeedProfile(
        strengths=persona.latent, confidence=np.ones(5), control_orientation=0.0
    )


def _bootstrap_beta(train, profiles, rng, n_boot: int = 40) -> tuple[float, float]:
    persona_ids = sorted({i for i, _, _ in train})
    by_persona: dict[int, list] = {}
    for i, p, c in train:
        by_persona.setdefault(i, []).append((profiles[i].strengths, p, c))
    betas = []
    for _ in range(n_boot):
        picks = rng.choice(persona_ids, size=len(persona_ids), replace=True)
        sample = [obs for pid in picks for obs in by_persona[int(pid)]]
        betas.append(fit(sample).beta)
    return float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1500, help="number of personas")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run(n=args.n, seed=args.seed)
    params: ContestParams = out["params"]
    print(
        f"Prediction lift: {out['n_train']} train / {out['n_test']} test personas, "
        f"{out['n_observations']} held-out choices\n"
    )
    for scored in out["results"]:
        print(scored.line)

    lo, hi = out["beta_ci"]
    print(f"\n  fitted beta = {params.beta:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  fitted alpha = { {k: round(v, 3) for k, v in params.as_dict()['alpha'].items()} }")

    by_name = {s.name: s for s in out["results"]}
    real = by_name["contest model, real profile (V0)"].accuracy
    shuffled = by_name["contest model, SHUFFLED profile"].accuracy
    modal_acc = by_name["modal option (no profile)"].accuracy
    lift = real - shuffled

    oracle = by_name["oracle (true latent + true params)"].accuracy
    print(f"\n  headroom above V0 before the ceiling: {oracle - real:+.1%} "
          f"(oracle {oracle:.1%})")
    print(f"\n  Gate 0  real vs shuffled profile : {real:.1%} vs {shuffled:.1%}"
          f"  (lift {lift:+.1%})  -> {'PASS' if lift > 0.03 else 'FAIL'}")
    print(f"  Gate 1  beats modal choice       : {real:.1%} vs {modal_acc:.1%}"
          f"  -> {'PASS' if real > modal_acc + 0.02 else 'FAIL'}")
    print(f"  Gate 1  beta CI excludes zero    : [{lo:.3f}, {hi:.3f}]"
          f"  -> {'PASS' if lo > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
