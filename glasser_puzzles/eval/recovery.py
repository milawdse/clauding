"""Does the questionnaire actually measure what it claims to?

Generate personas with known latent need vectors, simulate their questionnaire
responses through the IRT model, score those responses, and correlate the
result against the truth.  No language model, no GPU, a couple of seconds.

This runs first in the build order for a reason: if the instrument cannot
recover a need ordering from its own simulated data, nothing downstream can
work, and that is worth finding out before anything else is built.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from ..needs.constructs import NEEDS
from ..needs.scoring import score
from ..synth.irt import make_item_bank, simulate_responses
from ..synth.personas import sample_personas


@dataclass
class RecoveryResult:
    per_need_r: dict[str, float]
    mean_r: float
    dominant_accuracy: float
    top2_overlap: float
    n: int

    def passed(self, mean_threshold: float = 0.80, min_threshold: float = 0.75) -> bool:
        """Two thresholds, because per-need recovery genuinely varies.

        How well a need is recovered depends on the discrimination of the four
        items that happen to measure it.  Across seeds this lands between about
        r = 0.77 and r = 0.87, so gating on a single per-need floor of 0.80
        would fail on item draws that are perfectly acceptable.  The fix for a
        need that sits at the bottom of that range is better-written items, not
        more of them -- which is a note for whoever revises the bank, not a
        reason to fail the build.
        """
        return (
            self.mean_r >= mean_threshold
            and min(self.per_need_r.values()) >= min_threshold
        )


def run(n: int = 2000, seed: int = 0) -> RecoveryResult:
    rng = np.random.default_rng(seed)
    item_bank = make_item_bank(rng)
    personas = sample_personas(n, rng)

    truth = np.zeros((n, len(NEEDS)))
    scored = np.zeros((n, len(NEEDS)))
    for i, persona in enumerate(personas):
        responses = simulate_responses(persona, item_bank, rng)
        truth[i] = persona.latent
        scored[i] = score(responses).strengths

    per_need = {
        need.value: float(np.corrcoef(truth[:, j], scored[:, j])[0, 1])
        for j, need in enumerate(NEEDS)
    }
    dominant_hits = (truth.argmax(axis=1) == scored.argmax(axis=1)).mean()
    top2_truth = np.argsort(-truth, axis=1)[:, :2]
    top2_scored = np.argsort(-scored, axis=1)[:, :2]
    overlap = np.mean([len(set(a) & set(b)) / 2 for a, b in zip(top2_truth, top2_scored)])

    return RecoveryResult(
        per_need_r=per_need,
        mean_r=float(np.mean(list(per_need.values()))),
        dominant_accuracy=float(dominant_hits),
        top2_overlap=float(overlap),
        n=n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean-threshold", type=float, default=0.80)
    parser.add_argument("--min-threshold", type=float, default=0.75)
    args = parser.parse_args()

    result = run(n=args.n, seed=args.seed)
    print(f"Instrument recovery over {result.n} synthetic personas\n")
    for need, r in result.per_need_r.items():
        flag = "ok" if r >= args.min_threshold else "LOW"
        print(f"  {need:<12} r = {r:.3f}  {flag}")
    print(f"\n  mean r                 {result.mean_r:.3f}")
    print(f"  dominant need correct  {result.dominant_accuracy:.1%}")
    print(f"  top-2 overlap          {result.top2_overlap:.1%}")
    verdict = "PASS" if result.passed(args.mean_threshold, args.min_threshold) else "FAIL"
    print(
        f"\n  gate (mean r >= {args.mean_threshold}, "
        f"min r >= {args.min_threshold}): {verdict}"
    )


if __name__ == "__main__":
    main()
