import numpy as np

from glasser_puzzles.needs.constructs import NeedProfile, Need, need_index
from glasser_puzzles.predict.contest import ContestParams
from glasser_puzzles.puzzles.bank_loader import load_bank
from glasser_puzzles.update.belief import Belief, update

BANK = load_bank()
PUZZLE = next(p for p in BANK if p.id == "BF1")
PARAMS = ContestParams(beta=2.3, alpha=np.zeros(5))


def _belief():
    return Belief.from_profile(NeedProfile(np.zeros(5), np.ones(5), 0.0))


def test_chosen_need_moves_up():
    freedom_option = next(o for o in PUZZLE.options if o.need is Need.FREEDOM)
    after = update(_belief(), PUZZLE, freedom_option.id, PARAMS)
    assert after.mean[need_index(Need.FREEDOM)] > 0


def test_update_is_monotone_over_repeats():
    belief = _belief()
    option = next(o for o in PUZZLE.options if o.need is Need.FREEDOM)
    idx = need_index(Need.FREEDOM)
    previous = belief.mean[idx]
    for _ in range(4):
        belief = update(belief, PUZZLE, option.id, PARAMS)
        assert belief.mean[idx] > previous
        previous = belief.mean[idx]


def test_variance_shrinks_for_contested_needs():
    before = _belief()
    after = update(before, PUZZLE, PUZZLE.options[0].id, PARAMS)
    idx = need_index(PUZZLE.options[0].need)
    assert after.variance[idx] < before.variance[idx]


def test_needs_absent_from_the_puzzle_are_untouched():
    """A puzzle with no Survival option tells us nothing about Survival."""
    assert not any(o.need is Need.SURVIVAL for o in PUZZLE.options)
    before = _belief()
    after = update(before, PUZZLE, PUZZLE.options[0].id, PARAMS)
    idx = need_index(Need.SURVIVAL)
    assert after.variance[idx] == before.variance[idx]
    assert abs(after.mean[idx] - before.mean[idx]) < 1e-6


def test_low_confidence_needs_move_further():
    """We should revise hardest what we were least sure of."""
    option = next(o for o in PUZZLE.options if o.need is Need.FREEDOM)
    idx = need_index(Need.FREEDOM)

    confident = np.ones(5)
    unsure = np.ones(5)
    unsure[idx] = 0.2

    moved_confident = update(
        Belief.from_profile(NeedProfile(np.zeros(5), confident, 0.0)), PUZZLE, option.id, PARAMS
    ).mean[idx]
    moved_unsure = update(
        Belief.from_profile(NeedProfile(np.zeros(5), unsure, 0.0)), PUZZLE, option.id, PARAMS
    ).mean[idx]
    assert moved_unsure > moved_confident
