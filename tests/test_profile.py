import numpy as np

from glasser_puzzles.needs.constructs import Need, NeedProfile
from glasser_puzzles.profile.derive import GAP_THRESHOLD, MAX_HEURISTICS, derive
from glasser_puzzles.profile.heuristics import CONTRAST_HEURISTICS


def test_flat_profile_claims_nothing():
    """Asserting a pattern from a flat profile would be inventing one."""
    card = derive(NeedProfile(np.zeros(5), np.ones(5), 0.0))
    assert card.decision_heuristics == ()
    assert card.dominant is None


def test_strong_contrast_produces_its_heuristic():
    strengths = np.zeros(5)
    strengths[3] = 0.8  # freedom
    strengths[1] = -0.8  # belonging
    card = derive(NeedProfile(strengths, np.ones(5), 0.0))
    assert CONTRAST_HEURISTICS[(Need.FREEDOM, Need.BELONGING)] in card.decision_heuristics


def test_heuristics_are_capped():
    strengths = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    card = derive(NeedProfile(strengths, np.ones(5), 0.9))
    # up to MAX_HEURISTICS contrasts, plus at most one control heuristic
    assert len(card.decision_heuristics) <= MAX_HEURISTICS + 1


def test_gap_below_threshold_is_ignored():
    strengths = np.zeros(5)
    strengths[3] = GAP_THRESHOLD / 4
    strengths[1] = -GAP_THRESHOLD / 4
    card = derive(NeedProfile(strengths, np.ones(5), 0.0))
    assert CONTRAST_HEURISTICS[(Need.FREEDOM, Need.BELONGING)] not in card.decision_heuristics


def test_control_heuristic_follows_orientation():
    strengths = np.zeros(5)
    internal = derive(NeedProfile(strengths, np.ones(5), 0.8)).decision_heuristics
    external = derive(NeedProfile(strengths, np.ones(5), -0.8)).decision_heuristics
    assert internal and external
    assert internal != external


def test_vocabulary_covers_both_directions():
    from glasser_puzzles.puzzles.contrasts import CONTRAST_PAIRS

    for a, b in CONTRAST_PAIRS:
        assert (a, b) in CONTRAST_HEURISTICS
        assert (b, a) in CONTRAST_HEURISTICS
