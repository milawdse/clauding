import numpy as np
import pytest

from glasser_puzzles.needs.constructs import NEEDS, Need, need_index
from glasser_puzzles.needs.items import ITEMS, items_for
from glasser_puzzles.needs.scoring import NEUTRAL, neutral_responses, responses_from_vector, score


def test_yes_sayer_gets_a_flat_profile():
    """The whole point of ipsatization: agreeing with everything says nothing."""
    profile = score({item.id: 5.0 for item in ITEMS})
    assert np.allclose(profile.strengths, 0.0, atol=1e-9)
    assert profile.dominant is None


def test_no_sayer_gets_a_flat_profile():
    profile = score({item.id: 1.0 for item in ITEMS})
    assert np.allclose(profile.strengths, 0.0, atol=1e-9)


def test_neutral_responses_are_flat():
    profile = score(neutral_responses())
    assert np.allclose(profile.strengths, 0.0)
    assert profile.dominant is None


def test_reverse_items_are_flipped():
    """Agreeing with a reverse item must lower, not raise, that need."""
    responses = neutral_responses()
    reverse = next(i for i in items_for(Need.FREEDOM) if i.reverse)
    responses[reverse.id] = 5.0
    profile = score(responses)
    assert profile.strengths[need_index(Need.FREEDOM)] < 0


def test_forward_items_raise_their_need():
    responses = neutral_responses()
    forward = next(i for i in items_for(Need.POWER) if not i.reverse)
    responses[forward.id] = 5.0
    profile = score(responses)
    assert profile.strengths[need_index(Need.POWER)] > 0


def test_ordering_is_recovered():
    profile = score(
        responses_from_vector(
            {
                Need.SURVIVAL: -0.6,
                Need.BELONGING: -0.3,
                Need.POWER: 0.0,
                Need.FREEDOM: 0.8,
                Need.FUN: 0.4,
            }
        )
    )
    assert profile.dominant is Need.FREEDOM
    assert profile.secondary is Need.FUN


def test_disagreeing_items_lower_confidence():
    agree = neutral_responses()
    conflicted = neutral_responses()
    for i, item in enumerate(items_for(Need.FUN)):
        conflicted[item.id] = 1.0 if i % 2 else 5.0
    idx = need_index(Need.FUN)
    assert score(conflicted).confidence[idx] < score(agree).confidence[idx]


def test_missing_items_lower_confidence():
    full = neutral_responses()
    partial = {k: v for k, v in full.items() if k != items_for(Need.SURVIVAL)[0].id}
    idx = need_index(Need.SURVIVAL)
    assert score(partial).confidence[idx] < score(full).confidence[idx]


def test_out_of_range_response_rejected():
    responses = neutral_responses()
    responses["S1"] = 9.0
    with pytest.raises(ValueError):
        score(responses)


def test_unknown_item_rejected():
    with pytest.raises(KeyError):
        score({"NOPE": 3.0})


def test_every_need_has_balanced_keying():
    for need in NEEDS:
        items = items_for(need)
        assert len(items) == 4
        assert sum(i.reverse for i in items) == 2, f"{need} is not direction-balanced"
