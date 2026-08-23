from glasser_puzzles.puzzles.bank_loader import by_contrast, load_bank
from glasser_puzzles.puzzles.contrasts import CONTRAST_PAIRS
from glasser_puzzles.puzzles.validate import check


def test_bank_loads():
    assert len(load_bank()) == 30


def test_every_contrast_pair_is_covered():
    coverage = by_contrast()
    for pair in CONTRAST_PAIRS:
        assert len(coverage.get(pair, [])) >= 3, f"{pair} is under-covered"


def test_every_puzzle_passes_validation():
    problems = {p.id: check(p) for p in load_bank()}
    assert not any(problems.values()), {k: v for k, v in problems.items() if v}


def test_puzzle_ids_are_unique():
    ids = [p.id for p in load_bank()]
    assert len(set(ids)) == len(ids)


def test_contrast_needs_have_options():
    for puzzle in load_bank():
        needs = {o.need for o in puzzle.options}
        assert set(puzzle.contrast) <= needs
