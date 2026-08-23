"""Execute a PoT (Program of Thoughts) program and grade it.

Unlike the CoT setup, correctness here is decided by *running* the model's
generated program and checking what it prints -- not by comparing text. The
program is executed in a separate `python3 -I` subprocess (isolated mode:
no user site-packages, cwd not on sys.path) with a short timeout, so a
malformed or pathological model generation can't hang or touch this
process. This is a lightweight sandbox suitable for local research use, not
a hardened one -- don't point it at untrusted programs from strangers.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

from pot_library import POT_LIBRARY_SOURCE

TIMEOUT_SECONDS = 5


def run_program(program: str, timeout: float = TIMEOUT_SECONDS) -> Optional[str]:
    """Run `program` (a PoT completion) against the Cube library and return
    the last non-empty line it printed, lowercased/stripped -- or None if it
    errored, timed out, or printed nothing."""
    full_source = POT_LIBRARY_SOURCE + "\n" + program
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", full_source],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1].lower()


def score_program(program: str, expected_answer: str) -> float:
    """Mirrors reasoning-gym's score_answer: 1.0 exact match, 0.01 wrong but
    the program ran and printed something, 0.0 for no output/error/timeout."""
    output = run_program(program)
    if output is None:
        return 0.0
    return 1.0 if output == expected_answer.strip().lower() else 0.01


if __name__ == "__main__":
    demo = (
        'cube = Cube(top="pink", right="gray", front="orange", '
        'left="purple", back="indigo", bottom="cyan")\n'
        'cube.rotate_to_top("bottom")\n'
        "print(cube.back)\n"
    )
    print("output:", run_program(demo))
    print("score:", score_program(demo, "orange"))
