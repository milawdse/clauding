#!/usr/bin/env python3
"""Execute one notebook's code cells in a plain Python process.

Used by `tools/verify.py`, which runs this as a subprocess so that each
notebook gets a genuinely fresh interpreter. Two modes:

  stub      run the notebook exactly as a learner first opens it. Every cell
            must execute without raising — exercise stubs return None and
            their checkers report failures rather than throwing, so a fresh
            Run All is expected to be clean.

  solution  same, but after any cell containing a `# TODO:` marker, the
            module's reference solutions are exec'd into the live namespace,
            replacing the stubs. Every checker must then pass. This is what
            proves each exercise is solvable and each checker is correct.

Prints a JSON report on stdout. Exit status 0 means the cells ran; the
report says whether the checks passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parents[1]
TODO_MARKER = "# TODO:"


def code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    out = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        out.append("".join(source) if isinstance(source, list) else source)
    return out


def run(nb_path: Path, mode: str) -> dict:
    solutions = COURSE_DIR / "solutions" / f"m{nb_path.stem.split('_')[0]}.py"
    solution_src = solutions.read_text() if (mode == "solution" and solutions.exists()) else None

    os.chdir(COURSE_DIR / "notebooks")
    if str(COURSE_DIR) not in sys.path:
        sys.path.insert(0, str(COURSE_DIR))

    ns: dict = {"__name__": "__main__", "__file__": str(nb_path)}
    cells = code_cells(nb_path)
    for i, source in enumerate(cells, start=1):
        try:
            exec(compile(source, f"{nb_path.name}[cell {i}]", "exec"), ns)
        except Exception:
            return {
                "notebook": nb_path.name,
                "mode": mode,
                "ok": False,
                "failed_cell": i,
                "cells": len(cells),
                "error": traceback.format_exc(),
                "checks": None,
            }
        if solution_src is not None and TODO_MARKER in source:
            # Re-exec'ing is idempotent: it only rebinds the solution names.
            exec(compile(solution_src, str(solutions), "exec"), ns)

    from csai import check as check_mod

    failures = [
        {"title": r["title"], "failures": r["failures"], "crashed": r["crashed"]}
        for r in check_mod.RESULTS
        if not r["ok"]
    ]
    checks = check_mod.summary()
    ok = mode != "solution" or not failures
    return {
        "notebook": nb_path.name,
        "mode": mode,
        "ok": ok,
        "failed_cell": None,
        "cells": len(cells),
        "error": None,
        "checks": checks,
        "check_failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--mode", choices=("stub", "solution"), default="stub")
    args = ap.parse_args(argv)
    report = run(args.notebook.resolve(), args.mode)
    sys.stdout.write("\n__REPORT__" + json.dumps(report) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
