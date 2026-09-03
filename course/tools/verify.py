#!/usr/bin/env python3
"""Verify the whole course: every notebook runs, every exercise is solvable.

Checks, in order:

1. `build.py` is idempotent — rebuilding `src/` into a temp dir reproduces
   the committed `notebooks/` byte for byte, so sources and notebooks can
   never silently drift apart.
2. Every notebook is valid nbformat JSON.
3. **Stub mode** — each notebook's code cells run top to bottom in a fresh
   interpreter without raising. This is what a learner sees on Run All.
4. **Solution mode** — same, with `solutions/mNN.py` injected after each
   stub cell. Every checker must pass, which proves each exercise has a
   working reference solution and each checker actually accepts it.

Usage:
    python3 tools/verify.py                # everything
    python3 tools/verify.py 01 07          # just those modules
    python3 tools/verify.py --mode stub    # skip the solution pass
"""

from __future__ import annotations

import argparse
import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parents[1]
NB_DIR = COURSE_DIR / "notebooks"
SRC_DIR = COURSE_DIR / "src"

PASS, FAIL, SKIP = "✔", "✘", "-"


def check_build_idempotent() -> list[str]:
    """Return a list of problem descriptions (empty means all good)."""
    sys.path.insert(0, str(COURSE_DIR / "tools"))
    import build  # noqa: E402

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            build.build_all(SRC_DIR, tmp_dir)
        except build.SourceError as exc:
            return [f"source does not parse: {exc}"]
        for src in sorted(SRC_DIR.glob("*.py")):
            if src.name.startswith("_"):
                continue
            name = f"{src.stem}.ipynb"
            committed, rebuilt = NB_DIR / name, tmp_dir / name
            if not committed.exists():
                problems.append(f"{name}: missing from notebooks/ (run make build)")
            elif not filecmp.cmp(committed, rebuilt, shallow=False):
                problems.append(f"{name}: stale — src/ changed, run make build")
        for nb in sorted(NB_DIR.glob("*.ipynb")):
            if not (SRC_DIR / f"{nb.stem}.py").exists():
                problems.append(f"{nb.name}: orphan — no matching src/{nb.stem}.py")
    return problems


def run_notebook(nb: Path, mode: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(COURSE_DIR / "tools" / "run_notebook.py"),
         str(nb), "--mode", mode],
        capture_output=True,
        text=True,
    )
    marker = "\n__REPORT__"
    if marker in proc.stdout:
        return json.loads(proc.stdout.rsplit(marker, 1)[1])
    return {
        "notebook": nb.name, "mode": mode, "ok": False, "failed_cell": None,
        "cells": 0, "error": proc.stderr.strip() or "no report produced",
        "checks": None, "check_failures": [],
    }


def describe(report: dict) -> str:
    if report["ok"]:
        checks = report.get("checks") or {}
        detail = ""
        if checks.get("cases"):
            detail = f", {checks['cases_passed']}/{checks['cases']} checks"
        return f"{PASS} {report['notebook']:<34} {report['mode']:<9} " \
               f"{report['cells']} cells{detail}"
    head = f"{FAIL} {report['notebook']:<34} {report['mode']:<9} "
    if report["error"]:
        last = [l for l in report["error"].strip().splitlines() if l.strip()][-1]
        return head + f"cell {report['failed_cell']} raised: {last}"
    fails = report.get("check_failures") or []
    bits = []
    for f in fails:
        reasons = [f"{label}: {detail}" for label, detail in f["failures"]]
        if f["crashed"]:
            reasons.append(f"raised {f['crashed']}")
        bits.append(f"{f['title']} ({'; '.join(reasons)})")
    return head + "checks failed: " + " | ".join(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="*", help="module number prefixes, e.g. 01 07")
    ap.add_argument("--mode", choices=("stub", "solution", "both"), default="both")
    args = ap.parse_args(argv)

    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if args.modules:
        notebooks = [nb for nb in notebooks
                     if any(nb.name.startswith(m) for m in args.modules)]
    if not notebooks:
        print("no notebooks to verify (build them first: make build)")
        return 1

    failures = 0

    if not args.modules:
        print("== build idempotence ==")
        problems = check_build_idempotent()
        for p in problems:
            print(f"{FAIL} {p}")
        failures += len(problems)
        if not problems:
            print(f"{PASS} notebooks/ matches src/")
        print()

    print("== notebook JSON ==")
    for nb in notebooks:
        try:
            doc = json.loads(nb.read_text())
            assert doc["nbformat"] == 4, "not nbformat 4"
            assert doc["cells"], "no cells"
            print(f"{PASS} {nb.name:<34} {len(doc['cells'])} cells")
        except Exception as exc:
            print(f"{FAIL} {nb.name:<34} {exc}")
            failures += 1
    print()

    modes = ("stub", "solution") if args.mode == "both" else (args.mode,)
    for mode in modes:
        print(f"== execute: {mode} mode ==")
        for nb in notebooks:
            report = run_notebook(nb, mode)
            print(describe(report))
            if not report["ok"]:
                failures += 1
                if report["error"]:
                    print("     " + report["error"].strip().replace("\n", "\n     "))
        print()

    if failures:
        print(f"{FAIL} {failures} problem(s)")
        return 1
    print(f"{PASS} all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
