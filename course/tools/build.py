#!/usr/bin/env python3
"""Build `course/notebooks/*.ipynb` from the percent-format sources in
`course/src/`.

Why go through a source format at all? Because a `.ipynb` is a JSON blob:
awkward to review, awkward to diff, and impossible to run without Jupyter.
The `src/*.py` files are plain Python — `python3 src/01_....py` executes the
lecture, `git diff` on them is readable, and `tools/verify.py` can check
every cell in a container with no Jupyter installed. The notebooks are
generated, committed, and are what a learner opens.

Source format (the "percent" convention, also understood by jupytext):

    # %% [markdown]
    # ## A heading
    #
    # Prose. Leading "# " is stripped.

    # %%
    print("a code cell")

Cell ids are derived from the notebook name and cell index, so rebuilding
unchanged sources produces a byte-identical file (checked by verify.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

COURSE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = COURSE_DIR / "src"
NB_DIR = COURSE_DIR / "notebooks"

MARKER = "# %%"
NBFORMAT = (4, 5)


class SourceError(ValueError):
    """A percent-format source file we can't parse."""


def parse_cells(text: str, *, origin: str = "<string>") -> list[tuple[str, str]]:
    """Split percent-format text into a list of (cell_type, source) pairs."""
    lines = text.splitlines()
    cells: list[tuple[str, list[str]]] = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.rstrip()
        if stripped == MARKER or stripped.startswith(MARKER + " "):
            tag = stripped[len(MARKER):].strip()
            if tag in ("", "[code]"):
                cells.append(("code", []))
            elif tag == "[markdown]":
                cells.append(("markdown", []))
            else:
                raise SourceError(f"{origin}:{lineno}: unknown cell tag {tag!r}")
            continue
        if not cells:
            if stripped:
                raise SourceError(
                    f"{origin}:{lineno}: content before the first '# %%' marker"
                )
            continue
        cells[-1][1].append(line)

    out: list[tuple[str, str]] = []
    for kind, body in cells:
        source = _demote_comments(body) if kind == "markdown" else "\n".join(body)
        source = source.strip("\n")
        if source.strip():
            out.append((kind, source))
    return out


def _demote_comments(lines: list[str]) -> str:
    """Turn `# text` comment lines back into markdown text."""
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
        elif stripped == "#":
            out.append("")
        elif stripped.startswith("# "):
            out.append(stripped[2:])
        elif stripped.startswith("#"):
            out.append(stripped[1:])
        else:
            out.append(line)
    return "\n".join(out)


def _cell_id(notebook_name: str, index: int) -> str:
    digest = hashlib.sha1(f"{notebook_name}:{index}".encode()).hexdigest()
    return digest[:12]


def to_notebook(cells: list[tuple[str, str]], *, name: str) -> dict[str, Any]:
    nb_cells = []
    for i, (kind, source) in enumerate(cells):
        cell: dict[str, Any] = {
            "cell_type": kind,
            "id": _cell_id(name, i),
            "metadata": {},
            "source": _as_source_lines(source),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "file_extension": ".py",
                "mimetype": "text/x-python",
            },
        },
        "nbformat": NBFORMAT[0],
        "nbformat_minor": NBFORMAT[1],
    }


def _as_source_lines(source: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline
    except the last."""
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build_one(src: Path, out_dir: Path) -> Path:
    cells = parse_cells(src.read_text(), origin=str(src))
    if not cells:
        raise SourceError(f"{src}: no cells found")
    nb = to_notebook(cells, name=src.stem)
    out = out_dir / f"{src.stem}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


def build_all(src_dir: Path = SRC_DIR, out_dir: Path = NB_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(src_dir.glob("*.py")):
        if src.name.startswith("_"):
            continue
        written.append(build_one(src, out_dir))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=SRC_DIR)
    ap.add_argument("--out-dir", type=Path, default=NB_DIR)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        written = build_all(args.src_dir, args.out_dir)
    except SourceError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        for path in written:
            n = len(json.loads(path.read_text())["cells"])
            print(f"built {path.relative_to(COURSE_DIR)}  ({n} cells)")
        if not written:
            print("no sources found in", args.src_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
