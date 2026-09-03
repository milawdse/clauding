"""Shared helpers for the *Reasoning & System 2* course.

Standard library only, on purpose: every notebook in `course/notebooks/`
must run in a bare `python3` with nothing installed but Jupyter itself.

Submodules
----------
check   tiny self-test harness used by every exercise and project
trace   Trace/Step recording and trace comparison (built in Module 1)
render  text-mode tables, bar charts, trees and grids (no matplotlib needed)
data    loaders for the repo's `data/*.jsonl` and bridges to `data_gen/`
"""

__all__ = ["check", "trace", "render", "data"]
