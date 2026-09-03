"""Text-mode rendering: tables, bar charts, trees, grids.

Plots in this course are printed, not drawn. That is a deliberate choice —
the notebooks must run with nothing installed beyond Jupyter, and a text
bar chart survives being pasted into a terminal, a diff, or a commit
message. Where a real plot genuinely helps, the notebook offers it as a
clearly-marked optional matplotlib cell *below* the text version.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

BAR_FULL = "█"
BAR_PARTIALS = " ▏▎▍▌▋▊▉"  # eighths, for sub-character resolution


def table(rows: Sequence[Sequence[Any]],
          headers: Sequence[str] | None = None,
          *,
          align: str | Sequence[str] = "l") -> str:
    """Render rows as a padded text table.

    `align` is either one character applied to all columns or one per
    column: 'l' left, 'r' right, 'c' centre.
    """
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    head = [str(h) for h in headers] if headers else None
    ncols = max([len(r) for r in body] + [len(head) if head else 0] or [0])
    if ncols == 0:
        return ""
    for r in body:
        r.extend([""] * (ncols - len(r)))
    if head:
        head.extend([""] * (ncols - len(head)))

    aligns = list(align) if not isinstance(align, str) else list(align)
    if len(aligns) == 1:
        aligns = aligns * ncols
    aligns = (aligns + ["l"] * ncols)[:ncols]

    widths = [0] * ncols
    for r in ([head] if head else []) + body:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            w = widths[i]
            if aligns[i] == "r":
                out.append(cell.rjust(w))
            elif aligns[i] == "c":
                out.append(cell.center(w))
            else:
                out.append(cell.ljust(w))
        return "  ".join(out).rstrip()

    lines = []
    if head:
        lines.append(fmt_row(head))
        lines.append("  ".join("-" * w for w in widths))
    lines.extend(fmt_row(r) for r in body)
    return "\n".join(lines)


def bar_chart(pairs: Iterable[tuple[Any, float]],
              *,
              width: int = 40,
              title: str | None = None,
              value_fmt: str = "{:.3g}",
              maximum: float | None = None) -> str:
    """Horizontal text bar chart from (label, value) pairs."""
    items = [(str(k), float(v)) for k, v in pairs]
    if not items:
        return title or ""
    top = maximum if maximum is not None else max(max(v for _, v in items), 0.0)
    label_w = max(len(k) for k, _ in items)
    val_w = max(len(value_fmt.format(v)) for _, v in items)
    lines = [title] if title else []
    for label, value in items:
        frac = 0.0 if top <= 0 else max(0.0, value) / top
        lines.append(
            f"{label.ljust(label_w)} | {_bar(frac, width)} "
            f"{value_fmt.format(value).rjust(val_w)}"
        )
    return "\n".join(lines)


def _bar(fraction: float, width: int) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    eighths = round(fraction * width * 8)
    full, rem = divmod(eighths, 8)
    bar = BAR_FULL * full + (BAR_PARTIALS[rem] if rem else "")
    return bar.ljust(width)


def grid(cells: Sequence[Sequence[Any]], *, cell_width: int | None = None,
         border: bool = True) -> str:
    """Render a 2-D grid (gridworlds, N-queens boards, Sudoku)."""
    text = [[("" if c is None else str(c)) for c in row] for row in cells]
    w = cell_width or max((len(c) for row in text for c in row), default=1)
    ncols = max((len(row) for row in text), default=0)
    rule = "+" + "+".join("-" * (w + 2) for _ in range(ncols)) + "+"
    lines = []
    if border:
        lines.append(rule)
    for row in text:
        padded = list(row) + [""] * (ncols - len(row))
        lines.append("| " + " | ".join(c.center(w) for c in padded) + " |")
        if border:
            lines.append(rule)
    return "\n".join(lines)


def tree(root: Any,
         children: Callable[[Any], Sequence[Any]],
         label: Callable[[Any], str] = str,
         *,
         max_depth: int | None = None) -> str:
    """ASCII rendering of any tree, given child- and label- accessors."""
    lines: list[str] = []

    def walk(node: Any, prefix: str, is_last: bool, depth: int) -> None:
        connector = "" if depth == 0 else ("└── " if is_last else "├── ")
        lines.append(prefix + connector + label(node))
        if max_depth is not None and depth >= max_depth:
            return
        kids = list(children(node))
        if depth == 0:
            child_prefix = ""
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")
        for i, kid in enumerate(kids):
            walk(kid, child_prefix, i == len(kids) - 1, depth + 1)

    walk(root, "", True, 0)
    return "\n".join(lines)


def histogram(values: Iterable[float], *, bins: int = 10,
              width: int = 40, title: str | None = None) -> str:
    """Text histogram of a sequence of numbers."""
    data = [float(v) for v in values]
    if not data:
        return title or ""
    lo, hi = min(data), max(data)
    if hi == lo:
        hi = lo + 1.0
    counts = [0] * bins
    for v in data:
        idx = min(int((v - lo) / (hi - lo) * bins), bins - 1)
        counts[idx] += 1
    labels = [f"{lo + (hi - lo) * i / bins:.3g}" for i in range(bins)]
    return bar_chart(zip(labels, counts), width=width, title=title,
                     value_fmt="{:.0f}")
