"""A very small self-test harness for the course exercises.

Design constraints, in order of importance:

1. **A fresh notebook must run top to bottom without raising.** Exercise
   stubs return `None`; their checkers must therefore *report* failure
   rather than blow up, so a learner can Run All and see a list of what is
   still to do instead of a traceback wall.
2. **Failures must say what went wrong**, including when the learner's code
   raises — the exception is caught and shown as the failure reason.
3. **Results must be machine-readable**, so `tools/verify.py` can assert
   that every reference solution actually passes its own checker.

Usage in a notebook::

    from csai.check import checker

    @checker("Exercise 1 - negate")
    def check_ex1():
        yield "negates an atom", negate("p"), ("not", "p")
        yield "is an involution", negate(negate("p")) == "p"

    check_ex1()

A yielded 3-tuple ``(label, got, want)`` passes when ``got == want``; a
2-tuple ``(label, ok)`` passes when ``ok`` is truthy. That is the whole
language.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Iterable, Iterator

#: Every checker run appends one record here. `tools/verify.py` reads it.
RESULTS: list[dict[str, Any]] = []

PASS_MARK = "✔"  # heavy check mark
FAIL_MARK = "✘"  # heavy ballot x
TODO_MARK = "○"  # hollow circle: stub not attempted yet


def reset() -> None:
    """Forget all recorded results (used by the verifier between runs)."""
    RESULTS.clear()


def summary() -> dict[str, int]:
    """Aggregate counts across every checker run so far."""
    return {
        "checkers": len(RESULTS),
        "checkers_passed": sum(1 for r in RESULTS if r["ok"]),
        "cases": sum(r["total"] for r in RESULTS),
        "cases_passed": sum(r["passed"] for r in RESULTS),
    }


def fmt(value: Any, limit: int = 100) -> str:
    """`repr` a value, truncated so a huge structure can't flood the cell."""
    try:
        text = repr(value)
    except Exception:  # a broken __repr__ shouldn't break the report
        text = f"<unreprable {type(value).__name__}>"
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _judge(case: Any) -> tuple[str, bool, str]:
    """Turn one yielded case into (label, ok, detail)."""
    if not isinstance(case, tuple) or len(case) not in (2, 3):
        return (fmt(case), False, "checker yielded something that is not a "
                                  "(label, ok) or (label, got, want) tuple")
    if len(case) == 2:
        label, ok = case
        if ok is None:
            return (str(label), False, "not implemented yet (got None)")
        return (str(label), bool(ok), "" if ok else "expected a truthy value")
    label, got, want = case
    if got is None and want is not None:
        return (str(label), False, f"not implemented yet (got None, want {fmt(want)})")
    ok = bool(got == want)
    return (str(label), ok, "" if ok else f"got {fmt(got)}, want {fmt(want)}")


def run(title: str, cases: Iterable[Any], *, quiet: bool = False) -> bool:
    """Run `cases` (an iterable of yielded case tuples) and print a report.

    Returns True when every case passed. Never raises: an exception from the
    learner's code is caught and reported as a failed case, because a stub
    that isn't written yet must not stop the rest of the notebook.
    """
    outcomes: list[tuple[str, bool, str]] = []
    crashed: str | None = None
    iterator: Iterator[Any]
    try:
        iterator = iter(cases)
    except Exception:
        iterator = iter(())
        crashed = _last_line(traceback.format_exc())

    while crashed is None:
        try:
            case = next(iterator)
        except StopIteration:
            break
        except Exception:
            # A generator cannot be resumed after it raises, so stop here and
            # report whatever it managed to yield before falling over.
            crashed = _last_line(traceback.format_exc())
            break
        outcomes.append(_judge(case))

    passed = sum(1 for _, ok, _ in outcomes if ok)
    total = len(outcomes) + (1 if crashed else 0)
    ok_all = crashed is None and passed == len(outcomes) and total > 0

    RESULTS.append(
        {
            "title": title,
            "ok": ok_all,
            "passed": passed,
            "total": total,
            "crashed": crashed,
            "failures": [(label, detail) for label, ok, detail in outcomes if not ok],
        }
    )

    if not quiet:
        unattempted = crashed is None and _looks_unattempted(outcomes)
        head = PASS_MARK if ok_all else (TODO_MARK if unattempted else FAIL_MARK)
        print(f"{head} {title}  --  {passed}/{total} checks passed")
        for label, ok, detail in outcomes:
            mark = PASS_MARK if ok else FAIL_MARK
            print(f"   {mark} {label}" + (f"  --  {detail}" if detail else ""))
        if crashed:
            print(f"   {FAIL_MARK} your code raised  --  {crashed}")
    return ok_all


def checker(title: str) -> Callable[[Callable[[], Iterable[Any]]], Callable[..., bool]]:
    """Decorator turning a case-yielding generator into a runnable checker."""

    def decorate(fn: Callable[[], Iterable[Any]]) -> Callable[..., bool]:
        def wrapped(*, quiet: bool = False) -> bool:
            try:
                cases = fn()
            except Exception:
                cases = ()
                RESULTS.append(
                    {
                        "title": title,
                        "ok": False,
                        "passed": 0,
                        "total": 1,
                        "crashed": _last_line(traceback.format_exc()),
                        "failures": [],
                    }
                )
                if not quiet:
                    print(f"{FAIL_MARK} {title}  --  checker itself raised: "
                          f"{RESULTS[-1]['crashed']}")
                return False
            return run(title, cases, quiet=quiet)

        wrapped.__name__ = getattr(fn, "__name__", "checker")
        wrapped.__doc__ = f"Run the checks for: {title}"
        wrapped.title = title  # type: ignore[attr-defined]
        return wrapped

    return decorate


def _looks_unattempted(outcomes: list[tuple[str, bool, str]]) -> bool:
    fails = [detail for _, ok, detail in outcomes if not ok]
    return bool(fails) and all("not implemented yet" in d for d in fails)


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"
