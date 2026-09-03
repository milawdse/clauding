"""Recording and comparing chains of deliberation.

This is the course's through-line object. A `Trace` is just an ordered list
of `Step`s, each pairing the *action taken* with the *state that resulted*.
Every classical method in this course — DPLL's decision stack, A*'s expanded
path, a Prolog proof, a STRIPS plan, an MDP rollout — can be recorded as one,
and so can a language model's chain of thought.

That shared shape is what makes the central measurement of the course
possible: given a predicted trace and a ground-truth one, don't just ask
"was the final answer right?" but "at which step did they diverge?"
(`first_divergence`). A model that reaches the right answer through a trace
that diverged at step 1 got lucky; a model whose trace matches throughout is
executing the algorithm. Module 1 has you build this; Modules 2-12 import it.

A reference implementation lives here rather than in the notebook so later
modules have one canonical version to rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class Step:
    """One deliberation step: what was done, and what state it produced."""

    index: int
    action: str
    state: Any = None
    note: str = ""

    def __str__(self) -> str:
        head = f"Step {self.index}: {self.action}"
        if self.state is not None:
            head += f" -> {render_state(self.state)}"
        if self.note:
            head += f"   ({self.note})"
        return head


@dataclass
class Trace:
    """An ordered record of deliberation, with an optional initial state."""

    name: str = ""
    initial: Any = None
    steps: list[Step] = field(default_factory=list)
    result: Any = None

    def step(self, action: str, state: Any = None, note: str = "") -> Step:
        """Append a step and return it."""
        s = Step(index=len(self.steps) + 1, action=action, state=state, note=note)
        self.steps.append(s)
        return s

    def finish(self, result: Any) -> "Trace":
        """Record the final answer and return self (so it chains)."""
        self.result = result
        return self

    @property
    def states(self) -> list[Any]:
        """States after each step, not including the initial state."""
        return [s.state for s in self.steps]

    @property
    def actions(self) -> list[str]:
        return [s.action for s in self.steps]

    @property
    def final_state(self) -> Any:
        return self.steps[-1].state if self.steps else self.initial

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, i: int) -> Step:
        return self.steps[i]

    def render(self, *, show_initial: bool = True) -> str:
        lines: list[str] = []
        if self.name:
            lines.append(self.name)
        if show_initial and self.initial is not None:
            lines.append(f"Initial state: {render_state(self.initial)}")
        lines.extend(str(s) for s in self.steps)
        if self.result is not None:
            lines.append(f"Answer: {self.result}")
        return "\n".join(lines)

    def show(self) -> None:
        print(self.render())

    def __str__(self) -> str:
        return self.render()


def render_state(state: Any) -> str:
    """Compact one-line rendering of a state, stable across runs."""
    if isinstance(state, dict):
        return ", ".join(f"{k}={state[k]}" for k in state)
    if isinstance(state, (list, tuple)):
        return "[" + ", ".join(render_state(x) for x in state) + "]"
    if isinstance(state, (set, frozenset)):
        return "{" + ", ".join(sorted(str(x) for x in state)) + "}"
    return str(state)


# --------------------------------------------------------------------------
# Comparing two traces
# --------------------------------------------------------------------------


@dataclass
class TraceDiff:
    """The result of comparing a predicted trace against a gold one."""

    matches: list[bool]
    first_divergence: int | None  # 1-based step index, or None if identical
    length_mismatch: bool
    predicted_len: int
    gold_len: int

    @property
    def step_accuracy(self) -> float:
        """Fraction of gold steps whose state was reproduced exactly."""
        if not self.matches:
            return 1.0 if self.gold_len == 0 else 0.0
        return sum(self.matches) / self.gold_len if self.gold_len else 0.0

    @property
    def identical(self) -> bool:
        return self.first_divergence is None and not self.length_mismatch

    def render(self) -> str:
        lines = [
            f"predicted {self.predicted_len} steps, gold {self.gold_len} steps",
            "per-step state match: "
            + "".join("." if m else "X" for m in self.matches),
            f"step accuracy: {self.step_accuracy:.0%}",
        ]
        if self.first_divergence is None:
            lines.append("first divergence: none — traces agree")
        else:
            lines.append(f"first divergence: step {self.first_divergence}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _states_of(trace: Trace | Sequence[Any]) -> list[Any]:
    return trace.states if isinstance(trace, Trace) else list(trace)


def diff_traces(predicted: Trace | Sequence[Any],
                gold: Trace | Sequence[Any]) -> TraceDiff:
    """Compare two traces step by step.

    Accepts either `Trace` objects or bare sequences of states, so it works
    equally on a simulator's output and on states parsed out of a model's
    natural-language chain of thought.
    """
    pred_states = _states_of(predicted)
    gold_states = _states_of(gold)
    matches = [
        i < len(pred_states) and pred_states[i] == gold_states[i]
        for i in range(len(gold_states))
    ]
    first = next((i + 1 for i, m in enumerate(matches) if not m), None)
    return TraceDiff(
        matches=matches,
        first_divergence=first,
        length_mismatch=len(pred_states) != len(gold_states),
        predicted_len=len(pred_states),
        gold_len=len(gold_states),
    )


def first_wrong_step(predicted: Trace | Sequence[Any],
                     gold: Trace | Sequence[Any]) -> int | None:
    """1-based index of the first step whose state is wrong, else None."""
    return diff_traces(predicted, gold).first_divergence


def step_accuracy(predicted: Trace | Sequence[Any],
                  gold: Trace | Sequence[Any]) -> float:
    """Fraction of gold steps reproduced exactly."""
    return diff_traces(predicted, gold).step_accuracy
