"""STRIPS planning: action schemas, progression search, relaxed heuristics.

The canonical version of what you build in Module 9.

A **state** is a frozenset of ground fluents — tuples like `("on", "a", "b")`.
An **action schema** has parameters, preconditions, an add list and a delete
list; *grounding* substitutes objects for parameters to give concrete actions.

The representational commitment that makes planning work is the **closed-world
assumption**: whatever is not in the state is false. That is what lets an
action list only what it changes, and it is STRIPS's answer to the frame
problem.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from csai import fol, search
from csai.fol import Var

Fluent = tuple


@dataclass(frozen=True)
class Action:
    """A ground action: no variables left."""

    name: str
    args: tuple = ()
    preconditions: frozenset = frozenset()
    add: frozenset = frozenset()
    delete: frozenset = frozenset()

    def applicable(self, state: frozenset) -> bool:
        return self.preconditions <= state

    def apply(self, state: frozenset) -> frozenset:
        return (state - self.delete) | self.add

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.args))})"


@dataclass
class ActionSchema:
    """A parameterised action, to be ground over the problem's objects."""

    name: str
    parameters: tuple
    preconditions: tuple = ()
    add: tuple = ()
    delete: tuple = ()
    distinct: bool = True        # parameters must bind to different objects

    def ground(self, objects: Sequence[str]) -> list[Action]:
        actions = []
        for combo in itertools.product(objects, repeat=len(self.parameters)):
            if self.distinct and len(set(combo)) != len(combo):
                continue
            subst = dict(zip(self.parameters, combo))
            actions.append(Action(
                self.name, combo,
                frozenset(fol.substitute(p, subst) for p in self.preconditions),
                frozenset(fol.substitute(p, subst) for p in self.add),
                frozenset(fol.substitute(p, subst) for p in self.delete),
            ))
        return actions


@dataclass
class PlanningProblem:
    """Objects, action schemas, an initial state and a goal."""

    objects: tuple
    schemas: tuple
    initial: frozenset
    goal: frozenset
    _actions: list = field(default=None, repr=False)

    @property
    def actions(self) -> list[Action]:
        if self._actions is None:
            self._actions = [a for s in self.schemas for a in s.ground(self.objects)]
        return self._actions

    def applicable(self, state: frozenset) -> list[Action]:
        return [a for a in self.actions if a.applicable(state)]

    def achieved(self, state: frozenset) -> bool:
        return self.goal <= state


# --------------------------------------------------------------------------
# Validation — a plan is a claim, and claims get checked
# --------------------------------------------------------------------------

@dataclass
class Validation:
    """The verdict on a proposed plan, and where it went wrong."""

    valid: bool
    reason: str = ""
    failed_step: int | None = None      # 1-based
    states: list = field(default_factory=list)
    missing: frozenset = frozenset()

    def __bool__(self) -> bool:
        return self.valid

    def render(self) -> str:
        if self.valid:
            return f"valid: {len(self.states) - 1} steps reach the goal"
        head = f"invalid at step {self.failed_step}: {self.reason}"
        if self.missing:
            head += "\n  unsatisfied: " + ", ".join(
                sorted(map(str, self.missing)))
        return head

    def __str__(self) -> str:
        return self.render()


def validate(problem: PlanningProblem, plan: Sequence[Action]) -> Validation:
    """Execute a plan step by step and say precisely where it fails.

    This is the cheap half of planning: finding a plan is hard, checking one
    is linear. Any system that proposes plans should be paired with one of
    these.
    """
    state = problem.initial
    states = [state]
    for i, action in enumerate(plan, start=1):
        if action not in problem.actions:
            return Validation(False, f"{action} is not a legal action",
                              i, states)
        if not action.applicable(state):
            return Validation(False, f"preconditions of {action} do not hold",
                              i, states, action.preconditions - state)
        state = action.apply(state)
        states.append(state)
    if not problem.achieved(state):
        return Validation(False, "the plan ends without achieving the goal",
                          len(plan), states, problem.goal - state)
    return Validation(True, "", None, states)


# --------------------------------------------------------------------------
# Heuristics from the delete relaxation
# --------------------------------------------------------------------------

def relaxed_reachable(problem: PlanningProblem, state: frozenset) -> frozenset:
    """Everything reachable if actions never delete anything.

    The delete relaxation is the single most productive idea in domain-
    independent planning: drop the delete lists and the problem becomes
    monotone, so a fixpoint computes what is achievable in polynomial time.
    """
    reached = set(state)
    changed = True
    while changed:
        changed = False
        for action in problem.actions:
            if action.preconditions <= reached and not action.add <= reached:
                reached |= action.add
                changed = True
    return frozenset(reached)


def h_goals_unmet(problem: PlanningProblem, state: frozenset) -> float:
    """Count unachieved goal fluents. Cheap, weak, and not admissible."""
    return len(problem.goal - state)


def h_max(problem: PlanningProblem, state: frozenset) -> float:
    """Cost of the most expensive goal fluent in the relaxed problem.

    Admissible: no real plan can be shorter than the hardest single goal is
    to reach even with deletes ignored.
    """
    cost = _relaxed_costs(problem, state)
    if any(g not in cost for g in problem.goal):
        return float("inf")
    return max((cost[g] for g in problem.goal), default=0.0)


def h_add(problem: PlanningProblem, state: frozenset) -> float:
    """Sum of the relaxed costs of the goal fluents. Informative, inadmissible."""
    cost = _relaxed_costs(problem, state)
    if any(g not in cost for g in problem.goal):
        return float("inf")
    return sum(cost[g] for g in problem.goal)


def _relaxed_costs(problem: PlanningProblem, state: frozenset) -> dict:
    """Cheapest additive cost of each fluent in the delete-relaxed problem."""
    cost = {f: 0.0 for f in state}
    changed = True
    while changed:
        changed = False
        for action in problem.actions:
            if not action.preconditions <= set(cost):
                continue
            action_cost = 1.0 + sum(cost[p] for p in action.preconditions)
            for f in action.add:
                if action_cost < cost.get(f, float("inf")):
                    cost[f] = action_cost
                    changed = True
    return cost


def h_ff(problem: PlanningProblem, state: frozenset) -> float:
    """Length of a relaxed plan — the FF heuristic (Hoffmann & Nebel, 2001).

    Build the relaxed planning graph, then extract an actual (relaxed) plan
    by regressing from the goal. Counting *that* plan's actions is far more
    informative than summing per-fluent costs, because it does not
    double-count actions that achieve several goals at once.
    """
    layers, action_layers = _relaxed_graph(problem, state)
    if layers is None:
        return float("inf")

    needed = {len(layers) - 1: set(problem.goal)}
    chosen: set = set()
    for level in range(len(layers) - 1, 0, -1):
        for fluent in list(needed.get(level, ())):
            if fluent in layers[level - 1]:
                needed.setdefault(level - 1, set()).add(fluent)
                continue
            action = next((a for a in action_layers[level - 1]
                           if fluent in a.add), None)
            if action is None:
                continue
            chosen.add((level, action))
            for p in action.preconditions:
                needed.setdefault(level - 1, set()).add(p)
    return float(len(chosen))


def _relaxed_graph(problem: PlanningProblem, state: frozenset):
    """Fluent layers and the action layer that produced each."""
    layers = [frozenset(state)]
    action_layers = []
    while not problem.goal <= layers[-1]:
        applicable = [a for a in problem.actions if a.preconditions <= layers[-1]]
        nxt = layers[-1] | frozenset(f for a in applicable for f in a.add)
        if nxt == layers[-1]:
            return None, None           # goal unreachable even relaxed
        action_layers.append(applicable)
        layers.append(nxt)
    return layers, action_layers


HEURISTICS = {
    "none": lambda p, s: 0.0,
    "goals unmet": h_goals_unmet,
    "h_max": h_max,
    "h_add": h_add,
    "h_ff": h_ff,
}


# --------------------------------------------------------------------------
# Progression search
# --------------------------------------------------------------------------

class ForwardPlanning(search.Problem):
    """A planning problem viewed as state-space search, so A* applies."""

    def __init__(self, problem: PlanningProblem, heuristic: str = "h_ff"):
        self.problem = problem
        self.h = HEURISTICS[heuristic]

    def initial(self):
        return self.problem.initial

    def is_goal(self, state):
        return self.problem.achieved(state)

    def actions(self, state):
        return self.problem.applicable(state)

    def result(self, state, action):
        return action.apply(state)

    def heuristic(self, state):
        return self.h(self.problem, state)


def plan(problem: PlanningProblem, heuristic: str = "h_ff", **kw):
    """Find a plan by A* over the state space. Returns a SearchResult."""
    return search.astar(ForwardPlanning(problem, heuristic), **kw)


# --------------------------------------------------------------------------
# The blocks world
# --------------------------------------------------------------------------

B, FROM, TO = Var("B"), Var("From"), Var("To")

MOVE = ActionSchema(
    name="move",
    parameters=(B, FROM, TO),
    preconditions=(("on", B, FROM), ("clear", B), ("clear", TO), ("block", TO)),
    add=(("on", B, TO), ("clear", FROM)),
    delete=(("on", B, FROM), ("clear", TO)),
)

MOVE_TO_TABLE = ActionSchema(
    name="to_table",
    parameters=(B, FROM),
    preconditions=(("on", B, FROM), ("clear", B), ("block", FROM)),
    add=(("on", B, "table"), ("clear", FROM)),
    delete=(("on", B, FROM),),
)


def blocks_world(blocks: Sequence[str], initial_on: dict, goal_on: dict
                 ) -> PlanningProblem:
    """Build a blocks-world problem from `{block: what it sits on}` maps."""
    objects = tuple(blocks) + ("table",)
    state = set()
    for b in blocks:
        state.add(("block", b))
        state.add(("on", b, initial_on[b]))
    for b in blocks:
        if b not in initial_on.values():
            state.add(("clear", b))
    return PlanningProblem(
        objects=objects,
        schemas=(MOVE, MOVE_TO_TABLE),
        initial=frozenset(state),
        goal=frozenset(("on", b, t) for b, t in goal_on.items()),
    )


def show_blocks(state: frozenset) -> str:
    """Render a blocks-world state as towers."""
    on = {f[1]: f[2] for f in state if f[0] == "on"}
    tops = [b for b in on if b not in on.values()]
    towers = []
    for top in sorted(tops):
        stack, current = [], top
        while current != "table":
            stack.append(current)
            current = on.get(current, "table")
        towers.append(stack[::-1])          # bottom block first
    width = max((len(t) for t in towers), default=0)
    lines = []
    for level in range(width - 1, -1, -1):
        lines.append("  ".join(t[level] if level < len(t) else " "
                               for t in towers))
    lines.append("=" * max(len(line) for line in lines) if lines else "=")
    return "\n".join(lines)
