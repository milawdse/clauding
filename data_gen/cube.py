"""Cube rotation mechanics, ported verbatim from reasoning-gym so that our
generated data (and the intermediate reasoning traces we add on top) stays
bit-for-bit consistent with the upstream task's answer key and scoring.

Source: reasoning_gym/cognition/color_cube_rotation.py
(open-thought/reasoning-gym, Apache-2.0).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    WHITE = "white"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"
    GOLD = "gold"
    SILVER = "silver"
    INDIGO = "indigo"
    VIOLET = "violet"


class Side(str, Enum):
    TOP = "top"
    RIGHT = "right"
    FRONT = "front"
    LEFT = "left"
    BACK = "back"
    BOTTOM = "bottom"


@dataclass
class Cube:
    """Represents a cube with colored sides."""

    colors: dict[Side, Color]

    def rotate_front_to_top(self) -> None:
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.FRONT]
        self.colors[Side.FRONT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.TOP]

    def rotate_right_to_top(self) -> None:
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.RIGHT]
        self.colors[Side.RIGHT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.LEFT]
        self.colors[Side.LEFT] = old[Side.TOP]

    def rotate_back_to_top(self) -> None:
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.FRONT]
        self.colors[Side.FRONT] = old[Side.TOP]

    def rotate_left_to_top(self) -> None:
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.LEFT]
        self.colors[Side.LEFT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.RIGHT]
        self.colors[Side.RIGHT] = old[Side.TOP]

    def rotate_bottom_to_top(self) -> None:
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.TOP]
        self.colors[Side.FRONT] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.FRONT]

    def state(self) -> dict[str, str]:
        return {side.value: self.colors[side].value for side in Side}


ROTATION_METHOD = {
    Side.FRONT: Cube.rotate_front_to_top,
    Side.RIGHT: Cube.rotate_right_to_top,
    Side.BACK: Cube.rotate_back_to_top,
    Side.LEFT: Cube.rotate_left_to_top,
    Side.BOTTOM: Cube.rotate_bottom_to_top,
}


def rotate_to_top(cube: Cube, from_side: Side) -> None:
    """Rotate cube so that given side becomes top (BOTTOM never appears as
    from_side's target position here since it's excluded from the movable
    faces in generation, but the method exists for BOTTOM as a source too)."""
    ROTATION_METHOD[from_side](cube)


def generate_cube(rng: random.Random) -> Cube:
    colors = list(Color)
    rng.shuffle(colors)
    return Cube({side: color for side, color in zip(Side, colors)})


ROTATION_TEMPLATES = [
    "The cube is rotated so that the side which was before at the {side} is now at the top.",
    "Then the cube is rotated to bring the {side} side to the top.",
    "After that the cube is turned to make the {side} face the top.",
    "Now the cube is rotated to place its {side} side at the top.",
    "Next, the {side} side is rotated to become the top face.",
]
