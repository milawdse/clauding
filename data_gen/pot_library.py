"""The runtime API exposed to PoT (Program of Thoughts) programs.

Instead of writing a natural-language reasoning trace, the model synthesizes
a short Python program against this `Cube` class; an interpreter executes
the program and the *printed* value is the model's answer. The model never
has to track cube state itself -- it only has to translate the story into
the right sequence of API calls. This is the same rotation math as
data_gen/cube.py (ported from reasoning-gym), just re-expressed with plain
string attributes so it reads naturally as something a program would use.

POT_LIBRARY_SOURCE is the exact text shown to the model (in the system
prompt) and prepended to its generated program before execution, so the
model's program and the interpreter's namespace always agree.
"""

from __future__ import annotations

POT_LIBRARY_SOURCE = '''\
class Cube:
    def __init__(self, top, right, front, left, back, bottom):
        self.top, self.right, self.front = top, right, front
        self.left, self.back, self.bottom = left, back, bottom

    def rotate_to_top(self, side):
        """Rotate the cube so that `side` (one of "front", "right", "back",
        "left", "bottom") becomes the new top face."""
        if side == "front":
            self.top, self.front, self.bottom, self.back = self.front, self.bottom, self.back, self.top
        elif side == "right":
            self.top, self.right, self.bottom, self.left = self.right, self.bottom, self.left, self.top
        elif side == "back":
            self.top, self.back, self.bottom, self.front = self.back, self.bottom, self.front, self.top
        elif side == "left":
            self.top, self.left, self.bottom, self.right = self.left, self.bottom, self.right, self.top
        elif side == "bottom":
            self.top, self.bottom, self.front, self.back = self.bottom, self.top, self.back, self.front
        else:
            raise ValueError(f"unknown side: {side}")
'''


def _make_namespace() -> dict:
    ns: dict = {}
    exec(POT_LIBRARY_SOURCE, ns)
    return ns


if __name__ == "__main__":
    # Self-check against the known reasoning-gym GALLERY.md example.
    ns = _make_namespace()
    Cube = ns["Cube"]
    cube = Cube(top="pink", right="gray", front="orange", left="purple", back="indigo", bottom="cyan")
    cube.rotate_to_top("bottom")
    assert cube.back == "orange", cube.back
    print("pot_library self-check OK")
