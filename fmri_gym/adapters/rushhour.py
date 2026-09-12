"""Rush Hour adapter -- the DBP "puzzle" pick, via ``rushhour-gym``.

Slide cars out of a 6x6 grid to free the red car. The rules run in a Go engine
(https://github.com/chrplr/Rush-Hour); ``rushhour-gym`` (PyPI) wraps it and
fetches the engine binary of its matching release on first use.

The env this adapter drives is ``RushHourHuman-v0``: the experiment program's
own interface -- the eight meta-actions of its button scheme (choose a car,
slide it), its picture (``rgb_array``: board, white outline and legal-slide
arrows on the chosen car, status line), its trial flow (ready screen, blank
interval, "PUZZLE SOLVED!" hold) and the columns of its results file in
``info``. All of that lives in the package; this adapter is the keymap plus
the ``info`` fields to log.

Phase fields are passed to the env: ``puzzle_order`` (``"library"``: easiest
first), ``puzzle_indices``, ``min_moves_range``, ``movable_only`` (default
true), ``paced``, ``iti``, ``solved_feedback``; ``n_episodes`` names the
"Puzzle i of N" total. ``binary`` names an engine binary of your own (else
``$RUSHHOUR_ENV_BIN``, else the package's own lookup). A ``game`` other than
``RushHourHuman-v0`` (say ``RushHour-Easy-v0``, ``Discrete(32)``) has no
keymap a participant can use and is refused.

Use ``turn_based: true``; the env's time-driven screens need the loop's idle
redraw, which this adapter opts into with ``idle_redraw``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState
from .keyspec import SingleKeySpec

_ENV_ID = "RushHourHuman-v0"
_ENV_KWARGS = ("puzzle", "puzzle_indices", "min_moves_range", "puzzle_order",
               "movable_only", "paced", "iti", "solved_feedback", "binary")
# info fields logged as per-frame variables: the program's results-file
# columns, plus what ties a row to the engine (env_action) and to the UI.
_LOGGED = ("event", "phase", "trial", "puzzle", "puzzle_index", "min_moves",
           "car", "orientation", "from_row", "from_col", "to_row", "to_col",
           "n_slides", "solved", "t_ms", "trial_ms",
           "env_action", "selected", "slot", "moved", "illegal")


class RushHourAdapter(EnvAdapter):
    name: str = "rushhour"
    idle_redraw: bool = True

    def _make(self, spec: dict) -> Any:
        import gymnasium as gym
        import rushhour_gym  # noqa: F401  (registers RushHour*-v0)
        game = spec.get("game", _ENV_ID)
        if game != _ENV_ID:
            raise ValueError(
                f"rushhour backend: game must be {_ENV_ID!r} (a person's interface), "
                f"not {game!r}; the agent ids have no keymap a participant can use")
        kwargs = {k: spec[k] for k in _ENV_KWARGS if k in spec}
        if spec.get("mode") == "episode" and spec.get("n_episodes"):
            kwargs["n_trials"] = int(spec["n_episodes"])
        return gym.make(_ENV_ID, render_mode="rgb_array", **kwargs)

    def _keyspec(self) -> SingleKeySpec:
        from rushhour_gym.human import DEFAULT_KEYS, NOOP
        combos = {frozenset([k]): v for k, v in DEFAULT_KEYS.items()}
        return SingleKeySpec(combos=combos, noop=NOOP)

    def render(self) -> np.ndarray:
        return self.env.render()

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        info = info if isinstance(info, dict) else {}
        return FrameState(blob=None, variables={k: info.get(k) for k in _LOGGED if k in info})
