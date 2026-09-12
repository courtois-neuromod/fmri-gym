"""SuperTuxKart adapter (pystk2-gymnasium) -- the DBP "sports/racing" pick.

Uses the `pystk2-gymnasium` package (bpiwowar/pystk2-gymnasium) for the whole
race lifecycle: it registers `supertuxkart/full-v0`, a Gymnasium env whose Dict
action exposes acceleration / steering / brake / drift / fire / nitro, and it
handles reset/step/reward. We only translate held keys into that action.

pystk2-gymnasium has no rgb_array render mode (its ``render()`` is a no-op; its
only render mode, "human", opens SuperTuxKart's OWN window, which we do NOT want
-- it sits over our display and steals keyboard focus). So we pass an offscreen
``GraphicsConfig`` (no window) and read the frame straight off the in-process
race the env keeps (``env.unwrapped._stk.race.render_data[0].image``). That needs
``use_subprocess=False`` so the race lives in this process (not a worker), and a
real GL context -- SuperTuxKart's renderer does NOT work under
SDL_VIDEODRIVER=dummy. The fMRI presentation machine has a display, so this is
fine there; headless CI without GL cannot render it.

Controls: LEFT/RIGHT steer, UP accelerate, DOWN brake, SPACE fire item,
Z drift, X nitro. Reward + termination come from the gym env (progress / place /
finishing the race).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from .keyspec import PassthroughKeySpec
from .base import EnvAdapter, FrameState


class SuperTuxKartAdapter(EnvAdapter):
    name: str = "supertuxkart"

    def _make(self, spec: dict) -> gym.Env:
        import pystk2
        import pystk2_gymnasium  # noqa: F401  (registers supertuxkart/* env ids)
        from pystk2_gymnasium import AgentSpec

        # Render OFFSCREEN into an in-process buffer we read in render(): a plain
        # GraphicsConfig (NOT render_mode="human", which opens SuperTuxKart's own
        # window -- that would sit on top of our display and steal keyboard focus
        # so pygame gets no key presses). use_subprocess=False keeps the race in
        # THIS process so render() can reach its frame. AgentSpec(use_ai=False)
        # makes our kart player-controlled (else step() ignores the action).
        gc = pystk2.GraphicsConfig.sd()
        gc.screen_width = int(spec.get("width", 600))
        gc.screen_height = int(spec.get("height", 400))
        return gym.make(
            "supertuxkart/full-v0",
            render_mode=None,
            use_subprocess=False,
            graphics_config=gc,
            agent=AgentSpec(use_ai=False),
            num_kart=int(spec.get("num_kart", 3)),
            laps=int(spec.get("laps", 3)),
            difficulty=int(spec.get("difficulty", 2)),
            track=spec.get("track"),
        )

    def _keyspec(self) -> PassthroughKeySpec:
        # The action is assembled from the held-key set in step(); the combos
        # here just declare which keys are meaningful (resolve returns the set).
        keys = ["LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z", "X"]
        combos = {frozenset([k]): k for k in keys}
        return PassthroughKeySpec(combos=combos, noop="")

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        held = set(action.split("+")) if isinstance(action, str) and action else \
            (set(action) if action else set())
        act = {
            "acceleration": np.array([1.0 if "UP" in held else 0.0], np.float32),
            "steer": np.array([(1.0 if "RIGHT" in held else 0.0)
                               - (1.0 if "LEFT" in held else 0.0)], np.float32),
            "brake": int("DOWN" in held),
            "fire": int("SPACE" in held),
            "drift": int("Z" in held),
            "nitro": int("X" in held),
            "rescue": 0,
        }
        return self.env.step(act)

    def render(self) -> np.ndarray:
        # pystk2-gymnasium exposes no pixel obs; read the in-process race's frame.
        return np.asarray(self.env.unwrapped._stk.race.render_data[0].image)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        return FrameState(blob=None,
                          variables={"distance": float((info or {}).get("distance", 0.0))})
