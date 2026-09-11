"""Rush Hour adapter (chrplr/Rush-Hour) -- the DBP "puzzle" pick.

Slide cars out of a 6x6 grid to free the red car. The env (rushhour_gym) wraps a
Go engine binary and renders only ANSI (a letter grid). The action space is
Discrete(32): ``action = slot * 2 + direction`` (0=left/up, 1=right/down).

Humans do not press Discrete indices. Matching the experiment's own
``rushinput.DefaultMap``, we expose a small meta-action keymap:

  * LEFT/RIGHT, 3/4  -- select the previous / next car (board order)
  * UP/DOWN, 1/2, ,. -- slide the selected car back (left/up) or forward (right/down)

That is Rush-Hour's four-button response-box scheme, which its arrow keys carry
too. Its gamepad d-pad selects spatially instead; the spatial meta-actions are
kept here (``_SELECT_UP`` ...) so a curriculum can bind keys to them.

Select meta-actions update a local highlight and do not call ``env.step``.
Move meta-actions become a Discrete index and are what get logged as
``env_action`` for seed+action replay.

Two things the engine knows and a bare letter grid does not are used here. The
per-step ``action_mask`` says which of the selected car's two slides are legal,
so they are drawn as arrows on the car -- the same affordance Rush-Hour's own
SDL build gives (``rushui.DrawBoard``), without which a refused move and a
dropped keypress look identical. The phase flag ``movable_only`` (default on,
like Rush-Hour's ``-movable-only``) restricts selection to cars that can move at
all. Turning it off is a different task -- noticing which cars are stuck becomes
part of the search -- so keep it the same across sessions being compared.

Car geometry comes from the engine too, via ``obs_mode="cars"`` (slot-indexed
rows of row/col/length/horizontal), rather than from re-reading the letters.

The participant's experience of Rush-Hour's own SDL program (``main.go``,
``rushui``) is ported here so a curriculum can present the whole task without
leaving fmri-gym: the same look (1024x768, light board, flat cars in the same
palette, red exit marker, white outline on the selected car, status line), the
puzzles in library order (easiest first) when the phase says ``"puzzle_order":
"library"`` or lists ``"puzzle_indices"``, and -- ``"paced"``, on by default
with a puzzle sequence -- Rush-Hour's trial flow: a self-paced "Puzzle i of N,
press any key" screen, a blank inter-trial interval (``"iti"``, 0.8 s), the
board, and a "PUZZLE SOLVED!" hold (``"solved_feedback"``, 1.2 s). The logged
variables carry the columns of Rush-Hour's results file (``event``, ``car``,
``from_*``/``to_*``, ``n_slides``, ``solved``, ``t_ms``, ``trial_ms``).
Pacing needs ``turn_based: true`` and uses fmri-gym's idle redraw
(``idle_redraw``) for the time-driven screens.

Not ported, and not portable: Rush-Hour stamps every keypress and display flip
on SDL's monotonic clock and runs exclusive fullscreen; fmri-gym timestamps a
key when its loop processes it, so a keypress is quantised to ``1/fps`` (set
``fps`` high, e.g. 60, in a turn-based block: it only steps on keydown anyway)
and flips are not recorded. Nor the solved sound, nor mouse play.

Needs the Go binary ``rushhour-env`` built from the checkout:
    go build -o rushhour-env ./cmd/rushhour-env
Point at it via the phase "binary" field or the RUSHHOUR_ENV_BIN env var; this
adapter also auto-finds the vendored copy under vendor/rush-hour-src/.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# rushui's look (internal/rushui/render.go), in pixel coordinates (y down).
# The Go program draws in a 1024x768 logical space; fmri-gym scales the frame
# to the window with the aspect kept, so the same numbers give the same picture.
_W, _H = 1024, 768
_TILE = 90
_BOARD_X0 = _W // 2 - 3 * _TILE          # 242: board is centred horizontally
_BOARD_Y0 = _H // 2 - (3 * _TILE + 40)   # 74: shifted up to leave room for the status line
_CAR_INSET = 4                           # coloured body inset inside its black plate
_EXIT_W = 10                             # exit marker thickness at the right wall
_STATUS_Y = _BOARD_Y0 + 6 * _TILE + 40   # status line, 40 px below the board
_TARGET_ROW = 2
_BG = (240, 240, 240)
_GRID = (180, 180, 180)
_EXIT = (220, 50, 50)
_TEXT = (30, 30, 30)
_OUTLINE = (0, 0, 0)
_SELECT = (255, 255, 255)
_ARROW = (255, 255, 255)
# rushui.carColors: index 0 is the red target; the others cycle by the car's
# alphabetical index (rush.Car.ID), i.e. ord(label) - ord("A").
_CAR_COLORS: list[tuple[int, int, int]] = [
    (220, 50, 50), (50, 120, 220), (50, 180, 80), (220, 160, 40),
    (140, 60, 200), (200, 200, 50), (50, 180, 200),
]
# Arrow geometry, as fractions of a tile (rushui.arrowPoints).
_ARROW_TIP_INSET, _ARROW_LEN, _ARROW_HALF = 0.20, 0.34, 0.16
_FONT_SIZE = 28

# Trial flow, as in main.go.
_PHASE_READY, _PHASE_ITI, _PHASE_PLAY, _PHASE_SOLVED = "ready", "iti", "play", "solved"
# A press that arrived while the previous screen was up must not skip the
# ready screen (showScreen drains those); events reach us only after the
# solved hold, so the equivalent is to ignore presses in its first moments.
_READY_GRACE = 0.25
_VENDOR_BIN: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "vendor", "rush-hour-src", "rushhour-env")

# Meta-actions for the human keymap (NOT Discrete env indices). Same vocabulary
# as rushinput.DefaultMap: select a car, then slide it along its axis.
_SELECT_UP, _SELECT_DOWN, _SELECT_LEFT, _SELECT_RIGHT = 0, 1, 2, 3
_SELECT_PREV, _SELECT_NEXT = 4, 5
_MOVE_BACK, _MOVE_FORWARD = 6, 7
_NOOP = -1

# Mirrors https://github.com/chrplr/Rush-Hour/blob/main/internal/rushinput/keymap.go
_DEFAULT_KEYMAP: dict[str, int] = {
    "LEFT": _SELECT_PREV, "RIGHT": _SELECT_NEXT,
    "UP": _MOVE_BACK, "DOWN": _MOVE_FORWARD,
    "3": _SELECT_PREV, "4": _SELECT_NEXT,
    "1": _MOVE_BACK, "2": _MOVE_FORWARD,
    "COMMA": _MOVE_BACK, "PERIOD": _MOVE_FORWARD,
}

# Neighbour scoring constant from rush.Board.Neighbour (select.go).
_SIDEWAYS_PENALTY = 2 * 6


class RushHourAdapter(EnvAdapter):
    name: str = "rushhour"
    # Paced trials show time-driven screens; see session.py's turn-based loop.
    idle_redraw: bool = True

    def _make(self, spec: dict) -> Any:
        import gymnasium as gym
        import rushhour_gym  # noqa: F401  (registers RushHour*-v0)
        binary = spec.get("binary") or os.environ.get("RUSHHOUR_ENV_BIN")
        if not binary and os.path.exists(_VENDOR_BIN):
            binary = _VENDOR_BIN
        if binary:
            os.environ["RUSHHOUR_ENV_BIN"] = binary
        self._last_ansi = ""
        self._labels = ""
        self._cars: list[dict[str, Any]] = []
        self._by_slot: dict[int, dict[str, Any]] = {}
        self._mask: np.ndarray | None = None
        self._selected = 0
        self._last_obs: Any = None
        self._last_info: dict = {}
        # Skip cars that cannot move, as Rush-Hour does by default; see the
        # module docstring for why a study should not flip this mid-way.
        self._movable_only = bool(spec.get("movable_only", True))

        # Which puzzle each episode gets: an explicit list, the library's own
        # order (easiest first, what Rush-Hour's -n takes a prefix of), or --
        # neither given -- the env's seeded draw from its pool.
        self._puzzle_indices = spec.get("puzzle_indices")
        self._library_order = spec.get("puzzle_order") == "library"
        sequenced = self._library_order or bool(self._puzzle_indices)
        # Rush-Hour's trial flow (ready screen, blank ITI, solved hold).
        self._paced = bool(spec.get("paced", sequenced))
        self._iti = float(spec.get("iti", 0.8))
        self._solved_hold = float(spec.get("solved_feedback", 1.2))
        self._n_trials = int(spec.get("n_episodes", 0)) if spec.get("mode") == "episode" else 0
        self._episode = -1
        self._phase = _PHASE_PLAY
        self._phase_t0 = time.perf_counter()
        self._trial_onset: float | None = None
        self._event = ""
        self._trial_ms = -1.0
        self._font: Any = None

        # "cars" gives slot-indexed geometry straight from the engine; the
        # observation itself is never logged, so this costs nothing. The env
        # ids carry a TimeLimit sized for agents (200-500 steps) that would
        # truncate a participant on a hard puzzle, so it is lifted here.
        return gym.make(spec.get("game", "RushHour-Easy-v0"), render_mode="ansi",
                        obs_mode="cars",
                        max_episode_steps=int(spec.get("max_episode_steps", 10 ** 6)))

    def _keyspec(self) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=_NOOP)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        # The loop ends an episode on the solving move and comes straight
        # here, with the "PUZZLE SOLVED!" frame on screen: hold it, as
        # main.go does, before the next puzzle replaces it.
        self._hold_solved()
        self._episode += 1

        options = None
        if self._puzzle_indices:
            options = {"puzzle_index": int(
                self._puzzle_indices[self._episode % len(self._puzzle_indices)])}
        elif self._library_order:
            options = {"puzzle_index": self._episode}
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_ansi = self.env.render() or ""
        self._ingest(obs, info)
        self._selected = 0  # red car; same as the experiment's trial start
        self._last_obs, self._last_info = obs, info
        self._event = "trial_start"
        self._trial_ms = -1.0

        # Rush-Hour puts a ready screen before every puzzle but the first,
        # whose press was the instruction screen's, and a blank ITI before all.
        if not self._paced:
            self._begin_trial()
        elif self._episode > 0:
            self._set_phase(_PHASE_READY)
        else:
            self._set_phase(_PHASE_ITI)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        meta = int(action)
        self._advance()
        if self._phase == _PHASE_READY:
            if meta >= 0 and time.perf_counter() - self._phase_t0 >= _READY_GRACE:
                self._set_phase(_PHASE_ITI)
                self._event = "start"
            else:
                self._event = "ignored"
            return self._ui_only()
        if self._phase != _PHASE_PLAY:
            self._event = "ignored"
            return self._ui_only()

        if meta < 0:
            self._event = "noop"
            return self._ui_only()
        if meta <= _SELECT_NEXT:
            self._do_select(meta)
            self._event = "select"
            return self._ui_only()

        dir_bit = 0 if meta == _MOVE_BACK else 1
        discrete = int(self._selected) * 2 + dir_bit
        obs, reward, terminated, truncated, info = self.env.step(discrete)
        self._last_ansi = self.env.render() or ""
        self._ingest(obs, info)
        # Keep the highlight on the car that was just acted on when the engine
        # reports a slot (illegal clicks still name a slot).
        if isinstance(info, dict) and "slot" in info:
            try:
                slot = int(info["slot"])
                if slot in self._by_slot:
                    self._selected = slot
            except Exception:
                pass
        info = dict(info)
        info["env_action"] = discrete
        self._last_obs, self._last_info = obs, info
        self._event = "move" if info.get("moved") else "blocked"
        if terminated:
            # Timed at the move that solved it, as Rush-Hour's trial_ms is.
            self._event = "trial_end"
            self._trial_ms = self._t_ms()
            self._set_phase(_PHASE_SOLVED)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray:
        self._advance()
        if self._phase == _PHASE_READY:
            n = f" of {self._n_trials}" if self._n_trials else ""
            return _draw_screen([f"Puzzle {self._episode + 1}{n}", "",
                                 "Press any key or button to start."], self._get_font())
        if self._phase == _PHASE_ITI:
            return _draw_screen([], self._get_font())
        if self._phase == _PHASE_SOLVED:
            # No selection and no arrows on the solved board, as in main.go.
            return _draw_board(self._cars, None, [], "PUZZLE SOLVED!", self._get_font())

        car = self._selected_car()
        arrows: list[tuple[dict[str, Any], int]] = []
        if car is not None:
            back, forward = self._can_move(car)
            if back:
                arrows.append((car, -1))
            if forward:
                arrows.append((car, 1))
        n = f"/{self._n_trials}" if self._n_trials else ""
        status = f"Puzzle {self._episode + 1}{n} - free the RED car"
        return _draw_board(self._cars, car, arrows, status, self._get_font())

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables: dict[str, Any] = {}
        if isinstance(info, dict) and "slot" in info:
            try:
                variables["slot"] = int(info["slot"])
            except Exception:
                pass
        # Whether the press actually moved a car. Without these, a blocked
        # attempt and a successful move are indistinguishable in the saved data:
        # the reward is -1 either way, and the board looks the same.
        if isinstance(info, dict):
            variables["moved"] = bool(info.get("moved", False))
            variables["illegal"] = bool(info.get("illegal", False))
            variables["env_action"] = int(info.get("env_action", _NOOP))
        variables["selected"] = int(self._selected)

        # The columns of Rush-Hour's results file (internal/rushlog/row.go), so
        # the two programs' data read the same way. A row that records no
        # displacement -- a select, a blocked move -- has from == to.
        variables["event"] = self._event
        variables["phase"] = self._phase
        variables["trial"] = self._episode + 1
        variables["puzzle"] = str(info.get("puzzle", "")) if isinstance(info, dict) else ""
        variables["puzzle_index"] = int(info.get("puzzle_index", -1)) if isinstance(info, dict) else -1
        variables["min_moves"] = int(info.get("min_moves", -1)) if isinstance(info, dict) else -1
        car: dict[str, Any] | None = None
        frm = to = (-1, -1)
        if self._event in ("move", "blocked", "trial_end") and isinstance(info, dict):
            car = self._by_slot.get(int(info.get("slot", -1)))
            frm = tuple(int(x) for x in info.get("from", frm))
            to = tuple(int(x) for x in info.get("to", to))
        elif self._event == "select":
            car = self._selected_car()
            if car is not None:
                frm = to = (int(car["row"]), int(car["col"]))
        variables["car"] = str(car["label"]) if car else ""
        variables["orientation"] = ("H" if car["horizontal"] else "V") if car else ""
        variables["from_row"], variables["from_col"] = frm
        variables["to_row"], variables["to_col"] = to
        variables["n_slides"] = int(info.get("n_slides", 0)) if isinstance(info, dict) else 0
        variables["solved"] = self._event == "trial_end"
        variables["t_ms"] = self._t_ms()
        variables["trial_ms"] = self._trial_ms
        return FrameState(blob=None, variables=variables)

    def close(self) -> None:
        # The last puzzle's solved frame is on screen when the block ends.
        self._hold_solved()
        super().close()

    # ── Trial flow ───────────────────────────────────────────────────────────

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_t0 = time.perf_counter()

    def _begin_trial(self) -> None:
        self._set_phase(_PHASE_PLAY)
        self._trial_onset = self._phase_t0

    def _advance(self) -> None:
        """Time-driven transition: the blank ITI ends with the board's onset."""
        if self._phase == _PHASE_ITI and time.perf_counter() - self._phase_t0 >= self._iti:
            self._begin_trial()

    def _hold_solved(self) -> None:
        if self._phase == _PHASE_SOLVED:
            left = self._solved_hold - (time.perf_counter() - self._phase_t0)
            if left > 0:
                time.sleep(left)
            self._phase = _PHASE_PLAY

    def _t_ms(self) -> float:
        """Milliseconds since the board appeared (Rush-Hour's t_ms); -1 outside a trial."""
        if self._trial_onset is None or self._phase not in (_PHASE_PLAY, _PHASE_SOLVED):
            return -1.0
        ref = self._phase_t0 if self._phase == _PHASE_SOLVED else time.perf_counter()
        return (ref - self._trial_onset) * 1000.0

    def _get_font(self) -> Any:
        if self._font is None:
            import pygame
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.Font(pygame.font.get_default_font(), _FONT_SIZE)
        return self._font

    # ── Selection / board helpers ─────────────────────────────────────────────

    def _ui_only(self) -> tuple[Any, float, bool, bool, dict]:
        # A select calls no env.step, so the previous move's outcome fields would
        # otherwise persist and read, in the log, as a move that never happened.
        info = dict(self._last_info)
        info["env_action"] = -1
        info["moved"] = False
        info["illegal"] = False
        info["slot"] = -1
        return self._last_obs, 0.0, False, False, info

    def _ingest(self, obs: Any, info: dict) -> None:
        if isinstance(info, dict):
            self._labels = str(info.get("labels") or self._labels)
            mask = info.get("action_mask")
            self._mask = None if mask is None else np.asarray(mask, dtype=bool)
        cars = _cars_from_obs(obs, self._labels)
        if not cars:
            # Fallback for an obs_mode override: re-read the letter grid.
            board = (info.get("board") if isinstance(info, dict) else None) \
                or self._last_ansi
            cars = _cars_from_board(str(board or ""), self._labels)
        if cars:
            self._cars = cars
            self._by_slot = {int(c["slot"]): c for c in cars}

    def _selected_car(self) -> dict[str, Any] | None:
        """The highlighted car, by SLOT -- slots are padded, list indices are not."""
        car = self._by_slot.get(int(self._selected))
        return car or (self._cars[0] if self._cars else None)

    def _can_move(self, car: dict[str, Any]) -> tuple[bool, bool]:
        """(back, forward) legality for one car, straight from the engine mask."""
        if self._mask is None:
            return True, True
        i = 2 * int(car["slot"])
        if i + 1 >= len(self._mask):
            return True, True
        return bool(self._mask[i]), bool(self._mask[i + 1])

    def _candidates(self) -> list[dict[str, Any]]:
        """Cars the selection may land on. Empty when nothing qualifies, in
        which case the selection stays where it is (rush.Board.CycleAmong /
        NeighbourAmong return ``from``)."""
        if not self._movable_only:
            return self._cars
        return [c for c in self._cars if any(self._can_move(c))]

    def _do_select(self, meta: int) -> None:
        cars = self._candidates()
        if not cars:
            return
        cur = self._selected_car() or cars[0]
        if meta == _SELECT_PREV:
            nxt = _cycle(self._cars, cars, cur, -1)
        elif meta == _SELECT_NEXT:
            nxt = _cycle(self._cars, cars, cur, 1)
        else:
            d_row, d_col = {
                _SELECT_UP: (-1, 0), _SELECT_DOWN: (1, 0),
                _SELECT_LEFT: (0, -1), _SELECT_RIGHT: (0, 1),
            }[meta]
            nxt = _neighbour(cars, cur, d_row, d_col)
        self._selected = int(nxt["slot"])


def _cells(row: int, col: int, length: int, horizontal: bool) -> list[tuple[int, int]]:
    if horizontal:
        return [(row, col + i) for i in range(length)]
    return [(row + i, col) for i in range(length)]


def _cars_from_obs(obs: Any, labels: str) -> list[dict[str, Any]]:
    """Slot-indexed geometry from ``obs_mode="cars"``: row, col, length, horizontal.

    Rows are padded to max_vehicles, and a real vehicle is at least two cells
    long, so ``length == 0`` marks an empty slot.
    """
    arr = np.asarray(obs)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return []
    cars: list[dict[str, Any]] = []
    for slot, (row, col, length, horizontal) in enumerate(arr.tolist()):
        if int(length) == 0:
            continue
        row, col, length = int(row), int(col), int(length)
        horizontal = bool(horizontal)
        cars.append({
            "slot": slot,
            "label": labels[slot] if slot < len(labels) else "?",
            "row": row, "col": col,
            "length": length, "horizontal": horizontal,
            "cells": _cells(row, col, length, horizontal),
        })
    return cars


def _cars_from_board(board: str, labels: str) -> list[dict[str, Any]]:
    """Parse the ANSI/board notation into per-slot car geometries."""
    rows = []
    for line in (board or "").split("\n"):
        if not line:
            continue
        # Drop the " <" exit marker the ansi renderer appends.
        cell = line[:6] if len(line) >= 6 else line.rstrip()
        if cell:
            rows.append(cell)
    positions: dict[str, list[tuple[int, int]]] = {}
    for r, line in enumerate(rows[:6]):
        for c, ch in enumerate(line[:6]):
            if ch.isalpha():
                positions.setdefault(ch.upper(), []).append((r, c))

    cars: list[dict[str, Any]] = []
    # Prefer the engine's slot order; fall back to letters found on the board.
    order = list(labels) if labels else sorted(positions.keys())
    for slot, lab in enumerate(order):
        cells = positions.get(lab, [])
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        horizontal = len(set(rs)) == 1
        cars.append({
            "slot": slot, "label": lab,
            "row": min(rs), "col": min(cs),
            "length": len(cells), "horizontal": horizontal,
            "cells": cells,
        })
    return cars


def _span(car: dict[str, Any]) -> tuple[int, int, int, int]:
    row0, col0 = car["row"], car["col"]
    if car["horizontal"]:
        return row0, row0, col0, col0 + car["length"] - 1
    return row0, row0 + car["length"] - 1, col0, col0


def _center2(car: dict[str, Any]) -> tuple[int, int]:
    r0, r1, c0, c1 = _span(car)
    return r0 + r1, c0 + c1


def _gap(a0: int, a1: int, b0: int, b1: int) -> int:
    d = max(a0, b0) - min(a1, b1)
    return d if d > 0 else 0


def _neighbour(
    cars: list[dict[str, Any]], from_car: dict[str, Any], d_row: int, d_col: int
) -> dict[str, Any]:
    """Port of rush.Board.Neighbour — spatial select with wrap-around."""
    if (d_row != 0) == (d_col != 0):
        return from_car
    from_r2, from_c2 = _center2(from_car)
    f_r0, f_r1, f_c0, f_c1 = _span(from_car)
    best = wrapped = None
    best_score = wrap_score = 0
    for cand in cars:
        if cand is from_car or cand["slot"] == from_car["slot"]:
            continue
        cand_r2, cand_c2 = _center2(cand)
        ahead = (cand_r2 - from_r2) * d_row + (cand_c2 - from_c2) * d_col
        c_r0, c_r1, c_c0, c_c1 = _span(cand)
        sideways = (
            _gap(f_c0, f_c1, c_c0, c_c1) if d_row != 0
            else _gap(f_r0, f_r1, c_r0, c_r1)
        )
        score = ahead + _SIDEWAYS_PENALTY * sideways
        if ahead > 0:
            if best is None or score < best_score:
                best, best_score = cand, score
        else:
            if wrapped is None or score < wrap_score:
                wrapped, wrap_score = cand, score
    return best or wrapped or from_car


def _cycle(
    all_cars: list[dict[str, Any]], ok: list[dict[str, Any]],
    from_car: dict[str, Any], delta: int,
) -> dict[str, Any]:
    """Port of rush.Board.CycleAmong: from ``from_car``'s place in board
    order, step ``delta`` (±1) past cars not in ``ok`` until one is, wrapping.
    Walking on from the current position -- rather than re-entering the ring
    at an end -- is what makes a car stuck by its own move hand over to its
    board-order successor, as it does in Rush-Hour."""
    n = len(all_cars)
    if n == 0 or not ok:
        return from_car
    accepted = {c["slot"] for c in ok}
    idx = next((i for i, c in enumerate(all_cars) if c["slot"] == from_car["slot"]), 0)
    step = 1 if delta > 0 else -1
    for k in range(1, n + 1):
        cand = all_cars[(idx + k * step) % n]
        if cand["slot"] in accepted:
            return cand
    return from_car


def _fill_rect(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
               color: tuple[int, int, int]) -> None:
    xa, ya = max(int(round(x0)), 0), max(int(round(y0)), 0)
    xb, yb = min(int(round(x1)), img.shape[1]), min(int(round(y1)), img.shape[0])
    if xb > xa and yb > ya:
        img[ya:yb, xa:xb] = color


def _fill_triangle(img: np.ndarray, pts: list[tuple[float, float]],
                   color: tuple[int, int, int]) -> None:
    """Rasterise a filled triangle (pixel centres inside, by barycentric sign)."""
    (x0, y0), (x1, y1), (x2, y2) = pts
    xa, xb = int(np.floor(min(x0, x1, x2))), int(np.ceil(max(x0, x1, x2))) + 1
    ya, yb = int(np.floor(min(y0, y1, y2))), int(np.ceil(max(y0, y1, y2))) + 1
    xa, ya = max(xa, 0), max(ya, 0)
    xb, yb = min(xb, img.shape[1]), min(yb, img.shape[0])
    if xb <= xa or yb <= ya:
        return
    ys, xs = np.mgrid[ya:yb, xa:xb]
    px, py = xs + 0.5, ys + 0.5
    e0 = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
    e1 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    e2 = (x0 - x2) * (py - y2) - (y0 - y2) * (px - x2)
    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
    img[ya:yb, xa:xb][inside] = color


def _car_rect(car: dict[str, Any]) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of the car's cells, in pixels (rushui.CarRect)."""
    x0 = _BOARD_X0 + car["col"] * _TILE
    y0 = _BOARD_Y0 + car["row"] * _TILE
    w = _TILE * (car["length"] if car["horizontal"] else 1)
    h = _TILE * (1 if car["horizontal"] else car["length"])
    return x0, y0, x0 + w, y0 + h


def _car_color(car: dict[str, Any]) -> tuple[int, int, int]:
    """rushui.carColor: red for the target, else by alphabetical index."""
    label = str(car.get("label", "?"))
    if label == "A" or int(car.get("slot", -1)) == 0:
        return _CAR_COLORS[0]
    idx = ord(label) - ord("A") if label.isalpha() else int(car.get("slot", 1))
    return _CAR_COLORS[idx % (len(_CAR_COLORS) - 1) + 1]


def _arrow_points(car: dict[str, Any], direction: int) -> list[tuple[float, float]]:
    """rushui.arrowPoints, tip first: a triangle inside the leading end of the
    car, pointing the way a step in ``direction`` (-1 back, +1 forward) goes."""
    x0, y0, x1, y1 = _car_rect(car)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tip_inset, length, half = _TILE * _ARROW_TIP_INSET, _TILE * _ARROW_LEN, _TILE * _ARROW_HALF
    if car["horizontal"]:
        tip_x = cx + direction * ((x1 - x0) / 2 - tip_inset)
        back_x = tip_x - direction * length
        return [(tip_x, cy), (back_x, cy + half), (back_x, cy - half)]
    tip_y = cy + direction * ((y1 - y0) / 2 - tip_inset)
    back_y = tip_y - direction * length
    return [(cx, tip_y), (cx + half, back_y), (cx - half, back_y)]


def _blit_text(img: np.ndarray, text: str, cx: float, cy: float, font: Any,
               color: tuple[int, int, int] = _TEXT) -> None:
    """Draw one line of text centred on (cx, cy)."""
    if not text:
        return
    import pygame
    surf = font.render(text, True, color, _BG)
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    h, w = arr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if xb > xa and yb > ya:
        img[ya:yb, xa:xb] = arr[ya - y0:yb - y0, xa - x0:xb - x0]


def _draw_screen(lines: list[str], font: Any) -> np.ndarray:
    """A text screen (or, with no lines, the blank ITI), centred like
    goxpyriment's FittedTextBox."""
    img = np.empty((_H, _W, 3), dtype=np.uint8)
    img[:] = _BG
    if lines:
        pitch = int(_FONT_SIZE * 1.4)
        top = _H / 2 - pitch * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            _blit_text(img, line, _W / 2, top + i * pitch, font)
    return img


def _draw_board(cars: list[dict[str, Any]], selected: dict[str, Any] | None,
                arrows: list[tuple[dict[str, Any], int]], status: str,
                font: Any) -> np.ndarray:
    """Port of rushui.DrawBoard: grid, exit marker, cars (a black plate with
    the coloured body inset; white plate for the selection), the legal-slide
    arrows on top, and the status line."""
    img = np.empty((_H, _W, 3), dtype=np.uint8)
    img[:] = _BG
    x_end, y_end = _BOARD_X0 + 6 * _TILE, _BOARD_Y0 + 6 * _TILE
    for i in range(7):
        y = _BOARD_Y0 + i * _TILE
        x = _BOARD_X0 + i * _TILE
        img[y, _BOARD_X0:x_end + 1] = _GRID
        img[_BOARD_Y0:y_end + 1, x] = _GRID
    # Exit marker on the right wall of the target row.
    ey = _BOARD_Y0 + _TARGET_ROW * _TILE
    _fill_rect(img, x_end - _EXIT_W, ey, x_end, ey + _TILE, _EXIT)
    for car in cars:
        x0, y0, x1, y1 = _car_rect(car)
        plate = _SELECT if (selected is not None and car["slot"] == selected["slot"]) else _OUTLINE
        k = _CAR_INSET / 2
        _fill_rect(img, x0 + k, y0 + k, x1 - k, y1 - k, plate)
        k = 3 * _CAR_INSET / 2
        _fill_rect(img, x0 + k, y0 + k, x1 - k, y1 - k, _car_color(car))
    # Drawn last, so they sit on top of the selected car.
    for car, direction in arrows:
        _fill_triangle(img, _arrow_points(car, direction), _ARROW)
    _blit_text(img, status, _W / 2, _STATUS_Y, font)
    return img
