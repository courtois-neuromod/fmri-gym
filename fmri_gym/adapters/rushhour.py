"""Rush Hour adapter (chrplr/Rush-Hour) -- the DBP "puzzle" pick.

Slide cars out of a 6x6 grid to free the red car. The env (rushhour_gym) wraps a
Go engine binary and renders only ANSI (a letter grid). The action space is
Discrete(32): ``action = slot * 2 + direction`` (0=left/up, 1=right/down).

Humans do not press Discrete indices. Matching the experiment's own
``rushinput.DefaultMap``, we expose a small meta-action keymap:

  * arrows / 3,4  -- select a car (spatial neighbour, or cycle prev/next)
  * 1,2 / , .     -- slide the selected car back (left/up) or forward (right/down)

Select meta-actions update a local highlight and do not call ``env.step``.
Move meta-actions become a Discrete index and are what get logged as
``env_action`` for seed+action replay.

Two things the engine knows and a bare letter grid does not are used here. The
per-step ``action_mask`` says which of the selected car's two slides are legal,
so they are drawn as arrows on the car -- the same affordance Rush-Hour's own
SDL build gives (``rushui.DrawBoard``), without which a refused move and a
dropped keypress look identical. The phase flag ``movable_only`` additionally
restricts selection to cars that can move at all; it is off by default because
it is NOT Rush-Hour behaviour -- there, selection walks every car -- and it
removes part of the search.

Car geometry comes from the engine too, via ``obs_mode="cars"`` (slot-indexed
rows of row/col/length/horizontal), rather than from re-reading the letters.

Needs the Go binary ``rushhour-env`` built from the checkout:
    go build -o rushhour-env ./cmd/rushhour-env
Point at it via the phase "binary" field or the RUSHHOUR_ENV_BIN env var; this
adapter also auto-finds the vendored copy under vendor/rush-hour-src/.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# Distinct colors for car letters; 'A' (red player car) and exit are special.
_PALETTE: list[tuple[int, int, int]] = [
    (220, 60, 60), (70, 130, 220), (80, 190, 90), (230, 190, 60),
    (170, 90, 200), (230, 140, 60), (90, 200, 200), (230, 120, 170),
    (150, 110, 70), (120, 160, 90), (200, 200, 120), (110, 200, 160),
]
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
    "UP": _SELECT_UP, "DOWN": _SELECT_DOWN,
    "LEFT": _SELECT_LEFT, "RIGHT": _SELECT_RIGHT,
    "3": _SELECT_PREV, "4": _SELECT_NEXT,
    "1": _MOVE_BACK, "2": _MOVE_FORWARD,
    "COMMA": _MOVE_BACK, "PERIOD": _MOVE_FORWARD,
}

# Neighbour scoring constant from rush.Board.Neighbour (select.go).
_SIDEWAYS_PENALTY = 2 * 6


class RushHourAdapter(EnvAdapter):
    name: str = "rushhour"

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
        # Opt-in: skip cars that cannot move. Not Rush-Hour parity -- see module
        # docstring -- so a study comparing the two should leave it off.
        self._movable_only = bool(spec.get("movable_only", False))
        # "cars" gives slot-indexed geometry straight from the engine; the
        # observation itself is never logged, so this costs nothing.
        return gym.make(spec.get("game", "RushHour-Easy-v0"), render_mode="ansi",
                        obs_mode="cars")

    def _keyspec(self) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=_NOOP)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed)
        self._last_ansi = self.env.render() or ""
        self._ingest(obs, info)
        self._selected = 0  # red car; same as the experiment's trial start
        self._last_obs, self._last_info = obs, info
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        meta = int(action)
        if meta < 0:
            return self._ui_only()
        if meta <= _SELECT_NEXT:
            self._do_select(meta)
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
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray:
        car = self._selected_car()
        arrows: list[tuple[int, int, str]] = []
        if car is not None:
            back, forward = self._can_move(car)
            horizontal = bool(car["horizontal"])
            row, col, length = car["row"], car["col"], car["length"]
            if back:  # towards index 0: left for a horizontal car, up for a vertical one
                arrows.append((row, col, "left" if horizontal else "up"))
            if forward:
                tail_r = row if horizontal else row + length - 1
                tail_c = col + length - 1 if horizontal else col
                arrows.append((tail_r, tail_c, "right" if horizontal else "down"))
        return _board_to_rgb(self._last_ansi, selected=self._selected_label(),
                             arrows=arrows)

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {}
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
        return FrameState(blob=None, variables=variables)

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

    def _selected_label(self) -> str | None:
        car = self._selected_car()
        if car is not None:
            return car.get("label")
        if self._labels:
            return self._labels[0]
        return "A"

    def _can_move(self, car: dict[str, Any]) -> tuple[bool, bool]:
        """(back, forward) legality for one car, straight from the engine mask."""
        if self._mask is None:
            return True, True
        i = 2 * int(car["slot"])
        if i + 1 >= len(self._mask):
            return True, True
        return bool(self._mask[i]), bool(self._mask[i + 1])

    def _candidates(self) -> list[dict[str, Any]]:
        """Cars the selection may land on."""
        if not self._movable_only:
            return self._cars
        movable = [c for c in self._cars if any(self._can_move(c))]
        # A solved or wedged board has nothing movable; never strand the cursor.
        return movable or self._cars

    def _do_select(self, meta: int) -> None:
        cars = self._candidates()
        if not cars:
            return
        cur = self._selected_car() or cars[0]
        if meta == _SELECT_PREV:
            nxt = _cycle(cars, cur, -1)
        elif meta == _SELECT_NEXT:
            nxt = _cycle(cars, cur, 1)
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
    cars: list[dict[str, Any]], from_car: dict[str, Any], delta: int
) -> dict[str, Any]:
    """Port of rush.Board.Cycle — walk vehicles in board order, wrapping."""
    if not cars:
        return from_car
    idx = next((i for i, c in enumerate(cars) if c["slot"] == from_car["slot"]), None)
    if idx is None:
        # The current car was filtered out (movable_only): enter the ring at
        # its end, so one press still steps once rather than twice.
        return cars[0] if delta > 0 else cars[-1]
    return cars[(idx + delta) % len(cars)]


def _arrow_mask(size: int, direction: str) -> np.ndarray:
    """A filled triangle pointing `direction`, as a size x size boolean mask."""
    tri = np.zeros((size, size), dtype=bool)
    mid, last = size // 2, max(size - 1, 1)
    for j in range(size):                      # j: base (0) -> apex (size-1)
        half = ((last - j) * mid) // last
        tri[max(mid - half, 0):mid + half + 1, j] = True
    # rot90 turns counter-clockwise, and the triangle above points right.
    return np.rot90(tri, {"right": 0, "up": 1, "left": 2, "down": 3}[direction])


def _draw_arrow(img: np.ndarray, row: int, col: int, direction: str,
                cell: int, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Mark one legal slide, on the end cell of the car that would make it."""
    pad = max(cell // 5, 1)
    size = cell - 2 * pad
    y0, x0 = row * cell + pad, col * cell + pad
    if size <= 0 or y0 + size > img.shape[0] or x0 + size > img.shape[1]:
        return
    block = img[y0:y0 + size, x0:x0 + size]
    block[_arrow_mask(size, direction)] = color


def _board_to_rgb(
    ansi: str, cell: int = 64, selected: str | None = None,
    arrows: list[tuple[int, int, str]] | None = None,
) -> np.ndarray:
    """Render the ANSI letter grid to a colored pixel board."""
    rows = [r for r in (ansi or "").split("\n") if r != ""]
    if not rows:
        return np.zeros((cell * 6, cell * 6, 3), dtype=np.uint8)
    h = len(rows)
    w = max(len(r) for r in rows)
    img = np.full((h * cell, w * cell, 3), 30, dtype=np.uint8)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            y0, x0 = r * cell, c * cell
            if ch in (" ", "o"):
                color = (45, 45, 45)          # empty
            elif ch == "<":
                color = (255, 255, 255)       # exit marker
            elif ch == "A":
                color = (230, 40, 40)         # red player car
            elif ch.isalpha():
                color = _PALETTE[(ord(ch.upper()) - ord("A")) % len(_PALETTE)]
            else:
                color = (60, 60, 60)
            # draw a padded block so grid lines show
            img[y0 + 2:y0 + cell - 2, x0 + 2:x0 + cell - 2] = color
            if selected and ch.upper() == selected.upper():
                # Bright border so the selected car is obvious without arrows.
                img[y0:y0 + 3, x0:x0 + cell] = (255, 255, 255)
                img[y0 + cell - 3:y0 + cell, x0:x0 + cell] = (255, 255, 255)
                img[y0:y0 + cell, x0:x0 + 3] = (255, 255, 255)
                img[y0:y0 + cell, x0 + cell - 3:x0 + cell] = (255, 255, 255)
    # Drawn last, so they sit on top of the selected car (as rushui does).
    for row, col, direction in (arrows or []):
        _draw_arrow(img, row, col, direction, cell)
    return img
