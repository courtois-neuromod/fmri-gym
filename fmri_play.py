"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

Usage:
    python fmri_play.py --subject sub-01                        # built-in demo
    python fmri_play.py --subject sub-01 --curriculum my.json
    python fmri_play.py --subject sub-01 --dummy-trigger        # testing

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the curriculum schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from fmri_gym import Audio, Display, Session, Triggers


def build_demo_curriculum() -> list[dict]:
    """Mixed-backend demo: an Atari game, a retro game, and a survival game.

    All three are forgiving, free-roaming games (no instant game-over), so a
    first-time human can actually play them.
    """
    return [
        {"type": "message", "text": "Pong (Atari)", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        # Pong's paddle actions are RIGHT=2 / LEFT=3; remap them onto the
        # up/down arrows, which read more naturally for a vertical paddle.
        {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
         "duration": 10.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},

        {"type": "message", "text": "Airstriker (Genesis)", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "retro", "game": "Airstriker-Genesis-v0", "mode": "duration", "duration": 10.0, "fps": 60},

        # Crafter: an open-world survival game. You wander freely (arrows move,
        # SPACE interacts) with no instant death -- friendlier than CartPole,
        # which topples in ~2 s. Needs `pip install crafter`.
        {"type": "message", "text": "Crafter", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "crafter", "game": "crafter", "mode": "episode",
         "n_episodes": 1, "max_duration": 20.0, "fps": 15, "seed": 0},

        {"type": "fixation", "duration": 4.0},
        {"type": "survey", "questions": [
            "I was fully absorbed in the games.",
            "The games were too difficult.",
        ]},
    ]


def load_config(path: str) -> dict:
    """Load a config file: a bare curriculum list, or a dict with sections.

    The dict form carries ``"curriculum"`` plus optional ``"triggers"`` (sync +
    trigger codes; see :mod:`fmri_gym.triggers`). ``_``-prefixed keys are notes.

    :param path: JSON file path.
    :return: a dict with at least ``"curriculum"``.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {"curriculum": data}


def main() -> None:
    p = argparse.ArgumentParser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test")
    p.add_argument("--curriculum")
    p.add_argument("--outdir")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true",
                   help="do not lock flips to the monitor refresh (default: try to)")
    p.add_argument("--dummy-trigger", action="store_true")
    p.add_argument("--save-pixels", action="store_true",
                   help="ALE only: also store lossless pixels (large; warns).")
    p.add_argument("--vgdl-repo", default=os.environ.get("VGDL_REPO"),
                   help="path to the language_and_experience checkout (vgdl backend)")
    args = p.parse_args()

    config = (load_config(args.curriculum) if args.curriculum
              else {"curriculum": build_demo_curriculum()})
    curriculum = config["curriculum"]
    w, h = (int(x) for x in args.size.lower().split("x"))
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")

    # CLI-global backend options fold into the relevant game phases, so each
    # per-block EnvAdapter reads everything it needs from its own spec.
    for phase in curriculum:
        if phase.get("type") != "game":
            continue
        backend = phase.get("backend", "gym")
        if backend == "ale" and args.save_pixels:
            phase.setdefault("save_pixels", True)
        if backend == "vgdl" and args.vgdl_repo:
            phase.setdefault("repo", args.vgdl_repo)

    # Triggers first: a bad section or an unopenable port stops the run here,
    # at the desk, before any window opens -- not mid-session with a participant.
    triggers = Triggers.from_config(config.get("triggers"))
    print(f"triggers: {triggers.status()}", file=sys.stderr)
    if args.dummy_trigger:
        print("triggers: --dummy-trigger: the experimenter and scanner waits are skipped; "
              "this is a test run, not a session", file=sys.stderr)
    display = Display(size=(w, h), fullscreen=args.fullscreen, vsync=not args.no_vsync)
    audio = Audio()
    session = Session(args.subject, curriculum, display, outdir,
                      audio=audio, triggers=triggers, dummy_trigger=args.dummy_trigger)
    try:
        session.run()
    finally:
        display.close()
        audio.close()
        triggers.close()


if __name__ == "__main__":
    main()
