"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

Usage:
    python fmri_play.py --subject sub-01 --curriculum my.json
    python fmri_play.py --subject sub-01 --curriculum my.json --dummy-trigger   # testing
    python fmri_play.py --subject sub-01 --curriculum my.json --no-audio        # mute all games

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

# pygame prints a banner on import, to stdout -- where --next-ses answers a session script's
# SES=$(...). The versions it shows go in the manifest instead.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from fmri_gym import Audio, Display, Session, Triggers, bids  # noqa: E402 (after the line above)
from fmri_gym.config import EXIT_QUIT, load_config, validate_config  # noqa: E402
from fmri_gym.display import list_monitors, monitor_label  # noqa: E402

import pygame  # noqa: E402

def _fold_seeds(curriculum: list[dict], label: str) -> dict:
    """Give each game phase that pins no ``"seed"`` the one derived for this run.

    The session plays episode e with ``seed + e`` as for any phase, and the
    manifest's curriculum shows every seed; what was derived is said here.

    :return: the manifest's ``seeds`` entry: the run label, and phase -> how set.
    """
    how = {}
    for i, phase in enumerate(curriculum):
        if phase["type"] != "game":
            continue
        how[i] = "pinned" if "seed" in phase else "derived"
        phase.setdefault("seed", bids.phase_seed(label, i))
        print(f"seeds: phase {i} = {phase['seed']} ({how[i]})", file=sys.stderr)
    return {"derived_from": label, "phases": how}


def _quit_like_esc(_signum: int, _frame: object) -> None:
    """Ctrl+C: the window's quit event, which the session handles where it handles ESC.

    Python's own KeyboardInterrupt lands anywhere: between two fields of a
    frame, before a game block is saved. The quit event is read between frames.
    """
    if pygame.display.get_init():
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return
    print("Ctrl+C: no window to quit right now; press it again if the run goes on",
          file=sys.stderr)


def _check_monitor(monitor: int) -> None:
    """:raises ValueError: if this machine has no such monitor; the message lists them."""
    monitors = list_monitors()
    if not 0 <= monitor < len(monitors):
        labels = "; ".join(monitor_label(i, m) for i, m in enumerate(monitors))
        raise ValueError(f"--monitor {monitor}: this machine has {len(monitors)}: {labels}")


def _run_folder(args: argparse.Namespace) -> tuple[str, str]:
    """Where this run writes, and its label (the seed key); see :mod:`fmri_gym.bids`.

    :return: ``(folder, label)``, e.g. ``data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002``.
    :raises ValueError: on a subject or number BIDS refuses, or a ``--run`` that has data.
    """
    subject = bids.subject_label(args.subject)
    task = bids.task_label(args.curriculum)
    for flag in ("ses", "run"):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            raise ValueError(f"--{flag} counts from 1, got {getattr(args, flag)}")
    ses = args.ses if args.ses is not None else bids.next_session(args.data_root, subject)
    free = bids.next_run(args.data_root, subject, ses, task)
    run = args.run if args.run is not None else free
    folder = bids.run_dir(args.data_root, subject, ses, task, run)
    if os.path.exists(folder):
        raise ValueError(f"{folder} already has data: leave --run out to take the next free "
                         f"run of this task ({free}), or move that folder away")
    return folder, bids.run_label(subject, ses, task, run)


def _fold_cli_options(curriculum: list[dict], args: argparse.Namespace) -> None:
    """CLI-global backend options fold into the relevant game phases, so each
    per-block EnvAdapter reads everything it needs from its own spec."""
    for phase in curriculum:
        if phase.get("type") != "game":
            continue
        if args.no_audio:
            phase["audio"] = False
        if phase.get("backend") == "vgdl" and args.vgdl_repo:
            phase.setdefault("repo", args.vgdl_repo)
        if phase.get("backend") == "coom" and args.coom_repo:
            phase.setdefault("repo", args.coom_repo)


def _parser() -> argparse.ArgumentParser:
    """The command line; ``--gui`` shows the launch flags as a form."""
    p = argparse.ArgumentParser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test", help="BIDS subject: sub-<letters/digits>")
    p.add_argument("--curriculum", required=True, help="config JSON (see README)")
    p.add_argument("--data-root", default="data",
                   help="where the BIDS tree goes: <root>/sub-XX/ses-NNN/beh/<run>/")
    p.add_argument("--ses", type=int,
                   help="session number (default: the subject's next free one, from 1)")
    p.add_argument("--run", type=int, help="this task's run number in the session (default: "
                   "the next free one, from 1); refused if it already has data")
    p.add_argument("--next-ses", action="store_true",
                   help="print the subject's next free session number and exit (a session "
                   "script starts with it, so all its runs share one session)")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--monitor", type=int, default=0,
                   help="which monitor to open on, by index (0: the first); a wrong one stops "
                   "the run and lists this machine's")
    p.add_argument("--no-vsync", action="store_true",
                   help="do not lock flips to the monitor refresh (default: try to)")
    p.add_argument("--dummy-trigger", action="store_true")
    p.add_argument("--no-audio", action="store_true", help="mute game audio in every block (the curriculum saved "
                   "in the manifest shows \"audio\": false)")
    p.add_argument("--vgdl-repo", default=os.environ.get("VGDL_REPO"),
                   help="path to the language_and_experience checkout (vgdl backend)")
    p.add_argument("--coom-repo", default=os.environ.get("COOM_REPO"),
                   help="path to the TTomilin/COOM checkout (coom backend)")
    return p


def main() -> None:
    p = _parser()
    args = p.parse_args()
    if args.next_ses:
        subject = bids.subject_label(args.subject)
        print(f"{bids.next_session(args.data_root, subject):03d}")
        return
    config = load_config(args.curriculum)
    problems = validate_config(config)  # the editor's Check, so a file edited by hand gets it too
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))
    curriculum = config["curriculum"]
    w, h = (int(x) for x in args.size.lower().split("x"))
    outdir, label = _run_folder(args)
    print(f"output: {outdir}", file=sys.stderr)
    seeds = _fold_seeds(curriculum, label)
    _check_monitor(args.monitor)

    _fold_cli_options(curriculum, args)

    # Before the window: a bad section, an unopenable port or an unusable
    # output must stop the run before the session starts.
    triggers = Triggers.from_config(config.get("triggers"))
    print(f"triggers: {triggers.status()}", file=sys.stderr)
    if args.dummy_trigger:
        print("triggers: --dummy-trigger: the experimenter and scanner waits are skipped; "
              "this is a test run, not a session", file=sys.stderr)
    audio = Audio(enabled=not args.no_audio)
    print(f"audio: {audio.status()}", file=sys.stderr)
    display = Display(size=(w, h), fullscreen=args.fullscreen, vsync=not args.no_vsync,
                      monitor=args.monitor)
    session = Session(args.subject, curriculum, display, outdir,
                      audio=audio, triggers=triggers, dummy_trigger=args.dummy_trigger)
    session.logger.set_extra("seeds", seeds)
    session.logger.set_extra("versions", {  # the banner hidden above said these
        "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))})
    previous = signal.signal(signal.SIGINT, _quit_like_esc)
    try:
        completed = session.run()
    finally:
        display.close()
        audio.close()
        triggers.close()
        # Last: a terminal's Ctrl+C can come twice (to uv and to us), the second one late.
        signal.signal(signal.SIGINT, previous)
    if not completed:
        # A session script must not start the next run after an ESC.
        sys.exit(EXIT_QUIT)


if __name__ == "__main__":
    main()
