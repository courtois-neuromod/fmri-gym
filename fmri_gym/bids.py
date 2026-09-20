"""Where a run's data goes, BIDS-style, and the numbers in its name.

    data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/   manifest.json, block-*.npz
    data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002/   the same task again
    data/sub-01/ses-001/beh/sub-01_ses-001_task-crafter_run-001/

The folders are the counters: nothing is kept in sync beside them. A session
not given is the subject's next free ``ses-NNN``; a run not given is the next
free ``run-NNN`` *of that task* in the session, as BIDS counts runs. Delete a
failed run's folder and its number is free again. A session script picks its
session once, in its first line (``fmri-play --next-ses``), so all its runs
land in the same one.

The run's label (``sub-01_ses-001_task-pong_run-002``) also keys its seeds
(:func:`phase_seed`), so no two runs -- of a participant, or of two
participants -- replay each other's episodes unless a phase pins its
``"seed"``.

The names follow BIDS; the contents do not yet (a manifest and ``.npz`` blocks
in a folder per run, not ``_beh.tsv`` + ``_events.tsv`` sidecars).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Callable

_LABEL = re.compile(r"[A-Za-z0-9]+")


def subject_label(subject: str) -> str:
    """The ``sub-<label>`` a subject id must be.

    :param subject: e.g. ``sub-01``.
    :return: it, unchanged.
    :raises ValueError: if it is not ``sub-`` and letters or digits (BIDS allows no other).
    """
    if not (subject.startswith("sub-") and _LABEL.fullmatch(subject[4:])):
        raise ValueError(f"--subject {subject!r}: BIDS needs sub-<letters or digits>, "
                         "such as sub-01 or sub-pilot3")
    return subject


def task_label(config_path: str) -> str:
    """A config's task label: its file name, letters and digits only (``ale__pong`` -> ``alepong``).

    :raises ValueError: if nothing is left (a file named ``__.json``).
    """
    stem = os.path.splitext(os.path.basename(config_path))[0]
    label = "".join(re.findall(r"[A-Za-z0-9]", stem))
    if not label:
        raise ValueError(f"{config_path}: no letter or digit in its name to make a BIDS task "
                         "label from; rename the file")
    return label


def next_session(root: str, subject: str) -> int:
    """The subject's first ``ses-NNN`` with no folder under ``root``: 1, 2, 3..."""
    return _next_free(os.path.join(root, subject), lambda n: f"ses-{n:03d}")


def next_run(root: str, subject: str, session: int, task: str) -> int:
    """The first ``run-NNN`` of ``task`` with no folder in the session: 1, 2, 3..."""
    return _next_free(_beh(root, subject, session),
                      lambda n: run_label(subject, session, task, n))


def run_label(subject: str, session: int, task: str, run: int) -> str:
    """``sub-01_ses-001_task-pong_run-002``: the run's folder name and seed key."""
    return f"{subject}_ses-{session:03d}_task-{task}_run-{run:03d}"


def run_dir(root: str, subject: str, session: int, task: str, run: int) -> str:
    """The folder a run writes into; see the module docstring."""
    return os.path.join(_beh(root, subject, session), run_label(subject, session, task, run))


def phase_seed(label: str, index: int) -> int:
    """A game phase's base seed when it pins none: its episodes get ``seed, seed + 1, ...``.

    A hash of the run's label and the phase's index: stable across processes
    (unlike ``hash()``), so the editor can show what a launch will use, and
    different for every run, participant and phase. Being random 31-bit numbers,
    two blocks' ranges practically never meet (``1000 + phase`` made them share).

    :param label: :func:`run_label`.
    :param index: the phase's index in the curriculum.
    :return: a seed in ``0 .. 2**31 - 1``.
    """
    digest = hashlib.sha256(f"{label}|phase-{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def _beh(root: str, subject: str, session: int) -> str:
    return os.path.join(root, subject, f"ses-{session:03d}", "beh")


def _next_free(folder: str, name: Callable[[int], str]) -> int:
    n = 1
    while os.path.exists(os.path.join(folder, name(n))):
        n += 1
    return n
