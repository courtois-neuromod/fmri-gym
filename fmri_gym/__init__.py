"""fmri-gym: one fMRI experiment framework for any Gymnasium-compatible game.

The experiment loop is engine-agnostic; each game engine is a small EnvAdapter
that wraps the underlying environment for one game block.
"""

from __future__ import annotations

from .adapters import get_adapter
from .session import Session, Clock
from .display import Display
from .audio import SoundDeviceGameBlockStream
from .logging import Logger

__all__ = ["Session", "Clock", "Display", "Logger", "get_adapter"]
