"""Optional episode archives of engine frames and source PCM.

Encoding runs off the experiment thread with a bounded queue. A full queue or
failed encoder aborts visibly instead of silently losing experiment data. The
sound track contains engine PCM timestamped at frame flips, not the delayed,
resampled and sometimes trimmed signal delivered by the playback callback.
"""
from __future__ import annotations

import os
import pathlib
import queue
import sys
import threading
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .adapters.base import Sound

_VIDEO_TIME_BASE = Fraction(1, 1_000_000)
_AUDIO_FORMATS = {
    "uint8": ("u8", "pcm_u8"),
    "int16": ("s16", "pcm_s16le"),
    "int32": ("s32", "pcm_s32le"),
    "float32": ("flt", "pcm_f32le"),
    "float64": ("dbl", "pcm_f64le"),
}


def check_recording(codec: str = "libx265") -> None:
    """Validate the optional dependency and codec before opening the display.

    :param codec: the selected encoder.
    :raises RuntimeError: if PyAV or the selected encoder is unavailable.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("video recording requires PyAV: install fmri-gym[recording]") from exc
    if codec not in ("libx265", "ffv1"):
        raise ValueError("recording codec must be libx265 or ffv1")
    av.codec.Codec(codec, "w")


class EpisodeRecorder:
    """Write one episode without making the frame loop wait for its encoder."""

    def __init__(
        self, path: pathlib.Path | str, frame_size: tuple[int, ...], *,
        fps: float = 35, video_codec: str = "libx265", audio_layout: str | None = None,
        audio_sample_rate: float = 44100, audio_dtype: str = "int16",
        max_pending_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        """Prepare streams; :meth:`start` launches the worker.

        :param path: new MKV file, never an existing recording.
        :param frame_size: engine RGB shape (height, width, channels).
        :param fps: nominal frame rate; actual frame timestamps remain variable.
        :param video_codec: libx265 (lossless YUV) or ffv1 (lossless RGB).
        :param audio_layout: mono/stereo, or None for a video-only source.
        :param audio_sample_rate: integer native PCM rate, obtained at reset.
        :param audio_dtype: native numpy PCM dtype, obtained at reset.
        :param max_pending_bytes: maximum snapshots waiting for the encoder.
        :raises ValueError: for unsupported source formats or invalid limits.
        :raises FileExistsError: if the output path already exists.
        """
        check_recording(video_codec)
        import av

        if fps <= 0 or max_pending_bytes <= 0:
            raise ValueError("recording fps and max_pending_bytes must be positive")
        if audio_layout not in (None, "mono", "stereo"):
            raise ValueError("recording currently supports mono or stereo engine PCM")
        if audio_sample_rate <= 0 or audio_sample_rate != int(audio_sample_rate):
            raise ValueError("recording requires an integer native audio sample rate")
        if audio_dtype not in _AUDIO_FORMATS:
            raise ValueError(f"unsupported recording PCM dtype: {audio_dtype}")
        self.path = pathlib.Path(path)
        self._file = self.path.open("xb")
        self._container = av.open(self._file, mode="w", format="matroska")
        self._video = self._container.add_stream(video_codec, rate=Fraction(str(fps)))
        self._video.width, self._video.height = frame_size[1], frame_size[0]
        self._video.time_base = _VIDEO_TIME_BASE
        self._video.codec_context.time_base = _VIDEO_TIME_BASE
        self._video.codec_context.thread_count = 2
        self._video.pix_fmt = "bgr0" if video_codec == "ffv1" else "yuv444p"
        if video_codec == "libx265":
            self._video.options = {"preset": "ultrafast", "x265-params":
                                   "lossless=1:pools=none:frame-threads=2:log-level=error"}
        self._audio = None
        self._rate, self._dtype = int(audio_sample_rate), np.dtype(audio_dtype)
        self._layout = audio_layout
        if audio_layout is not None:
            fmt, codec = _AUDIO_FORMATS[audio_dtype]
            self._audio = self._container.add_stream(codec, rate=self._rate)
            self._audio.layout = audio_layout
            self._audio.codec_context.format = fmt
        self._container.metadata.update({
            "audio_source": "engine_pcm_at_frame_flip; not speaker/PortAudio output",
            "video_source": "engine_rgb; not desktop or physical display capture",
        })
        self._shape = tuple(frame_size[:2]) + (3,)
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._limit, self._pending_bytes, self._peak_bytes = max_pending_bytes, 0, 0
        self._error: BaseException | None = None
        self._end: float | None = None
        self._thread: threading.Thread | None = None
        self._last_timestamp = -1.0
        self._durations: dict[int, int] = {}
        self._frames = 0
        self._chunks = 0
        self._audio_end_sample: int | None = None
        self._audio_placement = {"gap_chunks": 0, "gap_samples": 0, "max_gap_samples": 0,
                                 "overlap_chunks": 0, "overlap_samples": 0,
                                 "max_overlap_samples": 0}

    @property
    def finished(self) -> bool:
        """Whether the worker has exited and can be joined without waiting."""
        return self._thread is not None and not self._thread.is_alive()

    def start(self) -> None:
        """Start the encoder once; no daemon can outlive an incomplete archive."""
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        self._thread = threading.Thread(target=self._run, name="episode-recorder", daemon=False)
        self._thread.start()

    def check(self) -> None:
        """Surface a background failure in the experiment thread.

        :raises RuntimeError: if encoding or writing failed.
        """
        if self._error is not None:
            raise RuntimeError(f"recording failed for {self.path}: {self._error}") from self._error

    def step(self, timestamp: float, frame: np.ndarray, sound: Sound | None) -> None:
        """Snapshot a presented frame and its engine sound without encoder waits.

        :param timestamp: seconds since the episode's initial flip.
        :param frame: uint8 RGB array.
        :param sound: this step's source PCM, not the audio callback output.
        :raises RuntimeError: if the encoder failed, stopped or exhausted its queue.
        :raises ValueError: if timestamps or source formats change unexpectedly.
        """
        self.check()
        if self._stopped.is_set():
            raise RuntimeError("cannot append to a stopped recording")
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= self._last_timestamp:
            raise ValueError("recording timestamps must be finite and strictly increasing")
        if frame.shape != self._shape or frame.dtype != np.uint8:
            raise ValueError(f"recording expects uint8 RGB frames shaped {self._shape}")
        pcm = self._validate_sound(sound)
        size = frame.nbytes + (pcm.nbytes if pcm is not None else 0)
        with self._lock:
            if self._pending_bytes + size > self._limit:
                raise RuntimeError("recording queue is full; lower resolution or select a faster "
                                   "encoder before retrying (no frames were silently dropped)")
            self._pending_bytes += size
            self._peak_bytes = max(self._peak_bytes, self._pending_bytes)
        try:
            item = (timestamp, frame.copy(), None if pcm is None else pcm.copy(), size)
        except BaseException:
            with self._lock:
                self._pending_bytes -= size
            raise
        self._queue.put_nowait(item)
        self._last_timestamp = timestamp
        self._frames += 1
        self._chunks += pcm is not None
        if pcm is not None:
            self._measure_audio_placement(timestamp, len(pcm))
        self.check()

    def _measure_audio_placement(self, timestamp: float, samples: int) -> None:
        start = round(timestamp * self._rate)
        if self._audio_end_sample is not None:
            gap = start - self._audio_end_sample
            if gap:
                kind = "gap" if gap > 0 else "overlap"
                self._audio_placement[kind + "_chunks"] += 1
                self._audio_placement[kind + "_samples"] += abs(gap)
                key = "max_" + kind + "_samples"
                self._audio_placement[key] = max(self._audio_placement[key], abs(gap))
        self._audio_end_sample = start + samples

    def _validate_sound(self, sound: Sound | None) -> np.ndarray | None:
        if sound is None or not len(sound.pcm):
            return None
        if self._audio is None:
            raise ValueError("recording got audio after a silent reset; the adapter must expose "
                             "its PCM format at reset before the MKV streams are opened")
        pcm = sound.pcm
        channels = 1 if self._layout == "mono" else 2
        if pcm.ndim != 2 or pcm.shape[1] != channels:
            raise ValueError("recording PCM channel count changed within the episode")
        if sound.sample_rate != self._rate or pcm.dtype != self._dtype:
            raise ValueError("recording PCM rate or dtype changed within the episode")
        return pcm

    def stop(self, end_timestamp: float | None = None) -> None:
        """Mark the final frame's display end and finish queued data.

        :param end_timestamp: episode end, relative to its initial flip; required
            for a precise final hold. None uses the last submitted timestamp.
        """
        if self._stopped.is_set():
            return
        if end_timestamp is not None and (not np.isfinite(end_timestamp)
                                          or end_timestamp < self._last_timestamp):
            raise ValueError("recording end cannot precede its last frame")
        self._end = max(self._last_timestamp, 0.) if end_timestamp is None else end_timestamp
        self._stopped.set()

    def join(self) -> None:
        """Wait for the archive to flush and re-raise any worker error."""
        if self._thread is None:
            self._container.close()
            self._file.close()
        else:
            self._thread.join()
        self.check()

    def describe(self) -> dict:
        """Return recording metadata for the block's manifest entry."""
        return {"file": self.path.name, "video_frames": self._frames, "audio_chunks": self._chunks,
                "audio_source": "engine_pcm_at_frame_flip", "speaker_output": False,
                "audio_timing": "source chunks at flip PTS; gaps and overlaps preserved",
                "audio_rate": self._rate if self._audio is not None else None,
                "audio_layout": self._layout, "audio_dtype": str(self._dtype),
                "queue_peak_bytes": self._peak_bytes, "queue_limit_bytes": self._limit,
                "end_seconds": self._end, "codec": self._video.codec_context.name,
                "timestamp_resolution_seconds": .001,
                "source_pcm_placement": self._audio_placement.copy(),
                "error": str(self._error) if self._error is not None else None}

    def _run(self) -> None:
        pending = None
        try:
            if sys.platform.startswith("linux"):
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
            while not (self._stopped.is_set() and self._queue.empty()):
                try:
                    item = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                timestamp, _frame, pcm, _size = item
                if pending is not None:
                    self._write_video(pending[0], pending[1], timestamp - pending[0])
                    self._release(pending[3])
                self._write_audio(timestamp, pcm)
                pending = item
            if pending is not None:
                self._write_video(pending[0], pending[1], max(1e-6, self._end - pending[0]))
                self._release(pending[3])
            self._mux_video(self._video.encode())
            if self._audio is not None:
                for packet in self._audio.encode():
                    self._container.mux(packet)
        except BaseException as exc:  # noqa: BLE001 - transferred to the caller by check/join
            self._error = exc
        finally:
            try:
                self._container.close()
                self._file.close()
            except BaseException as exc:  # noqa: BLE001 - preserve errors during encoder teardown
                if self._error is None:
                    self._error = exc
            finally:
                self._file.close()
                while not self._queue.empty():
                    self._queue.get_nowait()

    def _release(self, size: int) -> None:
        with self._lock:
            self._pending_bytes -= size

    def _write_video(self, timestamp: float, frame: np.ndarray, duration: float) -> None:
        import av

        picture = av.VideoFrame.from_ndarray(frame, format="rgb24")
        picture.time_base = _VIDEO_TIME_BASE
        picture.pts = round(timestamp / _VIDEO_TIME_BASE)
        # A zero duration after Matroska's 1 ms quantization becomes the nominal
        # frame period in players. Keep a short final hold to at least one tick.
        self._durations[picture.pts] = max(1000, round(duration / _VIDEO_TIME_BASE))
        self._mux_video(self._video.encode(picture))

    def _mux_video(self, packets) -> None:
        for packet in packets:
            # Encoder delay can reorder frames; durations belong to PTS, not output order.
            duration = self._durations.pop(packet.pts)
            packet.duration = duration
            self._container.mux(packet)

    def _write_audio(self, timestamp: float, pcm: np.ndarray | None) -> None:
        if pcm is None:
            return
        import av

        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1),
                                           format=self._audio.codec_context.format.name,
                                           layout=self._layout)
        frame.sample_rate = self._rate
        frame.time_base = Fraction(1, self._rate)
        frame.pts = round(timestamp * self._rate)
        for packet in self._audio.encode(frame):
            self._container.mux(packet)
