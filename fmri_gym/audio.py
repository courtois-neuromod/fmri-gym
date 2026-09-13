"""Streaming PCM output via a PortAudio callback (sounddevice).

:meth:`play` starts the stream; PortAudio then calls :meth:`callback` on
its own realtime thread (not from the caller). Each callback fills one
host buffer (``outdata``) with the next slice of samples and returns
immediately — it does not play a whole clip in one go. Keep the callback
short: if it blocks or the queue runs dry, the buffer underruns and you
hear silence or clicks.

The caller pushes numpy blocks with :meth:`put` on another thread; the
callback pulls from the queue. A queued block is often longer or shorter
than one host buffer, so the callback's inner loop splices across blocks
until that single slot is full, then returns while the caller keeps
running.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import sounddevice

# Stream lifecycle. ``NOT_STARTED`` is unused: we construct already
# ``STOPPED`` and only start the PortAudio stream in :meth:`play`.
NOT_STARTED = 0
PLAYING = 1
STOPPED = 2


class SoundDeviceGameBlockStream:
    """Queue PCM blocks from the caller onto a PortAudio output stream."""

    def __init__(
        self,
        sample_rate: float,
        block_size: int = 0,
        channels: int = 2,
        dtype=sounddevice.default.dtype[1],
    ) -> None:
        """Open an output stream; does not start playback until :meth:`play`.

        :param sample_rate: samples per second; must match the input blocks.
        :param block_size: PortAudio block size in frames; ``0`` lets the
            host pick. When the caller knows its chunk length, pass it so
            one queued block often fills one callback.
        :param channels: channel count of queued arrays (stereo is 2).
        :param dtype: numpy / PortAudio sample dtype. Default is the
            device's output dtype (``sounddevice.default.dtype[1]``).
        """
        self.blocks: queue.Queue = queue.Queue()
        # Seed ~100 ms of silence. Matches ``latency=0.1`` below so the
        # first callback does not underrun before real samples are queued.
        self.blocks.put(np.zeros((int(0.1 * sample_rate), channels), dtype=dtype))
        self.lock = threading.Lock()
        self.output_stream = sounddevice.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            latency=0.1,
            device=None,
            channels=channels,
            callback=self.callback,
            dtype=dtype,
            # Let PortAudio zero the initial buffers instead of calling us
            # before :meth:`play` (and before any real game audio is queued).
            prime_output_buffers_using_stream_callback=False,
        )
        self.current_block_idx = 0
        self.current_block = None
        self.status = STOPPED

    def callback(self, outdata, frames: int, time, status) -> None:
        """PortAudio output callback: copy queued PCM into ``outdata``.

        Runs on PortAudio's thread. Keep it short: fill ``outdata`` and
        return. Blocking here underruns and clicks.

        A single queued block is often longer or shorter than ``frames``, so
        we splice across queue items until this callback's slot is full.

        :param outdata: preallocated output array, shape ``(frames, C)``.
        :param frames: number of sample frames PortAudio wants this call.
        :param time: PortAudio timing info (unused).
        :param status: PortAudio status flags (unused).
        """
        if self.status == STOPPED:
            return
        if self.blocks.empty():
            outdata.fill(0)
            logging.debug("sound queue empty")
            return
        elif self.current_block is None:
            with self.lock:
                self.current_block = self.blocks.get()

        out_idx = 0
        while True:
            current_block_len = self.current_block.shape[0]

            split_idx = min(current_block_len - self.current_block_idx, frames - out_idx)
            split_end = self.current_block_idx + split_idx
            outdata[out_idx : out_idx + split_idx] = self.current_block[
                self.current_block_idx : split_end
            ]
            out_idx += split_idx

            self.current_block_idx = split_end
            if split_end == current_block_len:
                with self.lock:
                    try:
                        # Tiny timeout so a late producer put does not stall
                        # PortAudio; underrun is preferable to a hang.
                        self.current_block = self.blocks.get(timeout=0.01)
                    except queue.Empty:
                        logging.debug("sound queue empty")
                self.current_block_idx = 0
            if out_idx == frames:
                return

    def put(self, block: np.ndarray) -> None:
        """Enqueue one PCM chunk from the producer thread.

        :param block: array shaped ``(n_samples, channels)``, same dtype
            as the stream. The queue holds a reference; copy first if the
            underlying buffer will be reused.
        """
        with self.lock:
            self.blocks.put(block)

    def play(self) -> None:
        """Start the PortAudio stream (idempotent if already running)."""
        self.status = PLAYING
        self.output_stream.start()

    def stop(self) -> None:
        """Stop playback and drop any queued samples.

        Flush after stop so leftover PCM is not played on a later
        :meth:`play`.
        """
        self.status = STOPPED
        self.output_stream.stop()
        self.flush()

    def flush(self) -> None:
        """Drop queued blocks. Replaces the queue rather than draining it.

        Draining while the callback may still hold ``current_block`` is
        racy; a fresh ``Queue`` is the simple cutoff.
        """
        self.blocks = queue.Queue()
