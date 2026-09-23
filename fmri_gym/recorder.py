import threading
import av
from fractions import Fraction
import numpy as np
import pathlib
import queue
import time

class EpisodeRecorder():

    def __init__(
            self,
            path:pathlib.Path,
            frame_size:tuple[int, int],
            audio_layout:str | None='stereo',
            audio_sample_rate:int=44100,
            video_codec='libx265',
            audio_codec='flac',
            audio_sample_format='s16',
    ) -> None:
        self.container = av.open(path, mode="w")
        time_base = Fraction(1, 65535)
        self.last_pts = -1000
        self.video_stream = self.container.add_stream(video_codec, time_base=time_base)
        self.video_stream.width = frame_size[0]
        self.video_stream.height = frame_size[1]
        self.video_stream.options = {
            'lossless': '1',
        }
        #        self.video_stream.bit_rate = 200000 * 10e3

        if audio_layout is not None:
            self.audio_stream = self.container.add_stream(audio_codec, rate=audio_sample_rate)
            self.audio_stream.layout = audio_layout
            self.audio_stream.codec_context.format = audio_sample_format


        self.steps = queue.Queue()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()


    def _run(self)-> None:
        while True:
            if self.steps.empty():
                if self._stop_event.is_set():
                    break
                time.sleep(.01)
            else:
                with self.lock:
                    timestamp, frame, audio = self.steps.get()
                    self._step(timestamp, frame, audio)
        for packet in self.video_stream.encode(None):
            self.container.mux(packet)
        if hasattr(self, 'audio_stream'):
            for packet in self.audio_stream.encode(None):
                self.container.mux(packet)
        self.container.close()

    def step(self, timestamp, frame, audio) -> None:
        with self.lock:
            self.steps.put((timestamp, frame, audio))

    def _step(self, timestamp, frame, audio) -> None:
        v_frame = av.VideoFrame.from_image(frame)
        pts = int(timestamp / time_base)
        pts = max(pts, self.last_pts + 1)
        v_frame.pts = pts
        self.last_pts = pts
        for packet in self.video_stream.encode(v_frame):
            self.container.mux(packet)

        if hasattr(self, 'audio_stream'):
            a_frame = av.AudioFrame.from_ndarray(
                audio,
                format='s16',
                layout='stereo',
            )
            for packet in self.audio_stream.encode(a_frame):
                self.container.mux(packet)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self) -> None:
        self.thread.join()
