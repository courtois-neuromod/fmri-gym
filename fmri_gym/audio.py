import sounddevice
import threading
import queue
import numpy as np
import logging

NOT_STARTED = 0
PLAYING = 1
STOPPED = 2

class SoundDeviceGameBlockStream(object):

    def __init__(
        self,
        sample_rate,
        block_size=0,
        channels=2,
        dtype=sounddevice.default.dtype[1]):

        self.blocks = queue.Queue()
        self.blocks.put(np.zeros((int(.1*sample_rate),2), dtype=dtype))
        self.lock = threading.Lock()
        self.output_stream = sounddevice.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            latency=0.1,
            device=None,
            channels=2,
            callback=self.callback,
            dtype=dtype,
            prime_output_buffers_using_stream_callback=False
            )
        self.current_block_idx = 0
        self.current_block = None
        self.status = STOPPED

    def callback(self, outdata, frames, time, status):
        if self.status == STOPPED:
            return
        if self.blocks.empty():
            outdata.fill(0)
            logging.debug('sound queue empty')
            return
        elif self.current_block is None:
            with self.lock:
                self.current_block = self.blocks.get()

        out_idx = 0
        while True:
            current_block_len = self.current_block.shape[0]

            split_idx = min(current_block_len-self.current_block_idx, frames-out_idx)
            split_end = self.current_block_idx + split_idx
            #print(frames,  current_block_len, out_idx, split_idx, self.current_block_idx, split_end)
            outdata[out_idx:out_idx+split_idx] = self.current_block[self.current_block_idx:split_end]
            out_idx += split_idx

            self.current_block_idx = split_end
            if split_end == current_block_len:
                with self.lock:
                    try:
                        self.current_block = self.blocks.get(timeout=.01)
                    except queue.Empty:
                        logging.debug('sound queue empty')
                self.current_block_idx = 0
            if out_idx == frames:
                return

    def put(self, block):
        with self.lock:
            self.blocks.put(block)

    def play(self):
        self.status = PLAYING
        self.output_stream.start()

    def stop(self):
        self.status = STOPPED
        self.output_stream.stop()
        self.flush()

    def flush(self):
        self.blocks = queue.Queue()
