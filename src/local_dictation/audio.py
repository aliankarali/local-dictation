"""Microphone capture at 16 kHz mono in 32 ms frames, with a short pre-roll so
the first word is not cut off when you press the key."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from .engine import FRAME, SAMPLE_RATE

log = logging.getLogger(__name__)


_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}


def mic_permission() -> str:
    """Microphone permission of the app that launched us (e.g. Terminal).
    Without it macOS silently delivers zeros instead of failing."""
    import AVFoundation as A

    return _STATUS.get(A.AVCaptureDevice.authorizationStatusForMediaType_(A.AVMediaTypeAudio), "unknown")


def request_mic_permission(callback: Callable[[bool], None]) -> None:
    """Shows the system prompt if the user was never asked."""
    import AVFoundation as A

    A.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        A.AVMediaTypeAudio, lambda granted: callback(bool(granted))
    )


def find_input_device(name: str):
    import sounddevice as sd

    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    log.warning("input device %r not found, using default", name)
    return None


class Microphone:
    def __init__(self, device: str = "", preroll_ms: int = 300):
        self.device = device
        self.lock = threading.Lock()
        self.ring: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(preroll_ms / 32)))
        self.capturing = False
        self.sink: Callable[[np.ndarray], None] | None = None
        self.stream = None
        self.last_used = 0.0
        self.overflows = 0

    @property
    def is_open(self) -> bool:
        return self.stream is not None

    def ensure_open(self) -> None:
        import sounddevice as sd

        if self.stream is not None:
            return
        t = time.monotonic()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME,
            latency="high",
            device=find_input_device(self.device),
            callback=self._callback,
        )
        self.stream.start()
        log.info("microphone opened in %.0fms", (time.monotonic() - t) * 1000)

    def close(self) -> None:
        with self.lock:
            stream, self.stream = self.stream, None
            self.ring.clear()
        if stream is not None:
            stream.stop()
            stream.close()
            log.info("microphone closed")

    def _callback(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflows += 1
        frame = indata[:, 0].copy()
        with self.lock:
            if self.capturing and self.sink is not None:
                self.sink(frame)
            else:
                self.ring.append(frame)

    def start_capture(self, sink: Callable[[np.ndarray], None], on_begin: Callable[[list], None]) -> None:
        """on_begin(preroll_frames) runs under the lock, before any frame reaches sink."""
        with self.lock:
            on_begin(list(self.ring))
            self.ring.clear()
            self.sink = sink
            self.capturing = True
        self.last_used = time.monotonic()

    def stop_capture(self, on_end: Callable[[], None]) -> None:
        """on_end runs under the lock, after the last frame reached sink."""
        with self.lock:
            self.capturing = False
            self.sink = None
            on_end()
        self.last_used = time.monotonic()
