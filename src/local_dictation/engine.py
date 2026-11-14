"""Speech recognition (Parakeet on MLX) and voice activity detection (Silero)."""

from __future__ import annotations

import logging
import time
import urllib.request
from collections import deque

import numpy as np

from .config import VAD_MODEL_PATH, VAD_MODEL_URL

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms


class Transcriber:
    """Parakeet TDT. All calls must happen on the same thread (MLX)."""

    def __init__(self, model_id: str, boosts: dict[str, float] | None = None, beam_size: int = 4):
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        t = time.monotonic()
        self.mx = mx
        self.model = from_pretrained(model_id)
        log.info("loaded %s in %.1fs", model_id, time.monotonic() - t)
        self.set_boosts(boosts or {}, beam_size)

    def set_boosts(self, boosts: dict[str, float], beam_size: int) -> None:
        """Prefer these words/phrases (beam search with a word booster).
        Partials always use fast greedy decoding."""
        from parakeet_mlx import boost as b
        from parakeet_mlx.parakeet import Beam, DecodingConfig

        self.fast = DecodingConfig()
        boosts = {w: float(x) for w, x in boosts.items() if w and x > 0}
        if boosts:
            booster = b.lift(b.words(boosts), self.model.vocabulary)
            self.accurate = DecodingConfig(decoding=Beam(beam_size=beam_size, booster=booster))
        else:
            self.accurate = self.fast

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        from parakeet_mlx.audio import get_logmel

        mx = self.mx
        pad = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([pad, audio.astype(np.float32), pad])
        if len(audio) < SAMPLE_RATE // 2:
            audio = np.concatenate([audio, np.zeros(SAMPLE_RATE // 2 - len(audio), np.float32)])
        x = mx.array(audio, dtype=mx.float32)
        # get_logmel is wrapped in an unbounded lru_cache keyed on the array,
        # which would keep every utterance in memory - call the raw function.
        mel = getattr(get_logmel, "__wrapped__", get_logmel)(x, self.model.preprocessor_config)
        cfg = self.fast if partial else self.accurate
        result = self.model.generate(mel, decoding_config=cfg)[0]
        return result.text.strip()

    def warmup(self) -> None:
        t = np.random.default_rng(0).normal(0, 0.01, SAMPLE_RATE).astype(np.float32)
        self.transcribe(t)
        self.transcribe(t, partial=True)


def ensure_vad_model() -> None:
    if VAD_MODEL_PATH.exists():
        return
    VAD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading VAD model to %s", VAD_MODEL_PATH)
    urllib.request.urlretrieve(VAD_MODEL_URL, VAD_MODEL_PATH)


class SileroVAD:
    """Silero VAD v5 via onnxruntime. Takes 512-sample frames at 16 kHz."""

    def __init__(self):
        import onnxruntime as ort

        ensure_vad_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(VAD_MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = x[:, -64:]
        return float(out[0][0])


class Segmenter:
    """Splits a stream of frames into utterances separated by pauses."""

    def __init__(
        self,
        vad,
        threshold: float = 0.5,
        silence_ms: int = 600,
        max_utterance_s: float = 25.0,
        pad_ms: int = 200,
        min_speech_ms: int = 160,
    ):
        frame_ms = FRAME * 1000 / SAMPLE_RATE
        self.vad = vad
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.05)
        self.silence_frames = max(1, round(silence_ms / frame_ms))
        self.pad_frames = max(1, round(pad_ms / frame_ms))
        self.min_speech = max(1, round(min_speech_ms / frame_ms))
        self.max_frames = int(max_utterance_s * 1000 / frame_ms)
        self.reset()

    def reset(self) -> None:
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        self.pre: deque = deque(maxlen=self.pad_frames)
        self.active = False
        self.buf: list[np.ndarray] = []
        self.speech = 0
        self.silence = 0

    def _emit(self, frames) -> list[np.ndarray]:
        if self.speech >= self.min_speech and frames:
            return [np.concatenate(frames)]
        return []

    def feed(self, frame: np.ndarray) -> list[np.ndarray]:
        p = self.vad(frame)
        if not self.active:
            self.pre.append(frame)
            if p >= self.threshold:
                self.active = True
                self.buf = list(self.pre)
                self.speech, self.silence = 1, 0
            return []
        self.buf.append(frame)
        if p >= self.threshold:
            self.speech += 1
            self.silence = 0
        elif p < self.neg_threshold or self.silence > 0:
            self.silence += 1
        if self.silence >= self.silence_frames:
            keep = len(self.buf) - self.silence + self.pad_frames
            out = self._emit(self.buf[:keep])
            tail = self.buf[-self.pad_frames:]
            self.active = False
            self.buf = []
            self.pre = deque(tail, maxlen=self.pad_frames)
            return out
        if len(self.buf) >= self.max_frames:
            out = self._emit(self.buf)
            self.buf, self.speech, self.silence = [], 0, 0
            return out
        return []

    @property
    def in_speech(self) -> bool:
        return self.active

    def current(self) -> np.ndarray | None:
        return np.concatenate(self.buf) if self.active and self.buf else None

    def flush(self) -> np.ndarray | None:
        out = self._emit(self.buf) if self.active else []
        self.active = False
        self.buf, self.speech, self.silence = [], 0, 0
        return out[0] if out else None
