"""The app's brain: hotkey events in, audio in, actions out.

Threads:
* main thread  - hotkeys, menu bar, overlay; calls on_press/on_release/...
* audio thread - pushes 32 ms frames into the worker queue while capturing
* worker       - owns all models (MLX must stay on one thread): VAD,
                 transcription, command parsing, executing actions
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .audio import Microphone
from .commands import Action, CommandParser, apply_corrections
from .config import Config
from .engine import SAMPLE_RATE, Segmenter, SileroVAD, Transcriber
from .executor import Executor
from .formatter import Formatter

log = logging.getLogger(__name__)


class NullUI:
    def set_state(self, state: str, detail: str = "") -> None: ...
    def show_overlay(self, title: str, text: str) -> None: ...
    def hide_overlay(self, delay: float = 0.0) -> None: ...
    def show_transcript(self, title: str, text: str) -> None: ...
    def hide_transcript(self, delay: float = 0.0) -> None: ...
    def play(self, sound: str) -> None: ...


@dataclass
class Session:
    mode: str  # ptt | handsfree | command
    started: float = field(default_factory=time.monotonic)
    frames: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    shown: list = field(default_factory=list)
    partial: str = ""
    last_partial: float = 0.0
    utterances: int = 0
    peak: float = 0.0


TITLES = {"ptt": "Listening", "handsfree": "Hands-free", "command": "Command"}


def describe(actions: list[Action], text: str) -> str:
    if actions and all(a.kind == "text" for a in actions) and not any(a.label for a in actions):
        return text
    return " · ".join(f"⌘ {a.label or a.kind}" if a.kind != "text" else text for a in actions)


class Controller:
    def __init__(self, cfg: Config, ui=None, target=None):
        self.cfg = cfg
        self.ui = ui or NullUI()
        self.target = target
        self.q: queue.Queue = queue.Queue()
        self.mic = Microphone(cfg.input_device, cfg.preroll_ms)
        self.mode: str | None = None  # main-thread view of the session
        self.ready = False
        self.last_tap = 0.0
        self.last_latency_ms: float | None = None
        self.session: Session | None = None
        self._stop_requested = False
        self.transcriber: Transcriber | None = None
        self.llm = None
        self._build_text_stack(cfg)
        self.worker = threading.Thread(target=self._run, name="worker", daemon=True)

    def _build_text_stack(self, cfg: Config) -> None:
        self.parser = CommandParser(
            key_commands=cfg.key_commands,
            snippets=cfg.snippets,
            auto_punctuation=cfg.auto_punctuation,
            final_period=cfg.final_period,
            llm_enabled=cfg.llm_enabled,
        )
        self.formatter = Formatter(cfg.vocabulary)
        old = getattr(self, "executor", None)
        self.executor = Executor(self.target, self.formatter, self.parser, llm=self.llm, on_stop=self._request_stop)
        if old is not None:
            self.executor.history = old.history

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.worker.start()
        threading.Thread(target=self._idle_mic_closer, name="mic-idle", daemon=True).start()

    def reload(self, cfg: Config) -> None:
        self.q.put(("reload", cfg))

    def _idle_mic_closer(self) -> None:
        while True:
            time.sleep(5)
            if (
                self.mic.is_open
                and self.mode is None
                and time.monotonic() - self.mic.last_used > self.cfg.mic_idle_close_s
            ):
                self.mic.close()

    # ------------------------------------------------------------------
    # main thread: hotkey handler interface
    # ------------------------------------------------------------------

    def on_press(self, role: str) -> None:
        if not self.ready:
            self.ui.show_overlay("Starting", "Still loading the speech model…")
            self.ui.hide_overlay(1.5)
            return
        if role == "dictate" and self.mode is None:
            self._begin("ptt")
        elif role == "command" and self.mode is None:
            self._begin("command")

    def on_release(self, role: str) -> None:
        if (role == "dictate" and self.mode == "ptt") or (role == "command" and self.mode == "command"):
            self._finish()

    def on_tap(self, role: str) -> None:
        if role == "command":
            if self.mode == "command":
                self._abort()
            return
        if self.mode == "handsfree":
            self._finish()
            return
        if self.mode == "ptt":
            self._abort()
        if self.cfg.handsfree_tap == "single":
            self._begin("handsfree")
            return
        now = time.monotonic()
        if now - self.last_tap < self.cfg.double_tap_ms / 1000:
            self.last_tap = 0.0
            self._begin("handsfree")
        else:
            self.last_tap = now

    def on_cancel(self, role: str) -> None:
        if (role == "dictate" and self.mode == "ptt") or (role == "command" and self.mode == "command"):
            self._abort()

    def toggle_handsfree(self) -> None:
        if self.mode == "handsfree":
            self._finish()
        elif self.mode is None and self.ready:
            self._begin("handsfree")

    # ------------------------------------------------------------------

    def _begin(self, mode: str) -> None:
        try:
            self.mic.ensure_open()
        except Exception as e:
            log.exception("microphone failed")
            self.ui.set_state("error", f"Microphone: {e}")
            return
        self.mode = mode
        self.mic.start_capture(
            sink=lambda f: self.q.put(("frame", f)),
            on_begin=lambda pre: self.q.put(("start", mode, pre)),
        )
        self.ui.set_state(mode)
        self.ui.show_transcript(TITLES[mode], "")
        if mode == "handsfree":
            self.ui.play("Tink")

    def _finish(self) -> None:
        mode, self.mode = self.mode, None
        t = time.monotonic()
        self.mic.stop_capture(on_end=lambda: self.q.put(("stop", t)))
        self.ui.set_state("idle")
        if mode == "handsfree":
            self.ui.play("Pop")

    def _abort(self) -> None:
        self.mode = None
        self.mic.stop_capture(on_end=lambda: self.q.put(("cancel",)))
        self.ui.set_state("idle")
        self.ui.hide_transcript()

    def _request_stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    def _load(self) -> None:
        cfg = self.cfg
        self.ui.set_state("loading", "Loading speech model…")
        self.transcriber = Transcriber(cfg.model, cfg.boosts(), cfg.beam_size)
        self.transcriber.warmup()
        self.segmenter = Segmenter(
            SileroVAD(), cfg.vad_threshold, cfg.silence_ms, cfg.max_utterance_s
        )
        self.ready = True
        self.ui.set_state("idle")
        log.info("ready")
        if cfg.llm_enabled:
            self.q.put(("load_llm",))

    def _run(self) -> None:
        try:
            self._load()
        except Exception as e:
            log.exception("loading failed")
            self.ui.set_state("error", f"Loading failed: {e}")
            return
        while True:
            msg = self.q.get()
            try:
                self._handle(msg)
            except Exception as e:
                log.exception("worker error on %s", msg[0])
                self.session = None
                self.ui.show_overlay("Error", str(e))
                self.ui.hide_overlay(3)

    def _handle(self, msg) -> None:
        kind = msg[0]
        if kind == "frame":
            if self.session:
                self._frame(msg[1])
        elif kind == "start":
            self.segmenter.reset()
            self.session = Session(msg[1])
            self._stop_requested = False
            for f in msg[2]:
                self._frame(f, partials=False)
        elif kind == "stop":
            if self.session:
                self._finalize(self.session, msg[1])
        elif kind == "cancel":
            self.session = None
        elif kind == "load_llm":
            self._load_llm()
        elif kind == "reload":
            self._reload(msg[1])

    def _load_llm(self) -> None:
        from .llm import EditLLM

        try:
            self.llm = EditLLM(self.cfg.llm_model)
            self.llm.warmup(self.parser.command_names())
            self.executor.llm = self.llm
        except Exception:
            log.exception("could not load the language model; smart edits disabled")

    def _reload(self, cfg: Config) -> None:
        old = self.cfg
        self.cfg = cfg
        self._build_text_stack(cfg)
        self.transcriber.set_boosts(cfg.boosts(), cfg.beam_size)
        self.segmenter = Segmenter(SileroVAD(), cfg.vad_threshold, cfg.silence_ms, cfg.max_utterance_s)
        if cfg.llm_enabled and self.llm is None:
            self._load_llm()
        if old.input_device != cfg.input_device or old.preroll_ms != cfg.preroll_ms:
            self.mic.close()
            self.mic = Microphone(cfg.input_device, cfg.preroll_ms)
        log.info("config reloaded")

    # -- audio -------------------------------------------------------------

    def _frame(self, frame: np.ndarray, partials: bool = True) -> None:
        s = self.session
        s.frames.append(frame)
        s.peak = max(s.peak, float(np.abs(frame).max()))
        if s.mode == "command":
            if partials:
                self._partial(s, lambda: np.concatenate(s.frames))
            return
        for utt in self.segmenter.feed(frame):
            self._utterance(s, utt)
            if self.session is not s:  # "stop listening" ended the session
                return
        if partials and self.segmenter.in_speech:
            self._partial(s, self.segmenter.current)

    def _partial(self, s: Session, get_audio) -> None:
        now = time.monotonic()
        if now - s.last_partial < 0.45 or self.q.qsize() > 4:
            return
        audio = get_audio()
        if audio is None or len(audio) < SAMPLE_RATE * 0.4 or len(audio) > SAMPLE_RATE * 20:
            return
        s.last_partial = now
        s.partial = self.transcriber.transcribe(audio, partial=True)
        self._show(s)

    def _show(self, s: Session) -> None:
        text = " ".join(x for x in s.shown + [s.partial] if x)
        self.ui.show_transcript(TITLES[s.mode], text)

    def _utterance(self, s: Session, audio: np.ndarray) -> None:
        t = time.monotonic()
        text = self.transcriber.transcribe(audio)
        asr_ms = (time.monotonic() - t) * 1000
        s.utterances += 1
        s.partial = ""
        if not text:
            return
        text = apply_corrections(text, self.cfg.corrections)
        actions = self.parser.parse(text)
        log.info("utterance %.1fs asr=%.0fms %r -> %s", len(audio) / SAMPLE_RATE, asr_ms, text,
                 [a.kind for a in actions])
        s.shown.append(describe(actions, text))
        if s.mode == "handsfree":
            self.executor.run(actions)
            if self._stop_requested:
                self._end_handsfree_from_worker(s)
                return
        else:
            s.pending.extend(actions)
        self._show(s)

    def _end_handsfree_from_worker(self, s: Session) -> None:
        self._stop_requested = False
        self.mode = None
        self.mic.stop_capture(on_end=lambda: None)
        self.session = None
        self.ui.set_state("idle")
        self.ui.play("Pop")
        self.ui.hide_transcript(0.6)

    def _check_level(self, s: Session) -> bool:
        """Warn when the microphone delivered silence (usually a permission problem)."""
        seconds = len(s.frames) * 512 / SAMPLE_RATE
        log.info("session %s: %.1fs audio, peak level %.4f", s.mode, seconds, s.peak)
        if seconds > 0.5 and s.peak < 1e-4:
            from .audio import mic_permission

            perm = mic_permission()
            log.warning("microphone delivered silence (permission: %s)", perm)
            msg = (
                "The microphone delivered only silence. Allow Microphone access for your "
                "terminal app in System Settings \u203a Privacy & Security \u203a Microphone, then restart."
                if perm != "authorized"
                else "The microphone delivered only silence. Check the input device in System Settings \u203a Sound."
            )
            self.ui.show_overlay("No sound", msg)
            self.ui.hide_overlay(8)
            return False
        if seconds > 0.5 and s.peak < 0.01:
            log.warning("very quiet input (peak %.4f) - wrong microphone selected?", s.peak)
        return True

    def _finalize(self, s: Session, t_release: float) -> None:
        self.session = None
        if not self._check_level(s):
            return
        if s.mode == "command":
            self._finalize_command(s, t_release)
            return
        rest = self.segmenter.flush()
        if rest is not None:
            self._utterance(s, rest)
        elif s.utterances == 0 and len(s.frames) * 512 / SAMPLE_RATE > 0.4:
            # VAD heard nothing clear (quiet voice) - try the whole recording.
            self._utterance(s, np.concatenate(s.frames))
        if s.mode == "ptt" and s.pending:
            self.executor.run(s.pending)
        self.last_latency_ms = (time.monotonic() - t_release) * 1000
        log.info("release -> done %.0fms", self.last_latency_ms)
        if s.shown:
            self._show(s)
            self.ui.hide_transcript(0.8)
        else:
            self.ui.hide_transcript()

    def _finalize_command(self, s: Session, t_release: float) -> None:
        audio = np.concatenate(s.frames) if s.frames else np.zeros(0, np.float32)
        if len(audio) < SAMPLE_RATE * 0.25:
            self.ui.hide_transcript()
            return
        text = apply_corrections(self.transcriber.transcribe(audio), self.cfg.corrections)
        if not text:
            self.ui.hide_transcript()
            return
        actions = self.parser.parse(text, command_mode=True)
        labels = self.executor.run(actions)
        self.last_latency_ms = (time.monotonic() - t_release) * 1000
        log.info("command %r -> %s (%.0fms)", text, labels, self.last_latency_ms)
        result = ", ".join(labels) or "done"
        if "not understood" in labels:
            self.ui.play("Funk")
        self.ui.show_transcript("Command", f"{text}  →  {result}")
        self.ui.hide_transcript(1.2)
