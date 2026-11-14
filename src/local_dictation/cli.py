"""Command line entry point.

  dictation              run the menu bar app
  dictation check        show permissions, devices, models
  dictation download     fetch all models into ~/Developer/local-dictation/models
  dictation transcribe F run an audio file through the full pipeline
  dictation bench        accuracy / command / latency benchmark using macOS `say`
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from .config import CONFIG_PATH, LOG_DIR, MODELS_DIR, VAD_MODEL_PATH, ensure_user_config, load_config


def setup_logging(console_level=logging.INFO) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(LOG_DIR / "dictation.log", maxBytes=2_000_000, backupCount=2)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(console_level)
    root.addHandler(sh)
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------


def read_audio(path: str | Path) -> np.ndarray:
    """Any format macOS understands -> 16 kHz mono float32 (uses afconvert)."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.wav"
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(path), str(out)],
            check=True, capture_output=True,
        )
        with wave.open(str(out)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def say(text: str, voice: str = "Samantha", rate: int | None = None) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "s.aiff"
        cmd = ["say", "-v", voice, "-o", str(out)]
        if rate:
            cmd += ["-r", str(rate)]
        subprocess.run(cmd + [text], check=True)
        return read_audio(out)


# --------------------------------------------------------------------------
# pipeline driver (same code path as the app, with a fake text field)
# --------------------------------------------------------------------------


class Pipeline:
    def __init__(self, cfg, llm: bool = False):
        from .controller import Controller
        from .executor import FakeTarget

        self.target = FakeTarget()
        self.c = Controller(cfg, target=self.target)
        self.c._load()  # on this thread: MLX must stay on one thread
        if llm:
            self.c._load_llm()

    def run(self, audio: np.ndarray, mode: str = "ptt", before: str = "") -> tuple[str, list[str], float]:
        from .executor import FakeTarget

        self.target = FakeTarget(before)
        self.c.executor.target = self.target
        self.c.executor.history.clear()
        self.c.formatter.caps_on = self.c.formatter.allcaps_on = False
        self.c._handle(("start", mode, []))
        for i in range(0, len(audio) - 511, 512):
            self.c._handle(("frame", audio[i : i + 512]))
        tail = np.zeros(512 * 3, np.float32)
        for i in range(0, len(tail), 512):
            self.c._handle(("frame", tail[i : i + 512]))
        t = time.monotonic()
        self.c._handle(("stop", t))
        return self.target.text, self.target.keys, (time.monotonic() - t) * 1000


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_check(_args) -> None:
    import sounddevice as sd

    from .macos import accessibility_trusted

    print(f"config:          {ensure_user_config()}")
    print(f"models dir:      {MODELS_DIR}")
    print(f"logs:            {LOG_DIR / 'dictation.log'}")
    print(f"VAD model:       {'ok' if VAD_MODEL_PATH.exists() else 'missing (run: dictation download)'}")
    hub = MODELS_DIR / "huggingface" / "hub"
    cfg = load_config()
    for m in (cfg.model, cfg.llm_model):
        p = hub / ("models--" + m.replace("/", "--"))
        print(f"model {m}: {'ok' if p.exists() else 'missing (run: dictation download)'}")
    from .audio import mic_permission

    print(f"accessibility:   {'granted' if accessibility_trusted() else 'NOT granted'}")
    print(f"microphone:      {mic_permission()}")
    print("input devices:")
    default = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {'*' if i == default else ' '} {d['name']}")


def cmd_mic_test(args) -> None:
    """Record a few seconds, show the level live, then transcribe."""
    from .audio import Microphone, mic_permission

    cfg = load_config()
    print(f"microphone permission: {mic_permission()}")
    frames: list = []
    mic = Microphone(cfg.input_device)
    mic.ensure_open()
    mic.start_capture(sink=frames.append, on_begin=lambda pre: None)
    print(f"Speak now ({args.seconds}s)…")
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        time.sleep(0.1)
        recent = frames[-3:]
        level = max((float(np.abs(f).max()) for f in recent), default=0.0)
        bar = "#" * min(50, int(level * 100))
        print(f"\r  level {level:6.4f} |{bar:<50}|", end="", flush=True)
    mic.stop_capture(on_end=lambda: None)
    mic.close()
    audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    print(f"\n  captured {len(audio) / 16000:.1f}s, peak {peak:.4f}")
    if peak < 1e-4:
        print("  -> SILENCE: macOS is blocking the microphone for this terminal app.\n"
              "     System Settings > Privacy & Security > Microphone > enable your terminal, then restart it.")
        return
    from .engine import Transcriber

    tr = Transcriber(cfg.model)
    print(f"  heard: {tr.transcribe(audio)!r}")


def cmd_download(_args) -> None:
    from huggingface_hub import snapshot_download

    from .engine import ensure_vad_model

    cfg = load_config()
    ensure_vad_model()
    for m in (cfg.model, cfg.llm_model):
        print(f"downloading {m} …")
        snapshot_download(m)
    print(f"done. models are in {MODELS_DIR}")


def cmd_transcribe(args) -> None:
    cfg = load_config()
    p = Pipeline(cfg, llm=args.llm)
    audio = read_audio(args.file)
    text, keys, ms = p.run(audio, "command" if args.command else "ptt")
    print(f"--- output ({len(audio) / 16000:.1f}s audio, {ms:.0f} ms after release) ---")
    print(text)
    if keys:
        print(f"--- keys: {keys}")


# (spoken text for `say`, expected text, expected keys, needs llm)
# [[slnc N]] inserts N ms of silence so a command stands alone.
COMMAND_CASES = [
    ("hello comma world period", "Hello, world.", [], False),
    ("how are you question mark", "How are you?", [], False),
    ("open paren this is a test close paren", "(This is a test)", [], False),
    ("first line new line second line", "First line\nSecond line", [], False),
    ("dear team comma new paragraph thanks for your help period", "Dear team,\n\nThanks for your help.", [], False),
    ("I clicked undo and it worked", "I clicked undo and it worked", [], False),
    ("hello world [[slnc 900]] scratch that [[slnc 900]] goodbye world", "Goodbye world", [], False),
    ("this is a test [[slnc 900]] undo", "This is a test", ["cmd+z"], False),
    ("cap paris is lovely", "Paris is lovely", [], False),
    ("it costs ten dollars period", ("It costs $10.", "It costs ten dollars."), [], False),
    ("the meeting is at noon [[slnc 900]] put that in quotes", '"The meeting is at noon"', [], True),
]

WER_SENTENCES = [
    "Please send me the quarterly report before the meeting on Thursday.",
    "The new office has two meeting rooms, a kitchen, and parking for all employees.",
    "Can you check whether the deployment finished successfully last night?",
    "We should schedule a call with the client to discuss the renewal terms.",
    "I think the second option is cheaper, but the first one is more flexible.",
    "Remember to update the documentation after you merge the pull request.",
    "Our team will be out of the office from the 20th to the 24th.",
    "The subscription price increased by about 5% compared to last year.",
]


def _wer(ref: str, hyp: str) -> tuple[int, int]:
    import re

    norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()  # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def cmd_bench(args) -> None:
    cfg = load_config()
    if args.no_llm:
        cfg.llm_enabled = False
    print("loading models …", flush=True)
    t = time.monotonic()
    p = Pipeline(cfg, llm=cfg.llm_enabled)
    print(f"loaded in {time.monotonic() - t:.1f}s\n")

    print("== Accuracy (word error rate, lower is better) ==")
    errs = words = 0
    asr_ms = []
    for s in WER_SENTENCES:
        audio = say(s, args.voice)
        t = time.monotonic()
        hyp = p.c.transcriber.transcribe(audio)
        asr_ms.append((time.monotonic() - t) * 1000)
        e, n = _wer(s, hyp)
        errs, words = errs + e, words + n
        print(f"  {'ok ' if e == 0 else f'{e} err'}  {hyp}")
    print(f"  WER {100 * errs / words:.1f}%  ({errs}/{words} words)  "
          f"median ASR {sorted(asr_ms)[len(asr_ms) // 2]:.0f} ms per sentence\n")

    print("== Commands (full pipeline: VAD -> ASR -> commands -> formatting) ==")
    passed = 0
    lat = []
    for spoken, exp_text, exp_keys, needs_llm in COMMAND_CASES:
        if needs_llm and not cfg.llm_enabled:
            continue
        audio = say(spoken, args.voice)
        text, keys, ms = p.run(audio)
        lat.append(ms)
        keys = [k for k in keys if k != "backspace"]  # edits retype via backspace
        ok = text in (exp_text if isinstance(exp_text, tuple) else (exp_text,)) and keys == exp_keys
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {ms:5.0f} ms  {spoken!r}")
        if not ok:
            print(f"        got      {text!r} keys={keys}")
            print(f"        expected {exp_text!r} keys={exp_keys}")
    total = len(lat)
    lat.sort()
    print(f"  {passed}/{total} passed")
    print(f"\n== Latency from key release to text inserted ==\n"
          f"  median {lat[len(lat) // 2]:.0f} ms, worst {lat[-1]:.0f} ms "
          f"(LLM edits included; plain dictation is the lower end)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="dictation", description="Local dictation for macOS")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run", help="run the menu bar app (default)")
    sub.add_parser("check", help="show permissions, devices and models")
    sub.add_parser("download", help="download all models")
    mt = sub.add_parser("mic-test", help="record a few seconds and show what the app hears")
    mt.add_argument("--seconds", type=float, default=4)
    t = sub.add_parser("transcribe", help="run an audio file through the pipeline")
    t.add_argument("file")
    t.add_argument("--command", action="store_true", help="treat the audio as a spoken command")
    t.add_argument("--llm", action="store_true", help="load the language model for smart edits")
    b = sub.add_parser("bench", help="benchmark with synthetic speech")
    b.add_argument("--voice", default="Samantha")
    b.add_argument("--no-llm", action="store_true")
    args = ap.parse_args(argv)

    cmd = args.cmd or "run"
    if cmd == "download":
        import os

        os.environ["HF_HUB_OFFLINE"] = "0"
    setup_logging(logging.INFO if cmd == "run" else logging.WARNING)
    if cmd == "run":
        from .ui import run_app

        run_app()
    elif cmd == "check":
        cmd_check(args)
    elif cmd == "download":
        cmd_download(args)
    elif cmd == "mic-test":
        cmd_mic_test(args)
    elif cmd == "transcribe":
        cmd_transcribe(args)
    elif cmd == "bench":
        cmd_bench(args)


if __name__ == "__main__":
    main()
