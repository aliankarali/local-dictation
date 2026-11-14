"""Paths and configuration.

Large downloads (models) live under ~/Developer/local-dictation so they stay
out of the project folder. Override with LOCAL_DICTATION_DATA.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(
    os.environ.get("LOCAL_DICTATION_DATA", Path.home() / "Developer" / "local-dictation")
)
MODELS_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
VAD_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
VAD_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
CONFIG_DIR = Path(
    os.environ.get("LOCAL_DICTATION_CONFIG_DIR", Path.home() / ".config" / "local-dictation")
)
CONFIG_PATH = CONFIG_DIR / "config.toml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.toml")


def setup_environment() -> None:
    """Point Hugging Face downloads at the models dir. Call before importing HF/MLX libs."""
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Stay offline once models are downloaded (no network calls at startup).
    # `dictation download` switches this off.
    if (MODELS_DIR / "huggingface" / "hub").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    dictate_key: str = "right_option"
    command_key: str = "right_command"
    tap_max_ms: int = 250
    double_tap_ms: int = 400
    handsfree_tap: str = "single"

    input_device: str = ""
    silence_ms: int = 600
    max_utterance_s: float = 25.0
    vad_threshold: float = 0.5
    mic_idle_close_s: float = 300
    preroll_ms: int = 300

    model: str = "mlx-community/parakeet-tdt-0.6b-v2"
    beam_size: int = 4
    vocabulary_boost: float = 2.0
    command_boost: float = 2.0

    output_method: str = "paste"
    restore_clipboard: bool = True
    sounds: bool = True
    show_transcript: bool = False

    auto_punctuation: bool = True
    final_period: bool = False

    llm_enabled: bool = True
    llm_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    vocabulary: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    key_commands: dict[str, str] = field(default_factory=dict)
    snippets: dict[str, str] = field(default_factory=dict)

    def boosts(self) -> dict[str, float]:
        from .commands import command_boosts

        extra = list(self.key_commands) + list(self.snippets)
        out = command_boosts(self.command_boost, extra)
        out.update({w: self.vocabulary_boost for w in self.vocabulary})
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        g = lambda section, key, default: d.get(section, {}).get(key, default)  # noqa: E731
        c = cls()
        return cls(
            dictate_key=g("hotkeys", "dictate", c.dictate_key),
            command_key=g("hotkeys", "command", c.command_key),
            tap_max_ms=int(g("hotkeys", "tap_max_ms", c.tap_max_ms)),
            double_tap_ms=int(g("hotkeys", "double_tap_ms", c.double_tap_ms)),
            handsfree_tap=g("hotkeys", "handsfree_tap", c.handsfree_tap),
            input_device=g("audio", "device", c.input_device),
            silence_ms=int(g("audio", "silence_ms", c.silence_ms)),
            max_utterance_s=float(g("audio", "max_utterance_s", c.max_utterance_s)),
            vad_threshold=float(g("audio", "vad_threshold", c.vad_threshold)),
            mic_idle_close_s=float(g("audio", "mic_idle_close_s", c.mic_idle_close_s)),
            preroll_ms=int(g("audio", "preroll_ms", c.preroll_ms)),
            model=g("engine", "model", c.model),
            beam_size=int(g("engine", "beam_size", c.beam_size)),
            vocabulary_boost=float(g("engine", "vocabulary_boost", c.vocabulary_boost)),
            command_boost=float(g("engine", "command_boost", c.command_boost)),
            output_method=g("output", "method", c.output_method),
            restore_clipboard=bool(g("output", "restore_clipboard", c.restore_clipboard)),
            sounds=bool(g("output", "sounds", c.sounds)),
            show_transcript=bool(g("output", "show_transcript", c.show_transcript)),
            auto_punctuation=bool(g("text", "auto_punctuation", c.auto_punctuation)),
            final_period=bool(g("text", "final_period", c.final_period)),
            llm_enabled=bool(g("llm", "enabled", c.llm_enabled)),
            llm_model=g("llm", "model", c.llm_model),
            vocabulary=list(g("vocabulary", "words", [])),
            corrections=dict(d.get("corrections", {})),
            key_commands=dict(d.get("key_commands", {})),
            snippets=dict(d.get("snippets", {})),
        )


def ensure_user_config() -> Path:
    if not CONFIG_PATH.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEFAULT_CONFIG_PATH, CONFIG_PATH)
    return CONFIG_PATH


def load_config(path: Path | None = None) -> Config:
    path = path or ensure_user_config()
    with open(path, "rb") as f:
        return Config.from_dict(tomllib.load(f))
