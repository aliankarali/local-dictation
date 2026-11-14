"""Hotkey state machine: hold, tap, double-tap, shortcut cancel."""

from local_dictation.config import Config
from local_dictation.controller import Controller
from local_dictation.executor import FakeTarget


class FakeMic:
    is_open = True
    last_used = 0.0

    def ensure_open(self):
        pass

    def start_capture(self, sink, on_begin):
        on_begin([])

    def stop_capture(self, on_end):
        on_end()


def make():
    c = Controller(Config(llm_enabled=False), target=FakeTarget())
    c.mic = FakeMic()
    c.ready = True
    return c


def drain(c):
    out = []
    while not c.q.empty():
        out.append(c.q.get()[0])
    return out


def test_hold_to_talk():
    c = make()
    c.on_press("dictate")
    assert c.mode == "ptt"
    c.on_release("dictate")
    assert c.mode is None
    assert drain(c) == ["start", "stop"]


def test_single_tap_toggles_handsfree():
    c = make()
    c.on_press("dictate")
    c.on_tap("dictate")
    assert c.mode == "handsfree"
    c.on_press("dictate")  # ignored while hands-free
    c.on_tap("dictate")
    assert c.mode is None
    assert drain(c) == ["start", "cancel", "start", "stop"]


def test_double_tap_mode():
    c = make()
    c.cfg.handsfree_tap = "double"
    c.on_press("dictate")
    c.on_tap("dictate")
    assert c.mode is None
    c.on_press("dictate")
    c.on_tap("dictate")
    assert c.mode == "handsfree"
    c.on_press("dictate")  # ignored while hands-free
    c.on_tap("dictate")
    assert c.mode is None
    assert drain(c) == ["start", "cancel", "start", "cancel", "start", "stop"]


def test_shortcut_cancels():
    c = make()
    c.on_press("dictate")
    c.on_cancel("dictate")
    assert c.mode is None
    assert drain(c) == ["start", "cancel"]


def test_command_key():
    c = make()
    c.on_press("command")
    assert c.mode == "command"
    c.on_release("command")
    assert drain(c) == ["start", "stop"]
    c.on_press("dictate")
    c.on_press("command")  # ignored while dictating
    assert c.mode == "ptt"


def test_not_ready():
    c = make()
    c.ready = False
    c.on_press("dictate")
    assert c.mode is None
