"""Execute actions against a target (the focused app, or a fake for tests)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .commands import Action, CommandParser
from .formatter import Formatter

log = logging.getLogger(__name__)

RECENT_S = 2.0  # trust our own last insertion over reading the text field
HISTORY_S = 600.0  # how long "scratch that" / edits can refer back


class Target(Protocol):
    def insert_text(self, text: str) -> None: ...
    def press(self, combo: str) -> None: ...
    def backspace(self, n: int) -> None: ...
    def context_before(self) -> str | None: ...
    def selected_text(self) -> str | None: ...
    def frontmost_pid(self) -> int: ...


@dataclass
class Insertion:
    text: str
    pid: int
    t: float


class Executor:
    def __init__(
        self,
        target: Target,
        formatter: Formatter,
        parser: CommandParser,
        llm=None,
        on_stop: Callable[[], None] | None = None,
    ):
        self.target = target
        self.formatter = formatter
        self.parser = parser
        self.llm = llm
        self.on_stop = on_stop
        self.history: list[Insertion] = []

    # -- helpers ----------------------------------------------------------

    def _last(self, max_age: float = HISTORY_S) -> Insertion | None:
        if not self.history:
            return None
        last = self.history[-1]
        if last.pid != self.target.frontmost_pid() or time.monotonic() - last.t > max_age:
            return None
        return last

    def _context(self) -> str | None:
        recent = self._last(RECENT_S)
        if recent:
            return recent.text
        try:
            ctx = self.target.context_before()
        except Exception:  # accessibility can fail in many ways
            ctx = None
        if ctx is not None:
            return ctx
        older = self._last()
        return older.text if older else None

    def _insert(self, text: str) -> None:
        if not text:
            return
        self.target.insert_text(text)
        self.history.append(Insertion(text, self.target.frontmost_pid(), time.monotonic()))
        del self.history[:-50]

    def render(self, action: Action, context: str | None) -> str:
        return self.formatter.render(action.tokens, context)

    # -- run --------------------------------------------------------------

    def run(self, actions: list[Action]) -> list[str]:
        """Execute actions in order. Returns short labels for the UI."""
        labels: list[str] = []
        for a in actions:
            try:
                labels.append(self._run_one(a))
            except Exception as e:  # never let one bad action kill the session
                log.exception("action failed: %s", a)
                labels.append(f"error: {e}")
        return [lab for lab in labels if lab]

    def _run_one(self, a: Action) -> str:
        if a.kind == "text":
            text = self.render(a, self._context())
            self._insert(text)
            return a.label or ""
        if a.kind == "key":
            self.target.press(a.combo)
            if a.combo in ("cmd+z",) and self._last():
                self.history.pop()
            return a.label
        if a.kind == "scratch":
            last = self._last()
            if not last:
                return "nothing to scratch"
            self.target.backspace(len(last.text))
            self.history.pop()
            return a.label
        if a.kind == "stop":
            if self.on_stop:
                self.on_stop()
            return a.label
        if a.kind == "llm":
            return self._run_llm(a)
        return ""

    def _run_llm(self, a: Action) -> str:
        if self.llm is None or not self.llm.ready:
            if a.command_mode:
                return "not understood"
            # No language model: just type what was said.
            self._insert(self.formatter.render(self.parser.to_tokens(a.text), self._context()))
            return ""
        selection = None
        try:
            selection = self.target.selected_text()
        except Exception:
            pass
        last = self._last()
        target_text = selection or (last.text if last else None)
        res = self.llm.interpret(a.text, target_text, self.parser.command_names())
        log.info("llm %r -> %s", a.text, res)
        kind = res.get("action")
        if kind == "edit" and target_text and isinstance(res.get("text"), str):
            new = res["text"]
            lead = target_text[: len(target_text) - len(target_text.lstrip())]
            new = lead + new.lstrip()
            if selection:
                self.target.insert_text(new)  # replaces the selection
            else:
                self.target.backspace(len(target_text))
                self.history.pop()
                self._insert(new)
            return "edited"
        if kind == "key":
            name = str(res.get("command", "")).lower()
            acts = self.parser.parse(name, command_mode=True)
            acts = [x for x in acts if x.kind != "llm"]
            if acts:
                self.run(acts)
                return name
            return "not understood"
        if kind == "dictate" and not a.command_mode:
            self._insert(self.formatter.render(self.parser.to_tokens(a.text), self._context()))
            return ""
        return "not understood"


class FakeTarget:
    """In-memory text field used by tests and the benchmark."""

    def __init__(self, text: str = "", pid: int = 1, readable: bool = True):
        self.text = text
        self.sel = (len(text), len(text))
        self.pid = pid
        self.readable = readable
        self.keys: list[str] = []

    def _replace(self, s: str) -> None:
        a, b = self.sel
        self.text = self.text[:a] + s + self.text[b:]
        self.sel = (a + len(s), a + len(s))

    def insert_text(self, text: str) -> None:
        self._replace(text)

    def backspace(self, n: int) -> None:
        for _ in range(n):
            self.press("backspace")

    def press(self, combo: str) -> None:
        self.keys.append(combo)
        a, b = self.sel
        if combo == "backspace":
            if a != b:
                self._replace("")
            elif a > 0:
                self.sel = (a - 1, a)
                self._replace("")
        elif combo == "alt+backspace":
            i = len(self.text[:a].rstrip())
            while i > 0 and not self.text[i - 1].isspace():
                i -= 1
            self.sel = (i, b)
            self._replace("")
        elif combo == "alt+shift+left":
            i = len(self.text[:a].rstrip())
            while i > 0 and not self.text[i - 1].isspace():
                i -= 1
            self.sel = (i, b)
        elif combo == "cmd+a":
            self.sel = (0, len(self.text))
        elif combo == "return":
            self._replace("\n")
        elif combo == "tab":
            self._replace("\t")

    def context_before(self) -> str | None:
        return self.text[: self.sel[0]] if self.readable else None

    def selected_text(self) -> str | None:
        a, b = self.sel
        return self.text[a:b] or None

    def frontmost_pid(self) -> int:
        return self.pid
