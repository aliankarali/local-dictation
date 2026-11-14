"""Render tokens to text: spacing and capitalization, aware of the text that
is already before the cursor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .commands import Directive, Sym, Token, Word

SENTENCE_END = ".?!"
OPENERS = "([{\"'“‘"
NO_SPACE_AFTER = set("([{\n\t/\\@#$_“‘")
I_WORDS = {"i", "i'm", "i'll", "i've", "i'd"}


@lru_cache(maxsize=1)
def load_common_words() -> frozenset[str]:
    """Lowercase entries of the system word list = common (non-proper) words."""
    p = Path("/usr/share/dict/words")
    if not p.exists():
        return frozenset()
    return frozenset(w for w in p.read_text().split() if w.islower())


def context_state(before: str | None) -> tuple[bool | None, bool]:
    """Return (at_sentence_start, needs_space) for text that comes after `before`.

    `before=None` means unknown (we could not read the text field)."""
    if before is None:
        return None, False
    if before == "":
        return True, False
    last = before[-1]
    if last.isspace() or last in NO_SPACE_AFTER:
        needs_space = False
    elif last in "\"'":
        prev = before[-2] if len(before) > 1 else " "
        needs_space = not (prev.isspace() or prev in OPENERS)  # closing quote
    else:
        needs_space = True
    s = before.rstrip(" \t")
    if not s or s.endswith("\n"):
        return True, needs_space
    s = s.rstrip(OPENERS)
    if not s:
        return True, needs_space
    return s[-1] in SENTENCE_END, needs_space


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _decap(s: str) -> str:
    return s[:1].lower() + s[1:]


class Formatter:
    def __init__(self, vocabulary: list[str] | tuple = (), common_words=None):
        self.common = load_common_words() if common_words is None else common_words
        self.vocab = {v.lower(): v for v in vocabulary}
        self.caps_on = False
        self.allcaps_on = False

    def _is_common(self, word: str) -> bool:
        if not word[:1].isupper():
            return False
        if len(word) > 1 and word.isupper():  # acronym
            return False
        low = word.lower()
        if low in I_WORDS:
            return False
        base = low.replace("’", "'").split("'")[0].split("-")[0]
        return base in self.common

    def _case(self, w: Word, state: bool | None) -> str:
        text = w.text
        if w.force == "upper":
            return text.upper()
        if w.force == "lower":
            return text.lower()
        if w.force == "cap":
            return _cap(text)
        if self.allcaps_on:
            return text.upper()
        if self.caps_on:
            return _cap(text)
        canonical = self.vocab.get(text.lower())
        if canonical:
            return _cap(canonical) if state is True else canonical
        if state is True:
            return _cap(text)
        if state is None:
            return text
        if (w.first_in_utterance or w.after_dropped_end) and self._is_common(text):
            return _decap(text)
        return text

    def render(self, tokens: list[Token], before: str | None) -> str:
        """`before` is the text in front of the cursor (None if unknown)."""
        state, space_ok = context_state(before)
        out: list[str] = []
        no_space = False
        for t in tokens:
            if isinstance(t, Directive):
                if t.name == "no_space":
                    no_space = True
                elif t.name == "caps_on":
                    self.caps_on, self.allcaps_on = True, False
                elif t.name == "caps_off":
                    self.caps_on = False
                elif t.name == "allcaps_on":
                    self.allcaps_on, self.caps_on = True, False
                elif t.name == "allcaps_off":
                    self.allcaps_on = False
                continue
            if isinstance(t, Sym):
                if t.space_before and space_ok and not no_space:
                    out.append(" ")
                out.append(t.text)
                space_ok = t.space_after
                no_space = False
                if t.sentence_end:
                    state = None if t.auto else True
                elif t.text in OPENERS and not t.space_after:
                    pass  # "(" keeps the sentence state for the next word
                else:
                    state = False
                continue
            # Word
            text = t.text if t.raw else self._case(t, state)
            if space_ok and not no_space and text[:1] not in "\n":
                out.append(" ")
            out.append(text)
            space_ok = not text.endswith(("\n", " ", "\t"))
            no_space = False
            state = False
        return "".join(out)

    def snapshot(self) -> tuple[bool, bool]:
        return self.caps_on, self.allcaps_on

    def restore(self, snap: tuple[bool, bool]) -> None:
        self.caps_on, self.allcaps_on = snap
