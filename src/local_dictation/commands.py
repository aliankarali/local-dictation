"""Turn one transcribed utterance into actions.

Two kinds of commands:

* Inline commands - spoken punctuation ("comma", "open paren"), layout
  ("new line") and formatting ("cap", "all caps", "no space"). They can appear
  anywhere in a sentence.
* Standalone commands - editing ("undo", "scratch that", "select previous
  word", ...). They only fire when the whole utterance is the command, i.e.
  you said it on its own with a pause before and after. This is what prevents
  "I clicked undo" from undoing something.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Tokens and actions
# --------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    force: str | None = None  # "cap" | "upper" | "lower"
    first_in_utterance: bool = False
    after_dropped_end: bool = False  # a model-added sentence end before it was removed
    raw: bool = False  # insert exactly as is (snippets, literal)


@dataclass
class Sym:
    text: str
    space_before: bool = False
    space_after: bool = True
    sentence_end: bool = False
    auto: bool = False  # added by the speech model, not spoken

    @property
    def droppable(self) -> bool:
        return self.auto and self.text in DROPPABLE


@dataclass
class Directive:
    name: str  # no_space | caps_on | caps_off | allcaps_on | allcaps_off


Token = Word | Sym | Directive


@dataclass
class Action:
    kind: str  # text | key | scratch | stop | llm
    tokens: list = field(default_factory=list)
    combo: str = ""
    text: str = ""
    label: str = ""
    command_mode: bool = False


DROPPABLE = {".", ",", "?", "!", ";", ":", "..."}

# --------------------------------------------------------------------------
# Phrase tables
# --------------------------------------------------------------------------

# phrase -> (text, space_before, space_after, sentence_end)
_SYMBOLS: dict[tuple[str, ...], tuple[str, bool, bool, bool]] = {}


def _add_sym(names: list[str], text: str, before: bool, after: bool, end: bool = False):
    for n in names:
        _SYMBOLS[tuple(n.split())] = (text, before, after, end)


_add_sym(["comma", "coma"], ",", False, True)
_add_sym(["period", "full stop", "fullstop"], ".", False, True, True)
_add_sym(["question mark"], "?", False, True, True)
_add_sym(["exclamation mark", "exclamation point"], "!", False, True, True)
_add_sym(["colon"], ":", False, True)
_add_sym(["semicolon", "semi colon"], ";", False, True)
_add_sym(["ellipsis", "dot dot dot"], "...", False, True)
_add_sym(
    ["open paren", "open parenthesis", "open parentheses", "left paren",
     "left parenthesis", "open round bracket"],
    "(", True, False,
)
_add_sym(
    ["close paren", "close parenthesis", "close parentheses", "right paren",
     "right parenthesis", "close round bracket"],
    ")", False, True,
)
_add_sym(["open bracket", "open square bracket", "left bracket"], "[", True, False)
_add_sym(["close bracket", "close square bracket", "right bracket"], "]", False, True)
_add_sym(["open brace", "open curly brace", "open curly bracket", "left brace"], "{", True, False)
_add_sym(["close brace", "close curly brace", "close curly bracket", "right brace"], "}", False, True)
_add_sym(["open quote", "begin quote", "start quote", "open quotes"], '"', True, False)
_add_sym(["close quote", "end quote", "unquote", "close quotes", "end quotes"], '"', False, True)
_add_sym(["open single quote"], "'", True, False)
_add_sym(["close single quote"], "'", False, True)
_add_sym(["apostrophe"], "'", False, False)
_add_sym(["hyphen"], "-", False, False)
_add_sym(["dash"], "-", True, True)
_add_sym(["em dash"], "—", False, False)
_add_sym(["slash", "forward slash"], "/", False, False)
_add_sym(["backslash", "back slash"], "\\", False, False)
_add_sym(["underscore"], "_", False, False)
_add_sym(["at sign"], "@", False, False)
_add_sym(["hash sign", "hashtag", "pound sign"], "#", True, False)
_add_sym(["ampersand"], "&", True, True)
_add_sym(["percent sign"], "%", False, True)
_add_sym(["dollar sign"], "$", True, False)
_add_sym(["asterisk"], "*", False, False)
_add_sym(["equals sign", "equal sign"], "=", True, True)
_add_sym(["plus sign"], "+", True, True)
_add_sym(["new line", "newline", "next line", "line break"], "\n", False, False, True)
_add_sym(["new paragraph", "next paragraph"], "\n\n", False, False, True)

_DIRECTIVES = {
    ("no", "space"): "no_space",
    ("caps", "on"): "caps_on",
    ("caps", "off"): "caps_off",
    ("all", "caps", "on"): "allcaps_on",
    ("all", "caps", "off"): "allcaps_off",
}
_FORCES = {
    ("cap",): "cap",
    ("all", "caps"): "upper",
    ("no", "caps"): "lower",
}
_LITERAL = ("literal",)
_MAX_PHRASE = 4

# Standalone commands: normalized utterance -> key combo
DEFAULT_KEY_COMMANDS = {
    "undo": "cmd+z",
    "undo that": "cmd+z",
    "redo": "cmd+shift+z",
    "redo that": "cmd+shift+z",
    "delete last word": "alt+backspace",
    "delete previous word": "alt+backspace",
    "delete the last word": "alt+backspace",
    "backspace": "backspace",
    "select previous word": "alt+shift+left",
    "select last word": "alt+shift+left",
    "select next word": "alt+shift+right",
    "select to start of line": "cmd+shift+left",
    "select to end of line": "cmd+shift+right",
    "select all": "cmd+a",
    "copy that": "cmd+c",
    "cut that": "cmd+x",
    "paste that": "cmd+v",
    "press enter": "return",
    "press return": "return",
    "hit enter": "return",
    "press tab": "tab",
    "press escape": "escape",
    "go to end of line": "cmd+right",
    "go to start of line": "cmd+left",
    "go to beginning of line": "cmd+left",
    "go to top": "cmd+up",
    "go to bottom": "cmd+down",
    "move left": "left",
    "move right": "right",
    "move up": "up",
    "move down": "down",
}
SCRATCH_COMMANDS = {"scratch that", "delete that", "strike that", "scratch"}
STOP_COMMANDS = {"stop listening", "stop dictation", "go to sleep"}

EDIT_VERBS = {
    "put", "make", "change", "replace", "capitalize", "uppercase", "lowercase",
    "rewrite", "rephrase", "fix", "correct", "wrap", "turn", "remove", "format",
    "translate", "shorten", "bold", "italicize", "quote", "convert", "swap",
    "spell", "delete",
}
EDIT_REFERENCES = {
    "that", "this", "it", "last", "previous", "selection", "selected",
    "sentence", "word", "paragraph", "everything",
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9']")


def norm_word(w: str) -> str:
    return _NORM_RE.sub("", w.lower().replace("’", "'"))


def norm_phrase(text: str) -> str:
    return " ".join(n for n in (norm_word(w) for w in text.replace("-", " ").split()) if n)


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    for wrong in sorted(corrections, key=len, reverse=True):
        right = corrections[wrong]
        text = re.sub(
            r"(?<![\w'])" + re.escape(wrong) + r"(?![\w'])",
            lambda _m, r=right: r,
            text,
            flags=re.IGNORECASE,
        )
    return text


_LEAD = {"(": "(", "[": "[", "{": "{", '"': '"', "“": '"'}
_TRAIL = {".", ",", "?", "!", ":", ";", ")", "]", "}", '"', "”", "…"}
_DASHES = {"-", "–", "—"}


def _model_sym(c: str, leading: bool) -> Sym:
    if leading:
        return Sym(_LEAD[c], space_before=True, space_after=False, auto=True)
    if c in ".?!":
        return Sym(c, False, True, True, auto=True)
    if c == "…":
        return Sym("...", False, True, False, auto=True)
    if c in "\"”":
        return Sym('"', False, True, auto=True)
    return Sym(c, False, True, auto=True)


def tokenize(text: str) -> list[Token]:
    """Split model output into words and the punctuation the model added."""
    tokens: list[Token] = []
    for chunk in text.split():
        if chunk in _DASHES:
            tokens.append(Sym(chunk, True, True, auto=True))
            continue
        lead = []
        while chunk and chunk[0] in _LEAD:
            lead.append(chunk[0])
            chunk = chunk[1:]
        trail = []
        while chunk and chunk[-1] in _TRAIL:
            trail.insert(0, chunk[-1])
            chunk = chunk[:-1]
        tokens.extend(_model_sym(c, True) for c in lead)
        if chunk:
            tokens.append(Word(chunk))
        trail_s = "".join(trail)
        trail_s = trail_s.replace("...", "…")
        tokens.extend(_model_sym(c, False) for c in trail_s)
    return tokens


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


class CommandParser:
    def __init__(
        self,
        key_commands: dict[str, str] | None = None,
        snippets: dict[str, str] | None = None,
        auto_punctuation: bool = True,
        final_period: bool = False,
        llm_enabled: bool = False,
    ):
        self.key_commands = dict(DEFAULT_KEY_COMMANDS)
        for k, v in (key_commands or {}).items():
            self.key_commands[norm_phrase(k)] = v
        self.snippets = {norm_phrase(k): v for k, v in (snippets or {}).items()}
        self.auto_punctuation = auto_punctuation
        self.final_period = final_period
        self.llm_enabled = llm_enabled

    # -- public -----------------------------------------------------------

    def parse(self, text: str, *, command_mode: bool = False) -> list[Action]:
        phrase = norm_phrase(text)
        if not phrase:
            return []
        standalone = self._standalone(phrase) or self._fuzzy_standalone(phrase)
        if standalone:
            return standalone
        if command_mode:
            return self._command_mode(phrase, text)
        if self.llm_enabled and looks_like_edit(phrase):
            return [Action("llm", text=text, label="edit")]
        tokens = self.to_tokens(text)
        return [Action("text", tokens=tokens)] if tokens else []

    def command_names(self) -> list[str]:
        return sorted(set(self.key_commands) | SCRATCH_COMMANDS | STOP_COMMANDS)

    # -- standalone / command mode ---------------------------------------

    def _standalone(self, phrase: str) -> list[Action]:
        if phrase in self.key_commands:
            return [Action("key", combo=self.key_commands[phrase], label=phrase)]
        if phrase in SCRATCH_COMMANDS:
            return [Action("scratch", label=phrase)]
        if phrase in STOP_COMMANDS:
            return [Action("stop", label=phrase)]
        if phrase in self.snippets:
            return [Action("text", tokens=[Word(self.snippets[phrase], raw=True)], label=phrase)]
        return []

    def _fuzzy_standalone(self, phrase: str) -> list[Action]:
        """Tolerate one misheard word in multi-word commands ("scratch dad").
        Same word count, first word exact, very similar overall."""
        words = phrase.split()
        if len(words) < 2:
            return []
        candidates = [
            c for c in self.command_names() + list(self.snippets)
            if len(c.split()) == len(words) and c.split()[0] == words[0]
        ]
        match = difflib.get_close_matches(phrase, candidates, n=1, cutoff=0.75)
        return self._standalone(match[0]) if match else []

    def _command_mode(self, phrase: str, text: str) -> list[Action]:
        # Inline-only commands said in command mode ("comma", "new line", "caps on")
        words = tuple(phrase.split())
        if words in _SYMBOLS or words in _DIRECTIVES:
            return [Action("text", tokens=self.to_tokens(text), label=phrase)]
        names = self.command_names() + list(self.snippets)
        match = difflib.get_close_matches(phrase, names, n=1, cutoff=0.8)
        if match:
            return self._standalone(match[0])
        return [Action("llm", text=text, label="command", command_mode=True)]

    # -- inline -----------------------------------------------------------

    def _match(self, raw: list[Token], i: int):
        """Longest phrase starting at raw[i]. Returns (kind, value, end_index)."""
        words: list[tuple[str, int]] = []
        j = i
        while j < len(raw) and len(words) < _MAX_PHRASE:
            t = raw[j]
            if isinstance(t, Word):
                n = norm_word(t.text)
                if not n:
                    break
                words.append((n, j))
            elif not (isinstance(t, Sym) and t.droppable):
                break
            j += 1
        for n in range(len(words), 0, -1):
            key = tuple(w for w, _ in words[:n])
            end = words[n - 1][1] + 1
            if key in _SYMBOLS:
                return "sym", _SYMBOLS[key], end
            if key in _DIRECTIVES:
                return "directive", _DIRECTIVES[key], end
            if key in _FORCES and end < len(raw):
                return "force", _FORCES[key], end
            if key == _LITERAL and end < len(raw):
                return "literal", None, end
        return None

    def to_tokens(self, text: str) -> list[Token]:
        raw = tokenize(text)
        out: list[Token] = []
        force: str | None = None
        dropped_end = False
        first_word = True
        i = 0

        def skip_auto_after(k: int) -> int:
            nonlocal dropped_end
            while k < len(raw) and isinstance(raw[k], Sym) and raw[k].droppable:
                dropped_end = dropped_end or raw[k].sentence_end
                k += 1
            return k

        while i < len(raw):
            t = raw[i]
            if isinstance(t, Word):
                m = self._match(raw, i)
                if m:
                    kind, value, end = m
                    if kind == "literal":
                        m2 = self._match(raw, end)
                        if m2:
                            for w in raw[end:m2[2]]:
                                if isinstance(w, Word):
                                    out.append(Word(w.text.lower(), first_in_utterance=first_word))
                                    first_word = False
                            i = m2[2]
                            continue
                        i = end  # "literal" + ordinary word: drop "literal"
                        continue
                    if kind == "sym":
                        # The model often adds its own punctuation around spoken
                        # punctuation ("Hello, comma world.") - drop it.
                        while out and isinstance(out[-1], Sym) and out[-1].droppable:
                            out.pop()
                        out.append(Sym(*value))
                    elif kind == "directive":
                        out.append(Directive(value))
                    elif kind == "force":
                        force = value
                    i = skip_auto_after(end)
                    continue
                out.append(
                    Word(
                        t.text,
                        force=force,
                        first_in_utterance=first_word,
                        after_dropped_end=dropped_end,
                    )
                )
                force = None
                dropped_end = False
                first_word = False
            elif isinstance(t, Sym):
                if not self.auto_punctuation and t.droppable:
                    dropped_end = dropped_end or t.sentence_end
                else:
                    out.append(t)
            i += 1
        if not self.final_period:
            # The model ends nearly every phrase with "." - drop that one
            # (spoken "period" is not auto, so it stays; so do "?" and "!").
            while out and isinstance(out[-1], Sym) and out[-1].auto and out[-1].text in (".", "..."):
                out.pop()
        return out


# Spoken command words the speech model should prefer over look-alikes
# ("comma" not "common", "scratch that" not "scratch dad", "undo" not "on do").
BOOST_PHRASES = [
    "comma", "period", "full stop", "question mark", "exclamation mark",
    "exclamation point", "colon", "semicolon", "new line", "new paragraph",
    "open paren", "close paren", "open quote", "close quote", "open bracket",
    "close bracket", "hyphen", "undo", "redo", "scratch that", "delete that",
    "select all", "select previous word", "delete last word", "caps on",
    "caps off", "all caps", "no caps", "no space", "press enter",
    "stop listening", "literal",
]
STRONG_BOOST_PHRASES = ["cap"]  # short, easily heard as "cat"/"cab"


def command_boosts(weight: float, extra_phrases=()) -> dict[str, float]:
    if weight <= 0:
        return {}
    out = {p: weight for p in BOOST_PHRASES}
    out.update({p: weight * 1.5 for p in STRONG_BOOST_PHRASES})
    out.update({p: weight for p in extra_phrases})
    return out


def looks_like_edit(phrase: str) -> bool:
    words = phrase.split()
    return (
        2 <= len(words) <= 14
        and words[0] in EDIT_VERBS
        and any(w in EDIT_REFERENCES for w in words[1:])
    )
