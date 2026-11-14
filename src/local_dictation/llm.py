"""Local language model that interprets free-form edit commands.

Only used for phrases that look like edits ("put that in quotes") or for
command-mode phrases that match no built-in command."""

from __future__ import annotations

import json
import logging
import re
import time

log = logging.getLogger(__name__)

SYSTEM = """You are the command interpreter inside a voice dictation app.
The user spoke an INSTRUCTION. TARGET is the text the instruction refers to
(the current selection or the phrase the user dictated last); it may be empty.

Reply with exactly one line of JSON and nothing else:
{"action": "edit", "text": "<full new version of TARGET>"}  - the instruction changes TARGET (quote it, capitalize it, fix it, replace a word, rephrase, translate...). Apply only the requested change; keep everything else in TARGET identical.
{"action": "key", "command": "<name>"}  - the instruction means one of these editor commands: COMMANDS
{"action": "dictate"}  - the instruction is ordinary text the user wants typed, not a command.
{"action": "none"}  - it is a command but cannot be done.

Choose "dictate" whenever the INSTRUCTION could be a sentence meant to be written
down. Choose "edit" only when it explicitly talks about existing text ("that",
"it", "this", "the last word", "the selection", "the sentence")."""

EXAMPLES = [
    ("put that in quotes", "the meeting is at noon", {"action": "edit", "text": "\"the meeting is at noon\""}),
    ("make that a question", "You are coming tomorrow.", {"action": "edit", "text": "Are you coming tomorrow?"}),
    ("replace noon with three pm", "The meeting is at noon.", {"action": "edit", "text": "The meeting is at 3 pm."}),
    ("get rid of everything I just said", "", {"action": "key", "command": "scratch that"}),
    ("make sure that the door is locked", "Hi Sam,", {"action": "dictate"}),
    ("change the tires before winter", "Reminder:", {"action": "dictate"}),
    ("capitalize that", "berlin", {"action": "edit", "text": "Berlin"}),
    ("remove the last word", "See you on Monday then.", {"action": "edit", "text": "See you on Monday."}),
]


class EditLLM:
    def __init__(self, model_id: str):
        from mlx_lm import load

        t = time.monotonic()
        self.model, self.tokenizer = load(model_id)
        self.ready = True
        self._cache = None
        self._cached: list[int] = []
        log.info("loaded %s in %.1fs", model_id, time.monotonic() - t)

    def _messages(self, instruction: str, target: str | None, commands: list[str]):
        system = SYSTEM.replace("COMMANDS", ", ".join(commands))
        msgs = [{"role": "system", "content": system}]
        for instr, tgt, ans in EXAMPLES:
            msgs.append({"role": "user", "content": f"TARGET: {json.dumps(tgt)}\nINSTRUCTION: {json.dumps(instr)}"})
            msgs.append({"role": "assistant", "content": json.dumps(ans)})
        msgs.append(
            {"role": "user", "content": f"TARGET: {json.dumps(target or '')}\nINSTRUCTION: {json.dumps(instruction)}"}
        )
        return msgs

    def interpret(self, instruction: str, target: str | None, commands: list[str]) -> dict:
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        tokens = list(
            self.tokenizer.apply_chat_template(
                self._messages(instruction, target, commands), add_generation_prompt=True
            )
        )
        # Reuse the KV cache for the shared prefix (instructions + examples):
        # only the new TARGET/INSTRUCTION tokens need to be processed.
        if self._cache is None:
            self._cache = make_prompt_cache(self.model)
            self._cached = []
        same = 0
        for a, b in zip(self._cached, tokens):
            if a != b:
                break
            same += 1
        same = min(same, len(tokens) - 1)
        trim_prompt_cache(self._cache, self._cache[0].offset - same)

        max_tokens = 64 + int(len(target or "") * 0.6)
        t = time.monotonic()
        text, generated = "", []
        for r in stream_generate(
            self.model, self.tokenizer, tokens[same:], max_tokens=max_tokens, prompt_cache=self._cache
        ):
            text += r.text
            generated.append(r.token)
            if _json_complete(text):
                break
        self._cached = (tokens + generated)[: self._cache[0].offset]
        log.info("llm %.0fms (%d new prompt tokens): %s", (time.monotonic() - t) * 1000,
                 len(tokens) - same, text.strip()[:200])
        return parse_reply(text)

    def warmup(self, commands: list[str]) -> None:
        """Fills the prompt cache with the shared prefix."""
        self.interpret("undo that", "", commands)


def _json_complete(text: str) -> bool:
    start = text.find("{")
    if start < 0:
        return False
    depth, in_str, esc = 0, False, False
    for c in text[start:]:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def parse_reply(out: str) -> dict:
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return {"action": "none"}
    try:
        res = json.loads(m.group(0))
        return res if isinstance(res, dict) else {"action": "none"}
    except json.JSONDecodeError:
        return {"action": "none"}
