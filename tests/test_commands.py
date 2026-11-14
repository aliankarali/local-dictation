import pytest

from local_dictation.commands import CommandParser, apply_corrections, looks_like_edit
from local_dictation.executor import Executor, FakeTarget
from local_dictation.formatter import Formatter, context_state

COMMON = frozenset(
    "the quick brown fox world hello this is a test how are you and then more "
    "make sure that it works great see name value".split()
)


def run(utterances, before="", auto_punctuation=True, readable=True, final_period=True, **kw):
    target = FakeTarget(before, readable=readable)
    parser = CommandParser(auto_punctuation=auto_punctuation, final_period=final_period, **kw)
    ex = Executor(target, Formatter(common_words=COMMON), parser)
    for u in utterances:
        ex.run(parser.parse(u))
    return target


@pytest.mark.parametrize(
    "asr, expected",
    [
        # spoken punctuation, with the model's own punctuation around it
        ("Hello, comma world.", "Hello, world."),
        ("Hello comma world period", "Hello, world."),
        ("hello comma world period", "Hello, world."),
        ("Hello. Comma. World.", "Hello, world."),
        ("How are you question mark", "How are you?"),
        ("Open paren this is a test close paren.", "(This is a test)"),
        ("This is open quote great close quote period", 'This is "great".'),
        ("hello new line world", "Hello\nWorld"),
        ("Hello. New paragraph. This is more.", "Hello\n\nThis is more."),
        ("well hyphen known", "Well-known"),
        ("a dash b", "A - b"),
        ("name at sign example dot com", "Name@example dot com"),
        # formatting
        ("cap hello world", "Hello world"),
        ("all caps hello world", "HELLO world"),
        ("caps on the quick brown fox caps off", "The Quick Brown Fox"),
        ("hello no space world", "Helloworld"),
        # escape
        ("I need a literal comma here", "I need a comma here"),
        # model punctuation is kept when not next to a command
        ("Make sure that it works. See you.", "Make sure that it works. See you."),
    ],
)
def test_inline(asr, expected):
    assert run([asr]).text == expected


def test_final_period_dropped_by_default():
    p = CommandParser()
    assert p.final_period is False
    r = lambda *u: run(list(u), final_period=False).text  # noqa: E731
    assert r("Hello.") == "Hello"
    assert r("I'll be there at five, see you then.") == "I'll be there at five, see you then"
    # sentences inside the phrase keep their periods
    assert r("Thanks. See you tomorrow.") == "Thanks. See you tomorrow"
    # spoken "period", "?" and "!" stay
    assert r("Hello world period") == "Hello world."
    assert r("Hello world, period.") == "Hello world."
    assert r("How are you?") == "How are you?"
    assert r("Great!") == "Great!"
    # the next phrase continues the sentence
    assert r("Hello world.", "How are you?") == "Hello world how are you?"


def test_auto_punctuation_off():
    assert run(["Hello, world. How are you?"], auto_punctuation=False).text == "Hello world how are you"


def test_spacing_and_case_between_utterances():
    t = run(["Hello world.", "How are you?"])
    assert t.text == "Hello world. How are you?"
    # continuing a sentence: model capitalizes "Then", we lower it
    t = run(["I went home", "Then more."])
    assert t.text == "I went home then more."
    # "I" stays capitalized
    t = run(["and", "I think so"])
    assert t.text == "And I think so"


def test_context_from_text_field():
    assert run(["Hello world"], before="Some text. ").text == "Some text. Hello world"
    assert run(["Hello world"], before="Some text").text == "Some text hello world"
    assert run(["Hello world"], before="(").text == "(Hello world"


def test_unknown_context_keeps_model_case():
    assert run(["Hello world"], readable=False).text == "Hello world"


def test_standalone_commands():
    t = run(["Hello world.", "Undo."])
    assert t.keys == ["cmd+z"]
    t = run(["Hello world.", "Scratch that."])
    assert t.text == ""
    t = run(["Hello world.", "How are you?", "scratch that"])
    assert t.text == "Hello world."
    t = run(["hello world", "Select previous word."])
    assert t.keys == ["alt+shift+left"]
    assert t.selected_text() == "world"


def test_fuzzy_standalone():
    t = run(["Hello world.", "Scratch dad."])
    assert t.text == ""
    # different word count or first word: stays text
    assert run(["Select a ball."]).text == "Select a ball."
    assert run(["Snatch that."]).text == "Snatch that."


def test_commands_inside_sentence_are_text():
    t = run(["I clicked undo and it worked."])
    assert t.text == "I clicked undo and it worked."
    assert t.keys == []


def test_custom_key_command_and_snippet():
    t = run(["Save file."], key_commands={"save file": "cmd+s"})
    assert t.keys == ["cmd+s"]
    t = run(["Insert signature."], snippets={"insert signature": "Best,\nMe"})
    assert t.text == "Best,\nMe"


def test_corrections():
    assert apply_corrections("run cube control get pods", {"cube control": "kubectl"}) == "run kubectl get pods"
    assert apply_corrections("Cube Control rocks", {"cube control": "kubectl"}) == "kubectl rocks"


def test_command_mode():
    p = CommandParser()
    assert p.parse("Undo.", command_mode=True)[0].combo == "cmd+z"
    assert p.parse("undue", command_mode=True)[0].kind in ("key", "llm")
    assert p.parse("select all please now", command_mode=True)[0].kind == "llm"
    assert p.parse("new line", command_mode=True)[0].kind == "text"


def test_edit_detection():
    assert looks_like_edit("put that in quotes")
    assert looks_like_edit("capitalize the last sentence")
    assert not looks_like_edit("make dinner tonight")
    p = CommandParser(llm_enabled=True)
    assert p.parse("Put that in quotes.")[0].kind == "llm"
    # without an LLM, falls back to typing
    t = run(["Put that in quotes."], llm_enabled=True)
    assert t.text == "Put that in quotes."


def test_context_state():
    assert context_state("") == (True, False)
    assert context_state("Hi. ") == (True, False)
    assert context_state("Hi") == (False, True)
    assert context_state("Hi.\n") == (True, False)
    assert context_state('He said "') == (False, False)
    assert context_state('He said "yes."') == (True, True)
