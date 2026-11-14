# local-dictation

Private, offline dictation for macOS with voice commands. Speech is recognized
locally by NVIDIA Parakeet TDT 0.6B v2 on Apple's MLX; nothing leaves your Mac.

- **Hold Right ⌥** and speak; text is inserted into the focused app when you let go.
- **Tap Right ⌥** for hands-free mode; tap again (or say "stop listening") to end.
  The menu bar icon turns red and a sound plays while it listens.
- **Hold Right ⌘** to speak a command; nothing you say is typed.

## Requirements

- A Mac with Apple Silicon (M1 or newer) and 16 GB of memory recommended
- macOS 14 or newer
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)

## Setup

```bash
git clone https://github.com/aliankarali/local-dictation.git
cd local-dictation
uv sync
uv run dictation download   # ~4.5 GB into ~/Developer/local-dictation/models
uv run dictation check      # shows permissions, microphones, models
uv run dictation mic-test   # speak for 4 seconds, see what it hears
uv run dictation            # start the menu bar app
```

Models are stored in `~/Developer/local-dictation/models`; set
`LOCAL_DICTATION_DATA` to use another folder. After the download the app works
fully offline.

Run it from **Terminal.app**. macOS grants permissions to the app that launches
it, so Terminal needs:

1. **Microphone** (asked on first use)
2. **Accessibility** (System Settings › Privacy & Security › Accessibility)
3. **Input Monitoring** (same place), if the hotkeys don't react

Restart the app after granting.

## Voice commands

**Anywhere in a sentence**

| Say | Get |
|---|---|
| comma, period / full stop, question mark, exclamation mark, colon, semicolon, ellipsis | `, . ? ! : ; ...` |
| open/close paren, open/close bracket, open/close brace, open/close quote | `( ) [ ] { } " "` |
| hyphen, dash, em dash, slash, backslash, underscore, at sign, hash sign, ampersand, percent sign, dollar sign, asterisk, equals sign, plus sign | `- - — / \ _ @ # & % $ * = +` |
| new line, new paragraph | line breaks |
| cap *word*, all caps *word*, no caps *word* | Word, WORD, word |
| caps on / caps off, all caps on / all caps off | title case / upper case until off |
| no space | joins the next word |
| literal *comma* | types the word "comma" |

**On their own** (pause before and after, so "I clicked undo" stays text)

| Say | Does |
|---|---|
| scratch that / delete that | removes the last thing you dictated |
| undo, redo | ⌘Z, ⇧⌘Z |
| delete last word, backspace | ⌥⌫, ⌫ |
| select previous word, select next word, select all | selection |
| select to start of line, select to end of line | selection |
| copy that, cut that, paste that | ⌘C, ⌘X, ⌘V |
| press enter, press tab, press escape | keys |
| go to start/end of line, go to top/bottom, move left/right/up/down | cursor |
| stop listening | ends hands-free mode |

**Natural edits (local LLM, Qwen3 4B):** "put that in quotes", "capitalize
that", "change Friday to Monday", "make that a question", "remove the last
word"... These apply to the selected text, or to what you dictated last. In
command mode (Right ⌘) anything that isn't a built-in command goes to the LLM.

The speech model adds its own punctuation. The period it puts at the end of
every phrase is dropped by default (`[text] final_period = false`); commas,
question marks and a spoken "period" are kept. `auto_punctuation = false` turns
off all model punctuation.

There is no transcript window by default; set `[output] show_transcript = true`
to see what you say live. Warnings (no microphone sound, missing permissions)
always appear.

Add your own key commands, text snippets, vocabulary (names, jargon) and
corrections in `~/.config/local-dictation/config.toml` (menu bar › Edit
config…, then Reload config).

## Benchmark

```bash
uv run dictation bench          # accuracy, commands, latency with synthetic speech
uv run dictation transcribe some-recording.m4a
uv run pytest
```

On an M1 Pro: about 0.1–0.3 s from key release to inserted text, about 0.6–0.9 s
for LLM edits. Memory: roughly 1–2 GB for speech, plus ~2.5 GB with the LLM
(set `[llm] enabled = false` to skip it).

## Where things are

| What | Where |
|---|---|
| Models | `~/Developer/local-dictation/models` |
| Log | `~/Developer/local-dictation/logs/dictation.log` |
| Config | `~/.config/local-dictation/config.toml` |

## Known limits

- Spacing and capitalization use the text before the cursor, read through Accessibility.
  Some apps (Electron, some web editors) don't expose it; then the app relies
  on what it typed itself.
- The microphone stays open 5 minutes after use (`mic_idle_close_s`) so the first
  word isn't clipped. With AirPods this keeps them in the lower-quality headset
  mode during that time; set it to 0 if that bothers you.
- `paste` mode briefly uses the clipboard and restores it afterwards. Use
  `method = "type"` to avoid touching it (slower; "new line" then presses Return,
  which sends the message in chat apps).

## License

MIT, see [LICENSE](LICENSE). The models are downloaded separately and have
their own licenses: Parakeet TDT 0.6B v2 (CC-BY-4.0), Qwen3-4B-Instruct
(Apache 2.0), Silero VAD (MIT).
