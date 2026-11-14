"""macOS integration: insert text / press keys in the focused app, read the
text before the cursor (Accessibility), global hotkeys (event tap)."""

from __future__ import annotations

import ctypes
import logging
import re
import threading
import time
from ctypes import POINTER, byref, c_int32, c_uint8, c_uint16, c_uint32, c_ulong, c_void_p

import Quartz
from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString, NSWorkspace
from ApplicationServices import (
    AXIsProcessTrusted,
    AXIsProcessTrustedWithOptions,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateSystemWide,
    AXUIElementSetMessagingTimeout,
    AXValueGetValue,
    kAXFocusedUIElementAttribute,
    kAXSelectedTextAttribute,
    kAXSelectedTextRangeAttribute,
    kAXTrustedCheckOptionPrompt,
    kAXValueAttribute,
    kAXValueCFRangeType,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Keyboard layout: character -> virtual key code (so "cmd+z" works on QWERTZ)
# --------------------------------------------------------------------------

SPECIAL_KEYS = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "backspace": 51,
    "delete": 51, "escape": 53, "esc": 53, "forward_delete": 117,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
}
ARROWS = {123, 124, 125, 126}
ANSI = dict(zip("asdfhgzxcv", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
ANSI.update(dict(zip("bqweryt", [11, 12, 13, 14, 15, 16, 17])))
ANSI.update({"o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46})
MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "command": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "opt": Quartz.kCGEventFlagMaskAlternate,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "control": Quartz.kCGEventFlagMaskControl,
    "fn": Quartz.kCGEventFlagMaskSecondaryFn,
}


def layout_char_map() -> dict[str, int]:
    """Map characters to key codes for the current keyboard layout.
    Must run on the main thread (Text Input Sources API)."""
    try:
        carbon = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Carbon.framework/Carbon")
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        carbon.TISCopyCurrentKeyboardLayoutInputSource.restype = c_void_p
        carbon.TISGetInputSourceProperty.restype = c_void_p
        carbon.TISGetInputSourceProperty.argtypes = [c_void_p, c_void_p]
        carbon.LMGetKbdType.restype = c_uint8
        carbon.UCKeyTranslate.restype = c_int32
        carbon.UCKeyTranslate.argtypes = [
            c_void_p, c_uint16, c_uint16, c_uint32, c_uint32, c_uint32,
            POINTER(c_uint32), c_ulong, POINTER(c_ulong), POINTER(c_uint16),
        ]
        cf.CFDataGetBytePtr.restype = c_void_p
        cf.CFDataGetBytePtr.argtypes = [c_void_p]
        cf.CFRelease.argtypes = [c_void_p]
        prop = c_void_p.in_dll(carbon, "kTISPropertyUnicodeKeyLayoutData")
        src = carbon.TISCopyCurrentKeyboardLayoutInputSource()
        if not src:
            return {}
        try:
            data = carbon.TISGetInputSourceProperty(src, prop)
            if not data:
                return {}
            layout = cf.CFDataGetBytePtr(data)
            kbd = carbon.LMGetKbdType()
            result: dict[str, int] = {}
            for code in range(128):
                dead, length, buf = c_uint32(0), c_ulong(0), (c_uint16 * 4)()
                err = carbon.UCKeyTranslate(layout, code, 3, 0, kbd, 1, byref(dead), 4, byref(length), buf)
                if err == 0 and length.value:
                    ch = "".join(chr(buf[i]) for i in range(length.value)).lower()
                    if ch.isprintable() and ch.strip():
                        result.setdefault(ch, code)
            return result
        finally:
            cf.CFRelease(src)
    except Exception:
        log.exception("could not read keyboard layout, assuming US")
        return {}


# --------------------------------------------------------------------------
# Target: the focused application
# --------------------------------------------------------------------------


class MacTarget:
    def __init__(self, method: str = "paste", restore_clipboard: bool = True):
        self.method = method
        self.restore_clipboard = restore_clipboard
        self.charmap = layout_char_map()
        self.source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
        self.source_state_id = Quartz.CGEventSourceGetSourceStateID(self.source)
        self._clip_lock = threading.Lock()
        self._saved: list[dict] | None = None
        self._our_change = -1
        self._restore_at = 0.0
        self._restore_thread: threading.Thread | None = None

    # -- keys -------------------------------------------------------------

    def _keycode(self, key: str) -> int:
        if key in SPECIAL_KEYS:
            return SPECIAL_KEYS[key]
        if key in self.charmap:
            return self.charmap[key]
        if key in ANSI:
            return ANSI[key]
        raise ValueError(f"unknown key: {key}")

    def press(self, combo: str) -> None:
        parts = [p.strip() for p in combo.lower().split("+") if p.strip()]
        key = parts[-1]
        flags = 0
        for m in parts[:-1]:
            flags |= MODIFIERS[m]
        code = self._keycode(key)
        if code in ARROWS:
            flags |= Quartz.kCGEventFlagMaskSecondaryFn | Quartz.kCGEventFlagMaskNumericPad
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(self.source, code, down)
            Quartz.CGEventSetFlags(ev, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            time.sleep(0.004)

    def backspace(self, n: int) -> None:
        for i in range(n):
            self.press("backspace")
            if i % 20 == 19:
                time.sleep(0.01)

    # -- text -------------------------------------------------------------

    def insert_text(self, text: str) -> None:
        if self.method == "type":
            self._type(text)
        else:
            self._paste(text)

    def _type(self, text: str) -> None:
        for part in re.split(r"(\n)", text):
            if part == "\n":
                self.press("return")
                continue
            for i in range(0, len(part), 16):
                chunk = part[i : i + 16]
                n = len(chunk.encode("utf-16-le")) // 2
                for down in (True, False):
                    ev = Quartz.CGEventCreateKeyboardEvent(self.source, 0, down)
                    Quartz.CGEventKeyboardSetUnicodeString(ev, n, chunk)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
                time.sleep(0.004)

    @staticmethod
    def _snapshot(pb) -> list[dict]:
        items = []
        for item in pb.pasteboardItems() or []:
            d = {}
            for t in item.types():
                data = item.dataForType_(t)
                if data is not None:
                    d[t] = data
            items.append(d)
        return items

    def _paste(self, text: str) -> None:
        pb = NSPasteboard.generalPasteboard()
        with self._clip_lock:
            if self.restore_clipboard and self._saved is None:
                self._saved = self._snapshot(pb)
            pb.clearContents()
            # Mark as transient so clipboard managers don't record it.
            pb.declareTypes_owner_(
                [NSPasteboardTypeString, "org.nspasteboard.TransientType",
                 "org.nspasteboard.AutoGeneratedType"],
                None,
            )
            pb.setString_forType_(text, NSPasteboardTypeString)
            pb.setString_forType_("", "org.nspasteboard.TransientType")
            pb.setString_forType_("", "org.nspasteboard.AutoGeneratedType")
            self._our_change = pb.changeCount()
            self.press("cmd+v")
            self._restore_at = time.monotonic() + 0.5
        if self.restore_clipboard and not (self._restore_thread and self._restore_thread.is_alive()):
            self._restore_thread = threading.Thread(target=self._restore_loop, daemon=True)
            self._restore_thread.start()

    def _restore_loop(self) -> None:
        while True:
            with self._clip_lock:
                wait = self._restore_at - time.monotonic()
                if wait <= 0:
                    pb = NSPasteboard.generalPasteboard()
                    if self._saved is not None and pb.changeCount() == self._our_change:
                        pb.clearContents()
                        objs = []
                        for d in self._saved:
                            it = NSPasteboardItem.alloc().init()
                            for t, data in d.items():
                                it.setData_forType_(data, t)
                            objs.append(it)
                        if objs:
                            pb.writeObjects_(objs)
                    self._saved = None
                    return
            time.sleep(min(wait, 0.1))

    # -- accessibility ----------------------------------------------------

    @staticmethod
    def _focused():
        system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, 0.3)
        err, el = AXUIElementCopyAttributeValue(system, kAXFocusedUIElementAttribute, None)
        return el if err == 0 else None

    def context_before(self) -> str | None:
        el = self._focused()
        if el is None:
            return None
        err, value = AXUIElementCopyAttributeValue(el, kAXValueAttribute, None)
        if err != 0 or not isinstance(value, str):
            return None
        err, rng = AXUIElementCopyAttributeValue(el, kAXSelectedTextRangeAttribute, None)
        if err != 0 or rng is None:
            return None
        ok, r = AXValueGetValue(rng, kAXValueCFRangeType, None)
        if not ok:
            return None
        loc = r.location if hasattr(r, "location") else r[0]
        return value[max(0, loc - 300) : loc]

    def selected_text(self) -> str | None:
        el = self._focused()
        if el is None:
            return None
        err, value = AXUIElementCopyAttributeValue(el, kAXSelectedTextAttribute, None)
        return value if err == 0 and isinstance(value, str) and value else None

    def frontmost_pid(self) -> int:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app else -1


def accessibility_trusted(prompt: bool = False) -> bool:
    if prompt:
        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
    return bool(AXIsProcessTrusted())


# --------------------------------------------------------------------------
# Global hotkeys: modifier keys pressed on their own
# --------------------------------------------------------------------------

HOTKEYS = {
    # name: (key code, device-dependent flag bit)
    "right_option": (61, 0x40),
    "left_option": (58, 0x20),
    "right_command": (54, 0x10),
    "left_command": (55, 0x08),
    "right_shift": (60, 0x04),
    "left_shift": (56, 0x02),
    "right_control": (62, 0x2000),
    "left_control": (59, 0x01),
    "fn": (63, Quartz.kCGEventFlagMaskSecondaryFn),
}


class _KeyState:
    def __init__(self):
        self.down = False
        self.t = 0.0
        self.interrupted = False


class HotkeyListener:
    """Listen-only event tap. Calls handler.on_press/on_release/on_tap/on_cancel
    with role "dictate" or "command". Runs on the thread whose run loop it is
    added to (the main thread)."""

    def __init__(self, keys: dict[str, str], tap_max_ms: int, handler):
        unknown = [k for k in keys.values() if k not in HOTKEYS]
        if unknown:
            raise ValueError(f"unknown hotkey(s): {unknown}; options: {sorted(HOTKEYS)}")
        self.roles = {HOTKEYS[name][0]: (role, HOTKEYS[name][1]) for role, name in keys.items()}
        self.tap_max = tap_max_ms / 1000
        self.handler = handler
        self.state = {role: _KeyState() for role in keys}
        self.tap = None
        # Key events we synthesize ourselves (pasting, commands) must not
        # count as "another key pressed while holding the hotkey".
        self.ignore_source_states: set[int] = set()

    def start(self) -> bool:
        mask = (1 << Quartz.kCGEventFlagsChanged) | (1 << Quartz.kCGEventKeyDown)
        self.tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._callback,
            None,
        )
        if self.tap is None:
            return False
        src = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self.tap, True)
        return True

    def _callback(self, proxy, type_, event, refcon):
        try:
            if type_ in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(self.tap, True)
                return event
            if type_ == Quartz.kCGEventKeyDown:
                src = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceStateID)
                if src in self.ignore_source_states:
                    return event
                # Another key while a hotkey is held: it's a shortcut (e.g. Option+E).
                for st in self.state.values():
                    if st.down:
                        st.interrupted = True
                return event
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            if code not in self.roles:
                # Another modifier pressed while holding ours: treat as shortcut.
                for st in self.state.values():
                    if st.down:
                        st.interrupted = True
                return event
            role, bit = self.roles[code]
            down = bool(Quartz.CGEventGetFlags(event) & bit)
            self._edge(role, down)
        except Exception:
            log.exception("hotkey callback failed")
        return event

    def _edge(self, role: str, down: bool) -> None:
        st = self.state[role]
        now = time.monotonic()
        if down and not st.down:
            st.down, st.t, st.interrupted = True, now, False
            self.handler.on_press(role)
        elif not down and st.down:
            st.down = False
            if st.interrupted:
                self.handler.on_cancel(role)
            elif now - st.t < self.tap_max:
                self.handler.on_tap(role)
            else:
                self.handler.on_release(role)
