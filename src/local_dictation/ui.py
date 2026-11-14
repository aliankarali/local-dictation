"""Menu bar icon, floating overlay and app wiring (AppKit via PyObjC)."""

from __future__ import annotations

import logging
import subprocess

import objc
from AppKit import (
    NSApp,
    NSAppearance,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSFont,
    NSImage,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSPasteboard,
    NSPasteboardTypeString,
    NSScreen,
    NSSound,
    NSStatusBar,
    NSStatusWindowLevel,
    NSTextField,
    NSVariableStatusItemLength,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject, NSPointInRect
from PyObjCTools import AppHelper

from .config import CONFIG_PATH, LOG_DIR, ensure_user_config, load_config
from .controller import Controller
from .macos import HotkeyListener, MacTarget, accessibility_trusted

log = logging.getLogger(__name__)

KEY_NAMES = {
    "right_option": "Right ⌥", "left_option": "Left ⌥",
    "right_command": "Right ⌘", "left_command": "Left ⌘",
    "right_shift": "Right ⇧", "left_shift": "Left ⇧",
    "right_control": "Right ⌃", "left_control": "Left ⌃", "fn": "Fn",
}
ICONS = {
    "loading": "hourglass", "idle": "mic", "ptt": "mic.fill",
    "handsfree": "waveform", "command": "command", "error": "exclamationmark.triangle",
}


def _label(rect, size: float, bold: bool = False, alpha: float = 1.0):
    f = NSTextField.labelWithString_("")
    f.setFrame_(rect)
    f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    f.setTextColor_(NSColor.whiteColor().colorWithAlphaComponent_(alpha))
    f.setMaximumNumberOfLines_(2)
    f.cell().setWraps_(True)
    return f


class Overlay:
    W, H = 600, 78

    def __init__(self):
        rect = NSMakeRect(0, 0, self.W, self.H)
        p = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False,
        )
        p.setLevel_(NSStatusWindowLevel)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setHasShadow_(True)
        p.setIgnoresMouseEvents_(True)
        p.setHidesOnDeactivate_(False)
        p.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        p.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameVibrantDark"))
        v = NSVisualEffectView.alloc().initWithFrame_(rect)
        v.setMaterial_(13)  # HUD window
        v.setBlendingMode_(0)  # behind window
        v.setState_(1)  # active
        v.setWantsLayer_(True)
        v.layer().setCornerRadius_(16)
        v.layer().setMasksToBounds_(True)
        p.setContentView_(v)
        self.title = _label(NSMakeRect(18, self.H - 26, self.W - 36, 16), 10.5, bold=True, alpha=0.6)
        self.text = _label(NSMakeRect(18, 8, self.W - 36, 44), 15)
        v.addSubview_(self.title)
        v.addSubview_(self.text)
        self.panel = p
        self.gen = 0
        self.kind = None

    def show(self, title: str, text: str, kind: str = "message") -> None:
        self.gen += 1
        self.kind = kind
        self.title.setStringValue_(title.upper())
        text = text.replace("\n", " ↵ ")
        if len(text) > 150:
            text = "…" + text[-150:]
        self.text.setStringValue_(text or "…")
        mouse = NSEvent.mouseLocation()
        screen = next((s for s in NSScreen.screens() if NSPointInRect(mouse, s.frame())), NSScreen.mainScreen())
        vf = screen.visibleFrame()
        self.panel.setFrameOrigin_((vf.origin.x + (vf.size.width - self.W) / 2, vf.origin.y + 80))
        self.panel.orderFrontRegardless()

    def hide(self, delay: float = 0.0, kind: str | None = None) -> None:
        if kind is not None and kind != self.kind:
            return
        gen = self.gen
        if delay <= 0:
            self.panel.orderOut_(None)
        else:
            AppHelper.callLater(delay, lambda: gen == self.gen and self.panel.orderOut_(None))


class MenuTarget(NSObject):
    """Receives menu actions (needs to be an NSObject)."""

    def toggleHandsFree_(self, sender):
        self.app.controller.toggle_handsfree()

    def editConfig_(self, sender):
        ensure_user_config()
        subprocess.Popen(["open", "-t", str(CONFIG_PATH)])

    def reloadConfig_(self, sender):
        self.app.reload_config()

    def openLog_(self, sender):
        subprocess.Popen(["open", str(LOG_DIR / "dictation.log")])

    def permissions_(self, sender):
        accessibility_trusted(prompt=True)
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])

    def micPermissions_(self, sender):
        subprocess.Popen(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"])

    def copyRecent_(self, sender):
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(sender.representedObject(), NSPasteboardTypeString)

    def quit_(self, sender):
        NSApp.terminate_(None)

    def menuNeedsUpdate_(self, menu):
        self.app.refresh_menu()


class UIBridge:
    """Thread-safe UI calls for the controller (everything hops to the main thread)."""

    def __init__(self, app: "MenuBarApp"):
        self.app = app

    def set_state(self, state: str, detail: str = "") -> None:
        AppHelper.callAfter(self.app.set_state, state, detail)

    def show_overlay(self, title: str, text: str) -> None:
        AppHelper.callAfter(self.app.overlay.show, title, text)

    def hide_overlay(self, delay: float = 0.0) -> None:
        AppHelper.callAfter(self.app.overlay.hide, delay)

    def show_transcript(self, title: str, text: str) -> None:
        if self.app.cfg.show_transcript:
            AppHelper.callAfter(self.app.overlay.show, title, text, "transcript")

    def hide_transcript(self, delay: float = 0.0) -> None:
        AppHelper.callAfter(self.app.overlay.hide, delay, "transcript")

    def play(self, sound: str) -> None:
        if self.app.cfg.sounds:
            AppHelper.callAfter(lambda: (s := NSSound.soundNamed_(sound)) and s.play())


class MainThreadHandler:
    """Hotkey events: leave the event-tap callback quickly, act right after."""

    def __init__(self, controller: Controller):
        self.c = controller

    def on_press(self, role):
        AppHelper.callAfter(self.c.on_press, role)

    def on_release(self, role):
        AppHelper.callAfter(self.c.on_release, role)

    def on_tap(self, role):
        AppHelper.callAfter(self.c.on_tap, role)

    def on_cancel(self, role):
        AppHelper.callAfter(self.c.on_cancel, role)


class MenuBarApp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "loading"
        self.detail = ""
        self.overlay = Overlay()
        self.target_obj = MenuTarget.alloc().init()
        self.target_obj.app = self
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.menu = NSMenu.alloc().init()
        self.menu.setDelegate_(self.target_obj)
        self.menu.setAutoenablesItems_(False)
        self.item.setMenu_(self.menu)
        self.controller: Controller | None = None
        self.set_state("loading", "Loading…")

    # -- menu -----------------------------------------------------------

    def _add(self, menu, title, action=None, enabled=True, key=""):
        it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        it.setTarget_(self.target_obj)
        it.setEnabled_(enabled and action is not None)
        menu.addItem_(it)
        return it

    def refresh_menu(self) -> None:
        m = self.menu
        m.removeAllItems()
        dk = KEY_NAMES.get(self.cfg.dictate_key, self.cfg.dictate_key)
        ck = KEY_NAMES.get(self.cfg.command_key, self.cfg.command_key)
        status = {
            "loading": "Loading models…",
            "idle": "Ready",
            "ptt": "Listening…",
            "handsfree": "Hands-free: listening",
            "command": "Listening for a command…",
            "error": "Error",
        }[self.state]
        self._add(m, status + (f" — {self.detail}" if self.detail else ""))
        tap = "tap" if self.cfg.handsfree_tap == "single" else "double-tap"
        self._add(m, f"Hold {dk} to dictate · {tap} for hands-free")
        self._add(m, f"Hold {ck} for a command")
        m.addItem_(NSMenuItem.separatorItem())
        hf = self._add(m, "Hands-free mode", "toggleHandsFree:")
        c = self.controller
        hf.setState_(1 if c and c.mode == "handsfree" else 0)
        recent = self._add(m, "Recent", None)
        sub = NSMenu.alloc().init()
        history = list(reversed(c.executor.history[-10:])) if c else []
        for ins in history:
            title = ins.text.strip().replace("\n", " ↵ ")
            it = self._add(sub, (title[:60] + "…") if len(title) > 60 else title, "copyRecent:")
            it.setRepresentedObject_(ins.text)
        if not history:
            self._add(sub, "Nothing yet")
        recent.setSubmenu_(sub)
        recent.setEnabled_(True)
        if c and c.last_latency_ms is not None:
            self._add(m, f"Last: {c.last_latency_ms:.0f} ms from key release to text")
        smart = "on" if (c and c.llm is not None) else ("loading" if self.cfg.llm_enabled else "off")
        self._add(m, f"Smart edits (local LLM): {smart}")
        if not accessibility_trusted():
            self._add(m, "⚠︎ Grant Accessibility access…", "permissions:")
        m.addItem_(NSMenuItem.separatorItem())
        self._add(m, "Edit config…", "editConfig:", key=",")
        self._add(m, "Reload config", "reloadConfig:", key="r")
        self._add(m, "Open log", "openLog:")
        m.addItem_(NSMenuItem.separatorItem())
        self._add(m, "Quit", "quit:", key="q")

    def set_state(self, state: str, detail: str = "") -> None:
        self.state, self.detail = state, detail
        btn = self.item.button()
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(ICONS[state], "Dictation")
        if img is not None:
            img.setTemplate_(True)
            btn.setImage_(img)
        else:
            btn.setTitle_("\U0001f399")
        tint = {
            "ptt": NSColor.systemRedColor(),
            "handsfree": NSColor.systemRedColor(),
            "command": NSColor.systemOrangeColor(),
            "error": NSColor.systemYellowColor(),
        }.get(state)
        btn.setContentTintColor_(tint)
        if state == "error" and detail:
            self.overlay.show("Error", detail)
            self.overlay.hide(6)

    def reload_config(self) -> None:
        try:
            cfg = load_config()
        except Exception as e:
            self.overlay.show("Config error", str(e))
            self.overlay.hide(5)
            return
        self.cfg = cfg
        if self.controller:
            self.controller.reload(cfg)
        self.overlay.show("Config", "Reloaded. Hotkey changes need a restart.")
        self.overlay.hide(2)


def run_app() -> None:
    cfg = load_config()
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    ui = MenuBarApp(cfg)
    target = MacTarget(cfg.output_method, cfg.restore_clipboard)  # main thread: reads keyboard layout
    controller = Controller(cfg, UIBridge(ui), target)
    ui.controller = controller

    import os

    if not accessibility_trusted(prompt="LOCAL_DICTATION_NO_PROMPT" not in os.environ):
        log.warning("Accessibility access missing")
        ui.overlay.show(
            "Permission needed",
            "Allow Accessibility for your terminal in System Settings › Privacy & Security, then restart.",
        )
        ui.overlay.hide(10)

    from .audio import mic_permission, request_mic_permission

    perm = mic_permission()
    log.info("microphone permission: %s", perm)
    if perm == "not_determined":
        request_mic_permission(lambda ok: log.info("microphone permission granted: %s", ok))
    elif perm != "authorized":
        ui.overlay.show(
            "Microphone blocked",
            "Allow Microphone for your terminal app in System Settings \u203a Privacy & Security \u203a Microphone, then restart.",
        )
        ui.overlay.hide(12)

    hotkeys = HotkeyListener({"dictate": cfg.dictate_key, "command": cfg.command_key}, cfg.tap_max_ms,
                             MainThreadHandler(controller))
    hotkeys.ignore_source_states.add(target.source_state_id)
    if not hotkeys.start():
        log.error("could not install the keyboard event tap (Accessibility / Input Monitoring missing)")
        ui.set_state("error", "No keyboard access: allow Accessibility and Input Monitoring, then restart.")
    ui._hotkeys = hotkeys  # keep alive

    controller.start()
    ui.refresh_menu()
    print(f"local-dictation running. Hold {KEY_NAMES.get(cfg.dictate_key)} to dictate "
          f"(tap it for hands-free), "
          f"{KEY_NAMES.get(cfg.command_key)} for commands. Ctrl+C to quit.", flush=True)
    AppHelper.runEventLoop(installInterrupt=True)


# keep objc import used (selectors resolved by name)
_ = objc
