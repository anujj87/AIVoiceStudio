"""Access keys (Alt+letter) that work the same in every window.

Windows answers a button's ``&`` mnemonic itself, but only along one narrow
path: the key has to be translated into a character that matches a label, it
has to arrive at the *focused* control, and it has to survive whatever that
control's default window procedure does with it (measured on this build: a
``WM_SYSCHAR`` delivered to the dialog does nothing at all, while the same
message delivered to the focused control activates the button).  When any of
those links fails the key silently does nothing - which is exactly what
"Alt+R will not remove the selected model" looks like from the outside.

This module is the application's own, deterministic answer to that:

* ``install()`` hooks a window once.  A top-level window's char hook sees
  every keystroke in it, whatever has focus, so nothing has to be translated
  or forwarded for the key to work.
* the letter is matched against the label's ``&`` marker, layout independent,
  with the raw scan code and the physical key state as fallbacks for the wx
  builds that hand Alt+letter over untranslated;
* a target that is *disabled* is reported (beep, log line, and a window may
  answer ``show_access_key_hint(entry)`` to say why in words) instead of
  doing nothing;
* one keystroke activates one control: ``claim()`` keeps a window's own
  handler and this hook from both firing on the same key;
* a key that a menu already owns is left to the menu, which is what Windows
  does in a frame - the menu bar is checked before any child control.

Nothing here changes a label: the ``&`` markers stay in place, so the
underlines a sighted user sees and the names a screen reader announces are
untouched.
"""

from __future__ import annotations

import logging
import sys
import time

import wx

log = logging.getLogger(__name__)

#: How long a claimed keystroke blocks a second activation, in seconds.
#: Long enough to swallow the second path of the same key press and the
#: auto-repeat of a held key, short enough for a deliberate repeat.
REPEAT_GUARD_SECONDS = 0.6

#: The command event a real click on each control type sends.
_COMMAND_EVENTS: tuple[tuple[type, int], ...] = (
    (wx.ToggleButton, wx.EVT_TOGGLEBUTTON.typeId),
    (wx.RadioButton, wx.EVT_RADIOBUTTON.typeId),
    (wx.CheckBox, wx.EVT_CHECKBOX.typeId),
    (wx.Button, wx.EVT_BUTTON.typeId),
)

#: The Windows virtual key of the Alt key, used by ``alt_is_down``.
_VK_MENU = 0x12

#: PC scan codes of the letter keys (1 is Escape, so A starts at 0x1E).
_SCAN_CODES = {chr(ord("A") + i): code for i, code in enumerate((
    0x1E, 0x30, 0x2E, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17, 0x24, 0x25, 0x26,
    0x32, 0x31, 0x18, 0x19, 0x10, 0x13, 0x1F, 0x14, 0x16, 0x2F, 0x11, 0x2D,
    0x15, 0x2C,
))}


def mnemonic(label: str) -> str | None:
    """The letter ``label`` underlines, lower-cased (``None`` when there is none).

    A menu item's label may carry its accelerator after a tab; only the label
    half is searched.  ``&&`` is an escaped ampersand and introduces no
    access key, so it is skipped and the first single ``&`` wins.
    """
    text = label.split("\t")[0]
    index = 0
    while index < len(text):
        if text[index] == "&":
            if index + 1 < len(text) and text[index + 1] == "&":
                index += 2
                continue
            if index + 1 < len(text):
                return text[index + 1].lower()
            return None
        index += 1
    return None


class AccessKey:
    """One control, the letter its label underlines, and whether it can act."""

    __slots__ = ("control", "label", "key", "enabled")

    def __init__(self, control: wx.Window, label: str, key: str, enabled: bool):
        self.control = control
        self.label = label
        self.key = key
        self.enabled = enabled

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AccessKey(Alt+{self.key.upper()}, {self.label!r}, "
                f"enabled={self.enabled})")


def top_level(window: wx.Window | None) -> wx.Window | None:
    """The top-level window ``window`` belongs to (``window`` itself if none)."""
    if window is None:
        return None
    win = window
    while not isinstance(win, wx.TopLevelWindow):
        parent = win.GetParent()
        if parent is None:
            return win
        win = parent
    return win


def shown_on_screen(window: wx.Window) -> bool:
    """True when ``window`` and every ancestor up to its top-level is shown.

    This is the rule Windows itself uses when it looks for a mnemonic: a
    control inside a hidden panel is invisible and therefore cannot answer
    a key, even though ``IsShown()`` on the control itself is still True.

    The walk stops at the top-level window: a dialog's owner frame being
    hidden (a minimised main window, a dialog opened from a frame that is
    not shown in a test) does not hide the dialog.
    """
    win: wx.Window | None = window
    while win is not None:
        if not win.IsShown():
            return False
        if isinstance(win, wx.TopLevelWindow):
            break
        win = win.GetParent()
    return True


def access_keys(window: wx.Window) -> list[AccessKey]:
    """Every actionable access key inside ``window``, in creation order.

    Only controls that can be *activated* are returned: buttons, check boxes,
    radio buttons and toggle buttons.  A static text's ``&`` (a label for the
    control next to it) is Windows' own business and is left alone.
    """
    root = top_level(window) or window
    found: list[AccessKey] = []

    def walk(container: wx.Window) -> None:
        for child in container.GetChildren():
            if not isinstance(child, wx.Window):
                continue
            if isinstance(child, (wx.Button, wx.CheckBox, wx.RadioButton,
                                  wx.ToggleButton)) and shown_on_screen(child):
                label = child.GetLabel()
                letter = mnemonic(label)
                if letter:
                    found.append(AccessKey(child, label, letter,
                                           child.IsEnabled()))
            if not isinstance(child, (wx.ComboBox, wx.Choice, wx.SpinCtrl,
                                      wx.SpinCtrlDouble, wx.RadioBox)):
                walk(child)

    walk(root)
    return found


def receiver(window: wx.Window, key: str) -> AccessKey | None:
    """The first visible control that answers Alt+``key`` (enabled or not)."""
    for entry in access_keys(window):
        if entry.key == key:
            return entry
    return None


def enabled_receiver(window: wx.Window, key: str) -> AccessKey | None:
    """The first *enabled* visible control that answers Alt+``key``.

    Windows skips a disabled control when it resolves a mnemonic and keeps
    looking, so a duplicate key on a disabled control does not block the live
    one.  This mirrors that.
    """
    for entry in access_keys(window):
        if entry.key == key and entry.enabled:
            return entry
    return None


def menu_claims(window: wx.Window, key: str) -> bool:
    """True when a menu item of ``window``'s menu bar already owns Alt+``key``.

    A frame's menu bar is answered by Windows before any child control, so
    the menu keeps the key (Alt+R in the main window is "&Resume Recording",
    not the welcome panel's remove button).
    """
    root = top_level(window) or window
    get_bar = getattr(root, "GetMenuBar", None)
    if get_bar is None:
        return False
    bar = get_bar()
    if bar is None:
        return False

    def walk_menu(menu: wx.Menu) -> bool:
        for item in menu.GetMenuItems():
            if item.GetKind() == wx.ITEM_SEPARATOR:
                continue
            label = item.GetItemLabel()
            if mnemonic(label) == key:
                return True
            sub = item.GetSubMenu()
            if sub is not None and walk_menu(sub):
                return True
        return False

    for index in range(bar.GetMenuCount()):
        menu = bar.GetMenu(index)
        if menu is not None and walk_menu(menu):
            return True
    return False


def alt_letter(evt: wx.KeyEvent) -> str | None:
    """The letter of an Alt+letter keystroke, or ``None``.

    AltGr arrives as Ctrl+Alt and is therefore not an access key.  wx can
    hand an Alt+letter combination over as an untranslated key (key code 0)
    on this build, which is why the raw scan code and, as a last resort, the
    physical key state are consulted before giving up - that is what makes
    the key work on a keyboard layout whose Alt+letter produces no character.
    """
    if not evt.AltDown() or evt.ControlDown():
        return None
    code = evt.GetKeyCode()
    if 0x41 <= code <= 0x5A:
        return chr(code).lower()
    if 0x61 <= code <= 0x7A:
        return chr(code)
    if code != 0 or sys.platform != "win32":
        return None
    try:
        scan = (evt.GetRawKeyFlags() >> 16) & 0xFF
    except Exception:  # noqa: BLE001 - raw flags are not always available
        scan = 0
    for letter, code_of_letter in _SCAN_CODES.items():
        if code_of_letter == scan:
            return letter.lower()
    import ctypes  # noqa: PLC0415 - Windows-only fallback

    user32 = ctypes.windll.user32
    for letter in _SCAN_CODES:
        if user32.GetKeyState(ord(letter)) & 0x8000:
            return letter.lower()
    return None


def alt_is_down() -> bool:
    """True while the Alt key is physically held down.

    This tells a key-driven activation from a mouse click, which is what lets
    ``once()`` de-duplicate the first without ever dropping the second.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # noqa: PLC0415 - Windows-only check

        return bool(ctypes.windll.user32.GetKeyState(_VK_MENU) & 0x8000)
    except Exception:  # noqa: BLE001 - never let a key handler fail here
        return False


def _control_of(source) -> wx.Window | None:
    """The control an event, a control or ``None`` refers to."""
    if source is None:
        return None
    if isinstance(source, wx.Window):
        return source
    get_object = getattr(source, "GetEventObject", None)
    if get_object is None:
        return None
    obj = get_object()
    return obj if isinstance(obj, wx.Window) else None


def once(source, *, seconds: float = REPEAT_GUARD_SECONDS) -> bool:
    """True when this access-key activation should go ahead.

    Windows answers one Alt+letter keystroke *twice*: the dialog manager
    matches the mnemonic on ``WM_SYSKEYDOWN`` and again on the ``WM_SYSCHAR``
    the message loop derives from it (both are delivered to the focused
    control for a real key press, and both were measured to activate the
    button here).  A window's own hook may see the same key a third time.
    Without a guard a button bound to an access key therefore runs its
    handler two or three times per press: two confirmation dialogs, two
    downloads, two previews.

    Handlers that start something call this first, with their event (or the
    control, when they are invoked programmatically)::

        def _on_remove(self, evt):
            if not access_keys.once(evt):
                return

    Only Alt-driven activations are de-duplicated - a mouse click, Enter or
    Space always passes - and the dispatcher's own activation is passed
    through even though it may be delivered after the native one.
    """
    control = _control_of(source)
    if control is None:
        return True
    now = time.monotonic()
    last = getattr(control, "_access_key_run", None)
    fresh = last is not None and now - last[1] < seconds
    if getattr(control, "_access_key_in_flight", False):
        # This activation is the dispatcher's own (see activate()).
        if fresh and last[2] == "native":
            return False
        control._access_key_run = (last[0] if last else None, now, "ours")  # type: ignore[attr-defined]
        return True
    if not alt_is_down():
        return True
    if fresh:
        return False
    control._access_key_run = (None, now, "native")  # type: ignore[attr-defined]
    return True


def claim(window: wx.Window, key: str,
          seconds: float = REPEAT_GUARD_SECONDS) -> bool:
    """True for the first handler that sees this Alt+``key`` keystroke.

    A window has two independent paths to the same access key (its own
    handler plus this module's hook, in either order), and a fixed dialog may
    deliver one key press twice.  Whichever path arrives first wins; the
    other sees the claim and keeps quiet, so one keystroke does one thing.
    """
    root = top_level(window) or window
    now = time.monotonic()
    last = getattr(root, "_access_key_claim", None)
    if last is not None and last[0] == key and now - last[1] < seconds:
        return False
    root._access_key_claim = (key, now)  # type: ignore[attr-defined]
    return True


def _hint_sink(control: wx.Window) -> wx.Window | None:
    """The nearest window able to explain an unavailable access key."""
    win: wx.Window | None = control
    while win is not None:
        if hasattr(win, "show_access_key_hint"):
            return win
        win = win.GetParent()
    return None


def _report_unavailable(window: wx.Window, entry: AccessKey) -> None:
    """Say that Alt+<key> is there but cannot act, instead of nothing."""
    log.info("Access key Alt+%s: %r is disabled - telling the user",
             entry.key.upper(), entry.label)
    try:
        wx.Bell()
    except Exception:  # noqa: BLE001 - a beep must never raise
        pass
    sink = _hint_sink(entry.control)
    if sink is None or not hasattr(sink, "show_access_key_hint"):
        return
    try:
        sink.show_access_key_hint(entry)
    except Exception:  # noqa: BLE001 - an explanation must never raise
        log.exception("Access key Alt+%s: show_access_key_hint failed",
                      entry.key.upper())


def activate(entry: AccessKey) -> None:
    """Do what a click on ``entry.control`` does.

    The control's own event is sent to its own handlers, so every listener
    the control has (the panel's, the dialog's) sees a normal activation.
    Check boxes, radio buttons and toggle buttons are switched first, exactly
    as the mouse would leave them, and focus moves to the control so a screen
    reader announces where the user landed.
    """
    control = entry.control
    log.info("Access key Alt+%s: %s (%s)", entry.key.upper(), entry.label,
             type(control).__name__)
    control._access_key_in_flight = True  # type: ignore[attr-defined]
    try:
        if isinstance(control, (wx.CheckBox, wx.ToggleButton)):
            control.SetValue(not control.GetValue())
        elif isinstance(control, wx.RadioButton):
            control.SetValue(True)
        for kind, event_type in _COMMAND_EVENTS:
            if isinstance(control, kind):
                control.SetFocus()
                event = wx.CommandEvent(event_type, control.GetId())
                event.SetEventObject(control)
                control.GetEventHandler().ProcessEvent(event)
                return
    except Exception:  # noqa: BLE001 - report, never crash the window
        log.exception("Access key Alt+%s: activating %r failed",
                      entry.key.upper(), entry.label)
    finally:
        control._access_key_in_flight = False  # type: ignore[attr-defined]


def handle_key(window: wx.Window, evt: wx.KeyEvent) -> bool:
    """Answer one char-hook event; True when this module dealt with it."""
    if wx.IsBusy():
        return False
    letter = alt_letter(evt)
    if letter is None:
        return False
    root = top_level(evt.GetEventObject() or window) or window
    if menu_claims(root, letter):
        return False
    entry = enabled_receiver(root, letter)
    if entry is None:
        disabled = receiver(root, letter)
        if disabled is None:
            return False
        _report_unavailable(root, disabled)
        return True
    if not claim(root, letter):
        return True
    activate(entry)
    return True


def install(window: wx.Window) -> None:
    """Make every access key of ``window`` work.  Safe to call twice."""
    if getattr(window, "_access_keys_installed", False):
        return
    window._access_keys_installed = True  # type: ignore[attr-defined]

    def on_char_hook(evt: wx.KeyEvent) -> None:
        if not handle_key(window, evt):
            evt.Skip()

    window.Bind(wx.EVT_CHAR_HOOK, on_char_hook)
    log.debug("Access keys installed on %s", type(window).__name__)


class AccessKeyHints:
    """Mixin: explain an access key that lands on a disabled control.

    A window that mixes this in answers ``show_access_key_hint``, which is
    what ``_report_unavailable`` looks for when Alt+letter reaches a control
    that cannot act.  Panels override ``access_key_hint`` to say *why* in
    their own words; the base wording is the honest fallback.
    """

    #: Status labels a panel may own, best first (see access_key_hint_target).
    HINT_TARGET_NAMES = ("progress_label", "preview_status", "clone_status",
                         "design_status", "lib_status", "status")

    def access_key_hint_target(self) -> wx.StaticText | None:
        """The status line this window uses for access-key explanations."""
        for name in self.HINT_TARGET_NAMES:
            candidate = getattr(self, name, None)
            if isinstance(candidate, wx.StaticText):
                return candidate
        return None

    def access_key_hint(self, entry: AccessKey) -> str:
        """The sentence to show when access key ``entry`` cannot act."""
        tip = (entry.control.GetToolTipText() or "").strip()
        if tip:
            return f"Alt+{entry.key.upper()}: {tip}"
        return (f"Alt+{entry.key.upper()}: {entry.label} is unavailable in "
                "the current state.")

    def show_access_key_hint(self, entry: AccessKey) -> None:
        """Say why an access key could not act (called by this module)."""
        message = self.access_key_hint(entry)
        target = self.access_key_hint_target()
        if target is not None:
            target.SetLabel(message)
        log.info("Access key hint: %s", message)


def describe(window: wx.Window) -> list[str]:
    """One line per access key of ``window`` (used by the audit tool)."""
    lines = []
    for entry in access_keys(window):
        state = "enabled" if entry.enabled else "DISABLED"
        lines.append(f"Alt+{entry.key.upper():<2} {entry.label!r} ({state})")
    return lines


__all__ = [
    "AccessKey", "AccessKeyHints", "REPEAT_GUARD_SECONDS", "access_keys",
    "activate", "alt_is_down", "alt_letter", "claim", "describe",
    "enabled_receiver", "handle_key", "install", "menu_claims", "mnemonic",
    "once", "receiver", "shown_on_screen", "top_level",
]
