"""Keyboard-gesture audit: every access key and accelerator of every window.

The companion tools answer narrower questions - ``a11y_mnemonics.py`` lists
access keys and spots two buttons sharing one, ``a11y_adjacency.py`` compares
labels with accessible names.  This tool asks the question the user actually
hits: *does every gesture do something, and does it do the thing it says?*

For every window of the application it reports

  Alt+<letter>  each action key of the top-level window that is visible now
                (a control inside a hidden category counts as invisible),
                with the control's enabled state, and flags

  !DUP     two visible controls claim the same letter.  Windows activates the
           first and the other becomes unreachable from the keyboard.
  !SHADOW  a menu item of the window's menu bar owns the letter.  A frame
           answers its menu first, so a child button with the same letter can
           never be reached (the label would be lying).
  !SILENT  the letter resolves only to a *disabled* control and the window has
           no ``show_access_key_hint`` to explain why - the key does nothing
           and says nothing.
  !MISS    the letter cannot be resolved back to the control that carries it
           (a bug in the resolver itself).

and, for windows with a menu bar, every menu accelerator (Ctrl+Shift+N and
friends) with duplicates flagged (!ACC) and every menu mnemonic per menu with
duplicates flagged (!DUP).

Run: .venv/Scripts/python.exe tools/a11y_gestures.py
"""

from __future__ import annotations

import os
import sys

import wx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ai_voice_studio.gui import access_keys  # noqa: E402


def _hint_sink(window: wx.Window) -> wx.Window | None:
    """The nearest window that can explain an unavailable access key."""
    win: wx.Window | None = window
    while win is not None:
        if hasattr(win, "show_access_key_hint"):
            return win
        win = win.GetParent()
    return None


def _menu_items(window: wx.Window):
    """Yield ``(menu_name, item)`` for every item of ``window``'s menu bar."""
    root = access_keys.top_level(window) or window
    get_bar = getattr(root, "GetMenuBar", None)
    if get_bar is None:
        return
    bar = get_bar()
    if bar is None:
        return
    for index in range(bar.GetMenuCount()):
        menu = bar.GetMenu(index)
        if menu is None:
            continue
        label = bar.GetMenuLabel(index).replace("&", "")
        for item in menu.GetMenuItems():
            if item.GetKind() == wx.ITEM_SEPARATOR:
                continue
            yield label, item
            sub = item.GetSubMenu()
            if sub is not None:
                for sub_item in sub.GetMenuItems():
                    if sub_item.GetKind() != wx.ITEM_SEPARATOR:
                        yield label, sub_item


def audit(title: str, window: wx.Window) -> list[str]:
    """Print the gestures of one window; return the problem lines."""
    root = access_keys.top_level(window) or window
    problems: list[str] = []
    entries = access_keys.access_keys(window)

    print(f"  -- window: {type(root).__name__}")
    if not entries:
        print("     (no action keys)")
    seen: dict[str, str] = {}
    for entry in entries:
        state = "enabled" if entry.enabled else "DISABLED"
        line = f"     Alt+{entry.key.upper():<2} {entry.label!r} ({state})"
        if entry.key in seen:
            line += f"   !DUP with {seen[entry.key]!r}"
            problems.append(f"!DUP   Alt+{entry.key.upper()} is claimed by "
                            f"{entry.label!r} and {seen[entry.key]!r}")
        else:
            seen[entry.key] = entry.label
        if access_keys.menu_claims(root, entry.key):
            line += "   !SHADOW"
            problems.append(f"!SHADOW Alt+{entry.key.upper()} ({entry.label!r}) "
                            "is owned by a menu item and can never reach this "
                            "control")
        resolved = access_keys.receiver(root, entry.key)
        if resolved is None or resolved.control is not entry.control:
            line += "   !MISS"
            problems.append(f"!MISS  Alt+{entry.key.upper()} ({entry.label!r}) "
                            "does not resolve back to its own control")
        if not entry.enabled and access_keys.enabled_receiver(root, entry.key) is None:
            line += "   (inert: disabled)"
            if _hint_sink(entry.control) is None:
                line += "   !SILENT"
                problems.append(
                    f"!SILENT Alt+{entry.key.upper()} ({entry.label!r}) lands on "
                    "a disabled control and the window has no "
                    "show_access_key_hint to explain it")
        print(line)

    # Menu gestures: accelerators (Ctrl+Shift+N ...) and per-menu mnemonics.
    accelerators: dict[str, str] = {}
    mnemonics: dict[tuple[str, str], str] = {}
    for menu_name, item in _menu_items(root):
        label = item.GetItemLabel()
        text, _, accel = label.partition("\t")
        if accel:
            if accel in accelerators:
                problems.append(f"!ACC   {accel} is on both {text!r} and "
                                f"{accelerators[accel]!r}")
                print(f"     {accel:<14} {text!r}   !ACC with "
                      f"{accelerators[accel]!r}")
            else:
                accelerators[accel] = text
                print(f"     {accel:<14} {text!r}")
        letter = access_keys.mnemonic(text)
        if letter:
            key = (menu_name, letter)
            if key in mnemonics:
                problems.append(f"!DUP   menu {menu_name!r}: Alt+{letter.upper()} "
                                f"is on both {text!r} and {mnemonics[key]!r}")
                print(f"     menu {menu_name} Alt+{letter.upper()} {text!r}   "
                      f"!DUP with {mnemonics[key]!r}")
            else:
                mnemonics[key] = text
    return problems


def main() -> int:
    from a11y_windows import iter_windows

    app = wx.App(False)  # noqa: F841 - must stay referenced

    all_problems: list[str] = []
    for title, window in iter_windows():
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        for problem in audit(title, window):
            all_problems.append(f"{title}: {problem}")

    print("\n" + "=" * 78)
    if all_problems:
        print(f"{len(all_problems)} gesture problem(s):")
        for problem in all_problems:
            print("  " + problem)
        return 1
    print("No keyboard-gesture problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
