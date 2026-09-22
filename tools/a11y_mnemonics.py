"""Access-key (mnemonic) audit.

Prints, for every button of every window, the Alt+<key> combination that
activates it, and flags the two defects that make access keys useless:

  * !DUP  - two *visible* buttons in the same window answer to the same
            Alt+key, so Windows fires only the first one (the other is
            unreachable by keyboard).
  * !BAD  - a label whose '&' does not introduce a real access key
            (a trailing '&', or '&&'), which shows a stray ampersand and
            registers no shortcut at all.

An access key that leaks into the accessible name is reported as well
(!NAME): Windows strips the '&' from a Win32 button's name, so a name that
still contains one means the label was copied into SetName() by mistake.

Run: .venv/Scripts/python.exe tools/a11y_mnemonics.py
"""

from __future__ import annotations

import os
import sys

import wx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def access_key(label: str) -> str | None:
    """The letter Windows underlines for ``label`` (None when there is none).

    ``&&`` is an escaped ampersand and introduces no access key, so it is
    skipped; the first single ``&`` wins.
    """
    index = 0
    while index < len(label):
        if label[index] == "&":
            if index + 1 < len(label) and label[index + 1] == "&":
                index += 2
                continue
            if index + 1 < len(label):
                return label[index + 1].lower()
            return None
        index += 1
    return None


def _visible(window: wx.Window) -> bool:
    win = window
    while win is not None:
        if not win.IsShown():
            return False
        win = win.GetParent()
    return True


def audit(title: str, window: wx.Window) -> list[str]:
    """Print one window's access keys; return the problem lines."""
    rows: list[tuple[str, str, str]] = []  # (label, key, name)

    def walk(container: wx.Window) -> None:
        for child in container.GetChildren():
            if not isinstance(child, wx.Window):
                continue
            if isinstance(child, wx.Button) and _visible(child):
                label = child.GetLabel()
                key = access_key(label)
                rows.append((label, key or "-", child.GetName()))
            if not isinstance(child, (wx.ComboBox, wx.Choice, wx.SpinCtrl,
                                      wx.SpinCtrlDouble, wx.RadioBox)):
                walk(child)

    walk(window)

    problems: list[str] = []
    seen: dict[str, str] = {}
    for label, key, name in rows:
        line = f"  Alt+{key.upper():<2} {label!r}"
        if key == "-" and ("&&" in label or label.endswith("&")):
            line += "   !BAD"
            problems.append(f"!BAD  {label!r} has no usable access key")
        if key != "-":
            if key in seen:
                line += f"   !DUP with {seen[key]!r}"
                problems.append(
                    f"!DUP  Alt+{key.upper()} is claimed by {label!r} "
                    f"and {seen[key]!r}"
                )
            else:
                seen[key] = label
        if "&" in (name or ""):
            line += "   !NAME"
            problems.append(f"!NAME {name!r} still contains '&'")
        print(line)

    if not rows:
        print("  (no buttons)")
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
        print(f"{len(all_problems)} access-key problem(s):")
        for problem in all_problems:
            print("  " + problem)
        return 1
    print("No access-key problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
