"""Measure how long the Settings dialog takes to open, and what it builds.

Run:  .venv/Scripts/python.exe tools/measure_settings_open.py

Reports the wall time of ``SettingsDialog(...)``, how many category panels it
constructed, and the window-handle count of the dialog right after opening
(versus the same dialog with every category visited, which is what the old
eager implementation always paid).
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore


def _child_count(window: wx.Window) -> int:
    stack = list(window.GetChildren())
    total = 0
    while stack:
        child = stack.pop()
        total += 1
        stack.extend(child.GetChildren())
    return total


def main() -> int:
    app = wx.App(False)
    frame = wx.Frame(None, title="measure")
    folder = tempfile.mkdtemp(prefix="aivs_measure_settings_")
    settings = Settings(path=os.path.join(folder, "settings.json"))
    store = ModelStore(state_file=os.path.join(folder, "models.json"))

    opens = []
    for _ in range(3):
        start = time.perf_counter()
        dlg = SettingsDialog(frame, settings, store)
        opens.append(time.perf_counter() - start)
        built = len(dlg.built_panels())
        total = dlg.panel_count()
        children = _child_count(dlg)
        dlg.Destroy()

    print(f"open: best {min(opens):.3f}s  median {statistics.median(opens):.3f}s  "
          f"worst {max(opens):.3f}s  over {len(opens)} opens")
    print(f"opening builds {built} of {total} categories, "
          f"{children} child windows")

    dlg = SettingsDialog(frame, settings, store)
    per_category = []
    for index in range(dlg.panel_count()):
        start = time.perf_counter()
        dlg.ensure_panel(index)
        per_category.append((dlg.CATEGORIES[index].title,
                             time.perf_counter() - start))
    print(f"every category built ({len(dlg.built_panels())} panels), "
          f"{_child_count(dlg)} child windows")
    for title, seconds in sorted(per_category, key=lambda row: -row[1])[:6]:
        print(f"    {seconds * 1000:7.0f} ms  {title}")
    dlg.Destroy()
    frame.Destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
