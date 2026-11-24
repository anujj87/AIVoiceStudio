"""Temporary diagnostic: where does a scaled SettingsDialog's size come from?

Prints the size, minimum, maximum and best size of a real SettingsDialog
before and after a scale change, and what an explicit SetSize does.  Run by the
throw-away ``.github/workflows/diag.yml`` on a CI runner.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wx

from ai_voice_studio.gui import dpi
from ai_voice_studio.gui.settings_dialog import SettingsDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore


def report(tag: str, window: wx.Window) -> None:
    state = getattr(window, "_aivs_dpi_state", {}) or {}
    print(f"{tag}: size={tuple(window.GetSize())} "
          f"min={tuple(window.GetMinSize())} max={tuple(window.GetMaxSize())} "
          f"best={tuple(window.GetBestSize())} "
          f"state_size={state.get('size')} state_scale={state.get('scale')} "
          f"content={state.get('size_from_content')}")


def main() -> None:
    app = wx.App(False)
    print("wx", wx.version())
    display = wx.Display(0)
    print("display", display.GetGeometry(), "client", display.GetClientArea())

    folder = tempfile.mkdtemp()
    settings = Settings(path=os.path.join(folder, "settings.json"))
    store = ModelStore(state_file=os.path.join(folder, "models.json"))
    frame = wx.Frame(None)

    with dpi.at_scale(1.0):
        dlg = SettingsDialog(frame, settings, store)
        report("1.0 dialog", dlg)

    dpi.set_scale_override(2.0)
    dpi.rescale(dlg)
    report("rescaled to 2.0", dlg)

    dlg.SetSize((1640, 1120))
    report("explicit SetSize 1640x1120", dlg)

    dlg.SetMinSize((-1, -1))
    dlg.SetMaxSize((-1, -1))
    dlg.SetSize((1640, 1120))
    report("same, min/max cleared", dlg)

    with dpi.at_scale(2.0):
        dlg2 = SettingsDialog(frame, settings, store)
        report("dialog built at 2.0", dlg2)

    style = (wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX
             | wx.MINIMIZE_BOX)
    probe = wx.Dialog(None, size=(100, 100), style=style)
    probe.SetSize((100_000, 100_000))
    print("plain resizable dialog caps at", tuple(probe.GetSize()))
    probe.Destroy()
    app.Destroy()


if __name__ == "__main__":
    main()
