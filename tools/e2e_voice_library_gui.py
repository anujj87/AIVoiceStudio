"""Headless E2E drive of Settings -> "OmniVoice engines" voice library.

Creates a clone voice and a design voice through the real panel handlers
(no GPU synthesis - creation/rename/delete are local store work), then
verifies the voices surface in the Available TTS panel lists.

Run:  .venv/Scripts/python tools/e2e_voice_library_gui.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wx  # noqa: E402

from ai_voice_studio import paths  # noqa: E402
from ai_voice_studio.gui.settings_dialog import SettingsDialog  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="aivs_vlib_")
    fd, state = tempfile.mkstemp(prefix="aivs_models_", suffix=".json")
    os.close(fd)

    sample = os.path.join(tmp, "ref.wav")
    with open(sample, "wb") as fh:
        fh.write(b"RIFFxxxxWAVE not really audio but copy only")

    orig_models_dir = paths.models_dir
    paths.models_dir = lambda: os.path.join(tmp, "models")
    orig_msg = wx.MessageBox
    messages = []

    def fake_msg(text, caption="", style=wx.OK):
        messages.append((caption, str(text)))
        return wx.YES if style & wx.YES_NO else wx.OK

    wx.MessageBox = fake_msg

    class _FakeEntryDlg:
        def __init__(self, *a, **k):
            self._value = "Renamed Voice"

        def SetValue(self, v):
            self._value = v

        def GetValue(self):
            return self._value

        def ShowModal(self):
            return wx.ID_OK

        def Destroy(self):
            pass

    orig_entry = wx.TextEntryDialog
    wx.TextEntryDialog = _FakeEntryDlg

    try:
        app = wx.App(False)
        frame = wx.Frame(None)
        store = ModelStore(state_file=state)
        dlg = SettingsDialog(frame, Settings(), store)
        panel = dlg._panels[3]
        assert panel.title == "OmniVoice engines", panel.title

        # --- clone voice ------------------------------------------------
        panel.sample_ctrl.SetValue(sample)
        panel.clone_name_ctrl.SetValue("My Clone")
        panel._on_create("clone")
        voices = [v for v in store.custom_voices() if v["name"] == "My Clone"]
        assert voices, "clone not registered"
        assert voices[0]["kind"] == "omni_clone"
        assert os.path.isfile(voices[0]["sample"]), "sample not copied"
        assert panel.voices_list.GetCount() == 1
        print("OK clone created:", panel.voices_list.GetString(0))

        # --- design voice -----------------------------------------------
        idx = panel.mode_combo.FindString("Describe a new voice")
        assert idx >= 0
        panel.mode_combo.SetSelection(idx)
        panel.design_desc_ctrl.SetValue("female, young adult, indian accent")
        panel.design_name_ctrl.SetValue("Narrator")
        panel._on_create("design")
        assert store.custom_voices() and any(
            v["name"] == "Narrator" for v in store.custom_voices()
        ), "design voice missing"
        assert panel.voices_list.GetCount() == 2
        print("OK design created:", panel.voices_list.GetString(1))

        # --- rename (stubbed text entry dialog) -------------------------
        panel.voices_list.SetSelection(0)
        panel._on_rename(None)
        names = {v["name"] for v in store.custom_voices()}
        assert "Renamed Voice" in names and "My Clone" not in names, names
        print("OK renamed clone -> 'Renamed Voice'")

        # --- library voice is engine-agnostic in Available TTS ----------
        # (category activation must refresh the list built at dialog open)
        dlg.available_panel.on_activated()
        shown = {v["voice"] for v in dlg.available_panel._voices
                 if v.get("custom_omni")}
        assert shown == {"Renamed Voice", "Narrator"}, shown
        print("OK Available TTS lists library voices:", sorted(shown))

        dlg._panels[6].on_activated()  # Punctuation panel
        shown_p = {v["voice"] for v in dlg._panels[6]._voices
                   if v.get("custom_omni")}
        assert shown_p == {"Renamed Voice", "Narrator"}, shown_p
        print("OK Punctuation panel lists library voices:", sorted(shown_p))

        # --- delete (MessageBox answered YES) ---------------------------
        sel = next(i for i, v in enumerate(panel._library)
                   if v["name"] == "Narrator")
        panel.voices_list.SetSelection(sel)
        panel._on_delete(None)
        remaining = {v["name"] for v in store.custom_voices()}
        assert "Narrator" not in remaining and "Renamed Voice" in remaining
        print("OK design voice deleted; clone kept:", sorted(remaining))

        print("VOICE LIBRARY GUI E2E: PASS")
        return 0
    finally:
        wx.MessageBox = orig_msg
        wx.TextEntryDialog = orig_entry
        paths.models_dir = orig_models_dir
        for p in (state,):
            if os.path.exists(p):
                os.remove(p)
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
