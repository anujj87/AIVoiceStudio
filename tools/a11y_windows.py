"""Every auditable window of the application, built once for the a11y tools.

``tools/a11y_audit.py`` (screen-reader names), ``tools/a11y_adjacency.py``
(visible label vs accessible name) and ``tools/a11y_mnemonics.py`` (access
keys) all need the same set of windows; building them here keeps the three
reports in step - a window added below is automatically audited by all of
them.

``iter_windows()`` yields ``(title, window)`` pairs.  A generator suspends
between yields, so the window stays alive until the consumer asks for the
next one (it is destroyed right after resuming).

Nothing here is imported by the application.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import wx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from a11y_sample import ensure_sample_project  # noqa: E402

from ai_voice_studio.settings import Settings  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402

PROJECT_DIR_FALLBACK = os.path.join("build", "e2e_omnivoice", "design")


def _laid_out(window: wx.Window) -> wx.Window:
    """Resize ``window`` sensibly and lay it out before it is audited.

    The geometry-based audits read control positions, and a window that was
    only just created has everything at (0, 0); a window squeezed smaller
    than its content (a wizard page, for example) additionally stacks its
    rows on top of each other and invents label mismatches that the real,
    properly sized window does not have.  Giving every window room for its
    best size and then laying it out keeps their reports truthful.
    """
    try:
        best = window.GetBestSize()
        current = window.GetSize()
        window.SetSize((max(current.width, best.width, 720),
                        max(current.height, best.height, 480)))
        window.Layout()
    except Exception:  # noqa: BLE001 - a dev tool must never raise
        pass
    return window


def _store() -> ModelStore:
    os.makedirs(os.path.join("build", "a11y_models"), exist_ok=True)
    return ModelStore(
        state_file=os.path.join(os.getcwd(), "build", "a11y_models",
                                "models_state.json")
    )


def iter_windows():
    """Yield ``(title, window)`` for every window of the application."""
    store = _store()
    settings = Settings()

    # -- Settings dialog: one window per category panel -------------------
    import ai_voice_studio.gui.settings_dialog as sd

    dlg = sd.SettingsDialog(None, settings, store)
    dlg.SetSize((1000, 900))
    dlg.Show()
    for index, category in enumerate(dlg.CATEGORIES):
        dlg._show_category(index)
        dlg.Layout()
        yield f"[{category.title}] panel", _laid_out(dlg._panels[index])
    dlg.Destroy()

    # -- Recording window (needs a project folder) ------------------------
    from ai_voice_studio.gui.recording_dialog import RecordingDialog

    project_dir = PROJECT_DIR_FALLBACK
    if not os.path.isfile(os.path.join(project_dir, "project.json")):
        project_dir = ensure_sample_project()
    recording = RecordingDialog(None, project_dir, settings, store)
    recording.SetSize((900, 800))
    recording.Show()
    yield "Recording dialog", _laid_out(recording)
    recording.Destroy()

    # -- OmniVoice voice options dialog -----------------------------------
    from ai_voice_studio.gui.omnivoice_options_dialog import OmniVoiceOptionsDialog

    omni = OmniVoiceOptionsDialog(None, engine_label="OmniVoice", omni={},
                                 project_name="demo")
    omni.SetSize((760, 800))
    omni.Show()
    yield "OmniVoice options dialog", _laid_out(omni)
    omni.Destroy()

    # -- Voice Lab tuning dialog, one per engine --------------------------
    from ai_voice_studio.gui.voicelab_options_dialog import VoiceLabOptionsDialog
    from ai_voice_studio.voicelab import engines as voice_lab

    for engine_id in voice_lab.engine_ids():
        tuning = VoiceLabOptionsDialog(None, engine_id, values={},
                                       project_name="demo")
        tuning.SetSize((760, 800))
        tuning.Show()
        yield f"Voice Lab options dialog ({engine_id})", _laid_out(tuning)
        tuning.Destroy()

    # -- First-launch terms dialog ----------------------------------------
    from ai_voice_studio.gui.accept_dialog import AcceptanceDialog

    terms = AcceptanceDialog(None)
    terms.Show()
    yield "First-launch terms dialog", _laid_out(terms)
    terms.Destroy()

    # -- Main window (welcome panel) --------------------------------------
    from ai_voice_studio.gui.main_frame import MainFrame

    settings_dir = tempfile.mkdtemp(prefix="aivs_a11y_")
    frame_settings = Settings(path=os.path.join(settings_dir, "settings.json"))
    frame_settings.add_recent_project("A11y sample", project_dir)
    frame = MainFrame(settings=frame_settings, store=store)
    frame.SetSize((900, 700))
    frame.Show()
    yield "Main window (welcome panel)", _laid_out(frame)
    frame.Destroy()
    shutil.rmtree(settings_dir, ignore_errors=True)

    # -- New Project wizard, every page -----------------------------------
    from ai_voice_studio.gui.new_project_wizard import NewProjectWizard

    wizard = NewProjectWizard(None, settings, store)
    wizard.SetSize((1000, 900))
    wizard.Show()
    for page in (wizard.page_details, wizard.page_mode, wizard.page_daisy,
                 wizard.page_info):
        # A step that is not current is hidden, and a screen reader does not
        # see hidden controls, so each page is shown before it is audited.
        wizard.ShowPage(page)
        yield f"New Project wizard - {type(page).__name__}", _laid_out(page)
    wizard.Destroy()

    # -- "Start selected recording" picker --------------------------------
    from ai_voice_studio import project as project_mod
    from ai_voice_studio.gui.main_frame import _StartRecordingDialog, _picker_entries

    data = project_mod.load_project(project_dir)
    saved = project_mod.recorded_files(project_dir)
    picker = _StartRecordingDialog(None, _picker_entries(data, saved))
    picker.Show()
    yield "Start selected recording picker", _laid_out(picker)
    picker.Destroy()

    # -- Progress dialog and reusable action dialog -----------------------
    from ai_voice_studio.gui.dialogs import _ActionDialog
    from ai_voice_studio.gui.progress import TaskProgressDialog

    progress = TaskProgressDialog(None, title="Working", message="Working...",
                                  cancel_label="Cancel")
    progress.Show()
    yield "Progress dialog", _laid_out(progress)
    progress.Destroy()

    action = _ActionDialog(
        None, "Recording complete", "All segments recorded.",
        [("open", "Open project folder"), ("ok", "OK")], default_key="ok",
    )
    action.Show()
    yield "Action dialog", _laid_out(action)
    action.Destroy()
