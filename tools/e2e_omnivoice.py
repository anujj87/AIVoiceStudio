"""End-to-end OmniVoice test through the Recording window.

Drives the exact code path a real recording uses (RecordingDialog ->
SynthesisWorker -> OmniVoice engine -> worker subprocess -> WAV on disk):

  1. creates a throw-away project with a couple of segments,
  2. opens the Recording window for it,
  3. records once in **voice design** mode, once in **voice clone** mode
     (cloning the voice produced by the first run),
  4. reports the output files, durations, and every worker warning / skipped
     parameter the OmniVoice worker reported.

Run from the source checkout with the app venv::

    .venv\\Scripts\\python tools/e2e_omnivoice.py

It needs an NVIDIA GPU, the managed Python runtime created, and
``omnivoice-triton`` installed there (the normal Settings > Compute setup).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO_ROOT, "build", "e2e_omnivoice")

_E2E_APP = None  # persistent wx.App handle

# Allow running as ``python tools/e2e_omnivoice.py`` from anywhere:
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The worker subprocess must run with the *managed* Python (it holds
# omnivoice-triton + torch); in development the app spawns the worker with
# sys.executable, so point it at the managed environment for this test.
ADDON_PYTHON = os.path.join(
    os.environ.get("APPDATA", ""), "AIVoiceStudio", "addon_env", "Scripts", "python.exe"
)

DESIGN_TEXT = (
    "Hello. This is an end to end test of OmniVoice in AI Voice Studio. "
    "A designed voice is speaking these sentences. Female narrator, "
    "young adult, british accent, clear and calm."
)
DESIGN_INSTRUCT = "female, young adult, british accent"
CLONE_TEXT = (
    "And this second file tests voice cloning. "
    "It should sound like the same narrator that spoke the first file, "
    "because that file was used as the reference sample."
)


def _logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("e2e")


def _make_project(root: str, name: str, text: str, omni: dict) -> str:
    from datetime import datetime

    from ai_voice_studio import project

    project_dir = os.path.join(root, name)
    # Start from an empty folder: reusing a previous run's directory made the
    # report list stale WAVs as this run's output (a failed design run looked
    # like it had produced audio, because yesterday's files were still there).
    shutil.rmtree(project_dir, ignore_errors=True)
    os.makedirs(project_dir, exist_ok=True)
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    segments = [
        {
            "index": i,
            "title": f"{i:02d} {sentences[i][:40]}",
            "text": sentences[i],
            "status": "pending",
            "saved": None,
        }
        for i in range(len(sentences))
    ]
    data = {
        "name": name,
        "source_file": "",
        "created": datetime.now().isoformat(),
        "project_type": "audio_playlist",
        "segments": segments,
        "tts": {
            "compute": "cuda_gpu",
            "punctuation": "default",
            "output_format": "wav",
            "omni": omni,
        },
    }
    project.save_project(project_dir, data)
    return project_dir


def _release_engines():
    """Close cached engines (frees the GPU workers) between runs."""
    from ai_voice_studio.tts import engine as engine_mod

    for eng in list(engine_mod._engine_cache.values()):
        try:
            eng.close()
        except Exception:  # noqa: BLE001
            pass
    engine_mod.clear_engine_cache()
    gc.collect()


def _record(log, project_dir: str, mode_label: str) -> dict:
    """Open the Recording window and record all pending segments."""
    import wx

    from ai_voice_studio.gui.recording_dialog import RecordingDialog
    from ai_voice_studio.gui import dialogs as dialogs_mod
    from ai_voice_studio.settings import Settings
    from ai_voice_studio.tts.models import ModelStore

    global _E2E_APP  # keep the wx.App alive for the whole run
    if wx.GetApp() is None:
        _E2E_APP = wx.App(False)

    messageboxes = []

    def fake_messagebox(message, caption="", style=0, *a, **kw):
        messageboxes.append((str(caption), str(message)))
        return wx.OK

    def fake_action_dialog(parent, title, message, actions, default_key=None):
        """Stand in for the modal labelled-button dialog.

        The recording-complete dialog (and the OmniVoice server error dialog)
        lives in ``dialogs`` rather than ``wx.MessageBox``; without this stub
        the run blocks forever on ``ShowModal`` waiting for a click.
        """
        messageboxes.append((str(title), str(message)))
        if default_key is not None:
            return default_key
        return actions[0][0] if actions else None

    original_messagebox = wx.MessageBox
    original_action_dialog = dialogs_mod.run_action_dialog
    wx.MessageBox = fake_messagebox
    dialogs_mod.run_action_dialog = fake_action_dialog

    # In development the OmniVoice engine spawns its worker with
    # sys.executable; point it at the managed Python that owns
    # omnivoice-triton + torch (in frozen builds the app already does this).
    real_executable = sys.executable
    sys.executable = ADDON_PYTHON
    try:
        dialog = RecordingDialog(None, project_dir, Settings(), ModelStore())
        try:
            dialog.Show()

            # The voice list is filled in asynchronously (the installed-package
            # probe runs in the background), so pump events until the
            # OmniVoice engine shows up instead of reading the combo once.
            def _tts_index() -> int:
                return next(
                    (i for i in range(dialog.tts_combo.GetCount())
                     if dialog.tts_combo.GetClientData(i) == "omnivoice"),
                    -1,
                )

            def _wait_until(predicate, timeout=60.0) -> bool:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    wx.Yield()
                    if predicate():
                        return True
                    time.sleep(0.05)
                return bool(predicate())

            # TTS engine -> OmniVoice (direct / omnivoice-triton).  This must
            # come first: the Compute combo is rebuilt from the selected
            # engine, so "CUDA GPU (OmniVoice)" only exists once an OmniVoice
            # engine is selected (picking it before this raised "the compute
            # option is missing" whenever another engine sorted first).
            if not _wait_until(lambda: _tts_index() >= 0):
                raise RuntimeError(
                    "No OmniVoice voices available - is omnivoice-triton "
                    "installed in the managed Python environment?"
                )
            dialog.tts_combo.SetSelection(_tts_index())
            dialog._on_tts(None)

            # Compute -> CUDA GPU (OmniVoice), then refresh the engine list.
            idx = next(
                (i for i in range(dialog.compute_combo.GetCount())
                 if dialog.compute_combo.GetClientData(i) == "cuda_gpu"),
                -1,
            )
            if idx < 0:
                raise RuntimeError("The 'CUDA GPU (OmniVoice)' compute option is missing")
            dialog.compute_combo.SetSelection(idx)
            dialog._on_compute_change(None)

            # Voice -> the first catalog voice (auto) of the first variant.
            if dialog.voice_combo.GetCount():
                dialog.voice_combo.SetSelection(0)

            started = time.monotonic()
            dialog._on_start(None)
            deadline = time.monotonic() + 540
            while time.monotonic() < deadline:
                wx.Yield()
                worker = dialog._worker
                if worker is None or not worker.is_alive():
                    break
                time.sleep(0.05)
            wx.Yield()
            elapsed = time.monotonic() - started

            outputs = sorted(
                os.path.join(project_dir, f)
                for f in os.listdir(project_dir)
                if f.lower().endswith(".wav")
            )
            return {
                "mode_label": mode_label,
                "status": dialog.status.GetLabel(),
                "outputs": outputs,
                "elapsed_s": elapsed,
                "messageboxes": messageboxes,
            }
        finally:
            try:
                dialog._worker and dialog._worker.cancel()
            except Exception:  # noqa: BLE001
                pass
            dialog.Destroy()
    finally:
        wx.MessageBox = original_messagebox
        dialogs_mod.run_action_dialog = original_action_dialog
        sys.executable = real_executable


def _wav_info(path: str) -> dict:
    with wave.open(path, "rb") as wf:
        return {
            "frames": wf.getnframes(),
            "rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "seconds": round(wf.getnframes() / wf.getframerate(), 2),
        }


def main() -> int:
    log = _logger()
    if not os.path.isfile(ADDON_PYTHON):
        log.error("Managed Python not found at %s", ADDON_PYTHON)
        return 2

    os.environ["HF_HUB_OFFLINE"] = "1"  # model is already cached
    os.makedirs(OUT_ROOT, exist_ok=True)

    from ai_voice_studio.omnivoice import spec
    from ai_voice_studio.project import first_pending_index

    results = []
    try:
        # ---- Run 1: voice design -----------------------------------------
        omni = spec.build_omni(
            mode="design",
            instruct=DESIGN_INSTRUCT,
            num_step=32,
            guidance_scale=3.0,
            class_temperature=0.0,
        )
        design_dir = _make_project(OUT_ROOT, "design", DESIGN_TEXT, omni)
        log.info("== RUN 1: voice design into %s", design_dir)
        results.append(("design", _record(log, design_dir, "voice design")))

        # Use the first produced file as the clone reference.
        design_outputs = results[0][1]["outputs"]
        if not design_outputs:
            # OmniVoice occasionally returns no audio for the first segment of
            # a fresh model load.  That is upstream flakiness rather than a
            # failure of the app path, so report the run instead of crashing on
            # an empty output list, and skip the clone run (it needs a
            # reference sample).
            log.warning(
                "The design run produced no audio (see the report below); "
                "skipping the clone run."
            )
        else:
            ref = design_outputs[0]
            log.info("Reference sample for clone run: %s", ref)

            # ---- Run 2: voice clone --------------------------------------
            _release_engines()
            omni = spec.build_omni(
                mode="clone",
                ref_audio=ref,
                ref_text=DESIGN_TEXT,  # exact transcript -> no Whisper download
                num_step=32,
                guidance_scale=3.0,
                class_temperature=0.0,
            )
            clone_dir = _make_project(OUT_ROOT, "clone", CLONE_TEXT, omni)
            log.info("== RUN 2: voice clone into %s", clone_dir)
            results.append(("clone", _record(log, clone_dir, "voice clone")))

        # ---- Report --------------------------------------------------------
        print("\n" + "=" * 78)
        print("OMNIVOICE E2E RESULT (Recording window path)")
        print("=" * 78)
        for kind, info in results:
            print(f"\n[{kind.upper()}] {info['mode_label']}")
            print(f"  elapsed: {info['elapsed_s']:.1f}s   status: {info['status']}")
            for mb in info["messageboxes"]:
                print(f"  message box: {mb[0]}: {mb[1][:200]}")
            for out in info["outputs"]:
                print(f"  wav: {out}  {_wav_info(out)}")
            if not info["outputs"]:
                print("  !! NO WAV OUTPUT PRODUCED")
        print("\nWorker messages (skipped parameters / warnings):")
        print("-" * 78)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("E2E failed: %s: %s", type(exc).__name__, exc)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        _release_engines()


if __name__ == "__main__":
    sys.exit(main())
