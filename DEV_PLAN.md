# AI Voice Studio — Development Plan

Milestones map to the implementation order. Each phase ends with something runnable/verifiable.
Status is tracked at the top; check boxes are updated as phases complete.

## Phase 0 — Research & specification ✅
- [x] Study [bookworm](https://github.com/blindpandas/bookworm) and other accessible readers
      (wxPython for screen-reader support, heading-based navigation, Inno Setup installer).
- [x] Survey ONNX neural TTS ecosystems: **Piper** (rhasspy), **Kokoro-82M**, and the
      **sherpa-onnx** runtime (Piper/VITS/Kokoro/Matcha/MMS with CPU/CUDA/DirectML).
- [x] Decide runtime: **sherpa-onnx** = single synthesis backend; models downloaded from
      official HF/GitHub URLs (no model files bundled).
- [x] Write `PROJECT_SPEC.md` (the full self-prompt specification).

## Phase 1 — Project scaffolding & data layer ✅
- [x] Package layout, `main.py`, `requirements*.txt` (core/gpu/npu).
- [x] `paths.py` (user dir under `%APPDATA%\AIVoiceStudio`), `settings.py` (JSON store
      with defaults/reset/recent projects), `compute.py` (CPU/CUDA/DirectML + Auto).

## Phase 2 — Model catalog & downloader ✅
- [x] `catalog.py` + `models_catalog.json`: Piper (30 languages, real verified URLs),
      Kokoro-82M (38 voices), VITS/MMS experimental.
- [x] `downloader.py`: resumable downloads (Range + `.part`), progress callbacks,
      cancellation, safe extraction, install/remove state (`models.json`).

## Phase 3 — TTS engine ✅
- [x] `engine.py`: sherpa-onnx wrapper (VITS for Piper/custom, Kokoro config), sample-rate
      aware, engine cache per (voice, provider), graceful degradation.
- [x] Punctuation (none/math/all/default) + pitch (resample) + volume applied on samples.

## Phase 4 — Documents & splitting ✅
- [x] `parsers.py`: TXT/MD/HTML/PDF/DOCX (+DOC via Word COM), clipboard; heading extraction.
- [x] `splitter.py`: 4 modes + naming rules, unique-title guard, filename sanitising.

## Phase 5 — Audio output ✅
- [x] `output.py` (stdlib WAV writer), `ffmpeg.py` (detect, download gyan.dev build to
      `%APPDATA%\AIVoiceStudio\ffmpeg`, convert WAV→MP3/FLAC).

## Phase 6 — GUI ✅
- [x] `theme.py` (system/light/dark) + `a11y.py` (accessible labels via `SetName`).
- [x] `main_frame.py` (menu + accelerators + recent projects + status bar).
- [x] `settings_dialog.py` + `model_panels.py`: all 7 tabs.
- [x] `new_project_wizard.py` (2 pages), `recording_dialog.py` (per-project TTS,
      Start/Pause/Resume/Stop, instant save, resume-after-restart).

## Phase 7 — Background jobs & resilience ✅
- [x] `synthesizer.py` worker + `wx.PostEvent` events; per-segment instant disk save;
      resume from first unsaved segment.

## Phase 8 — Packaging ✅ (scaffold)
- [x] PyInstaller spec (onedir, bundles runtime + catalog), Inno Setup 32/64-bit scripts,
      `build.ps1`. Installer options: location, Start Menu, desktop icon, readme, launch.
- [ ] Build smoke test of the installers themselves (needs Inno Setup + per-arch venvs).

## Phase 9 — Tests ✅ / manual QA open
- [x] 22 unit tests pass (splitter naming, parsers, settings, punctuation).
- [x] End-to-end verified: real Piper + Kokoro downloads, synthesis to WAV, worker job,
      resume logic, GUI construction smoke tests.
- [ ] Manual screen-reader walkthrough with NVDA on the final installers.
