# AI Voice Studio — Development Plan

Milestones map to the implementation order. Each phase ends with something runnable/verifiable.
Status is tracked at the top; check boxes are updated as phases complete.

Current release: **2026.3.7** (`constants.APP_VERSION`;
`TERMS_VERSION` stays 2026.3.1 because the terms text did not change, so an
upgrade does not ask the user to accept them again; `installer_common.iss`).
Test suite: unit, GUI smoke, engine-environment, book, documentation,
accessibility and winget-manifest guards. Installer:
`dist\AI-Voice-Studio-v-2026-3-7-Setup-x64.exe`.

2026.3.7 adds the update path and the winget package. **Help → Check for
updates** (`Alt+D`, and *Check for updates now* in Settings → General) asks
GitHub for the newest release, shows what changed and downloads the installer
for this Windows version; a quiet check runs once a day at start-up and the
dialog can mute a single version. Nothing is downloaded or installed without
the user asking. The same release carries the generated winget manifests
(`winget/manifests/a/AnujSharma/AIVoiceStudio/<version>/`) so
`winget install AnujSharma.AIVoiceStudio` and `winget upgrade` use exactly
that installer; `tools/winget_validate.py` checks them against winget's rules
in the test suite.

The test suite itself no longer crashes on the way out: every GUI test module
built its own `wx.App`, and a second one shuts the first down while its
windows are still alive, so the interpreter died with an access violation
*after* the last test passed (pytest exit code 139 on a green run). All five
GUI modules now share one application from `tests/wx_test_app.py`, and
`tests/test_wx_test_app.py` fails the build if a module ever constructs its
own again.

2026.3.6 settles two gestures the user reported. **Remove selected variant** no
longer carries an Alt+R access key at all (a key that looks available and then
does nothing is worse than none; the button stays on Tab and Enter).
**Ctrl+Shift+N** is now owned solely by the welcome panel's *Create New
Project* button: the File-menu item carries no accelerator and the frame's key
handler sends the button its own click event, so gesture and mouse click cannot
diverge. While fixing it, the frame's key handler was measured against real
keystrokes and now confirms Ctrl/Shift against the physical keyboard, because
wx 3.3.3 can deliver a Shift+letter combination with the modifier bits missing
("Ctrl+Shift+N does nothing" was reproduced exactly that way).

2026.3.5 makes the keyboard gestures dependable: one shared access-key handler
(`gui/access_keys.py`) activates buttons in every window whatever has the focus,
explains a key whose target is disabled instead of doing nothing, and runs one
action per keystroke (Windows answers a mnemonic twice per press). The Download
and remove page opens on a downloaded model, removal reports what it did, the
main window's button keys no longer collide with the menu bar, and
`tools/a11y_gestures.py` audits every window's keys, shadows and accelerators.

## Phase 0 — Research & specification ✅
- [x] Study [bookworm](https://github.com/blindpandas/bookworm) and other accessible readers
      (wxPython for screen-reader support, heading-based navigation, Inno Setup installer).
- [x] Survey ONNX neural TTS ecosystems: **Piper** (rhasspy), **Kokoro-82M**, and the
      **sherpa-onnx** runtime (Piper/VITS/Kokoro/Matcha/MMS with CPU/CUDA).
- [x] Decide runtime: **sherpa-onnx** = single synthesis backend; models downloaded from
      official HF/GitHub URLs (no model files bundled).
- [x] Write `PROJECT_SPEC.md` (the full self-prompt specification).

## Phase 1 — Project scaffolding & data layer ✅
- [x] Package layout, `main.py`, `requirements*.txt`.
- [x] `paths.py` (user dir under `%APPDATA%\AIVoiceStudio`), `settings.py` (JSON store
      with defaults/reset/recent projects), `compute.py` (CPU/CUDA + Auto, CPU thread budget).

## Phase 2 — Model catalog & downloader ✅
- [x] `catalog.py` + `models_catalog.json`: nine TTS families (Piper, Kokoro-82M, Kitten,
      VITS zh-AISHELL3, Matcha, OmniVoice, OmniVoice Server, SAPI5, Windows Core).
- [x] `downloader.py`: resumable downloads (Range + `.part`), progress callbacks,
      cancellation, safe extraction, install/remove state (`models.json`).

## Phase 3 — TTS engine ✅
- [x] `engine.py`: sherpa-onnx wrapper (VITS for Piper/custom, Kokoro config), sample-rate
      aware, engine cache per (voice, provider), graceful degradation.
- [x] Punctuation (none/math/all/default) + pitch (resample) + volume applied on samples.
- [x] `windows_tts.py`: SAPI5 and Windows Core voices as first-class engines.

## Phase 4 — Documents & splitting ✅
- [x] `parsers.py`: TXT/MD/HTML/PDF/DOCX (+DOC via Word COM), clipboard; heading extraction.
- [x] `splitter.py`: 4 modes + naming rules, unique-title guard, filename sanitising.

## Phase 5 — Audio output ✅
- [x] `output.py` (stdlib WAV writer), `ffmpeg.py` (detect, download gyan.dev build to
      `%APPDATA%\AIVoiceStudio\ffmpeg`, convert WAV→MP3/FLAC).

## Phase 6 — GUI ✅
- [x] `theme.py` (system/light/dark) + `a11y.py` (accessible labels; MSAA names via
      `finalize_accessibility`).
- [x] `main_frame.py` (menu + accelerators + recent projects + status bar + context menu).
- [x] `settings_dialog.py` + `model_panels.py` + `clone_engines_panel.py`: thirteen categories.
- [x] `new_project_wizard.py`, `recording_dialog.py` (per-project TTS,
      Start/Pause/Resume/Stop, instant save, resume-after-restart).

## Phase 7 — Background jobs & resilience ✅
- [x] `synthesizer.py` worker + `wx.PostEvent` events; per-segment instant disk save;
      resume from first unsaved segment; optional exclusive `end_index` for single-file runs.

## Phase 8 — Packaging ✅
- [x] PyInstaller spec (onedir, bundles runtime + catalog + docs/book), Inno Setup scripts
      (32/64-bit), `build.ps1`. Installer options: location, Start Menu, desktop icon,
      readme, launch, licence page.
- [x] Installer built and smoke-tested (frozen build launched with an isolated `%APPDATA%`).

## Phase 9 — Tests ✅
- [x] Unit + GUI smoke + environment tests (22 files, 396 cases).
- [x] Documentation guards (`tests/test_docs.py`): the Help menu, the installer's document
      list and the documents themselves must agree, no retired component may be described as
      current, and every version statement must name the release.
- [x] End-to-end verified: real Piper + Kokoro downloads, synthesis to WAV, worker job,
      resume logic, GUI construction smoke tests, start-selected-recording end to end.
- [ ] Manual screen-reader walkthrough with NVDA on the final installers.

## Phase 10 — OmniVoice (direct + server) ✅
- [x] `omnivoice/` (triton runner in a worker subprocess, `spec.py` as the single vocabulary,
      voice store) and `omnivoice_server/` (OpenAI-compatible HTTP server manager).
- [x] Full generation surface (design, clone, parameters, presets, voice profiles, auth).
- [x] Settings categories for both engines and a per-project options dialog.

## Phase 11 — DAISY and playlists ✅
- [x] `playlist_builder.py` (`.m3u8` / `.pls` / `.wpl`), `daisy_builder.py` (2.02),
      `daisy3_builder.py` (Z39.86-2005 with images) and "Export DAISY as ZIP".

## Phase 12 — Voice Lab engines ✅
- [x] `voicelab/engines.py` registry (Pocket TTS, Bark, F5-TTS) with built-in voices and
      licence metadata; Bark's 130 speaker presets generated from its own library.
- [x] JSON-lines worker protocol; `VoicelabEngine` adapter behind the normal engine cache.
- [x] Per-engine tuning (`voicelab/options.py`) with Settings defaults and project overrides.
- [x] NeuTTS removed: it needed a gated Hugging Face login the application cannot provide.

## Phase 13 — One Python environment per engine ✅
- [x] `python_runtime` creates `tts_envs\<engine>` per pip-installed engine;
      `SHARED_ENVIRONMENTS` folds the two OmniVoice engines into one.
- [x] `venv_packages` per-environment cached probes + a direct site-packages check for the
      decisions that must be right immediately (which interpreter starts a worker).
- [x] `verify_engine_import` after each install; `PYTORCH_CUDA_INDEX` (cu128) so every CUDA
      wheel comes from one index and Python 3.13 is supported.
- [x] Audio I/O shim: TorchCodec probe and soundfile fallback, so FFmpeg is optional.

## Phase 14 — Compute choice ✅
- [x] `compute.cpu_threads()` (80–95 % of logical CPUs) and full-card CUDA tuning.
- [x] `gui/compute_choice.py`: a Compute combo beside every Preview button, remembered per
      category (`preview_compute.<category>`).

## Phase 15 — Onboarding and licensing ✅
- [x] First-launch terms of use (`gui/accept_dialog.py`) gated in `main.py`; versioned
      acceptance, all check boxes required, no way to dismiss into consent.
- [x] `LICENSE` and `docs/THIRD-PARTY-LICENSES.html` covering every bundled and
      downloaded component, linked from the Help menu.

## Phase 16 — Start Selected Recording ✅
- [x] Replaced "Restart Selected Recording" with a picker: recorded-files mode vs.
      file-break mode, and one-file vs. from-here scope; no "no recordings yet" dead end.

## Phase 17 — Documentation ❌ open
- [x] `docs/book`: 82 chapters and 3 appendices (basic Python → rebuilding the
      application), rebuilt as a single HTML file and a PDF.
- [x] `README.html`, `UserGuide.html`, `AddonDevelopmentGuide.html`,
      `AccessibilityGuide.html`, `THIRD-PARTY-LICENSES.html`, root `README.md`,
      `PROJECT_SPEC.md`, this plan — all brought up to the current implementation.
- [ ] Rewrite the original application chapters (43–56) in the same depth as Part X so that
      no chapter still describes the pre-Voice-Lab design.

## Phase 18 — Updates and the winget package ✅
- [x] `updates.py`: version comparison, the GitHub `releases/latest` payload, the
      installer for this architecture, and a resumable download through the model
      downloader. A failed check is data, never an exception that reaches the UI.
- [x] `gui/update_dialog.py`: the quiet daily check, the Help menu's *Check for
      updates*, and the download-and-install flow (`/SILENT`, then close the app).
- [x] `updates.auto_check` / `updates.last_check` / `updates.skip_version` in
      `settings.json`, with a switch and a *Check for updates now* button in
      Settings → General → Updates.
- [x] Help menu item *Check for updates* (Alt+D, unique inside the menu).
- [x] `winget/manifests/a/AnujSharma/AIVoiceStudio/<version>/` generated by
      `tools/make_winget.py`; `tools/winget_validate.py` + `tests/test_winget.py`
      enforce winget's rules (identifier, dotted version, SPDX licence, inno,
      `/VERYSILENT`, versioned URL, ProductCode, SHA256).
- [x] Installer registers proper ARP links (`AppPublisherURL`, `AppSupportURL`,
      `AppUpdatesURL`); the Inno `AppId` stays unchanged so upgrades keep working.

## Next
- [ ] Manual NVDA walkthrough of the installer and the first-launch dialog.
- [ ] 32-bit build verification (a 32-bit Python 3.13 venv is required).
- [ ] Signing the installer so SmartScreen stops warning on first run.
- [ ] Submit the generated manifests to `microsoft/winget-pkgs` once the release exists
      (`winget/README.md` has the steps).
