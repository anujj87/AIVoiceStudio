# AI Voice Studio — Project Specification (Self-Prompt)

> This document is the "AI prompt for yourself": the complete, unambiguous specification that
> an engineer (human or AI) can follow to build, extend, and test the software. It consolidates
> the original requirements, the research findings, and the technical decisions made during
> development.

## 1. Product summary

**AI Voice Studio** is a Windows desktop application that converts text documents into
spoken audio files (audiobooks, study notes, lectures) using **neural text-to-speech (TTS)
models that run locally through ONNX Runtime**. It is built with **Python + wxPython** and is
designed from the ground up to be **fully accessible with screen readers** (NVDA, JAWS, Narrator).

Key properties:

- 100% local, privacy-friendly synthesis (no cloud API).
- The application and the ONNX runtime are shipped inside the installer.
- TTS models are **not** bundled: the user downloads them from the Settings → Download/Remove
  tab (official open-source model URLs). This keeps the installer small and licenses transparent.
- Output formats: **WAV, MP3, FLAC** (MP3/FLAC via FFmpeg, downloaded on demand to the app data folder).
- Input formats: **PDF, TXT, DOC/DOCX, HTML, MD, and clipboard text**.
- Compute back-ends: **CPU, GPU (CUDA), NPU (DirectML)** with auto-detection; unavailable
  back-ends are hidden from the UI.

## 2. Research summary (what we read before coding)

### 2.1 Similar projects studied

| Project | What we learned from it |
|---|---|
| [blindpandas/bookworm](https://github.com/blindpandas/bookworm) | Accessible document reader. Lessons: real screen-reader support needs correct label/name propagation on **every** control; use wxPython (not Qt/Tk) because wxWidgets has the best native accessibility; support many document formats; structured navigation by headings; keyboard-friendly menus; NSIS/Inno-based Windows installer; `uv`-managed builds; GPL-2+ licensing. |
| rhasspy/piper | Fast local neural TTS; voice catalog is a 4-level hierarchy (language → voice → quality → model files). The voice catalog URL scheme became the model-registry design. |
| k2-fsa/sherpa-onnx | The reference ONNX TTS runtime. Supports Piper, VITS, Kokoro-82M, Matcha-TTS, MMS TTS, Coqui VITS, with CPU / CUDA / DirectML providers. Its Python API is our single synthesis backend. |
| kokoro-onnx | Showed that Kokoro-82M (a very high-quality 82M-param model) is available as pure ONNX and is freely usable (Apache-2.0). |

### 2.2 ONNX neural TTS model catalog (verified sources)

| TTS | License | Languages | Variants | Voice cloning | Source / download host |
|---|---|---|---|---|---|
| **Piper** | MIT | 20+ (en, de, fr, es, it, pt, ru, uk, pl, nl, sv, hu, cs, fi, tr, vi, ar, zh, ja…) | quality tiers: `x_low`, `low`, `medium`, `high` | No | `huggingface.co/rhasspy/piper-voices` (official catalog) |
| **Kokoro-82M** | Apache-2.0 | en (US/GB/AU), zh, ja, es, fr, hi, it, pt, ko | `v1.0` … `v1.5` snapshots | No (needs fine-tune) | `huggingface.co/hexgrad/Kokoro-82M` and sherpa-onnx mirrors |
| **VITS (multilingual)** | MPL-2.0 (Coqui) | en, de, es, fr, ja, ko, zh… | `vctk`, `ljspeech`, `aishell3`… | No | sherpa-onnx sample models on GitHub Releases |
| **Matcha-TTS** | CC-BY-4.0 / MIT | en, de, es, fr, nl… | single | No | sherpa-onnx releases |
| **MMS TTS** | CC-BY-NC-4.0 (note: non-commercial) | 1100+ languages | `eng`, `deu`, `fra`, `spa`… | No | sherpa-onnx releases |
| **Coqui VITS / XTTS v2 (ONNX)** | MPL-2.0 / CPML | multi | `xtts_v2` | **Yes** (reference + speaker embedding) | HF mirrors; XTTS v2 has an official ONNX export |

Voice-cloning support (Settings → Voice Clone tab) targets **Coqui XTTS v2 (ONNX export)** and
**Piper fine-tuned voice uploads** (a `.onnx` + `.onnx.json` pair trained from the user's own
voice is a valid "clone" workflow that is fully offline and free). The voice-clone tab lists only
TTS engines flagged `voice_cloning: true` in the catalog.

### 2.3 Compute providers

| Back-end | ONNX Runtime package | Execution providers |
|---|---|---|
| CPU | `onnxruntime` (bundled) | `CPUExecutionProvider` |
| GPU | `onnxruntime-gpu` (user-installable at setup/update) | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| NPU | `onnxruntime-directml` (user-installable, Windows) | `DmlExecutionProvider` (Intel/AMD/Qualcomm NPU via DirectML) |

Detection rules:

- **CPU** is always available.
- **GPU** is offered only if a CUDA-capable provider is present (check `onnxruntime.get_available_providers()`
  and/or presence of `onnxruntime-gpu` import + `nvidia-smi`).
- **NPU** is offered only if `DmlExecutionProvider` is available **and** a DirectML/NPU-capable
  accelerator exists on the system (queried via `wmi` / PnP device enumeration; gracefully
  skipped if the query fails).
- If more than one compute option is available, an **Auto** entry is added (preference order:
  GPU → NPU → CPU).

## 3. Functional requirements

### 3.1 Settings dialog (File → Settings / `Ctrl+,`)

A tabbed dialog with **OK / Cancel / Apply** buttons. Tabs:

1. **General** — Theme combo box: `System default`, `Light`, `Dark`. Applies live.
2. **Download and remove** (model manager):
   - Combo 1: Select TTS (catalog top-level, e.g. Piper, Kokoro).
   - Combo 2: Select language (of that TTS).
   - Combo 3: Select variant (e.g. `medium`, `low`, `v1.5`).
   - **Download** button (enabled when the selected voice is not fully downloaded) and
     **Remove** button (enabled when downloaded). Show per-file progress.
3. **Available TTS**:
   - Combo 1: TTS, Combo 2: Language, Combo 3: Variant, Combo 4: Voice.
   - Lists only **downloaded** voices so the user can preview/use them.
   - Voice-clone uploads appear here too (voice name = file name of the uploaded sample).
4. **Voice clone**:
   - Combo 1: TTS, Combo 2: Language, Combo 3: Variant (only entries of cloning-capable TTS).
   - **Upload voice** button → file dialog for audio sample (`.wav/.mp3/.flac/.onnx`).
   - For Piper: a `.onnx` + `.onnx.json` pair is copied into the user model folder and registered
     under the uploaded file name. For XTTS: a reference audio path is recorded.
   - Registered clone names appear in **Available TTS → Voice combo** and are usable everywhere.
5. **Recording settings**:
   - Punctuation combo: `Default (TTS default)`, `None`, `Math`, `All`.
   - Speed slider, Pitch slider, Volume slider (with value labels).
   - Edit box prefilled with sample text + **Preview** button (plays a short synthesis).
6. **Audio file creation**:
   - Radio group (single choice): **Page by page with heading style 1**,
     **Page by page only**, **Heading style 1 only**, **Break on every heading**.
   - Description text for each mode (see §3.4).
7. **Reset** — **Reset to default** button; shows confirmation "All settings will go back to
   their default values." and clears models cache info.

### 3.2 Accessibility guidelines (must-follow)

- Every interactive control has a proper **label** (wx `StaticText` immediately before the
  control, or `SetName()` + label association) so screen readers announce it.
- Menu bar with mnemonics (`&File`, `&Edit`, `&Tools`, `&Settings`, `&Help`) and accelerators.
- All dialogs use `wx.StandardDialogButtonSizer` (OK/Cancel/Apply) which is natively accessible.
- Use **combo boxes** for multi-choice selections; use **check boxes** for single booleans; use
  **radio buttons** for small single-choice sets (e.g. audio file creation modes).
- Focus lands on the first control when a dialog opens; Tab order follows reading order.
- All status messages are written to a status bar (announced by screen readers) and to logs.
- No modal dead-ends: long operations run in worker threads with a cancellable progress dialog.

### 3.3 Hardware support

As described in §2.3. The **Recording** window and **Settings** show a "Compute" combo with only
the available options (plus `Auto` when >1). Changing compute back-end at runtime re-creates the
TTS engine (cache by `(provider, model fingerprint)`).

### 3.4 Audio file creation modes & naming

Input document is prepared (parsed) and split according to the selected mode:

1. **Page by page with heading style 1** — send text page by page; when a Heading-1 is
   encountered, the text before it is sent first, then the heading is sent together with the
   remaining text of its section (so the heading is spoken at the start of its own section).
2. **Page by page only** — pages are sent one by one.
3. **Heading style 1 only** — pages are grouped by Heading-1 (each heading + its content).
4. **Break on every heading** — same grouping as mode 3, but a new file starts at
   every heading (any level 1–6); each heading is joined with the content that
   follows it up to the next heading of any level.

Naming convention (both modes 1 & 2): `01 page 1`, `02 page 2`, … ; if a Heading-1 is found in
mode 1, the file containing the heading is named after it, e.g.
`01 page 1`, `02 My Heading`, `03 page 2`. Mode 3: `01 First Heading`, `02 Second Heading`, …
Mode 4: `01 Heading One`, `02 Heading Two`, … (zero-padded index + sanitized heading text).

### 3.5 New Project wizard (File → New Project / `Ctrl+Shift+N`)

Two pages:

1. **Project page** — Project name field + **Open document** button (file dialog; supports
   PDF/TXT/DOC/DOCX/HTML/MD). Document is parsed on "Finish".
2. **Audio file creation** page — the same radio group as in Settings (§3.4) + naming preview.

Buttons: **Cancel / < Back / Next > / OK (Finish)**. On finish: prepare the document, split it
per the chosen mode, create the project folder (under `AppData\Roaming\AIVoiceStudio\projects`),
and open the **Recording** window for this project. Projects are listed in **File → Recent projects**.

### 3.6 Recording window (Tools → Record / `Ctrl+Shift+R`)

- Selectors: TTS, Language, Variant, Voice, plus **Rate, Pitch, Volume** sliders and
  **Punctuation** combo — these settings apply **only to this project** (stored in the project file).
- **Start / Pause / Resume** recording.
- Instant persistence: every synthesized segment is **saved to disk immediately** after synthesis.
  If the app is closed/crashes, the next launch finds the last saved segment and offers to
  **resume** from there (re-reading the split list, continuing at the first unsaved segment).
- Progress: `Segment 12 of 140` in the status bar + progress bar. Per-segment file names follow
  §3.4. Output format chosen in Recording window (WAV default; MP3/FLAC prompt to fetch FFmpeg).
- When FFmpeg is missing and MP3/FLAC is chosen: message box "FFmpeg is required to create
  MP3/FLAC files. Download FFmpeg now? (saved to %APPDATA%\AIVoiceStudio\ffmpeg)" — **Yes**
  downloads it (official gyan.dev build) and auto-converts; **No** switches back to WAV.

### 3.7 Project file

Each project folder contains `project.json`:

```json
{
  "name": "My Audiobook",
  "source_file": "C:\\docs\\book.pdf",
  "created": "2026-08-15T10:00:00",
  "audio_mode": "page_with_h1",
  "segments": [
    {"index": 1, "title": "01 page 1", "text": "...", "saved": "01 page 1.wav", "status": "done"}
  ],
  "tts": {"tts": "Piper", "language": "en_US", "variant": "medium", "voice": "lessac",
          "rate": 1.0, "pitch": 1.0, "volume": 1.0, "punctuation": "default",
          "output_format": "wav", "compute": "auto"}
}
```

## 4. Architecture

```
AI-voice-studio/
├── PROJECT_SPEC.md            # this document
├── DEV_PLAN.md                # phased development plan
├── README.md                  # user + developer docs
├── requirements.txt           # core deps (bundled)
├── requirements-gpu.txt       # optional GPU runtime
├── requirements-npu.txt       # optional NPU/DirectML runtime
├── main.py                    # entry point
├── ai_voice_studio/
│   ├── __init__.py
│   ├── paths.py               # user data dirs (%APPDATA%\AIVoiceStudio)
│   ├── settings.py            # persisted settings (JSON)
│   ├── constants.py           # app constants + theme enum
│   ├── compute.py             # CPU/GPU/NPU/Auto detection
│   ├── tts/
│   │   ├── catalog.py         # model catalog (embedded JSON) + helpers
│   │   ├── downloader.py      # resumable downloads (requests, Range)
│   │   ├── engine.py          # sherpa-onnx wrapper, engine cache
│   │   └── models.py          # installed-model state (downloads.json)
│   ├── documents/
│   │   ├── parsers.py         # pdf/txt/docx/html/md/clipboard → plain text
│   │   └── splitter.py        # §3.4 splitting + naming
│   ├── audio/
│   │   ├── output.py          # WAV writer (stdlib) + MP3/FLAC via ffmpeg
│   │   └── ffmpeg.py          # ffmpeg download/install to user dir
│   ├── jobs/
│   │   └── synthesizer.py     # background synthesis worker + resume logic
│   ├── gui/
│   │   ├── main_frame.py      # main window, menu, status bar
│   │   ├── settings_dialog.py # all 7 tabs
│   │   ├── model_panels.py    # Download/Remove, Available TTS, Voice clone panels
│   │   ├── new_project_wizard.py
│   │   ├── recording_dialog.py
│   │   ├── theme.py           # light/dark/system theming helpers
│   │   └── a11y.py            # accessible label helpers
│   └── util.py                # misc (sanitize filenames, wx thread helpers)
├── packaging/
│   ├── ai_voice_studio.spec   # PyInstaller spec
│   ├── installer_32.iss       # Inno Setup 32-bit
│   ├── installer_64.iss       # Inno Setup 64-bit
│   └── build.ps1              # one-command build pipeline
└── tests/
    ├── test_splitter.py
    ├── test_parsers.py
    └── test_settings.py
```

### 4.1 Threading model

- GUI thread: wxPython only.
- `SynthesisWorker` (one thread) pulls segments from the queue, calls the TTS engine
  (blocking, releases GIL in onnxruntime), writes audio immediately, posts `wx` events
  (`EVT_SYNTH_PROGRESS`, `EVT_SYNTH_DONE`, `EVT_SYNTH_ERROR`) to the UI via `wx.PostEvent`.
- `Downloader` runs per-file with progress events; cancellable.
- All engine/download calls go through a lock; engines are cached per `(provider, model)`.

### 4.2 Resilience

- Heavy third-party imports (`sherpa_onnx`, `onnxruntime`) are **lazy** so the GUI always
  starts, even when the runtime is missing — with actionable error dialogs pointing to the
  Settings panel.
- Document parser failures are per-file and reported in the log, not fatal.
- ffmpeg download is resumable; conversion failures fall back to WAV with a warning.

## 5. Packaging & installer requirements

- **32-bit and 64-bit** installable packages (Inno Setup; Python 3.13 x86 for 32-bit build,
  x86-64 for 64-bit build).
- Installer options: install location, Start Menu folder, desktop icon, "Open readme after
  install" checkbox, "Start AI Voice Studio after setup" checkbox.
- PyInstaller `--onefile`/`--onedir` spec bundling the app + `onnxruntime` + `sherpa-onnx`
  + parsers; model data lives in user dir (never in `Program Files`).
- Optional runtime add-ons downloaded by the user inside the app (GPU/NPU runtimes, FFmpeg).

## 6. Out of scope (first release)

- Streaming/listening preview while recording (single-pass generation; Preview uses the same
  engine synchronously).
- Cloud voices and commercial models.
- Non-Windows installers (app is Windows-first; code is portable in principle).
