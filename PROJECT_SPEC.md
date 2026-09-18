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
- Compute back-ends: **CPU, GPU (CUDA)** with auto-detection; unavailable back-ends are
  hidden from the UI. (DirectML/NPU was planned but is never offered — sherpa-onnx ships no
  DirectML execution provider to load.)

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
### 2.2.1 The engines the catalog now ships (2026.3.1)

`models_catalog.json` describes **nine** families (Piper, Kokoro-82M, Kitten TTS, VITS
zh-AISHELL3, Matcha-TTS, OmniVoice, OmniVoice Server, SAPI5 and the Windows Core voices)
and `tts/catalog.get_tts_list()` appends the **Voice Lab** engines, whose metadata lives
next to their built-in voices in `voicelab/engines.py`:

| Engine | id | Install package(s) | Licence |
|---|---|---|---|
| Pocket TTS (Kyutai) | `pocket_tts` | `pocket-tts`, soundfile, scipy | Apache-2.0 (per-voice terms) |
| Bark (Suno AI) | `bark` | `transformers`, `torch`, soundfile, scipy | MIT |
| F5-TTS | `f5tts` | `f5-tts`, `torch`, `torchaudio`, soundfile | MIT code, **CC-BY-NC-4.0 weights** |
| OmniVoice | `omnivoice` | `omnivoice-triton` (+ PyTorch, CUDA) | Apache-2.0 |
| OmniVoice Server | `omnivoice_server` | `omnivoice-server` (+ PyTorch, CUDA) | Apache-2.0 |

Voice cloning (Settings → Voice Clone) is provided by the Voice Lab engines and by
OmniVoice; **Coqui XTTS v2 was retired** in favour of them (it was the cloning engine of
earlier releases and needed a ~2.5 GB engine download plus a ~2.1 GB model on first use).
Piper fine-tuned voice uploads (an `.onnx` + `.onnx.json` pair trained on the user's own
voice) remain a valid fully offline "clone" workflow. Engines are flagged in the catalog
with `requires_package` when they are installed as Python packages rather than downloaded
as ONNX artifacts; the panels turn that into an installed / not-installed state.

### 2.3 Compute providers

| Back-end | ONNX Runtime package | Execution providers |
|---|---|---|
| CPU | `onnxruntime` (bundled) | `CPUExecutionProvider` |
| GPU | `onnxruntime-gpu` (user-installable at setup/update) | `CUDAExecutionProvider`, `TensorrtExecutionProvider` |
| NPU (DirectML) | not shipped | none — sherpa-onnx has no DirectML provider to load, so the option is deliberately absent |

Detection rules (current implementation):

- **CPU** is always available.
- **GPU** is offered only if a CUDA-capable provider is present (check `onnxruntime.get_available_providers()`
  and/or presence of `onnxruntime-gpu` import + `nvidia-smi`).
- **NPU (DirectML)** is **never** offered: sherpa-onnx has no DirectML execution provider to
  load, so an NPU entry would be a choice that cannot run. `compute.py` documents this and
  the combo never lists it.
- If more than one compute option is available, an **Auto** entry is added (preference order:
  GPU → CPU).

**How much compute is used** (whatever back-end is selected):

- **GPU** uses the whole available NVIDIA card. Any CUDA-capable card the driver exposes is
  accepted - there is no model, VRAM or compute-capability requirement - and the run is tuned for
  maximum throughput: the first (usually only) CUDA device, cuDNN's autotuner, the TensorFloat-32
  fast paths, `high` float32 matmul precision and the autograd engine switched off.
- **CPU** uses between **80% and 95%** of the machine's logical CPUs (`compute.cpu_threads()`),
  so a long recording is fast while the desktop, the Recording window and the audio writer stay
  responsive. Machines with fewer than four logical CPUs keep one core free instead; a
  single-core machine uses its one core. The value is passed to every back-end - sherpa-onnx's
  `num_threads` and `torch.set_num_threads` in the pip-installed engines' worker - so the same
  budget applies everywhere.

**A Compute combo with every Preview button**: *Available TTS*, *Recording settings*,
*Punctuation* and *OmniVoice engines* each show a small **Compute** combo (CPU, GPU when an
NVIDIA GPU is detected, Auto when there is a choice) next to their Preview button. The choice is
remembered per category in Settings (`preview_compute.<category>`) and is resolved to the
provider the preview passes to `get_engine`.

## 3. Functional requirements

### 3.1 Settings dialog (File → Settings / `Ctrl+,`)

A tabbed dialog with **OK / Cancel / Apply** buttons. The categories, in the order
they appear (`SettingsDialog.CATEGORIES`):

| # | Category class | Title |
|---|---|---|
| 1 | `_GeneralPanel` | General |
| 2 | `DownloadPanel` | Download and remove |
| 3 | `AvailablePanel` | Available TTS |
| 4 | `VoiceClonePanel` | Voice clone (the Voice Lab engines) |
| 5 | `_OmniVoiceEnginesPanel` | OmniVoice engines |
| 6 | `_OmniVoiceServerPanel` | OmniVoice Server |
| 7 | `_RecordingSettingsPanel` | Recording settings |
| 8 | `_PunctuationPanel` | Punctuation |
| 9 | `_AudioModePanel` | Audio file creation |
| 10 | `_DaisySettingsPanel` | DAISY settings |
| 11 | `_ComputePanel` | Compute |
| 12 | `_DeveloperPanel` | Developer (only with Developer Mode) |
| 13 | `_ResetPanel` | Reset |

Details:

1. **General** — Theme combo box: `System default`, `Light`, `Dark` (applies live),
   the recordings and models folders (each with Browse...), and Enable Developer Mode,
   which reveals the Developer category and heavy logging.
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
4. **Voice Clone** (Pocket TTS by Kyutai, Suno Bark, F5-TTS) — the CPU-first clone
   engines:
   - 1. **Engine** combo + **Install engine... / Remove engine** (pip into the engine's *own*
     virtualenv, `%APPDATA%\AIVoiceStudio\tts_envs\<engine>`) with the install status
     underneath.
   - 2. **Device** combo: **CPU always**; **GPU (CUDA)** and **Auto** are added when an
     NVIDIA GPU is detected (the CPU is never removed).
   - 3. **Voice**: *Use a built-in voice* (the engine's pre-made voices) or *Clone a voice
     from a sample* (reference recording + optional transcript + voice name → **Create
     cloned voice**; Bark takes a `.npz` speaker embedding).
   - 4. **Voices ready to use**: every built-in and cloned voice, with **Preview selected
     voice** and **Delete cloned voice**.
   - **Engine tuning...** edits the generation defaults of the selected engine (diffusion
     steps, CFG strength, sampling temperatures, int8 quantization, seed, ...) and saves
     them in Settings (`clone_engines.options.<engine>`); each project can override them.
     A control starts at the engine's own default and only values that differ from it are
     stored, so an untouched engine keeps its documented behaviour.
   - Cloned voices are stored in the model store (`kind=clone_reference`) and appear in
     **Available TTS**, the Preview cascades and the Recording window.
   - Built-in voices of installed engines are injected into those panels too, so they can
     be used everywhere without extra configuration.
5. **Voice clone (uploaded model)**:
   - Combo 1: TTS, Combo 2: Language, Combo 3: Variant (only entries of cloning-capable TTS).
   - **Upload voice** button → file dialog for audio sample (`.wav/.mp3/.flac/.onnx`).
   - For Piper: a `.onnx` + `.onnx.json` pair is copied into the user model folder and registered
     under the uploaded file name. For the Voice Lab engines: a reference audio path is
     recorded (this entry is legacy - the XTTS cloner was retired).
   - Registered clone names appear in **Available TTS → Voice combo** and are usable everywhere.
6. **Recording settings**:
   - Punctuation combo: `Default (TTS default)`, `None`, `Math`, `All`.
   - Speed slider, Pitch slider, Volume slider (with value labels).
   - Edit box prefilled with sample text + **Preview** button (plays a short synthesis).
7. **Audio file creation**:
   - Radio group (single choice): **Page by page with heading style 1**,
     **Page by page only**, **Heading style 1 only**, **Break on every heading**.
   - Description text for each mode (see §3.4).
8. **Reset** — **Reset to default** button; shows confirmation "All settings will go back to
   their default values." and clears models cache info.
9. **Compute** — detected CPU/GPU capability, and download/remove of the optional ONNX GPU
   runtime (about 1.8 GB). Restart requested after a change.
10. **Developer** — heavy logging, addon install/enable/disable/remove, the addon Python
    environment (pip install, package listing) and diagnostics. Hidden without Developer Mode.
11. **DAISY settings** — language, publisher and whether the DAISY 3 book embeds the text.

Each category that shows a **Preview** button also carries a **Compute** combo; the choice is
stored per category under `preview_compute.<category>`.

### 3.1.1 First launch — terms of use

Before any subsystem is initialised (`main.py` → `gui.accept_dialog.require_terms_acceptance`),
the application shows a **Terms of Use** dialog on the first launch and again after an update
whose terms text changed:

- Six check boxes state the conditions of use: no voice cloning without rights or prior
  permission; no illegal use; commercial users must read each TTS licence and use only engines
  that permit commercial use (or have written permission); use only SAPI5 / Windows Core voices
  the user has the right to use; the author provides no warranty and accepts no responsibility
  for legal or illegal use; and voice cloning is a normal feature of modern neural TTS that the
  user will use carefully and legally.
- A final, bold check box **"I agree with all above terms"** is the master switch: ticking it
  ticks the six conditions, unticking it clears them.
- **I Agree** and **Disagree (Close Program)** stay disabled until that final check box is ticked.
  Escape is swallowed at the key-hook level, and closing the window counts as Disagree.
- The application starts **only** from the I Agree path. Acceptance is stored as
  `terms_version` (compared with `constants.TERMS_VERSION`), so it is remembered until the next
  version whose terms changed, a fresh profile or a reinstall.

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

### 3.3.1 Python environments (one per TTS engine)

Every TTS engine that is installed with pip runs from its **own** Python virtualenv
(`%APPDATA%\AIVoiceStudio\tts_envs\<engine>`, created and managed by
`python_runtime.PythonRuntime`), so conflicting dependency versions (torch, torchaudio,
transformers, triton) can never break another engine:

| Environment | Holds |
|---|---|
| `tts_envs\pocket_tts` | Pocket TTS (Kyutai) |
| `tts_envs\bark` | Suno Bark (`transformers` + `torch`) |
| `tts_envs\f5tts` | F5-TTS |
| `tts_envs\omnivoice` | **both** OmniVoice engines (direct + HTTP server) |
| `addon_env` | addons, the Developer tab and the sherpa-onnx GPU runtime |

Rules:

- The engine's id names the environment (`python_runtime.environment_id`);
  `SHARED_ENVIRONMENTS` maps the one deliberate exception - `omnivoice_server` onto
  `omnivoice` - because the two OmniVoice engines are the same model behind two front-ends and
  want the same base package and the same CUDA PyTorch.
- The engine's packages are installed into that environment only; each engine is driven by
  `voicelab.worker` (or the OmniVoice worker) running on that environment's interpreter, so
  PyTorch/CUDA never enters the GUI process.
- `venv_packages` probes are cached per *environment*, so the two OmniVoice engines share one
  answer and installing either refreshes both. Because the probe answers only once its background
  check has finished, the cached probe is for the **GUI** (labels, enabled buttons) and never for
  a decision that must be right immediately: `venv_packages.version`/`installed` fall back to a
  direct listing of the environment's `site-packages` (`venv_packages.package_present`), and
  `voicelab.CloneWorker._runtime` uses that same listing to choose which interpreter an engine's
  worker is started with. Without it the first preview after an install started the worker in the
  shared environment and Bark failed with `ModuleNotFoundError: No module named 'torch'` although
  its own environment had PyTorch. When neither environment holds the engine the error says so in
  the engine's own words instead of naming a missing module.
- The sherpa-onnx ONNX engines (Piper, Kokoro, Kitten, Matcha, VITS, the Windows system voices)
  ship with the application and are deliberately unchanged: they keep using the bundled runtime.
- Engines installed before the per-TTS environments existed are still found in the shared addon
  environment (`python_runtime.engine_runtime` prefers the engine's own environment and falls
  back to the shared one).
- **GPU support for every engine**: each Voice Clone engine takes the CUDA build of PyTorch from
  the official PyTorch wheel index `voicelab.engines.PYTORCH_CUDA_INDEX` when an NVIDIA GPU is
  detected (PyPI's plain `torch` wheel is CPU-only on Windows); the OmniVoice installers use the
  same constant, so every CUDA wheel comes from one index. The index must carry wheels for the
  Python the app runs on - `cu121` stops at Python 3.12 and made pip fail with "No matching
  distribution found for torch" on Python 3.13, which is why the index is `cu128`.
- If the CUDA wheel index cannot satisfy an install (no build for this interpreter, or the mirror
  is unreachable) the engine installer falls back to the plain PyPI packages, so the engine is
  still installed and usable on the CPU, with the GPU limitation reported to the user.
- **Install verification**: after the pip steps succeed the app imports the engine's module(s)
  (`voicelab.verify_engine_import`, using `engines.import_modules`) in the engine's own environment
  and reports a failure in the install dialog. pip succeeding is not the same as a working engine;
  this is what turns a dependency-resolution mistake into an immediate, explained message. The same
  `engines.import_modules` list decides which environment the worker is started with, so it is the
  single description of "what this engine needs". Package names are plain names only: the same list
  feeds "Remove engine", and a version specifier there would make `pip uninstall` fail.
- **Audio decoding without FFmpeg**: from TorchAudio 2.9 on `torchaudio.load`/`save` decode through
  TorchCodec (its `backend` argument is accepted and ignored), which on Windows needs FFmpeg's
  shared libraries - not something the app installs. The Voice Lab worker probes TorchCodec with a
  real file (`_torchcodec_works`) and only when it is broken replaces `torchaudio.load`/`save` with
  soundfile-backed equivalents (`_install_audio_io_shim`, opt-out via
  `AIVS_VOICELAB_NO_AUDIO_SHIM=0`); soundfile ships libsndfile in its wheel, so WAV/FLAC/OGG/MP3
  decode with nothing installed, and a machine with FFmpeg keeps TorchCodec's wider format support.
  F5-TTS additionally cleans its reference clip with pydub, which shells out to FFmpeg for non-WAV
  input, so a non-WAV reference is converted to a WAV once with soundfile before the engine runs.

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
- **Engine tuning...** (shown for the Voice Clone engines) stores this project's generation
  overrides under `tts.engine_options` (`{"engine": id, "values": {...}}`); a project without
  its own overrides uses the Settings defaults for that engine. The summary next to the button
  names the active values and where they come from, and unsupported ones are reported.
- **Start / Pause / Resume** recording.
- Instant persistence: every synthesized segment is **saved to disk immediately** after synthesis.
  If the app is closed/crashes, the next launch finds the last saved segment and offers to
  **resume** from there (re-reading the split list, continuing at the first unsaved segment).
- Progress: `Segment 12 of 140` in the status bar + progress bar. Per-segment file names follow
  §3.4. Output format chosen in Recording window (WAV default; MP3/FLAC prompt to fetch FFmpeg).
- When FFmpeg is missing and MP3/FLAC is chosen: message box "FFmpeg is required to create
  MP3/FLAC files. Download FFmpeg now? (saved to %APPDATA%\AIVoiceStudio\ffmpeg)" — **Yes**
  downloads it (official gyan.dev build) and auto-converts; **No** switches back to WAV.

### 3.6.1 Start Selected Recording (Edit menu)

`Edit → Start Selected Recording...` opens a dialog whose sidebar first decides *what the
list means*, then what to do with the choice:

| Sidebar choice | Combo label | Combo lists |
|---|---|---|
| **Select an audio file** (default) | "Choose which recorded file to record again:" | only the project's recorded audio files |
| **Select by file break** | "Choose which file to record by file break:" | every file break of the project, recorded or not |

and a second pair decides the scope:

- **Only record selected file** (default) — records that one file and stops; the following
  file is not touched whether or not it was already recorded.
- **Record all files from here** — records the chosen file and every file after it to the end
  of the project (the previous behaviour of "restart selected recording", now opt-in).

Mechanics: the chosen recording is deleted (its segment returns to `pending`), the
`RecordingDialog` opens with the chosen start position and (for the single-file scope) an
exclusive `end_index`, so `jobs.synthesizer.SynthesisWorker` stops after that segment. The
dialog **never** refuses to open for lack of recordings; with none, it starts in the
by-file-break mode, so one chapter of a long book can be recorded before chapter 1. The only
blocked case is a project whose document was never split into file breaks.

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
├── requirements-npu.txt       # legacy record; DirectML/NPU is never offered
├── main.py                    # entry point
├── ai_voice_studio/
│   ├── __init__.py
│   ├── paths.py               # user data dirs (%APPDATA%\AIVoiceStudio)
│   ├── settings.py            # persisted settings (JSON)
│   ├── constants.py           # app constants + theme enum
│   ├── compute.py             # CPU/GPU detection, NVIDIA probe, CPU thread budget (80-95%)
│   ├── python_runtime/        # managed virtualenvs (one per TTS engine + addon_env)
│   ├── venv_packages.py       # cached per-environment package probes
│   ├── tts/
│   │   ├── catalog.py         # model catalog (embedded JSON) + helpers
│   │   ├── downloader.py      # resumable downloads (requests, Range)
│   │   ├── engine.py          # sherpa-onnx wrapper, engine cache
│   │   └── models.py          # installed-model state (downloads.json)
│   ├── voicelab/
│   │   ├── engines.py         # CPU/GPU clone engines + their built-in voices
│   │   ├── options.py         # per-engine tuning vocabulary (defaults, ranges)
│   │   ├── worker.py          # standalone worker subprocess (PyTorch lives here)
│   │   └── __init__.py        # device choice, worker client, VoicelabEngine, clones
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
│   │   ├── settings_dialog.py # all settings categories
│   │   ├── clone_engines_panel.py # Voice Clone category (CPU/GPU clone engines)
│   │   ├── voicelab_options_dialog.py # per-engine tuning dialog (project/defaults)
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

#### 4.0.1 Current module map (2026.3.1)

The tree above is the original layout. The application has since grown the following
modules; this is the authoritative map:

```
AI-voice-studio/
├── main.py                    # entry point: terms gate, then the GUI
├── ai_voice_studio/
│   ├── constants.py           # APP_VERSION, TERMS_VERSION, theme + format constants
│   ├── paths.py               # %APPDATA%\AIVoiceStudio layout (projects, models, tts_envs, logs)
│   ├── settings.py            # persisted settings (defaults, reset, recent projects)
│   ├── project.py             # project.json read/write, segments, saved files
│   ├── compute.py             # NVIDIA probe, CPU thread budget (80-95%), provider choice
│   ├── runtime.py             # sherpa-onnx / onnxruntime loading and provider setup
│   ├── heavy_logging.py       # per-subsystem logs + crash log (Developer Mode)
│   ├── venv_packages.py       # cached per-environment package probes
│   ├── python_runtime/        # managed virtualenvs: one per TTS engine + addon_env
│   ├── addons/                # addon discovery, manifest validation, registration
│   ├── tts/
│   │   ├── catalog.py         # JSON catalog + helpers; appends the Voice Lab engines
│   │   ├── models_catalog.json# nine ONNX / built-in TTS families
│   │   ├── downloader.py      # resumable downloads
│   │   ├── models.py          # installed-model state
│   │   ├── engine.py          # sherpa-onnx wrapper + engine cache
│   │   └── windows_tts.py     # SAPI5 and Windows Core voice enumeration
│   ├── voicelab/              # Pocket TTS, Bark, F5-TTS
│   │   ├── engines.py         # registry, built-in voices, PYTORCH_CUDA_INDEX
│   │   ├── options.py         # per-engine tuning vocabulary
│   │   ├── worker.py          # standalone worker subprocess (torch lives here)
│   │   └── __init__.py        # device choice, worker client, VoicelabEngine, verify_engine_import
│   ├── omnivoice/             # direct OmniVoice engine (spec, worker, voice store)
│   ├── omnivoice_server/      # OpenAI-compatible OmniVoice HTTP server manager
│   ├── documents/             # parsers, splitter, playlist builder, DAISY 2.02 / 3 builders
│   ├── audio/                 # WAV writer, MP3/FLAC conversion, FFmpeg download
│   ├── jobs/synthesizer.py    # background synthesis worker (+ start/end range)
│   └── gui/
│       ├── main_frame.py      # menus, recent projects, Start Selected Recording picker
│       ├── settings_dialog.py # all thirteen categories
│       ├── model_panels.py    # Download/Remove, Available TTS
│       ├── clone_engines_panel.py # Voice Clone (Voice Lab engines)
│       ├── compute_choice.py  # the per-category Preview Compute combo
│       ├── accept_dialog.py   # first-launch terms of use
│       ├── voicelab_options_dialog.py / omnivoice_options_dialog.py
│       ├── new_project_wizard.py, recording_dialog.py, dialogs.py, progress.py
│       ├── theme.py, a11y.py, events.py
├── docs/                      # README/UserGuide/Addon/Accessibility/THIRD-PARTY-LICENSES + book/
├── packaging/                 # PyInstaller spec, installer_common.iss, installer_32/64.iss, build.ps1
├── tools/                     # a11y audits, end-to-end probes, book PDF builder
└── tests/                     # unit, GUI smoke, engine-environment and book guards (22 files)
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
  + parsers + the `docs` folder (including the book); model data lives in user dir (never in
  `Program Files`). Built output: `dist\AI-Voice-Studio-v-<version>-Setup-x64.exe`
  (or `-x86`); the version lives in `installer_common.iss` and must match
  `constants.APP_VERSION`.
- Optional runtime add-ons downloaded by the user inside the app (the ONNX GPU runtime, FFmpeg,
  and the per-engine Python environments with their PyTorch/CUDA wheels).

## 6. Out of scope (first release)

- Streaming/listening preview while recording (single-pass generation; Preview uses the same
  engine synchronously).
- Cloud voices and commercial models.
- Non-Windows installers (app is Windows-first; code is portable in principle).
