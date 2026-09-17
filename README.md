# AI Voice Studio

Convert documents (PDF, TXT, DOC/DOCX, HTML, Markdown, clipboard text) into
spoken **WAV / MP3 / FLAC** audio files using free, **offline neural TTS** models
that run locally through **ONNX Runtime** — no cloud, no account, full privacy.

Built with **Python + wxPython** and designed to be **fully accessible with
screen readers** (NVDA, JAWS, Narrator).

![License](https://img.shields.io/badge/license-GPL--3.0-blue)

---

## Features

- **Addon system** — install third-party TTS engines and voices as addons,
  inspired by NVDA's addon architecture. Addons can add new engines, voices,
  and capabilities. The addon development guide is available in Help menu.
- **Developer Mode** — enables heavy logging (diagnostic files for every
  subsystem), addon management, pip environment control, and diagnostic
  tools. Toggle in Settings → General.
- **Embedded Python runtime** — a managed Python virtual environment for
  addon dependencies, with pip support for installing any PyPI package.
- **Local neural TTS**: [Piper](https://github.com/rhasspy/piper) (MIT, 30+
  languages), [Kokoro-82M](https://huggingface.com/hexgrad/Kokoro-82M)
  (Apache-2.0, incl. the **multi-lang v1.0** (53 speakers) and **v1.1**
  (103 speakers) English+Chinese models, each with a smaller int8-quantized
  variant),
  [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (flow-matching,
  with its shared `vocos-22khz` vocoder downloaded automatically),
  [MMS TTS](https://huggingface.com/facebook/mms-tts) (VITS, 8 languages,
  non-commercial), and VITS **zh-AISHELL3** (174 speakers) — all downloaded
  from official open-source releases.
- **Download / Remove model manager** with TTS → language → variant → voice
  cascading combo boxes and per-file progress.
- **Voice clone tab** (XTTS v2): clone a voice from a 4–5 second WAV sample
  of the person's voice — choose a name and language, the voice becomes
  selectable everywhere and speaks 17 languages. The cloning engine (about
  2.5 GB) is an optional download into the user folder, like the GPU runtime.
- **Voice Clone category (CPU first)** — open-source voice-cloning engines
  that run on a normal **CPU** and additionally on the **GPU** when an NVIDIA
  card is detected: [Pocket TTS by
  Kyutai](https://github.com/kyutai-labs/pocket-tts),
  [Suno Bark](https://github.com/suno-ai/bark) and
  [F5-TTS](https://github.com/SWivid/F5-TTS). Each engine is installed with
  one button into the app's managed Python environment, its **pre-made
  voices** (26 Pocket TTS voices, 130 Bark speaker presets, the bundled
  F5-TTS references) appear in **Available TTS**, in the Preview lists and in
  the Recording window, and you can clone your own voice from a 5–15 second
  recording (Bark clones from a `.npz` speaker embedding).
- **Compute back-ends**: CPU always; **GPU (CUDA)** and **NPU (DirectML)** are
  shown automatically when present; an **Auto** option is offered when more
  than one back-end exists. The Voice Clone engines add their own device
  choice — CPU always, plus GPU and Auto when an NVIDIA GPU is present — both
  in Settings and per project in the Recording window.
- **A Compute combo next to every Preview button**: in **Available TTS**,
  **Recording settings**, **Punctuation** and **OmniVoice engines** you choose
  CPU / GPU / Auto right where you press Preview. The choice is remembered per
  category in Settings. **GPU** uses the whole available NVIDIA card (whatever
  card the driver exposes, no model or VRAM requirement); **CPU** uses between
  80% and 95% of the machine's logical CPUs, leaving the desktop responsive.
- **Input**: PDF, TXT, Markdown, HTML, DOCX (and DOC via Word automation),
  clipboard text. Output: WAV natively; MP3/FLAC via FFmpeg, which the app
  downloads for you into `%APPDATA%\AIVoiceStudio\ffmpeg` on first use.
- **Audio file creation modes** with the naming rules from the spec:

  | Mode | Files |
  |---|---|
  | Page by page with heading style 1 | `01 page 1`, `02 <heading>`, `03 page 2`, … |
  | Page by page only | `01 page 1`, `02 page 2`, … |
  | Heading style 1 only | `01 <heading>`, `02 <heading>`, … |
  | Break on every heading | one file per heading level 1–6, each with its content up to the next heading |
- **Recording window** (`Ctrl+Shift+R`): shows the text to record first, then
  per-project TTS, language, variant, voice, rate, pitch, volume and output
  format. Every segment is **saved to disk immediately**; if the app stops,
  reopening the project **resumes** from the first unsaved segment. The exact
  text sent to the TTS is logged to `processed_text.json` next to the audio
  files.
- **New Project wizard** (`Ctrl+Shift+N`): project name → open document →
  choose punctuation (spoken-word expansion) → choose audio mode → Finish
  opens the Recording window. Punctuation for a project is fixed here, so the
  Recording window no longer shows it.
- **Punctuation** (Settings → Punctuation tab): the default mode for new
  projects — Default (TTS decides), None (strip), Math (keep symbols) or **All**
  (speak every mark as a word: `quote`, `dot`, `left paren`, `tic`, …), with a
  live example that shows exactly how your text will be spoken.
- **Screen-reader friendly**: every control is labelled and has a proper
  accessible name; menu bar with mnemonics; OK/Cancel/Apply dialogs;
  check boxes for single choices, combo boxes for multi choices, radio buttons
  for small single-choice sets.
- **Theme**: System default / Light / Dark.- Installer (32-bit and 64-bit): **"Install for all users / for me only"** choice
  (defaults to per-user, **no admin rights needed**), install location, Start
  Menu folder choice, optional desktop icon, readme, launch-after-setup.

## Quick start (from source)

Requires Python 3.11+ on Windows.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

Then:

1. **Settings** (`Ctrl+,`) → **Download and remove** → pick
   `Piper` / `English (United States)` / `Medium quality` → **Download**.
2. **File → New Project** (`Ctrl+Shift+N`) → name it, open a document, choose
   an audio mode, press Finish.
3. In the Recording window press **Start recording**. Audio files land in the
   project folder.

## Data locations

| What | Where |
|---|---|
| Settings | `%APPDATA%\AIVoiceStudio\settings.json` |
| Downloaded voices | `%APPDATA%\AIVoiceStudio\models` |
| Cloned voices (Voice Clone) | `%APPDATA%\AIVoiceStudio\models\custom\<voice name>` |
| One virtualenv per pip-installed TTS engine | `%APPDATA%\AIVoiceStudio\tts_envs\<engine>` |
| Managed environment for addons | `%APPDATA%\AIVoiceStudio\addon_env` |
| Projects & audio | `%APPDATA%\AIVoiceStudio\projects\<project name>` |
| FFmpeg | `%APPDATA%\AIVoiceStudio\ffmpeg` |
| Logs | `%APPDATA%\AIVoiceStudio\logs\app.log` |

## GPU / NPU

- **GPU**: install `onnxruntime-gpu` (see `requirements-gpu.txt`) and have
  NVIDIA CUDA drivers; the app then shows "GPU (CUDA)".
- **NPU**: install `onnxruntime-directml` (see `requirements-npu.txt`); the app
  shows "NPU (DirectML)" only when an NPU-capable accelerator is detected.

## Voice Clone engines (CPU, plus GPU when available)

Settings → **Voice Clone** installs open-source voice-cloning engines, each
into **its own Python virtualenv** under
`%APPDATA%\AIVoiceStudio\tts_envs\<engine>`. None of them *needs* a GPU: they
all speak on the CPU, and the moment an NVIDIA GPU is detected the **Device**
selector offers **GPU (CUDA)** and **Auto** next to the CPU — the CPU option
never disappears. The Recording window offers the same device choice per
project.

### Python environments (one per TTS engine)

Every TTS engine that is installed with pip gets its **own** Python
virtualenv, so one engine's dependency pins (torch, torchaudio, transformers,
triton) can never break another's:

| Environment | Holds |
|---|---|
| `tts_envs\pocket_tts` | Pocket TTS (Kyutai) |
| `tts_envs\bark` | Suno Bark (`transformers` + `torch`) |
| `tts_envs\f5tts` | F5-TTS |
| `tts_envs\omnivoice` | **both** OmniVoice engines |
| `addon_env` | addons, the Developer tab and the sherpa-onnx GPU runtime |

The two OmniVoice engines (the direct Triton runner and the HTTP server) are
the *same model* behind two front-ends; they want the same base package and
the same CUDA PyTorch, so they deliberately share one environment instead of
duplicating several gigabytes of wheels. Engines you installed before these
per-engine environments existed keep working: the app falls back to the shared
`addon_env` until you reinstall them from their category.

### GPU support for every engine

Every Voice Clone engine supports CUDA, so each one gets the **CUDA build of
PyTorch** installed into its own environment whenever an NVIDIA GPU with a
working driver is detected (`download.pytorch.org/whl/cu128`; on Windows the
plain PyPI wheel is CPU-only). `cu128` is deliberate: the older `cu121` index
stops at Python 3.12 and has **no wheels for the Python 3.13** the app runs
on, which made pip fail with "No matching distribution found for torch". The
OmniVoice engines are GPU-only and use the same index. Engines without a GPU
keep the ordinary CPU wheels, and if a CUDA wheel is unavailable for the
installed Python the app falls back to the CPU build and tells you so — the
engine is always installable.

| Engine | Package | Pre-made voices | Rough download |
|---|---|---|---|
| Pocket TTS (Kyutai) | `pocket-tts` | 26 voices in 6 languages | ~450 MB + PyTorch |
| Bark (Suno AI, transformer) | `transformers` + `torch` | 130 speaker presets in 12 languages | ~2 GB (`bark-small`: ~700 MB) |
| F5-TTS (transformer) | `f5-tts` | 2 bundled reference voices | ~1.4 GB |

Each engine's model is downloaded from Hugging Face the first time you
synthesize with it, and stays in the Hugging Face cache afterwards.

### Which environment an engine speaks in

Each engine's worker is started with the interpreter of the engine's **own**
environment. That decision is read straight from the environment's
`site-packages` on disk (`venv_packages.package_present`) rather than from the
cached package probe the Settings pages use: the probe answers "not
installed" until its background check has finished, and a worker started in
the wrong environment fails with a baffling `ModuleNotFoundError`. (That is
exactly how Suno Bark reported "No module named 'torch'" while its own
environment had PyTorch installed.) An engine found only in the shared
`addon_env` still runs there; an engine in neither environment says so in its
own words instead.

After a successful install the app **imports the engine once** in its own
environment (`voicelab.verify_engine_import`) and reports it when that fails,
so a dependency combination the engine cannot load shows up in the install
dialog instead of on your first preview.

### Audio decoding without FFmpeg

From TorchAudio 2.9 on, `torchaudio.load`/`save` decode through **TorchCodec**
(its `backend` argument is accepted and ignored), and TorchCodec on Windows
needs FFmpeg's *shared* libraries — which the app does not install. F5-TTS
reads its reference clip that way, so previewing any F5-TTS voice failed with
"Could not load libtorchcodec". The engine worker now probes TorchCodec on a
real file first and, **only when it is genuinely broken**, backs
`torchaudio.load`/`save` with `soundfile` (whose wheel contains libsndfile, so
nothing has to be installed on the computer). WAV, FLAC, OGG and MP3 work that
way; a machine that has FFmpeg keeps TorchCodec and its wider format support.
F5-TTS also cleans the reference clip with pydub, which shells out to FFmpeg
for anything that is not a WAV, so a non-WAV reference (MP3/FLAC/OGG — the
formats the Voice Clone page offers) is converted to a WAV once with
soundfile before the engine sees it.

Cloning works from a short recording of the voice you want (5–15 seconds of
clean speech) plus, optionally, a transcript of it; the reference is copied
into the voice folder, so the clone keeps working after you move the original
file. Bark is the exception: it clones from a speaker-embedding `.npz` file
and otherwise uses one of its 130 speaker presets.

Pocket TTS, Bark and F5-TTS run in a separate worker process, so PyTorch
never loads inside the application itself.

### Per-engine tuning

Every engine has its own generation knobs, reachable from two places:

* **Settings → Voice Clone → Engine tuning...** saves the *defaults* of the
  selected engine (used by every project that has not changed them),
* **Recording window → Engine tuning...** saves *per-project overrides*; the
  line next to the button shows which values the project will use, and where
  they come from.

| Engine | Tuning options |
|---|---|
| Pocket TTS | int8 quantization (CPU speed-up), seed |
| Bark | text temperature, waveform temperature, seed |
| F5-TTS | diffusion steps (NFE), CFG strength, sway sampling, cross-fade, target RMS, fixed duration, trim silence, seed |

A control always starts at the engine's own default and **only values that
differ from it are stored and sent**, so a project you never tune behaves
exactly like a freshly installed engine, and *Reset to engine defaults* is the
same as "let the engine decide". Parameters the installed version of an engine
does not support (the engines rename them between releases) are skipped, logged
and never fail a recording.

> Kyutai's *voice-cloning* Pocket TTS weights live behind a Hugging Face
> repository that may require accepting its terms and logging in
> (`huggingface-cli login`) before the first synthesis.

## Packaging

See `packaging/` — `build.ps1` drives PyInstaller (per-arch) and Inno Setup to
produce `AI-Voice-Studio-Setup-32.exe` / `-64.exe`. The installer offers the
same wizard widgets as the ProgramLauncher reference project: a
"for all users / for me only" radio on the directory page
(`PrivilegesRequired=lowest` + `PrivilegesRequiredOverridesAllowed=dialog`,
so per-user installs need **no administrator rights**), a Start Menu folder
choice, optional desktop icon, license page, and launch/readme options.
Models are never bundled; they are downloaded by the user in-app.

## License

GPL-3.0-or-later (see `LICENSE`). TTS models keep their own licenses:
Piper = MIT, Kokoro-82M = Apache-2.0, MMS = CC-BY-NC (non-commercial).
The application bundles `sherpa-onnx` (Apache-2.0) and `onnxruntime` (MIT).

## Author

Developed by **Anuj Sharma**.

## Acknowledgements

Inspired by accessible reading software such as
[Bookworm](https://github.com/blindpandas/bookworm), and built on
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [Piper](https://github.com/rhasspy/piper),
and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M).
