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
- **Compute back-ends**: CPU always; **GPU (CUDA)** and **NPU (DirectML)** are
  shown automatically when present; an **Auto** option is offered when more
  than one back-end exists.
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
| Projects & audio | `%APPDATA%\AIVoiceStudio\projects\<project name>` |
| FFmpeg | `%APPDATA%\AIVoiceStudio\ffmpeg` |
| Logs | `%APPDATA%\AIVoiceStudio\logs\app.log` |

## GPU / NPU

- **GPU**: install `onnxruntime-gpu` (see `requirements-gpu.txt`) and have
  NVIDIA CUDA drivers; the app then shows "GPU (CUDA)".
- **NPU**: install `onnxruntime-directml` (see `requirements-npu.txt`); the app
  shows "NPU (DirectML)" only when an NPU-capable accelerator is detected.

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
