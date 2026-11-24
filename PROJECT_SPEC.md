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
| **Supertonic 3** | OpenRAIL | 31 (ar, bg, cs, da, de, el, en, es, et, fi, fr, hi, hr, hu, id, it, ja, ko, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi) | one int8 archive for every language, 10 voice styles (sid 0–4 = F1–F5, sid 5–9 = M1–M5) | No | `github.com/k2-fsa/sherpa-onnx` releases (one 128 MB archive, downloaded once into `models/shared/supertonic-3`) |
| **MMS TTS** | CC-BY-NC-4.0 (note: non-commercial) | 1100+ languages | `eng`, `deu`, `fra`, `spa`… | No | sherpa-onnx releases |
### 2.2.1 The engines the catalog now ships (2026.3.1)

`models_catalog.json` describes **ten** families (Piper, Kokoro-82M, Kitten TTS, VITS
zh-AISHELL3, Matcha-TTS, Supertonic 3, OmniVoice, OmniVoice Server, SAPI5 and the Windows
Core voices)
and `tts/catalog.get_tts_list()` appends the **Voice Lab** engines, whose metadata lives
next to their built-in voices in `voicelab/engines.py`:

| Engine | id | Install package(s) | Licence |
|---|---|---|---|
| Pocket TTS (Kyutai) | `pocket_tts` (voice cloning needs the gated `kyutai/pocket-tts` weights) | `pocket-tts`, soundfile, scipy | Apache-2.0 (per-voice terms) |
| F5-TTS | `f5tts` | `f5-tts`, `torch`, `torchaudio`, soundfile | MIT code, **CC-BY-NC-4.0 weights** |
| KokoClone | `kokoclone` | `kokoclone`, `kanade-tokenizer` (pinned source archives) | Apache-2.0 |
| Chatterbox | `chatterbox` | `chatterbox-tts`, `torch`, `torchaudio`, soundfile | MIT |
| Qwen3-TTS | `qwen3tts` | `qwen-tts`, `torch`, `torchaudio`, soundfile | Apache-2.0 |
| VibeVoice | `vibevoice` | `vibevoice` (pinned source archive), `torch`, soundfile | MIT |
| OmniVoice | `omnivoice` | `omnivoice-triton` (+ PyTorch, CUDA) | Apache-2.0 |
| OmniVoice Server | `omnivoice_server` | `omnivoice-server` (+ PyTorch, CUDA) | Apache-2.0 |
| Indic Parler-TTS | `indic_parler` | `parler-tts` (+ PyTorch - no compiled front-end) | Apache-2.0 |
| Kitten TTS 2.0 | `kitten_tts_2` | `kittenml` (+ PyTorch) | Apache-2.0 code; **Stellon Labs Community License weights** |

Kitten TTS 2.0 is a **second** entry beside the ONNX `kitten` engine, not a new variant of
it: it is KittenML's 1.7B speech language model driven from the `kittenml` package, it has
its own voices (47 built-in plus cloning), its own ten languages and its own environment
(`tts_envs\kitten_tts_2`), and the ONNX entry is untouched by it.

The Indian-language engine (`indic_parler`) ships in 2027.1.0 as the installable half of
a survey of Hindi/Indic TTS options (checked 2026-10-13); the half that is not installable
is recorded in §2.2.3.


**The engine was installed into its own environment and made to speak before it was
documented** (`tools/e2e_voice_lab_engine.py` runs one segment through the app's own path
- voice entry, `VoicelabEngine`, the engine's interpreter, its worker and its model
download - and reports the sample rate, the length, the real-time factor, the RMS, and
the **speech-versus-drone verdict the application itself uses**: a take with no speech in
it is *loud*, so levels alone cannot tell "it produced a file" from "it produced speech",
which is how a booming engine passed this probe once).  The probe then **transcribes that
take with the application's own Whisper large-v3** (the same `stt_envs\whisper_large_v3`
environment, started through `ai_voice_studio.stt.engine` exactly as a recording is) and
compares the words with the text the engine was given: a word error rate above **0.35**
fails the run, as does a take whose words could not be checked at all, so speech-shaped
noise cannot pass.  The limit is measured, not chosen - the engines that spoke the text
sit at 0.00-0.23 and the ones that only made noise at 0.52-1.00.  The rate is printed
with the decoding setting it was measured at (the probe's default is `balanced`,
`--asr-accuracy fast` is the quickest), so a figure is never read against another one
taken differently. `indic_parler` 0.2.3 (5.11 s of Hindi, 44.1 kHz, 68 s on an RTX 3060)
installs and imports cleanly on Python 3.13/Windows, and draws its weights from a
**gated** repository, so a run needs a Hugging Face token with the licence accepted
(§3.4).

**Kitten TTS 2.0 fills a GPU the same way, and its cost per segment was measured before
it was declared** (`tools/measure_engine_vram.py kitten_tts_2`, 2026-10-16, 12 GB RTX
3060).  The tool drives the recording path itself - the catalogue voice entry, the
app-facing `VoicelabEngine`, the shared worker pool, the engine's own environment and its
worker subprocess - and samples `nvidia-smi` around it, so the figure is the driver's
video memory, not PyTorch's allocator.  It reads the card in four states (idle, loading,
loaded and quiet, generating) and finds the cost of a copy from them.  One copy of the
1.7B checkpoint loading and generating peaked at **6115 MiB** over an idle 1207 MiB
(the checkpoint is ternary at 0.95 GB on disk but is materialised at bf16, and the codec
and the process's own CUDA context come on top); with two copies resident and both
generating at once the peak was **10645 MiB** over an idle 1216 MiB, **4714 MiB a copy**,
and the same two segments took **45.8 s of wall time against 87.8 s through one copy**
(1.9x) - this engine generates at roughly a fifth of real time, so the second slot is
worth several minutes on a book.  A third copy asks for about 15 GB and
was not demonstrated: on the 12 GB card it saturates the card and starts spilling to host
memory instead of failing, which is why the declared engine ceiling (3, as for the other
one-text-per-call engines) is documented as a ceiling rather than as a measurement.  The
declaration that shipped is `mode: instances`, `max_slots: 3`, `vram_mb: 5000` - the
dearest of the two runs, rounded up, so a slot is never charged less than the card was
seen to use - and the plan the application settles on follows from it: one copy on a
12 GB card (a second fits the card but not the budget a recording leaves after its 1.5 GB
reserve), two on a 16 GB card, three from 24 GB up.  `tests/test_kitten_tts2.py` keeps the
declared figure tied to these runs, and `tests/test_measure_engine_vram.py` holds the
tool's arithmetic to account.

**Two faults of the engines' own dependencies were found by running them, and both are
fixed in the install plan rather than left as a note to the user.** `parler-tts`
declares `protobuf>=4.0.0`, but `descript-audiotools` - the audio-tools half of the DAC
decoder the checkpoint decodes its audio with, which the plan installs by name - still
caps protobuf at `<3.20`; on 3.19.6 the engine did not start at all, because the
`sentencepiece` that comes with it ships generated code wanting
`google.protobuf.internal.builder`, which 3.19 has not got (*"cannot import name
'builder'"*). Nothing in `audiotools` or `dac` imports protobuf at all, so the engine's
own requirement wins: `protobuf>=4.25.0` is installed as a final step. 









Voice cloning (Settings → Voice Clone) is provided by the Voice Lab engines and by
OmniVoice; **Coqui XTTS v2 was retired** in favour of them (it was the cloning engine of
earlier releases and needed a ~2.5 GB engine download plus a ~2.1 GB model on first use).
Piper fine-tuned voice uploads (an `.onnx` + `.onnx.json` pair trained on the user's own
voice) remain a valid fully offline "clone" workflow. Engines are flagged in the catalog
with `requires_package` when they are installed as Python packages rather than downloaded
as ONNX artifacts; the panels turn that into an installed / not-installed state.

### 2.2.2 Deliberately not offered: Voxtral 4B TTS (checked 2026-10-08)

`mistralai/Voxtral-4B-TTS-2603` — Mistral AI's 4B text-to-speech model (24 kHz
mono, 20 preset voices, 9 languages, CC-BY-NC-4.0) — is **not** a Voice Lab
engine or a catalog entry, and that is a recorded decision so it is not
attempted again. Four verified reasons:

1. **No runtime this application can install.** The model card names vLLM-Omni
   as its only supported library; `transformers` answers *"the checkpoint you
   are trying to load has model type `voxtral_tts` but Transformers does not
   recognize this architecture"*, and there is no ONNX export, no GGUF and no
   pip package. **vLLM and vLLM-Omni publish no Windows wheels** (checked
   against PyPI's complete file index: 0 Windows wheels out of 209 and 24), so
   the per-engine environment story (`tts_envs\<engine>`) cannot hold it on
   Windows; upstream's own answer for Windows is WSL2.
2. **No CPU path.** vLLM is a CUDA inference server, so unlike the Voice Lab
   engines (whose CPU run is merely slow) there is nothing to offer a machine
   without an NVIDIA card.
3. **Above the video-memory ceiling.** The model card asks for 16 GB or more;
   the published runs used 20.5 GiB (RTX 4090 24 GB) and 27.3 GiB (RTX 5090
   32 GB). Every engine this catalog ships is held to **under 12 GB**.
4. **Cloning is gated, and emotions are voices.** The open-weights checkpoint
   omits the codec's encoder, so a reference-audio request fails with
   `encode_waveforms requires encoder weights which are not available in the
   open-source checkpoint`; and the model accepts no emotion tags or text
   instructions (its paper says emotion is steered by choosing a voice prompt),
   so an Emotion control could only pick among its preset voices, of which
   `neutral`, `casual` and `cheerful` carry an emotion.

The Voxtral **speech-to-text** model (Voxtral Mini 3B, `transformers`) is
unaffected and stays in the STT catalog.

### 2.2.3 Deliberately not offered: Svara-TTS and AI4Bharat Indic-TTS (checked 2026-10-13)

Two Hindi/Indic synthesis models were surveyed alongside the three that were added
(§2.2.1). Both were rejected for a reason that a per-engine virtualenv cannot fix,
and that is recorded here so they are not attempted again.

**Svara-TTS** (`kenpath/svara-tts`, ~150M, LoRA fine-tune of `Qwen/Qwen3-0.6B-Base`)
has **no runtime that runs on Windows**. Its own README names vLLM-Omni as the
serving library, and that stack (vLLM + FlashInfer) publishes Linux wheels only, so
`tts_envs\svara_tts` could not hold it here. The kernel path is not a way out either:
there is no `llama-cpp-python` **cp313 Windows** wheel, and this application installs
engines by plain `pip install` into a venv built from its own Python 3.13 interpreter
(§3.3.1), so a source build is not available to it either.

**AI4Bharat Indic-TTS** (`AI4Bharat/Indic-TTS`, FastPitch + HiFi-GAN v1) is
**installable but not worth an engine slot**: its text front-end needs `pynini`, which
publishes **no Windows wheel on PyPI** (it is distributed through conda), and its
pinned harness needs `numba==0.56.2`, which has no cp313 wheel. Removing the front-end
and feeding it a hand-written g2p would leave a model whose own repository is archived,
whose checkpoints are v1, and which brings no capability the shipped engines lack —
whereas the same text front-end goal is already met by Indic Parler-TTS
(which takes text directly, needing no pynini). The rejection is about the cost of
maintaining the front-end, not about audio quality.

Indic **Parler-TTS** is the one that survived: it installs from PyPI into its own
environment on Python 3.13/Windows and runs on an NVIDIA GPU (its 0.9B transformer is far
too slow on a CPU to offer).

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
*Punctuation* and *Voice Clone* (the OmniVoice voice library) each show a small **Compute** combo (CPU, GPU when an
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
| 2 | `_AppearancePanel` | Appearance |
| 3 | `DownloadPanel` | Download and remove ONNX TTS |
| 4 | `AvailablePanel` | Available TTS |
| 5 | `_VoiceCloneCategoryPanel` | Voice Clone (Voice Lab engines + OmniVoice engines, one page) |
| 6 | `_OmniVoiceServerPanel` | OmniVoice Server |
| 7 | `_RecordingSettingsPanel` | Recording settings |
| 8 | `_PunctuationPanel` | Punctuation |
| 9 | `_AudioModePanel` | Audio file creation |
| 10 | `_DaisySettingsPanel` | DAISY settings |
| 11 | `_ManageSTTPanel` | Manage the STT (speech-recognition models) |
| 12 | `_ManageDescribePanel` | Manage the video describer (audio description) |
| 13 | `_HuggingFacePanel` | Hugging Face account |
| 14 | `_InputGesturesPanel` | Keyboard gestures |
| 15 | `_ComputePanel` | Compute (the ONNX GPU runtime only) |
| 16 | `_DeveloperPanel` | Developer (only with Developer Mode) |
| 17 | `_ResetPanel` | Reset |

**A category is built the first time it is shown.** The dialog opens by
constructing only General; each other category is constructed when it is first
visited (`SettingsDialog.ensure_panel`), which brings the whole dialog up in
about 0.07 s and 21 child windows instead of the 1.8–2.0 s and 392 child windows
building all seventeen up front cost (the heaviest category, Voice Clone - the
Voice Lab engines plus the OmniVoice studio - is the one someone opening
Settings to change the theme never looks at). Waiting costs nothing: a category that is never opened
has no controls and so *writes* nothing either — its stored values are already
correct, and validation and saving only ever run over the panels that exist, so
a value the user set in a category they did not open survives Apply, OK and
Cancel untouched. A category that *is* opened is built, named (real MSAA
accNames), themed and validated exactly as before, and a Reset throws the built
panels away so the next visit shows the defaults. `tools/measure_settings_open.py`
measures both numbers.

Details:

1. **General** — **Enable Developer Mode**, which reveals the Developer category
   and heavy logging, the project save location and the models folder (each with
   its own named button: **Browse project location…** and **Browse model
   location…**), and the update behaviour.
2. **Appearance** — Theme combo box: `System default`, `Light`, `Dark` (applies
   live), the ten original predefined dark themes and the ten new *vibrant* dark
   themes (added for 2027; the original ten are untouched), and any custom theme,
   with a **Create custom theme…** button opening the theme studio (six colours, a
   painted style, a wallpaper; any number of themes may be created, edited and
   deleted), and the **wallpaper** controls for the selected theme (on/off, an
   image of your own, and the painted background style).
3. **Download and remove ONNX TTS** (model manager):
   - Combo 1: Select TTS (catalog top-level, e.g. Piper, Kokoro).
   - Combo 2: Select language (of that TTS).
   - Combo 3: Select variant (e.g. `medium`, `low`, `v1.5`).
   - **Download** button (enabled when the selected voice is not fully downloaded) and
     **Remove** button (enabled when downloaded). Show per-file progress.
   - A variant whose first download stopped part-way (or that an older build
     installed only one voice of) is *completable*: the same button fetches the
     voices that are still missing and leaves the installed ones alone.
4. **Available TTS**:
   - Combo 1: TTS, Combo 2: Language, Combo 3: Variant, Combo 4: Voice.
   - Lists only **downloaded** voices so the user can preview/use them.
   - Voice-clone uploads appear here too (voice name = file name of the uploaded sample).
5. **Voice Clone** (Pocket TTS by Kyutai, F5-TTS, KokoClone, Chatterbox, Qwen3-TTS,
   VibeVoice, MOSS-TTS, GPT-SoVITS, Indic Parler-TTS) — the
   clone engines:
   - 1. **Engine** combo + **Install engine... / Remove engine** (pip into the engine's *own*
     virtualenv, `%APPDATA%\AIVoiceStudio\tts_envs\<engine>`) with the install status
     underneath.
   - 2. **Device** combo: **CPU always**; **GPU (CUDA)** and **Auto** are added when an
     NVIDIA GPU is detected (the CPU is never removed).
   - 3. **Voice**: *Use a built-in voice* (the engine's pre-made voices) or *Clone a voice
     from a sample* (reference recording + optional transcript + voice name → **Create
     cloned voice**; F5-TTS needs the transcript, the others condition on the audio alone).
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
   - Below the Voice Lab half sits the **OmniVoice engines** half (the old *OmniVoice engines*
     category, merged into this one): **Install / Remove OmniVoice dependency**
     (`omnivoice-triton`) and **Install / Remove OmniVoice Server** (`omnivoice-server`) with
     their install status, the three engine cards (Server, Triton, Hybrid) and the OmniVoice
     voice library (clone a sample or describe a voice, rename, delete, preview with a
     **Compute** combo). The packages install into the shared `tts_envs\omnivoice`
     environment; nothing OmniVoice-related installs from **Compute**, which keeps only the
     ONNX GPU (CUDA) runtime.
6. **Voice clone (uploaded model)**:
   - Combo 1: TTS, Combo 2: Language, Combo 3: Variant (only entries of cloning-capable TTS).
   - **Upload voice** button → file dialog for audio sample (`.wav/.mp3/.flac/.onnx`).
   - For Piper: a `.onnx` + `.onnx.json` pair is copied into the user model folder and registered
     under the uploaded file name. For the Voice Lab engines: a reference audio path is
     recorded (this entry is legacy - the XTTS cloner was retired).
   - Registered clone names appear in **Available TTS → Voice combo** and are usable everywhere.
7. **Recording settings**:
   - Punctuation combo: `Default (TTS default)`, `None`, `Math`, `All`. A mode applies to the
     text around an engine's own expression markup, never to the markup itself: Kitten TTS 2.0
     reads a leading `[emotion]` tag, inline `<event>` tags and `(((emphasis)))` spans as
     director's notes, so they survive every mode exactly as typed (`tts.markup`).
   - Speed slider, Pitch slider, Volume slider (with value labels).
   - Edit box prefilled with sample text + **Preview** button (plays a short synthesis).
8. **Audio file creation**:
   - Radio group (single choice): **Page by page with heading style 1**,
     **Page by page only**, **Heading style 1 only**, **Break on every heading**.
   - Description text for each mode (see §3.4).
9. **Reset** — **Reset to default** button; shows confirmation "All settings will go back to
   their default values." and clears models cache info.
10. **Compute** — detected CPU/GPU capability, and download/remove of the optional ONNX GPU
   runtime (about 1.8 GB). Restart requested after a change.
11. **Developer** — heavy logging, addon install/enable/disable/remove, the addon Python
    environment (pip install, package listing) and diagnostics. Hidden without Developer Mode.
12. **DAISY settings** — language, publisher and whether the DAISY 3 book embeds the text.
13. **Manage the STT** — the speech-recognition models of `stt/catalog.py` in a
    table (name, licence, languages, fit, notes), the variant combo and **Install
    selected model** / **Remove model**. Each model is installed into a Python
    environment *of its own* (`stt_envs\<model id>`) because their stacks
    (NeMo, transformers, two PyTorch pins, two protobuf versions) conflict. A GPU
    machine also gets the CUDA-only packages the model declares (Whisper’s
    `nvidia-cublas-cu12`); when an existing environment predates one, the category
    names the missing wheel and turns *Install* into a repair. No model needs an
    account, a token or a licence click-through.
14. **Keyboard gestures** — NVDA-style Input Gestures: the commands (`gestures.py`)
    grouped by category in a tree, the selected command’s keys in a list, and
    **Add gesture…** / **Remove gesture** / **Restore defaults**. A key another
    command already owns is named before it is reassigned, and a real clash is
    reported rather than resolved silently. Bindings are stored in
    `settings["gestures.bindings"]` and reloaded when Settings closes.

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
- All status messages are written to a status bar and to logs.
- **Changes are pushed, not waited for** (`gui/announce.py`): a screen reader only hears a control
  when the user navigates to it, so progress messages, status-line updates and inline validation
  errors are also announced through an MSAA **live region** — a 1x1 `wx.StaticText` carrying
  `STATE_SYSTEM_LIVE_REGION` whose accName is set to the message and whose window handle is sent
  `EVENT_OBJECT_LIVEREGIONCHANGED` (plus `EVENT_OBJECT_NAMECHANGE` for readers that only watch name
  changes). Progress is announced in 10-point steps, an error is *assertive* (it interrupts and drops
  the queued percentages), and a polite message is coalesced per key and paced at most once every
  0.6 s, so a download reporting every megabyte cannot bury the sentence the listener is hearing.
  Everything the layer announces is also in a visible control — the layer never becomes the only
  place a message exists.
- A window's accessible name is its own **title** (`gui/a11y.py`): the screen reader's window-title
  command must announce "AI Voice Studio", never the name wxWidgets gives a window of its own
  making ("frame", "dialog"); a control nobody named is left unnamed rather than announced as
  `listBox` or `panel`.
- The frozen executable carries a Windows **VERSIONINFO** resource — product name `AI Voice
  Studio`, product version `APP_VERSION`, built by `tools/make_version_info.py` for
  `packaging/ai_voice_studio.spec` — so the screen reader's "report the application version"
  gesture answers with the name and version instead of "Application unknown, version not
  detected".
- No modal dead-ends: long operations run in worker threads with a cancellable progress dialog.
- **Access keys** (`&` in a label, e.g. `&Preview`, `&Remove selected variant`, `&Start
  recording`): one letter per action per window, never shadowed by a menu item (a frame
  answers its menu bar first, so `&X` in the main window would be unreachable), and never
  silent - a key that lands on a disabled control beeps and says why in the window's own
  status line (`access_keys.install` + `show_access_key_hint`). Windows answers a mnemonic on
  both messages of one key press, so handlers that start something call `access_keys.once()`;
  `tools/a11y_gestures.py` audits keys, shadows and accelerators for every window.

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
| `tts_envs\f5tts` | F5-TTS |
| `tts_envs\kokoclone` | KokoClone (Kokoro-ONNX + Kanade) |
| `tts_envs\chatterbox` | Chatterbox (its own pinned PyTorch) |
| `tts_envs\qwen3tts` | Qwen3-TTS (its own pinned `transformers`) |
| `tts_envs\vibevoice` | VibeVoice (its own pinned `transformers`) |
| `tts_envs\omnivoice` | **both** OmniVoice engines (direct + HTTP server) |
| `tts_envs\indic_parler` | Indic Parler-TTS |
| `tts_envs\kitten_tts_2` | Kitten TTS 2.0 (KittenML) |
| `stt_envs\<model id>` | one speech-to-text model |
| `addon_env` | addons, the Developer tab and the sherpa-onnx GPU runtime |

Rules:

- The engine's id names the environment (`python_runtime.environment_id`);
  `SHARED_ENVIRONMENTS` maps the one deliberate exception - `omnivoice_server` onto
  `omnivoice` - because the two OmniVoice engines are the same model behind two front-ends and
  want the same base package and the same CUDA PyTorch.
- An environment outlives the installation that created it (it lives in `%APPDATA%`), and a
  virtualenv runs on top of the interpreter it was built from, whose absolute path it records in
  `pyvenv.cfg`. The base interpreter must therefore not be one inside the program folder: the
  environments are created from a per-user copy of the bundled runtime
  (`%APPDATA%\AIVoiceStudio\python_runtime`, staged by `python_runtime._stage_python_runtime`),
  and an environment that names a Python which has gone is re-pointed at the current one
  (`PythonRuntime.repair_env`, `python -m venv --upgrade`, which keeps its packages) instead of
  being rebuilt. Reinstalling the application - per-user after all-users, a new folder, a new
  drive - must never leave a model that is installed unable to run; every worker launch uses
  `PythonRuntime.usable_python_exe` for the same reason. Without it a reinstall produced
  `Audio description failed: the model's process exited with code 103 ... did not find executable
  at 'C:\Program Files\AI Voice Studio\python-runtime\python.exe'`.
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
  shared environment, where the engine died with `ModuleNotFoundError: No module named 'torch'`
  although its own environment had PyTorch. When neither environment holds the engine the error says so in
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

### 3.8 Update checks and the winget package (2026.3.7)

The project is published on GitHub, and that is the only source of update
information.

- **Source of truth**: `https://api.github.com/repos/anujj87/AIVoiceStudio/releases/latest`
  (`updates.LATEST_RELEASE_API`). A draft release is ignored; the tag is
  parsed into numbers so `2026.10.0` is newer than `2026.3.7`.
- **Help → Check for updates** (access key `Alt+D`, and **Check for updates now**
  in Settings → General → Updates) always answers: a newer version (the update
  dialog), “you are using the newest version”, or the reason the check could
  not be made.
- **Automatic check**: once a day at start-up when `updates.auto_check` is true
  (default). A successful check records `updates.last_check`; a version the
  user muted is remembered as `updates.skip_version`. A check is silent
  unless there is something to offer, runs on a worker thread, and can never
  block or fail the start-up.
- **The update dialog** shows the installed and the new version, the published
  release notes, and three actions: *Download and install* (resumable download
  with progress and Cancel, then Inno Setup `/SILENT`), *Open release page*, and
  *Remind me later*. Nothing is downloaded without this dialog.
- **winget compliance**: `winget/manifests/<letter>/AnujSharma/AIVoiceStudio/<version>/`
  holds the three manifests winget requires (version, installer, locale),
  generated by `tools/make_winget.py` from `APP_VERSION`, the built installer's
  SHA256 and the release tag. `tools/winget_validate.py` (run by
  `tests/test_winget.py`) checks the rules locally: valid identifier and
  dotted-numeric version, required metadata, SPDX licence, lowercase tags,
  `InstallerType: inno`, `Scope: user`, `InstallModes`, `UpgradeBehavior: install`,
  the `/VERYSILENT` switch the installer answers, a versioned (never “latest”)
  download URL, the published `ProductCode` (the Inno `AppId` + `_is1`) and a
  64-character uppercase SHA256 that matches the built bytes.
- **The downloaded installer is verified before it is offered.** GitHub
  publishes a SHA-256 `digest` for each release asset; `updates.verify_download`
  hashes the file, and a mismatch deletes it and reports the error instead of
  handing it to `launch_installer`. A release published without a digest (older
  than GitHub's digests) is logged as unverified and allowed, so nobody is
  stranded on an old version. The installer itself is still unsigned (see
  DEV_PLAN), which is why this check matters.
- The Inno `AppId` is never changed for packaging reasons: it is the key
  Windows uses to recognise an installed copy.

#### 3.8.1 Credentials at rest

`settings.json` holds two credentials: the Hugging Face **read token**
(`hf.token`) and the OmniVoice server's `api_key`. Both are encrypted with
**DPAPI** (`secret_store.py`: `CryptProtectData` with `CRYPTPROTECT_UI_FORBIDDEN`),
so only the same Windows user on the same machine can decrypt them and no
password or key file is involved. The stored value is `dpapi:<base64>`, only
the value is encrypted (key names, structure and every other setting stay
readable), and a settings file from an older version is migrated - read as
plain text, then rewritten encrypted - the first time it is loaded. Where DPAPI
is unavailable the plain-text footing is kept and `secret_store.available()`
says so: a token that cannot be encrypted must not become a token that cannot
be used.

### 3.9 Speech to text (transcription)

A project has a **base**: *recording* (text → audio) or *transcription*
(audio/video → text). The base is chosen first in the New Project wizard, and
the Edit menu’s wording follows it (a transcription project has *files*, a
recording project has *recordings*; *Start Selected Files*, *Transcribe all files
from here*).

- **Sources**: audio (WAV, MP3, FLAC, M4A, AAC, OGG/Opus, WMA, AIFF) and video
  (MP4, MKV, MOV, WebM, AVI, M4V, MPG, MPEG, WMV, TS, M2TS). A file or a folder is
  accepted; a folder is walked recursively. Video is decoded by the FFmpeg the
  application already downloads for MP3/FLAC output.
- **Models** live in `ai_voice_studio/stt/catalog.py`: Whisper (faster-whisper),
  Qwen3-ASR, NVIDIA Canary-Qwen, NVIDIA Parakeet TDT, Moonshine, SpeechBrain,
  Voxtral Mini 3B and IBM Granite Speech. A model declares its language, licence,
  variants, back-end, pip packages, CUDA-only packages, whether its back-end really
  reports *timings*, how a transformers/NeMo model must be driven, and whether it
  genuinely cannot run without a GPU. A model whose repository is gated on Hugging
  Face is deliberately absent, so nothing needs an account or a token.
- **The Compute combo tells the truth**: only a GPU-only model (Canary-Qwen 2.5B,
  Granite Speech 8B) is offered the GPU alone; every other model offers *both* the
  CPU and the GPU when an NVIDIA card is present (the preferred device first) and
  the CPU alone when there is none. The combo is rebuilt when the variant changes.
  `Auto` is never offered.
- **The dialog** offers model, variant, language (or automatic detection), accuracy
  (*fast* = greedy, *balanced* = beam 5, *most accurate* = beam 8, the same labels
  for every model) and, only where the model supports it, a translation target.
- **Outputs**: TXT, Markdown, HTML, JSON, CSV, SRT and VTT, with or without
  timestamps. SRT/VTT always carry their timing. *Time by time separation* splits a
  recording into documents of *n* minutes without ever cutting a segment in half,
  and two sources with the same base name are disambiguated rather than overwritten.
- **The job** runs on a worker thread, writes each finished document to disk
  immediately (a crash costs one file), reports progress per file, can be cancelled,
  and offers **Open project folder** when it finishes.

### 3.10 Audio description (video → narration)

A third project base: a vision-language model watches a video and writes the
narration a blind or low-vision viewer needs, which is then documented, spoken
and mixed back into the film.

- **Sources**: video containers only (MP4, MKV, MOV, AVI, WebM, FLV, WMV, M4V,
  MPG/MPEG, 3GP, TS, M2TS, OGV). A file or a folder is accepted; a folder is
  walked recursively. FFmpeg decodes the frames and reads the duration.
- **Models** live in `ai_voice_studio/describe/catalog.py`: Qwen3-VL, Qwen2.5-VL,
  Qwen2.5-Omni, Qwen3-Omni, Gemma 3, Gemma 4, MiniCPM-o 2.6, MiniCPM-o 4.5,
  MiniCPM-V 4, SmolVLM2 and Moondream, each declaring its licence, notes,
  `trust_remote_code`, timestamp
  support, pip packages, **CUDA-only packages** and per-variant size / VRAM /
  cost. A model’s
  environment (`describe_envs\<model id>`) and a variant’s checkpoint are
  separate layers: the packages are shared by every variant, the checkpoint is
  not, and the UI keeps them apart (one Install/Repair button per model, one
  Download/Remove button per variant, with a status column).
- **A model's own modelling code decides part of its environment.**
  `trust_remote_code` models run the checkpoint's Python, so a package that code
  imports - and the library release it was written against - belongs in the
  catalogue entry, not in an instruction to the user. MiniCPM-o 2.6 is the
  worked example: `torchaudio` (its processor) and `librosa` (its utilities,
  which brings `soxr` with it — the pair is never installed apart) are declared
  packages, and `transformers` is pinned
  below 4.53 because its code imports whisper internals that 4.53.0 removed.
  `environments.missing_packages()` is the second half of that rule: it compares
  the declared packages against the environment's own `site-packages`, so an
  environment built before a requirement was declared is *repaired* (the button
  names what is missing) instead of having to be deleted and built again.
- **A checkpoint also decides which class loads it and how it is asked.** The
  catalogue carries `auto_class` (the name the checkpoint's own `auto_map`
  registers), `driver` (`generate`, `chat` for the MiniCPM-o/V family, whose
  processors ship no chat template that carries the standard multimodal message,
  or `query` for Moondream 2, which ships no processor or template at all and is
  asked through its own `query(image, question)` skill, one frame per
  description), `dtype` (the checkpoint's own, for a model whose forward pass
  fixes it — Moondream 2 casts its crops to bfloat16) and `load_kwargs`.
  The MiniCPM-o/V entries are the `chat` case: their checkpoints
  register `AutoModel` and nothing for the task-specific classes, so a run that
  asked for `AutoModelForImageTextToText` was refused with *"Unrecognized
  configuration class … for this kind of AutoModel"* from a correct environment;
  they are asked for `AutoModel`, driven through their own `chat`, and the omni
  ones are loaded vision-only (`init_audio`/`init_tts` off), because a
  description reads the picture and their speech half is a stack a description
  never uses. A generation that needs a package its predecessor does not (4.5
  wants `minicpmo-utils`, which pins an older Pillow and librosa) gets an entry —
  and therefore an environment — of its own.

  Pins are
  per model for the same reason the environments are: a pin can only affect the
  environment it is installed into.
- **Gated repositories** carry `"gated": True` in the describing catalog
  (Gemma 3's three checkpoints) or a `gated_repo` entry in the Voice Lab engine
  registry (`kyutai/pocket-tts`, which Pocket TTS's *voice-cloning* weights live
  in — its 26 pre-made voices need nothing). `ai_voice_studio/hf_access.py`
  turns the flag into something usable: `GATED_REPOS` is the registry of the
  gated repositories the application ships, Settings > **Hugging Face account**
  holds the user's own *read* token (verified against `whoami-v2` by *Check
  token*, stored in their own `settings.json`), the panel's **Open the model
  page** button opens the page whose own accept button grants the licence, and
  **Check access** (`hf_access.check_access`, which fetches the smallest file the
  repository lists) answers the question the failure itself leaves open. Every
  model worker is started with `HF_TOKEN` in its environment, and a gated run
  without a token stops **before** the download with the page to accept the
  licence on and the panel to paste into — instead of the library's bare
  `401 — Cannot access gated repo ... Please log in`, and instead of Kyutai's
  advice to run an `hf auth login` this installation does not have.
  `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` and a previous `huggingface-cli login`
  are honoured too, and every other model (including all STT models) still
  downloads with no account at all.
- **One description per window, not per chunk.** `AD_WINDOW_SECONDS = 20.0`
  (capped by `AD_MAX_WINDOW_SECONDS = 60.0`) is what one description covers; the
  chunk (1–10 minutes) only bounds memory, progress and resume. Each window is
  told what the previous description said, so the model reports the change —
  on-screen text, code, a gesture — rather than the scene, and a window whose
  picture has not changed is skipped (reported as skipped) instead of repeated.
- **Placement** is one choice with two values: *standard* speaks into a pause of
  the soundtrack (optionally fitted to the real silences by the speech-to-text
  model; a description no pause can hold is dropped rather than spoken over
  dialogue), and *extended* stops the film for each description and goes on.- **Track priority** decides which of the two tracks gives way when a description
  is mixed over a film's own soundtrack (standard placement only; extended
  placement switches the combo off, because there the film is silent while the
  description speaks). `AD_TRACK_PRIORITY_CHOICES` lists the four answers from
  the gentlest to the film to the most forceful and opens on the first — *same
  volume*, the default, which is what every run before the choice produced, and
  what a project carrying no value at all is still mixed as — then *duck, don't
  lower*, then *25% higher* and *50% higher*; `AD_TRACK_PRIORITY_GAIN` is the film's own gain
  (1.0, 0.75, 0.5, and 1.0 for the ducking, which does not lower the film).
  `extended.mix_graph(film_volume=…)` emits one FFmpeg `volume` stage on the
  film's input and never touches the description, which keeps the level the
  narration settings gave it. Turning the description up instead would clip
  (full-scale speech already peaks at full scale) and would change a level the
  user had chosen by ear. **Duck, don't lower** is the same one stage with a
  different gain: `extended.track_priority_duck()` answers `AD_DUCK_LEVEL = 0.25`
  for it and 1.0 for every other choice, and
  `extended.mix_graph(film_volume, duck_level, duck_turns)` builds a per-frame
  `volume` expression (evaluated once per audio frame, not per sample) that is 1
  everywhere and a trapezoid dipping to 0.25 over each spoken description
  (`AD_DUCK_EDGE_SECONDS = 0.06` either side), so the film's own sound is dipped
  to about −12 dB exactly while it is spoken over and left as it is between
  descriptions. The moments are measured, not estimated: `narration.write_track()`
  records `turns` — where each description is really heard, from the placed clips
  — and `DescribeJob._render_mix()` hands them to the mixer; two descriptions
  closer than `AD_DUCK_MERGE_SECONDS = 0.4` are merged into one dip. The dip is a
  refinement, never a reason to lose the film: if the expression would not fit on
  one command line (`AD_DUCK_MAX_EXPRESSION`), or FFmpeg refuses it, the next
  attempt writes the plain mix — both tracks as they are — before the
  narration-only attempt that exists for a video with no soundtrack. Chosen in
  the Audio description window, remembered in `settings['describe.track_priority']`
  and written into a new project by the wizard, and recorded in the run's own
  output record (`track_priority`, plus `duck_level` when the film was dipped).

- **The narration voice's own volume** is a slider directly under the voice
  combo in the describe window (`Narration voice volume`), on the recording
  side's range (`VOLUME_MIN`–`VOLUME_MAX`, 1.00 the voice as downloaded). It is
  the description's own level — *Track priority* moves the film's track, never
  the description — it is carried into the run as `tts.volume`, and it is
  remembered in `settings['describe.narration_volume']` and written into a new
  project by the wizard.
- **Three outputs per video**: the description document in the chosen format
  (script / HTML / SRT / VTT / JSON, with the reviewable script always written
  beside it), the spoken narration read by an installed offline voice on the
  film’s own clock, and `<source> - described.mp4` — a **new** file, never the
  source: the original picture stream copied with `-c:v copy` plus the original
  soundtrack mixed with the narration (`normalize=0`, `duration=first`), or the
  film with its pauses baked in for extended description.
- **FFmpeg is required** (frames, duration, described video). When it is missing
  the describe window offers to download it into the app’s user folder instead
  of refusing to run.
- **A model larger than the card is loaded in 4 bits** (`bitsandbytes`, a
  CUDA-only package) rather than being offloaded to the disk, which is what
  ended a real run with `0xC0000005`; placement is always explicit
  (`device_map={"": 0}`), and a model that can run neither way is refused with
  its reason.
- **The job** runs on a worker thread (`describe/job.py`), writes each finished
  piece into the script on disk as it goes (closing a long run loses nothing),
  reports progress per piece, can be cancelled, and its completion dialog names
  the three outputs and any warnings.

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
│   │   ├── downloader.py      # resumable downloads, per-voice install layout
│   │   ├── models.py          # installed-model state
│   │   ├── engine.py          # sherpa-onnx wrapper + engine cache
│   │   └── windows_tts.py     # SAPI5 and Windows Core voice enumeration
│   ├── voicelab/              # Pocket TTS, F5-TTS, KokoClone, Chatterbox, ...
│   │   ├── engines.py         # registry, built-in voices, PYTORCH_CUDA_INDEX
│   │   ├── options.py         # per-engine tuning vocabulary
│   │   ├── benchmark.py       # per-engine/device speed measurement
│   │   ├── worker.py          # standalone worker subprocess (torch lives here)
│   │   └── __init__.py        # device choice, worker client, VoicelabEngine, verify_engine_import
│   ├── stt/                   # speech to text: one environment per model
│   │   ├── catalog.py         # model registry: back-ends, packages, device fit, timings
│   │   ├── environments.py    # per-model venv lifecycle, install plan, import check
│   │   ├── engine.py          # transcribe_audio + the worker request
│   │   ├── worker.py          # standalone worker subprocess (its own interpreter)
│   │   ├── job.py             # background transcription job
│   │   └── output.py          # TXT / SRT / VTT / HTML / JSON / CSV documents
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
- **Silent install** must work with winget's switches: `PrivilegesRequired=lowest` (per-user by
  default, so no elevation is needed), every `[Run]` entry flagged `skipifsilent` so a silent
  install never launches the application, and `/VERYSILENT` advertised in the winget manifests.
  `packaging\silent_install.ps1` performs that unattended install, and the matching uninstall,
  with the same switches: it is immune to the POSIX-shell path rewriting that turns
  `/VERYSILENT` into `C:/Program Files/Git/VERYSILENT`, and it waits for Inno's uninstaller,
  which exits 0 before it has finished deleting the files. It also clears the `__pycache__`
  bytecode caches the bundled runtime writes beside its own modules when it runs: Inno removes
  only what Setup installed, so a wait for an empty folder would never end.
- The winget manifests are part of the release, not of the installed application: they ship in
  the repository under `winget/` and are submitted to `microsoft/winget-pkgs` per version.

## 6. Out of scope (first release)

- Streaming/listening preview while recording (single-pass generation; Preview uses the same
  engine synchronously).
- Cloud voices and commercial models.
- Non-Windows installers (app is Windows-first; code is portable in principle).
