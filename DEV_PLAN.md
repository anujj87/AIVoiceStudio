# AI Voice Studio — Development Plan

Milestones map to the implementation order. Each phase ends with something runnable/verifiable.
Status is tracked at the top; check boxes are updated as phases complete.

Current release: **2027.1.0** (`constants.APP_VERSION`;
`TERMS_VERSION` stays 2026.3.1 because the terms text did not change, so an
upgrade does not ask the user to accept them again; `installer_common.iss`).
Test suite: unit, GUI smoke, engine-environment, book, documentation,
accessibility and winget-manifest guards. Installer:
`dist\AI-Voice-Studio-v-2027-1-0-Setup-x64.exe`.

2027.1.0 keeps the installer's interactive "install for all users / install for
me only" choice and makes a GPU recording use far more of the card.  sherpa-onnx
prepares and runs one sentence per call and the Kokoro implementation ignores
`max_num_sentences`, so a single engine left the GPU waiting on the CPU for
every sentence.  The GPU back-end now keeps two segments in flight (up to four
on a larger machine) - `compute.synthesis_workers` and `tts/engine.TtsEnginePool`
- and each segment is still produced by the same model with the same arguments,
so the audio is unchanged.  Measured on an RTX 3060 with a six-core CPU:
0.186x realtime at ~16% average GPU utilisation before, 0.144x realtime at ~22%
after; per-segment audio lengths agree to within 0.03%.

2027.1.0 also settles the two Canary failures a user reported.  The reported
"error" was a model download's progress bar and a `transformers` deprecation
warning: a NeMo model killed by an out-of-memory condition writes no response,
so the worker's last stderr lines became the headline.  `_worker_error_detail`
now drops progress bars, warnings and NeMo's own log lines, keeps the real
cause, and names the exit code when the worker was killed.  The cause itself
was two-fold: NeMo 3.0 segfaults against `transformers` 5 (both NeMo models now
pin `transformers<5`, and Canary-Qwen loads and transcribes again), and the
"Canary-1B" variant is an ordinary Canary ASR checkpoint, not a SALM, so it is
now driven through `ASRModel` instead of the speech-language-model loader that
reported "Missing key audio_locator_tag".  The same release adds a modal
cancellable transcription progress window with *Open project folder* on
completion, a project-type-aware Edit menu (a transcription project has
*files*, a recording project *recordings*), *What's New* in the Help menu with
`docs/WHATS-NEW.html`, the New Project wizard focusing the project base first,
and Settings > **Keyboard gestures** - a configurable gesture manager modelled on
NVDA's, with per-command defaults, user overrides, conflict reporting and a key
capture dialog.

2027.1.0 also makes the one *gated* model usable instead of merely failing.
Every Gemma 3 checkpoint lives in Google's gated Hugging Face repository, so
asking for Gemma 3 12B ended in the library's bare `401 Client Error ... Cannot
access gated repo ... Please log in` part-way through a 24 GB download - a
message that names neither the page that grants access nor any way to give the
application a token (`huggingface-cli login` is not available to someone who
installed the setup file).  `ai_voice_studio/hf_access.py` and the new Settings >
**Hugging Face account** category answer it: paste a read token, *Check token*
reports the account it belongs to, every model worker is started with `HF_TOKEN`
in its environment, and a gated run stops *before* the download with the page to
accept the licence on and the panel to paste into.  `HF_TOKEN` and an existing
`huggingface-cli login` are honoured too, and every other model - all 30 other
repositories, checked one by one against the live API - still downloads with no
account, no token and no click-through.

A user then asked the obvious question the first version could not answer: *how
do I make a token for Gemma 3?*  There is no such thing - one read token serves
every gated model, and what grants the access is accepting *that* model's
licence - so `hf_access.GATED_REPOS` is now the registry of the gated
repositories the application ships (Gemma 3's three and Kyutai's
gated `kyutai/pocket-tts`), and the panel does both halves of the job: a
**Model** picker, **Open the model page...** (the page whose own *Agree and
access repository* button grants the access) and **Check access**.  That last one
is `hf_access.check_access`: it fetches the **smallest file** the repository
lists, so a *yes* costs a few hundred bytes, and on a refusal it also asks
`whoami-v2` so the answer separates the two problems that share one error message
- a token that is not accepted (fixed in the panel) and a licence that was never
accepted (fixed on the model's page, and nowhere else).

The same gate turned up in the Voice Lab, exactly as the user reported it: Pocket
TTS speaks fine but its *Clone a voice from a sample* asked for Hugging Face.
Kyutai's voice-cloning weights are in `kyutai/pocket-tts`, which is gated; the 26
pre-made voices are not, so the engine loads, the built-in voices work, and the
library - rather than failing - quietly falls back to its cloning-free weights
and then answers a clone with advice to log in with an `hf auth login` command
this installation does not have.  The engine registry now declares
`gated_repo: kyutai/pocket-tts`, the worker records that it loaded without voice
cloning (a note in the run) and refuses a clone with the two steps that work, and
`voicelab` translates any gated failure that reaches the application into the
same message naming the page and the panel.  `tools/check_hf_access.py` grew a
fifth live check: every gated repository is asked for with no token (must answer
"no token") and with a token that does not exist (must be refused, and must say
whether the token or the licence is the problem).

2027.1.0 also adds the Voice Lab engine a survey of Hindi/Indic TTS turned up,
and records the three it did not. **Indic Parler-TTS** (AI4Bharat) speaks the
Indian languages and takes its text directly, so it needs no `pynini` front-end
(which has no Windows wheel); it is *description-conditioned* (its 65 named
speakers are picked by rewriting a sentence about the voice, which is also its
voice-design mode) and needs the card. **IndicF5** (AI4Bharat) and an
**OmniVoice ONNX** export (the same model through ONNX Runtime, the one engine
here with no PyTorch at all) were added in the same survey and **removed again
before release**: neither spoke the text it was handed (the faults below are the
evidence), and the application's own OmniVoice engines already cover that model.
Two models were rejected on evidence
rather than taste, and the reasons are in `PROJECT_SPEC.md` §2.2.3 so they are
not attempted again: **Svara-TTS** (vLLM and FlashInfer are Linux-only and it
has no CPython 3.13 Windows wheel) and AI4Bharat's **Indic-TTS** (its front-end
needs `pynini`, and its pinned `numba` has no cp313 build).

Each engine was installed into its own environment and run both ways before it
was documented; `tools/e2e_voice_lab_engine.py` is that check, kept as a
tool: it drives a segment through the app's own path (voice entry →
`VoicelabEngine` → the engine's interpreter → its worker → the model download)
and reports the sample rate, the length, the real-time factor, the RMS **and the
speech-versus-drone verdict the application itself uses** - a take with no
speech in it is loud, so the RMS alone once passed an engine that only boomed.
It then **transcribes its own take with the application's Whisper large-v3** (the
same `stt_envs\whisper_large_v3` environment, started through
`ai_voice_studio.stt.engine` as a recording is) and compares the words with the
text it was given: a word error rate above **0.35** fails the run, and a take
whose words could not be checked at all fails it too, because a take nobody has
listened to is not a pass (`--wer` moves the limit, `--no-words` is the explicit
way out). The tool is what caught the faults below, and it is also what failed
the two engines that were removed: Indic Parler-TTS's Hindi take measures
**0.23** through the probe's default `balanced` decoding,
while the 0.10-0.14 figures quoted above came from the quickest setting
(`--asr-accuracy fast`), which is what the first survey used.  The probe prints the rate and the decoding setting together, so one
figure is never read against another taken differently.

**That verdict is not enough, and this is the lesson the removed engines taught.**
It separates a boom from speech; it cannot separate speech from *babble*.
OmniVoice ONNX passed it - 5.12 s of audio, 0.35-0.61 of its audible windows
speech-like, no drone - and **did not speak the text it is given**: run through
the application's own Whisper large-v3 (from `stt_envs\whisper_large_v3`), the
same Hindi sentence that Indic Parler-TTS returns at a word error rate of 0.10
came back from it as unrelated Hindi at **0.95-1.00**, in auto-voice and in
clone mode, on the int4 and the fp16 backbone alike - which is why that engine is
gone. The Indic engines were therefore re-verified with a transcript rather than
with the detector, and that is what caught IndicF5's fault: `indic_parler` came
back at 0.10, while `indicf5` measured **0.52** on the probe's own sentence and
**0.89** on the sentence the checkpoint's own model card ships. The words that
survived made it look like F5 swallowing the reference prompt, which is how the
first reading explained it, and it was wrong: the checkpoint was loading without
its text-embedding weights at all. With them re-applied by hand the same two
takes measure **0.14** and **0.00** - and IndicF5 was removed anyway, because an
engine that must speak correctly cannot depend on the application re-applying a
checkpoint's own weights file itself (the fault below). The ONNX export's README
claimed 0.48x real time on a CPU, which did not reproduce here either (44 s on
the CPU, 29 s on the RTX 3060), and its take was never the words. Both engines
are gone; the app's own OmniVoice engines, which run the PyTorch checkpoint, are
the ones to use for speech.

Eight real faults were found by running the new engines. Those that applied to
code that survives are fixed (Indic Parler-TTS's install, for instance); the
rest are kept as the record of why two of the engines were removed:

* **OmniVoice ONNX did not speak at all**, and the export is why: its own
  `inference.py` feeds the text bare, runs one forward pass per step and picks
tokens greedily. Upstream frames the text with seven added tokens, runs a second
  pass over the audio span alone and combines the two with classifier-free
guidance, and draws the committed positions from seeded Gumbel noise on a
  shifted schedule. Ported faithfully at first, the engine rendered a steady boom
  at full level - **5% of its audible windows sounding like speech, 43 distinct
  codebook entries in the whole clip with one of them on 117 of 1024 cells** -
  which every level test passed. The loop now follows upstream (`_omni_prompt`,
  `_omni_schedule`, `_omni_unmask`), the same clip measures 61% (Hindi) / 50%
  (English voice entry), and the seed is a real tuning option: an empty box is the
  engine's own fixed seed, so the same text repeats, and another one re-draws a
  take that came out bad. **The boom is gone, but the words never arrived**, and
  the second half of this diagnosis came later: with the same prompt and mask
  span, an unrelated sentence in another language leaves **89% of the export's
  predicted codes identical** (per codebook 62-99%) and moves its logits by
  only **6% of their spread**, so the backbone barely conditions on its input -
  and that is true of the int4 build and of the export's fp16 `cuda/` build. The codec is not the
  problem: a known-intelligible clip encoded and decoded through the four Higgs
  models transcribes at 0.10, and the loop matches the export's own
  `omnivoice_manifest.json`. What is left is the export itself: a speaking build
  needs a corrected one (a re-export from the PyTorch checkpoint, e.g. with
  Olive).
* **Indic Parler-TTS did not start**: `parler-tts` declares `protobuf>=4.0.0` but
  `descript-audiotools`, installed by name for its DAC decoder, still caps
  protobuf at `<3.20` - and on 3.19.6 the `sentencepiece` that comes with it
  cannot import `google.protobuf.internal.builder` ("cannot import name
  'builder'"). Nothing in `audiotools` or `dac` imports protobuf, so the engine's
  own requirement wins and `protobuf>=4.25.0` is installed last. Verified by
  speaking: 5.11 s of Hindi at 44.1 kHz, 68 s on the card.
* **IndicF5 did not start either, for three reasons in a row**, each found by
  installing it and running one sentence: `pydub` - required by the checkpoint's
  remote code - imports the `audioop` module CPython 3.13 removed, so it is
  installed together with `audioop-lts`; the archive's own inference imports
  `matplotlib`, `wandb` and `datasets` at module level and its `--no-deps`
  install brings none of them; and with the `transformers` 5.x an unconstrained
  name resolves to, loading the checkpoint dies in its 4.x-era model code with
  "Tensor on device cpu is not on the expected device meta!", which the archive's
  own `transformers<4.50` prevents (`datasets<4` in the same step keeps the
  huggingface-hub requirement consistent). Verified by cloning from a reference
  recording: 5.04 s of Hindi at 24 kHz, 24 s on the card.
* **IndicF5 spoke gibberish, because the checkpoint was only half loading.** The
  engine installed and ran, and what came back was speech-shaped nonsense: run
  through the application's own Whisper large-v3, the checkpoint's own demo - a
  Punjabi reference and the Hindi line its model card documents - transcribed at
  a word error rate of **0.89**, with only its first and last words close. The
  cause is `transformers` 4.46.1, the version the archive's own
  `transformers<4.50` pins: while loading a checkpoint it renames every tensor
  whose *name* contains `beta` or `gamma` to `bias`/`weight` (a substring rule
  written for its own renamed parameters, which cannot tell `grn.beta` from a
  rename). Both halves of this checkpoint name parameters exactly that way -
  f5_tts's text-embedding `ConvNeXtV2Block` keeps a `GRN` with `beta` and
  `gamma`, and the vocos vocoder keeps a layer scale in `gamma` - so those
  tensors never reached the model and stayed at their initialisation, zeros. A
  GRN block with zero parameters is an identity, so the text conditioning no
  longer described the words at all. Nothing looked broken: the load only warns,
  and the warning names keys (`grn.bias`, `convnext.0.weight`) which exist in no
  file, while every key and shape the model holds does match the published file,
  so a comparison that stops at names and shapes finds nothing wrong (only the
  *values* gave it away). The engine now applies the checkpoint's own
  `model.safetensors` a second time, one tensor at a time, and counts anything it
  cannot place. The same two takes afterwards: **0.14** on the probe's sentence
  and **0.00** on the model card's demo - the words, in the cloned voice. The
  lesson generalises past this engine: a load that reports success has not
  necessarily applied every tensor, and only a transcript-level check says
  whether a voice engine spoke.

* the session dictionary the worker loaded (`audio_embeddings_encoder`, by file
  stem) and the keys the forward pass read (`audio_embeddings`) had drifted
  apart, which no unit test noticed because the test's stubs were keyed to match
  the *forward pass*; the names are now module constants used by both, and the
  stub sessions are keyed from the loader.
* `onnxruntime-gpu` lists the CUDA provider as available the moment it is
  installed, and only fails at its first `Conv` node when cuDNN is missing -
  a crash rather than a message. The engine now puts the CUDA folders on the
  DLL path **and** `PATH` and asks for the card only when cuDNN 9 is on disk;
  otherwise it runs on the CPU and says why. Both halves were needed: with
  `os.add_dll_directory` alone cuDNN's own loader reported "Could not locate
  cudnn_graph64_9.dll" on a machine where the provider then ran perfectly.
* the engine installs no PyTorch, so nothing supplied cuDNN 9 and cuBLAS 12 -
  every other engine gets them from `torch/lib`. Its GPU install now brings
  `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12` and
  `nvidia-cufft-cu12`, and those names are part of its package list so *Remove
  engine* takes them away again.

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
- [x] `theme.py` (system/light/dark, kept unchanged) plus **twenty predefined dark
      themes** (the original ten, and ten vibrant ones added on request) and a **custom
      theme studio** (`theme_studio.py`: colours, style, name, unlimited themes).
- [x] **Settings → Appearance**: the theme list, the custom theme studio and the
      wallpaper moved out of General into a category of their own.
- [x] `a11y.py` (accessible labels; MSAA names via `finalize_accessibility`).
- [x] **Wallpapers** for any theme (built-in, predefined or custom): the user's image or a
      painted pattern, drawn by `_WallpaperPanel` in the welcome screen's empty region only.
      That screen is three regions - a full-width action bar (Remove selected project Alt+J,
      Create New Project, Record, Settings), a left projects sidebar with the recent-projects
      list, and the remaining empty area the wallpaper fills - so the wallpaper never paints
      behind a button or the project list.
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
- [x] `voicelab/engines.py` registry (Pocket TTS, F5-TTS, KokoClone, Chatterbox,
      Qwen3-TTS, VibeVoice) with built-in voices and licence metadata, all generated from
      the engines' own voice lists rather than typed out per voice.
- [x] Bark (Suno AI) removed: its 130 speaker presets and `.npz`-only "cloning" made it
      the one engine the rest of the Voice Clone page could not express.
- [x] KokoClone (Kokoro-ONNX + Kanade voice conversion) installed from pinned source
      archives (it is not on PyPI), with the CUDA ONNX runtime on a GPU machine and an
      `own_environment` flag that keeps it out of the shared OmniVoice environment.
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
- [x] `docs/book`: 88 chapters and 3 appendices (basic Python → rebuilding the
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
      the full Inno silent switch set, versioned URL, ProductCode, SHA256).
      The suite also holds the installer to it: the script must keep the
      install-mode choice (`PrivilegesRequiredOverridesAllowed=dialog`) while
      staying per-user by default, and the switches must be complete, because
      winget-pkgs' installation test is unattended (`SilentSwitchTest`).
- [x] Installer registers proper ARP links (`AppPublisherURL`, `AppSupportURL`,
      `AppUpdatesURL`); the Inno `AppId` stays unchanged so upgrades keep working.

## Phase 19 — Speech to text (transcription) ✅
- [x] `stt/catalog.py`: nine models (Whisper Large V3 / Small-Base, Qwen3-ASR,
      Canary-Qwen, Parakeet TDT, Moonshine, SpeechBrain, Voxtral, Granite Speech) with
      their back-ends, pip packages, licence, languages, translation targets, timing
      support and the **device each one is for**.
- [x] `stt/environments.py`: one Python environment per model (`stt_envs\<model id>`) so
      NeMo, `transformers` and two different PyTorch pins can never break each other.
- [x] `stt/worker.py`: one worker run on the model's own interpreter, with a runner per
      back-end (faster-whisper, OpenAI Whisper, `transformers` pipeline / chat prompt /
      audio-LLM request API, NeMo ASR + SALM, SpeechBrain), 16 kHz decoding without
      torchaudio, CUDA→CPU fallback, and per-model timestamp handling.
- [x] Device fit: only a model that genuinely cannot run without a GPU (Canary-Qwen 2.5B,
      Granite Speech 8B) is offered the GPU alone; every other model offers both the CPU
      and the GPU when an NVIDIA card is present (preferred device first) and the CPU alone
      without one, because many users have no card. The fit is decided per variant, so the
      combo is rebuilt when the variant changes. No model is offered `Auto`, and a GPU
      choice no longer waits for the optional sherpa-onnx GPU runtime — the models bring
      their own PyTorch.
- [x] A GPU run needs its CUDA runtime in the model's own environment: CTranslate2 needs
      cuBLAS, which `faster-whisper` does not ship, so `nvidia-cublas-cu12` is a CUDA-only
      catalog package. An environment installed before that wheel existed (Whisper Large V3
      here) fell back to the CPU — Manage the STT now names the missing wheel and turns
      Install into Repair, and the transcribe dialog warns when the selected GPU run would
      fall back.
- [x] Gated models are not offered at all: Cohere Transcribe only downloads after signing
      in to Hugging Face and accepting a licence, so it was removed from the catalog.
- [x] Voxtral 24B removed: ~48 GB of bfloat16 weights cannot load on the 12 GB cards the
      catalog targets (or in the 16 GB of RAM this project is developed against); the 3B
      model is the one that fits.
- [x] Verified end to end on real downloaded weights: Whisper (CPU and CUDA), Qwen3-ASR
      1.7B, Voxtral Mini 3B, Moonshine, Granite Speech 1B and 3.3 2B, Parakeet TDT,
      Canary-Qwen, SpeechBrain.
- [ ] Granite Speech 3.3 **8B** is unverified: its ~17 GB of weights exceed both this
      machine's RAM and the download budget it was tested under.
- [x] A gated model is now supported deliberately: `hf_access.py` + Settings > **Hugging
      Face account** (an explicit token field, a *Check token* button that asks whoami-v2,
      and the "accept the licence" step named in the failure itself), the token handed to
      every model worker as `HF_TOKEN`, and `catalog.gated()` turning the bare
      `401 — Cannot access gated repo` into the two steps that fix it *before* a download
      starts. Gemma 3 is the describing catalog's one gated entry; no STT model is gated
      (checked against the live API), so that side keeps refusing them.
- [x] Both halves of a gated model's setup are in the panel: `hf_access.GATED_REPOS`
      (Gemma 3 4B/12B/27B + `kyutai/pocket-tts`), a **Model** picker, **Open the model
      page...**, and *Check access* (`check_access`: smallest-file probe, plus whoami-v2 on
      a refusal, so "wrong token" and "licence not accepted" are told apart).
- [x] The Voice Lab's gated weight set is handled the same way: `engines.gated_repo`,
      a worker note when Pocket TTS loads *without* voice cloning, a clone refused with the
      page and the panel named (instead of the library's `hf auth login` advice), and
      `voicelab._error_text` translating any gated failure that reaches the application.

## Phase 20 — Keyboard gestures and the 2027 project polish ✅
- [x] `gestures.py`: the command catalog (Projects, Recording, Help, Application), a
      normalized `kb:<mods+key>` identifier, and a user map that **replaces** a command's
      defaults (an empty list unbinds it). Two commands asking for the same key are
      reported as a conflict, never resolved silently.
- [x] Settings > **Keyboard gestures** (`_InputGesturesPanel`): a category tree of commands,
      the selected command's gestures, Add gesture... (a key-capture dialog), Remove
      gesture and Restore defaults. Stealing a key from another command asks first.
- [x] The main window's key handler offers every key press to the configured gestures
      before the built-in shortcuts, and reloads them when Settings closes.
- [x] A transcription now shows a modal, cancellable progress window and offers
      *Open project folder* when it finishes (the recording flow's behaviour).
- [x] The Edit menu (and the recent-project context menu) follow the selected project:
      *Resume the project*, *Restart All Recording* / *Restart All Files*, *Start Selected
      Recording...* / *Start Selected Files...*.
- [x] *What's New* in the Help menu, backed by `docs/WHATS-NEW.html` (2026 -> 2027).
- [x] The New Project wizard opens with the **Project base** focused, since it decides
      which fields the page shows.
- [x] NeMo models pin `transformers<5`; the Canary-1B variant goes through `ASRModel` and
      Canary-Qwen 2.5B through the SALM loader, chosen per variant.

## Phase 21 — Audio description ✅
- [x] `describe/catalog.py`: nine vision-language model families (Qwen3-VL, Qwen2.5-VL,
      Qwen2.5-Omni, Qwen3-Omni, Gemma 3, Gemma 4, MiniCPM-o, SmolVLM2, Moondream) with the
      licence, notes, pip packages, CUDA-only packages, `trust_remote_code` and, per
      variant, the download size, the VRAM it wants and its cost per minute of video.
- [x] `describe/environments.py`: one Python environment per model
      (`describe_envs\<model id>`), the same shape as the STT one; a missing CUDA-only
      package (here `bitsandbytes`, the 4-bit quantiser) turns Install into **Repair**.
- [x] `describe/worker.py`: one standalone script, run by the model’s own interpreter, that
      samples frames with FFmpeg, skips a window whose picture has not changed, asks the
      model for one description and streams each finished piece back as `AIVS-SEGMENT`.
- [x] `describe/pipeline.py`: pure planning — chunks, **windows** (one description per
      twenty seconds, capped at sixty), estimates, the picture-change comparison and
      gap fitting (described as *where a description may be spoken*, tested exactly).
- [x] `describe/extended.py`: the extended plan (where the film stops, how long it is
      held, what the described film’s clock says) **and** both FFmpeg graphs: `render_*`
      stops the film with `tpad`/`apad`, `mix_*` mixes the narration over the original
      soundtrack with the picture copied (`-c:v copy`), each with a fallback chain.
- [x] **Three outputs per video**: the description document, the spoken narration and the
      described video (`<source> - described.mp4`) — a new file; the source is never
      written to.
- [x] Loader: explicit placement (`device_map={"": 0}`), a 4-bit `nf4` load when the
      weights exceed the card, a refusal with its reason when neither fits, and a crash
      note that turns `0xC0000005` into a sentence.
- [x] A missing FFmpeg is an offer to download it, asked off the GUI thread.
- [x] Docs: a full User Guide section, README/README.html/WHATS-NEW entries,
      `PROJECT_SPEC` §3.10, and **Part XI of the book** (chapters 89–94).
- [x] Verified end to end on this machine: Qwen3-VL 4B on the RTX 3060 and the Piper
      voice the user's project uses, through `DescribeJob` on a real three-minute lecture -
      three outputs (`.txt`, `- narration.wav`, `- described.mp4`), no warnings, 7
      descriptions of 10 windows, 155 s of which the mix was 6 s. The described film was
      probed afterwards: same duration and the *same* h264 1920x1080 50 fps 534 kb/s video
      stream as the source (`-c:v copy`), with the audio carrying the description.
- [x] The prompt budget was corrected by that measurement: a window's longer prompt (the
      change instruction plus the previous description) did not fit the card at four
      frames, and the driver spilled - 71 s a window against 10 s. `MAX_PROMPT_PATCHES` is
      now 6912 (three frames), and the same run takes 128 s instead of 477 s.
- [ ] The extended render graph has still not been executed on a real film (its syntax and
      its plan are covered by tests; the run above was standard placement).
- [ ] The 4-bit load against a real oversized checkpoint is still exercised with
      `from_pretrained` faked (section 14 of `AUDIO_DESCRIPTION_PLAN.md`).

- [x] OmniVoice ONNX was **dropped on 2026-10-14 instead of fixed**: the
      published export does not speak the text it is given (89% of its predicted
      codes survive an unrelated sentence, WER 0.95-1.00), the export's fp16
      `cuda/` backbone was downloaded (891 MB) and run and does not fix it
      either, and the application's own OmniVoice engines cover the model. The
      way back, if it is ever taken, is recorded here: `optimize.py` in the
      export repository is the Olive recipe that produced the published files
      (`torch>=2.10`, `transformers>=5.4`, `onnxruntime-genai`), so a corrected
      export could be reproduced on a machine with room for it and published to
      a repository an engine could download from.
- [x] IndicF5 was **dropped on the same day**: its checkpoint only loaded under
      the `transformers` the archive pins once the engine re-applied the
      published `model.safetensors` by hand, and an engine that must speak
      correctly should not stand on that.
- [ ] Manual NVDA walkthrough of the installer and the first-launch dialog.
- [ ] 32-bit build verification (a 32-bit Python 3.13 venv is required).
- [ ] Signing the installer so SmartScreen stops warning on first run.
- [ ] Get PR #440575 (`New package: AnujSharma.AIVoiceStudio version 2026.3.7`)
      merged. Steps 01-10 all passed on the second run (`c5d484de`, 2026-09-25);
      the first run failed at 08. Installation Validation because the manifest's
      silent switches were incomplete (the install waited for a click). It now
      waits on the community moderator approval the check-in policy requires.
      `SUBMISSION-STEPS.txt` has the runbook, the measured evidence and the traps.
- [ ] Ship the hardened installer with the next version. `installer_common.iss`
      keeps the install-mode choice (`PrivilegesRequired=lowest`,
      `PrivilegesRequiredOverridesAllowed=dialog`) with `DisableStartupPrompt=yes`,
      so an interactive install can still be elevated for all users while an
      unattended one takes the per-user path; verified by hand to install
      unattended and uninstall cleanly. Nothing to do beyond the normal version
      bump - do not replace a released asset.
- [x] A track priority for standard audio description, so a description is not buried
      by a film whose music and dialogue never stop
      (`constants.AD_TRACK_PRIORITY_CHOICES`). *Same volume* is the default and leaves
      both tracks as they were produced; *25% higher* and *50% higher* keep the
      description at the level the narration settings gave it and mix the film's own
      soundtrack at 0.75 and 0.5 (`extended.mix_graph(film_volume=...)` emits one
      `volume` stage on the film input and touches nothing else; the default emits
      none, so an old run's graph is unchanged byte for byte). Only the film's track
      moves, because raising the description would clip and would change a level the
      user had chosen by ear. The choice is made in the Audio description window,
      below Placement and above *Start audio description*, remembered in
      `settings['describe.track_priority']`, written into a new project by the wizard,
      recorded in the run's own record, and switched off with its reason under
      extended placement, where the two tracks are never heard together. Covered by
      tests: the graph and the gains, every mix attempt, the job's hand-off to the
      mixer, and the combo itself.
- [x] *Duck, don't lower*, a fourth track priority asked for on 2026-10-11: the film's own soundtrack
      keeps its level and is dipped to a quarter of it (about -12 dB) only while a
      description is actually being spoken (`AD_DUCK_LEVEL`,
      `extended.track_priority_duck`, `extended.ducking_expression`). The moments are
      measured, not guessed at: `narration.write_track()` records `turns` from the
      placed clips, the job hands them to the mixer, and descriptions closer than
      `AD_DUCK_MERGE_SECONDS` are merged into one dip. The gain is a per-frame
      FFmpeg `volume` expression - a trapezoid around every spoken description with
      `AD_DUCK_EDGE_SECONDS` either side - parsed once and evaluated once per audio
      frame, which cost 0.6 s of the 12.4 s a ten-minute film takes to mix. *Same
      volume* stays the default, so a project carrying no value is mixed exactly as
      it always was; so is a film whose list of moments would not fit on one command
      line (`AD_DUCK_MAX_EXPRESSION`), and the ducked mix is tried before the plain
      one so a dip that cannot be applied never costs the film its own sound.
      Verified against the app's own FFmpeg 9.0.2: the film alone is -30.1 dB, and
      through the dip it is -30.1 dB between descriptions and -42.1 dB while the
      voice speaks (12.0 dB down, the quarter asked for), with the picture copied
      rather than re-encoded; 1427 tests pass.
- [x] Re-verified that path against the 2027.1.0 setup on 2026-10-10: a silent
      install exited 0 (913 files, Start Menu entry, HKCU uninstall entry), the
      installed copy was checked to be the new build and reached its main
      window, and a silent uninstall removed every file, the entry and the
      shortcut. `packaging\silent_install.ps1` now drives both directions, so
      the switches cannot be lost to a shell: in Git Bash the hand-run
      `/VERYSILENT` arrives as `C:/Program Files/Git/VERYSILENT`, Setup then
      ignores it and the "unattended" install waits on its wizard. The driver
      also waits for the uninstaller, which exits 0 before it has finished.
- [x] Fixed the audio-description failure that hit anyone who reinstalled the
      application somewhere else: every per-model environment records the
      absolute path of the Python it was built from, and the ones made by the
      all-users install named
      `C:\Program Files\AI Voice Studio\python-runtime\python.exe`, which a
      per-user reinstall leaves behind. The worker then exited 103 -
      `did not find executable at '...': The system cannot find the path
      specified.` - and the dialog reported it as a model crash. Environments
      are now created from a per-user copy of the bundled runtime
      (`%APPDATA%\AIVoiceStudio\python_runtime`; 35 MB, measured at 2.2 s to
      copy, made once), and an environment that names a Python that has gone is
      re-pointed in place with `python -m venv --upgrade` rather than rebuilt -
      measured at 0.12 s, with `pip` and the installed packages untouched.
      `PythonRuntime.usable_python_exe` is what every worker launch now uses
      (describe, speech to text, Voice Lab, OmniVoice), so the repair happens
      before the run rather than as an error after it. Reproduced and verified
      on this machine: the real venv launcher fails with exit 103 and the exact
      message when its base is deleted, and a real runtime/environment built
      from the packaged runtime goes "venv -> delete the base -> repair ->
      `import torch` works again, `pip` intact". The environments that were
      broken here (four `describe_envs`, six `stt_envs`) were all named by the
      17:02 Program Files install.
- [x] Fixed `packaging\silent_install.ps1 -Uninstall` hanging on a *successful*
      uninstall: the bundled runtime writes `__pycache__` files beside its own
      modules the first time anything runs it (here: 47 `.pyc` files, because
      the repair probe ran the shipped runtime directly), Inno's uninstaller
      removes only the files Setup installed, so the wait for an empty folder
      ran its full 600 s and called a complete uninstall a timeout. The driver
      now recognises those bytecode caches, clears them and reports them. The
      artifacts were rebuilt after this change so the shipped setup matches the
      finished source (`tools\check_frozen_env_repair.py` re-run against it).
- [x] Fixed two screen-reader defects on 2026-10-11, both reported from a real
      NVDA session. **The window title never reached NVDA**: wxWidgets names a
      window after its own class, so `GetName()` answered "frame" for the main
      window and "dialog" for every dialog, `finalize_accessibility` copied that
      leftover into the MSAA accName, and NVDA+T announced "frame" instead of
      "AI Voice Studio". A top-level window's accName is now its title, read
      live (the describe window becomes "Audio description - <film>"), and a
      control the application never named is left unnamed instead of being
      announced as "listBox"/"panel"; `tests/test_a11y_names.py` holds both and
      sweeps every widget class the source instantiates against the list of
      names wxWidgets invents (`gui/a11y.py::_DEFAULT_WINDOW_NAMES`). **The
      executable carried no version resource**: NVDA's *report the application
      version* gesture asks Windows for the product name and product version,
      and a PyInstaller build without a VERSIONINFO resource answers nothing -
      "Application unknown, version not detected (AMD64)".
      `tools/make_version_info.py` builds that resource from
      `constants.APP_VERSION` (product name `AI Voice Studio`, product version
      `2027.1.0`, `OriginalFilename` `AI-Voice-Studio.exe`) and the spec hands it
      to the `EXE` target; `tests/test_packaging.py` checks the fields, the four
      number binary version and the built executable itself, and the tool reads
      any built exe back (`python tools/make_version_info.py <exe>`). Rebuilt the
      artifacts after the change, and verified the shipped
      `AI-Voice-Studio.exe` reports `AI Voice Studio 2027.1.0`.
- [x] Reordered *Track priority* on 2026-10-11 to read from the gentlest answer
      to the film to the most forceful - *Same volume*, *Duck, don't lower*,
      *Description 25% higher*, *Description 50% higher* - so the list opens on
      its default and the ducking sits with the other two choices rather than
      above them. Added a **Narration voice volume** slider directly under the
      narration voice combo in the describe window, on the recording side's own
      range: it sets the description's level (1.00 is the voice as downloaded),
      is carried into the run as `tts.volume`, remembered in
      `settings['describe.narration_volume']`, written into a new project by the
      wizard, and covered by tests for its opening value, its label and
      accessible name, its enable/disable with the voice, and the value the run
      is told.
