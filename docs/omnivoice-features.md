# OmniVoice in AI Voice Studio — full feature guide

Research summary for "enable all OmniVoice features in both the direct and the
server versions". Every claim below was checked against the authoritative
sources (September 2026):

| Project | Source | Role in the studio |
|---|---|---|
| [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) | GitHub README + docs | Base model + generation vocabulary |
| [newgrit1004/omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton) | GitHub README + `src/omnivoice_triton` source | **Direct** engine (`pip install omnivoice-triton`) |
| [maemreyo/omnivoice-server](https://github.com/maemreyo/omnivoice-server) | GitHub docs + published wheel `omnivoice-server 0.2.5` | **Server** engine (`pip install omnivoice-server`) |

The studio's two "versions" of OmniVoice are:

* **Direct** — engine id `omnivoice`; an `omnivoice-triton` runner lives in a
  worker subprocess (`ai_voice_studio/omnivoice/worker.py`) so its PyTorch /
  CUDA runtime never conflicts with sherpa-onnx.
* **Server** — engine id `omnivoice_server`; `omnivoice-server` runs as an
  OpenAI-compatible HTTP subprocess (`ai_voice_studio/omnivoice_server/`).

---

## 1. What OmniVoice actually is

OmniVoice is a massively multilingual, zero-shot TTS model from the k2-fsa
team (sherpa-onnx authors). Highlights:

* **600+ languages** — auto-detected from the text; you can pass a language
  hint (`en`, `zh`, `hi`, `vi`, ...) for pronunciation.
* **Three generation modes**: auto voice, voice cloning, voice design.
* **Voice design** controls the speaker through a small attribute vocabulary:
  gender, age, pitch, whisper style, English accent, Chinese dialect.
* **Voice cloning** copies a speaker from a 3–15 s reference sample. A
  transcript is optional: when omitted the engine auto-transcribes the sample
  with Whisper.
* **Fine-grained control inline in the text**: non-verbal symbols
  (`[laughter]`, `[breath]`, `[sigh]`, `[sniff]`, question / surprise /
  confirmation tags) and pronunciation hints (pinyin tone numbers for
  Chinese, CMU dictionary for English).
* **Generation parameters**: `num_step` (diffusion steps 1–64), `speed`,
  optional fixed `duration`, CFG `guidance_scale`, sampling temperatures and
  (server mode) a reproducible `seed`.

### Canonical voice-design attributes (verified)

| Group | Allowed values |
|---|---|
| Gender | `male`, `female` |
| Age | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| Pitch | `very low pitch` … `very high pitch` (5 steps) |
| Style | `whisper` |
| English accent | `american / british / australian / chinese / canadian / indian / korean / portuguese / russian / japanese accent` |
| Chinese dialect | `河南话, 陕西话, 四川话, 贵州话, 云南话, 桂林话, 济南话, 石家庄话, 甘肃话, 宁夏话, 青岛话, 东北话` |

Instructions are comma separated and freely combinable, e.g.
`female, young adult, high pitch, british accent, whisper`.

### Inline non-verbal tags (verified)

`[laughter]` `[breath]` `[sigh]` `[sniff]` `[confirmation-en]` `[question-en]`
`[question-ah]` `[question-oh]` `[question-ei]` `[question-yi]`
`[surprise-ah]` `[surprise-oh]` `[surprise-wa]` `[surprise-yo]`
`[dissatisfaction-hnn]`.

Caveat measured by the server project: a tag with fewer than ~8–10 words of
normal text around it can make OmniVoice emit a low drone with no speech
(reported via `X-No-Speech-Detected`). Tags outside the list are read as
literal text — they are not errors.

---

## 2. Direct engine (`omnivoice-triton`)

### API surface (read from the package source)

```python
from omnivoice_triton import create_runner
runner = create_runner("triton")   # "triton" | "hybrid" (also base/faster in the package)
runner.load_model()                # downloads ~2 GB from HuggingFace on first use
runner.generate(text, language=None, *, num_step=32, guidance_scale=2.0, class_temperature=0.0)
runner.generate_voice_clone(text, ref_audio, ref_text="", language=None, *, num_step=..., guidance_scale=..., class_temperature=...)
runner.generate_voice_design(text, instruct, language=None, *, num_step=..., guidance_scale=..., class_temperature=...)
runner.unload_model()
# result = {"audio": np.ndarray @24 kHz, "sample_rate": 24000, "time_s": ..., "peak_vram_gb": ...}
```

Facts that mattered for the integration:

* There is **no speaker id / sid** — catalog "voices" (auto/female/male/
  child/elderly) only make sense translated into *design instructions*.
* **`language` must be omitted (or `None`) for auto-detect** — the studio
  previously sent the string `"auto"`, which is not a language code.
* `num_step`/`guidance_scale`/`class_temperature` are applied through a
  `generation_config`; **`speed`/`duration` are not accepted by the runner
  methods**, so the studio applies `speed` natively when supported and
  otherwise approximates it by resampling (reported in the worker response).
* Different package versions accept different parameter sets → the worker
  introspects the runner method signature and forwards only supported
  parameters (`skipped` reported back, nothing crashes).
* `hybrid` mode (CUDA graph) is ~3.4x faster than stock but the project warns
  about a VRAM-leak bug; `triton` is the safe default.

### How the studio enables it (this feature set)

| Feature | Studio wiring |
|---|---|
| Auto / design / clone mode | Recording window → **OmniVoice voice options...** dialog; stored per project under `project.json → tts.omni` |
| Design attributes + free instructions | Dialog builder over the canonical vocabulary; engine sends `instruct` |
| Catalog voice → design mapping | `ai_voice_studio/omnivoice/spec.py` (`DIRECT_VOICE_INSTRUCT`): picking "Female voice" now really requests `female` |
| Cloning (ref audio + optional transcript) | Dialog file picker; blank transcript → Whisper auto-transcribes |
| `num_step`, `guidance_scale`, `class_temperature`, `duration`, `seed`, `language` | Request fields; worker filters by installed runner capability |
| Speed | Rate slider; native when supported, else resampled in worker |
| Non-verbal / pronunciation control | Pass-through — type them in the document text |
| Result metrics | Worker reports `time_ms` (from `time_s`) + `peak_vram_gb` |

Files: `ai_voice_studio/omnivoice/spec.py`, `worker.py`, `__init__.py` (engine
+ request builder + frozen worker copy), `ai_voice_studio/gui/
omnivoice_options_dialog.py`, Recording window integration, engine cache key
now includes the OmniVoice identity so two projects with different clones or
instructions never share one engine.

---

## 3. Server engine (`omnivoice-server`)

### API surface (read from the published 0.2.5 wheel, not just the docs)

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible TTS (JSON) |
| `POST /v1/audio/speech/clone` | One-shot cloning (multipart: `text`, `ref_audio`, `ref_text`…) |
| `GET /v1/voices` | Voices + `design_attributes` vocabulary |
| `POST /v1/voices/profiles` | Store a reusable clone profile (`profile_id`, `ref_audio`, `ref_text`, `overwrite`) |
| `GET/PATCH/DELETE /v1/voices/profiles/{id}` | Manage one profile |
| `GET /v1/models`, `GET /health`, `GET /metrics` | Models, health, Prometheus metrics |
| `/ui` | Built-in web UI |

`/v1/audio/speech` body fields (JSON): `model`, `input`, `voice`, `speaker`,
`instructions`, `response_format` (wav/pcm/…), `speed` (0.25–4.0), `stream`,
`num_step` (1–64), `guidance_scale` (0–10), `denoise`, `t_shift` (0–2),
`position_temperature` (0–10), `class_temperature` (0–2), `duration`
(0.1–60 s, overrides `speed`), `language`, `layer_penalty_factor`,
`preprocess_prompt`, `postprocess_output`, `audio_chunk_duration`,
`audio_chunk_threshold`, `request_timeout_s`, `seed`.

Parameter precedence: `instructions` (strongest) → `speaker` preset →
`voice` preset → server default design prompt
(`male, middle-aged, moderate pitch, british accent`).

Server-only extensions: 13 OpenAI preset aliases, sentence-level **streaming**
(raw PCM chunks, `response_format=pcm`), voice-profile storage, **named voice
files** (a `.txt` in the voice directory names a design + parameters and is
selectable by filename), bearer-token auth, CORS, concurrency pool, health +
metrics, seed-based reproducible output.

CLI (verified from the wheel — note the **model flag is `--model`**, the
docs' `--model-id` table is stale): `--host` `--port` (default **8880**)
`--device` (default `cpu`; `auto|cuda|mps|cpu`) `--num-step` `--guidance-scale`
`--denoise` `--t-shift` `--position-temperature` `--class-temperature`
`--seed` `--stream` `--max-concurrent` `--profile-dir` `--voice-dir`
`--api-key` `--cors-origins` … every option also has an `OMNIVOICE_` env var.

### How the studio enables it (this feature set)

| Feature | Studio wiring |
|---|---|
| All 13 official presets | `spec.OPENAI_VOICE_PRESETS` + catalog (`models_catalog.json`); descriptions corrected to the server's real prompts |
| Full per-request generation surface | `OmniVoiceServerManager.synthesize()` / `synthesize_clone()` now send every documented field (previously only a subset) |
| Voice design | `instructions` sent explicitly when set (strongest control) |
| Language hint | `language` field |
| Voice profiles (CRUD) | `save_profile / list_profiles / get_profile / delete_profile` client methods |
| Models / voice info | `get_models()`, `get_voice_info()` (incl. `design_attributes`) |
| Reproducible output | optional `seed` (per request), stable per voice |
| Clone via multipart | full optional fields (`num_step`, temperatures, `seed`, …) |
| Streaming | `stream=True` produces PCM chunks — the studio records whole files, so streaming stays a transport option for external API users (documented, not used by the recorder) |

Settings panel (existing "OmniVoice Server" category) controls host / port /
device / steps / concurrency / auth / CORS; per-project advanced knobs live in
the same **OmniVoice voice options** dialog as the direct engine, stored under
`tts.omni` and applied by the engine through
`spec.apply_omni_to_voice()`.

---

## 4. Shared feature vocabulary (single source of truth)

`ai_voice_studio/omnivoice/spec.py` holds everything both engines and the GUI
agree on:

* sample rate, defaults and ranges for every knob;
* `DIRECT_VOICE_INSTRUCT`, `OPENAI_VOICE_PRESETS` (13, official), the design
  attribute lists, the non-verbal tag list and language examples;
* `clean_language()` ("auto" → `None`), `resolve_instruct()` (explicit >
  catalog voice > empty), `filter_kwargs()` (version-adaptive calling),
  `build_omni()` (normalised per-project settings), and
  `apply_omni_to_voice()` (enrich a voice entry without mutating it).

## 5. Where settings live

Per-project options are stored in `project.json`:

```json
"tts": {
  "tts": "omnivoice", "variant": "triton", "voice": "auto",
  "rate": 1.0, "pitch": 1.0, "volume": 1.0,
  "omni": {
    "mode": "design",
    "instruct": "female, young adult, british accent",
    "language": null, "num_step": 32, "guidance_scale": 3.0,
    "class_temperature": 0.0, "position_temperature": 5.0,
    "denoise": true, "duration": null, "seed": null,
    "ref_audio": "", "ref_text": ""
  }
}
```

Defaults are chosen so that a project that never opens the options dialog
behaves exactly as before (auto voice, runner/server defaults). The dialog's
guidance-scale default is 3.0 (the studio's previous server behaviour).

## 6. What is intentionally not surfaced

* **Streaming playback preview** — the recorder writes files, it does not
  play; streaming stays an HTTP-API feature.
* **Voice-prompt files (`.pt`)** — `create_voice_clone_prompt()` reuse exists
  in the base model but needs the model in-process; not surfaced.
* **Web UI (`/ui`), FlashInfer, batch inference, training** — server/batch
  tooling, out of scope for a desktop recorder.
* **The 600+ language list** — auto-detect plus an optional free-text hint is
  the sensible surface; listing every language would add noise, not control.
