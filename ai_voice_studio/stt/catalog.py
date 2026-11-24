"""Speech-to-Text (STT) model catalog.

The catalog is the single description of every speech-to-text model the
application knows about.  It is deliberately plain Python (no ONNX download
metadata like the TTS catalog uses): an STT model is an ordinary Python
package that runs in one of the managed environments, so the ``Manage the STT``
Settings category and the transcribe widget both read their choices from here.

Every model entry carries what the transcribe UI needs:

* ``name`` / ``license`` / ``languages`` / ``fit`` / ``notes`` - the columns of
  the model table (Settings > Manage the STT).
* ``variants`` - the size/checkpoint choices the *variant* combo lists.
* ``backend`` / ``pip_packages`` - how the model is run and what has to be
  installed before it can run (see :mod:`ai_voice_studio.stt.engine`).
* ``translation`` / ``translation_targets`` - whether the model can translate
  speech and into which languages, which drives the translation combo.
* ``timestamps`` - whether the back-end really reports *when* each phrase was
  spoken.  transformers raises "We cannot return_timestamps yet on non-CTC
  models apart from Whisper!" for every audio language model, so this has to
  be declared per model rather than requested everywhere.
* ``trust_remote_code`` - whether the repository ships custom modelling code
  that only loads with ``trust_remote_code=True``.
* ``runner`` - *how* a transformers-backed model is driven.  The generic ASR
  pipeline cannot run every architecture, so a model says which of the
  worker's strategies it needs: ``pipeline`` (the default), ``chat`` (a
  processor that takes a chat prompt plus the audio, as Granite Speech does)
  or ``transcription_request`` (a processor with the audio-LLM request API,
  as Qwen3-ASR and Voxtral have).
* ``cuda_packages`` - pip packages the model's own environment additionally
  needs before it can use a CUDA device (an NVIDIA cuBLAS runtime, for
  instance).  They are a separate step from ``pip_packages`` because a
  CPU-only machine must not download them.
* ``fit`` - the device this model is *preferred* on: ``"cpu"`` or ``"gpu"``.
  It only decides the order of the transcribe dialog's compute combo (and the
  wording of the table's fit column); almost every model here can run on both
  a CPU and a CUDA GPU, so both are offered and the preferred one is selected
  first.  ``variant_fit`` overrides it for a single variant (Canary-Qwen's
  small checkpoint prefers the CPU, the 2.5B the GPU).
* ``gpu_only`` / ``gpu_only_variants`` - the models (or variants) that really
  cannot run without an NVIDIA GPU: a speech-language model too large to
  execute on a CPU at all.  Those are offered the GPU alone, never a CPU run
  that would only fail (see :func:`device_options`).

The rule the transcribe dialog follows is therefore: **only a model that
cannot run without a GPU shows the GPU alone; every other model offers both
the CPU and the GPU**, because many users have no NVIDIA card at all and must
still be able to transcribe.

A model whose repository is **gated** - one that only downloads after signing
in to Hugging Face and accepting a licence, such as Cohere Transcribe - is
deliberately *not* in this catalog.  Every model listed here runs with no
account, no token and no click-through, which is what the transcribe dialog
promises.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Whisper's 99 languages (code, spoken name).  Shared by the Whisper entries.
# ---------------------------------------------------------------------------
WHISPER_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Detect automatically"),
    ("en", "English"), ("zh", "Chinese"), ("de", "German"), ("es", "Spanish"),
    ("ru", "Russian"), ("ko", "Korean"), ("fr", "French"), ("ja", "Japanese"),
    ("pt", "Portuguese"), ("tr", "Turkish"), ("pl", "Polish"), ("ca", "Catalan"),
    ("nl", "Dutch"), ("ar", "Arabic"), ("sv", "Swedish"), ("it", "Italian"),
    ("id", "Indonesian"), ("hi", "Hindi"), ("fi", "Finnish"),
    ("vi", "Vietnamese"), ("he", "Hebrew"), ("uk", "Ukrainian"),
    ("el", "Greek"), ("ms", "Malay"), ("cs", "Czech"), ("ro", "Romanian"),
    ("da", "Danish"), ("hu", "Hungarian"), ("ta", "Tamil"), ("no", "Norwegian"),
    ("th", "Thai"), ("ur", "Urdu"), ("hr", "Croatian"), ("bg", "Bulgarian"),
    ("lt", "Lithuanian"), ("la", "Latin"), ("mi", "Maori"),
    ("ml", "Malayalam"), ("cy", "Welsh"), ("sk", "Slovak"), ("te", "Telugu"),
    ("fa", "Persian"), ("lv", "Latvian"), ("bn", "Bengali"), ("sr", "Serbian"),
    ("az", "Azerbaijani"), ("sl", "Slovenian"), ("kn", "Kannada"),
    ("et", "Estonian"), ("mk", "Macedonian"), ("br", "Breton"),
    ("eu", "Basque"), ("is", "Icelandic"), ("hy", "Armenian"), ("ne", "Nepali"),
    ("mn", "Mongolian"), ("bs", "Bosnian"), ("kk", "Kazakh"), ("sq", "Albanian"),
    ("sw", "Swahili"), ("gl", "Galician"), ("mr", "Marathi"), ("pa", "Punjabi"),
    ("si", "Sinhala"), ("km", "Khmer"), ("sn", "Shona"), ("yo", "Yoruba"),
    ("so", "Somali"), ("af", "Afrikaans"), ("oc", "Occitan"), ("ka", "Georgian"),
    ("be", "Belarusian"), ("tg", "Tajik"), ("sd", "Sindhi"), ("gu", "Gujarati"),
    ("am", "Amharic"), ("yi", "Yiddish"), ("lo", "Lao"), ("uz", "Uzbek"),
    ("fo", "Faroese"), ("ht", "Haitian Creole"), ("ps", "Pashto"),
    ("tk", "Turkmen"), ("nn", "Nynorsk"), ("mt", "Maltese"), ("sa", "Sanskrit"),
    ("lb", "Luxembourgish"), ("my", "Burmese"), ("bo", "Tibetan"),
    ("tl", "Tagalog"), ("mg", "Malagasy"), ("as", "Assamese"), ("tt", "Tatar"),
    ("haw", "Hawaiian"), ("ln", "Lingala"), ("ha", "Hausa"), ("ba", "Bashkir"),
    ("jw", "Javanese"), ("su", "Sundanese"), ("yue", "Cantonese"),
]

#: Canary's 25 European languages.
_CANARY_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Detect automatically"),
    ("bg", "Bulgarian"), ("hr", "Croatian"), ("cs", "Czech"), ("da", "Danish"),
    ("nl", "Dutch"), ("en", "English"), ("et", "Estonian"), ("fi", "Finnish"),
    ("fr", "French"), ("de", "German"), ("el", "Greek"), ("hu", "Hungarian"),
    ("it", "Italian"), ("lv", "Latvian"), ("lt", "Lithuanian"), ("mt", "Maltese"),
    ("pl", "Polish"), ("pt", "Portuguese"), ("ro", "Romanian"), ("sk", "Slovak"),
    ("sl", "Slovenian"), ("es", "Spanish"), ("sv", "Swedish"), ("ru", "Russian"),
    ("uk", "Ukrainian"),
]

#: Qwen3-ASR's main languages (52 languages and dialects in total).
_QWEN_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Detect automatically"),
    ("zh", "Chinese (Mandarin)"), ("yue", "Chinese (Cantonese)"),
    ("en", "English"), ("ja", "Japanese"), ("ko", "Korean"), ("fr", "French"),
    ("de", "German"), ("es", "Spanish"), ("it", "Italian"), ("pt", "Portuguese"),
    ("ru", "Russian"), ("ar", "Arabic"), ("hi", "Hindi"), ("bn", "Bengali"),
    ("ta", "Tamil"), ("te", "Telugu"), ("ur", "Urdu"), ("id", "Indonesian"),
    ("ms", "Malay"), ("th", "Thai"), ("vi", "Vietnamese"), ("tr", "Turkish"),
    ("pl", "Polish"), ("nl", "Dutch"), ("sv", "Swedish"), ("uk", "Ukrainian"),
]

#: Voxtral's documented languages (13 in the newest release).
_VOXTRAL_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Detect automatically"),
    ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("pt", "Portuguese"),
    ("hi", "Hindi"), ("de", "German"), ("nl", "Dutch"), ("it", "Italian"),
    ("zh", "Chinese"), ("ar", "Arabic"), ("ru", "Russian"), ("ja", "Japanese"),
]

#: Granite Speech's languages.
_GRANITE_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Detect automatically"),
    ("en", "English"), ("fr", "French"), ("de", "German"), ("es", "Spanish"),
    ("pt", "Portuguese"), ("ja", "Japanese"),
]

_ENGLISH_ONLY: list[tuple[str, str]] = [("en", "English")]

#: English translation target used by the models that translate into English.
_ENGLISH_TARGET: list[tuple[str, str]] = [("en", "English")]

#: Back-end id -> the importable module that proves it is installed in a
#: model's own environment.  Used for availability checks and the post-install
#: import verification.
BACKEND_MODULES: dict[str, str] = {
    "faster_whisper": "faster_whisper",
    "whisper": "whisper",
    "transformers": "transformers",
    "nemo": "nemo",
    "speechbrain": "speechbrain",
}

#: How carefully a model decodes.  Every model offers the same three levels;
#: the label under the combo shows the model's own reported accuracy.
ACCURACY_FAST = "fast"
ACCURACY_BALANCED = "balanced"
ACCURACY_BEST = "best"
ACCURACY_CHOICES: list[tuple[str, str]] = [
    (ACCURACY_FAST, "Fast (greedy decoding, quickest)"),
    (ACCURACY_BALANCED, "Balanced (beam search, recommended)"),
    (ACCURACY_BEST, "Most accurate (wide beam search, slowest)"),
]
DEFAULT_ACCURACY = ACCURACY_BALANCED

#: Beam sizes the accuracy level maps to for the supported back-ends.
BEAM_SIZES = {ACCURACY_FAST: 1, ACCURACY_BALANCED: 5, ACCURACY_BEST: 8}

#: How a transformers-backed model is driven.  The worker mirrors these names
#: (it cannot import this package), so they are the contract between the two.
RUNNER_PIPELINE = "pipeline"
RUNNER_TRANSCRIPTION_REQUEST = "transcription_request"
RUNNER_CHAT = "chat"

#: How a NeMo-backed model is loaded.  A plain ASR model has a ``.nemo``
#: checkpoint and is opened with ``ASRModel.from_pretrained``; a speech-language
#: model (SALM) is a transformers-style repository and must not be sent to the
#: ASR entry point at all.
RUNNER_ASR = "asr"
RUNNER_SALM = "salm"

#: NeMo 3.0 predates transformers 5 and crashes with it, so every NeMo model
#: pins the last compatible major series.
_NEMO_TRANSFORMERS = "transformers<5"

#: The device a model (or one of its variants) is preferred on.  This is only
#: a preference - both devices are offered wherever the model can run on both.
#: A model that genuinely needs a GPU says so with ``gpu_only`` instead.
FIT_CPU = "cpu"
FIT_GPU = "gpu"

#: Fit label shown in Settings > Manage the STT for a model that runs on both
#: devices (variant-specific wording is set per entry where it matters).
_FIT_BOTH_GPU_FIRST = "\u2705 GPU or CPU"
_FIT_BOTH_CPU_FIRST = "\u2705 CPU or GPU"


def _whisper_variants() -> list[tuple[str, str]]:
    return [
        ("large-v3", "Large v3 (most accurate)"),
        ("large-v3-turbo", "Large v3 Turbo (fast, near-large accuracy)"),
        ("large-v2", "Large v2"),
        ("medium", "Medium"),
        ("small", "Small"),
        ("base", "Base"),
        ("tiny", "Tiny (fastest)"),
    ]


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------
MODELS: list[dict] = [
    {
        "id": "whisper_large_v3",
        "name": "Whisper Large V3",
        "license": "MIT",
        "languages_label": "99+",
        "languages": WHISPER_LANGUAGES,
        "fit": "gpu",
        "fit_label": _FIT_BOTH_GPU_FIRST,
        "notes": "Multilingual and accurate; runs on both a GPU (~10-12 GB "
                 "VRAM for Large) and a CPU",
        "variants": _whisper_variants(),
        "default_variant": "large-v3",
        "accuracy_hint": "~10.3% WER (large-v3)",
        "backend": "faster_whisper",
        "pip_packages": ["faster-whisper"],
        # CTranslate2 is a CUDA build on Windows: without the cuBLAS runtime
        # from these wheels every run failed with "Library cublas64_12.dll is
        # not found or cannot be loaded".  The wheel ships cuDNN itself.
        "cuda_packages": ["nvidia-cublas-cu12"],
        "timestamps": True,
        "alt_backends": ["whisper", "transformers"],
        "translation": True,
        "translation_targets": _ENGLISH_TARGET,
    },
    {
        "id": "whisper_small_base",
        "name": "Whisper Small/Base (CPU)",
        "license": "MIT",
        "languages_label": "99+",
        "languages": WHISPER_LANGUAGES,
        "fit": "cpu",
        "fit_label": _FIT_BOTH_CPU_FIRST,
        "notes": "Lighter versions of Whisper; fast on the CPU and quicker "
                 "still on a GPU",
        "variants": [
            ("small", "Small (recommended on CPU)"),
            ("base", "Base"),
            ("tiny", "Tiny (fastest)"),
        ],
        "default_variant": "small",
        "accuracy_hint": "~12% WER (small); base/tiny are faster but less exact",
        "backend": "faster_whisper",
        "pip_packages": ["faster-whisper"],
        "cuda_packages": ["nvidia-cublas-cu12"],
        "timestamps": True,
        "alt_backends": ["whisper", "transformers"],
        "translation": True,
        "translation_targets": _ENGLISH_TARGET,
    },
    {
        "id": "qwen3_asr",
        "name": "Qwen3-ASR",
        "license": "Apache 2.0",
        "languages_label": "52",
        "languages": _QWEN_LANGUAGES,
        "fit": "gpu",
        "fit_label": _FIT_BOTH_GPU_FIRST,
        "notes": "Multilingual; runs fastest on a mid-range GPU, works on a "
                 "CPU too",
        "variants": [
            ("1.7b", "Qwen3-ASR-1.7B (most accurate)"),
            ("0.6b", "Qwen3-ASR-0.6B (faster)"),
        ],
        "default_variant": "1.7b",
        "accuracy_hint": "state of the art on 52 languages and dialects",
        "backend": "transformers",
        "runner": RUNNER_TRANSCRIPTION_REQUEST,
        # ``accelerate`` is what lets a model be placed with ``device_map``,
        # which is how the audio language models are loaded.
        "pip_packages": ["transformers", "torch", "accelerate", "librosa",
                         "soundfile"],
        # An audio language model: it transcribes through its own processor,
        # and transformers refuses to return timestamps for it.
        "timestamps": False,
        "alt_backends": [],
        "translation": True,
        "translation_targets": _ENGLISH_TARGET,
    },
    {
        "id": "canary_qwen",
        "name": "NVIDIA Canary-Qwen",
        "license": "Apache 2.0",
        "languages_label": "25",
        "languages": _CANARY_LANGUAGES,
        "fit": "gpu",
        "fit_label": "\u2705 1B: CPU or GPU \u00b7 2.5B: GPU only",
        "notes": "High accuracy, multilingual.  The 2.5B is a speech-language "
                 "model and needs a GPU; the 1B runs on a CPU",
        # The two checkpoints are different kinds of model: the 2.5B is a SALM
        # that has to have a GPU, while Canary-1B is an ordinary ASR model that
        # runs happily on a CPU.
        "variant_fit": {"1b": "cpu"},
        "gpu_only_variants": ["2.5b"],
        "variants": [
            ("2.5b", "Canary-Qwen-2.5B (recommended, GPU)"),
            ("1b", "Canary-1B (smaller, CPU or GPU)"),
        ],
        "default_variant": "2.5b",
        "accuracy_hint": "leading English WER; strong across 25 European languages",
        "backend": "nemo",
        # The 2.5B is a SALM (a speech-augmented language model), which NeMo
        # builds with a LoRA adapter - hence ``peft``; the 1B is an ordinary
        # Canary ASR model with a ``.nemo`` checkpoint.  NeMo 3.0 needs
        # transformers below 5 (with 5.x the SALM load segfaults).
        "runners": {"2.5b": RUNNER_SALM, "1b": RUNNER_ASR},
        "pip_packages": ["nemo_toolkit[asr]", "torch", "peft",
                         _NEMO_TRANSFORMERS],
        "timestamps": True,
        "alt_backends": [],
        "translation": True,
        "translation_targets": _ENGLISH_TARGET,
    },
    {
        "id": "parakeet_tdt",
        "name": "NVIDIA Parakeet TDT",
        "license": "Apache 2.0",
        "languages_label": "English",
        "languages": _ENGLISH_ONLY,
        "fit": "gpu",
        "fit_label": _FIT_BOTH_GPU_FIRST,
        "notes": "Real-time streaming, low latency; runs on a GPU or a CPU",
        "variants": [
            ("0.6b-v2", "Parakeet TDT 0.6B v2 (English)"),
            ("0.6b", "Parakeet TDT 0.6B"),
        ],
        "default_variant": "0.6b-v2",
        "accuracy_hint": "very low English WER with low latency",
        "backend": "nemo",
        "pip_packages": ["nemo_toolkit[asr]", "torch", _NEMO_TRANSFORMERS],
        "timestamps": True,
        "alt_backends": [],
        "translation": False,
        "translation_targets": [],
    },
    {
        "id": "moonshine",
        "name": "Moonshine (CPU)",
        "license": "MIT",
        "languages_label": "English (expanding)",
        "languages": _ENGLISH_ONLY,
        "fit": "cpu",
        "fit_label": _FIT_BOTH_CPU_FIRST,
        "notes": "Lightweight (27M-61M params) and fast on the CPU, built for "
                 "edge devices and NVDA add-ons; a GPU also works",
        "variants": [
            ("base", "Base (61M, recommended)"),
            ("tiny", "Tiny (27M, fastest)"),
        ],
        "default_variant": "base",
        "accuracy_hint": "close to Whisper on English, much faster on CPU",
        # Moonshine runs through transformers: the ``useful-moonshine`` wheel
        # pins ``torch==2.4.1``, which has no wheel for the Python this
        # application builds its environments with, so the install failed.
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "soundfile"],
        "timestamps": False,
        "alt_backends": [],
        "translation": False,
        "translation_targets": [],
    },
    {
        "id": "speechbrain",
        "name": "SpeechBrain (CPU)",
        "license": "Apache 2.0",
        "languages_label": "Multi",
        "languages": [
            ("auto", "Detect automatically"),
            ("en", "English"), ("hi", "Hindi"), ("de", "German"),
            ("es", "Spanish"), ("fr", "French"), ("it", "Italian"),
        ],
        "fit": "cpu",
        "fit_label": _FIT_BOTH_CPU_FIRST,
        "notes": "Toolkit, modular, runs on CPU (or a GPU); good for custom "
                 "Hindi/Indian-accent training",
        "variants": [
            ("base", "SpeechBrain ASR (base, recommended)"),
            ("hindi", "Hindi / Indian-accent recipe"),
        ],
        "default_variant": "base",
        "accuracy_hint": "depends on the recipe; tunable on your own data",
        "backend": "speechbrain",
        "pip_packages": ["speechbrain", "torch", "torchaudio"],
        "timestamps": False,
        "alt_backends": [],
        "translation": False,
        "translation_targets": [],
    },
    {
        "id": "voxtral",
        "name": "Voxtral",
        "license": "Apache 2.0",
        "languages_label": "Multilingual",
        "languages": _VOXTRAL_LANGUAGES,
        "fit": "gpu",
        "fit_label": _FIT_BOTH_GPU_FIRST,
        "notes": "Designed for production pipelines, multilingual; the 3B "
                 "model needs ~9.5 GB VRAM for the GPU path",
        "variants": [
            ("mini-3b", "Voxtral Mini 3B"),
        ],
        "default_variant": "mini-3b",
        "accuracy_hint": "state of the art for transcription and translation",
        "backend": "transformers",
        "runner": RUNNER_TRANSCRIPTION_REQUEST,
        # Voxtral's processor reads the recording itself (``load_audio_as``),
        # which resamples with librosa.
        "pip_packages": ["transformers", "torch", "accelerate",
                         "mistral-common", "librosa", "soundfile"],
        "timestamps": False,
        "alt_backends": [],
        "translation": True,
        "translation_targets": _VOXTRAL_LANGUAGES[1:],
    },
    {
        "id": "granite_speech",
        "name": "IBM Granite Speech (CPU)",
        "license": "Apache 2.0",
        "languages_label": "English +7",
        "languages": _GRANITE_LANGUAGES,
        "fit": "cpu",
        "fit_label": "\u2705 1B/2B: CPU or GPU \u00b7 8B: GPU only",
        "notes": "Very large.  The 1B and 2B run on a CPU; the 8B is ~17 GB "
                 "of weights and needs a GPU",
        # The 8B checkpoint is ~17 GB of weights: it cannot be held in the RAM
        # of an ordinary machine, so it is a GPU-only choice.
        "variant_fit": {"8b": "gpu"},
        "gpu_only_variants": ["8b"],
        "variants": [
            ("1b", "Granite 4.0 1B Speech (~4.6 GB, CPU or GPU)"),
            ("2b", "Granite Speech 3.3 2B (~6 GB, CPU or GPU)"),
            ("8b", "Granite Speech 3.3 8B (~17 GB, GPU only)"),
        ],
        "default_variant": "1b",
        "accuracy_hint": "strong English ASR and speech translation",
        "backend": "transformers",
        # Granite Speech is driven by a chat prompt plus the audio, and its
        # feature extractor rejects the ASR pipeline's ``sampling_rate``.
        "runner": RUNNER_CHAT,
        # The 3.3 revisions ship their speech alignment as a LoRA adapter
        # (``has_lora_adapter`` in their config), which transformers applies
        # through ``peft``; without it the checkpoints do not load.
        "pip_packages": ["transformers", "torch", "torchaudio", "peft",
                         "soundfile"],
        "timestamps": False,
        "alt_backends": [],
        "translation": True,
        "translation_targets": _ENGLISH_TARGET,
    },
]

_BY_ID: dict[str, dict] = {m["id"]: m for m in MODELS}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def models() -> list[dict]:
    """Every catalog entry (in table order)."""
    return MODELS


def model(model_id: str) -> dict:
    """One catalog entry; raises ``KeyError`` for an unknown id."""
    return _BY_ID[model_id]


def model_ids() -> list[str]:
    return [m["id"] for m in MODELS]


def variants(model_id: str) -> list[tuple[str, str]]:
    return list(model(model_id)["variants"])


def default_variant(model_id: str) -> str:
    return model(model_id)["default_variant"]


def languages(model_id: str) -> list[tuple[str, str]]:
    return list(model(model_id)["languages"])


def supports_translation(model_id: str) -> bool:
    return bool(model(model_id)["translation"])


def translation_targets(model_id: str) -> list[tuple[str, str]]:
    return list(model(model_id).get("translation_targets", []))


def accuracy_hint(model_id: str) -> str:
    return model(model_id).get("accuracy_hint", "")


def pip_packages(model_id: str) -> list[str]:
    return list(model(model_id)["pip_packages"])


def cuda_packages(model_id: str) -> list[str]:
    """Extra packages the model needs *only* to use a CUDA device."""
    return list(model(model_id).get("cuda_packages", []))


def supports_timestamps(model_id: str) -> bool:
    """Whether this model's back-end can report when each phrase was spoken.

    transformers raises "We cannot return_timestamps yet on non-CTC models
    apart from Whisper!" for every audio language model, so a model that has
    no timings says so here rather than failing at transcribe time.
    """
    return bool(model(model_id).get("timestamps", False))


def needs_remote_code(model_id: str) -> bool:
    """Whether the model's repository ships code that must be trusted."""
    return bool(model(model_id).get("trust_remote_code", False))


def runner(model_id: str) -> str:
    """How the worker should drive a transformers-backed model."""
    return runner_for(model_id)


def runner_for(model_id: str, variant: str = "") -> str:
    """How the worker should drive this model (and variant, when given).

    A per-variant ``runners`` map wins: one catalog entry can offer both a
    NeMo ASR checkpoint and a NeMo speech-language model, which are loaded by
    completely different classes.  Otherwise the model-level ``runner`` (or the
    generic pipeline) applies.
    """
    entry = model(model_id)
    runners = entry.get("runners") or {}
    if variant and variant in runners:
        return str(runners[variant])
    return str(entry.get("runner") or RUNNER_PIPELINE)


def backend(model_id: str) -> str:
    return model(model_id)["backend"]


def backend_modules(model_id: str) -> list[str]:
    """Import modules that prove this model can run, primary back-end first.

    The first entry is the model's own back-end (what the installer provides);
    the rest are the alternative back-ends that can also drive it, so a model
    is usable the moment any of them is installed in its environment.
    """
    entry = model(model_id)
    ids = [entry["backend"], *entry.get("alt_backends", [])]
    return [BACKEND_MODULES[b] for b in ids if b in BACKEND_MODULES]


def fit(model_id: str, variant: str = "") -> str:
    """The device this model (or one variant) is *preferred* on.

    ``"cpu"`` or ``"gpu"``.  It is only a preference: it decides which option
    the transcribe dialog selects first and how the table's fit column reads,
    never which devices are offered (see :func:`device_options`).  A variant's
    ``variant_fit`` overrides the model-level value, because one catalog entry
    can hold both a GPU-only checkpoint and a CPU-friendly one.
    """
    entry = model(model_id)
    overrides = entry.get("variant_fit") or {}
    if variant and variant in overrides:
        return str(overrides[variant])
    return str(entry.get("fit") or FIT_CPU)


def gpu_only(model_id: str, variant: str = "") -> bool:
    """Whether this model (or variant) genuinely cannot run without a GPU.

    A speech-language model whose weights alone exceed the RAM of an ordinary
    machine has no usable CPU path, so offering a CPU run would only set the
    user up for a failure.
    """
    entry = model(model_id)
    if entry.get("gpu_only"):
        return True
    return bool(variant) and variant in (entry.get("gpu_only_variants") or [])


def gpu_fit_model_ids() -> list[str]:
    """Ids of the models preferred on a CUDA GPU."""
    return [m["id"] for m in MODELS if m.get("fit") == FIT_GPU]


def device_options(model_id: str, variant: str = "",
                   has_cuda: bool = True) -> list[tuple[str, str]]:
    """Compute choices the transcribe widget may offer for this model/variant.

    The rule is deliberately generous, because many users have no NVIDIA card
    at all:

    * a **GPU-only** model (or variant) is offered the GPU alone - a CPU run
      could not work, so it is not dangled in front of the user.  On a machine
      with no NVIDIA card the single option still says a GPU is required,
      so the dialog can explain why the run cannot start.
    * **every other model** is offered *both* the CPU and the GPU whenever the
      machine has an NVIDIA card (the preferred device first).  On a machine
      with no card the CPU is the only option, because a GPU choice that
      cannot run is worse than none.

    ``Auto`` is deliberately never offered: it would quietly pick a device
    rather than let the user decide, and this combo is about a model's device,
    not the machine's preference.
    """
    if gpu_only(model_id, variant):
        if has_cuda:
            return [("cuda", "GPU (CUDA)")]
        return [
            (
                "cuda",
                "GPU (CUDA) - no NVIDIA GPU detected, and this model "
                "cannot run on a CPU",
            )
        ]
    if not has_cuda:
        label = "CPU"
        if fit(model_id, variant) == FIT_GPU:
            label = (
                "CPU (no NVIDIA GPU detected - this model runs on the CPU, "
                "but it is faster on a GPU)"
            )
        return [("cpu", label)]
    gpu_option = ("cuda", "GPU (CUDA)")
    cpu_option = ("cpu", "CPU")
    if fit(model_id, variant) == FIT_GPU:
        return [gpu_option, cpu_option]
    return [cpu_option, gpu_option]


def summary_rows() -> list[tuple[str, str, str, str, str]]:
    """The five columns of the Settings > Manage the STT table."""
    return [
        (m["name"], m["license"], m["languages_label"], m["fit_label"], m["notes"])
        for m in MODELS
    ]
