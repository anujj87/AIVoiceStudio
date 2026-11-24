"""Per-engine tuning options for the Voice Lab engines.

Each engine has its own generation knobs: F5-TTS offers diffusion steps and a
CFG strength, Pocket TTS can quantize its model to int8 for a much faster CPU
run, and the transformer engines (Chatterbox, Qwen3-TTS, VibeVoice) their own
sampling controls.

This module is the single source of truth for those knobs, so the Settings
panel (per-engine defaults), the Recording window (per-project overrides), the
worker (which applies them) and the tests all agree on one vocabulary:

* the option *keys* are the argument names the engines themselves use, so the
  worker can pass a dict straight through after filtering it against the
  installed engine's signature,
* the *default* of every option is the engine's own default, and
* only values that **differ from the default** are stored and sent.

That last rule is what keeps a project honest: a project that never touched the
tuning dialog sends no options at all and therefore behaves exactly like a
freshly installed engine, and "reset to defaults" simply clears the overrides.

Nothing here imports the heavy engine packages, so the module is safe to use in
the GUI thread and in unit tests.  Options an installed engine version cannot
apply are skipped and reported by the worker instead of failing a recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# -- option kinds (the GUI picks a widget for each) --------------------------
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_BOOL = "bool"
KIND_TEXT = "text"
KIND_CHOICE = "choice"
KIND_SEED = "seed"


@dataclass(frozen=True)
class Option:
    """One tuning knob of one engine."""

    key: str
    label: str
    kind: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    increment: float = 0.1
    choices: tuple[str, ...] = ()
    help: str = ""
    #: Load-time options change which model the worker loads; they take part in
    #: the worker's backend cache key so switching them really reloads.
    load_time: bool = False

    @property
    def optional(self) -> bool:
        """True when "not set" is a meaningful value (blank = engine default)."""
        return self.kind in (KIND_TEXT, KIND_SEED) or self.default is None


# The shared RNG seed: supported by every engine (natively where the engine has
# a seed argument, otherwise by seeding Torch/NumPy before generation).
_SEED = Option(
    key="seed",
    label="Random seed",
    kind=KIND_SEED,
    default=None,
    help=(
        "Optional RNG seed. The same text and seed reproduce the same audio; "
        "leave it empty for fresh randomness."
    ),
)


_OPTIONS: dict[str, tuple[Option, ...]] = {
    # ------------------------------------------------------------------ Kyutai
    "pocket_tts": (
        Option(
            key="quantize",
            label="Quantize the model to int8 (faster on CPU)",
            kind=KIND_BOOL,
            default=False,
            load_time=True,
            help=(
                "Dynamic int8 quantization: a much smaller and faster model on "
                "a CPU with a small quality cost. Ignored on the GPU."
            ),
        ),
        _SEED,
    ),
    # ------------------------------------------------------------------ F5-TTS
    "f5tts": (
        Option(
            key="nfe_step",
            label="Diffusion steps (NFE)",
            kind=KIND_INT,
            default=32,
            minimum=4,
            maximum=64,
            increment=1,
            help=(
                "Function evaluations of the flow-matching sampler. 16 is "
                "fast, 32 is the engine default, 64 is the highest quality."
            ),
        ),
        Option(
            key="cfg_strength",
            label="CFG strength",
            kind=KIND_FLOAT,
            default=2.0,
            minimum=0.0,
            maximum=5.0,
            increment=0.1,
            help=(
                "How strongly the reference voice is followed. Higher = closer "
                "to the reference (and less stable above ~3)."
            ),
        ),
        Option(
            key="sway_sampling_coef",
            label="Sway sampling coefficient",
            kind=KIND_FLOAT,
            default=-1.0,
            minimum=-2.0,
            maximum=2.0,
            increment=0.1,
            help=(
                "Sway-sampling shift of the flow-matching schedule. -1 is the "
                "engine default; 0 disables sway sampling."
            ),
        ),
        Option(
            key="cross_fade_duration",
            label="Cross-fade between segments (seconds)",
            kind=KIND_FLOAT,
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help="Smoothing used when a long text is generated in batches.",
        ),
        Option(
            key="target_rms",
            label="Target loudness (RMS)",
            kind=KIND_FLOAT,
            default=0.1,
            minimum=0.0,
            maximum=1.0,
            increment=0.01,
            help="Loudness the reference clip is normalised to (0 = leave it).",
        ),
        Option(
            key="fix_duration",
            label="Fixed duration (seconds, 0 = auto)",
            kind=KIND_FLOAT,
            default=0.0,
            minimum=0.0,
            maximum=60.0,
            increment=0.5,
            help=(
                "Force the output to a fixed length. Mostly a speed-control "
                "trick for very short lines; 0 lets the engine decide."
            ),
        ),
        Option(
            key="remove_silence",
            label="Trim silence",
            kind=KIND_BOOL,
            default=False,
            help="Remove the leading/trailing silence from the generated audio.",
        ),
        _SEED,
    ),
    # --------------------------------------------------------------- KokoClone
    # KokoClone's own call (``generate(text, lang, reference_audio,
    # output_path)``) has no generation knobs: the Kokoro stage is a single
    # deterministic ONNX pass and the Kanade conversion has no sampler.  The
    # seed is still offered - the worker seeds Torch/NumPy with it, so the same
    # text and seed reproduce the same audio.
    "kokoclone": (
        _SEED,
    ),
    # ------------------------------------------------------------- Chatterbox
    # Chatterbox's own ``generate`` takes these.  ``exaggeration`` is the
    # model's expressiveness dial and ``cfg_weight`` how closely the reference
    # is followed; the 350M Turbo model takes neither, and the worker reports
    # them as skipped rather than failing the run.
    "chatterbox": (
        Option(
            key="exaggeration",
            label="Expressiveness (exaggeration)",
            kind=KIND_FLOAT,
            default=0.5,
            minimum=0.0,
            maximum=2.0,
            increment=0.05,
            help=(
                "How dramatic the delivery is. 0.5 is the engine's default; "
                "0.7 or more suits expressive text, and higher values speak "
                "faster."
            ),
        ),
        Option(
            key="cfg_weight",
            label="Reference guidance (CFG weight)",
            kind=KIND_FLOAT,
            default=0.5,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help=(
                "How strongly the voice is followed. 0.5 is the default; "
                "about 0.3 gives a fast-talking reference a calmer pace, and "
                "0.0 ignores its accent when the text is another language."
            ),
        ),
        Option(
            key="temperature",
            label="Sampling temperature",
            kind=KIND_FLOAT,
            default=0.8,
            minimum=0.05,
            maximum=2.0,
            increment=0.05,
            help="Lower is steadier and more predictable, higher is livelier.",
        ),
        Option(
            key="repetition_penalty",
            label="Repetition penalty",
            kind=KIND_FLOAT,
            default=1.2,
            minimum=1.0,
            maximum=2.0,
            increment=0.05,
            help="Raises the cost of repeating what was just said.",
        ),
        Option(
            key="min_p",
            label="Nucleus floor (min-p)",
            kind=KIND_FLOAT,
            default=0.05,
            minimum=0.0,
            maximum=1.0,
            increment=0.01,
            help="Keeps only tokens above this share of the best one.",
        ),
        Option(
            key="top_p",
            label="Nucleus sampling (top-p)",
            kind=KIND_FLOAT,
            default=1.0,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help="1.0 disables it; lower values narrow the choices.",
        ),
        _SEED,
    ),
    # ------------------------------------------------------------- Qwen3-TTS
    "qwen3tts": (
        Option(
            key="instruct",
            label="Style instruction",
            kind=KIND_TEXT,
            default=None,
            help=(
                "A plain-language description of how to speak, for example "
                "'Speak slowly and calmly, as if reading a bedtime story.' "
                "Leave it empty to let the model decide from the text."
            ),
        ),
        Option(
            key="temperature",
            label="Sampling temperature",
            kind=KIND_FLOAT,
            default=0.7,
            minimum=0.05,
            maximum=2.0,
            increment=0.05,
            help="Lower is steadier and more predictable, higher is livelier.",
        ),
        Option(
            key="top_p",
            label="Nucleus sampling (top-p)",
            kind=KIND_FLOAT,
            default=0.8,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help="Keeps the most likely tokens that together reach this share.",
        ),
        Option(
            key="top_k",
            label="Top-k sampling",
            kind=KIND_INT,
            default=50,
            minimum=1,
            maximum=200,
            increment=1,
            help="How many candidate tokens each step may choose from.",
        ),
        Option(
            key="repetition_penalty",
            label="Repetition penalty",
            kind=KIND_FLOAT,
            default=1.05,
            minimum=1.0,
            maximum=2.0,
            increment=0.05,
            help="Raises the cost of repeating what was just said.",
        ),
        Option(
            key="max_new_tokens",
            label="Longest audio (tokens)",
            kind=KIND_INT,
            default=4096,
            minimum=256,
            maximum=8192,
            increment=256,
            help=(
                "Upper bound on the speech tokens one call may generate. "
                "Raise it only if a long segment is cut off."
            ),
        ),
        Option(
            key="x_vector_only_mode",
            label="Clone the timbre only (no transcript needed)",
            kind=KIND_BOOL,
            default=False,
            help=(
                "With a transcript of the reference recording the clone is "
                "more faithful; without one, this copies the timbre alone."
            ),
        ),
        _SEED,
    ),
    # --------------------------------------------------------------- VibeVoice
    # VibeVoice's own ``generate`` takes ``cfg_scale``; the diffusion sampler's
    # step count is set through the model (``set_ddpm_inference_steps``), so
    # the worker consumes that one itself.  The engine's own defaults are 1.3
    # for the guidance scale and 10 steps.
    "vibevoice": (
        Option(
            key="cfg_scale",
            label="Voice guidance (CFG scale)",
            kind=KIND_FLOAT,
            default=1.3,
            minimum=1.0,
            maximum=3.0,
            increment=0.05,
            help=(
                "How strongly the voice and the text steer generation. 1.3 is "
                "the engine's default; higher follows the prompt more closely "
                "and can sound more forced."
            ),
        ),
        Option(
            key="ddpm_steps",
            label="Diffusion steps (DDPM)",
            kind=KIND_INT,
            default=10,
            minimum=4,
            maximum=40,
            increment=1,
            help=(
                "Steps of the diffusion head that renders the audio. 10 is "
                "the engine's default; more is slower and marginally cleaner, "
                "fewer is faster and a little rougher."
            ),
        ),
        _SEED,
    ),
    # --------------------------------------------------------------- MOSS-TTS
    # The Local-Transformer checkpoints are driven by the Hugging Face
    # ``generate`` call: a token budget and an optional duration.  The worker
    # forwards only the knobs the loaded model accepts - an engine version
    # without one reports it as skipped rather than failing the run.
    "moss_tts": (
        Option(
            key="max_new_tokens",
            label="Longest audio (tokens)",
            kind=KIND_INT,
            default=4096,
            minimum=256,
            maximum=8192,
            increment=256,
            help=(
                "Upper bound on the speech tokens one call may generate "
                "(Local Transformer models). Raise it only if a long segment "
                "is cut off."
            ),
        ),
        Option(
            key="tokens",
            label="Fixed duration (tokens, 0 = auto)",
            kind=KIND_INT,
            default=0,
            minimum=0,
            maximum=8192,
            increment=64,
            help=(
                "Force the audio to a token count (Local Transformer models): "
                "the family's own duration control. 0 lets the model decide "
                "from the text."
            ),
        ),
        _SEED,
    ),
    # ------------------------------------------------------------- GPT-SoVITS
    # Every key here is an argument of the engine's own ``TTS.run`` call, so
    # the worker passes the overrides straight through after filtering them
    # against the installed release's signature.
    "gptsovits": (
        Option(
            key="text_split_method",
            label="Text split method",
            kind=KIND_CHOICE,
            default="cut5",
            choices=(
                "cut0", "cut1", "cut2", "cut3", "cut4", "cut5",
                "auto_cut0", "auto_cut1", "auto_cut2", "auto_cut3",
                "auto_cut4", "auto_cut5",
            ),
            help=(
                "How a long text is broken into sentences before synthesis. "
                "cut5 is the project's default; the 'auto' forms also pick the "
                "language of each part."
            ),
        ),
        Option(
            key="temperature",
            label="Sampling temperature",
            kind=KIND_FLOAT,
            default=1.0,
            minimum=0.05,
            maximum=2.0,
            increment=0.05,
            help="Lower is steadier and more predictable, higher is livelier.",
        ),
        Option(
            key="top_k",
            label="Top-k sampling",
            kind=KIND_INT,
            default=15,
            minimum=1,
            maximum=200,
            increment=1,
            help="How many candidate tokens each step may choose from.",
        ),
        Option(
            key="top_p",
            label="Nucleus sampling (top-p)",
            kind=KIND_FLOAT,
            default=1.0,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help="1.0 disables it; lower values narrow the choices.",
        ),
        Option(
            key="repetition_penalty",
            label="Repetition penalty",
            kind=KIND_FLOAT,
            default=1.35,
            minimum=1.0,
            maximum=2.0,
            increment=0.05,
            help="Raises the cost of repeating what was just said.",
        ),
        Option(
            key="speed_factor",
            label="Speaking speed",
            kind=KIND_FLOAT,
            default=1.0,
            minimum=0.5,
            maximum=2.0,
            increment=0.05,
            help=(
                "The engine's own speed control, applied before the vocoder "
                "(a true time-stretch, unlike the resampling the application "
                "would otherwise apply)."
            ),
        ),
        Option(
            key="fragment_interval",
            label="Pause between sentences (seconds)",
            kind=KIND_FLOAT,
            default=0.3,
            minimum=0.0,
            maximum=2.0,
            increment=0.05,
            help="Silence inserted between the parts of a split text.",
        ),
        Option(
            key="sample_steps",
            label="Diffusion steps (v3/v4 only)",
            kind=KIND_INT,
            default=32,
            minimum=4,
            maximum=64,
            increment=1,
            help=(
                "Sampling steps of the v3/v4 SoVITS stage. 32 is the engine's "
                "default; ignored by v1, v2, v2Pro and v2ProPlus."
            ),
        ),
        Option(
            key="super_sampling",
            label="Super-sampling (v3/v4 only)",
            kind=KIND_BOOL,
            default=False,
            help=(
                "Run the v3/v4 vocoder at a higher rate for a little more "
                "clarity, at some speed cost. Ignored by the other releases."
            ),
        ),
        Option(
            key="parallel_infer",
            label="Parallel inference",
            kind=KIND_BOOL,
            default=True,
            help=(
                "Generate the parts of a split text together. Turning it off "
                "is slower but uses less memory."
            ),
        ),
        _SEED,
    ),
    # ------------------------------------------------------- Indic Parler-TTS
    "indic_parler": (
        Option(
            key="description",
            label="Voice description",
            kind=KIND_TEXT,
            default=None,
            help=(
                "Parler-TTS takes no voice id: this sentence *is* the voice. "
                "Leave it empty to use the description of the voice you "
                "selected; type your own to design one - name the speaker, "
                "the pace, the expressiveness and the recording quality, for "
                "example 'A calm female speaker reads at a slow pace, in a "
                "very high quality recording with no background noise.'"
            ),
        ),
        Option(
            key="temperature",
            label="Sampling temperature",
            kind=KIND_FLOAT,
            default=1.0,
            minimum=0.1,
            maximum=2.0,
            increment=0.05,
            help=(
                "1.0 is the model's own default. Lower is steadier and more "
                "monotone, higher is more varied - and can be unstable above "
                "~1.3."
            ),
        ),
        Option(
            key="max_new_tokens",
            label="Longest call (audio tokens)",
            kind=KIND_INT,
            default=1024,
            minimum=64,
            maximum=4096,
            increment=64,
            help=(
                "A ceiling on what one call may generate. 1024 (the "
                "default) is well past the longest sentence; raise it only "
                "if the end of a very long segment goes missing, and lower "
                "it to stop a run that will not stop."
            ),
        ),
        _SEED,
    ),
    # --------------------------------------------------------- Kitten TTS 2.0
    # KittenTTS 2 is a 1.7B speech language model driven by one call
    # (``model.generate(text, voice=... / reference=...)``).  Its decoding
    # preset sets temperature/top-k/top-p/min-p in one go, so the preset is the
    # knob offered here rather than the four values it stands for - overriding
    # them one at a time is a job for somebody who knows which preset they are
    # departing from, and the engine's own defaults for them are not published.
    "kitten_tts_2": (
        Option(
            key="preset",
            label="Decoding preset",
            kind=KIND_CHOICE,
            default="stable",
            choices=("stable", "expressive"),
            help=(
                "'stable' is the engine's own default and the one a book "
                "wants; 'expressive' loosens sampling for a livelier, less "
                "predictable delivery."
            ),
        ),
        Option(
            key="max_new_tokens",
            label="Longest audio (tokens)",
            kind=KIND_INT,
            default=1000,
            minimum=100,
            maximum=8192,
            increment=64,
            help=(
                "Upper bound on the speech tokens one chunk may generate "
                "(about 25 tokens is one second of audio). 1000 is the "
                "engine's default; raise it only if a long segment is cut "
                "off, and lower it to stop a run that will not stop."
            ),
        ),
        Option(
            key="normalize",
            label="Expand written forms before speaking",
            kind=KIND_BOOL,
            default=True,
            help=(
                "Turn numbers, dates, units and abbreviations into words. "
                "The normalizer is English-tuned: it is already switched off "
                "for a voice in another language, and turning it off here is "
                "what English text needs only if the expansion gets a "
                "written form wrong."
            ),
        ),
        Option(
            key="weights",
            label="Weight packing",
            kind=KIND_CHOICE,
            default="packed",
            choices=("packed", "emb4", "full"),
            load_time=True,
            help=(
                "Which packing of the same weights is loaded: 'packed' "
                "(947 MiB, lossless, the engine's default), 'emb4' (506 MiB, "
                "a 4-bit token embedding for a smaller download and a little "
                "less quality) or 'full' (plain bfloat16, 3.4 GB). Only the "
                "one selected is downloaded."
            ),
        ),
        Option(
            key="decoder",
            label="Audio decoder",
            kind=KIND_CHOICE,
            default="default",
            choices=("default", "student_w4"),
            load_time=True,
            help=(
                "'default' is the full-quality 459 MB decoder; "
                "'student_w4' is a distilled one packed to 4 bits (39 MB), "
                "which trades a little fidelity for a much smaller "
                "footprint - not for speed."
            ),
        ),
        _SEED,
    ),
}


def engine_ids() -> tuple[str, ...]:
    """Engines that have tuning options (every Voice Lab engine)."""
    return tuple(_OPTIONS)


def specs(engine_id: str) -> tuple[Option, ...]:
    """The tuning options of an engine, in display order."""
    return _OPTIONS.get(engine_id or "", ())


def spec(engine_id: str, key: str) -> Option | None:
    """One tuning option by key, or ``None`` when the engine has no such knob."""
    for option in specs(engine_id):
        if option.key == key:
            return option
    return None


def has_options(engine_id: str) -> bool:
    return bool(specs(engine_id))


def option_keys(engine_id: str) -> tuple[str, ...]:
    return tuple(option.key for option in specs(engine_id))


def defaults(engine_id: str) -> dict[str, Any]:
    """Every option of an engine at the engine's own default value."""
    return {option.key: option.default for option in specs(engine_id)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _coerce(option: Option, value: Any) -> Any:
    """Coerce one raw value into the option's type, or raise ``ValueError``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if option.kind == KIND_BOOL:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if option.kind in (KIND_INT, KIND_SEED):
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{option.label} must be a whole number.") from exc
        return number
    if option.kind == KIND_FLOAT:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{option.label} must be a number.") from exc
    if option.kind == KIND_CHOICE:
        text = str(value).strip()
        if option.choices and text not in option.choices:
            raise ValueError(f"{option.label} must be one of: "
                             + ", ".join(option.choices))
        return text
    return str(value).strip()


def _clamp(option: Option, value: Any) -> Any:
    if value is None or option.kind not in (KIND_INT, KIND_FLOAT, KIND_SEED):
        return value
    if option.kind == KIND_SEED:
        return value
    number = float(value)
    if option.minimum is not None:
        number = max(float(option.minimum), number)
    if option.maximum is not None:
        number = min(float(option.maximum), number)
    return round(number) if option.kind == KIND_INT else number


def clean(engine_id: str, values: dict[str, Any] | None) -> dict[str, Any]:
    """Validate ``values`` against the engine and keep only the *overrides*.

    Unknown keys are dropped, out-of-range numbers are clamped, and any value
    equal to the engine's own default is removed - what is left is exactly what
    the worker has to pass to the engine.  Raises ``ValueError`` for values that
    cannot be understood at all (e.g. text in a numeric field).
    """
    out: dict[str, Any] = {}
    for option in specs(engine_id):
        if not isinstance(values, dict) or option.key not in values:
            continue
        value = _clamp(option, _coerce(option, values[option.key]))
        if value is None or value in ("", option.default):
            continue
        out[option.key] = value
    return out


def problems(engine_id: str, values: dict[str, Any] | None,
             device: str = "cpu") -> list[str]:
    """Human-readable warnings about a set of overrides (never fatal)."""
    notes: list[str] = []
    cleaned = clean(engine_id, values)
    if "quantize" in cleaned and str(device).lower() == "cuda":
        notes.append(
            "int8 quantization is a CPU optimisation; on the GPU it is "
            "ignored."
        )
    return notes


def describe_overrides(engine_id: str, values: dict[str, Any] | None) -> str:
    """Short human summary of the overrides, e.g. ``"32 steps, cfg 2"``."""
    cleaned = clean(engine_id, values)
    if not cleaned:
        return ""
    bits: list[str] = []
    for option in specs(engine_id):
        if option.key not in cleaned:
            continue
        value = cleaned[option.key]
        if option.kind == KIND_BOOL:
            bits.append(option.label.split(" (")[0].lower())
        elif option.kind == KIND_FLOAT:
            bits.append(f"{option.label.split(' (')[0].lower()} {float(value):g}")
        else:
            bits.append(f"{option.label.split(' (')[0].lower()} {value}")
    return ", ".join(bits)


def max_seconds_note(engine_id: str) -> str:
    """One-line hint about which knobs cost the most time on a CPU."""
    if engine_id == "f5tts":
        return ("Diffusion steps dominate the runtime on a CPU: 16 is roughly "
                "twice as fast as 32.")
    if engine_id == "pocket_tts":
        return "Quantization is the single biggest speed-up on a CPU."
    if engine_id == "kokoclone":
        return ("The first run downloads KokoClone's Kokoro and Kanade models "
                "(~1.5 GB) and loads them once; after that each segment is "
                "synthesised and re-voiced in one pass.")
    if engine_id == "chatterbox":
        return ("Chatterbox is a 350-500M transformer that runs on the GPU: "
                "the first run downloads the model it needs, and Turbo is the "
                "fastest of the three.")
    if engine_id == "vibevoice":
        return ("VibeVoice is a 1.5-7B language model with a diffusion head "
                "that runs on the GPU: the first run downloads the model it "
                "needs, and the 1.5B model is the one that fits a 12 GB card.")
    if engine_id == "qwen3tts":
        return ("Qwen3-TTS is a 0.6-1.7B language model that runs on the GPU:"
                "the first run downloads it, and within one run a cloned "
                "voice is prepared once and reused for every later segment.")
    if engine_id == "moss_tts":
        return ("MOSS-TTS runs on the GPU: the first run downloads the model "
                "it needs, and the 4B one is the largest that fits a 12 GB "
                "card.")
    if engine_id == "gptsovits":
        return ("GPT-SoVITS runs on the GPU: the first run downloads its "
                "pretrained weights (~1.5-2 GB) and the reference recording "
                "is prepared once, then reused for every later segment.")
    if engine_id == "kitten_tts_2":
        return ("Kitten TTS 2.0 runs on the CPU or the GPU: the first run "
                "downloads its 1.7B weights (~950 MB, or ~470 MB with the "
                "emb4 packing). A CPU run is given every core this computer "
                "allows, but the engine's PyTorch path is the one upstream "
                "calls demo-only, so it is the slower device; the GPU is the "
                "fast one.")
    return ""


# ---------------------------------------------------------------------------
# Per-project / per-engine application
# ---------------------------------------------------------------------------
def resolve(engine_id: str, project: dict[str, Any] | None,
            fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """The effective overrides: project values win, then the engine defaults.

    ``fallback`` is what the Settings > Voice Clone page saved for the engine
    ("use these tuning options for new projects"); a project that has its own
    overrides keeps them.
    """
    cleaned = clean(engine_id, project)
    if cleaned:
        return cleaned
    return clean(engine_id, fallback)


def apply_to_voice(voice_entry: dict[str, Any],
                   values: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``voice_entry`` carrying the tuning overrides.

    The Recording window feeds the result to the synthesis worker, so the
    options ride on the voice entry and the ``engine.synthesize`` contract is
    untouched.  The original entry is never mutated.  Defaults are *not*
    stored: an empty set of overrides removes the key entirely.
    """
    out = dict(voice_entry)
    engine_id = out.get("engine") or out.get("tts") or ""
    cleaned = clean(engine_id, values)
    if cleaned:
        out["options"] = cleaned
    else:
        out.pop("options", None)
    return out


def settings_key(engine_id: str) -> str:
    """Settings key holding the default tuning options of an engine."""
    return f"clone_engines.options.{engine_id}"

