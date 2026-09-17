"""Per-engine tuning options for the Voice Lab engines.

Each engine has its own generation knobs: F5-TTS offers diffusion steps and a
CFG strength, Bark offers its text/waveform sampling temperatures, and Pocket
TTS can quantize its model to int8 for a much faster CPU run.

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

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    increment: float = 0.1
    choices: Tuple[str, ...] = ()
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


_OPTIONS: Dict[str, Tuple[Option, ...]] = {
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
    # -------------------------------------------------------------------- Bark
    "bark": (
        Option(
            key="text_temp",
            label="Text temperature",
            kind=KIND_FLOAT,
            default=0.7,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help=(
                "Sampling temperature of the semantic (text) stage - the "
                "engine's documented default is 0.7."
            ),
        ),
        Option(
            key="waveform_temp",
            label="Waveform temperature",
            kind=KIND_FLOAT,
            default=0.7,
            minimum=0.0,
            maximum=1.0,
            increment=0.05,
            help=(
                "Sampling temperature of the audio (coarse + fine) stages - "
                "the engine's documented default is 0.7."
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
}


def engine_ids() -> Tuple[str, ...]:
    """Engines that have tuning options (every Voice Lab engine)."""
    return tuple(_OPTIONS)


def specs(engine_id: str) -> Tuple[Option, ...]:
    """The tuning options of an engine, in display order."""
    return _OPTIONS.get(engine_id or "", ())


def spec(engine_id: str, key: str) -> Optional[Option]:
    """One tuning option by key, or ``None`` when the engine has no such knob."""
    for option in specs(engine_id):
        if option.key == key:
            return option
    return None


def has_options(engine_id: str) -> bool:
    return bool(specs(engine_id))


def option_keys(engine_id: str) -> Tuple[str, ...]:
    return tuple(option.key for option in specs(engine_id))


def defaults(engine_id: str) -> Dict[str, Any]:
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
    return int(round(number)) if option.kind == KIND_INT else number


def clean(engine_id: str, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate ``values`` against the engine and keep only the *overrides*.

    Unknown keys are dropped, out-of-range numbers are clamped, and any value
    equal to the engine's own default is removed - what is left is exactly what
    the worker has to pass to the engine.  Raises ``ValueError`` for values that
    cannot be understood at all (e.g. text in a numeric field).
    """
    out: Dict[str, Any] = {}
    for option in specs(engine_id):
        if not isinstance(values, dict) or option.key not in values:
            continue
        value = _clamp(option, _coerce(option, values[option.key]))
        if value is None or value == "" or value == option.default:
            continue
        out[option.key] = value
    return out


def problems(engine_id: str, values: Optional[Dict[str, Any]],
             device: str = "cpu") -> List[str]:
    """Human-readable warnings about a set of overrides (never fatal)."""
    notes: List[str] = []
    cleaned = clean(engine_id, values)
    if "quantize" in cleaned and str(device).lower() == "cuda":
        notes.append(
            "int8 quantization is a CPU optimisation; on the GPU it is "
            "ignored."
        )
    return notes


def describe_overrides(engine_id: str, values: Optional[Dict[str, Any]]) -> str:
    """Short human summary of the overrides, e.g. ``"32 steps, cfg 2"``."""
    cleaned = clean(engine_id, values)
    if not cleaned:
        return ""
    bits: List[str] = []
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
    if engine_id == "bark":
        return ("Bark generates audio autoregressively; a lower waveform "
                "temperature does not make it faster.")
    if engine_id == "pocket_tts":
        return "Quantization is the single biggest speed-up on a CPU."
    return ""


# ---------------------------------------------------------------------------
# Per-project / per-engine application
# ---------------------------------------------------------------------------
def resolve(engine_id: str, project: Optional[Dict[str, Any]],
            fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The effective overrides: project values win, then the engine defaults.

    ``fallback`` is what the Settings > Voice Clone page saved for the engine
    ("use these tuning options for new projects"); a project that has its own
    overrides keeps them.
    """
    cleaned = clean(engine_id, project)
    if cleaned:
        return cleaned
    return clean(engine_id, fallback)


def apply_to_voice(voice_entry: Dict[str, Any],
                   values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


def iter_options(engine_id: str) -> Iterable[Tuple[Option, Any]]:
    """``(option, default)`` pairs, for building the dialogs."""
    for option in specs(engine_id):
        yield option, option.default
