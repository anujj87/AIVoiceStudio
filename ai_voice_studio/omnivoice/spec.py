"""OmniVoice feature specifications (shared vocabulary).

This module centralises everything AI Voice Studio knows about OmniVoice
features, so the direct engine (``omnivoice`` / ``omnivoice-triton``), the
server engine (``omnivoice_server`` / ``omnivoice-server``), the GUI and the
tests all agree on one canonical feature vocabulary.

The values below were researched from the authoritative sources:

* ``k2-fsa/OmniVoice`` (base model) - voice cloning, voice design attributes,
  auto voice, non-verbal tags, pinyin / CMU pronunciation control, and
  ``num_step`` / ``speed`` / ``duration`` generation parameters.
* ``newgrit1004/omnivoice-triton`` - ``create_runner("triton"|"hybrid"|...)``
  runners whose ``generate`` / ``generate_voice_clone`` /
  ``generate_voice_design`` accept ``num_step``, ``guidance_scale``,
  ``class_temperature`` and an optional ``language`` hint.
* ``maemreyo/omnivoice-server`` (PyPI ``omnivoice-server``) - the
  OpenAI-compatible HTTP server: presets, canonical design attributes,
  non-verbal tags, per-request generation parameters and voice profiles.

Nothing in this module imports the heavy OmniVoice packages, so it is safe
to use everywhere (including unit tests and the GUI thread).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import languages

# The two engine ids OmniVoice is reachable through: the direct engine
# (``omnivoice-triton``) and the OpenAI-compatible HTTP server
# (``omnivoice-server``).  Both share one model, one voice library and one
# language table, so the GUI asks this tuple instead of keeping its own copy.
ENGINE_IDS: Tuple[str, ...] = ("omnivoice", "omnivoice_server")


# OmniVoice output sample rate (kHz) - same for the base model, the triton
# runners and the HTTP server.
SAMPLE_RATE = 24000


def is_engine(value: Optional[str]) -> bool:
    """True when ``value`` names one of the OmniVoice engines."""
    return bool(value) and str(value) in ENGINE_IDS


# ---------------------------------------------------------------------------
# Generation defaults (AI Voice Studio level)
# ---------------------------------------------------------------------------
DEFAULT_NUM_STEP = 32          # diffusion steps 1-64 (16 = faster, 32 = balanced)
DEFAULT_GUIDANCE_SCALE = 3.0   # CFG strength 0-10 (studio default; server demo used 3.0)
DEFAULT_CLASS_TEMPERATURE = 0.0  # token sampling temperature 0-2 (0 = greedy)
DEFAULT_POSITION_TEMPERATURE = 5.0  # voice-diversity temperature 0-10 (server default)
DEFAULT_T_SHIFT = 0.1          # noise-schedule shift 0-2 (server default)
DEFAULT_DENOISE = True         # denoising token (recommended on, server default)

# Speed is 0.25-4.0 on the HTTP server; the recording window's rate slider is
# 0.5-2.0, which fits comfortably inside that range.
SPEED_MIN = 0.5
SPEED_MAX = 2.0

# Parameter ranges (used by the GUI spin controls).
NUM_STEP_MIN, NUM_STEP_MAX = 1, 64
GUIDANCE_MIN, GUIDANCE_MAX = 0.0, 10.0
CLASS_TEMP_MIN, CLASS_TEMP_MAX = 0.0, 2.0
POSITION_TEMP_MIN, POSITION_TEMP_MAX = 0.0, 10.0
T_SHIFT_MIN, T_SHIFT_MAX = 0.0, 2.0
DURATION_MIN, DURATION_MAX = 0.1, 60.0

# ---------------------------------------------------------------------------
# Direct engine (omnivoice-triton) catalog voices -> voice-design instruct.
#
# The direct catalog ships "auto / female / male / child / elderly" voice
# entries.  OmniVoice has no speaker ids - those entries only make sense when
# translated into the model's *voice design* instruction vocabulary.
# ---------------------------------------------------------------------------
DIRECT_VOICE_INSTRUCT = {
    "auto": "",
    "female": "female",
    "male": "male",
    "child": "child",
    "elderly": "elderly",
    "teenager": "teenager",
    "young": "young adult",
    "middle": "middle-aged",
    "whisper": "whisper",
}

# ---------------------------------------------------------------------------
# Server (omnivoice-server) OpenAI-compatible presets.
#
# These are the official mappings shipped inside the omnivoice-server package
# (``omnivoice_server/voice_presets.py``).  The client resolves them itself so
# a designed voice does not depend on the server's copy of the table.
# ---------------------------------------------------------------------------
OPENAI_VOICE_PRESETS = {
    "alloy": "female, young adult, moderate pitch, american accent",
    "ash": "male, young adult, low pitch, american accent",
    "ballad": "male, middle-aged, low pitch, british accent",
    "cedar": "male, middle-aged, low pitch, american accent",
    "coral": "female, young adult, high pitch, australian accent",
    "echo": "male, middle-aged, moderate pitch, canadian accent",
    "fable": "female, middle-aged, moderate pitch, british accent",
    "marin": "female, middle-aged, moderate pitch, canadian accent",
    "nova": "female, young adult, high pitch, american accent",
    "onyx": "male, middle-aged, very low pitch, british accent",
    "sage": "female, elderly, low pitch, british accent",
    "shimmer": "female, young adult, very high pitch, american accent",
    "verse": "male, young adult, moderate pitch, british accent",
}

SERVER_DEFAULT_INSTRUCT = "male, middle-aged, moderate pitch, british accent"

# ---------------------------------------------------------------------------
# Canonical voice-design attributes (upstream OmniVoice vocabulary, mirrored
# by the HTTP server's ``/v1/voices`` response and used by the GUI builders).
# ---------------------------------------------------------------------------
DESIGN_ATTRIBUTES: Dict[str, List[str]] = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": [
        "very low pitch",
        "low pitch",
        "moderate pitch",
        "high pitch",
        "very high pitch",
    ],
    "style": ["whisper"],
    "accent_en": [
        "american accent",
        "british accent",
        "australian accent",
        "chinese accent",
        "canadian accent",
        "indian accent",
        "korean accent",
        "portuguese accent",
        "russian accent",
        "japanese accent",
    ],
    "dialect_zh": [
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    ],
}

# Non-verbal symbols understood inline by OmniVoice (pass-through feature).
# A tag in this list is spoken as a sound; anything else is read literally.
NONVERBAL_TAGS = [
    "laughter", "breath", "sigh", "sniff",
    "confirmation-en", "question-en",
    "question-ah", "question-oh", "question-ei", "question-yi",
    "surprise-ah", "surprise-oh", "surprise-wa", "surprise-yo",
    "dissatisfaction-hnn",
]

# The full language table of the model lives in ``languages.py`` (646 entries,
# copied from upstream ``docs/languages.md``); these examples stay as the short
# "well known languages" list used by the documentation and by quick hints.
LANGUAGE_EXAMPLES = [
    "en (English)", "zh (Chinese)", "hi (Hindi)", "arb (Standard Arabic)",
    "es (Spanish)", "fr (French)", "de (German)", "ja (Japanese)",
    "ko (Korean)", "vi (Vietnamese)", "ru (Russian)", "pt (Portuguese)",
]


def clean_language(value: Optional[str]) -> Optional[str]:
    """Normalise a language hint to the id the engine expects.

    ``"auto"`` / ``""`` / ``None`` all mean "let OmniVoice detect it" and are
    mapped to ``None`` (the model's native auto-detect behaviour).  Anything
    else is resolved against the model's own language table
    (:mod:`ai_voice_studio.omnivoice.languages`), so an English name
    (``"English"``), an ISO 639-3 code (``"eng"``) and a picker label
    (``"English (en)"``) all become the ``"en"`` the request must carry, while
    a code the table does not know is passed through lower-cased so a newer
    engine version is never blocked.
    """
    return languages.normalise(value)


def voice_id_instruct(engine: Optional[str], voice_id: Optional[str]) -> str:
    """Return the built-in instruct for a catalog voice id, or ``""``.

    * Direct engine: catalog voices ``female``, ``male``, ``child`` ... map to
      the matching upstream voice-design attribute.
    * Server engine: OpenAI preset names (``alloy`` ... ``verse``) map to the
      server's official design prompts.
    """
    if not voice_id:
        return ""
    voice_id = str(voice_id).strip().lower()
    if engine == "omnivoice_server":
        return OPENAI_VOICE_PRESETS.get(voice_id, "")
    return DIRECT_VOICE_INSTRUCT.get(voice_id, "")


def resolve_instruct(
    engine: Optional[str],
    voice_id: Optional[str],
    explicit_instruct: Optional[str] = None,
) -> str:
    """Compute the effective voice-design instruction for an engine.

    Precedence: an explicit instruction (the strongest OmniVoice control)
    wins; otherwise the catalog voice / preset mapping is used; otherwise
    ``""`` means "let the engine choose" (auto voice on the direct engine,
    server default design prompt on the HTTP server).
    """
    if explicit_instruct and str(explicit_instruct).strip():
        return str(explicit_instruct).strip()
    return voice_id_instruct(engine, voice_id)


def clamp(value: Optional[float], lo: float, hi: float, default: Optional[float]) -> Optional[float]:
    """Clamp an optional numeric setting into ``[lo, hi]``."""
    if value is None or value == "":
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, num))


def filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only the kwargs a callable can actually accept.

    Version-adaptive calling: different omnivoice-triton / omnivoice releases
    accept different generation parameters.  Returns ``(supported, skipped)``
    where ``skipped`` lists the requested parameters the callable cannot take,
    so callers can report them instead of silently dropping them.

    Callables with ``**kwargs`` accept everything.
    """
    if not kwargs:
        return {}, []
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        # Cannot introspect - assume it accepts everything (forward-compatible).
        return dict(kwargs), []
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return dict(kwargs), []
    supported: Dict[str, Any] = {}
    skipped: List[str] = []
    for name, value in kwargs.items():
        if name in params:
            supported[name] = value
        else:
            skipped.append(name)
    return supported, skipped


def build_omni(
    instruct: str = "",
    mode: str = "auto",
    ref_audio: str = "",
    ref_text: str = "",
    language: Optional[str] = None,
    num_step: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    class_temperature: Optional[float] = None,
    position_temperature: Optional[float] = None,
    denoise: Optional[bool] = None,
    t_shift: Optional[float] = None,
    duration: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the canonical per-project ``omni`` settings dict."""
    return {
        "mode": mode if mode in ("auto", "design", "clone") else "auto",
        "instruct": (instruct or "").strip(),
        "ref_audio": (ref_audio or "").strip(),
        "ref_text": (ref_text or "").strip(),
        "language": clean_language(language),
        "num_step": int(num_step) if num_step else None,
        "guidance_scale": float(guidance_scale) if guidance_scale is not None else None,
        "class_temperature": float(class_temperature) if class_temperature is not None else None,
        "position_temperature": float(position_temperature) if position_temperature is not None else None,
        "denoise": bool(denoise) if denoise is not None else None,
        "t_shift": float(t_shift) if t_shift is not None else None,
        "duration": float(duration) if duration else None,
        "seed": int(seed) if seed is not None and str(seed).strip() != "" else None,
    }


def apply_omni_to_voice(
    voice_entry: Dict[str, Any],
    omni: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a copy of ``voice_entry`` enriched with the OmniVoice options.

    This is what the Recording window feeds into the synthesis worker for the
    direct and server engines: cloning references, design instructions, the
    language hint and the advanced generation knobs all ride on the voice
    entry, so the engine cache keys and ``engine.synthesize`` interface stay
    untouched.

    The original ``voice_entry`` is never mutated.
    """
    out = dict(voice_entry)
    omni = build_omni(**(omni or {}))
    out["omni"] = omni

    engine = out.get("engine") or "omnivoice"
    mode = omni.get("mode", "auto")

    if mode == "clone":
        # Voice cloning: reference audio (+ optional transcript).  The
        # reference file must exist; the caller validates that before saving.
        out["ref_audio"] = omni.get("ref_audio", "") or out.get("ref_audio", "")
        out["ref_text"] = omni.get("ref_text", "") or out.get("ref_text", "")
        out["instruct"] = ""
    elif mode == "design":
        # Voice design: natural-language description wins over the voice id.
        out["instruct"] = omni.get("instruct", "") or voice_id_instruct(engine, out.get("voice"))
        out["ref_audio"] = ""
        out["ref_text"] = ""
    else:
        # Auto voice: only the catalog voice id may still imply an instruct
        # (e.g. the "female" entry on the direct engine).
        out["instruct"] = voice_id_instruct(engine, out.get("voice"))
        out["ref_audio"] = ""
        out["ref_text"] = ""

    # Always normalise: "auto" / empty become None so the request never
    # sends an invalid language code to the engine.  An explicit project
    # hint wins; otherwise keep whatever language the voice entry carried.
    hint = omni.get("language") or voice_entry.get("language")
    out["language"] = clean_language(hint)
    return out
