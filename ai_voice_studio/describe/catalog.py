"""Video-description (audio description) model catalog.

The catalog is the single description of every model this application can use to
*describe the picture* of a video.  It mirrors :mod:`ai_voice_studio.stt.catalog`
deliberately: a describing model is an ordinary Python package (``transformers``
plus PyTorch and a multimodal processor) that runs in a Python environment of its
own, so *Manage the video describer* in Settings and the New Project wizard both
read their choices from here.

Every model entry carries:

* ``name`` / ``license`` / ``languages_label`` / ``fit_label`` / ``notes`` - the
  table columns of Settings > Manage the video describer (after the variant).
* ``variants`` - the size/checkpoint choices the *variant* combo lists.  Unlike
  an STT entry, a variant here is a **dict**, because the Settings table has to
  show what each choice costs: its download size and the video memory it needs.
* ``hallucination`` - a plain-language note on how much the model tends to invent
  detail.  A blind listener cannot check a description against the picture, so
  invented detail is the single most serious failure mode and it is stated per
  model instead of being hidden behind benchmarks.
* ``backend`` / ``pip_packages`` - how the model is run and what has to be
  installed before it can run.
* ``hf_models`` - variant -> Hugging Face repository.  A model with no entry for
  a variant cannot be downloaded yet, and the UI says so.
* ``auto_class`` / ``driver`` / ``load_kwargs`` - the three things a checkpoint
  that ships its *own* modelling code can need beyond a plain describing stack:
  the ``transformers`` Auto class its ``auto_map`` actually registers, the way a
  run asks it for a description (``generate``, or the checkpoint's own ``chat``),
  and any extra ``from_pretrained`` keyword it wants.  See
  :func:`auto_class`, :func:`driver` and :func:`load_kwargs`.

Every model here runs with no account and no token, with **one exception that is
marked as such**: Gemma 3's checkpoints live in Google's *gated* repository, so
its entry carries ``"gated": True`` and the application asks for a Hugging Face
token instead of failing with the library's bare ``401`` (see
:mod:`ai_voice_studio.hf_access`).  The flag is what lets a run say so *before* a
24 GB download starts; the describing models other than Gemma 3 download
anonymously, and the STT catalogue keeps its own rule of refusing gated entries
entirely.

Every entry states its licence in the terms its own publisher uses, and a licence
that is *not* OSI-approved says so - Gemma 3 is the one entry of that kind, and
its description says why it is here anyway (the user asked for it, and it is a
strong small vision model) together with what that costs the user: a gated
download that needs Google's licence accepted with an account of their own.

Why these models and not others is recorded in ``AUDIO_DESCRIPTION_PLAN.md``;
briefly: the Qwen omni/vision lines and Gemma 4 are Apache-2.0 with real video
understanding, Qwen3-Omni-Captioner is the only open checkpoint fine-tuned for
low-hallucination captions, and SmolVLM2/Moondream keep the app's
"works with no NVIDIA card" promise.  Meta's Llama 3.2 Vision and Gemma 1/2 are
left out: their licences are not OSI-approved and they add nothing the entries
above do not already cover.
"""

from __future__ import annotations

#: The languages the describing models answer in.  Vision-language models are
#: overwhelmingly English-first, so the list is short and honest.
DESCRIBE_LANGUAGES: list[tuple[str, str]] = [
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
]

_ENGLISH_FIRST: list[tuple[str, str]] = [("en", "English")] + [
    (code, label) for code, label in DESCRIBE_LANGUAGES if code != "en"
]

#: Back-end id -> the importable module that proves it is installed in a model's
#: own environment.  Every describing model runs through ``transformers``.
BACKEND_MODULES: dict[str, str] = {
    "transformers": "transformers",
}

#: The device a model (or one of its variants) is *preferred* on.  This is only a
#: preference: both devices are offered wherever the model can run on both.
FIT_CPU = "cpu"
FIT_GPU = "gpu"

_FIT_BOTH_GPU_FIRST = "\u2705 GPU or CPU"
_FIT_BOTH_CPU_FIRST = "\u2705 CPU or GPU"
_FIT_GPU_ONLY = "\u26a0 GPU only"

#: The ``transformers`` series a model's *own* modelling code still runs on.
#: A checkpoint that ships ``trust_remote_code`` imports the library's internals,
#: not just its public API, so the release it was written against is part of what
#: its environment has to contain.  MiniCPM-o 2.6 imports
#: ``WHISPER_ATTENTION_CLASSES`` from ``transformers.models.whisper``, which
#: **4.53.0** removed (the file still defines it in 4.52.0), so the pin stops one
#: minor release short of that and resolves to the newest 4.52.x.  The exact
#: release the model card names - 4.44.2, which is also what the checkpoint's own
#: ``config.json`` records as ``transformers_version`` - is deliberately not what
#: the pin asks for: 4.44.2 requires ``tokenizers<0.20``, whose newest Windows
#: wheel is cp312, so an environment on the Python 3.13 this application builds
#: (and on the PyTorch 2.11 it installs) cannot have it without a Rust toolchain.
#: The pin is safe because every describing model has a virtualenv of its own
#: (``describe_envs\\<model id>``): it cannot reach another model, a TTS engine or
#: a speech-to-text model.  NeMo's speech-to-text models are pinned below five
#: for the same kind of reason - see :mod:`ai_voice_studio.stt.catalog`.
#:
#: MiniCPM-V 4 shares the pin for the same kind of reason: its ``config.json``
#: records ``transformers_version`` 4.51.0 and its own files import
#: ``transformers.modeling_attn_mask_utils``, a helper the library itself calls
#: deprecated and the 5 series does not carry.  MiniCPM-o 4.5 records 4.51.0 as
#: well and its card names that release as the one it was tested with, so it
#: takes the same pin for the same reason: all three are checkpoints whose own
#: code is written against the library's internals, one minor series back.
_REMOTE_CODE_TRANSFORMERS = "transformers<4.53"

#: How a run asks a model for a description.  ``generate`` is the general path -
#: the processor builds the inputs from the standard multimodal message and the
#: model generates from them - and it is what every checkpoint ``transformers``
#: supports natively takes.  The other two are the ways a checkpoint that
#: ``transformers`` does *not* support natively is asked, and each is a property
#: of the checkpoint rather than a preference:
#:
#: * ``DRIVER_CHAT`` - the model's own ``chat`` method (the MiniCPM-o/V family,
#:   whose processors carry no template the standard message fits).  See
#:   :func:`ai_voice_studio.describe.worker._describe_with_chat`.
#: * ``DRIVER_QUERY`` - the model's own single-image question method (Moondream
#:   2, which ships no processor and no chat template at all).  See
#:   :func:`ai_voice_studio.describe.worker._describe_with_query`.
DRIVER_GENERATE = "generate"
DRIVER_CHAT = "chat"
DRIVER_QUERY = "query"

#: Seconds of compute per *minute of video* at :data:`DEFAULT_FPS_COST`, used by
#: :mod:`ai_voice_studio.describe.pipeline` to tell the user how long a run will
#: take before it starts.  They are deliberately round, conservative figures for
#: an 8-12 GB card, not benchmark scores: the point is to warn, not to promise.
_DEFAULT_FPS_COST = 0.5


def _variant(
    value: str,
    label: str,
    *,
    size_gb: float,
    vram_gb: float,
    fit: str,
    cpu_ok: bool,
    gpu_cost: float,
    cpu_cost: float,
    hf: str = "",
    note: str = "",
) -> dict:
    """One variant choice with everything the Settings table needs.

    ``gpu_cost`` / ``cpu_cost`` are seconds of compute per minute of video at
    :data:`_DEFAULT_FPS_COST`; the estimator scales them by the chosen frame
    rate.  ``cpu_ok`` is ``False`` for a checkpoint that is simply too large to
    hold in an ordinary machine's memory, which is what makes a variant GPU-only.
    """
    return {
        "value": value,
        "label": label,
        "size_gb": float(size_gb),
        "vram_gb": float(vram_gb),
        "fit": fit,
        "cpu_ok": bool(cpu_ok),
        "gpu_cost": float(gpu_cost),
        "cpu_cost": float(cpu_cost),
        "hf": hf,
        "note": note,
    }


MODELS: list[dict] = [
    {
        "id": "qwen3_omni_captioner",
        "name": "Qwen3-Omni Captioner",
        "license": "Apache 2.0",
        "languages_label": "Multilingual",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u26a0 GPU only - about 37 GB of video memory",
        "notes": "The one open model fine-tuned specifically for detailed, "
                 "low-hallucination captions.  Excellent, but far above this "
                 "app's 12 GB rule.",
        "hallucination": "Lowest of the family: this checkpoint was tuned for "
                         "detailed captions with as little invented detail as "
                         "possible.",
        "variants": [
            _variant(
                "30b-a3b",
                "Qwen3-Omni-30B-A3B-Captioner",
                size_gb=37.0, vram_gb=37.0, fit="gpu", cpu_ok=False,
                gpu_cost=85.0, cpu_cost=600.0,
                hf="Qwen/Qwen3-Omni-30B-A3B-Captioner",
                note="Best descriptions of any open model here.",
            ),
        ],
        "default_variant": "30b-a3b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        "timestamps": True,
        "trust_remote_code": False,
    },
    {
        "id": "qwen2_5_omni",
        "name": "Qwen2.5-Omni",
        "license": "Apache 2.0",
        "languages_label": "Multilingual",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u2705 7B: GPU \u00b7 3B: CPU or GPU",
        "notes": "Takes video AND audio at once, which is what makes the "
                 "description follow the dialogue; the 7B is the sweet spot, "
                 "the 3B runs anywhere.",
        "hallucination": "Moderate: it will invent a name or a place if asked "
                         "to be specific.  The guideline prompt keeps it "
                         "descriptive rather than speculative.",
        "variants": [
            _variant(
                "7b", "Qwen2.5-Omni-7B (recommended)",
                size_gb=16.5, vram_gb=10.0, fit="gpu", cpu_ok=False,
                gpu_cost=90.0, cpu_cost=540.0,
                hf="Qwen/Qwen2.5-Omni-7B",
                note="Fits an 8-12 GB card.",
            ),
            _variant(
                "3b", "Qwen2.5-Omni-3B (small GPU or CPU)",
                size_gb=7.5, vram_gb=5.0, fit="cpu", cpu_ok=True,
                gpu_cost=45.0, cpu_cost=210.0,
                hf="Qwen/Qwen2.5-Omni-3B",
                note="Keeps the app's CPU-only promise.",
            ),
        ],
        "default_variant": "7b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "librosa", "soundfile"],
        "timestamps": True,
        "trust_remote_code": False,
    },
    {
        "id": "qwen3_vl",
        "name": "Qwen3-VL",
        "license": "Apache 2.0",
        "languages_label": "Multilingual",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u2705 2B: CPU or GPU \u00b7 4B and up: GPU",
        "notes": "The best open family for very long video: 256K context and "
                 "second-level grounding, which is exactly what a chunked run "
                 "over a lecture or a film needs.",
        "hallucination": "Moderate.  Strong grounding on the sampled frames, but "
                         "it answers more confidently than it sees.",
        "variants": [
            _variant(
                "2b", "Qwen3-VL-2B (CPU or GPU)",
                size_gb=4.4, vram_gb=4.0, fit="cpu", cpu_ok=True,
                gpu_cost=22.0, cpu_cost=95.0,
                hf="Qwen/Qwen3-VL-2B-Instruct",
                note="Lightest Qwen3-VL; runs with no card.",
            ),
            _variant(
                "4b", "Qwen3-VL-4B",
                size_gb=8.0, vram_gb=7.0, fit="gpu", cpu_ok=True,
                gpu_cost=40.0, cpu_cost=180.0,
                hf="Qwen/Qwen3-VL-4B-Instruct",
            ),
            _variant(
                "8b", "Qwen3-VL-8B (recommended)",
                size_gb=16.0, vram_gb=10.0, fit="gpu", cpu_ok=False,
                gpu_cost=70.0, cpu_cost=420.0,
                hf="Qwen/Qwen3-VL-8B-Instruct",
                note="Best quality that fits a 12 GB card.",
            ),
            _variant(
                "30b-a3b", "Qwen3-VL-30B-A3B (MoE, big GPU)",
                size_gb=60.0, vram_gb=34.0, fit="gpu", cpu_ok=False,
                gpu_cost=75.0, cpu_cost=560.0,
                hf="Qwen/Qwen3-VL-30B-A3B-Instruct",
                note="Above this app's 12 GB rule.",
            ),
        ],
        "default_variant": "8b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        "timestamps": True,
        "trust_remote_code": False,
    },
    {
        "id": "qwen2_5_vl",
        "name": "Qwen2.5-VL",
        "license": "Apache 2.0",
        "languages_label": "Multilingual",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u2705 3B: CPU or GPU \u00b7 7B: GPU",
        "notes": "Well-tested value baseline; comprehends over an hour of "
                 "video with sparse frames and can point at the moment an "
                 "event happens.",
        "hallucination": "Moderate.  Detailed but talkative: without the "
                         "guideline prompt it describes far more than matters.",
        "variants": [
            _variant(
                "3b", "Qwen2.5-VL-3B (CPU or GPU)",
                size_gb=6.2, vram_gb=4.5, fit="cpu", cpu_ok=True,
                gpu_cost=30.0, cpu_cost=130.0,
                hf="Qwen/Qwen2.5-VL-3B-Instruct",
            ),
            _variant(
                "7b", "Qwen2.5-VL-7B (recommended)",
                size_gb=15.5, vram_gb=9.0, fit="gpu", cpu_ok=False,
                gpu_cost=60.0, cpu_cost=380.0,
                hf="Qwen/Qwen2.5-VL-7B-Instruct",
            ),
        ],
        "default_variant": "7b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        "timestamps": True,
        "trust_remote_code": False,
    },
    {
        "id": "gemma4",
        "name": "Gemma 4",
        "license": "Apache 2.0",
        "languages_label": "140+",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u2705 E2B/E4B: CPU or GPU \u00b7 12B: GPU",
        "notes": "Google's natively multimodal family: text, image and audio "
                 "input (video as frames) on the E2B, E4B and 12B models, and "
                 "the 12B is encoder-free so all modalities enter one decoder.",
        "hallucination": "Moderate.  A strong reasoner, so it will occasionally "
                         "add context the frames do not show; the guideline "
                         "prompt is what stops it.",
        "variants": [
            _variant(
                "e2b", "Gemma 4 E2B (2.3B effective, CPU or GPU)",
                size_gb=5.0, vram_gb=4.0, fit="cpu", cpu_ok=True,
                gpu_cost=16.0, cpu_cost=70.0,
                hf="google/gemma-4-e2b-it",
                note="Designed for phones and laptops.",
            ),
            _variant(
                "e4b", "Gemma 4 E4B (4.5B effective, CPU or GPU)",
                size_gb=8.0, vram_gb=6.5, fit="cpu", cpu_ok=True,
                gpu_cost=30.0, cpu_cost=120.0,
                hf="google/gemma-4-e4b-it",
            ),
            _variant(
                "12b", "Gemma 4 12B Unified (audio and video, GPU)",
                size_gb=24.0, vram_gb=16.0, fit="gpu", cpu_ok=False,
                gpu_cost=65.0, cpu_cost=420.0,
                hf="google/gemma-4-12b-it",
                note="Above 12 GB in full precision; a quantised load fits.",
            ),
        ],
        "default_variant": "e4b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        "timestamps": True,
        "trust_remote_code": False,
    },
    {
        "id": "gemma3",
        "name": "Gemma 3",
        "license": "Gemma Terms of Use (not OSI-approved)",
        "languages_label": "140+",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u2705 4B: CPU or GPU \u00b7 12B and 27B: GPU",
        "notes": "Google's Gemma 3, with a vision encoder on the 4B, 12B and "
                 "27B checkpoints; the 1B is text-only and cannot describe "
                 "video, so it is not offered here.  Two things to know before "
                 "choosing it: the download comes from a gated Hugging Face "
                 "repository, so Google's Gemma terms have to be accepted "
                 "on the model page and the account signed in before the first "
                 "run; and Gemma 3 does not ground to a second, so its "
                 "descriptions sit at their chunk boundary rather than in a "
                 "pause of their own.",
        "hallucination": "Moderate.  A strong general reasoner, so it "
                         "occasionally adds context the sampled frames do not "
                         "show; the guideline prompt is what keeps it "
                         "descriptive rather than speculative.",
        "variants": [
            _variant(
                "4b", "Gemma 3 4B (CPU or GPU)",
                size_gb=8.5, vram_gb=7.0, fit="cpu", cpu_ok=True,
                gpu_cost=35.0, cpu_cost=170.0,
                hf="google/gemma-3-4b-it",
                note="Smallest Gemma 3 with a vision encoder.",
            ),
            _variant(
                "12b", "Gemma 3 12B (recommended on a GPU)",
                size_gb=24.0, vram_gb=16.0, fit="gpu", cpu_ok=False,
                gpu_cost=65.0, cpu_cost=480.0,
                hf="google/gemma-3-12b-it",
                note="Above 12 GB in full precision; a quantised load fits.",
            ),
            _variant(
                "27b", "Gemma 3 27B (big GPU)",
                size_gb=54.0, vram_gb=30.0, fit="gpu", cpu_ok=False,
                gpu_cost=95.0, cpu_cost=900.0,
                hf="google/gemma-3-27b-it",
                note="Above this app's 12 GB rule.",
            ),
        ],
        "default_variant": "4b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        "timestamps": False,
        "trust_remote_code": False,
        # Every Gemma 3 checkpoint lives in Google's *gated* repository: the
        # licence has to be accepted on the model's own page and the download
        # authenticated with a token from that account (checked against the live
        # API for all three variants).  The flag is not decoration - it is what
        # makes the application say so *before* a 24 GB download is attempted,
        # instead of surfacing a bare 401 in the middle of one.  See
        # :mod:`ai_voice_studio.hf_access`.
        "gated": True,
    },
    {
        "id": "minicpm_o",
        "name": "MiniCPM-o",
        "license": "Open weights (Apache-style)",
        "languages_label": "Multilingual",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": _FIT_GPU_ONLY,
        "notes": "Full-duplex omni model: sees and hears at 8B, so it is a "
                 "good draft engine for an 8-12 GB card.  A description reads "
                 "the picture, so its audio and speech halves are switched "
                 "off when it is loaded (see ``load_kwargs``).",
        "hallucination": "Moderate.  Good on faces and expression, weaker at "
                         "deciding what matters in a busy scene.",
        "variants": [
            _variant(
                "8b", "MiniCPM-o 2.6 (8B)",
                size_gb=17.0, vram_gb=11.0, fit="gpu", cpu_ok=False,
                gpu_cost=110.0, cpu_cost=700.0,
                hf="openbmb/MiniCPM-o-2_6",
                note="Needs trust_remote_code (its own modelling code), which "
                     "imports torchaudio and librosa and needs the "
                     "transformers 4.52 series.  Its checkpoint registers "
                     "itself as AutoModel, not as a task-specific class.",
            ),
        ],
        "default_variant": "8b",
        "backend": "transformers",
        # ``torchaudio`` and ``librosa`` are not optional here, and neither is
        # the pin.  This checkpoint runs its own modelling code with
        # ``trust_remote_code``, and that code imports packages that no other
        # describing model needs: ``processing_minicpmo.py`` imports
        # ``torchaudio`` (``transformers`` refuses to load the file without it -
        # "This modeling file requires the following packages that were not found
        # in your environment: torchaudio"), and ``utils.py`` imports
        # ``librosa``, which is pulled in by ``modeling_minicpmo.py``.  ``librosa``
        # must never be installed alone: it requires ``soxr``, and from
        # ``transformers`` 5 on ``transformers.audio_utils`` imports ``soxr``
        # whenever ``librosa`` is importable, so an environment holding one
        # without the other cannot even ``import transformers``.  Declaring
        # ``librosa`` is therefore enough - pip brings ``soxr`` with it, as it
        # did in the environment this was fixed in - and declaring ``soxr``
        # separately would only risk the two disagreeing on a version.
        # ``torchaudio`` is in
        # ``_CUDA_PACKAGES`` (see :mod:`ai_voice_studio.describe.environments`),
        # so on a machine with an NVIDIA card it is installed from the same
        # PyTorch wheel index as ``torch`` - a CUDA ``torch`` beside a CPU
        # ``torchaudio`` is a runtime crash, not a warning.
        "pip_packages": [_REMOTE_CODE_TRANSFORMERS, "torch", "torchvision",
                         "torchaudio", "accelerate", "pillow", "numpy",
                         "librosa", "soundfile"],
        # The class its ``auto_map`` registers, and the call its own code
        # documents.  Both are properties of the checkpoint, not preferences:
        # asking MiniCPM-o for ``AutoModelForImageTextToText`` is refused before
        # a weight is read (the error a user reported), and its processor has no
        # chat template that carries the standard multimodal message, so the way
        # in is ``model.chat`` with the frames in the user turn, as its card
        # writes it.
        "auto_class": "AutoModel",
        "driver": DRIVER_CHAT,
        # Vision-only, which is what a description is: the whisper encoder and
        # the speech-synthesis module are not built at all.  Its own code
        # asserts the TTS packages rather than skipping them when they are
        # requested, so this is also what keeps the environment from needing a
        # speech stack that describing never touches.
        "load_kwargs": {"init_audio": False, "init_tts": False},
        "timestamps": True,
        "trust_remote_code": True,
    },
    {
        "id": "minicpm_o_4_5",
        "name": "MiniCPM-o 4.5",
        "license": "Apache 2.0",
        "languages_label": "30+",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u26a0 GPU only - about 19 GB of video memory in bf16; a "
                     "12 GB card runs it in 4-bit",
        "notes": "The family's current flagship, a generation on from "
                 "MiniCPM-o 2.6: SigLIP2 vision, a Whisper-medium audio "
                 "encoder, speech synthesis and a Qwen3-8B backbone, about 9B "
                 "in all, reading high-resolution images and video at up to "
                 "10 fps.  Its environment is its own rather than a second "
                 "variant of MiniCPM-o 2.6, because ``minicpmo-utils`` has to "
                 "be installed before its first load - its own ``utils.py`` "
                 "imports ``minicpmo.utils``, which is that distribution's "
                 "module, and ``transformers`` imports every module a remote-"
                 "code file names when it caches it - "
                 "and it pins a Pillow and a librosa older than the ones "
                 "MiniCPM-o 2.6's environment carries.",
        "hallucination": "Low for its size: the model card reports "
                         "MMHal-Bench results around those of Gemini 2.5 "
                         "Flash, and this is the family's most current "
                         "checkpoint.  The guideline prompt still governs - a "
                         "description is not a caption competition.",
        "variants": [
            _variant(
                "9b", "MiniCPM-o 4.5 (9B, video and audio)",
                size_gb=18.7, vram_gb=19.0, fit="gpu", cpu_ok=False,
                gpu_cost=120.0, cpu_cost=800.0,
                hf="openbmb/MiniCPM-o-4_5",
                note="Needs trust_remote_code (its own modelling code) and the "
                     "transformers 4.52 series; it registers itself as "
                     "AutoModel.  Larger than a 12 GB card, so a GPU run loads "
                     "it in 4-bit.",
            ),
        ],
        "default_variant": "9b",
        "backend": "transformers",
        # Written from what the checkpoint's own files import, because that is
        # what ``transformers`` checks: its remote-code import check walks the
        # whole file, imports *inside* functions count, and only ``try`` blocks
        # are skipped.  ``modeling_minicpmo.py`` imports ``librosa``,
        # ``soundfile`` and ``tqdm`` beside numpy, PIL, torch and
        # ``huggingface_hub``; ``utils.py`` - imported by the modelling file, so
        # checked with it - imports ``minicpmo.utils``.  ``minicpmo.utils`` is
        # the ``minicpmo-utils`` distribution's own top-level package, so that
        # distribution is required rather than optional: the check runs on a
        # module as it is cached, which is the first load of a revision in a
        # fresh cache, and it refuses that load before a weight is read.  The
        # rest is covered elsewhere -
        # ``tqdm``, ``huggingface_hub`` and ``requests`` come with
        # ``transformers``, and numpy, PIL and ``soundfile`` are declared here.
        #
        # ``librosa`` has to be declared *and* importable, and on today's Python
        # that second half is what the extra pin buys.  ``minicpmo-utils`` pins
        # ``librosa==0.9.0``, whose ``librosa.util.files`` does ``from
        # pkg_resources import resource_filename``, and ``pkg_resources`` was
        # removed from ``setuptools`` 81.  An environment built with a current
        # setuptools therefore holds librosa and cannot import it - and the
        # checkpoint's own import check then refuses the load with
        # "requires the following packages that were not found in your
        # environment: librosa", which is true of the import and false of the
        # install.  ``setuptools<81`` is declared beside it for that reason, and
        # both halves were checked on the Python 3.13 this application builds:
        # with ``setuptools<81`` (80.10.2) ``import librosa`` and
        # ``import minicpmo.utils`` both succeed, while setuptools 84 fails the
        # first with ``No module named 'pkg_resources'``.  ``librosa`` is
        # declared unpinned - the pin that arrives with ``minicpmo-utils``
        # decides the version - so that the requirement is visible in the panel
        # and survives a future release that drops it.
        #
        # What the rest of ``minicpmo-utils`` costs is named in the entry: its
        # own ``Pillow==10.4.0`` pin, ``librosa==0.9.0``, and ``decord`` and
        # ``moviepy`` that a description never calls.  Every one of those has a
        # Windows path on Python 3.13 (``decord`` 0.6.0 ships a
        # ``py3-none-win_amd64`` wheel, Pillow 10.4.0 ships cp313 wheels), and
        # the chain that carries the pins was installed and imported for real to
        # confirm it - numpy, Pillow 10.4.0, librosa, soundfile and
        # ``minicpmo.utils`` all landed and imported, with ``setuptools<81``
        # present.  (That distribution also ships ``s3tokenizer``, which needs
        # ``einops`` and is not part of this checkpoint's requirement set; it is
        # left alone rather than declared for a load that never names it.)
        "pip_packages": [_REMOTE_CODE_TRANSFORMERS, "torch", "torchvision",
                         "torchaudio", "accelerate", "pillow", "numpy",
                         "librosa", "soundfile", "setuptools<81",
                         "minicpmo-utils>=1.0.5"],
        # Same properties of the checkpoint as MiniCPM-o 2.6: it registers
        # ``AutoModel`` (nothing for the task-specific classes), its own
        # ``chat`` is the documented way in, and the audio and speech halves are
        # switched off, because a description reads the picture.
        "auto_class": "AutoModel",
        "driver": DRIVER_CHAT,
        "load_kwargs": {"init_audio": False, "init_tts": False},
        "timestamps": True,
        "trust_remote_code": True,
    },
    {
        "id": "minicpm_v_4",
        "name": "MiniCPM-V 4",
        "license": "Apache 2.0",
        "languages_label": "Multilingual",
        "languages": _ENGLISH_FIRST,
        "fit": "gpu",
        "fit_label": "\u2705 4.1B: CPU or GPU",
        "notes": "The 4B model of the MiniCPM-o/V family, and the one that "
                 "answers \"is there a small MiniCPM that reads video?\": "
                 "there is no 4B MiniCPM-o - the -o line (2.6, 4.5) starts at "
                 "8B - while MiniCPM-V 4 is 4.1B, image/video and multi-image "
                 "understanding at GPT-4.1-mini level on its own benchmarks, "
                 "and built for on-device use.  Vision only: it has no audio "
                 "or speech half at all, so it needs none of the packages "
                 "MiniCPM-o needs.",
        "hallucination": "Higher than the 8B-and-up models, as a 4B model "
                         "usually is: it can add detail the sampled frames do "
                         "not show.  The guideline prompt is what keeps it "
                         "descriptive rather than speculative.",
        "variants": [
            _variant(
                "4.1b", "MiniCPM-V 4.0 (4.1B, video)",
                size_gb=8.1, vram_gb=9.0, fit="gpu", cpu_ok=True,
                gpu_cost=40.0, cpu_cost=180.0,
                hf="openbmb/MiniCPM-V-4",
                note="Needs trust_remote_code (its own modelling code) and the "
                     "transformers 4.52 series; it registers itself as "
                     "AutoModel.  Fits an 8-12 GB card, and runs on a CPU.",
            ),
        ],
        "default_variant": "4.1b",
        "backend": "transformers",
        # Lighter than MiniCPM-o's environment by exactly the parts it does not
        # have: no ``torchaudio`` (nothing here decodes audio),
        # no ``librosa`` and no speech stack.  What it does need is the pin: its
        # own files import ``transformers.modeling_attn_mask_utils``, and its
        # ``config.json`` records ``transformers_version`` 4.51.0.
        "pip_packages": [_REMOTE_CODE_TRANSFORMERS, "torch", "torchvision",
                         "accelerate", "pillow", "numpy", "soundfile"],
        "auto_class": "AutoModel",
        "driver": DRIVER_CHAT,
        "timestamps": True,
        "trust_remote_code": True,
    },
    {
        "id": "smolvlm2",
        "name": "SmolVLM2 (CPU)",
        "license": "Apache 2.0",
        "languages_label": "English",
        "languages": [("en", "English")],
        "fit": "cpu",
        "fit_label": _FIT_BOTH_CPU_FIRST,
        "notes": "Small enough to run on a laptop with no graphics card at all, "
                 "which keeps the app's CPU-only promise for description.",
        "hallucination": "Higher relative to its size: it invents objects more "
                         "readily than the larger models.  Use it for short "
                         "clips.",
        "variants": [
            _variant(
                "2.2b", "SmolVLM2 2.2B",
                size_gb=4.5, vram_gb=4.0, fit="cpu", cpu_ok=True,
                gpu_cost=25.0, cpu_cost=90.0,
                hf="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            ),
            _variant(
                "500m", "SmolVLM2 500M",
                size_gb=1.1, vram_gb=1.0, fit="cpu", cpu_ok=True,
                gpu_cost=10.0, cpu_cost=35.0,
                hf="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            ),
            _variant(
                "256m", "SmolVLM2 256M (fastest)",
                size_gb=0.6, vram_gb=0.6, fit="cpu", cpu_ok=True,
                gpu_cost=6.0, cpu_cost=20.0,
                hf="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            ),
        ],
        "default_variant": "2.2b",
        "backend": "transformers",
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        "timestamps": True,
        "trust_remote_code": False,
    },
    {
        "id": "moondream2",
        "name": "Moondream 2 (CPU)",
        "license": "Apache 2.0",
        "languages_label": "English",
        "languages": [("en", "English")],
        "fit": "cpu",
        "fit_label": _FIT_BOTH_CPU_FIRST,
        "notes": "Tiny and fast on a CPU; useful as a first run to see the "
                 "whole pipeline work before downloading a 16 GB model.  It "
                 "answers one picture at a time (its own API takes one image "
                 "per question), so each window is described from its last "
                 "frame.",
        "hallucination": "Higher: short answers, and it will fill a gap rather "
                         "than admit it cannot see.  Treat the output as a draft.",
        "variants": [
            _variant(
                "1.9b", "Moondream 2 (1.9B)",
                size_gb=3.8, vram_gb=3.5, fit="cpu", cpu_ok=True,
                gpu_cost=14.0, cpu_cost=55.0,
                hf="vikhyatk/moondream2",
                note="Needs trust_remote_code (its own modelling code) and its "
                     "own ``query`` API; one picture per description.  Runs on "
                     "a CPU.",
            ),
        ],
        "default_variant": "1.9b",
        "backend": "transformers",
        # Its own files name ``transformers``, ``torch``, ``torchvision``,
        # ``PIL``, ``numpy`` and ``tokenizers`` - and ``einops`` only in
        # ``vision_encoder.py``, which nothing in the repository imports, so
        # that one is not required to load it (``transformers`` checks the
        # modules a checkpoint's files reach, not every file it ships).  What
        # arrives with ``transformers`` covers ``tokenizers``.
        "pip_packages": ["transformers", "torch", "torchvision", "accelerate",
                         "pillow", "numpy", "soundfile"],
        # The class its ``auto_map`` registers, and the only one that fits: its
        # ``config.json`` maps ``AutoModelForCausalLM`` to ``hf_moondream.
        # HfMoondream`` and maps nothing to the task-specific classes, so a run
        # that starts from those reaches this one only after two refusals.
        "auto_class": "AutoModelForCausalLM",
        # No processor and no chat template exist here - the repository carries
        # a tokenizer and its own modelling code, and the multimodal message the
        # generic path builds cannot reach either.  What its own code documents
        # is a set of skills: ``query(image, question)``, ``caption(image,
        # length)``, ``detect`` and ``point``.  A description is a question, so
        # ``query`` is what this model is asked, one picture at a time.
        "driver": DRIVER_QUERY,
        # The dtype is part of the checkpoint too, and this one is why the field
        # exists.  Its ``config.json`` records ``torch_dtype`` bfloat16, its own
        # vision code casts every crop to bfloat16 in the forward pass, and the
        # layers it builds take that dtype from the model's constructor default.
        # A CPU run that forced float32 - the choice every other describing
        # model takes, and the reason the worker defaults to it - would then
        # hand float32 weights to a bfloat16 activation and fail in the first
        # linear layer.  ``auto`` asks ``transformers`` for the checkpoint's own
        # dtype instead: the config's ``torch_dtype`` when it has one, the
        # weights' dtype otherwise.
        "dtype": "auto",
        "timestamps": True,
        "trust_remote_code": True,
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
    """``(value, label)`` choices for a combo box."""
    return [(v["value"], v["label"]) for v in model(model_id)["variants"]]


def variant_entries(model_id: str) -> list[dict]:
    """The full variant records (size, VRAM, fit, cost)."""
    return [dict(v) for v in model(model_id)["variants"]]


def variant_entry(model_id: str, variant: str) -> dict:
    """One variant record; falls back to the model's default variant."""
    entries = model(model_id)["variants"]
    for entry in entries:
        if entry["value"] == variant:
            return dict(entry)
    for entry in entries:
        if entry["value"] == default_variant(model_id):
            return dict(entry)
    return dict(entries[0])


def default_variant(model_id: str) -> str:
    return model(model_id)["default_variant"]


def languages(model_id: str) -> list[tuple[str, str]]:
    return list(model(model_id)["languages"])


def hf_model(model_id: str, variant: str = "") -> str:
    """The Hugging Face repository for this model/variant (``""`` if none)."""
    return str(variant_entry(model_id, variant or default_variant(model_id)).get("hf") or "")


def gated(model_id: str, variant: str = "") -> bool:
    """Whether this variant's repository is *gated*.

    A gated repository downloads only for an account that has accepted its
    licence, so a run needs a Hugging Face token (Settings > Hugging Face
    account).  Marked per model and per variant, because a family can mix the
    two - Google's Gemma 4 is open while every Gemma 3 checkpoint is gated.
    Gemma 3 is the one entry of this catalogue of that kind; the flag exists so
    the application can say so before the download rather than after it.
    """
    entry = model(model_id)
    if variant:
        return bool(variant_entry(model_id, variant).get("gated")
                    or entry.get("gated", False))
    return bool(entry.get("gated", False))


def gated_note(model_id: str, variant: str = "") -> str:
    """A short line naming the extra step a gated repository needs.

    Appended to the status the Settings table and the wizard show beside a
    variant, so the requirement is visible at the moment the model is chosen
    rather than only when a run fails.
    """
    if not gated(model_id, variant):
        return ""
    return ("Gated repository: accept its licence on Hugging Face and add a "
            "token in Settings > Hugging Face account")


def pip_packages(model_id: str) -> list[str]:
    return list(model(model_id)["pip_packages"])


def cuda_packages(model_id: str) -> list[str]:
    """Extra packages needed *only* for a CUDA run.

    Every describing model gets ``bitsandbytes``: a checkpoint larger than the
    card (Gemma 3 12B is about 24 GB of weights on a 12 GB card) is loaded in
    4-bit through it, which is the difference between a run and the process
    dying when the model is spilled to the disk.  It is a CUDA-only dependency,
    so a machine with no NVIDIA card installs none of it - which is also why an
    entry may add its own:

    * ``cuda_packages`` - pip packages the model's own environment additionally
      needs to run on the GPU.
    """
    packages = list(model(model_id).get("cuda_packages", []))
    if "bitsandbytes" not in packages:
        packages.append("bitsandbytes")
    return packages


def backend(model_id: str) -> str:
    return model(model_id)["backend"]


def backend_modules(model_id: str) -> list[str]:
    """Import modules that prove this model can run, primary back-end first."""
    entry = model(model_id)
    ids = [entry["backend"], *entry.get("alt_backends", [])]
    return [BACKEND_MODULES[b] for b in ids if b in BACKEND_MODULES]


def needs_remote_code(model_id: str) -> bool:
    """Whether the model's repository ships code that must be trusted."""
    return bool(model(model_id).get("trust_remote_code", False))


def auto_class(model_id: str) -> str:
    """The ``transformers`` Auto class this checkpoint's ``auto_map`` registers.

    ``""`` means the ordinary task-specific class is the right one, which is the
    case for every checkpoint ``transformers`` supports natively.  A repository
    that ships its own modelling code names the implementation it provides in
    its ``config.json``, and *that name is part of the checkpoint*: MiniCPM-o
    2.6 and MiniCPM-V 4 map ``AutoModel`` to ``MiniCPMO``/``MiniCPMV`` and map
    nothing at all to ``AutoModelForImageTextToText``, so a run that asks for
    the task-specific class is refused before a single weight is read::

        Unrecognized configuration class <class '...MiniCPMOConfig'> for this
        kind of AutoModel: AutoModelForImageTextToText.  Model type should be
        one of AriaConfig, AyaVisionConfig, ...

    That refusal has nothing to do with the model's own environment being
    wrong: the environment was right and the *question* was wrong.  The field is
    how the catalogue answers it correctly.
    """
    return str(model(model_id).get("auto_class") or "")


def driver(model_id: str) -> str:
    """How a run asks this model for a description (see :data:`DRIVER_CHAT`)."""
    return str(model(model_id).get("driver") or DRIVER_GENERATE)


def dtype(model_id: str) -> str:
    """The dtype to load this checkpoint in, or ``""`` for the default.

    Empty for every model but Moondream 2, which is the one whose own code
    fixes bfloat16 in its forward pass: a CPU run forcing float32 would fail in
    its first linear layer.  ``"auto"`` is how that is asked for - the
    checkpoint's own dtype, from its ``config.json`` or its weights.
    """
    return str(model(model_id).get("dtype") or "")


def load_kwargs(model_id: str) -> dict:
    """Extra ``from_pretrained`` keywords this checkpoint wants.

    MiniCPM-o 2.6 is an omni model: its config builds a Whisper audio encoder
    and a speech-synthesis module beside the vision stack, and its own code
    asserts the TTS packages rather than skipping them ("please make sure
    vector_quantize_pytorch and vocos are installed").  Audio description reads
    the *picture* - this application never hands a describing model the
    soundtrack - so the audio and speech halves are switched off at load.  That
    is the model card's own advice for vision-only use, and it is what keeps a
    description from needing a speech stack it will never use.
    """
    return dict(model(model_id).get("load_kwargs") or {})


def supports_timestamps(model_id: str) -> bool:
    """Whether a description can be tied to the chunk's own timecodes."""
    return bool(model(model_id).get("timestamps", True))


def hallucination_note(model_id: str) -> str:
    """Plain-language note on how much this model invents."""
    return str(model(model_id).get("hallucination", ""))


# ---------------------------------------------------------------------------
# Variant numbers (size, video memory, fit, compute)
# ---------------------------------------------------------------------------
def variant_size_gb(model_id: str, variant: str = "") -> float:
    """Approximate download size of this variant, in gigabytes."""
    return float(variant_entry(model_id, variant)["size_gb"])


def variant_vram_gb(model_id: str, variant: str = "") -> float:
    """Approximate video memory this variant needs, in gigabytes."""
    return float(variant_entry(model_id, variant)["vram_gb"])


def fit(model_id: str, variant: str = "") -> str:
    """The device this variant is *preferred* on (``"cpu"`` or ``"gpu"``)."""
    entry = variant_entry(model_id, variant)
    return str(entry.get("fit") or model(model_id).get("fit") or FIT_CPU)


def cpu_ok(model_id: str, variant: str = "") -> bool:
    """Whether this variant can genuinely be run on a CPU (slowly)."""
    return bool(variant_entry(model_id, variant).get("cpu_ok", True))


def gpu_only(model_id: str, variant: str = "") -> bool:
    """Whether this variant cannot run without an NVIDIA GPU."""
    return not cpu_ok(model_id, variant)


def cost_per_video_minute(model_id: str, variant: str = "", device: str = "cpu") -> float:
    """Seconds of compute per minute of video at the catalog's reference rate."""
    entry = variant_entry(model_id, variant)
    if device == "cuda":
        return float(entry.get("gpu_cost", 60.0))
    return float(entry.get("cpu_cost", 240.0))


def variant_rows(model_id: str) -> list[tuple[str, str, str, str, str, str]]:
    """The variant table rows of Settings > Manage the video describer.

    ``(value, label, size, vram, fit_label, note)`` - the size and VRAM columns
    are what let a user pick by hardware instead of by name.
    """
    rows: list[tuple[str, str, str, str, str, str]] = []
    for entry in variant_entries(model_id):
        rows.append((
            entry["value"],
            entry["label"],
            f"{entry['size_gb']:.1f} GB",
            f"{entry['vram_gb']:.1f} GB",
            "GPU" if not entry["cpu_ok"] else "CPU or GPU",
            str(entry.get("note") or ""),
        ))
    return rows


def device_options(model_id: str, variant: str = "",
                   has_cuda: bool = True) -> list[tuple[str, str]]:
    """Compute choices the describe widget may offer for this model/variant.

    The rule is the STT catalogue's, for the same reason - many users have no
    NVIDIA card at all:

    * a **GPU-only** variant is offered the GPU alone, because a CPU run could
      not work and should not be dangled in front of the user;
    * **every other variant** is offered both the CPU and the GPU whenever the
      machine has an NVIDIA card (the preferred device first), and the CPU alone
      when it has not.

    ``Auto`` is deliberately never offered: the combo is about the model, not
    about a machine-wide preference.
    """
    if gpu_only(model_id, variant):
        if has_cuda:
            return [("cuda", "GPU (CUDA)")]
        return [(
            "cuda",
            "GPU (CUDA) - no NVIDIA GPU detected, and this model "
            "cannot run on a CPU",
        )]
    if not has_cuda:
        label = "CPU"
        if fit(model_id, variant) == FIT_GPU:
            label = (
                "CPU (no NVIDIA GPU detected - this model runs on the CPU, "
                "but it is much faster on a GPU)"
            )
        return [("cpu", label)]
    gpu_option = ("cuda", "GPU (CUDA)")
    cpu_option = ("cpu", "CPU")
    if fit(model_id, variant) == FIT_GPU:
        return [gpu_option, cpu_option]
    return [cpu_option, gpu_option]


def summary_rows() -> list[tuple[str, str, str, str]]:
    """``(name, license, languages, fit_label)`` one row per model."""
    return [
        (m["name"], m["license"], m["languages_label"], m["fit_label"])
        for m in MODELS
    ]


def cheapest_cpu_model() -> str | None:
    """The smallest CPU-capable model id (fine for a first run)."""
    best: tuple[float, str] | None = None
    for entry in MODELS:
        for variant in entry["variants"]:
            if not variant["cpu_ok"]:
                continue
            size = float(variant["size_gb"])
            if best is None or size < best[0]:
                best = (size, entry["id"])
    return best[1] if best else None


def default_model_id() -> str:
    """The model the wizard and the dialog preselect when there is no memory.

    Deliberately the smallest CPU-capable one, not the first table row: the
    table is ordered by quality, but a *first* run must not open on a 37 GB
    model that cannot run on the machine in front of the user.  A saved choice
    always wins over this; the preselect only decides the unconfigured case.
    """
    return cheapest_cpu_model() or MODELS[0]["id"]


#: Seconds of compute per minute of video at :data:`_DEFAULT_FPS_COST`, kept as a
#: module attribute so the estimator does not have to restate it.
REFERENCE_FPS = _DEFAULT_FPS_COST
