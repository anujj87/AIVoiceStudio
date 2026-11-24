"""Voice Lab engine registry (the voice-clone engines).

These engines are **not** sherpa-onnx ONNX artifacts: each one is a Python
package that runs in the app's managed virtualenv (see ``python_runtime``) and
is driven by ``voicelab.worker`` in a subprocess, exactly like OmniVoice.

Every one of them is a *voice cloning* engine.  The smaller ones run on a
**CPU** and on the **GPU** when an NVIDIA card is present; the large
transformer engines (their ``devices`` is ``gpu``) run on the GPU alone,
because their CPU run is many times slower than real time.  The choice is
offered both here and per project in the Recording window.

Because the voices of these engines are *provided by the installed package*
rather than downloaded as artifacts, they are defined here in Python instead of
``tts/models_catalog.json``: F5-TTS's built-in voices, for instance, point at
reference files *inside* the installed package
(``f5_tts:infer/examples/basic/basic_ref_en.wav``), and KokoClone's are derived
from the Kokoro voices it synthesises with.

Each voice entry has the same shape as the ones ``ModelStore.installed_voices``
produces, so the Download/Available/Recording panels need no special casing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------------
# Engine metadata
# ---------------------------------------------------------------------------

#: Kyutai Pocket TTS pre-made voices: (voice id, language code).
_POCKET_VOICES: tuple[tuple[str, str], ...] = (
    ("alba", "en"),
    ("anna", "en"),
    ("azelma", "en"),
    ("bill_boerst", "en"),
    ("caro_davy", "en"),
    ("charles", "en"),
    ("cosette", "en"),
    ("eponine", "en"),
    ("eve", "en"),
    ("fantine", "en"),
    ("george", "en"),
    ("jane", "en"),
    ("javert", "en"),
    ("jean", "en"),
    ("marius", "en"),
    ("mary", "en"),
    ("michael", "en"),
    ("paul", "en"),
    ("peter_yearsley", "en"),
    ("stuart_bell", "en"),
    ("vera", "en"),
    ("estelle", "fr"),
    ("juergen", "de"),
    ("giovanni", "it"),
    ("lola", "es"),
    ("rafael", "pt"),
)

#: Chatterbox Multilingual's 23 languages (the model's own language ids).
_CHATTERBOX_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
)

#: The variant ids of the three Chatterbox models.
CHATTERBOX_ENGLISH_VARIANT = "chatterbox_v1"
CHATTERBOX_TURBO_VARIANT = "chatterbox_turbo"
CHATTERBOX_MULTILINGUAL_VARIANT = "chatterbox_mtl"

#: Chatterbox's voice id for "just use the model's own default speaker" -
#: it ships no named voices, so this is the only built-in voice it has.
CHATTERBOX_DEFAULT_VOICE = "default"

#: Qwen3-TTS models.  Each variant is one model on Hugging Face; ``kind``
#: decides which of the package's three entry points speaks with it (its
#: ``CustomVoice`` models have nine named speakers, its ``Base`` models clone a
#: voice from a recording).
_QWEN_MODELS: tuple[tuple[str, str, str, str], ...] = (
    (
        "qwen3_custom_1_7b",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "custom",
        "Qwen3-TTS 1.7B CustomVoice (best quality, nine voices)",
    ),
    (
        "qwen3_custom_0_6b",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "custom",
        "Qwen3-TTS 0.6B CustomVoice (lighter, nine voices)",
    ),
    (
        "qwen3_clone_1_7b",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "clone",
        "Qwen3-TTS 1.7B Base (clones a voice from a recording)",
    ),
    (
        "qwen3_clone_0_6b",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "clone",
        "Qwen3-TTS 0.6B Base (clones a voice from a recording)",
    ),
)

#: Qwen3-TTS's nine pre-made voices: ``(voice id, native language, blurb)``.
#: The package calls both the id and the human name ``speaker``, so the id *is*
#: the name it is asked for by.  Qwen recommends each speaker's own language,
#: but every speaker can speak every language the models support.
_QWEN_SPEAKERS: tuple[tuple[str, str, str], ...] = (
    ("Vivian", "zh", "Bright, slightly edgy young female voice."),
    ("Serena", "zh", "Warm, gentle young female voice."),
    ("Uncle_Fu", "zh", "Seasoned male voice with a low, mellow timbre."),
    ("Dylan", "zh", "Youthful Beijing male voice with a clear, natural timbre."),
    ("Eric", "zh", "Lively Chengdu male voice with a slightly husky brightness."),
    ("Ryan", "en", "Dynamic male voice with strong rhythmic drive."),
    ("Aiden", "en", "Sunny American male voice with a clear midrange."),
    ("Ono_Anna", "ja", "Playful Japanese female voice with a light, nimble timbre."),
    ("Sohee", "ko", "Warm Korean female voice with rich emotion."),
)

#: The ten languages Qwen3-TTS covers.
_QWEN_LANGUAGES: tuple[str, ...] = (
    "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
)

#: VibeVoice's two models: ``(variant id, Hugging Face repo, display label)``.
#:
#: Both are the same architecture at two sizes (see the community fork's model
#: zoo): the 1.5B model keeps a 64K context and can speak up to ~90 minutes in
#: one pass, the 7B ("Large") model a 32K context and ~45 minutes.  The
#: weights are downloaded from Hugging Face on first use, like every other
#: engine's; they are not part of the install.
_VIBEVOICE_MODELS: tuple[tuple[str, str, str], ...] = (
    (
        "vibevoice_1_5b",
        "vibevoice/VibeVoice-1.5B",
        "VibeVoice 1.5B (needs ~8 GB VRAM, up to 90 minutes per run)",
    ),
    (
        "vibevoice_7b",
        "vibevoice/VibeVoice-7B",
        "VibeVoice Large 7B (needs ~20 GB VRAM, up to 45 minutes per run)",
    ),
)

#: VibeVoice's voice presets: ``(voice id, language code, repo file, label)``.
#:
#: The fork ships these as reference recordings (``demo/voices/*.wav``): with
#: no clip of your own the model conditions on one of them, which is how its
#: built-in voices are defined.  They are fetched from the repository the
#: engine is installed from and cached in the engine's own folder on first
#: use, because the pip package does not carry the ``demo`` folder.
#: ``Mary`` and ``Anchen`` are deliberately absent: those presets contain
#: background music, which the model then tends to reproduce.
_VIBEVOICE_VOICES: tuple[tuple[str, str, str, str], ...] = (
    ("Alice", "en", "en-Alice_woman.wav", "Alice - English woman"),
    ("Maya", "en", "en-Maya_woman.wav", "Maya - English woman"),
    ("Carter", "en", "en-Carter_man.wav", "Carter - English man"),
    ("Frank", "en", "en-Frank_man.wav", "Frank - English man"),
    ("Samuel", "en", "in-Samuel_man.wav", "Samuel - Indian English man"),
    ("Bowen", "zh", "zh-Bowen_man.wav", "Bowen - Chinese man"),
    ("Xinran", "zh", "zh-Xinran_woman.wav", "Xinran - Chinese woman"),
)

# ---------------------------------------------------------------------------
# MOSS-TTS (OpenMOSS / MOSI.AI)
# ---------------------------------------------------------------------------
#: The MOSS-TTS checkpoints this build offers: ``(variant id, Hugging Face repo,
#: label, video memory in MB, devices)``.
#:
#: Every variant listed here fits inside **12 GB of video memory** in bfloat16,
#: which is the ceiling this integration is held to: the 8B ``MOSS-TTS-v1.5``
#: checkpoint needs roughly 19 GB, so it is deliberately absent, and what
#: remains is the latest of each size that does fit.  Both load through the
#: standard ``AutoProcessor``/``AutoModel`` interface with their repository's
#: own remote code.
_MOSS_MODELS: tuple[tuple[str, str, str, int, str], ...] = (
    (
        "moss_tts_local_v15_4b",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "MOSS-TTS Local Transformer v1.5 4B "
        "(48 kHz stereo, latest)",
        9500,
        "gpu",
    ),
    (
        "moss_tts_local_1_7b",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        "MOSS-TTS Local Transformer 1.7B (48 kHz stereo)",
        6500,
        "gpu",
    ),
)

#: MOSS-TTS is published as a **source tree on GitHub** (its ``pyproject.toml``
#: builds the ``moss-tts`` distribution, which is not on PyPI), so it is
#: installed from a pinned source archive.  The pin is the commit this
#: integration was reviewed against (2026-10-03); an unpinned ``main`` would
#: change under the user's feet.
MOSS_SOURCE_URL = (
    "https://github.com/OpenMOSS/MOSS-TTS/archive/"
    "934d6826b084c46a0d033402174d5f8ac4ed2519.zip"
)

#: The 31 languages MOSS-TTS-v1.5 and its Local-Transformer sibling cover.
#: ``(code, name)``; the code is the short tag the engine is asked for.
_MOSS_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("zh", "Chinese"),
    ("yue", "Cantonese"),
    ("en", "English"),
    ("ar", "Arabic"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sw", "Swahili"),
    ("sv", "Swedish"),
    ("tl", "Tagalog"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
)

#: The same tags as a ``code -> name`` mapping, which is the shape the
#: Local-Transformer API takes and the worker keeps its own copy of.
_MOSS_LANGUAGE_NAMES: dict[str, str] = dict(_MOSS_LANGUAGES)

#: MOSS-TTS's voice id for "speak without recording a reference".  The Local
#: Transformer checkpoints synthesise from the text alone, so this is the one
#: built-in voice every variant has, and every other voice is a clone.
MOSS_DEFAULT_VOICE = "default"

# ---------------------------------------------------------------------------
# GPT-SoVITS (RVC-Boss)
# ---------------------------------------------------------------------------
#: The GPT-SoVITS releases: ``(variant id, version, label, video memory MB)``.
#:
#: Every one of them is far smaller than the 12 GB ceiling - the GPT stage is a
#: ~155 MB checkpoint and the SoVITS stage a 77-770 MB one, with a shared
#: RoBERTa (~650 MB) and HuBERT (~190 MB) on top.  The version string is the
#: engine's own ``version`` key, which is what picks its weights and languages.
_GPTSOVITS_MODELS: tuple[tuple[str, str, str, int], ...] = (
    (
        "gptsovits_v2proplus",
        "v2ProPlus",
        "GPT-SoVITS v2ProPlus (latest, best quality)",
        2600,
    ),
    ("gptsovits_v2pro", "v2Pro", "GPT-SoVITS v2Pro", 2400),
    ("gptsovits_v4", "v4", "GPT-SoVITS v4 (48 kHz vocoder)", 3200),
    ("gptsovits_v3", "v3", "GPT-SoVITS v3 (407M, best cloning)", 3000),
    ("gptsovits_v2", "v2", "GPT-SoVITS v2 (fast, stable)", 1600),
    ("gptsovits_v1", "v1", "GPT-SoVITS v1 (original)", 1400),
)

#: GPT-SoVITS has **no** ``pyproject.toml``/``setup.py``, so it cannot be
#: ``pip install``-ed from its archive the way the other source engines are.
#: The worker fetches this pinned archive into the engine's own folder, adds it
#: to ``sys.path`` and downloads the pretrained weights beside it instead (see
#: ``_SOURCE_ARCHIVES`` and the worker's ``GptSovitsBackend``).
GPT_SOVITS_SOURCE_URL = (
    "https://github.com/RVC-Boss/GPT-SoVITS/archive/"
    "48b1a0169a28582a8984402f82cf438d3bfa6aca.zip"
)

#: The Hugging Face repository holding every GPT-SoVITS pretrained weight.
GPT_SOVITS_WEIGHTS_REPO = "lj1995/GPT-SoVITS"

#: The languages GPT-SoVITS v2 and later cover, as its own ``*_lang`` codes.
#: ``(code, name)``.
#: GPT-SoVITS's language coverage.  Japanese is deliberately **absent**: its
#: front-end imports ``pyopenjtalk``, which upstream publishes as a source
#: distribution only - there is no wheel for the Python 3.13 this application
#: runs on, and building it needs CMake, MSVC and the OpenJTalk submodules.
#: Offering it would only turn a Japanese voice into an install-time failure,
#: so the engine is listed with the languages it can really speak.
_GPTSOVITS_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
    ("yue", "Cantonese"),
)

#: GPT-SoVITS is a **voice-cloning** engine with no voices of its own: it always
#: speaks from a reference recording and its transcript.  It therefore ships no
#: built-in voices at all - the ones that appear are the voices the user creates
#: or clones on the Voice Clone page.
GPTSOVITS_BUILTIN_VOICES: tuple[dict[str, Any], ...] = ()

#: VibeVoice is published as a **source tree on GitHub**, not on PyPI, so it
#: is installed from a pinned source archive - the same way KokoClone is.  The
#: pin is the commit this integration was reviewed against (2026-08-29); an
#: unpinned ``main`` would change under the user's feet, and the engine's own
#: modules import each other by path.
VIBEVOICE_SOURCE_URL = (
    "https://github.com/vibevoice-community/VibeVoice/archive/"
    "952326ddb264062466a888cf32a5b2f4e803e16e.zip"
)

#: Raw-file base for :data:`_VIBEVOICE_VOICES`, at the same pinned commit, so
#: a preset the worker fetches is the one this build was checked against.
VIBEVOICE_VOICES_BASE_URL = (
    "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/"
    "952326ddb264062466a888cf32a5b2f4e803e16e/demo/voices/"
)

#: F5-TTS reference voices bundled with the ``f5-tts`` package.
_F5_REFS: tuple[tuple[str, str, str, str], ...] = (
    (
        "basic_ref_en",
        "en",
        "infer/examples/basic/basic_ref_en.wav",
        "Some call me nature, others call me mother nature.",
    ),
    (
        "basic_ref_zh",
        "zh",
        "infer/examples/basic/basic_ref_zh.wav",
        "对，这就是我，万人敬仰的太乙真人。",
    ),
)

#: KokoClone's built-in voices: the Kokoro voices it synthesises with *before*
#: a reference voice is applied - one for each language it speaks.  Every other
#: KokoClone voice is a reference recording of your own (see "Clone a voice from
#: a sample").
#:
#: The voice ids are Kokoro's own names, and the language decides which of
#: KokoClone's Kokoro models and phonemizers is used for it.  Japanese is
#: deliberately absent: the phonemiser behind it (``pyopenjtalk``) is published
#: as source only and needs a C++ toolchain, which the application's installer
#: cannot require.
#:
#: ``(voice id, language code, display name)``
_KOKOCLONE_VOICES: tuple[tuple[str, str, str], ...] = (
    ("af_bella", "en", "Bella"),
    ("hf_alpha", "hi", "Alpha"),
    ("ff_siwis", "fr", "Siwis"),
    ("em_alex", "es", "Alex"),
    ("im_nicola", "it", "Nicola"),
    ("pf_dora", "pt", "Dora"),
    ("zf_001", "zh", "Voice 001"),
)

#: KokoClone itself and the Kanade voice-conversion model it clones with are
#: published as **source trees on GitHub**, not on PyPI, so they are installed
#: from a pinned source archive.  The pins are the commits this integration was
#: reviewed against: KokoClone is young, and an unpinned ``main`` would change
#: under the user's feet.  Nothing here is fetched at import time - ``pip``
#: downloads the archive only when the user installs the engine.
KOKOCLONE_SOURCE_URL = (
    "https://github.com/Ashish-Patnaik/kokoclone/archive/"
    "dd6bd3acd3010dc223978839761db74957195f98.zip"
)
KANADE_SOURCE_URL = (
    "https://github.com/frothywater/kanade-tokenizer/archive/"
    "961f20bf892c59f391d0b6c5f7b88e70ed919b99.zip"
)

#: ``pip`` steps an engine needs *besides* the plain PyPI names in
#: ``packages``, as ``(arguments, index_url, names)``:
#:
#: * ``arguments`` are handed to ``pip install`` as they are and may hold pip
#:   flags next to requirement specs,
#: * ``names`` are the **distribution names** those arguments install, so
#:   ``install_plan`` never also asks for them by name (they are not on PyPI)
#:   while ``packages`` still lists them for ``pip uninstall``.
#:
#: The KokoClone steps are:
#:
#: * ``misaki`` - the phonemiser every KokoClone call routes text through.  It
#:   declares ``Requires-Python: <3.13`` while this application runs on 3.13,
#:   and the flag is what makes ``pip`` install it anyway.  The package is pure
#:   Python and imports fine on 3.13 (only its optional extras, which need
#:   Japanese/English toolchains, do not), and KokoClone only uses its eSpeak
#:   phonemiser, whose dependencies come with ``kokoro-onnx``.
#: * KokoClone and Kanade themselves, from their pinned source archives.
#:   ``--no-deps``: their own dependency lists would replace the CUDA build of
#:   PyTorch installed first with PyPI's CPU-only wheel, which would quietly
#:   take the GPU option away from the user.


_EXTRA_REQUIREMENTS: dict[str, tuple[tuple[tuple[str, ...], str | None,
                                           tuple[str, ...]], ...]] = {
    # VibeVoice is a **source archive**, not a PyPI package.  Its own
    # dependency list (pinned transformers==4.51.3 and accelerate==1.6.0, plus
    # the Gradio the fork's demo uses) rides along in this same step, which is
    # safe because this engine gets an environment of its own (see
    # ``own_environment``) and PyTorch is already installed from the CUDA index
    # on a GPU machine, so pip leaves the CUDA build in place.
    "vibevoice": (
        ((VIBEVOICE_SOURCE_URL,), None, ("vibevoice",)),
    ),
    # MOSS-TTS builds the ``moss-tts`` distribution from its source archive
    # (the archive's root holds the ``pyproject.toml``, so pip accepts the bare
    # URL); its own dependency list rides along in the same step, which is safe
    # because this engine gets an environment of its own.  PyTorch is installed
    # first from the CUDA index on a GPU machine and ``moss-tts`` does not pin
    # it, so pip leaves the CUDA build in place.  A bare URL is used because
    # ``python_runtime.pip_install`` strips pip flags and splits on spaces.
    #
    # The second step pins ``transformers`` to the release MOSS-TTS's remote
    # code is written against.  The archive declares that pin only in its
    # ``torch-runtime`` extra, which a bare URL install does not select, so
    # without this step pip takes the newest ``transformers`` - and the
    # checkpoint's own code then breaks against it in ways that surface one at
    # a time (a missing ``sentencepiece``, then ``TypeError: unsupported
    # operand type(s) for |: 'list' and 'set'`` from a changed internal, then a
    # refused import because the newer release wants a newer ``safetensors``
    # than the archive pins).  Naming ``transformers`` here also keeps it out
    # of the by-name step, so it is never installed twice at two versions.
    "moss_tts": (
        ((MOSS_SOURCE_URL,), None, ("moss-tts",)),
        (("transformers==5.0.0",), None, ("transformers",)),
    ),
    # KittenTTS 2 is installed from PyPI by name, with its own dependency list:
    # ``kittenml`` pins nothing that another engine would need (torch>=2.6,
    # torchaudio>=2.6, transformers>=4.55), and this engine gets an environment
    # of its own, so its resolution reaches nothing else.  PyTorch itself is
    # installed *first* from the CUDA index on a GPU machine (see
    # ``_CUDA_PACKAGES``); ``kittenml`` asks for ``torch>=2.6``, which a local
    # CUDA build (``2.9.0+cu128``) satisfies, so pip leaves it in place.
    "kokoclone": (
        (("--ignore-requires-python", "misaki"), None, ("misaki",)),
        (
            ("--no-deps", KOKOCLONE_SOURCE_URL, KANADE_SOURCE_URL),
            None,
            ("kokoclone", "kanade-tokenizer"),
        ),
    ),
    # Indic Parler-TTS is installed from PyPI, but **not** with its own
    # dependency list: ``parler-tts`` pins ``transformers==4.46.1`` (which the
    # step below installs explicitly) and asks for its audio-tools dependency
    # through a GitHub URL, which would make the install need ``git`` and a
    # source build.  ``--no-deps`` plus the by-name list in ``packages`` is what
    # keeps the install to wheels: the real runtime imports are the model code,
    # Transformers, PyTorch and soundfile, and the description is tokenized with
    # the T5 tokenizer Transformers already ships.
    # ``protobuf`` comes last, and deliberately so.  ``parler-tts`` itself
    # declares ``protobuf>=4.0.0`` (modern protobuf), but ``descript-audiotools``
    # - the audio-tools half of the DAC codec the checkpoint decodes its audio
    # with, installed by name in ``packages`` - still caps it at ``<3.20``, from
    # a release that predates protobuf 4.  On protobuf 3.19.6 the engine does
    # not run at all: ``parler-tts`` pulls in the current ``sentencepiece``,
    # whose generated ``*_pb2.py`` is built for protobuf 4 and imports
    # ``google.protobuf.internal.builder``, a module 3.19 does not have, so the
    # very first import failed with *"cannot import name 'builder'"*.  That cap
    # is stale on the audiotools side: nothing in ``audiotools`` or ``dac``
    # imports protobuf at all (both were checked), so the engine's own
    # requirement is the one to honour.  Installing it as its own last step is
    # what makes pip resolve it there - the earlier steps have already placed
    # the packages and pip does not re-resolve what is installed.
    "indic_parler": (
        (("--no-deps", "parler-tts==0.2.3"), None, ("parler-tts",)),
        (("transformers==4.46.1",), None, ("transformers",)),
        (("protobuf>=4.25.0",), None, ("protobuf",)),
    ),
}

# ---------------------------------------------------------------------------
# Indic Parler-TTS (AI4Bharat)
# ---------------------------------------------------------------------------
#: ``ai4bharat/indic-parler-tts`` is a **description-conditioned** TTS: it takes
#: what to say *and* a prose description of the voice and delivery, and the
#: speaker's name inside that description is what selects the voice.  There is
#: therefore no voice-id argument to pass, which is why the names below - the
#: ones the model card documents - are the engine's voices and
#: :func:`_parler_description` is what is actually sent for each of them.
#:
#: ``(language code, speaker names)``.  The two languages the model card calls
#: *unofficial* (Chhattisgarhi and Punjabi) are deliberately absent: offering a
#: language the checkpoint was not trained on would promise an accent it does
#: not have, and a book recorded with it would have to be recorded again.
_INDIC_PARLER_VOICES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("as", ("Amit", "Sita", "Poonam", "Rakesh")),
    ("bn", ("Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya")),
    ("brx", ("Bikram", "Maya", "Kalpana")),
    ("doi", ("Karan",)),
    ("en", (
        "Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin", "Aakash",
        "Sneha", "Kabir", "Tisha", "Chingkhei", "Thoiba", "Priya", "Tarun",
        "Gauri", "Nisha", "Raghav", "Kavya", "Ravi", "Vikas", "Riya",
    )),
    ("gu", ("Yash", "Neha")),
    ("hi", ("Rohit", "Divya", "Aman", "Rani")),
    ("kn", ("Suresh", "Anu", "Chetan", "Vidya")),
    ("ml", ("Anjali", "Anju", "Harish")),
    ("mni", ("Laishram", "Ranjit")),
    ("mr", ("Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha")),
    ("ne", ("Amrita",)),
    ("or", ("Manas", "Debjani")),
    ("sa", ("Aryan",)),
    ("ta", ("Kavitha", "Jaya")),
    ("te", ("Prakash", "Lalitha", "Kiran")),
)

#: The one variant of the par-ler engine (one checkpoint speaks every language
#: it covers, the language being detected from the transcript).
INDIC_PARLER_VARIANT = "indic_parler_v1"

#: The repository the checkpoint is downloaded from.  It is **gated** (the
#: model card asks for contact details before the files are served), so it is
#: listed in ``hf_access.GATED_REPOS`` and the download needs the user's own
#: Hugging Face token - see ``ai_voice_studio.hf_access``.
INDIC_PARLER_WEIGHTS_REPO = "ai4bharat/indic-parler-tts"

#: The text encoder whose tokenizer the *description* is tokenized with.  The
#: checkpoint names it (``google/flan-t5-large``), and the model card loads it
#: separately from the checkpoint, so it is downloaded on first use.
INDIC_PARLER_DESCRIPTION_TOKENIZER = "google/flan-t5-large"


# ---------------------------------------------------------------------------
# Kitten TTS 2.0 (KittenML)
# ---------------------------------------------------------------------------
#: The one model of this engine.  KittenTTS 2 is a 1.7B speech language model
#: that reads text and writes S3 codec tokens, which its own vocoder turns into
#: 24 kHz audio - one checkpoint speaks every language it covers, so a voice and
#: a language are all a request needs (the language rides on the voice).
KITTEN2_REPO = "KittenML/kitten-tts-2"

#: Its single variant: which *weights* are loaded (the 947 MiB lossless
#: ``packed`` packing or the 469 MiB quantised ``emb4`` one) is a load-time
#: tuning option rather than a second variant, exactly like Pocket TTS's int8
#: quantization - the voices and languages are the same either way.
KITTEN2_VARIANT = "kitten_tts_2_1_7b"

#: The eight voices KittenTTS 0.8 already had, followed by the thirty this
#: model adds: ``(voice id, what its reference clip sounds like)``.  The id *is*
#: the name the engine is asked for by (``model.available_voices`` lists the
#: same strings), and the character text is what the voice list shows, so two
#: voices are never confused for one another.
_KITTEN2_ENGLISH_VOICES: tuple[tuple[str, str], ...] = (
    ("Bella", "KittenTTS 0.8 voice, female"),
    ("Jasper", "KittenTTS 0.8 voice, male"),
    ("Luna", "KittenTTS 0.8 voice, female"),
    ("Bruno", "KittenTTS 0.8 voice, male"),
    ("Rosie", "KittenTTS 0.8 voice, female"),
    ("Hugo", "KittenTTS 0.8 voice, male"),
    ("Kiki", "KittenTTS 0.8 voice, female"),
    ("Leo", "KittenTTS 0.8 voice, male"),
    ("Herbert", "Wise elderly professor, male"),
    ("Diana", "Stern military officer, female"),
    ("Laurence", "Dramatic stage actor, male"),
    ("Maeve", "Cozy warm storyteller, female"),
    ("Walter", "Warm grandfather, male"),
    ("Edith", "Soft elderly grandmother, female"),
    ("Miles", "Gentle librarian, male"),
    ("Grace", "Soothing nurse, female"),
    ("Matthew", "Exhausted mechanic, male"),
    ("Reginald", "Distinguished elderly diplomat, male"),
    ("Elliot", "Grieving young, male"),
    ("Iris", "Hushed museum guide, female"),
    ("Willow", "Hushed young, female"),
    ("Frank", "Gravelly grizzled war veteran, male"),
    ("Dolores", "Southern elderly, female"),
    ("Serena", "Gentle yoga instructor, female"),
    ("Victor", "Controlled fury older, male"),
    ("Julian", "Dreamy poet soft, male"),
    ("Dante", "Playful smooth young, male"),
    ("Eleanor", "Somber historian, female"),
    ("Alfred", "Tender older, male"),
    ("Otis", "Rumbling blues musician, male"),
    ("Saoirse", "Joyful Irish, female"),
    ("Vincent", "Hushed museum guide, male"),
    ("Claire", "Playful professional, female"),
    ("Martha", "Weathered radio journalist, female"),
    ("Raven", "Controlled fury young, female"),
    ("Sable", "Hushed conspirator, female"),
    ("Marcus", "Smooth radio DJ, male"),
    ("Victoria", "Regal narrator, female"),
)

#: The nine voices that are named after a language rather than a person: the
#: reference clip *is* what supplies that language's accent and phonotactics,
#: so it is how a non-English language is reached (``(language, voice id)``).
_KITTEN2_LANGUAGE_VOICES: tuple[tuple[str, str], ...] = (
    ("ar", "Arabic"),
    ("zh", "Chinese"),
    ("fr", "French"),
    ("de", "German"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("es", "Spanish"),
)


def _kitten2_languages() -> list[dict[str, Any]]:
    """KittenTTS 2's ten languages, each offering the one checkpoint.

    English is the thirty-eight-voice set; every other language is reached
    through its own voice (its reference clip), which is why the language list
    is exactly the languages those nine voices cover rather than the twenty the
    project advertises - offering a language no voice can speak would promise an
    accent the checkpoint cannot deliver.
    """
    codes = ("en", *tuple(code for code, _voice in _KITTEN2_LANGUAGE_VOICES))
    return _languages(codes, KITTEN2_VARIANT, "KittenTTS 2 1.7B (24 kHz)")


def _parler_description(name: str) -> str:
    """The description text that makes Parler-TTS speak with one voice.

    Parler-TTS has no concept of a voice id: the description *is* the voice.
    Naming the speaker is what selects them; the rest of the sentence steers the
    delivery, and it is deliberately the plainest one the model was trained on
    (moderate pace, no background noise, very close microphone) because that is
    what a book wants.  The tuning dialog can replace this text entirely, which
    is the model's own "voice design" mode.
    """
    return (
        f"{name} speaks naturally at a moderate pace with a slightly "
        "expressive delivery. The recording is of very high quality, with no "
        "background noise and the speaker very close to the microphone."
    )


#: ``voice`` id used on the Voice Clone panel when the user wants to clone a
#: voice from their own recording instead of a built-in voice.
CLONE_VOICE_ID = "__clone__"

#: The official PyTorch wheel index with CUDA builds.  The plain PyPI
#: ``torch`` wheel on Windows is CPU-only, so a GPU run needs the wheels from
#: this index (installed into the engine's own environment when an NVIDIA GPU
#: with a working driver is detected).
#:
#: ``cu128`` is deliberate and must stay in step with the Python the app runs
#: on: the older ``cu121`` index stops at Python 3.12 and carries no wheels for
#: Python 3.13, which made pip fail with "No matching distribution found for
#: torch" while installing an engine.  The OmniVoice installers use this same
#: constant, so all CUDA wheels come from one index.
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

#: Packages each engine takes from PyTorch's CUDA wheel index when an NVIDIA
#: GPU with a working driver is present.  Every Voice Lab engine runs on the
#: GPU (``devices`` is ``cpu+gpu`` or ``gpu``), so *every* engine gets the CUDA
#: build of PyTorch: without it the plain PyPI wheel is CPU-only on Windows and
#: the "GPU" option would be a label with nothing behind it.  Engines that need
#: torchaudio pull it from the same index, so the pair can never mix builds.
_CUDA_PACKAGES: dict[str, tuple[str, ...]] = {
    "pocket_tts": ("torch",),
    "f5tts": ("torch", "torchaudio"),
    "kokoclone": ("torch", "torchaudio"),
    # Chatterbox pins PyTorch to exactly 2.6.0, and the CUDA build of that
    # release only exists on the CUDA 12.6 index (the newer ``cu128`` index
    # starts at 2.7).  Installing the pinned pair *first* is what makes pip
    # keep the CUDA build: a local version (``2.6.0+cu126``) still satisfies
    # the ``torch==2.6.0`` the engine asks for (PEP 440 ignores the local
    # segment when the specifier has none), so nothing is downgraded to the
    # CPU-only wheel from PyPI.
    "chatterbox": ("torch==2.6.0", "torchaudio==2.6.0"),
    "qwen3tts": ("torch", "torchaudio"),
    # VibeVoice needs PyTorch only (its audio is handled with soundfile and
    # librosa, not torchaudio), and it does not pin a release, so the
    # application-wide CUDA index is the right one for it.
    "vibevoice": ("torch",),
    # MOSS-TTS's 48 kHz codec and GPT-SoVITS's vocoder both go through
    # torchaudio, so the pair comes from the same index and can never mix
    # builds.
    "moss_tts": ("torch", "torchaudio"),
    # KittenTTS 2 declares both PyTorch and TorchAudio, so both come from the
    # same index: a CUDA ``torch`` beside PyPI's CPU-only ``torchaudio`` is a
    # crash, not a warning.
    "kitten_tts_2": ("torch", "torchaudio"),
    "gptsovits": ("torch", "torchaudio"),
    # Indic Parler-TTS decodes its 44.1 kHz waveform with PyTorch (its DAC
    # codec is a torch module), so both wheels come from one index.
    "indic_parler": ("torch", "torchaudio"),
}

#: The CUDA 12.6 wheel index.  Chatterbox pins ``torch==2.6.0``, whose CUDA
#: build is published here and nowhere newer (the ``cu128`` index starts at
#: 2.7), so this is where its PyTorch comes from.
CHATTERBOX_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"

#: Wheel index per engine, for the engines whose pins cannot come from the
#: application-wide ``cu128`` index.  Everything else uses
#: :data:`PYTORCH_CUDA_INDEX`.
_CUDA_INDEX: dict[str, str] = {
    "chatterbox": CHATTERBOX_CUDA_INDEX,
}

#: Packages installed **in addition** on a machine with an NVIDIA GPU.
#:
#: KokoClone's TTS stage is a Kokoro ONNX model, which by default runs on the
#: CPU whatever else does; this is the CUDA build of the ONNX runtime, so the
#: card drives *both* stages.  (``kokoro-onnx`` finds an accelerated ONNX
#: runtime by distribution name and uses its providers, which is exactly what
#: upstream's ``pip install kokoro-onnx[gpu]`` sets up.  When the GPU cannot be
#: used it falls back to the CPU providers by itself.)
_CUDA_EXTRA_PACKAGES: dict[str, tuple[str, ...]] = {
    "kokoclone": ("onnxruntime-gpu",),
}

#: How much video memory one *slot* of a GPU-only engine needs, and how many
#: slots one call may hold.
#:
#: A "slot" is one segment the engine speaks while another segment is being
#: spoken: exactly what the OmniVoice engine does by itself, and what makes the
#: difference between a card that idles at a third of its capacity while a book
#: is recorded and one that is actually used.  Two ways of filling a card are
#: supported, and each engine declares the one its own code can do:
#:
#: ``instances``
#:     The engine takes **one text per call**, so a second segment at the same
#:     time needs a second loaded model - a second worker process with its own
#:     copy of the weights (Chatterbox).  ``vram_mb`` is that copy's peak.
#: ``batch``
#:     The engine's own call takes a **list of texts** and generates them in one
#:     pass, so several segments share one model copy (Qwen3-TTS).  ``vram_mb``
#:     is the loaded model, ``per_text_mb`` what one more text in the batch adds
#:     to it (its share of the attention cache and the batched activations).
#:
#: The figures are the **peaks the driver reported** (``nvidia-smi``, not
#: PyTorch's allocator - the card is what has to hold them), measured on a
#: 12 GB RTX 3060 with roughly fifteen seconds of speech per call.  How many
#: slots a *machine* can afford is decided at recording time from the memory
#: actually free then (``voicelab.parallel_plan``); these numbers only say what
#: one slot costs and what the engine can do.  Larger models of the same
#: family cost more; an unknown variant is charged the dearest one declared, so
#: a newer model can never over-fill a card.
#:
#: ``chars_per_slot`` is how much text a batched slot was measured with (a
#: sentence of a couple of hundred characters).  A batch of whole paragraphs
#: would ask for several times the memory of a batch of sentences, so the
#: application keeps a batch within ``chars_per_slot`` characters per slot
#: (``voicelab.batch_chars``): a project of ordinary segments still fills the
#: card, and a project of very long ones is spoken in smaller batches - each
#: segment still generated as one piece of its own, so the audio is unchanged.
_GPU_PARALLEL: dict[str, dict[str, Any]] = {
    "chatterbox": {
        "mode": "instances",
        "max_slots": 3,
        "slots": {
            CHATTERBOX_ENGLISH_VARIANT: {"vram_mb": 4700},
            CHATTERBOX_TURBO_VARIANT: {"vram_mb": 3600},
            CHATTERBOX_MULTILINGUAL_VARIANT: {"vram_mb": 4800},
        },
    },
    # VibeVoice speaks one text per call and generates it autoregressively, so
    # a second segment at the same time needs a second loaded model.  The
    # figures are the peaks each size needs on the card: the 1.5B model leaves
    # room for a second copy on a 24 GB card, the 7B one does not on anything
    # smaller than a 48 GB card.
    "vibevoice": {
        "mode": "instances",
        "max_slots": 2,
        "slots": {
            "vibevoice_1_5b": {"vram_mb": 8000},
            "vibevoice_7b": {"vram_mb": 20000},
        },
    },
    # MOSS-TTS speaks one text per call and generates it autoregressively, so a
    # second segment at the same time needs a second loaded model.  The figures
    # are the peaks each size needs on the card: the 4B model fills a 12 GB card
    # on its own, the 1.7B one leaves room for a second copy.
    "moss_tts": {
        "mode": "instances",
        "max_slots": 2,
        "slots": {
            "moss_tts_local_v15_4b": {"vram_mb": 9500},
            "moss_tts_local_1_7b": {"vram_mb": 6500},
        },
    },
    # GPT-SoVITS is a small two-stage model (a GPT text-to-token stage and a
    # SoVITS vocoder), one text per call, so a second segment needs a second
    # loaded copy.  A single copy is well under 4 GB, so an ordinary card can
    # hold several.
    "gptsovits": {
        "mode": "instances",
        "max_slots": 3,
        "slots": {
            "gptsovits_v2proplus": {"vram_mb": 2600},
            "gptsovits_v2pro": {"vram_mb": 2400},
            "gptsovits_v4": {"vram_mb": 3200},
            "gptsovits_v3": {"vram_mb": 3000},
            "gptsovits_v2": {"vram_mb": 1600},
            "gptsovits_v1": {"vram_mb": 1400},
        },
    },
    # Kitten TTS 2.0's own ``generate`` takes **one** text (a longer one is cut
    # into chunks internally and decoded chunk by chunk), so a second segment at
    # the same time needs a second loaded copy of the 1.7B model.  The package's
    # only batched path is its optional vLLM backend, which needs Linux.
    #
    # It is the dearest slot declared here, and the figure is measured, not
    # guessed (``tools/measure_engine_vram.py kitten_tts_2``, run under the
    # application's own interpreter, 65-character segments after a warm-up):
    # the checkpoint is ternary (0.95 GB on disk) but unpacking materialises the
    # bf16 weights (~3.5 GB), and the codec, the embeddings and the process's own
    # CUDA context come on top.  One copy, loading and generating, peaked at
    # 6115 MiB over an idle 1207 MiB - 4908 MiB; two copies resident and both
    # generating at once peaked at 10645 MiB over an idle 1216 MiB - 4714 MiB a
    # copy.  The dearest of the two is declared, rounded up, so a slot is never
    # charged less than the card was seen to use.  What that buys, with the
    # memory free and the reserve a recording has: a 12 GB card keeps one copy
    # (two fit the card itself but not the budget left after the 1.5 GB reserve),
    # a 16 GB card keeps two, and a card of 24 GB or more reaches the ceiling of
    # three that the other one-text-per-call engines declare.  The reference
    # 12 GB card could only demonstrate two: a third copy asks for about 15 GB,
    # and instead of failing it saturates the card and starts spilling to host
    # memory, which is why the ceiling is not read off this machine alone.
    # The gain is measured with the same two segments: 45.8 s of wall time
    # through two copies against 87.8 s through one (1.9x), and this engine runs
    # at roughly a fifth of real time, so a book is where the second slot pays.
    "kitten_tts_2": {
        "mode": "instances",
        "max_slots": 3,
        "slots": {
            KITTEN2_VARIANT: {"vram_mb": 5000},
        },
    },
    "qwen3tts": {
        "mode": "batch",
        "max_slots": 6,
        "slots": {
            "qwen3_custom_1_7b": {"vram_mb": 6100, "per_text_mb": 900,
                                 "chars_per_slot": 150},
            "qwen3_custom_0_6b": {"vram_mb": 3300, "per_text_mb": 500,
                                 "chars_per_slot": 150},
            "qwen3_clone_1_7b": {"vram_mb": 6300, "per_text_mb": 900,
                                 "chars_per_slot": 150},
            "qwen3_clone_0_6b": {"vram_mb": 3400, "per_text_mb": 500,
                                 "chars_per_slot": 150},
        },
    },
}

#: Modules an engine needs before it can speak - the worker answers "is this
#: engine available?" from the same list.  Kept here (not only in the worker)
#: so the application can *verify* an installation by importing them in the
#: engine's own environment; ``tests/test_voicelab.py`` fails when the two
#: copies drift apart.  The same list decides which environment an engine's
#: worker is started with (see ``voicelab.CloneWorker._runtime``), so a wrong
#: entry here is what makes an engine fail with a missing-module error.
_IMPORT_MODULES: dict[str, tuple[str, ...]] = {
    "pocket_tts": ("pocket_tts",),
    "f5tts": ("f5_tts",),
    # ``core`` is KokoClone's own package (built from its source archive) and
    # ``kanade_tokenizer`` is the voice-conversion model it clones with.
    "kokoclone": ("core", "kanade_tokenizer"),
    "chatterbox": ("chatterbox", "torch", "torchaudio"),
    "qwen3tts": ("qwen_tts", "torch", "torchaudio"),
    "vibevoice": ("vibevoice", "torch", "transformers"),
    # MOSS-TTS is a Hugging Face transformer loaded with its repository's own
    # remote code, so transformers (and torch) are what prove it can speak.
    "moss_tts": ("transformers", "torch"),
    # GPT-SoVITS is a source tree, not an installed package: its worker adds the
    # fetched tree to ``sys.path`` itself, so the modules that prove it can
    # speak are its own runtime dependencies.
    # ``pytorch_lightning`` is on the import path the worker actually walks
    # (``GPT_SoVITS/AR/models/t2s_lightning_module.py``), and ``ffmpeg`` is the
    # module whose absence broke the first preview; naming them here is what
    # makes the post-install check catch this class of failure instead of
    # leaving it for the first synthesis.
    "gptsovits": (
        "torch", "torchaudio", "librosa", "ffmpeg", "pytorch_lightning",
    ),
    # The model code, the description tokenizer and PyTorch are what prove
    # Indic Parler-TTS can speak.
    "indic_parler": ("parler_tts", "transformers", "torch", "torchaudio"),
    # ``kittenml`` is the engine's own package (it carries the model, the
    # codec and the packaging) and PyTorch is what it runs on, so importing
    # the pair is what proves this environment can speak.
    "kitten_tts_2": ("kittenml", "torch"),
}

_LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
    # The Indic languages (Indic Parler-TTS).  The catalog names are
    # spelled out here rather than taken from a library, because the language a
    # voice is filed under is what the Punctuation and Recording settings
    # categories and the per-language speaking-rate defaults follow.
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    # ``hi`` (Hindi) is already listed above with the other Latin/Greek-script
    # languages; repeating it here would be a second entry for one key.
    "kn": "Kannada",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
}


def _languages(codes: Iterable[str], variant: str, variant_name: str) -> list[dict[str, Any]]:
    """Catalog-shaped language list (one variant, voices added separately)."""
    return [
        {
            "code": code,
            "name": _LANGUAGE_NAMES.get(code, code),
            "variants": [{"id": variant, "name": variant_name, "voices": []}],
        }
        for code in codes
    ]


def _qwen_languages() -> list[dict[str, Any]]:
    """Qwen3-TTS's languages, each offering all four of its models.

    Every language is one catalog entry holding the four variants, so the
    language boxes in the app list it once however many models speak it.
    """
    return [
        {
            "code": code,
            "name": _LANGUAGE_NAMES.get(code, code),
            "variants": [
                {"id": variant, "name": name, "voices": []}
                for variant, _repo, _kind, name in _QWEN_MODELS
            ],
        }
        for code in _QWEN_LANGUAGES
    ]


def _moss_languages() -> list[dict[str, Any]]:
    """MOSS-TTS's languages, each offering both variants that fit a card.

    Both Local-Transformer checkpoints speak all 31 languages.  A language is
    listed once and holds every variant that speaks it, so the language boxes
    stay short.
    """
    languages: list[dict[str, Any]] = []
    for code, name in _MOSS_LANGUAGES:
        variants = [
            {"id": variant, "name": label, "voices": []}
            for variant, _repo, label, _vram, _devices in _MOSS_MODELS
        ]
        if variants:
            languages.append({"code": code, "name": name, "variants": variants})
    return languages


def _gptsovits_languages() -> list[dict[str, Any]]:
    """GPT-SoVITS's languages, each offering every release.

    All six releases speak the same five languages (v1 in its own narrower
    form, which the worker maps onto the same codes), so each language lists
    them all and the voice list follows the chosen variant.
    """
    return [
        {
            "code": code,
            "name": name,
            "variants": [
                {"id": variant, "name": label, "voices": []}
                for variant, _version, label, _vram in _GPTSOVITS_MODELS
            ],
        }
        for code, name in _GPTSOVITS_LANGUAGES
    ]


def _vibevoice_languages() -> list[dict[str, Any]]:
    """VibeVoice's languages, each offering both model sizes.

    English and Chinese are the languages the bundled voice presets cover,
    and every preset can speak with either model, so both variants are listed
    for both languages and the voice list follows the chosen variant (the same
    shape Qwen3-TTS's languages have).  The model itself can transfer to other
    languages unevenly; offering them as selectable would promise more than it
    delivers.
    """
    return [
        {
            "code": code,
            "name": _LANGUAGE_NAMES.get(code, code),
            "variants": [
                {"id": variant, "name": label, "voices": []}
                for variant, _repo, label in _VIBEVOICE_MODELS
            ],
        }
        for code in ("en", "zh")
    ]


ENGINES: tuple[dict[str, Any], ...] = (
    {
        "id": "pocket_tts",
        "name": "Pocket TTS (Kyutai)",
        "engine": "pocket_tts",
        "package": "pocket-tts",
        "packages": ("pocket-tts", "soundfile", "scipy"),
        "license": "Apache-2.0 (code); per-voice licenses",
        "homepage": "https://github.com/kyutai-labs/pocket-tts",
        "size_hint": "~450 MB (model) + PyTorch",
        "devices": "cpu+gpu",
        "voice_cloning": True,
        # Kyutai's *voice-cloning* weights are gated (the cloning-free ones are
        # not), so cloning needs a Hugging Face token like Gemma 3 does.  See
        # ``ai_voice_studio.hf_access``.
        "gated_repo": "kyutai/pocket-tts",
        "note": (
            "A 100M-parameter voice-cloning TTS that runs faster than real "
            "time on a normal CPU. 26 pre-made voices in 6 languages; clone "
            "any voice from a 5-10 second sample. Cloning itself needs the "
            "gated kyutai/pocket-tts weights: accept their licence and add a "
            "Hugging Face token in Settings > Hugging Face account. Until "
            "then the 26 pre-made voices work and cloning does not."
        ),
        "languages": _languages(
            ("en", "fr", "de", "es", "it", "pt"),
            "pocket_v1", "Pocket TTS 100M",
        ),
    },
    {
        "id": "f5tts",
        "name": "F5-TTS (transformer)",
        "engine": "f5tts",
        "package": "f5-tts",
        "packages": ("f5-tts", "torch", "torchaudio", "soundfile"),
        "license": "MIT (code); CC-BY-NC-4.0 (weights)",
        "homepage": "https://github.com/SWivid/F5-TTS",
        "size_hint": "~1.4 GB",
        "devices": "cpu+gpu",
        "voice_cloning": True,
        "note": (
            "Flow-matching transformer TTS with zero-shot voice cloning from "
            "a 5-15 second reference clip plus its transcript. English and "
            "Chinese, with a bundled reference voice for each."
        ),
        "languages": _languages(("en", "zh"), "f5_v1_base", "F5-TTS v1 Base"),
    },
    {
        "id": "kokoclone",
        "name": "KokoClone (Kokoro + Kanade)",
        "engine": "kokoclone",
        "package": "kokoclone",
        # ``kokoclone`` and ``kanade-tokenizer`` are not on PyPI: they are
        # installed from the pinned source archives declared above (see
        # ``_EXTRA_REQUIREMENTS``).  They are listed here because this is also
        # the list "Remove engine" uninstalls by name.
        "packages": (
            "kokoclone", "kanade-tokenizer",
            "torch", "torchaudio",
            "onnxruntime", "onnxruntime-gpu",
            "kokoro-onnx", "misaki",
            "soundfile", "huggingface-hub", "numpy",
            "safetensors", "tqdm", "vocos", "jsonargparse",
            "cn2an", "jieba", "ordered-set", "pypinyin", "pypinyin-dict",
        ),
        "license": "Apache-2.0",
        "homepage": "https://github.com/Ashish-Patnaik/kokoclone",
        "size_hint": "~1.5 GB (Kokoro + Kanade models) + PyTorch",
        "devices": "cpu+gpu",
        "voice_cloning": True,
        # KokoClone needs a newer PyTorch and a newer ONNX runtime than the
        # other engines, and it must never pick up the packages another engine
        # installed: it runs in its own environment, and only there.  (The
        # other engines may fall back to the old *shared* addon environment,
        # where engines installed before the per-engine environments existed
        # live - for KokoClone that environment is somebody else's.)
        "own_environment": True,
        "note": (
            "Kokoro's fast multilingual ONNX speech with Kanade zero-shot "
            "voice conversion on top: give it 3-10 seconds of a voice and it "
            "speaks in that voice. English, Hindi, French, Spanish, Italian, "
            "Portuguese and Chinese. Its built-in voices are the Kokoro "
            "voices it starts from before a clone is applied."
        ),
        "languages": _languages(
            ("en", "hi", "fr", "es", "it", "pt", "zh"),
            "kokoclone_v1", "Kokoro 82M (ONNX) + Kanade 12.5 Hz",
        ),
    },
    {
        "id": "chatterbox",
        "name": "Chatterbox (Resemble AI)",
        "engine": "chatterbox",
        "package": "chatterbox-tts",
        # ``chatterbox-tts`` pulls its whole dependency list from PyPI (the
        # pinned ``transformers``/``torch`` versions included); only PyTorch
        # itself is redirected to the CUDA index on a GPU machine (see
        # ``_CUDA_PACKAGES``).
        "packages": (
            "chatterbox-tts", "torch", "torchaudio", "soundfile",
        ),
        "license": "MIT",
        "homepage": "https://github.com/resemble-ai/chatterbox",
        "size_hint": (
            "~2.5 GB (three models: 500M x2 + 350M, plus its dependencies) "
            "+ PyTorch"
        ),
        # GPU only: all three models are half-gigabyte transformers whose CPU
        # run is many times slower than real time, so the CPU is not offered as
        # a usable device for it (see ``gpu_only``).
        "devices": "gpu",
        "voice_cloning": True,
        # Chatterbox pins ``torch==2.6.0``/``torchaudio==2.6.0``; a second
        # engine in the same environment with a different torch would break
        # one of them.  It gets an environment of its own, like KokoClone.
        "own_environment": True,
        "note": (
            "Resemble AI's zero-shot voice cloning family. Chatterbox (500M) "
            "and Chatterbox Turbo (350M, with [laugh]/[chuckle] tags) speak "
            "English; Chatterbox Multilingual (500M) speaks 23 languages. "
            "Clone a voice from a 5-15 second sample, or use the model's own "
            "default voice. Needs an NVIDIA GPU."
        ),
        # English gets all three models; every other language only the
        # multilingual one.  The whole engine clones from a reference clip, so
        # the variant a cloned voice is stored with is the multilingual model
        # (it speaks every language, the English ones only speak English).
        "clone_variant": CHATTERBOX_MULTILINGUAL_VARIANT,
        "languages": [
            {
                "code": "en",
                "name": "English",
                "variants": [
                    {
                        "id": CHATTERBOX_ENGLISH_VARIANT,
                        "name": "Chatterbox 500M (English, best quality)",
                        "voices": [],
                    },
                    {
                        "id": CHATTERBOX_TURBO_VARIANT,
                        "name": "Chatterbox Turbo 350M (English, fastest)",
                        "voices": [],
                    },
                    {
                        "id": CHATTERBOX_MULTILINGUAL_VARIANT,
                        "name": "Chatterbox Multilingual 500M (23 languages)",
                        "voices": [],
                    },
                ],
            },
            *_languages(
                (code for code, _label in _CHATTERBOX_LANGUAGES if code != "en"),
                CHATTERBOX_MULTILINGUAL_VARIANT,
                "Chatterbox Multilingual 500M (23 languages)",
            ),
        ],
    },
    {
        "id": "qwen3tts",
        "name": "Qwen3-TTS (Alibaba Qwen)",
        "engine": "qwen3tts",
        "package": "qwen-tts",
        "packages": (
            "qwen-tts", "torch", "torchaudio", "soundfile",
        ),
        "license": "Apache-2.0",
        "homepage": "https://github.com/QwenLM/Qwen3-TTS",
        "size_hint": (
            "~2-9 GB (0.6B or 1.7B model, downloaded on first use) "
            "+ PyTorch"
        ),
        # GPU only: the engine is a 0.6B-1.7B language model run in bfloat16,
        # generation is far slower than real time on a CPU, and the package is
        # built for CUDA (it recommends FlashAttention 2).
        "devices": "gpu",
        "voice_cloning": True,
        # ``qwen-tts`` pins ``transformers==4.57.3``; its own environment keeps
        # that pin away from every other engine.
        "own_environment": True,
        "note": (
            "Qwen3-TTS speaks and clones with a 0.6B or 1.7B Qwen3 model "
            "over its 12 Hz speech tokenizer, in ten languages. Its "
            "CustomVoice models ship nine named voices; its Base models "
            "clone a voice from a 3-10 second sample and its transcript. "
            "Needs an NVIDIA GPU."
        ),
        # A cloned voice must be stored with a *Base* variant: the CustomVoice
        # models speak one of their nine voices and cannot clone.
        "clone_variant": "qwen3_clone_1_7b",
        "languages": _qwen_languages(),
    },
    {
        "id": "vibevoice",
        "name": "VibeVoice (Microsoft, community fork)",
        "engine": "vibevoice",
        "package": "vibevoice",
        # ``vibevoice`` is not on PyPI: it is installed from the pinned source
        # archive declared above (see ``_EXTRA_REQUIREMENTS``), which also
        # brings transformers/accelerate/Gradio.  ``torch`` is installed first
        # from the CUDA index on a GPU machine; ``soundfile`` reads the voice
        # presets and the reference recordings.
        "packages": ("vibevoice", "torch", "soundfile"),
        "license": "MIT",
        "homepage": "https://github.com/vibevoice-community/VibeVoice",
        "size_hint": (
            "~4 GB (1.5B) or ~18 GB (7B) model on first use, plus PyTorch "
            "and the engine's own dependencies"
        ),
        # GPU only: the model is an autoregressive language model with a
        # diffusion head, and its CPU run is several times slower than real
        # time even at 1.5B, so the CPU is not offered as a usable device
        # (see ``gpu_only``).
        "devices": "gpu",
        "voice_cloning": True,
        # The engine pins ``transformers==4.51.3`` (and ``accelerate==1.6.0``):
        # another engine's pinned transformers in a shared environment would
        # break one of the two, so this one may only run in its own.
        "own_environment": True,
        "note": (
            "Microsoft's long-form conversational TTS: a language model plans "
            "the speech and a diffusion head renders it, so one run can speak "
            "up to 90 minutes (1.5B) or 45 minutes (7B) with up to four "
            "speakers. Seven bundled voice presets in English and Chinese, "
            "and a voice of your own from a 5-15 second recording. Needs an "
            "NVIDIA GPU."
        ),
        # A cloned voice lands on the lighter model: both sizes support voice
        # cloning, and the 1.5B one is the one that fits an ordinary card.
        "clone_variant": "vibevoice_1_5b",
        "languages": _vibevoice_languages(),
    },
    {
        "id": "moss_tts",
        "name": "MOSS-TTS (OpenMOSS / MOSI.AI)",
        "engine": "moss_tts",
        "package": "moss-tts",
        # ``moss-tts`` is not on PyPI: it is built from the pinned source
        # archive declared above (see ``_EXTRA_REQUIREMENTS``).  Its own
        # dependency list brings transformers, librosa, soundfile and the
        # rest; ``torch``/``torchaudio`` come from the CUDA index on a GPU
        # machine and from PyPI otherwise.
        # ``sentencepiece`` is what the checkpoints' own tokenizer code asks
        # for (``transformers`` only reports it as an optional import), and
        # ``transformers`` is installed at the pinned release by
        # ``_EXTRA_REQUIREMENTS`` rather than from this list, which is why the
        # two never land at two different versions.
        "packages": (
            "moss-tts", "torch", "torchaudio", "transformers",
            "soundfile", "librosa", "numpy", "safetensors", "accelerate",
            "sentencepiece",
        ),
        "license": "Apache-2.0",
        "homepage": "https://github.com/OpenMOSS/MOSS-TTS",
        "size_hint": (
            "~6.5 GB (1.7B) or ~9.5 GB (4B) model on first use, plus PyTorch"
        ),
        # The two Local-Transformer checkpoints are GPU-only: their CPU run is
        # many times slower than real time.
        "devices": "gpu",
        "voice_cloning": True,
        # MOSS-TTS pins ``transformers==5.0.0`` in its ``torch-runtime`` extra:
        # another engine's pinned transformers in a shared environment would
        # break one of the two, so this one may only run in its own.
        "own_environment": True,
        "note": (
            "OpenMOSS's production TTS family. The 48 kHz Local Transformer "
            "checkpoints (4B latest, 1.7B) clone a voice from a short sample "
            "and read long text stably in 31 languages. Both fit inside "
            "12 GB of video memory. Needs an NVIDIA GPU."
        ),
        # Every variant clones from a reference clip, and they all speak the
        # family's 31 languages, so a clone lands on the latest 4B model.
        "clone_variant": "moss_tts_local_v15_4b",
        "languages": _moss_languages(),
    },
    {
        "id": "gptsovits",
        "name": "GPT-SoVITS (RVC-Boss)",
        "engine": "gptsovits",
        # GPT-SoVITS has no distribution of its own to probe for, so the probe
        # is a distinctive package its dependency set brings and no other
        # engine needs: ``rotary_embedding_torch``.  Its presence proves the
        # engine's packages are installed; the worker fetches the source tree
        # itself on first use and reports plainly if that fails.
        "package": "rotary_embedding_torch",
        # GPT-SoVITS has no packaging metadata, so nothing is installed from
        # its archive by pip: the worker fetches the pinned tree into the
        # engine's own folder and downloads the pretrained weights beside it.
        # The packages listed here are the ones its runtime actually needs, and
        # they are also what "Remove engine" uninstalls.
        #
        # The list follows upstream's ``requirements.txt``: pip installing the
        # tree is not an option (it has no ``pyproject.toml``), so its declared
        # dependencies are installed by name instead.  ``ffmpeg-python`` is the
        # one whose absence broke the very first preview (``import ffmpeg`` at
        # the top of ``TTS_infer_pack/TTS.py``), and ``pytorch-lightning``,
        # ``peft`` and ``matplotlib`` are imported on that same path - a lean
        # list that only looked plausible was not enough.  ``pyopenjtalk`` is
        # deliberately absent: upstream ships no wheel for Python 3.13 and it
        # cannot be built here, so the worker substitutes it (see
        # ``worker._ensure_gptsovits_japanese``) and Japanese is simply not
        # offered by this integration.  ``jieba_fast`` is absent for the same
        # reason: it is a C extension with no wheel for Python 3.13, so pip
        # fails the whole install trying to build it ("Microsoft Visual C++
        # 14.0 or greater is required").  The worker stands in plain ``jieba``
        # under the ``jieba_fast`` name (see
        # ``worker._ensure_gptsovits_jieba_fast``).
        "packages": (
            "torch", "torchaudio", "transformers", "librosa",
            "soundfile", "numpy", "scipy", "numba", "tqdm",
            "PyYAML", "psutil", "sentencepiece", "cn2an", "pypinyin",
            "jieba", "wordsegment", "split-lang",
            "fast-langdetect", "opencc", "g2p_en", "ToJyutping",
            "g2pk2", "ko_pron", "rotary_embedding_torch", "x_transformers",
            "onnxruntime", "requests", "huggingface_hub", "pytorch-lightning",
            "peft", "tensorboard", "matplotlib", "ffmpeg-python",
            "chardet", "funasr", "modelscope", "av", "ctranslate2",
            "fastapi",
        ),
        "license": "MIT",
        "homepage": "https://github.com/RVC-Boss/GPT-SoVITS",
        "size_hint": (
            "~1.5-2 GB of pretrained weights (downloaded on first use) plus "
            "PyTorch"
        ),
        # GPU only: the GPT stage is a small autoregressive transformer whose
        # CPU run is several times slower than real time, and the project is
        # built for CUDA.  It is small enough for any NVIDIA card.
        "devices": "gpu",
        "voice_cloning": True,
        # It runs from a source tree that its worker fetches itself, so it can
        # never share another engine's packages.
        "own_environment": True,
        "note": (
            "Few-shot voice cloning from as little as five seconds of audio. "
            "Six releases (v2ProPlus latest, then v2Pro, v4, v3, v2 and v1) "
            "speak English, Chinese, Japanese, Korean and Cantonese, all "
            "inside 12 GB of video memory. It has no voices of its own: clone "
            "a voice from a recording to speak with it. Needs an NVIDIA GPU."
        ),
        # A clone always lands on the latest release.
        "clone_variant": "gptsovits_v2proplus",
        "languages": _gptsovits_languages(),
    },
    {
        "id": "indic_parler",
        "name": "Indic Parler-TTS (AI4Bharat)",
        "engine": "indic_parler",
        "package": "parler-tts",
        # ``parler-tts`` itself is installed from a pinned wheel with
        # ``--no-deps`` and the rest come from PyPI; ``transformers`` is listed
        # because it is installed at the release ``parler-tts`` requires (see
        # ``_EXTRA_REQUIREMENTS``), which is also what "Remove engine"
        # uninstalls by name.
        "packages": (
            "parler-tts", "transformers", "tokenizers", "sentencepiece",
            "torch", "torchaudio", "numpy", "soundfile", "librosa",
            "accelerate", "huggingface-hub", "safetensors", "tqdm",
            "protobuf", "descript-audio-codec", "descript-audiotools",
            "argbind", "einops", "julius",
        ),
        "license": "Apache-2.0",
        "homepage": "https://github.com/huggingface/parler-tts",
        "size_hint": (
            "~3.8 GB model (downloaded on first use) plus the T5 description "
            "tokenizer and PyTorch"
        ),
        # GPU only: this is a 0.9B autoregressive transformer that renders its
        # CD-quality audio through a neural codec, and its CPU run is many times
        # slower than real time.  (The Settings panel offers the CPU for such an
        # engine once "Measure speed" has shown it is fast enough here.)
        "devices": "gpu",
        # Not a cloning engine: it speaks with the 69 voices its card documents,
        # selected - and steered - by the description text.
        "voice_cloning": False,
        # ``parler-tts`` pins ``transformers==4.46.1``; another engine's
        # Transformers in a shared environment would break one of the two.
        "own_environment": True,
        # Gated: the model card asks for contact details before the files are
        # served, so the download needs a token (Settings > Hugging Face
        # account) and the licence accepted on the model's page.
        "gated_repo": INDIC_PARLER_WEIGHTS_REPO,
        "note": (
            "AI4Bharat's Indic fine-tune of Parler-TTS: 20 Indian languages "
            "and English spoken by the 65 named voices its model card lists. "
            "There is no voice id - you pick the speaker's name, and the "
            "description of the voice and delivery that goes with it can be "
            "rewritten in Engine tuning, which is the model's own voice-design "
            "mode. Its repository is gated, so downloading it needs a Hugging "
            "Face token (Settings > Hugging Face account) with the licence "
            "accepted on the model's page. Needs an NVIDIA GPU."
        ),
        "languages": _languages(
            (code for code, _names in _INDIC_PARLER_VOICES),
            INDIC_PARLER_VARIANT,
            "Indic Parler-TTS 0.9B (44.1 kHz)",
        ),
    },
    {
        "id": "kitten_tts_2",
        "name": "Kitten TTS 2.0 (KittenML)",
        "engine": "kitten_tts_2",
        "package": "kittenml",
        # ``kittenml`` is on PyPI with a working dependency list, so it is
        # installed by name and pip resolves its own requirements (torch,
        # torchaudio, transformers, soundfile, the espeak-ng that its phonemizer
        # bundles...).  The names are listed here as well because this is also
        # the list "Remove engine" works from.
        "packages": (
            "kittenml", "torch", "torchaudio", "transformers", "safetensors",
            "numpy", "scipy", "soundfile", "librosa", "huggingface-hub",
            "brotli", "einops", "diffusers", "omegaconf", "hydra-core",
            "s3tokenizer", "resemble-perth", "pyloudnorm",
            "kitten-text-processing", "espeakng-loader", "phonemizer",
            "onnxruntime",
        ),
        "license": (
            "Apache-2.0 (code); Stellon Labs Community License (weights)"
        ),
        "homepage": "https://github.com/KittenML/KittenTTS",
        "size_hint": (
            "~950 MB of 1.7B weights (or ~470 MB with the emb4 packing), "
            "downloaded on first use, plus PyTorch; about 6 GB of RAM for a "
            "CPU run"
        ),
        # Both devices are real for this engine.  Its own documentation says
        # the model "runs on CPU out of the box - ``device`` is auto-detected"
        # and asks for about 6 GB of RAM, so the CPU is a mode it has, not a
        # fallback this application invented: it is declared here and the
        # worker passes ``device="cpu"`` to the package.  The package's
        # PyTorch path is the one its documentation calls demo-only, so a CPU
        # run is slower than real time; the application's measured-speed rule
        # (``voicelab.cpu_allowed``) still keeps the CPU out of the device list
        # when a benchmark on *this* computer says it is slower than
        # ``CPU_RTF_LIMIT`` and an NVIDIA card is there to use instead.
        "devices": "cpu+gpu",
        "voice_cloning": True,
        # This engine reads expression markup in the text it is given: a leading
        # ``[emotion]`` tag, inline ``<event>`` tags and ``(((emphasis)))`` spans
        # are director's notes the model conditions on, not words to pronounce
        # (``kittenml.kittentts2.text``).  The application's own text pipeline
        # hands them over untouched for it - see ``tts.markup``.
        "expression_markup": True,
        # ``kittenml`` asks for ``transformers>=4.55``; another engine's pinned
        # Transformers in a shared environment would break one of the two, so
        # this engine may only ever run in its own environment.
        "own_environment": True,
        # Every weight packing and decoder is the same model at a different
        # size, and both speak all ten languages, so a clone lands on the one
        # variant there is.
        "clone_variant": KITTEN2_VARIANT,
        "note": (
            "KittenML's 1.7B speech language model, the successor to the small "
            "ONNX voices the catalogue already ships. 47 built-in voices - the "
            "eight from KittenTTS 0.8, thirty more English ones and nine named "
            "after a language - plus voice cloning from a 5-30 second sample, "
            "and optional [emotion] tags, <event> tags and (((emphasis))). Its "
            "weights download from Hugging Face on first use and need no "
            "account. Runs on an NVIDIA GPU or on the CPU (about 6 GB of "
            "RAM); Measure speed shows what this computer manages on each."
        ),
        "languages": _kitten2_languages(),
    },
)

_ENGINES_BY_ID: dict[str, dict[str, Any]] = {e["id"]: e for e in ENGINES}


def engine_ids() -> tuple[str, ...]:
    """Ids of every Voice Lab engine."""
    return tuple(e["id"] for e in ENGINES)


def is_engine(engine_id: str | None) -> bool:
    """True when ``engine_id`` is one of the Voice Lab engines."""
    return bool(engine_id) and engine_id in _ENGINES_BY_ID


def engine(engine_id: str) -> dict[str, Any] | None:
    return _ENGINES_BY_ID.get(engine_id)


def packages(engine_id: str) -> tuple[str, ...]:
    """pip packages this engine needs in the managed virtualenv."""
    info = _ENGINES_BY_ID.get(engine_id)
    if not info:
        return ()
    return tuple(info.get("packages") or (info.get("package"),))


def own_environment(engine_id: str) -> bool:
    """True when this engine must run in **its own** environment only.

    Every engine is installed into its own virtualenv, but the older engines
    may still fall back to the shared *addon* environment (the one OmniVoice
    and addons use) when they were installed before the per-engine
    environments existed.  An engine marked here never does: its packages live
    in ``tts_envs/<engine>`` and nowhere else, so it can neither borrow
    another engine's libraries nor have its own taken away by one (see
    ``voicelab.is_installed`` and ``voicelab.CloneWorker._runtime``).
    """
    info = _ENGINES_BY_ID.get(engine_id)
    return bool(info and info.get("own_environment"))


def expression_markup(engine_id: str) -> bool:
    """True when the engine reads expression markup from the text it is given.

    Kitten TTS 2.0 is the one: a leading ``[emotion]`` tag, inline ``<event>``
    tags and ``(((emphasis)))`` spans condition its delivery, so the text
    pipeline must not expand or strip their punctuation (``tts.markup``).
    """
    info = _ENGINES_BY_ID.get(engine_id)
    return bool(info and info.get("expression_markup"))


def gpu_supported(engine_id: str) -> bool:
    """True when this engine can run on an NVIDIA GPU.

    Most Voice Lab engines are CPU-first and additionally run on the GPU
    (``devices`` is ``cpu+gpu``); the large transformer engines cannot run fast
    enough on a CPU to be useful and are GPU-only (``devices`` is ``gpu``).
    """
    info = _ENGINES_BY_ID.get(engine_id)
    return bool(info and "gpu" in str(info.get("devices", "")))


def cpu_supported(engine_id: str) -> bool:
    """True when this engine is usable on a CPU."""
    info = _ENGINES_BY_ID.get(engine_id)
    return bool(info and "cpu" in str(info.get("devices", "")))


def gpu_only(engine_id: str) -> bool:
    """True when this engine needs an NVIDIA GPU and has no CPU mode.

    Offered as one device, not two: a CPU run of a 0.5-1.7B transformer is
    many times slower than real time, so presenting it as an option would only
    promise something the engine cannot deliver.
    """
    return gpu_supported(engine_id) and not cpu_supported(engine_id)


def gated_repo(engine_id: str) -> str:
    """The **gated** Hugging Face repository this engine's weights live in ("").

    Every other engine here downloads anonymously.  Kyutai's Pocket TTS does
    not: its voice-cloning weights are in ``kyutai/pocket-tts``, which is
    gated, so a clone needs the licence accepted on that page and a token
    (Settings > Hugging Face account).  Without them the engine does not fail:
    it silently loads its cloning-free weights and only the pre-made voices can
    be spoken - which is why the application has to name the cause when a clone
    is attempted.
    """
    info = _ENGINES_BY_ID.get(engine_id) or {}
    return str(info.get("gated_repo") or "")


def gpu_parallel(engine_id: str, variant: str = "") -> dict[str, Any] | None:
    """How this engine/variant fills a GPU, or ``None`` when it can not.

    ``{"mode": "instances"|"batch", "max_slots": int, "vram_mb": int,
    "per_text_mb": int}``.  An engine that declares nothing is spoken one
    segment at a time, exactly as before: filling a card is something an engine
    has to be able to do, not something the application can do for it.
    """
    info = _GPU_PARALLEL.get(engine_id)
    if not info:
        return None
    slots = info.get("slots") or {}
    spec: dict[str, Any] = dict(slots.get(variant) or {})
    if not spec and slots:
        # A variant with no figures of its own (a voice created for a model
        # this build does not know) is charged the dearest one declared.
        spec = dict(
            max(slots.values(), key=lambda item: int(item.get("vram_mb") or 0))
        )
    if not spec:
        return None
    return {
        "mode": str(info.get("mode") or "instances"),
        "max_slots": max(1, int(info.get("max_slots") or 1)),
        "vram_mb": max(0, int(spec.get("vram_mb") or 0)),
        "per_text_mb": max(0, int(spec.get("per_text_mb") or 0)),
        "chars_per_slot": max(0, int(spec.get("chars_per_slot") or 0)),
    }


def gpu_chars_per_slot(engine_id: str) -> int:
    """Text a batched slot was measured with, across all of an engine's models.

    The most conservative (shortest) figure the engine declares: a batch is
    sized for the model whose measurement is tightest.
    """
    declared = _GPU_PARALLEL.get(engine_id) or {}
    if str(declared.get("mode")) != "batch":
        return 0
    values = [
        int(spec.get("chars_per_slot") or 0)
        for spec in (declared.get("slots") or {}).values()
    ]
    values = [value for value in values if value > 0]
    return min(values) if values else 0


def clone_variant(engine_id: str) -> str:
    """The variant a cloned voice of this engine is stored with.

    Usually the engine's first variant.  It is declared separately where the
    models differ in *what they can do*: Chatterbox's English models cannot
    speak another language, and Qwen3-TTS's CustomVoice models cannot clone at
    all, so a clone has to land on the variant that can speak it.
    """
    info = _ENGINES_BY_ID.get(engine_id) or {}
    declared = str(info.get("clone_variant") or "")
    if declared:
        return declared
    for lang in info.get("languages", []):
        for variant in lang.get("variants", []):
            return variant["id"]
    return "default"


def qwen_model(variant: str) -> dict[str, str] | None:
    """The Qwen3-TTS model behind a variant id (``None`` for other engines)."""
    for variant_id, repo, kind, label in _QWEN_MODELS:
        if variant_id == variant:
            return {"variant": variant_id, "repo": repo, "kind": kind,
                    "label": label}
    return None


def vibevoice_model(variant: str) -> dict[str, str] | None:
    """The VibeVoice model behind a variant id (``None`` for other engines)."""
    for variant_id, repo, label in _VIBEVOICE_MODELS:
        if variant_id == variant:
            return {"variant": variant_id, "repo": repo, "label": label}
    return None


def moss_model(variant: str) -> dict[str, str] | None:
    """The MOSS-TTS checkpoint behind a variant id (``None`` for other engines)."""
    for variant_id, repo, label, vram, devices in _MOSS_MODELS:
        if variant_id == variant:
            return {"variant": variant_id, "repo": repo, "label": label,
                    "vram_mb": vram, "devices": devices}
    return None


def gptsovits_model(variant: str) -> dict[str, str] | None:
    """The GPT-SoVITS release behind a variant id (``None`` elsewhere)."""
    for variant_id, version, label, vram in _GPTSOVITS_MODELS:
        if variant_id == variant:
            return {"variant": variant_id, "version": version, "label": label,
                    "vram_mb": vram}
    return None


def import_modules(engine_id: str) -> tuple[str, ...]:
    """Modules the engine needs before it can speak (worker's list)."""
    return _IMPORT_MODULES.get(engine_id, ())


def cuda_packages(engine_id: str) -> tuple[str, ...]:
    """Packages to install from the CUDA wheel index on a GPU machine."""
    return _CUDA_PACKAGES.get(engine_id, ())


def cuda_index_url(engine_id: str) -> str | None:
    """The wheel index for ``cuda_packages`` (``None`` when the engine has none).

    :data:`PYTORCH_CUDA_INDEX` for every engine but the ones whose pinned
    PyTorch release is not on it (see ``_CUDA_INDEX``).
    """
    if not cuda_packages(engine_id):
        return None
    return _CUDA_INDEX.get(engine_id, PYTORCH_CUDA_INDEX)


def cuda_extra_packages(engine_id: str) -> tuple[str, ...]:
    """Packages installed in addition on a machine with an NVIDIA GPU."""
    return _CUDA_EXTRA_PACKAGES.get(engine_id, ())


def extra_requirements(
    engine_id: str,
) -> tuple[tuple[tuple[str, ...], str | None, tuple[str, ...]], ...]:
    """``pip`` steps beyond the engine's plain PyPI packages.

    Each entry is ``(arguments, index_url, names)``: pip arguments (flags and
    requirement specs), the index they come from, and the distribution names
    they install.
    """
    return _EXTRA_REQUIREMENTS.get(engine_id, ())


def packaged_elsewhere(engine_id: str) -> tuple[str, ...]:
    """Names installed by ``extra_requirements`` instead of by name from PyPI.

    They are part of :func:`packages` (so ``pip uninstall`` removes them) but
    must never be requested from an index by name, because they are not there.
    """
    names: list[str] = []
    for _arguments, _index, installed in extra_requirements(engine_id):
        for name in installed:
            if name not in names:
                names.append(name)
    for name in cuda_extra_packages(engine_id):
        if name not in names:
            names.append(name)
    return tuple(names)


def probe_package(engine_id: str) -> str:
    """The single package whose presence means "this engine is installed"."""
    info = _ENGINES_BY_ID.get(engine_id)
    return (info or {}).get("package", "")


def engine_name(engine_id: str) -> str:
    return (_ENGINES_BY_ID.get(engine_id) or {}).get("name", engine_id)


# ---------------------------------------------------------------------------
# Built-in (pre-made) voices
# ---------------------------------------------------------------------------
def _entry(
    engine_id: str,
    language: str,
    variant: str,
    voice: str,
    voice_name: str,
    **extra: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "tts": engine_id,
        "tts_name": engine_name(engine_id),
        "language": language,
        "variant": variant,
        "voice": voice,
        "voice_name": voice_name,
        "sid": 0,
        "engine": engine_id,
        "dir": "",
        "builtin_voice": True,
        "requires_package": probe_package(engine_id),
    }
    entry.update(extra)
    return entry


def builtin_voices(engine_id: str) -> list[dict[str, Any]]:
    """Every pre-made voice the engine ships with (no download needed)."""
    if not is_engine(engine_id):
        return []
    if engine_id == "pocket_tts":
        return [
            _entry(
                "pocket_tts", language, "pocket_v1", voice,
                f"{voice.replace('_', ' ').title()} ({_LANGUAGE_NAMES.get(language, language)})",
            )
            for voice, language in _POCKET_VOICES
        ]
    if engine_id == "f5tts":
        return [
            _entry(
                "f5tts", language, "f5_v1_base", voice,
                f"Bundled {_LANGUAGE_NAMES.get(language, language)} reference voice",
                ref_resource=f"f5_tts:{ref}",
                ref_text=text,
            )
            for (voice, language, ref, text) in _F5_REFS
        ]
    if engine_id == "kokoclone":
        return [
            _entry(
                "kokoclone", language, "kokoclone_v1", voice,
                f"{name} - Kokoro voice "
                f"({_LANGUAGE_NAMES.get(language, language)})",
            )
            for (voice, language, name) in _KOKOCLONE_VOICES
        ]
    if engine_id == "chatterbox":
        # Chatterbox ships no named voices: with no reference clip it speaks in
        # the model's own default voice, which is exactly what these entries
        # ask for.  Every other voice is one you cloned from a recording.
        voices = []
        for variant, label in (
            (CHATTERBOX_ENGLISH_VARIANT, "Chatterbox 500M"),
            (CHATTERBOX_TURBO_VARIANT, "Chatterbox Turbo 350M"),
        ):
            voices.append(
                _entry(
                    "chatterbox", "en", variant, CHATTERBOX_DEFAULT_VOICE,
                    f"Default voice - {label} (English)",
                )
            )
        for code, label in _CHATTERBOX_LANGUAGES:
            voices.append(
                _entry(
                    "chatterbox", code, CHATTERBOX_MULTILINGUAL_VARIANT,
                    CHATTERBOX_DEFAULT_VOICE,
                    f"Default voice - Chatterbox Multilingual ({label})",
                )
            )
        return voices
    if engine_id == "vibevoice":
        # Every preset is a reference recording the model conditions on, and
        # both sizes can speak with it, so each preset is listed once per
        # variant - the same shape Qwen3-TTS's named voices have.
        voices: list[dict[str, Any]] = []
        for variant, _repo, label in _VIBEVOICE_MODELS:
            for voice, language, filename, display in _VIBEVOICE_VOICES:
                voices.append(
                    _entry(
                        "vibevoice", language, variant, voice,
                        f"{display} - {label}",
                        ref_resource=f"vibevoice:demo/voices/{filename}",
                    )
                )
        return voices
    if engine_id == "moss_tts":
        # MOSS-TTS synthesises directly from text, so every variant has one
        # "default" voice of its own (the model's own speaker); every other
        # voice is one cloned from a recording.
        voices = []
        for variant, _repo, label, _vram, _devices in _MOSS_MODELS:
            for code, name in _MOSS_LANGUAGES:
                voices.append(
                    _entry(
                        "moss_tts", code, variant, MOSS_DEFAULT_VOICE,
                        f"Default voice - {label} ({name})",
                    )
                )
        return voices
    if engine_id == "gptsovits":
        # GPT-SoVITS always speaks from a reference recording and its
        # transcript, so it ships no voices at all: the ones that appear are
        # the voices the user creates or clones on the Voice Clone page.
        return list(GPTSOVITS_BUILTIN_VOICES)
    if engine_id == "qwen3tts":
        # The CustomVoice models' nine named speakers.  The Base models have no
        # voice of their own - they clone one from a recording - so they
        # contribute no entries here (they are the Voice Clone page's job).
        return [
            _entry(
                "qwen3tts", language, variant, speaker,
                f"{speaker} - {_LANGUAGE_NAMES.get(language, language)} "
                f"({blurb})",
            )
            for variant, _repo, kind, _label in _QWEN_MODELS
            if kind == "custom"
            for (speaker, language, blurb) in _QWEN_SPEAKERS
        ]
    if engine_id == "kitten_tts_2":
        # Every built-in voice is a reference clip shipped *inside* the model
        # repository, and the clip is also its transcript, so nothing has to be
        # downloaded to define them - the engine fetches the one a voice asks
        # for when it first speaks with it.  The nine language voices are listed
        # with the language they carry, because that is the only way to reach a
        # non-English language with this model.
        voices = [
            _entry(
                "kitten_tts_2", "en", KITTEN2_VARIANT, name,
                f"{name} - English ({character})",
            )
            for name, character in _KITTEN2_ENGLISH_VOICES
        ]
        voices.extend(
            _entry(
                "kitten_tts_2", code, KITTEN2_VARIANT, name,
                f"{name} language voice",
            )
            for code, name in _KITTEN2_LANGUAGE_VOICES
        )
        return voices
    if engine_id == "indic_parler":
        # Every voice is one of the model card's named speakers.  The
        # description travels with the entry because Parler-TTS has no voice id
        # to pass: the entry's ``voice`` is the speaker's name and its
        # ``description`` the sentence that asks for them (the tuning dialog can
        # replace the sentence, which is how a described voice is designed).
        return [
            _entry(
                "indic_parler", code, INDIC_PARLER_VARIANT, name,
                f"{name} - {_LANGUAGE_NAMES.get(code, code)}",
                description=_parler_description(name),
            )
            for code, names in _INDIC_PARLER_VOICES
            for name in names
        ]
    return []


def all_builtin_voices() -> list[dict[str, Any]]:
    voices: list[dict[str, Any]] = []
    for engine_id in engine_ids():
        voices.extend(builtin_voices(engine_id))
    return voices


def count_builtin_voices(engine_id: str) -> int:
    return len(builtin_voices(engine_id))
