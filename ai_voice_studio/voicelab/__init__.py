"""Voice Lab: the voice-clone engines.

Nine engines live here.  The smaller ones run on a plain CPU and switch to the
GPU when one is present; the large transformer engines are GPU-only because
their CPU run is many times slower than real time:

============== ============================ ==================================
Engine         Package                      Notes
============== ============================ ==================================
Pocket TTS     ``pocket-tts``               Kyutai, 100M params, 26 voices
F5-TTS         ``f5-tts``                   Flow-matching transformer
KokoClone      ``kokoclone`` (source)        Kokoro ONNX + Kanade voice cloning
Chatterbox     ``chatterbox-tts``           Resemble AI, 23 languages (GPU only)
Qwen3-TTS      ``qwen-tts``                 Qwen 0.6B/1.7B, 9 voices (GPU only)
VibeVoice      ``vibevoice`` (source)       Microsoft 1.5B/7B, 7 voices (GPU only)
MOSS-TTS       ``moss-tts`` (source)        OpenMOSS, 4B/1.7B (GPU only)
GPT-SoVITS     source tree                  RVC-Boss, 6 releases, clone only
Kitten TTS 2.0 ``kittenml``                  KittenML 1.7B, 47 voices (GPU only)
============== ============================ ==================================

Each one is installed into the app's managed virtualenv and driven from a
subprocess (``voicelab.worker``), so PyTorch never enters the GUI process.  The
engine wrapper below plugs into ``tts.engine.get_engine`` and implements the
same ``synthesize`` contract as the sherpa-onnx engines.

Per-engine tuning (diffusion steps, sampling temperatures, quantization, seeds,
...) lives in ``voicelab.options``: the Settings page stores defaults per
engine, the Recording window stores per-project overrides, and both ride along
with a voice entry into the worker, which applies whatever the installed
engine version actually supports.

Nothing in this module is imported by the GUI at start-up: the heavy pieces
(the worker, the venv) are created on first use.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, ClassVar

import numpy as np

from .. import hf_access
from . import engines
from . import options as _options

log = logging.getLogger(__name__)

#: Re-exported at package level so callers can write
#: ``from ..voicelab import is_engine`` (the Settings panels and the
#: Recording window do exactly that).
is_engine = engines.is_engine

#: The **gated** Hugging Face repository an engine's weights live in (``""`` for
#: the engines that download anonymously).  Kyutai's Pocket TTS is the one:
#: its voice-cloning weights are gated, its pre-made voices are not.
gated_repo = engines.gated_repo

#: ``kind`` used for voices the user cloned from their own recording.
CLONE_KIND = "clone_reference"

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_AUTO = "auto"

#: Settings key holding the device chosen on the Voice Clone settings page.
DEVICE_SETTING = "clone_engines.device"

#: Interpreter used to run the worker; defaults to the managed virtualenv.
_python_override = os.environ.get("AIVS_VOICELAB_PYTHON", "")


_DEFAULT_SAMPLE_RATE = 24000

#: Fields copied from a voice entry into a worker request.
_VOICE_KEYS = (
    "voice",
    "language",
    "variant",
    "ref_audio",
    "ref_text",
    "ref_resource",
    "ref_text_resource",
    "repo",
    # Parler-TTS has no voice-id argument: a built-in voice *is* the description
    # of it, and the entry carries that description here.  Every other engine
    # ignores the key.
    "description",
)

_WORKER_NAME = "_voicelab_worker.py"


class VoicelabError(Exception):
    """Raised when a Voice Lab engine is missing or fails to synthesize."""


def _error_text(engine_id: str, text, fallback: str = "Synthesis failed.") -> str:
    """A failure reported by the worker, in the user's own terms.

    A gated download happens *inside* the engine's environment, so the message
    that comes back is the library's - a bare ``401``, or Kyutai's advice to
    accept the terms and then log in locally with the ``hf`` command - and
    neither names anything a person who installed the setup file can do.
    :mod:`ai_voice_studio.hf_access` turns that into the licence page and this
    application's own token panel; anything that is not an access failure is
    passed through untouched, so a real bug is never dressed up as a missing
    licence.
    """
    message = str(text or "").strip() or fallback
    return (hf_access.access_message(message, engines.gated_repo(engine_id),
                                     engines.engine_name(engine_id))
            or message)


# ---------------------------------------------------------------------------
# Device detection / selection
# ---------------------------------------------------------------------------
_cuda_cache: bool | None = None


def has_cuda(force: bool = False) -> bool:
    """True when an NVIDIA GPU with a working driver is present.

    Checked with ``nvidia-smi`` (a few milliseconds, no PyTorch import) and
    cached for the session: it only decides whether the *option* to run on the
    GPU is offered, never whether an engine works at all.
    """
    global _cuda_cache
    if _cuda_cache is not None and not force:
        return _cuda_cache
    found = False
    try:
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=flags,
            check=False,
        )
        found = result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        found = False
    _cuda_cache = found
    return found


#: A measured CPU run may be this much slower than real time and still be
#: offered for an engine that is otherwise GPU-only: 2.0 means an hour of audio
#: takes at most two hours to generate.  Above it the CPU is left out whenever
#: there is an NVIDIA GPU to use instead (and kept as the only entry when there
#: is not: a slow run is still better than no run at all).
CPU_RTF_LIMIT = 2.0

#: Measured real-time factors of CPU runs, by engine: generation seconds per
#: second of audio, or ``None`` for a run that did not finish in time (and is
#: therefore above the limit by an unknown margin).  Filled from the settings
#: file when the application starts (``voicelab.benchmark.load_verdicts``) and
#: updated whenever a benchmark finishes, so the device list of the Settings
#: panel, the Recording window and the previews all agree on what *this*
#: computer can do with each engine.
_cpu_rtf: dict[str, float | None] = {}


def format_rtf(rtf: float | None) -> str:
    """How a measured real-time factor is written for the user."""
    if rtf is None:
        return "did not finish"
    if rtf < 10:
        return f"{rtf:.2f}x real time"
    return f"{rtf:.0f}x real time"


def set_cpu_rtf(engine_id: str, rtf: float | None) -> None:
    """Record a measured CPU real-time factor (``None`` = did not finish)."""
    if not engine_id:
        return
    _cpu_rtf[str(engine_id)] = None if rtf is None else max(0.0, float(rtf))


def cpu_rtf(engine_id: str | None) -> float | None:
    """The measured CPU real-time factor, or ``None`` when there is none."""
    return _cpu_rtf.get(str(engine_id or ""))


def cpu_measured(engine_id: str | None) -> bool:
    """True when a CPU run of this engine has been measured at all."""
    return str(engine_id or "") in _cpu_rtf


def cpu_verdict(engine_id: str | None) -> bool | None:
    """Whether the CPU passed the limit: ``None`` before it was measured."""
    if not cpu_measured(engine_id):
        return None
    rtf = cpu_rtf(engine_id)
    return rtf is not None and rtf <= CPU_RTF_LIMIT


def clear_cpu_rtf(engine_id: str | None = None) -> None:
    """Forget measured CPU speeds (tests, and a benchmark reset)."""
    if engine_id is None:
        _cpu_rtf.clear()
    else:
        _cpu_rtf.pop(str(engine_id), None)


def cpu_allowed(engine_id: str | None) -> bool:
    """True when the CPU should be offered for this engine right now.

    * An engine declared **GPU-only** gets the CPU only once a measurement
      says this computer runs it fast enough to be worth offering - that is
      what turns a declaration into a decision.
    * Every other engine keeps the CPU (the device that works on every
      computer) unless it was measured slower than :data:`CPU_RTF_LIMIT` *and*
      an NVIDIA GPU is present to use instead.
    """
    if not engine_id:
        return True
    verdict = cpu_verdict(engine_id)
    if engines.gpu_only(engine_id):
        return verdict is True
    return not (verdict is False and has_cuda())


def cpu_option_label(engine_id: str | None) -> str:
    """The CPU entry's label, carrying the measured figure when there is one."""
    if not engine_id or not cpu_measured(engine_id):
        return "CPU (works on every computer)"
    speed = format_rtf(cpu_rtf(engine_id))
    if cpu_verdict(engine_id) is True:
        return f"CPU (measured {speed})"
    return f"CPU (measured {speed} - slower than real time)"


def cpu_measurement_note(engine_id: str | None) -> str:
    """A sentence about a measured CPU speed, or "" when there is none."""
    if not engine_id or not cpu_measured(engine_id):
        return ""
    if cpu_verdict(engine_id) is True:
        return (
            "Measured on this computer's CPU at "
            f"{format_rtf(cpu_rtf(engine_id))}, so the CPU is offered for it."
        )
    if cpu_rtf(engine_id) is None:
        return (
            "A CPU run was measured and did not finish in time, so the CPU is "
            "not offered for it here."
        )
    return (
        "Measured on this computer's CPU at "
        f"{format_rtf(cpu_rtf(engine_id))} - slower than real time, so the "
        "CPU is not offered here while a GPU is available."
    )


def device_options(engine_id: str | None = None) -> list[tuple[str, str]]:
    """``(value, label)`` pairs for the device selector.

    The CPU is offered for every engine that can run on it; when an NVIDIA GPU
    is detected, the GPU is offered *in addition* - never instead - plus an
    Auto entry that picks the GPU when one is available at synthesis time.

    Two things change that list, and both come from a **measured** speed
    (``voicelab.benchmark``) rather than from a guess about the model's size:

    * an engine declared **GPU-only** (``engines.gpu_only``) gains the CPU
      entry when this computer measured it fast enough (at most
      :data:`CPU_RTF_LIMIT` times real time);
    * an engine whose measured CPU run is slower than that keeps the CPU only
      while no NVIDIA GPU is present, because with a card to use instead the
      CPU entry would only promise an hours-long run.

    Until an engine has been measured its declared devices decide, which is
    exactly what the application did before the benchmark existed.
    """
    gpu = has_cuda()
    if engine_id and engines.gpu_only(engine_id):
        label = "GPU (CUDA - NVIDIA, required by this engine)"
        if not gpu:
            label = (
                "GPU (CUDA - NVIDIA) - no NVIDIA GPU was found on this "
                "computer"
            )
        options: list[tuple[str, str]] = [(DEVICE_CUDA, label)]
        if cpu_allowed(engine_id):
            # Only ever after a measurement: this computer ran it fast enough
            # for the CPU to be worth offering.
            options.append((DEVICE_CPU, cpu_option_label(engine_id)))
        return options
    options = []
    if cpu_allowed(engine_id):
        options.append((DEVICE_CPU, cpu_option_label(engine_id)))
    if gpu:
        options.append((DEVICE_CUDA, "GPU (CUDA - NVIDIA)"))
        options.append((DEVICE_AUTO, "Auto (GPU when available)"))
    if not options:
        # A measured-slow CPU and no card to use instead: the CPU is all this
        # computer has, and a slow run is better than no run.
        options.append((DEVICE_CPU, cpu_option_label(engine_id)))
    return options


def resolve_device(choice: str | None,
                   engine_id: str | None = None) -> str:
    """Turn a device choice (``cpu`` / ``cuda`` / ``auto``) into a concrete one.

    A GPU-only engine resolves to the GPU unless a measurement has enabled the
    CPU for it: the CPU is not a mode it has, so a stored ``cpu`` choice (from
    before the engine was installed, or from a shared setting) must not send it
    there.  The engine itself reports the problem plainly when the computer has
    no usable NVIDIA GPU.

    The measured verdict is what makes the choice safe in the other direction
    too: an engine whose measured CPU run is above the limit has no CPU entry
    while a card is present, so a stored ``cpu`` must not start the very run
    the measurement ruled out.
    """
    if engine_id and engines.gpu_only(engine_id) and not cpu_allowed(engine_id):
        return DEVICE_CUDA
    value = (choice or "").strip().lower()
    if engine_id and not cpu_allowed(engine_id) and has_cuda() and value not in (
        DEVICE_CUDA, DEVICE_AUTO, "gpu", "cuda_gpu"
    ):
        return DEVICE_CUDA
    if value == DEVICE_CUDA:
        return DEVICE_CUDA if has_cuda() else DEVICE_CPU
    if value == DEVICE_AUTO:
        return DEVICE_CUDA if has_cuda() else DEVICE_CPU
    if value in ("gpu", "cuda_gpu"):
        return DEVICE_CUDA if has_cuda() else DEVICE_CPU
    return DEVICE_CPU


# ---------------------------------------------------------------------------
# Engine availability
# ---------------------------------------------------------------------------
def is_installed(engine_id: str) -> bool:
    """True when the engine's package is installed in its own environment.

    Every Voice Lab engine lives in its own virtualenv
    (``%APPDATA%\\AIVoiceStudio\\tts_envs\\<engine>``), so engines with
    conflicting dependencies cannot break each other.  An engine that was
    installed into the older *shared* addon environment keeps working: the
    probe falls back to it - except for an engine that requires its own
    environment (``engines.own_environment``), which must never be reported as
    installed because of something another engine put in the shared one.
    """
    package = engines.probe_package(engine_id)
    if not package:
        return False
    try:
        from .. import venv_packages

        if venv_packages.installed(package, engine=engine_id):
            return True
        if engines.own_environment(engine_id):
            return False
        # Engines installed before the per-TTS environments exist in the
        # shared addon environment; honour them until they are reinstalled.
        return venv_packages.installed(package)
    except Exception:
        return False


def installed_version(engine_id: str) -> str | None:
    package = engines.probe_package(engine_id)
    if not package:
        return None
    try:
        from .. import venv_packages

        found = venv_packages.version(package, engine=engine_id)
        if found or engines.own_environment(engine_id):
            return found
        return venv_packages.version(package)
    except Exception:
        return None


def requirement_name(requirement: str) -> str:
    """Distribution name of a requirement spec (``torch==2.6.0`` -> ``torch``).

    Used to recognise the same package whether it is named plainly or with a
    version pin.  ``_CUDA_PACKAGES`` needs the pins (Chatterbox's PyTorch has
    to be its pinned release, from the CUDA index that still carries it) while
    the rest of an engine's list holds bare names, which ``pip uninstall``
    needs; normalising both onto the name is what keeps the two in step.
    """
    name = str(requirement or "").split(";")[0].strip()
    for separator in ("==", ">=", "<=", "!=", "~=", "===", ">", "<", "[", " "):
        name = name.split(separator)[0]
    return name.strip().lower().replace("_", "-").replace(".", "-")


def install_plan(engine_id: str) -> list[tuple[str, str | None]]:
    """The pip steps that install an engine into its own environment.

    Returns ``[(packages, index_url_or_None), ...]``.  On a machine with an
    NVIDIA GPU the CUDA build of PyTorch is installed *first*, from the
    official PyTorch wheel index; the remaining packages then come from PyPI -
    two steps, because the CUDA index does not mirror the other packages.
    (PyPI's plain ``torch`` wheel is CPU-only on Windows, so a GPU run needs
    the wheels from this index.)

    The engine's compatibility pins ride along in the second step, so pip
    resolves them together with the engine's own dependencies (see
    ``engines.pins``).

    An engine whose requirements are **not** plain PyPI names contributes more
    steps of its own (``engines.extra_requirements``): KokoClone's own package
    and its Kanade model are source archives on GitHub and are installed with
    the pip flags they need.  Steps that must not replace the CUDA PyTorch
    (the GPU ONNX runtime, the ``--no-deps`` source archives) come after the
    PyPI step, so the GPU build is never undone by a later one.
    """
    from .. import venv_packages

    packages = list(packages_for(engine_id))
    steps: list[tuple[list[str], str | None]] = []
    cuda = list(engines.cuda_packages(engine_id))
    index = engines.cuda_index_url(engine_id)
    if cuda and index and has_cuda() and not venv_packages.installed(
        "torch", engine=engine_id
    ):
        steps.append((cuda, index))
        # By *name*, so a pinned CUDA requirement (``torch==2.6.0``) also takes
        # the plain ``torch`` off the by-name step - otherwise pip would fetch
        # the CPU-only wheel from PyPI and replace the CUDA build with it.
        cuda_names = {requirement_name(package) for package in cuda}
        packages = [
            p for p in packages if requirement_name(p) not in cuda_names
        ]
    # Names that another step installs (from a source archive, or with a pip
    # flag): asking for them by name would fail, they are not on the index.
    elsewhere = set(engines.packaged_elsewhere(engine_id))
    by_name = [p for p in packages if p not in elsewhere]
    if by_name:
        steps.append((by_name, None))
    if engines.cuda_extra_packages(engine_id) and has_cuda():
        steps.append((list(engines.cuda_extra_packages(engine_id)), None))
    for arguments, extra_index, _names in engines.extra_requirements(engine_id):
        steps.append((list(arguments), extra_index))
    return steps


def packages_for(engine_id: str) -> tuple[str, ...]:
    """The ``pip`` packages this engine needs in its own environment.

    This is everything that ends up installed, and therefore also what
    "Remove engine" uninstalls, so it holds plain names only - never
    ``name<version`` requirements, which ``pip uninstall`` cannot take.  A name
    that is installed from somewhere other than an index (KokoClone's own
    package, its Kanade model) is still listed here; ``install_plan`` skips
    those in the "install by name" step because they are not on PyPI.
    """
    return engines.packages(engine_id)


def _engine_source_tree(engine_id: str) -> str | None:
    """The fetched source tree an engine's worker imports from, if any.

    Only GPT-SoVITS works this way: nothing installs it into the environment,
    so the worker unpacks the pinned archive into its own folder and puts that
    folder on ``sys.path``.  ``verify_engine_import`` has to look there too,
    or it would report a healthy environment for a tree that cannot import.
    """
    if engine_id != "gptsovits":
        return None
    workspace = gptsovits_workspace()
    if os.path.isdir(os.path.join(workspace, "GPT_SoVITS")):
        return workspace
    return None


def _source_tree_probe_script(modules: tuple[str, ...], tree: str) -> str:
    """A ``python -c`` script that imports a source-tree engine for real.

    It mirrors what the worker does before importing: the repository root and
    its ``GPT_SoVITS`` folder go on ``sys.path`` (upstream's own entry point
    adds both), plain ``jieba`` stands in for ``jieba_fast`` - which has no
    buildable release for this Python - and then the engine's entry point is
    imported.  A missing dependency is then caught here, at install time,
    instead of on the first preview.
    """
    gptsovits_dir = os.path.join(tree, "GPT_SoVITS")
    names = ", ".join(modules)
    return (
        "import os, sys;\n"
        f"sys.path.insert(0, {tree!r});\n"
        f"sys.path.insert(0, {gptsovits_dir!r});\n"
        "os.chdir(sys.path[1]);\n"
        "try:\n"
        "    import jieba, jieba.posseg;\n"
        "    sys.modules.setdefault('jieba_fast', jieba);\n"
        "    sys.modules.setdefault('jieba_fast.posseg', jieba.posseg);\n"
        "except Exception:\n"
        "    pass;\n"
        f"import {names};\n"
        "from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config;\n"
    )


def verify_engine_import(engine_id: str, timeout: float = 300.0) -> str | None:
    """Import the engine in its own environment; return a problem, or ``None``.

    Installing the packages is not the same as having a working engine: pip
    can resolve a dependency combination that the engine's own code cannot
    import.  Running the import once, right after the install, turns that into
    a message the user sees immediately instead of a puzzling failure on the
    first preview.

    For an engine that runs from a **source tree** rather than from an
    installed package (GPT-SoVITS), the plain import is not enough: the tree
    is fetched and put on ``sys.path`` by the worker, so the check has to do
    the same.  It adds the tree the worker would add and imports the engine's
    real entry point, which is what catches a dependency missing from the
    engine's package list - the failure that produced a first-preview error
    rather than an install-time one.
    """
    modules = engines.import_modules(engine_id)
    if not modules:
        return None
    try:
        from ..python_runtime import get_runtime

        runtime = get_runtime(engine_id)
        if not runtime.is_created:
            return None
        # An environment built by an earlier installation runs on a Python that
        # has gone; ``usable_python_exe`` re-points it at this one first.
        python_exe = runtime.usable_python_exe()
    except Exception as exc:
        return f"Could not start the engine's Python environment: {exc}"
    if not os.path.isfile(python_exe):
        return None
    script = "import " + ", ".join(modules)
    tree = _engine_source_tree(engine_id)
    if tree:
        script = _source_tree_probe_script(modules, tree)
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            f"Loading {engines.engine_name(engine_id)} took too long "
            f"({int(timeout)}s) and was stopped."
        )
    except OSError as exc:
        return f"Could not run the engine's Python environment: {exc}"
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    last = detail[-1] if detail else f"exit code {result.returncode}"
    return f"{engines.engine_name(engine_id)} is installed but failed to load: {last}"


# ---------------------------------------------------------------------------
# Built-in voices
# ---------------------------------------------------------------------------
def builtin_voice_entries(engine_id: str | None = None) -> list[dict[str, Any]]:
    """Voice entries for the pre-made voices usable right now.

    Only engines whose package is installed contribute entries: a built-in
    voice is offered exactly when the engine that speaks it is ready.
    """
    ids = (engine_id,) if engine_id else engines.engine_ids()
    entries: list[dict[str, Any]] = []
    for engine_key in ids:
        if not is_installed(engine_key):
            continue
        entries.extend(engines.builtin_voices(engine_key))
    return entries


# ---------------------------------------------------------------------------
# Cloned voices (user recordings)
# ---------------------------------------------------------------------------
def clone_voices(store, engine_id: str | None = None) -> list[dict[str, Any]]:
    """Custom voices the user cloned with a Voice Lab engine."""
    try:
        voices = store.custom_voices(engine_id) if engine_id else store.custom_voices()
    except Exception:
        return []
    return [v for v in voices if v.get("kind") == CLONE_KIND]


def create_clone(
    store,
    engine_id: str,
    name: str,
    ref_audio: str,
    ref_text: str = "",
    language: str = "en",
    variant: str = "",
) -> dict[str, Any]:
    """Register a cloned voice from a reference recording.

    The audio file is copied into the voices folder so the clone keeps working
    when the original file is moved or deleted.  Returns the stored entry.
    """
    if not engines.is_engine(engine_id):
        raise ValueError(f"'{engine_id}' is not a Voice Lab engine.")
    voice_name = (name or "").strip()
    if not voice_name:
        raise ValueError("Type a voice name first.")
    source = (ref_audio or "").strip()
    if not source or not os.path.isfile(source):
        raise ValueError("Choose a reference audio file first (Browse...).")
    directory = store.custom_dir(voice_name)
    extension = os.path.splitext(source)[1].lower() or ".wav"
    destination = os.path.join(directory, "reference" + extension)
    try:
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise ValueError(f"Could not copy the reference audio: {exc}") from exc
    entry = {
        "engine": engine_id,
        "language": language or "en",
        # The variant decides which *model* speaks a cloned voice, and not
        # every model of an engine can clone (Qwen3-TTS's CustomVoice models
        # cannot) or speak every language (Chatterbox's English models cannot) -
        # so the engine names the variant a clone belongs on.
        "variant": variant or engines.clone_variant(engine_id),
        "voice": engines.CLONE_VOICE_ID,
        "ref_audio": destination,
        "ref_text": (ref_text or "").strip(),
        "source_audio": source,
        "mode": "clone",
    }
    store.add_custom_voice(
        voice_name, engine_id, directory, CLONE_KIND, extra=entry
    )
    for stored in store.custom_voices(engine_id):
        if stored.get("name") == voice_name:
            return stored
    return {"name": voice_name, "tts": engine_id, **entry}


def rename_clone(store, old_name: str, new_name: str) -> bool:
    return bool(store.rename_custom_voice(old_name, new_name))


def delete_clone(store, name: str) -> bool:
    return bool(store.remove_custom_voice(name))


# ---------------------------------------------------------------------------
# Worker request builder (pure, unit-testable)
# ---------------------------------------------------------------------------
def cpu_thread_count() -> int:
    """Threads a CPU run may use (80-95% of the machine, see ``compute``)."""
    try:
        from .. import compute

        return compute.cpu_threads()
    except Exception:
        return 0


def kokoclone_workspace() -> str:
    """A writable folder KokoClone may keep its downloaded model in.

    KokoClone downloads its weights into ``./model`` and ``./voice`` **relative
    to the working directory**, so its worker needs a directory it owns - the
    application itself may well be installed in ``Program Files``, where
    nothing may be written.  The folder lives inside the engine's own
    environment directory, so removing the environment removes it too.
    """
    try:
        from ..python_runtime import engine_env_dir

        return os.path.join(engine_env_dir("kokoclone"), "kokoclone_home")
    except Exception:
        return os.path.join(tempfile.gettempdir(), "aivs_voicelab", "kokoclone")


def vibevoice_workspace() -> str:
    """A writable folder VibeVoice may keep its bundled voice presets in.

    The presets (``demo/voices/*.wav`` in the engine's repository) are not
    part of the pip package, so the worker downloads the one a voice asks for
    the first time it is used.  The folder lives inside the engine's own
    environment directory, so removing the environment removes it too - the
    same arrangement KokoClone's downloads have.
    """
    try:
        from ..python_runtime import engine_env_dir

        return os.path.join(engine_env_dir("vibevoice"), "vibevoice_home")
    except Exception:
        return os.path.join(tempfile.gettempdir(), "aivs_voicelab", "vibevoice")


def gptsovits_workspace() -> str:
    """A writable folder GPT-SoVITS keeps its source tree and weights in.

    GPT-SoVITS has no packaging metadata, so nothing installs it into the
    environment's ``site-packages``: its worker fetches the pinned source tree
    into this folder and downloads the pretrained weights beside it, then adds
    the folder to ``sys.path``.  The folder lives inside the engine's own
    environment directory, so removing the environment removes it too - the
    same arrangement KokoClone's and VibeVoice's downloads have.
    """
    try:
        from ..python_runtime import engine_env_dir

        return os.path.join(engine_env_dir("gptsovits"), "gptsovits_home")
    except Exception:
        return os.path.join(tempfile.gettempdir(), "aivs_voicelab", "gptsovits")


def build_synthesize_request(
    *,
    engine_id: str,
    device: str,
    text: str,
    voice_entry: dict[str, Any] | None = None,
    speed: float = 1.0,
    tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON request sent to the Voice Lab worker.

    ``tuning`` holds the per-engine generation overrides (``voicelab.options``);
    entries equal to the engine's own default are dropped, so an untouched
    project sends no options at all.  A voice entry may carry them itself
    instead - the Recording window does exactly that.
    """
    entry = voice_entry or {}
    resolved = resolve_device(device, engine_id)
    request: dict[str, Any] = {
        "cmd": "synthesize",
        "engine": engine_id,
        "device": resolved,
        "text": text,
        "speed": float(speed),
        # How much of the machine a run may use: the worker applies this with
        # torch.set_num_threads (CPU) or maximizes the CUDA card (GPU).
        "threads": cpu_thread_count(),
    }
    for key in _VOICE_KEYS:
        value = entry.get(key)
        if value:
            request[key] = value
    if engine_id == "kokoclone":
        # KokoClone loads its Kokoro and Kanade weights from the *current
        # directory*; tell the worker where that is (and where the downloads
        # may land) instead of letting it use whatever directory the worker
        # happens to have been started in.
        request["workspace"] = kokoclone_workspace()
    if engine_id == "vibevoice":
        # VibeVoice's built-in voices are reference recordings from the
        # engine's own repository, not files inside the installed package:
        # tell the worker where the downloaded presets may live.
        request["workspace"] = vibevoice_workspace()
    if engine_id == "gptsovits":
        # GPT-SoVITS runs from a source tree its worker fetches itself, so it
        # needs a directory it owns (the application may be installed in
        # ``Program Files``, where nothing may be written).
        request["workspace"] = gptsovits_workspace()
    overrides = _options.clean(engine_id, tuning)
    if not overrides:
        # The voice entry is the fallback carrier of the options.
        overrides = _options.clean(engine_id, entry.get("options"))
    if overrides:
        request["options"] = overrides
    return request


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------
def _bundled_worker_path() -> str | None:
    """Path of ``worker.py`` on disk, when it is a real (readable) file.

    In a frozen build this module lives inside PyInstaller's PYZ archive, so
    ``__file__`` does not exist; the packaging spec copies ``worker.py`` next to
    the other data files instead (see ``packaging/ai_voice_studio.spec``).
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"),
    ]
    bundle_dir = getattr(sys, "_MEIPASS", "")
    if bundle_dir:
        candidates.append(
            os.path.join(bundle_dir, "ai_voice_studio", "voicelab", "worker.py")
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _read_worker_source() -> str | None:
    """The worker script's source, or ``None`` when this build has no copy.

    A source checkout and a frozen build both resolve here: ``worker.py`` is a
    real file beside this module, and ``packaging/ai_voice_studio.spec`` copies
    it next to the bundled data files so the same lookup works inside the
    executable.
    """
    path = _bundled_worker_path()
    if path:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return None


def _worker_environment(engine_id: str) -> dict[str, str]:
    """The environment an engine's worker process is started with.

    :func:`hf_access.worker_environment` (so a download inside the worker is
    authenticated) plus one thing every engine needs: **Python's UTF-8 mode**.

    A text file opened without an explicit encoding is read with the
    *platform's* default - Windows-1252 on Windows - so a package that ships
    a file with any non-ASCII character in it dies on the first load.
    ``kittenml``'s ``voices.json`` holds typographic quotes, which is why
    Kitten TTS 2.0 failed every synthesis with ``UnicodeDecodeError: 'charmap'
    codec can't decode byte 0x9d`` even though the model itself was fine.  The
    mode cannot be switched on inside the process (the file-encoding default is
    fixed when the interpreter starts, and patching ``locale`` does not change
    it), so it is set here, on the child, for every engine alike.
    """
    env = hf_access.worker_environment()
    env["PYTHONUTF8"] = "1"
    return env


def _worker_script_path(env_dir: str) -> str:
    """A ``worker.py`` the managed virtualenv's interpreter can run.

    The bundled file is used directly when present; otherwise the source is
    written next to the venv (a frozen app keeps ``worker.py`` inside
    PyInstaller's PYZ archive, which another interpreter cannot read).
    """
    bundled = _bundled_worker_path()
    if bundled:
        return bundled
    script = os.path.join(env_dir, _WORKER_NAME)
    source = _read_worker_source()
    if not source:
        raise VoicelabError(
            "The Voice Lab worker script is missing from this installation. "
            "Reinstall the application."
        )
    try:
        needs_write = True
        if os.path.isfile(script):
            with open(script, encoding="utf-8") as fh:
                needs_write = fh.read() != source
        if needs_write:
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(source)
    except OSError as exc:
        raise VoicelabError(f"Could not prepare the Voice Lab worker: {exc}") from exc
    return script


class CloneWorker:
    """A persistent worker subprocess for one engine + device.

    The worker loads its model once and answers ``synthesize`` requests over a
    JSON-lines pipe, so consecutive segments do not pay the model-load cost.
    """

    def __init__(self, engine_id: str, device: str = DEVICE_CPU,
                 timeout: float = 1800.0):
        self.engine_id = engine_id
        self.device = resolve_device(device, engine_id)
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._timeout = timeout
        self._next_id = 0

    # -- lifecycle ----------------------------------------------------------
    def _runtime(self):
        """The interpreter environment this engine's worker runs in.

        Its own per-TTS environment - and for an engine that requires its own
        environment (``engines.own_environment``, KokoClone) **only** that one:
        such an engine must never be started in the shared addon environment,
        which is where OmniVoice and the addons keep their packages.

        The other engines may still fall back to the shared addon environment,
        where engines installed before the per-TTS environments existed live.

        The check reads each environment's ``site-packages`` **directly**
        (``venv_packages.package_present``) instead of asking the cached
        package probe: the answer has to be right the first time.  A probe
        that has not finished yet would otherwise send the worker into the
        wrong environment, where it dies with a baffling
        "ModuleNotFoundError: No module named 'torch'" - the module the
        engine's *own* environment does have.
        """
        from .. import venv_packages
        from ..python_runtime import get_runtime

        required = tuple(engines.import_modules(self.engine_id)) or (
            engines.probe_package(self.engine_id),
        )
        required = tuple(name for name in required if name)
        engine_runtime = get_runtime(self.engine_id)
        if engine_runtime.is_created and (
            not required
            or all(
                venv_packages.package_present(name, engine=self.engine_id)
                for name in required
            )
        ):
            return engine_runtime
        name = engines.engine_name(self.engine_id)
        if engines.own_environment(self.engine_id):
            # Never the shared environment: a KokoClone that ran there would
            # mix its packages with OmniVoice's (and could be broken by the
            # other engine's next upgrade).
            raise VoicelabError(
                f"{name} is not installed in its own Python environment. "
                "Install it from Settings > Voice Clone."
            )
        runtime = get_runtime()
        if not runtime.is_created:
            runtime.ensure_env()
        if not required or all(venv_packages.package_present(name) for name in required):
            return runtime
        raise VoicelabError(
            f"{name} is not installed in its own Python environment any more. "
            "Install it again from Settings > Voice Clone."
        )


    def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        runtime = self._runtime()
        # ``usable_python_exe``: an environment built by an earlier installation
        # of the application is re-pointed at this one before it is started.
        python_exe = runtime.usable_python_exe()
        script = _worker_script_path(runtime.env_dir)
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                [python_exe, script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=flags,
                # The worker's environment: the Hugging Face token (a cloning
                # engine downloads its weights inside this process) and UTF-8
                # mode (see ``_worker_environment``).
                env=_worker_environment(self.engine_id),
            )
        except OSError as exc:
            raise VoicelabError(
                f"Could not start the {engines.engine_name(self.engine_id)} "
                f"worker: {exc}"
            ) from exc
        response = self._request({"cmd": "ping"})
        if not response.get("ok"):
            raise VoicelabError(_error_text(
                self.engine_id, response.get("error"),
                "The Voice Lab worker failed to start.",
            ))

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            log.debug("Ignoring a failure in CloneWorker.close", exc_info=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                log.debug("Ignoring a failure in CloneWorker.close", exc_info=True)

    def kill(self) -> None:
        """Stop the worker at once, without waiting for what it is doing.

        Used when the engine's own environment is about to be removed or
        rebuilt: the worker's interpreter and the DLLs it loaded are exactly
        what Windows will not let go of, and waiting for a request in flight
        would block the install behind a whole synthesis.  The ``_proc``
        reference is read without the lock on purpose - interrupting is the
        point, and a request in flight then fails with "the worker stopped",
        which is true: it is about to be replaced by one from the new
        environment.
        """
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            log.debug("Ignoring a failure in CloneWorker.kill", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            log.debug("Ignoring a failure in CloneWorker.__del__", exc_info=True)

    # -- protocol -----------------------------------------------------------
    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise VoicelabError("The Voice Lab worker is not running.")
            payload = dict(request)
            payload["id"] = self._next_id
            self._next_id += 1
            try:
                proc.stdin.write(json.dumps(payload) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                # A blank line is never an answer.  (Anything an engine prints
                # is kept off this pipe by the worker itself - see
                # ``worker._protocol_stream``.)
                while line and not line.strip():
                    line = proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise VoicelabError(f"The Voice Lab worker failed: {exc}") from exc
            if not line:
                raise VoicelabError(
                    "The Voice Lab worker stopped. Install the engine again "
                    "from Settings > Voice Clone."
                )
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise VoicelabError(
                    "Invalid response from the Voice Lab worker."
                ) from exc

    # -- public API ---------------------------------------------------------

    def available_voices(self, engine_id: str | None = None) -> dict[str, bool]:
        """Which built-in voices of an engine are usable right now."""
        self._ensure_proc()
        if engine_id and engine_id != self.engine_id:
            worker = worker_for(engine_id, self.device)
            return worker.available_voices(engine_id)
        response = self._request({"cmd": "voices", "engine": self.engine_id,
                                  "device": self.device})
        if not response.get("ok"):
            raise VoicelabError(_error_text(
                self.engine_id, response.get("error"), "Could not list voices."
            ))
        return dict(response.get("available") or {})

    def synthesize(self, request: dict[str, Any]):
        """Speak one text.

        Returns ``(int16 samples, sample rate, report)`` where ``report`` lists
        the tuning options the engine applied and those it had to skip.
        """
        self._ensure_proc()
        return self._read_audio(self._request(request), self.engine_id)

    def synthesize_batch(self, requests: list[dict[str, Any]]):
        """Speak several texts in **one** worker call.

        The requests must describe one voice with one set of tuning options and
        differ only in their text - the pool only ever gathers such calls (see
        ``WorkerPool``).  They go to the worker as a single ``texts`` list,
        which is how an engine whose own call takes a list of texts (Qwen3-TTS)
        generates the whole batch in one pass.  Every engine is still spoken
        one text per call at heart: an engine that cannot take a list is given
        one worker process per segment instead.

        Returns one ``(samples, sample rate, report)`` per request, in order.
        """
        if not requests:
            return []
        self._ensure_proc()
        template = dict(requests[0])
        template.pop("text", None)
        template["texts"] = [str(item.get("text", "")) for item in requests]
        response = self._request(template)
        if not response.get("ok"):
            raise VoicelabError(_error_text(self.engine_id,
                                            response.get("error")))
        wavs = list(response.get("wavs") or ())
        if not wavs and response.get("wav"):
            wavs = [response["wav"]]
        if len(wavs) != len(requests):
            raise VoicelabError(
                f"The engine returned {len(wavs)} recording(s) for "
                f"{len(requests)} segment(s)."
            )
        return [self._read_audio(dict(response, wav=wav), self.engine_id)
                for wav in wavs]

    @staticmethod
    def _read_audio(response: dict[str, Any], engine_id: str = ""):
        """Turn one worker response into ``(samples, rate, report)``.

        ``engine_id`` only names the engine whose gated repository a failure
        should point at (see :func:`_error_text`); it is optional so the audio
        decoding can be exercised on its own.
        """
        if not response.get("ok"):
            raise VoicelabError(_error_text(engine_id, response.get("error")))
        samples = np.frombuffer(
            base64.b64decode(response["wav"]), dtype=np.int16
        ).copy()
        try:
            sample_rate = int(response.get("sample_rate") or _DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            sample_rate = _DEFAULT_SAMPLE_RATE
        report = {
            "applied": list(response.get("applied") or []),
            "skipped": list(response.get("skipped") or []),
            "device_used": response.get("device_used"),
            # What the engine's card needed for this call (0 on a CPU run):
            # the number that answers "how much of my GPU is it using?".
            "vram_peak_mb": response.get("vram_peak_mb") or 0,
            # Explanations from the engine process (for example "audio is
            # decoded with soundfile because there is no FFmpeg here").
            "notes": list(response.get("notes") or []),
        }
        return samples, sample_rate, report


#: Video memory a GPU run leaves to the rest of the machine: the desktop, the
#: browser, the driver and this application itself.  A slot is only afforded
#: out of what is free *beyond* this, so filling a card never starts swapping
#: the desktop out - which is exactly what makes a machine stutter while a
#: book records.
GPU_VRAM_RESERVE_MB = 1500

#: How long a batch waits for the other segments of the same recording.  The
#: recording pipeline hands its segments over in one tight loop (see
#: ``tts.engine.planned_workers``), so in practice they are all waiting when
#: the first one arrives; the short grace period only covers the microseconds
#: between the calls of the thread pool that dispatches them.
BATCH_GRACE_SECONDS = 0.05


def parallel_plan(engine_id: str, device: str = DEVICE_CPU,
                  variant: str = "") -> tuple[int, str]:
    """How many segments this engine may speak at once, and how.

    Returns ``(slots, mode)``.  ``mode`` is how the engine is *given* them:

    ``instances``  one text per call, so one worker process - one loaded model
        - per segment (Chatterbox).
    ``batch``      one worker process whose call takes a list of texts and
        generates them all in one pass (Qwen3-TTS).
    ``solo``       one segment at a time (a CPU run, or an engine that can do
        neither).

    The card decides how many slots are affordable: what is free *now*, minus
    the reserve above, divided by what one slot costs (``engines.gpu_parallel``
    declares those figures, measured on a 12 GB card).  A card with no room for
    a second slot - a 4 GB card, or any card already busy - gets one, which is
    the behaviour every engine had before; a 24 GB card gets as many as the
    engine itself can use.  Nothing here reduces quality: every slot speaks its
    own segment with the model at full precision.  When the driver does not
    answer, one slot it is.
    """
    resolved = resolve_device(device, engine_id)
    declared = engines.gpu_parallel(engine_id, variant)
    if resolved != DEVICE_CUDA or not declared:
        return 1, "solo"
    try:
        from .. import compute

        free = compute.free_vram_mb()
    except Exception:
        free = None
    if not free:
        return 1, "solo"
    budget = int(free) - GPU_VRAM_RESERVE_MB
    first = int(declared["vram_mb"])
    extra = int(declared["per_text_mb"]) or first
    if first <= 0 or extra <= 0:
        return 1, "solo"
    slots = 1 + (budget - first) // extra
    return max(1, min(int(declared["max_slots"]), slots)), str(declared["mode"])


def batch_chars(engine_id: str, slots: int) -> int:
    """How much text one batched call may hold (0 = no limit of its own).

    A slot's video-memory figure was measured with a short segment (see
    ``engines._GPU_PARALLEL``); a batch of whole paragraphs would ask the card
    for several times that.  The budget is that measured length times the
    slots, so a project of ordinary sentences still fills the card while a
    project of very long ones is spoken in smaller batches - each segment still
    generated as one piece of its own, so its audio is unchanged.
    """
    per_slot = engines.gpu_chars_per_slot(engine_id)
    if not per_slot:
        return 0
    return per_slot * max(1, int(slots or 1))


class _Call:
    """One caller waiting for its own audio from a pooled worker."""

    __slots__ = ("_event", "error", "request", "result")

    def __init__(self, request: dict[str, Any]):
        self.request = request
        self._event = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None

    def settle(self, result: Any = None, error: BaseException | None = None) -> None:
        """Give the caller its answer; the first one wins (never double-settles)."""
        if self._event.is_set():
            return
        self.result = result
        self.error = error
        self._event.set()

    def wait(self):
        self._event.wait()
        if self.error is not None:
            raise self.error
        return self.result


class WorkerPool:
    """The worker process(es) one engine on one device uses, and how.

    ``slots`` is how many segments may be in flight at the same time and
    ``mode`` says how the engine is given them (see :func:`parallel_plan`):

    * ``instances`` - ``slots`` worker processes, each with its own loaded
      model, ``synthesize`` borrowing a free one;
    * ``batch`` - one worker process whose call takes several texts: callers
      that arrive together are gathered into **one** call, which is what lets
      a single model copy decode several segments at once.  A batch is bounded
      by ``chars_per_batch`` as well as by ``slots``, because a slot's memory
      figure was measured with a short segment;
    * ``solo`` - one worker process, one segment at a time (what every engine
      did before; still what a CPU run and an unknown card get).

    Either way a caller just waits for its own audio, so the recording pipeline
    needs no special casing: it already asks for several segments at once.
    """

    def __init__(self, engine_id: str, device: str = DEVICE_CPU,
                 slots: int = 1, mode: str = "solo",
                 chars_per_batch: int | None = None):
        self.engine_id = engine_id
        self.device = resolve_device(device, engine_id)
        self.slots = max(1, int(slots or 1))
        self.mode = str(mode or "solo") if self.slots > 1 else "solo"
        #: Text one batched call may hold (0 = no budget of its own).
        self.chars_per_batch = (
            batch_chars(engine_id, self.slots)
            if chars_per_batch is None else max(0, int(chars_per_batch))
        )
        #: One process per slot for an engine that speaks one text per call;
        #: a single process for one that takes a batch (or runs solo).
        count = self.slots if self.mode == "instances" else 1
        self._workers = [
            CloneWorker(engine_id, self.device) for _ in range(max(1, count))
        ]
        self._idle: queue.Queue = queue.Queue()
        for worker in self._workers:
            self._idle.put(worker)
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self._dispatcher: threading.Thread | None = None
        self._closed = False

    # -- access --------------------------------------------------------------
    @property
    def worker(self) -> CloneWorker:
        """The pool's first worker (probing, cloning, ``available_voices``)."""
        return self._workers[0]

    @property
    def workers(self) -> list[CloneWorker]:
        return list(self._workers)

    def describe(self) -> str:
        """A line for the log: what this engine will do with the card."""
        name = engines.engine_name(self.engine_id)
        if self.mode == "batch" and self.slots > 1:
            return (
                f"{name}: up to {self.slots} segments are spoken in one "
                "batched call on the GPU"
            )
        if self.mode == "instances" and self.slots > 1:
            return (
                f"{name}: {self.slots} segments are spoken at once, one "
                "worker process (one loaded model) each"
            )
        return f"{name}: one segment at a time"

    # -- synthesis -----------------------------------------------------------
    def synthesize(self, request: dict[str, Any]):
        """Speak one text, waiting for a slot if the pool is busy."""
        if self._closed:
            raise VoicelabError("The Voice Lab worker was shut down.")
        if self.mode == "batch":
            return self._batched(request)
        worker = self._idle.get()
        try:
            return worker.synthesize(request)
        finally:
            self._idle.put(worker)

    def synthesize_batch(self, requests: list[dict[str, Any]]):
        """Speak several texts: through the batch path, or one after another."""
        if self.mode == "batch":
            return self._speak_batch(requests)
        return [self.synthesize(request) for request in requests]

    # -- batching ------------------------------------------------------------
    @staticmethod
    def _same_call(left: dict[str, Any], right: dict[str, Any]) -> bool:
        """True when two requests differ only in the text they ask for."""
        keys = {key for key in (*left, *right) if key not in ("text", "texts", "id")}
        return all(left.get(key) == right.get(key) for key in keys)

    @staticmethod
    def _text_length(request: dict[str, Any]) -> int:
        """How much text a request asks for (its share of a batch's budget)."""
        texts = request.get("texts")
        if texts is None:
            return len(str(request.get("text") or ""))
        if isinstance(texts, str):
            return len(texts)
        return sum(len(str(text)) for text in texts)

    def _batched(self, request: dict[str, Any]):
        """Hand a single request to the batching dispatcher and wait for it."""
        self._ensure_dispatcher()
        call = _Call(request)
        self._queue.put(call)
        return call.wait()

    def _ensure_dispatcher(self) -> None:
        with self._lock:
            if self._dispatcher is not None and self._dispatcher.is_alive():
                return
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop,
                name=f"voicelab-batch-{self.engine_id}",
                daemon=True,
            )
            self._dispatcher.start()

    def _dispatch_loop(self) -> None:
        """Gather the callers that arrive together into one engine call."""
        carry: _Call | None = None
        while True:
            call = carry if carry is not None else self._queue.get()
            carry = None
            if call is None:  # the sentinel ``close`` puts in the queue
                return
            batch = [call]
            try:
                chars = self._text_length(call.request)
                deadline = time.monotonic() + BATCH_GRACE_SECONDS
                while len(batch) < self.slots:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if nxt is None:
                        break
                    if not self._same_call(batch[0].request, nxt.request):
                        # A different voice or tuning: it cannot share this
                        # call, and it must not be left in the queue for the
                        # next round to lose.  Run this batch, then start on
                        # it.
                        carry = nxt
                        break
                    length = self._text_length(nxt.request)
                    if self.chars_per_batch and chars + length > self.chars_per_batch:
                        # Long segments: one more of them would ask the card
                        # for more memory than a slot's figure covers.  This
                        # batch is spoken as it is, and the long one starts
                        # the next.
                        carry = nxt
                        break
                    batch.append(nxt)
                    chars += length
            except BaseException as exc:
                # Whatever went wrong, these callers must not wait forever:
                # a segment that fails is a failed recording, a caller that
                # never answers is a hung one.
                log.exception("The Voice Lab batch dispatcher failed")
                for item in batch:
                    item.settle(
                        error=VoicelabError(
                            f"Could not gather the segments for this engine: {exc}"
                        )
                    )
                continue
            self._run_batch(batch)

    def _run_batch(self, batch: list[_Call]) -> None:
        """Speak one gathered batch and give every caller its own audio."""
        if not batch:
            return
        if len(batch) == 1:
            try:
                batch[0].settle(self._workers[0].synthesize(batch[0].request))
            except BaseException as exc:
                batch[0].settle(error=exc)
            return
        try:
            results = self._speak_batch([call.request for call in batch])
        except BaseException as exc:
            # The card may simply not have had room for the whole batch (or a
            # text was one the engine refuses).  One call per text is what the
            # engine would have done on its own, and it keeps a single bad
            # segment from failing the segments recorded beside it.
            log.warning(
                "A batch of %d %s segments failed (%s); retrying them one at a "
                "time",
                len(batch), engines.engine_name(self.engine_id), exc,
            )
            for call in batch:
                try:
                    call.settle(self._workers[0].synthesize(call.request))
                except BaseException as one:
                    call.settle(error=one)
            return
        for call, result in zip(batch, results, strict=False):
            call.settle(result)

    def _speak_batch(self, requests: list[dict[str, Any]]):
        """One worker call for several texts (the worker does the batching)."""
        if len(requests) == 1:
            return [self._workers[0].synthesize(requests[0])]
        return self._workers[0].synthesize_batch(requests)

    # -- lifecycle -----------------------------------------------------------
    def close(self, kill: bool = False) -> None:
        """Shut the pool down (``kill`` stops requests in flight at once)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            dispatcher = self._dispatcher
            self._dispatcher = None
        try:
            self._queue.put(None)
        except Exception:
            log.debug("Ignoring a failure in WorkerPool.close", exc_info=True)
        if dispatcher is not None:
            dispatcher.join(timeout=5)
        for worker in self._workers:
            try:
                worker.kill() if kill else worker.close()
            except Exception:
                log.debug("Ignoring a failure in WorkerPool.close", exc_info=True)


_worker_lock = threading.RLock()
_pools: dict[tuple[str, str], WorkerPool] = {}


def pool_for(engine_id: str, device: str = DEVICE_CPU,
             slots: int | None = None,
             mode: str | None = None) -> WorkerPool:
    """The worker pool of ``(engine, device)``: models stay loaded.

    One pool per engine and device, shared by everything in the process - a
    preview, a recording and the Settings panels all use the same loaded
    models - so the number of segments in flight is a property of the machine
    (see :func:`parallel_plan`), not of whoever asked first.  ``slots`` and
    ``mode`` override the plan (tests, probing).
    """
    resolved = resolve_device(device, engine_id)
    key = (engine_id, resolved)
    with _worker_lock:
        pool = _pools.get(key)
        if pool is None:
            if slots is None:
                slots, auto_mode = parallel_plan(engine_id, resolved)
                mode = mode or auto_mode
            pool = WorkerPool(engine_id, resolved, slots=slots, mode=mode or "solo",
                              chars_per_batch=batch_chars(engine_id, slots))
            _pools[key] = pool
            log.info("Voice Lab pool: %s", pool.describe())
        return pool


def worker_for(engine_id: str, device: str = DEVICE_CPU) -> CloneWorker:
    """The first worker of an engine's pool (probing, cloning, previews)."""
    return pool_for(engine_id, device).worker


def close_workers(engine_id: str | None = None) -> None:
    """Shut every worker down, or only one engine's (tests, app exit).

    One engine's workers are closed before that engine is removed or
    reinstalled: a running worker keeps its interpreter and the DLLs it loaded
    open, and Windows refuses to delete those while the process lives - which
    is what left a half-deleted environment behind and made the *next* install
    fail.  The cache entry goes too, so the next request starts a worker from
    the freshly installed environment.
    """
    with _worker_lock:
        if engine_id is None:
            pools = list(_pools.values())
            _pools.clear()
        else:
            keys = [key for key in _pools if key[0] == engine_id]
            pools = [_pools.pop(key) for key in keys]
    for pool in pools:
        # Removing or rebuilding one environment must not wait for a request
        # in flight (a synthesis can take minutes); the whole-cache shutdown
        # keeps the polite, waiting one.
        pool.close(kill=engine_id is not None)


# ---------------------------------------------------------------------------
# Engine wrapper (drops into the app's get_engine interface)
# ---------------------------------------------------------------------------
class VoicelabEngine:
    """Wraps a Voice Lab worker so it plugs into the TTS engine cache.

    Implements the same ``synthesize`` contract as ``TtsEngine``: returns int16
    samples at ``sample_rate``, honouring the pitch/volume helpers the app
    applies to every engine.
    """

    def __init__(self, voice_entry: dict[str, Any], device: str = DEVICE_CPU):
        engine_id = voice_entry.get("engine")
        if not engines.is_engine(engine_id):
            raise VoicelabError(f"'{engine_id}' is not a Voice Lab engine.")
        self.voice_entry = voice_entry
        self.engine_id = engine_id
        #: Per-engine tuning overrides carried by this voice entry.
        self.tuning = _options.clean(engine_id, voice_entry.get("options"))
        self.device = resolve_device(
            voice_entry.get("device") or device or DEVICE_CPU, engine_id
        )
        self.sample_rate = _DEFAULT_SAMPLE_RATE
        # The engine's workers, with however many segments the card can hold
        # in flight (see ``parallel_plan``).  On a CPU run - and on a card with
        # no room for a second model copy - that is the single worker this
        # always was.
        self._pool = pool_for(engine_id, self.device)
        #: Kept for callers that reach for the worker itself (probing, tests).
        self._worker = self._pool.worker
        #: How many segments this engine may speak at once.
        self.slots = self._pool.slots
        #: How it is given them: ``solo``, ``instances`` or ``batch``.
        self.parallel_mode = self._pool.mode

    @property
    def size(self) -> int:
        """How many segments this engine speaks at the same time.

        The same question ``TtsEnginePool.size`` answers for a sherpa-onnx pool,
        and the recording pipeline reads it back for exactly the same reason: it
        keeps that many segments in flight, and an engine that could only load
        part of what it was asked for must say so.  A Voice Lab engine's pool is
        already settled by the time it is asked (see ``parallel_plan``).
        """
        return max(1, int(self._pool.slots))

    # -- synthesis ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        request = build_synthesize_request(
            engine_id=self.engine_id,
            device=self.device,
            text=text,
            voice_entry=self.voice_entry,
            speed=speed,
            tuning=self.tuning,
        )
        samples, sample_rate, report = self._pool.synthesize(request)
        self._report_unused_options(report)
        self._report_notes(report)
        self._report_vram(report)
        if sample_rate:
            self.sample_rate = sample_rate
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch

            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume

            samples = _apply_volume(samples, volume)
        return samples

    @staticmethod
    def _report_unused_options(report: dict[str, Any] | None) -> None:
        """Log the tuning values the installed engine could not apply.

        Never fatal: an older/newer engine version may simply not have the
        knob, and the recording should still be produced with the default.
        """
        skipped = list((report or {}).get("skipped") or [])
        if skipped:
            log.warning(
                "The installed engine does not accept these tuning options "
                "(they were skipped): %s", ", ".join(sorted(skipped))
            )

    @staticmethod
    def _report_notes(report: dict[str, Any] | None) -> None:
        """Log the explanations the engine process sent back.

        A note means the run took a different route than the obvious one - for
        example audio was decoded with soundfile because this computer has no
        FFmpeg - which is worth putting in the log even though it is not an
        error.
        """
        for note in list((report or {}).get("notes") or []):
            log.info("Voice Lab engine note: %s", note)

    #: Highest video-memory peak reported per engine so far.  The first call of
    #: a run is the interesting one (it includes loading the model), so a new
    #: high goes into the log and everything below it stays quiet.
    _vram_seen: ClassVar[dict[str, int]] = {}

    def _report_vram(self, report: dict[str, Any] | None) -> None:
        """Log how much of the card the engine actually used for a call.

        The answer to "is it using my whole GPU?": the engine's own peak for
        the call that just finished, as the driver's allocator saw it.
        """
        peak = int((report or {}).get("vram_peak_mb") or 0)
        seen = VoicelabEngine._vram_seen
        if peak <= 0 or peak <= seen.get(self.engine_id, 0):
            return
        seen[self.engine_id] = peak
        log.info(
            "%s used %.1f GB of video memory for this call",
            engines.engine_name(self.engine_id), peak / 1024,
        )

    def close(self) -> None:
        """Release this engine.  The shared worker pool stays loaded.

        A preview closing the engine must not shut down the models a recording
        beside it is speaking with: the pool belongs to the engine and the
        device, not to this wrapper.  ``voicelab.close_workers`` is what stops
        the workers (an engine being reinstalled, or the application exiting).
        """


