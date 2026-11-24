"""One Python environment per describing model.

The copy of :mod:`ai_voice_studio.stt.environments` for the describe side, and
for the same reason: a describing model pulls in a vision-language stack
(``transformers`` + PyTorch + a multimodal processor) whose transitive pins clash
with both the TTS engines and the speech-to-text models.  **Each model gets a
virtualenv of its own** - ``%APPDATA%/AIVoiceStudio/describe_envs/<model id>``,
a sibling of ``tts_envs`` and ``stt_envs`` - so installing a describer can never
break a voice or a transcription model, and vice versa.

The lifecycle is identical to the STT side: the environment path, package probes
read straight from ``site-packages`` on disk (no subprocess, so the GUI stays
responsive), the pip plan (CUDA PyTorch first, then the rest from PyPI), the
post-install import verification, and removal.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable

from . import catalog

log = logging.getLogger(__name__)

#: name -> (timestamp, {normalized distribution name: version})
_DISK_CACHE: dict[str, tuple[float, dict[str, str]]] = {}
_LOCK = threading.RLock()

#: Packages that must come from the CUDA PyTorch wheel index on a GPU machine
#: (PyPI's plain ``torch`` wheel is CPU-only on Windows).
_CUDA_PACKAGES = {"torch", "torchaudio", "torchvision"}


def normalize(name: str) -> str:
    """PEP 503 normalisation (``torchvision`` == ``TorchVision``)."""
    parts = (p for p in re.split(r"[-_.]+", (name or "").strip().lower()) if p)
    return "-".join(parts)


def base_name(requirement: str) -> str:
    """Distribution name of a requirement (``torch>=2.4`` -> ``torch``)."""
    text = str(requirement or "").split(";")[0].strip()
    for separator in ("==", ">=", "<=", "!=", "~=", "===", ">", "<", "[", " "):
        text = text.split(separator)[0]
    return normalize(text)


# ---------------------------------------------------------------------------
# Environment lookup
# ---------------------------------------------------------------------------
def runtime(model_id: str):
    """The model's own ``PythonRuntime`` (see ``python_runtime``)."""
    from ..python_runtime import get_describe_runtime

    return get_describe_runtime(model_id)


def env_dir(model_id: str) -> str:
    from ..python_runtime import describe_env_dir

    return describe_env_dir(model_id)


def is_created(model_id: str) -> bool:
    try:
        return bool(runtime(model_id).is_created)
    except Exception:
        return False


def _has_cuda() -> bool:
    try:
        from .. import compute

        return compute.has_nvidia_gpu()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Package probes (from disk - never a subprocess)
# ---------------------------------------------------------------------------
def _distributions(model_id: str) -> dict[str, str]:
    """``{normalized distribution name: version}`` from the model's site-packages."""
    try:
        folders = runtime(model_id).site_packages_dirs
    except Exception:
        return {}
    found: dict[str, str] = {}
    for folder in folders:
        try:
            stamp = os.path.getmtime(folder)
        except OSError:
            continue
        with _LOCK:
            entry = _DISK_CACHE.get(folder)
        if entry is not None and entry[0] == stamp:
            found.update(entry[1])
            continue
        listed: dict[str, str] = {}
        try:
            for name in os.listdir(folder):
                if not name.endswith(".dist-info"):
                    continue
                distribution, _, version_text = name[: -len(".dist-info")].partition("-")
                listed[normalize(distribution)] = version_text
        except OSError:
            continue
        with _LOCK:
            _DISK_CACHE[folder] = (stamp, listed)
        found.update(listed)
    return found


def module_present(model_id: str, module: str) -> bool:
    """True when ``import module`` would work in this model's environment."""
    stem = normalize(module).replace("-", "_")
    if not stem:
        return False
    if stem in _distributions(model_id):
        return True
    try:
        folders = runtime(model_id).site_packages_dirs
    except Exception:
        return False
    for folder in folders:
        if os.path.isdir(os.path.join(folder, stem)):
            return True
        if os.path.isfile(os.path.join(folder, stem + ".py")):
            return True
    return False


def _package_for_module(model_id: str, module: str) -> str | None:
    stem = normalize(module)
    candidates = catalog.pip_packages(model_id)
    for package in candidates:
        name = base_name(package)
        if name == stem or stem in name or name in stem:
            return name
    return base_name(candidates[0]) if candidates else None


def installed_version(model_id: str) -> str | None:
    """Version of the model's primary back-end package, or ``None``."""
    modules = catalog.backend_modules(model_id)
    if not modules:
        return None
    package = _package_for_module(model_id, modules[0])
    if not package:
        return None
    return _distributions(model_id).get(package)


def is_installed(model_id: str) -> bool:
    """True when any back-end of this model is importable in its own env."""
    return any(
        module_present(model_id, module)
        for module in catalog.backend_modules(model_id)
    )


def installed_modules(model_id: str) -> list[str]:
    return [
        module for module in catalog.backend_modules(model_id)
        if module_present(model_id, module)
    ]


def missing_cuda_runtime(model_id: str) -> list[str]:
    """The CUDA-only packages this model's environment has not installed yet."""
    if not _has_cuda():
        return []
    have = _distributions(model_id)
    missing: list[str] = []
    for package in catalog.cuda_packages(model_id):
        name = base_name(package)
        if name and name not in have:
            missing.append(name)
    return missing


def missing_packages(model_id: str) -> list[str]:
    """The packages this model's catalogue entry declares that its env lacks.

    :func:`is_installed` answers a narrower question - is the *back-end*
    importable - and that is not the same as the environment being complete.  A
    model that runs its own modelling code can need packages no other describer
    needs: MiniCPM-o 2.6's own files import ``torchaudio`` and ``librosa``, so an
    environment built before that requirement was declared reports itself
    installed and then fails on the checkpoint's first import ("Could not load
    the model: This modeling file requires the following packages that were not
    found in your environment: torchaudio").  Naming the missing packages is what
    turns *Install* into *Repair* for an environment that already exists,
    instead of making the user delete several gigabytes of PyTorch and build the
    environment again.

    An environment whose packages cannot be listed at all is reported as having
    nothing missing - a directory that cannot be read is not evidence that every
    package is absent, and the caller's ``is_installed`` probe already covers the
    really empty case.
    """
    if not is_created(model_id):
        return []
    have = _distributions(model_id)
    if not have:
        return []
    declared = list(catalog.pip_packages(model_id))
    if _has_cuda():
        declared += catalog.cuda_packages(model_id)
    missing: list[str] = []
    for package in declared:
        name = base_name(package)
        if name and name not in have:
            missing.append(name)
    return list(dict.fromkeys(missing))


# ---------------------------------------------------------------------------
# Install / remove
# ---------------------------------------------------------------------------
def packages_for(model_id: str) -> tuple[str, ...]:
    """Plain pip names for this model's environment (install *and* uninstall)."""
    names = [base_name(p) for p in catalog.pip_packages(model_id)]
    names += [base_name(p) for p in catalog.cuda_packages(model_id)]
    return tuple(dict.fromkeys(names))


def install_plan(model_id: str) -> list[tuple[list[str], str | None]]:
    """The pip steps that install a model into its own environment.

    On a machine with an NVIDIA card the CUDA build of PyTorch is installed
    *first* from the official PyTorch wheel index; everything else comes from
    PyPI.  On a CPU-only machine a single PyPI step installs the lot.
    """
    packages = list(catalog.pip_packages(model_id))
    steps: list[tuple[list[str], str | None]] = []
    if _has_cuda():
        cuda = [p for p in packages if base_name(p) in _CUDA_PACKAGES]
        if cuda:
            from ..voicelab.engines import PYTORCH_CUDA_INDEX

            steps.append((cuda, PYTORCH_CUDA_INDEX))
            packages = [p for p in packages if base_name(p) not in _CUDA_PACKAGES]
        packages.extend(catalog.cuda_packages(model_id))
    if packages:
        steps.append((packages, None))
    return steps


def verify_import(model_id: str, timeout: float = 300.0) -> str | None:
    """Import the model's back-end in its own env; return a problem, or ``None``."""
    modules = catalog.backend_modules(model_id)
    if not modules:
        return None
    if not is_created(model_id):
        return None
    python_exe = runtime(model_id).python_exe
    if not os.path.isfile(python_exe):
        return None
    script = "import " + modules[0]
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True, text=True, timeout=timeout, creationflags=flags,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Loading {catalog.model(model_id)['name']} took too long and was stopped."
    except OSError as exc:
        return f"Could not run the environment: {exc}"
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip().splitlines()
    last = detail[-1] if detail else f"exit code {result.returncode}"
    return (
        f"{catalog.model(model_id)['name']} is installed but its back-end "
        f"failed to load: {last}"
    )


def invalidate(model_id: str | None = None) -> None:
    """Forget the on-disk package cache (after an install or removal)."""
    with _LOCK:
        if model_id is None:
            _DISK_CACHE.clear()
            return
        try:
            folders = runtime(model_id).site_packages_dirs
        except Exception:
            folders = []
        for folder in folders:
            _DISK_CACHE.pop(folder, None)
        _DISK_CACHE.pop(os.path.join(env_dir(model_id), "Lib", "site-packages"), None)


ProgressCallback = Callable[[str], None]


def install(model_id: str, progress: ProgressCallback | None = None,
            cancel_event: threading.Event | None = None) -> dict[str, object]:
    """Install a model's back-end into its own environment."""

    def _say(message: str) -> None:
        if progress is not None:
            try:
                progress(message)
            except Exception:
                log.debug("install progress callback failed", exc_info=True)

    rt = runtime(model_id)
    try:
        rt.ensure_pip()
    except Exception as exc:
        return {"ok": False, "error": str(exc), "output": ""}

    steps = install_plan(model_id)
    if not steps:
        return {"ok": False, "error": "No packages to install.", "output": ""}

    output: list[str] = []
    for packages, index_url in steps:
        _say(f"Installing {', '.join(packages)}...")
        result = rt.pip_install(
            packages,
            progress=(lambda line, _done, _total: _say(line)) if progress else None,
            cancel_event=cancel_event,
            index_url=index_url,
        )
        output.append(str(result.get("output", "")))
        if not result.get("ok"):
            invalidate(model_id)
            return {"ok": False, "error": str(result.get("error", "pip failed")),
                    "output": "\n".join(output)}

    invalidate(model_id)
    problem = verify_import(model_id)
    if problem:
        return {"ok": False, "error": problem, "output": "\n".join(output)}
    return {"ok": True, "error": "", "output": "\n".join(output)}


def remove(model_id: str) -> str | None:
    """Delete a model's environment; returns a note when it was not clean."""
    try:
        note = runtime(model_id).remove_env()
    except Exception as exc:
        return f"Could not remove the environment: {exc}"
    invalidate(model_id)
    return note


# ---------------------------------------------------------------------------
# Worker script
# ---------------------------------------------------------------------------
_WORKER_NAME = "worker.py"


def bundled_worker_path() -> str | None:
    """Path of ``worker.py`` on disk, when it is a real (readable) file."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _WORKER_NAME),
    ]
    bundle_dir = getattr(sys, "_MEIPASS", "")
    if bundle_dir:
        candidates.append(
            os.path.join(bundle_dir, "ai_voice_studio", "describe", _WORKER_NAME)
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def worker_script() -> str:
    """A ``worker.py`` a model environment's interpreter can run."""
    bundled = bundled_worker_path()
    if bundled:
        return bundled
    raise RuntimeError(
        "The audio-description worker script is missing from this installation. "
        "Reinstall the application."
    )


def env_size_bytes(model_id: str) -> int:
    """Rough size of the model's environment folder (0 when not created)."""
    root = env_dir(model_id)
    total = 0
    if not os.path.isdir(root):
        return 0
    for current, _dirs, names in os.walk(root):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(current, name))
            except OSError:
                continue
    return total
