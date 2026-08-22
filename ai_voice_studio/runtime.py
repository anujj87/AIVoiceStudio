"""Optional compute runtimes (the GPU dependency).

The application bundles a CPU-only ONNX Runtime together with sherpa-onnx
(the "main ONNX dependency"). An optional **GPU (CUDA) runtime** can be
downloaded from Settings to enable the GPU back-end. It is stored in the
per-user data folder (``%APPDATA%\\AIVoiceStudio\\runtime\\cuda``) -- never
in the installed program folder -- and is activated at startup, so a restart
is required after downloading or removing it.

Why separate runtimes? ONNX Runtime binaries are large (the CUDA runtime
with the CUDA 12 / cuDNN 9 DLLs is ~2.4 GB unpacked). Bundling it for every
user would bloat the installer, so GPU users opt in. The runtime's
``onnxruntime.dll`` is preloaded into the process before sherpa-onnx
imports, which shadows the bundled CPU copy -- Windows then resolves
sherpa-onnx's ``onnxruntime.dll`` dependency to the GPU build, giving the
same process a single ONNX Runtime (mixing two would corrupt the graph
loader).
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import tarfile
import threading
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional

from . import paths

log = logging.getLogger(__name__)

# Lazy imports from tts.downloader to avoid circular / PyInstaller bundle issues.
# These are only needed at runtime (install/activate), not at startup.
_downloader_loaded = False
DownloadCancelled = type("DownloadCancelled", (Exception,), {})
DownloadError = type("DownloadError", (Exception,), {})

def _load_dl():
    global _downloader_loaded, DownloadCancelled, DownloadError  # noqa: PLW0603
    if _downloader_loaded:
        return
    from .tts.downloader import DownloadCancelled as _DC, DownloadError as _DE  # noqa: PLC0415
    DownloadCancelled = _DC
    DownloadError = _DE
    _downloader_loaded = True

def download_file(*a, **kw):
    """Lazy wrapper — imports the real download_file on first call."""
    _load_dl()
    from .tts.downloader import download_file as _df  # noqa: PLC0415
    return _df(*a, **kw)

ProgressCallback = Callable[[str, int, int], None]

_RUNTIME_KIND = "cuda"
_MARKER = "installed.json"

# Official CUDA-enabled sherpa-onnx build, same version as the bundled CPU
# runtime (sherpa-onnx 1.13.5 / ONNX Runtime 1.27.1). We take only the
# onnxruntime.dll and the CUDA execution-provider plugins from it.
_CUDA_SHERPA_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.5/"
    "sherpa-onnx-v1.13.5-cuda-12.x-cudnn-9.x-onnxruntime1.27.1-win-x64-cuda.tar.bz2"
)
_CUDA_TARBALL_FILES = (
    "lib/onnxruntime.dll",
    "lib/onnxruntime_providers_cuda.dll",
    "lib/onnxruntime_providers_shared.dll",
)

# NVIDIA redistributable wheels (win_amd64), pinned to the versions validated
# with the CUDA build above. Each entry: (package, version, folder-in-wheel,
# list of DLL names to install).
_NVIDIA_WHEELS: List[tuple] = [
    (
        "nvidia-cudnn-cu12",
        "9.9.0.52",
        "nvidia/cudnn/bin",
        [
            "cudnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
        ],
    ),
    (
        "nvidia-cublas-cu12",
        "12.9.2.10",
        "nvidia/cublas/bin",
        ["cublas64_12.dll", "cublasLt64_12.dll"],
    ),
    ("nvidia-cufft-cu12", "11.4.1.4", "nvidia/cufft/bin", ["cufft64_11.dll"]),
    (
        "nvidia-cuda-runtime-cu12",
        "12.9.79",
        "nvidia/cuda_runtime/bin",
        ["cudart64_12.dll"],
    ),
]

#: Approximate download size (MB) for the UI note.
DOWNLOAD_SIZE_MB = 1800


def runtime_dir() -> str:
    """The folder holding the optional GPU runtime (created on demand)."""
    return paths.runtime_dir(_RUNTIME_KIND)


def is_installed() -> bool:
    """True when a complete GPU runtime is present."""
    d = runtime_dir()
    if not os.path.isfile(os.path.join(d, _MARKER)):
        return False
    return (
        os.path.isfile(os.path.join(d, "onnxruntime.dll"))
        and os.path.isfile(os.path.join(d, "onnxruntime_providers_cuda.dll"))
    )


def installed_version() -> Optional[str]:
    try:
        with open(os.path.join(runtime_dir(), _MARKER), "r", encoding="utf-8") as fh:
            return json.load(fh).get("version")
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def _wheel_url(package: str, version: str) -> str:
    """Resolve the win_amd64 wheel URL for ``package==version`` on PyPI."""
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{package}/{version}/json", timeout=30
        ) as resp:
            data = json.load(resp)
        for f in data.get("urls", []):
            if "win_amd64" in f.get("filename", ""):
                return f["url"]
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not resolve %s %s: %s", package, version, exc)
    raise DownloadError(f"Could not resolve {package} {version} on PyPI")


def _download_and_extract(
    url: str,
    filename: str,
    cache_dir: str,
    dest_dir: str,
    wanted: Dict[str, str],
    progress: Optional[ProgressCallback],
    cancel_event: Optional[threading.Event],
) -> None:
    """Download one artifact and extract the requested members by name."""
    archive = download_file(
        url,
        cache_dir,
        filename=filename,
        progress=progress,
        cancel_event=cancel_event,
    )
    if cancel_event and cancel_event.is_set():
        raise DownloadCancelled()
    if filename.endswith(".tar.bz2"):
        _extract_tarball(archive, dest_dir, wanted)
    else:
        _extract_zip(archive, dest_dir, wanted)
    try:
        os.remove(archive)
    except OSError:
        pass


def _extract_tarball(archive: str, dest_dir: str, wanted: Dict[str, str]) -> None:
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            # The archive has a top-level folder
            # ("sherpa-onnx-v.../lib/onnxruntime.dll"); match against the
            # path with that prefix stripped as well.
            parts = name.split("/")
            rel = "/".join(parts[1:]) if len(parts) > 1 else name
            target = wanted.get(rel) or wanted.get(name)
            if target is None:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(os.path.join(dest_dir, target), "wb") as fh:
                shutil.copyfileobj(src, fh)


def _extract_zip(archive: str, dest_dir: str, wanted: Dict[str, str]) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not info.is_dir() and name in wanted:
                with zf.open(info) as src, open(
                    os.path.join(dest_dir, wanted[name]), "wb"
                ) as fh:
                    shutil.copyfileobj(src, fh)


def install(
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Download and install the optional GPU runtime; returns its folder."""
    dest = runtime_dir()
    if is_installed():
        return dest
    os.makedirs(dest, exist_ok=True)
    cache = os.path.join(paths.user_data_dir(), "runtime", ".cache")
    os.makedirs(cache, exist_ok=True)
    try:
        # 1) sherpa-onnx CUDA build: onnxruntime.dll + CUDA provider plugins.
        _download_and_extract(
            _CUDA_SHERPA_URL,
            "sherpa-onnx-cuda-win-x64.tar.bz2",
            cache,
            dest,
            {f: os.path.basename(f) for f in _CUDA_TARBALL_FILES},
            progress,
            cancel_event,
        )
        # 2) NVIDIA runtime DLLs (cuDNN, cuBLAS, cuFFT, CUDA runtime).
        for package, version, folder, names in _NVIDIA_WHEELS:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled()
            url = _wheel_url(package, version)
            wanted = {f"{folder}/{n}": n for n in names}
            _download_and_extract(
                url,
                f"{package}-{version}.whl",
                cache,
                dest,
                wanted,
                progress,
                cancel_event,
            )
        with open(os.path.join(dest, _MARKER), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "runtime": _RUNTIME_KIND,
                    "version": "sherpa-onnx-1.13.5-cuda12-cudnn9-ort1.27.1",
                    "files": sorted(os.listdir(dest)),
                },
                fh,
                indent=2,
            )
    finally:
        # Leave the cache folder for resumable downloads; clear empty cache.
        try:
            if os.path.isdir(cache) and not os.listdir(cache):
                os.rmdir(cache)
        except OSError:
            pass
    return dest


def remove() -> None:
    """Delete the optional GPU runtime (user data only)."""
    shutil.rmtree(runtime_dir(), ignore_errors=True)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
def activate() -> bool:
    """Preload the GPU runtime so it shadows the bundled CPU runtime.

    Must run before ``import sherpa_onnx`` (the engine does this). Returns
    True when the GPU runtime was loaded; the rest of the app then offers
    the GPU (CUDA) back-end for this session.
    """
    if not is_installed():
        return False
    d = runtime_dir()
    if not os.path.isfile(os.path.join(d, "onnxruntime.dll")):
        return False
    try:
        # Load the CUDA onnxruntime.dll into the process first. Windows then
        # resolves any later "onnxruntime.dll" load to this module, so the
        # bundled CPU copy is never mixed in.
        ctypes.WinDLL(os.path.join(d, "onnxruntime.dll"))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)  # type: ignore[attr-defined]
        path = os.environ.get("PATH", "")
        if d not in path:
            os.environ["PATH"] = d + os.pathsep + path
        log.info("Activated optional GPU runtime from %s", d)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not activate GPU runtime: %s", exc)
        return False
