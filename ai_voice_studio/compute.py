"""Compute back-end detection.

Rules:
* CPU is always available (the bundled sherpa-onnx runtime).
* GPU (CUDA) is offered only when the optional GPU runtime has been
  downloaded from Settings *and* an NVIDIA GPU with a working driver is
  present. No CUDA toolkit install is needed - the runtime ships the
  CUDA/cuDNN DLLs, only the NVIDIA driver must be installed.
* NPU (DirectML) is never offered: no DirectML/NPU runtime exists for
  sherpa-onnx yet.
* "Auto" is offered when more than one back-end is available (preference
  order: GPU -> CPU).

Detection deliberately does NOT import onnxruntime: sherpa-onnx bundles its
own runtime, and importing a second ``onnxruntime`` package into the same
process corrupts the graph loader (two DLLs with the same module name). The
GPU runtime, when downloaded, is preloaded by the engine before sherpa-onnx
is imported.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional

from . import runtime
from .constants import COMPUTE_AUTO, COMPUTE_CPU, COMPUTE_CUDA, COMPUTE_DML

log = logging.getLogger(__name__)

#: A CPU run uses this share of the machine's logical CPUs: at least 80% and
#: at most 95%, so a long recording is as fast as the CPU can make it while
#: the desktop (and the Recording window) stays responsive.
CPU_THREAD_MIN_RATIO = 0.80
CPU_THREAD_MAX_RATIO = 0.95


_nvidia_cache: bool | None = None


def has_nvidia_gpu(force: bool = False) -> bool:
    """True when an NVIDIA GPU with a working driver is present.

    Driver-only check (``nvidia-smi``, no runtime download involved): it
    decides whether the *option* "GPU" is offered at all.  The sherpa-onnx
    ONNX engines additionally need the downloaded CUDA runtime (see
    :func:`detect`), while the pip-installed engines bring their own PyTorch.
    """
    global _nvidia_cache
    if _nvidia_cache is not None and not force:
        return _nvidia_cache
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
        )
        found = result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        found = False
    _nvidia_cache = found
    return found


def _has_cuda(force: bool = False) -> bool:
    # The GPU back-end needs the optional CUDA runtime (downloaded from
    # Settings) plus an NVIDIA GPU with a usable driver.
    if not runtime.is_installed():
        return False
    return has_nvidia_gpu(force)


def _has_npu() -> bool:
    # No DirectML/NPU runtime is bundled or downloadable for sherpa-onnx.
    return False


def detect(force: bool = False) -> Dict[str, bool]:
    """Return {'cpu': bool, 'cuda': bool, 'dml': bool} (cached).

    ``force=True`` re-runs every probe, *including* the (otherwise cached)
    driver check, so a test or a "Re-detect" button sees today's hardware.
    """
    cache = getattr(detect, "_cache", None)
    if cache is not None and not force:
        return cache
    result = {
        COMPUTE_CPU: True,
        COMPUTE_CUDA: _has_cuda(force),
        COMPUTE_DML: _has_npu(),
    }
    detect._cache = result  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# How much of the machine a run may use
# ---------------------------------------------------------------------------
def cpu_threads(total: Optional[int] = None) -> int:
    """How many CPU threads a local run should use.

    The CPU back-end uses as much of the machine as it can while staying
    inside the 80-95% band of the logical CPUs: the **upper** end of the band
    (e.g. 15 of 16, 30 of 32), never quite pinning every core - so a long
    recording is as fast as the CPU can make it while the desktop, the
    Recording window and the audio writer stay responsive.

    Machines with very few cores keep one core free instead (on two or three
    cores the band is empty), and a single-core machine uses its one core.
    """
    try:
        cpus = int(total if total is not None else (os.cpu_count() or 1))
    except (TypeError, ValueError):
        cpus = 1
    if cpus <= 1:
        return 1
    low = max(1, int(math.ceil(cpus * CPU_THREAD_MIN_RATIO)))
    high = max(1, int(math.floor(cpus * CPU_THREAD_MAX_RATIO)))
    if high < low:
        # Too few cores for a non-empty 80-95% band: leave one core free.
        return max(1, cpus - 1)
    # As much CPU as the rule allows: the top of the band.
    return max(1, min(high, cpus))


def cuda_device_index() -> int:
    """The CUDA device a GPU run uses.

    Whichever NVIDIA card the driver exposes is used - a single-process
    engine drives one device, so this is always the first (and usually only)
    one.  It is deliberately *not* tied to a specific model or a VRAM
    minimum: any CUDA-capable NVIDIA card is "good enough" here.
    """
    return 0


def gpu_summary() -> str:
    """Human-readable name of the NVIDIA card a GPU run would use."""
    if not has_nvidia_gpu():
        return ""
    try:
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=flags,
        )
        first = (result.stdout or "").strip().splitlines()
        if result.returncode == 0 and first:
            return first[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def available_compute() -> List[str]:
    """Compute options the UI may offer (plus Auto when >1)."""
    det = detect()
    options = []
    for key in (COMPUTE_CUDA, COMPUTE_DML, COMPUTE_CPU):
        if det.get(key):
            options.append(key)
    if len(options) > 1:
        options.insert(0, COMPUTE_AUTO)
    if not options:
        options = [COMPUTE_CPU]
    return options


def resolve_compute(choice: str) -> str:
    """Turn a UI selection ('auto'|'cpu'|'cuda'|'dml') into a concrete back-end."""
    if choice != COMPUTE_AUTO:
        return choice
    det = detect()
    for key in (COMPUTE_CUDA, COMPUTE_DML, COMPUTE_CPU):
        if det.get(key):
            return key
    return COMPUTE_CPU


def provider_for(compute: str) -> str:
    """Map a compute back-end to the string sherpa-onnx expects."""
    mapping = {
        COMPUTE_CPU: "cpu",
        COMPUTE_CUDA: "cuda",
        COMPUTE_DML: "dml",
    }
    return mapping.get(compute, "cpu")


def runtime_hint(compute: str) -> str:
    """Human-readable note about which runtime a back-end needs."""
    hints = {
        COMPUTE_CPU: "bundled with the application",
        COMPUTE_CUDA: "optional GPU runtime (download from Settings; ~1.8 GB)",
        COMPUTE_DML: "not available for this application",
    }
    return hints.get(compute, hints[COMPUTE_CPU])


def env_is_packaged() -> bool:
    """True when running from a frozen (PyInstaller) build."""
    return bool(getattr(sys, "frozen", False))
