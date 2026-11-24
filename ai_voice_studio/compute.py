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
            check=False,
        )
        found = result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
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


def detect(force: bool = False) -> dict[str, bool]:
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
def cpu_threads(total: int | None = None) -> int:
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
    low = max(1, math.ceil(cpus * CPU_THREAD_MIN_RATIO))
    high = max(1, math.floor(cpus * CPU_THREAD_MAX_RATIO))
    if high < low:
        # Too few cores for a non-empty 80-95% band: leave one core free.
        return max(1, cpus - 1)
    # As much CPU as the rule allows: the top of the band.
    return max(1, min(high, cpus))


#: A GPU run keeps this many segments in flight (see :func:`synthesis_workers`).
#: Two is the measured sweet spot on a six-core machine; four is the ceiling,
#: because past that the CPU-side work starts competing with itself.
GPU_WORKERS_MIN = 2
GPU_WORKERS_MAX = 4

#: Roughly how many logical CPUs one segment in flight needs: the front-end
#: (text normalisation and espeak-ng phonemisation) is CPU work that sherpa-onnx
#: runs for one sentence at a time, while the card is idle waiting for it.
GPU_WORKER_CPUS = 4

#: Free video memory a GPU run wants before it keeps more than one segment in
#: flight.  Each engine holds its own copy of the model (Kokoro fp32 is ~330 MB
#: plus the ONNX Runtime arena), so a nearly-full card is left to one engine.
GPU_WORKER_MIN_FREE_VRAM_MB = 2048

def free_vram_mb() -> int | None:
    """Free video memory in MB, or ``None`` when it cannot be read.

    Driver-only, like :func:`has_nvidia_gpu`: it is a hint for how many model
    copies a GPU run may hold, never a requirement.  Deliberately *not* cached
    (unlike the driver check): how much is free changes while the application
    runs, and a recording reads it once when it starts.
    """
    value: int | None = None
    try:
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=flags,
            check=False,
        )
        first = (result.stdout or "").strip().splitlines()
        if result.returncode == 0 and first:
            value = int(float(first[0].strip()))
    except Exception:
        value = None
    return value


def synthesis_workers(compute: str, total: int | None = None) -> int:
    """How many segments the chosen back-end synthesizes at the same time.

    A CPU run already uses as much of the machine as it is allowed
    (:func:`cpu_threads`) and is strictly one segment at a time.

    A GPU run is different: sherpa-onnx prepares and runs **one sentence** per
    call, and the Kokoro implementation ignores ``max_num_sentences`` (see
    ``TtsEngine._build_config``), so the card spends most of each call waiting
    for the CPU to phonemise the next sentence.  Feeding it a second (or third)
    segment at the same time overlaps that CPU work with GPU inference, which is
    what raises utilisation and cuts wall time - measured on an RTX 3060 with a
    six-core CPU: one worker 0.185x realtime at ~15% GPU, two workers 0.127x
    realtime at ~25% GPU, while three or four only made the CPU compete with
    itself.  The audio of each segment is produced exactly as before; only how
    many segments are in flight changes.
    """
    if provider_for(compute) != COMPUTE_CUDA:
        return 1
    try:
        cpus = int(os.cpu_count() or 1)
    except (TypeError, ValueError):
        cpus = 1
    workers = max(GPU_WORKERS_MIN, min(GPU_WORKERS_MAX, cpus // GPU_WORKER_CPUS))
    free = free_vram_mb()
    if free is not None and free < GPU_WORKER_MIN_FREE_VRAM_MB:
        # Not enough room for a second copy of the model: one engine it is.
        workers = 1
    if total is not None:
        try:
            workers = min(workers, max(1, int(total)))
        except (TypeError, ValueError):
            log.debug("Ignoring a failure in synthesis_workers", exc_info=True)
    return max(1, workers)


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
            check=False,
        )
        first = (result.stdout or "").strip().splitlines()
        if result.returncode == 0 and first:
            return first[0].strip()
    except Exception:
        log.debug("Ignoring a failure in gpu_summary", exc_info=True)
    return ""


def available_compute() -> list[str]:
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

