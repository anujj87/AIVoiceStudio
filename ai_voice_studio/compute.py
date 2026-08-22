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
import subprocess
import sys
from typing import Dict, List

from . import runtime
from .constants import COMPUTE_AUTO, COMPUTE_CPU, COMPUTE_CUDA, COMPUTE_DML

log = logging.getLogger(__name__)


def _has_cuda() -> bool:
    # The GPU back-end needs the optional CUDA runtime (downloaded from
    # Settings) plus an NVIDIA GPU with a usable driver.
    if not runtime.is_installed():
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def _has_npu() -> bool:
    # No DirectML/NPU runtime is bundled or downloadable for sherpa-onnx.
    return False


def detect(force: bool = False) -> Dict[str, bool]:
    """Return {'cpu': bool, 'cuda': bool, 'dml': bool} (cached)."""
    cache = getattr(detect, "_cache", None)
    if cache is not None and not force:
        return cache
    result = {
        COMPUTE_CPU: True,
        COMPUTE_CUDA: _has_cuda(),
        COMPUTE_DML: _has_npu(),
    }
    detect._cache = result  # type: ignore[attr-defined]
    return result


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
