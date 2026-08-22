"""Tests for compute back-end detection (GPU option visibility)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import compute  # noqa: E402


def _reset_cache():
    compute.detect._cache = None  # type: ignore[attr-defined]


class _FakeResult:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


class CudaDetectionTest(unittest.TestCase):
    def setUp(self):
        _reset_cache()

    def _patch(self, runtime_installed: bool, nvidia: bool):
        patches = [
            mock.patch("ai_voice_studio.compute.runtime.is_installed",
                       return_value=runtime_installed),
            mock.patch.object(
                compute.subprocess, "run",
                return_value=_FakeResult(0, "NVIDIA GeForce RTX 3060\n") if nvidia
                else _FakeResult(1, ""),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_cuda_offered_with_runtime_and_nvidia_gpu(self):
        self._patch(runtime_installed=True, nvidia=True)
        det = compute.detect(force=True)
        self.assertTrue(det["cuda"])
        self.assertIn("cuda", compute.available_compute())
        self.assertEqual(compute.resolve_compute("auto"), "cuda")

    def test_cuda_not_offered_without_downloaded_runtime(self):
        # Even with an NVIDIA GPU, no GPU option until the optional runtime
        # is downloaded from Settings -> Compute.
        self._patch(runtime_installed=False, nvidia=True)
        det = compute.detect(force=True)
        self.assertFalse(det["cuda"])
        self.assertNotIn("cuda", compute.available_compute())

    def test_cuda_not_offered_without_nvidia_gpu(self):
        self._patch(runtime_installed=True, nvidia=False)
        det = compute.detect(force=True)
        self.assertFalse(det["cuda"])

    def test_cpu_always_available(self):
        self._patch(runtime_installed=False, nvidia=False)
        det = compute.detect(force=True)
        self.assertTrue(det["cpu"])
        self.assertIn("cpu", compute.available_compute())

    def test_dml_never_offered(self):
        self._patch(runtime_installed=True, nvidia=True)
        det = compute.detect(force=True)
        self.assertFalse(det["dml"])

    def test_provider_mapping(self):
        self.assertEqual(compute.provider_for("cuda"), "cuda")
        self.assertEqual(compute.provider_for("cpu"), "cpu")
        self.assertEqual(compute.provider_for("auto"), "cpu")


if __name__ == "__main__":
    unittest.main()
