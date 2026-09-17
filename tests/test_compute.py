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
    compute._nvidia_cache = None


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

    def test_force_re_runs_the_driver_check(self):
        # A cached "no NVIDIA card" answer must not outlive the card itself:
        # force=True re-probes instead of returning the cached False.
        with mock.patch.object(compute.subprocess, "run",
                               return_value=_FakeResult(1, "")):
            self.assertFalse(compute.has_nvidia_gpu(force=True))
        self.assertFalse(compute.has_nvidia_gpu(), "the answer is cached")
        with mock.patch.object(compute.subprocess, "run",
                               return_value=_FakeResult(0, "NVIDIA RTX 4090\n")):
            self.assertTrue(compute.has_nvidia_gpu(force=True))
            self.assertTrue(compute.has_nvidia_gpu())

    def test_detect_force_leaves_no_stale_gpu_answer_behind(self):
        # detect(force=True) must re-run the driver check as well, otherwise a
        # probe that ran earlier in the session would hide a GPU that is there.
        for found, expected in ((False, False), (True, True)):
            result = _FakeResult(0, "NVIDIA RTX 4090\n") if found \
                else _FakeResult(1, "")
            with mock.patch("ai_voice_studio.compute.runtime.is_installed",
                            return_value=True), \
                    mock.patch.object(compute.subprocess, "run",
                                      return_value=result):
                self.assertEqual(compute.detect(force=True)["cuda"], expected)


class CpuThreadsTest(unittest.TestCase):
    """A CPU run uses 80-95% of the machine's logical CPUs."""

    def test_a_normal_machine_uses_the_top_of_the_band(self):
        # As much CPU as the rule allows (the upper end of 80-95%), never
        # every core: 16 -> 15, 32 -> 30, 8 -> 7.
        for cpus in (8, 12, 16, 24, 32, 64):
            threads = compute.cpu_threads(cpus)
            self.assertGreaterEqual(threads / cpus, compute.CPU_THREAD_MIN_RATIO,
                                    cpus)
            self.assertLessEqual(threads / cpus, compute.CPU_THREAD_MAX_RATIO,
                                 cpus)
            self.assertLess(threads, cpus, cpus)
            self.assertEqual(
                threads,
                int(cpus * compute.CPU_THREAD_MAX_RATIO),
                cpus,
            )

    def test_a_single_core_machine_still_gets_a_thread(self):
        self.assertEqual(compute.cpu_threads(1), 1)
        self.assertEqual(compute.cpu_threads(0), 1)
        self.assertEqual(compute.cpu_threads(-4), 1)

    def test_tiny_machines_keep_a_core_free(self):
        # The 80-95% band is empty on two or three cores, so one core is left
        # for the desktop instead of pinning the whole machine.
        self.assertEqual(compute.cpu_threads(2), 1)
        self.assertEqual(compute.cpu_threads(3), 2)
        self.assertLessEqual(compute.cpu_threads(4), 3)

    def test_the_machine_is_detected_when_no_count_is_given(self):
        with mock.patch.object(compute.os, "cpu_count", return_value=16):
            self.assertEqual(compute.cpu_threads(), compute.cpu_threads(16))

    def test_the_gpu_device_is_not_tied_to_a_specific_card(self):
        # Any CUDA-capable NVIDIA card is accepted: the first (usually only)
        # device is used, with no model or VRAM requirement.
        self.assertEqual(compute.cuda_device_index(), 0)


if __name__ == "__main__":
    unittest.main()
