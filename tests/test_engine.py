"""Tests for TTS engine audio-sample conversion (silent-WAV fix)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.tts.engine import samples_to_int16  # noqa: E402


class TestSamplesToInt16(unittest.TestCase):
    def test_float_normalized_samples_are_scaled(self):
        # sherpa-onnx returns floats in [-1, 1]; they must be scaled, not
        # truncated (truncation was producing silent WAV files).
        raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        out = samples_to_int16(raw)
        self.assertEqual(out.dtype, np.int16)
        self.assertEqual(out.tolist(), [-32767, -16383, 0, 16383, 32767])

    def test_int16_passthrough(self):
        raw = np.array([-32768, 0, 32767], dtype=np.int16)
        out = samples_to_int16(raw)
        self.assertEqual(out.dtype, np.int16)
        self.assertEqual(out.tolist(), [-32768, 0, 32767])

    def test_out_of_range_floats_are_clipped(self):
        raw = np.array([1.5, -1.5, 2.0], dtype=np.float32)
        out = samples_to_int16(raw)
        self.assertEqual(out.tolist(), [32767, -32768, 32767])


if __name__ == "__main__":
    unittest.main()
