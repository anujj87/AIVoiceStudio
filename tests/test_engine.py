"""Tests for the TTS engine: audio-sample conversion and the engine pool.

The pool is what lets a GPU run keep several segments in flight (the card
otherwise waits for the CPU on every single sentence); these tests pin down
who gets a pool, how many engines it holds and how it hands them out.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.tts import engine as engine_mod
from ai_voice_studio.tts.engine import (
    EngineUnavailableError,
    TtsEnginePool,
    get_engine_pool,
    is_local_sherpa_engine,
    samples_to_int16,
)


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


class _FakeEngine:
    """Stands in for a loaded sherpa-onnx engine."""

    sample_rate = 22050
    made: list = []
    busy: set = set()
    overlap = False

    def __init__(self, voice_entry, provider="cpu", num_threads=2):
        self.voice_entry = voice_entry
        self.provider = provider
        self.num_threads = num_threads
        self.texts: list[str] = []
        type(self).made.append(self)

    def synthesize(self, text, sid=0, speed=1.0, pitch=1.0, volume=1.0):
        if id(self) in _FakeEngine.busy:
            _FakeEngine.overlap = True
        _FakeEngine.busy.add(id(self))
        try:
            time.sleep(0.05)
            self.texts.append(text)
            return text
        finally:
            _FakeEngine.busy.discard(id(self))


def _reset_fake():
    _FakeEngine.made = []
    _FakeEngine.busy = set()
    _FakeEngine.overlap = False


class LocalSherpaEngineTest(unittest.TestCase):
    """Only the local sherpa-onnx voices can be pooled."""

    def test_local_models_are_recognised(self):
        for engine in ("vits", "kokoro", "kitten", "matcha", "piper"):
            with self.subTest(engine=engine):
                self.assertTrue(is_local_sherpa_engine({"engine": engine}))
        self.assertTrue(is_local_sherpa_engine({}))

    def test_voices_with_their_own_runtime_are_not_pooled(self):
        for engine in ("sapi5", "windows_core", "omnivoice", "omnivoice_server"):
            with self.subTest(engine=engine):
                self.assertFalse(is_local_sherpa_engine({"engine": engine}))

    def test_voice_lab_engines_are_not_pooled(self):
        with mock.patch("ai_voice_studio.voicelab.engines.is_engine",
                        return_value=True):
            self.assertFalse(is_local_sherpa_engine({"engine": "pocket"}))


class EnginePoolTest(unittest.TestCase):
    """The pool holds ``size`` engines and never uses one twice at once."""

    def setUp(self):
        _reset_fake()
        patcher = mock.patch.object(engine_mod, "TtsEngine", _FakeEngine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_pool_loads_the_requested_number_of_engines(self):
        pool = TtsEnginePool({"engine": "kokoro"}, provider="cuda", size=3)
        self.assertEqual(pool.size, 3)
        self.assertEqual(len(_FakeEngine.made), 3)
        self.assertEqual(pool.sample_rate, 22050)

    def test_a_pool_keeps_the_provider_and_thread_band(self):
        TtsEnginePool({"engine": "kokoro"}, provider="cuda", size=2, num_threads=7)
        for made in _FakeEngine.made:
            self.assertEqual(made.provider, "cuda")
            self.assertEqual(made.num_threads, 7)

    def test_synthesis_is_handed_to_a_free_engine(self):
        pool = TtsEnginePool({"engine": "kokoro"}, provider="cuda", size=2)
        self.assertEqual(pool.synthesize("hello"), "hello")
        self.assertEqual(pool.size, 2, "the engine went back into the pool")

    def test_two_segments_are_synthesized_at_the_same_time(self):
        pool = TtsEnginePool({"engine": "kokoro"}, provider="cuda", size=2)
        threads = [threading.Thread(target=pool.synthesize, args=(f"text {i}",))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(
            _FakeEngine.overlap,
            "one engine must never be used by two segments at once",
        )
        self.assertEqual(sorted(t for e in _FakeEngine.made for t in e.texts),
                         ["text 0", "text 1"])

    def test_a_card_that_runs_out_of_memory_keeps_the_engines_that_loaded(self):
        loaded = []

        class _Failing(_FakeEngine):
            def __init__(self, voice_entry, provider="cpu", num_threads=2):
                if len(loaded) >= 2:
                    raise EngineUnavailableError("out of memory")
                loaded.append(1)
                super().__init__(voice_entry, provider, num_threads)

        with mock.patch.object(engine_mod, "TtsEngine", _Failing):
            pool = TtsEnginePool({"engine": "kokoro"}, provider="cuda", size=4)
        self.assertEqual(pool.size, 2)

    def test_a_pool_that_cannot_load_anything_reports_the_engine_error(self):
        class _Broken(_FakeEngine):
            def __init__(self, *args, **kwargs):
                raise EngineUnavailableError("no model")

        with mock.patch.object(engine_mod, "TtsEngine", _Broken):
            with self.assertRaises(EngineUnavailableError):
                TtsEnginePool({"engine": "kokoro"}, provider="cuda", size=2)


class GetEnginePoolTest(unittest.TestCase):
    """A pool is only built when it can help."""

    def setUp(self):
        _reset_fake()
        patcher = mock.patch.object(engine_mod, "TtsEngine", _FakeEngine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_single_sized_pool_is_the_shared_cached_engine(self):
        sentinel = object()
        with mock.patch.object(engine_mod, "get_engine", return_value=sentinel):
            self.assertIs(get_engine_pool({"engine": "kokoro"}, size=1), sentinel)

    def test_a_voice_from_another_runtime_is_not_pooled(self):
        sentinel = object()
        with mock.patch.object(engine_mod, "get_engine", return_value=sentinel):
            self.assertIs(
                get_engine_pool({"engine": "omnivoice"}, size=4), sentinel
            )

    def test_a_local_model_gets_a_real_pool(self):
        pool = get_engine_pool({"engine": "kokoro"}, provider="cuda", size=2)
        self.assertIsInstance(pool, TtsEnginePool)
        self.assertEqual(pool.size, 2)


if __name__ == "__main__":
    unittest.main()
