"""Tests for the built-in Windows voice engines (SAPI5 / Windows Core).

PowerShell is never started here: the enumeration is injected through the
module cache and the WAV reader is exercised directly, so the tests stay fast
and hermetic on any machine.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.tts import windows_tts  # noqa: E402


class _CacheMixin(unittest.TestCase):
    """Isolate the module-level voice cache between tests."""

    def setUp(self):
        self._saved = (
            dict(windows_tts._cache),
            {key: list(value) for key, value in windows_tts._listeners.items()},
            set(windows_tts._loading),
        )
        windows_tts._cache.clear()
        windows_tts._listeners.clear()
        windows_tts._loading.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        cache, listeners, loading = self._saved
        windows_tts._cache.clear()
        windows_tts._cache.update(cache)
        windows_tts._listeners.clear()
        windows_tts._listeners.update(listeners)
        windows_tts._loading.clear()
        windows_tts._loading.update(loading)

    def _seed(self, engine_id: str, voices: list) -> None:
        windows_tts._cache[engine_id] = (time.time(), voices)


_ONE_VOICE = [{
    "id": "Microsoft Zira Desktop",
    "name": "Microsoft Zira Desktop",
    "gender": "Female",
    "language": "en-US",
}]


class VoiceEntryTest(_CacheMixin):
    def test_entry_carries_engine_voice_and_locale(self):
        self._seed(windows_tts.SAPI5, _ONE_VOICE)
        with mock.patch.object(windows_tts, "is_available", return_value=True):
            entries = windows_tts.voice_entries([windows_tts.SAPI5])
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["tts"], windows_tts.SAPI5)
        self.assertEqual(entry["engine"], windows_tts.SAPI5)
        self.assertEqual(entry["voice"], "Microsoft Zira Desktop")
        self.assertEqual(entry["language"], "en-US")
        self.assertEqual(entry["variant"], windows_tts.VARIANT)
        self.assertTrue(entry["builtin"])
        self.assertEqual(entry["sid"], 0)
        self.assertEqual(
            entry["voice_name"], "Microsoft Zira Desktop (Female, en-US)"
        )

    def test_voice_name_keeps_working_without_gender_or_locale(self):
        self._seed(windows_tts.WINDOWS_CORE, [{"id": "Bare", "name": "Bare"}])
        with mock.patch.object(windows_tts, "is_available", return_value=True):
            entries = windows_tts.voice_entries([windows_tts.WINDOWS_CORE])
        self.assertEqual(entries[0]["voice_name"], "Bare")
        self.assertEqual(entries[0]["language"], "en")

    def test_unavailable_platform_yields_nothing(self):
        self._seed(windows_tts.SAPI5, _ONE_VOICE)
        with mock.patch.object(windows_tts, "is_available", return_value=False):
            self.assertEqual(windows_tts.voice_entries(), [])
            self.assertEqual(windows_tts.voices(windows_tts.SAPI5), [])
            target: list = []
            windows_tts.add_installed_voices(target)
            self.assertEqual(target, [])

    def test_stale_cache_is_served_while_refreshing(self):
        windows_tts._cache[windows_tts.SAPI5] = (0.0, _ONE_VOICE)
        with mock.patch.object(windows_tts, "is_available", return_value=True), \
                mock.patch.object(windows_tts, "refresh") as refresh:
            voices = windows_tts.voices(windows_tts.SAPI5)
        refresh.assert_called_once_with(windows_tts.SAPI5)
        self.assertEqual(len(voices), 1)  # the stale list is still usable


class AddInstalledVoicesTest(_CacheMixin):
    def test_uses_the_cached_list_immediately(self):
        self._seed(windows_tts.SAPI5, _ONE_VOICE)
        target: list = []
        with mock.patch.object(windows_tts, "is_available", return_value=True):
            windows_tts.add_installed_voices(target)
        self.assertEqual([entry["voice"] for entry in target],
                         ["Microsoft Zira Desktop"])

    def test_callback_is_not_attached_to_a_cached_answer(self):
        """A callback that rebuilds the caller's list must not fire for an
        already-cached answer: that would recurse forever."""
        self._seed(windows_tts.SAPI5, _ONE_VOICE)
        self._seed(windows_tts.WINDOWS_CORE, _ONE_VOICE)
        calls: list = []
        with mock.patch.object(windows_tts, "is_available", return_value=True), \
                mock.patch.object(windows_tts, "refresh") as refresh:
            windows_tts.add_installed_voices([], lambda _v: calls.append(True))
        refresh.assert_not_called()
        self.assertEqual(calls, [])

    def test_callback_fires_for_a_pending_probe(self):
        started = threading.Event()
        release = threading.Event()

        def fake_enumerate(_engine_id):
            started.set()
            release.wait(10)
            return [{"id": "B", "name": "B", "gender": "Male", "language": "en-GB"}]

        seen: list = []
        with mock.patch.object(windows_tts, "is_available", return_value=True), \
                mock.patch.object(windows_tts, "_enumerate", side_effect=fake_enumerate):
            windows_tts.add_installed_voices([], lambda voices: seen.append(voices))
            self.assertTrue(started.wait(10))
            release.set()
            for _ in range(100):
                if seen:
                    break
                time.sleep(0.05)
        self.assertTrue(seen, "the listener never saw the enumeration")
        self.assertEqual(seen[0][0]["language"], "en-GB")


class _WavMixin(unittest.TestCase):
    def _tmp_wav(self) -> str:
        fd, path = tempfile.mkstemp(prefix="aivs_win_test_", suffix=".wav")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def _write(self, path, frames, width, rate=22050, channels=1):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            wf.writeframes(frames)


class ReadWavTest(_WavMixin):
    def test_int16_pcm_is_passed_through(self):
        path = self._tmp_wav()
        raw = np.array([-32768, 0, 32767], dtype="<i2")
        self._write(path, raw.tobytes(), 2)
        samples, rate = windows_tts._read_wav(path)
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(samples.tolist(), [-32768, 0, 32767])
        self.assertEqual(rate, 22050)

    def test_float32_pcm_is_scaled(self):
        path = self._tmp_wav()
        raw = np.array([-1.0, 0.0, 1.0], dtype="<f4")
        self._write(path, raw.tobytes(), 4, rate=16000)
        samples, rate = windows_tts._read_wav(path)
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(samples.tolist(), [-32767, 0, 32767])
        self.assertEqual(rate, 16000)

    def test_stereo_uses_the_first_channel(self):
        path = self._tmp_wav()
        stereo = np.array([1, 100, 2, 200], dtype="<i2")
        self._write(path, stereo.tobytes(), 2, channels=2)
        samples, _rate = windows_tts._read_wav(path)
        self.assertEqual(samples.tolist(), [1, 2])

    def test_empty_audio_is_an_error(self):
        path = self._tmp_wav()
        self._write(path, b"", 2)
        with self.assertRaises(RuntimeError):
            windows_tts._read_wav(path)


class EngineTest(_CacheMixin):
    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            windows_tts.WindowsVoiceEngine({"engine": "vits"})

    def test_non_windows_platform_is_rejected(self):
        with mock.patch.object(windows_tts, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                windows_tts.WindowsVoiceEngine({"engine": windows_tts.SAPI5})

    def test_voice_falls_back_to_the_speaker_id(self):
        self._seed(windows_tts.SAPI5, [
            {"id": "A", "name": "A", "gender": "Male", "language": "en-US"},
            {"id": "B", "name": "B", "gender": "Female", "language": "en-US"},
        ])
        engine = windows_tts.WindowsVoiceEngine(
            {"engine": windows_tts.SAPI5, "voice": "gone", "sid": 1}
        )
        with mock.patch.object(windows_tts, "is_available", return_value=True):
            self.assertEqual(
                engine._select_voice(windows_tts.voices(windows_tts.SAPI5)), "B"
            )

    def test_known_voice_is_used_as_is(self):
        self._seed(windows_tts.WINDOWS_CORE, [
            {"id": "Microsoft Mark", "name": "Microsoft Mark",
             "gender": "Male", "language": "en-US"},
        ])
        engine = windows_tts.WindowsVoiceEngine(
            {"engine": windows_tts.WINDOWS_CORE, "voice": "Microsoft Mark"}
        )
        with mock.patch.object(windows_tts, "is_available", return_value=True):
            self.assertEqual(
                engine._select_voice(
                    windows_tts.voices(windows_tts.WINDOWS_CORE)
                ),
                "Microsoft Mark",
            )

    def test_speed_is_clamped_per_engine(self):
        self._seed(windows_tts.SAPI5, _ONE_VOICE)
        engine = windows_tts.WindowsVoiceEngine(
            {"engine": windows_tts.SAPI5, "voice": "x", "sid": 0}
        )
        captured: dict = {}

        def fake_run(script, env=None, timeout=60):
            captured.update(env or {})
            raise RuntimeError("stop here")

        with mock.patch.object(windows_tts, "is_available", return_value=True), \
                mock.patch.object(windows_tts, "_run_powershell", side_effect=fake_run):
            with self.assertRaises(RuntimeError):
                engine.synthesize("hello", speed=9.0)
        self.assertEqual(captured["AIVS_SAPI_RATE"], "10")
        self.assertEqual(captured["AIVS_SPEAKING_RATE"], "6.0")


if __name__ == "__main__":
    unittest.main()
