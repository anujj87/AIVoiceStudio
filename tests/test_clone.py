"""Voice-cloning (XTTS v2) tests: worker protocol, voice registration and
the engine wrapper. The heavy coqui-tts runtime is never installed in tests;
the worker is exercised with the "ping" command and its error paths."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import clone  # noqa: E402
from ai_voice_studio.clone import CloneEngineError, XtTsCloneEngine  # noqa: E402
from ai_voice_studio.tts.models import ModelStore  # noqa: E402


def _write_wav(path: str, seconds: float = 4.0, rate: int = 22050) -> None:
    frames = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))


class WorkerProtocolTest(unittest.TestCase):
    def _spawn(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Isolate the worker's user-data lookup from the real profile.
        env = dict(os.environ)
        env["APPDATA"] = tempfile.mkdtemp(prefix="aivs_worker_")
        return subprocess.Popen(
            [sys.executable, "-m", "ai_voice_studio.clone.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=root,
            env=env,
        )

    def test_ping_reports_engine_missing_but_stays_healthy(self):
        proc = self._spawn()
        try:
            proc.stdin.write('{"cmd": "ping"}\n')
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            self.assertTrue(resp["ok"])
            # coqui-tts is not installed in the test environment.
            self.assertFalse(resp.get("engine"))
            self.assertIn("not installed", resp.get("error", "").lower())
        finally:
            proc.stdin.write('{"cmd": "quit"}\n')
            proc.stdin.flush()
            proc.wait(timeout=10)

    def test_synthesize_without_engine_returns_error(self):
        proc = self._spawn()
        try:
            proc.stdin.write(json.dumps({
                "cmd": "synthesize",
                "text": "hello",
                "sample_wav": "missing.wav",
                "language": "en",
            }) + "\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            self.assertFalse(resp["ok"])
            self.assertIn("not installed", resp.get("error", "").lower())
        finally:
            proc.stdin.write('{"cmd": "quit"}\n')
            proc.stdin.flush()
            proc.wait(timeout=10)

    def test_unknown_command_returns_error(self):
        proc = self._spawn()
        try:
            proc.stdin.write('{"cmd": "frobnicate"}\n')
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            self.assertFalse(resp["ok"])
            self.assertIn("Unknown command", resp["error"])
        finally:
            proc.stdin.write('{"cmd": "quit"}\n')
            proc.stdin.flush()
            proc.wait(timeout=10)


class CloneVoiceRegistrationTest(unittest.TestCase):
    def test_create_cloned_voice_registers_xtts_engine(self):
        fd, state = tempfile.mkstemp(prefix="aivs_clone_", suffix=".json")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(state) and os.remove(state))
        store = ModelStore(state_file=state)
        with tempfile.TemporaryDirectory() as sample_dir, \
                tempfile.TemporaryDirectory() as models_dir:
            sample = os.path.join(sample_dir, "voice.wav")
            _write_wav(sample)
            with mock.patch("ai_voice_studio.paths.models_dir",
                            return_value=models_dir):
                entry = clone.create_cloned_voice("Mum", sample, "hi", store)
            self.assertEqual(entry["name"], "Mum")
            self.assertTrue(os.path.isfile(entry["sample"]))
            voices = store.custom_voices()
            self.assertEqual(len(voices), 1)
            self.assertEqual(voices[0]["engine"], "xtts")
            self.assertEqual(voices[0]["kind"], "xtts")
            self.assertEqual(voices[0]["language"], "hi")
            self.assertTrue(os.path.isfile(voices[0]["sample"]))

    def test_missing_sample_raises(self):
        entry = {
            "engine": "xtts", "voice": "ghost", "dir": "C:/nope",
            "sample": "C:/nope/sample.wav", "xtts_lang": "en",
        }
        with self.assertRaises(CloneEngineError):
            XtTsCloneEngine(entry)

    def test_languages_cover_xtts_set(self):
        codes = [code for code, _ in clone.XTTS_LANGUAGES]
        for expected in ("en", "es", "fr", "de", "it", "pt", "pl", "tr",
                         "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"):
            self.assertIn(expected, codes)
        self.assertEqual(len(codes), 17)


if __name__ == "__main__":
    unittest.main()
