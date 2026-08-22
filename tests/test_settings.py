"""Tests for settings persistence and punctuation processing."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.constants import (
    PUNCTUATION_ALL,
    PUNCTUATION_DEFAULT,
    PUNCTUATION_MATH,
    PUNCTUATION_NONE,
)
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.engine import process_punctuation


class TestSettings(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            s = Settings(path)
            s.set("recording.rate", 1.25)
            s.set("theme", "dark")
            s.save()
            s2 = Settings(path)
            self.assertEqual(s2.get("recording.rate"), 1.25)
            self.assertEqual(s2.get("theme"), "dark")

    def test_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(os.path.join(tmp, "s.json"))
            self.assertEqual(s.get("recording.punctuation"), PUNCTUATION_DEFAULT)
            self.assertEqual(s.get("recording.rate"), 1.0)
            self.assertEqual(s.get("recording.volume"), 1.0)  # 100% by default
            self.assertEqual(s.get("audio_mode"), "page_with_h1")

    def test_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            s = Settings(path)
            s.set("theme", "light")
            s.reset()
            self.assertEqual(s.get("theme"), "system")

    def test_paths_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(os.path.join(tmp, "s.json"))
            self.assertEqual(s.get("paths.recordings_dir"), "")
            self.assertEqual(s.get("paths.models_dir"), "")

    def test_paths_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            s = Settings(path)
            s.set("paths.recordings_dir", "C:\\out\\recordings")
            s.set("paths.models_dir", "D:\\tts_models")
            s.save()
            s2 = Settings(path)
            self.assertEqual(s2.get("paths.recordings_dir"), "C:\\out\\recordings")
            self.assertEqual(s2.get("paths.models_dir"), "D:\\tts_models")

    def test_recent_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(os.path.join(tmp, "s.json"))
            s.add_recent_project("Book", "C:\\p\\book")
            s.add_recent_project("Doc", "C:\\p\\doc")
            recents = s.get("recent_projects")
            self.assertEqual(recents[0]["name"], "Doc")

    def test_remove_recent_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(os.path.join(tmp, "s.json"))
            s.add_recent_project("Book", "C:\\p\\book")
            s.add_recent_project("Doc", "C:\\p\\doc")
            s.remove_recent_project("C:\\p\\book")
            recents = s.get("recent_projects")
            self.assertEqual([r["name"] for r in recents], ["Doc"])


class TestPunctuation(unittest.TestCase):
    TEXT = "Hello, world! This is 3 + 4 = 7, right? (Yes.)"

    def test_default_passthrough(self):
        self.assertEqual(process_punctuation(self.TEXT, PUNCTUATION_DEFAULT), self.TEXT)

    def test_all_speaks_quotes_and_period(self):
        # User example: my name is "anujsharma". -> quote anujsharma quote dot
        out = process_punctuation('my name is "anujsharma".', PUNCTUATION_ALL)
        self.assertEqual(out, "my name is quote anujsharma quote dot")

    def test_all_speaks_code_like_tic_and_parens(self):
        # User example: print('hello') -> left paren tic hello tic right paren
        out = process_punctuation("print('hello')", PUNCTUATION_ALL)
        self.assertEqual(out, "print left paren tic hello tic right paren")

    def test_all_speaks_common_marks(self):
        out = process_punctuation(self.TEXT, PUNCTUATION_ALL)
        self.assertIn("comma", out)
        self.assertIn("exclamation mark", out)
        self.assertIn("question mark", out)
        self.assertIn("left paren", out)
        self.assertIn("right paren", out)
        self.assertNotIn(",", out)
        self.assertNotIn("!", out)

    def test_all_leaves_plain_words_untouched(self):
        out = process_punctuation("plain words only", PUNCTUATION_ALL)
        self.assertEqual(out, "plain words only")

    def test_none_strips(self):
        out = process_punctuation(self.TEXT, PUNCTUATION_NONE)
        self.assertNotIn(",", out)
        self.assertNotIn("!", out)
        self.assertNotIn("?", out)
        self.assertNotIn("(", out)

    def test_math_keeps_symbols(self):
        out = process_punctuation(self.TEXT, PUNCTUATION_MATH)
        self.assertIn("+", out)
        self.assertIn("=", out)
        self.assertNotIn(",", out)
        self.assertNotIn("?", out)


class TestProcessedTextLog(unittest.TestCase):
    def test_write_processed_text_json(self):
        from ai_voice_studio.constants import PROCESSED_TEXT_FILE_NAME
        from ai_voice_studio.jobs.synthesizer import write_processed_text

        with tempfile.TemporaryDirectory() as tmp:
            write_processed_text(tmp, {
                0: {"title": "01 page 1", "file": "01 page 1.wav", "text": "Hello dot"},
                1: {"title": "02 page 2", "file": "02 page 2.wav", "text": "Bye dot"},
            })
            path = os.path.join(tmp, PROCESSED_TEXT_FILE_NAME)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as fh:
                import json

                data = json.load(fh)
            entries = data["processed_text"]
            self.assertEqual(entries["0"]["file"], "01 page 1.wav")
            self.assertEqual(entries["0"]["text"], "Hello dot")
            self.assertEqual(entries["1"]["title"], "02 page 2")

    def test_write_processed_text_overwrites_incrementally(self):
        from ai_voice_studio.jobs.synthesizer import write_processed_text

        with tempfile.TemporaryDirectory() as tmp:
            entry = {"title": "a", "file": "a.wav", "text": "one"}
            second = {"title": "b", "file": "b.wav", "text": "two"}
            write_processed_text(tmp, {0: entry})
            write_processed_text(tmp, {0: entry, 1: second})
            path = os.path.join(tmp, "processed_text.json")
            with open(path, "r", encoding="utf-8") as fh:
                import json

                data = json.load(fh)
            self.assertEqual(len(data["processed_text"]), 2)


if __name__ == "__main__":
    unittest.main()
