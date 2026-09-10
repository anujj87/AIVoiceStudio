"""Tests for playlist file generation (audio_playlist projects)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.documents.playlist_builder import (  # noqa: E402
    build_playlists,
)


class TestPlaylistBuilder(unittest.TestCase):
    def test_writes_three_playlist_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = build_playlists(
                tmp, "My Book",
                [
                    {"index": 1, "title": "Chapter one", "status": "done",
                     "saved": "01 Chapter one.wav"},
                    {"index": 2, "title": "Chapter two", "status": "done",
                     "saved": "02 Chapter two.mp3"},
                    {"index": 3, "title": "Not recorded", "status": "pending",
                     "saved": None},
                ],
            )

            self.assertEqual(
                sorted(os.path.basename(p) for p in files),
                ["My Book.m3u8", "My Book.pls", "My Book.wpl"],
            )
            for p in files:
                self.assertTrue(os.path.isfile(p))

    def test_m3u8_lists_only_done_segments_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = build_playlists(
                tmp, "Book",
                [
                    {"index": 2, "title": "B", "status": "done", "saved": "B.wav"},
                    {"index": 1, "title": "A", "status": "done", "saved": "A.wav"},
                    {"index": 3, "title": "C", "status": "pending", "saved": None},
                ],
            )

            with open(files[0], encoding="utf-8") as fh:
                text = fh.read()
            self.assertTrue(text.startswith("#EXTM3U\n"))
            self.assertIn("#PLAYLIST:Book", text)
            self.assertIn("A.wav", text)
            self.assertIn("B.wav", text)
            self.assertNotIn("C.wav", text)
            self.assertLess(text.index("A.wav"), text.index("B.wav"))

    def test_pls_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = build_playlists(
                tmp, "Book",
                [{"index": 1, "title": "A <x>", "status": "done", "saved": "A.wav"}],
            )

            pls = next(p for p in files if p.endswith(".pls"))
            with open(pls, encoding="utf-8") as fh:
                text = fh.read()
            self.assertTrue(text.startswith("[playlist]\n"))
            self.assertIn("File1=A.wav", text)
            self.assertIn("Title1=A <x>", text)
            self.assertIn("NumberOfEntries=1", text)
            self.assertIn("Version=2", text)

    def test_wpl_escapes_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = build_playlists(
                tmp, "Book",
                [{"index": 1, "title": "A", "status": "done", "saved": 'A "x"&y.wav'}],
            )

            wpl = next(p for p in files if p.endswith(".wpl"))
            with open(wpl, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn('<media src="A &quot;x&quot;&amp;y.wav"/>', text)

    def test_returns_empty_when_nothing_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                build_playlists(
                    tmp, "Book",
                    [{"index": 1, "title": "A", "status": "pending", "saved": None}],
                ),
                [],
            )
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main()
