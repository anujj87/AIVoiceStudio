"""Tests for project lifecycle helpers (Edit menu actions)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import project  # noqa: E402
from ai_voice_studio.constants import MODE_PAGE_WITH_H1  # noqa: E402


def _make_project(tmp: str, segments: int) -> str:
    pdir = os.path.join(tmp, "proj")
    project.create_project(
        pdir, "Test project", "doc.txt", MODE_PAGE_WITH_H1,
        [{"index": i + 1, "title": f"Segment {i + 1}", "text": f"Text {i + 1}."}
         for i in range(segments)],
    )
    return pdir


class TestProjectLifecycle(unittest.TestCase):
    def test_reset_recordings_deletes_audio_and_resets_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_project(tmp, 3)
            saved = []
            for i, title in enumerate(("First", "Second", "Third"), start=1):
                p = os.path.join(pdir, f"{title}.wav")
                with open(p, "wb") as fh:
                    fh.write(b"\x00" * 44)
                project.mark_segment_done(pdir, i, p)
                saved.append(p)
            stray = os.path.join(pdir, "old.mp3")
            with open(stray, "wb") as fh:
                fh.write(b"\x00")
            data = project.load_project(pdir)
            self.assertEqual({s["status"] for s in data["segments"]}, {"done"})

            removed = project.reset_recordings(pdir)

            self.assertEqual(removed, 4)  # 3 segments + 1 stray audio file
            for p in saved + [stray]:
                self.assertFalse(os.path.exists(p))
            data = project.load_project(pdir)
            for seg in data["segments"]:
                self.assertEqual(seg["status"], "pending")
                self.assertIsNone(seg["saved"])

    def test_remove_recording_resets_only_that_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_project(tmp, 2)
            p1 = os.path.join(pdir, "First.wav")
            p2 = os.path.join(pdir, "Second.wav")
            for p in (p1, p2):
                with open(p, "wb") as fh:
                    fh.write(b"\x00" * 44)
            project.mark_segment_done(pdir, 1, p1)
            project.mark_segment_done(pdir, 2, p2)

            project.remove_recording(pdir, "First.wav")

            self.assertFalse(os.path.exists(p1))
            self.assertTrue(os.path.exists(p2))
            data = project.load_project(pdir)
            statuses = {s["index"]: s["status"] for s in data["segments"]}
            self.assertEqual(statuses, {1: "pending", 2: "done"})

    def test_remove_project_deletes_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_project(tmp, 2)
            self.assertTrue(os.path.isdir(pdir))
            project.remove_project(pdir)
            self.assertFalse(os.path.exists(pdir))

    def test_recorded_files_lists_audio_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_project(tmp, 2)
            # Files exist even though segment metadata was never updated
            # (this is the scenario behind the "Restart selected recording"
            # picker showing "no recorded files" for a fully-recorded project).
            names = ("Intro.wav", "Chapter 2.mp3", "notes.txt", "meta.flac")
            for name in names:
                with open(os.path.join(pdir, name), "wb") as fh:
                    fh.write(b"\x00" * 100)

            files = project.recorded_files(pdir)

            self.assertEqual(files, ["Chapter 2.mp3", "Intro.wav", "meta.flac"])

    def test_recorded_files_empty_when_folder_missing(self):
        self.assertEqual(project.recorded_files(r"Z:\\does\\not\\exist"), [])


if __name__ == "__main__":
    unittest.main()
