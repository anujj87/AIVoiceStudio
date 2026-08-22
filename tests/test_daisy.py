"""Tests for DAISY 2.02 book builder and ZIP export."""

import os
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.documents.daisy_builder import build_daisy_book, export_daisy_zip


class TestDaisyBuilder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_segments(self, count=3, include_audio=True):
        """Create fake segment data for testing."""
        segments = []
        for i in range(1, count + 1):
            saved = f"chapter{i:02d}.wav" if include_audio else None
            segments.append({
                "index": i,
                "title": f"Chapter {i}",
                "text": f"This is the text for chapter {i}.",
                "saved": saved,
                "status": "done" if include_audio else "pending",
            })
        return segments

    def test_build_daisy_audio_only(self):
        """Test building a DAISY book in audio-only mode."""
        segments = self._make_segments(include_audio=True)
        # Create dummy audio files
        audio_dir = os.path.join(self.tmpdir, "audio")
        os.makedirs(audio_dir)
        for seg in segments:
            open(os.path.join(audio_dir, seg["saved"]), "w").close()

        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Test Book",
            segments=segments,
            audio_format="wav",
            include_text=False,
            language="en",
            publisher="Test Publisher",
        )
        self.assertTrue(os.path.isfile(ncc_path))
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "package.opf")))
        # Check SMIL files exist
        for seg in segments:
            smil_name = f"{seg['title']}.smil"
            self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, smil_name)))

    def test_build_daisy_audio_text(self):
        """Test building a DAISY book in audio+text mode."""
        segments = self._make_segments(include_audio=True)
        audio_dir = os.path.join(self.tmpdir, "audio")
        os.makedirs(audio_dir)
        for seg in segments:
            open(os.path.join(audio_dir, seg["saved"]), "w").close()

        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Test Book",
            segments=segments,
            audio_format="mp3",
            include_text=True,
            language="hi",
            publisher="Hindi Press",
        )
        self.assertTrue(os.path.isfile(ncc_path))
        # Check text files exist for audio+text mode
        for seg in segments:
            txt_name = f"{seg['title']}.txt"
            self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, txt_name)))

    def test_build_daisy_no_ready_segments(self):
        """Test that building with no ready segments returns empty path."""
        segments = [{"index": 1, "title": "Ch1", "text": "text", "saved": None, "status": "pending"}]
        result = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Empty Book",
            segments=segments,
        )
        self.assertEqual(result, "")


class TestExportDaisyZip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_creates_zip_with_daisy_files(self):
        """Test that export_daisy_zip creates a valid ZIP with DAISY files."""
        segments = [
            {"index": 1, "title": "Ch1", "text": "Hello world", "saved": "ch1.wav", "status": "done"},
        ]
        # Create dummy files
        open(os.path.join(self.tmpdir, "ncc.html"), "w").close()
        open(os.path.join(self.tmpdir, "package.opf"), "w").close()
        audio_dir = os.path.join(self.tmpdir, "audio")
        os.makedirs(audio_dir)
        open(os.path.join(audio_dir, "ch1.wav"), "w").close()

        zip_path = os.path.join(self.tmpdir, "export.zip")
        result = export_daisy_zip(self.tmpdir, zip_path)

        self.assertTrue(os.path.isfile(result))
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            self.assertIn("ncc.html", names)
            self.assertIn("package.opf", names)
            self.assertIn("audio/ch1.wav", names)

    def test_export_with_smil_and_text(self):
        """Test ZIP includes SMIL and text files when present."""
        open(os.path.join(self.tmpdir, "ncc.html"), "w").close()
        open(os.path.join(self.tmpdir, "package.opf"), "w").close()
        smil_dir = os.path.join(self.tmpdir, "smil")
        os.makedirs(smil_dir)
        open(os.path.join(smil_dir, "ch1.smil"), "w").close()
        open(os.path.join(self.tmpdir, "ch1.txt"), "w").close()

        zip_path = os.path.join(self.tmpdir, "export2.zip")
        result = export_daisy_zip(self.tmpdir, zip_path)

        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            self.assertIn("ncc.html", names)
            self.assertIn("smil/ch1.smil", names)
            self.assertIn("ch1.txt", names)


if __name__ == "__main__":
    unittest.main()
