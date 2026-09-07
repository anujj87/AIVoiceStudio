"""Tests for the DAISY 2.02 book builder and ZIP export."""

import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.constants import DAISY_OUTPUT_DIR_NAME
from ai_voice_studio.documents.daisy_builder import build_daisy_book, export_daisy_zip


def _write_wav(path: str, seconds: float = 1.0, rate: int = 8000) -> None:
    """Write a small valid WAV of ``seconds`` seconds of silence."""
    n = int(rate * seconds)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n)


class TestDaisyBuilder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_segments(self, count=3, done=True):
        """Create segment data plus real WAV audio files in the project dir."""
        segments = []
        for i in range(1, count + 1):
            saved = f"chapter{i:02d}.wav"
            if done:
                _write_wav(os.path.join(self.tmpdir, saved), seconds=i)
            segments.append({
                "index": i,
                "title": f"Chapter {i}",
                "text": f"This is the text for chapter {i}.",
                "saved": saved,
                "status": "done" if done else "pending",
            })
        return segments

    def test_build_daisy_audio_only(self):
        """Audio-only books get NCC, OPF, SMILs, master.smil and copied audio."""
        segments = self._make_segments()
        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Test Book",
            segments=segments,
            audio_format="wav",
            include_text=False,
            language="en",
            publisher="Test Publisher",
        )
        daisy_dir = os.path.join(self.tmpdir, DAISY_OUTPUT_DIR_NAME)
        self.assertTrue(os.path.isfile(ncc_path))
        self.assertTrue(os.path.isfile(os.path.join(daisy_dir, "ncc.html")))
        self.assertTrue(os.path.isfile(os.path.join(daisy_dir, "package.opf")))
        self.assertTrue(os.path.isfile(os.path.join(daisy_dir, "master.smil")))

        # Flat fileset: one digit-only SMIL per segment, siblings of ncc.html
        daisy_files = sorted(os.listdir(daisy_dir))
        self.assertIn("ncc.html", daisy_files)
        self.assertIn("master.smil", daisy_files)
        self.assertIn("package.opf", daisy_files)
        for i in range(1, len(segments) + 1):
            self.assertIn(f"{i:04d}.smil", daisy_files)
            self.assertIn(f"{i:04d}.html", daisy_files)
            self.assertIn(f"aud{i:04d}.wav", daisy_files)
        with open(os.path.join(daisy_dir, "0001.smil"), encoding="utf-8") as fh:
            smil = fh.read()
        # SMIL 1.0 doctype (DAISY 2.02 is SMIL 1.0, not SMIL 2.0)
        self.assertIn('PUBLIC "-//W3C//DTD SMIL 1.0//EN"', smil)
        # layout region required for text rendering
        self.assertIn('<region id="txtView" />', smil)
        # audio is a sibling reference, wrapped in a nested <seq> like the
        # official samples
        self.assertIn(
            '<audio src="aud0001.wav" clip-begin="npt=0.000s" '
            'clip-end="npt=1.000s" id="aud_0001" />',
            smil,
        )
        self.assertIn("<seq>\n          <audio", smil)
        # SMIL carries the <text> sync target even for audio-only books
        self.assertIn('<text src="0001.html#seg_0001" id="txt_0001" />', smil)

        # Audio copied into DAISY/ under digit-only names (flat)
        for i in range(1, len(segments) + 1):
            self.assertTrue(os.path.isfile(
                os.path.join(daisy_dir, f"aud{i:04d}.wav")))

        # Mandatory NCC metadata
        with open(ncc_path, encoding="utf-8") as fh:
            ncc = fh.read()
        self.assertIn('<meta name="dc:format" content="Daisy 2.02" />', ncc)
        self.assertIn('<meta name="ncc:multimediaType" content="audioNcc" />', ncc)
        self.assertIn('<meta name="ncc:setInfo" content="1 of 1" />', ncc)
        self.assertIn('<meta name="ncc:tocItems" content="4" />', ncc)
        self.assertIn('<meta name="dc:language" content="en" scheme="ISO 639" />', ncc)
        self.assertIn('<meta name="dc:publisher" content="Test Publisher" />', ncc)
        # totalTime for 1+2+3 = 6 seconds
        self.assertIn('<meta name="ncc:totalTime" content="00:00:06" scheme="hh:mm:ss" />', ncc)
        # title heading + chapters link into SMIL <text> ids (flat, sibling
        # references), not <body> ids
        self.assertIn('<h1 class="title" id="ncc_0000">', ncc)
        self.assertIn('<a href="0001.smil#txt_0001">Chapter 1</a>', ncc)
        self.assertIn('<a href="0003.smil#txt_0003">Chapter 3</a>', ncc)

    def test_build_daisy_audio_text(self):
        """Audio+text books carry text/ documents and SMIL text sync."""
        segments = self._make_segments()
        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Test Book",
            segments=segments,
            audio_format="wav",
            include_text=True,
            language="hi",
            publisher="Hindi Press",
        )
        daisy_dir = os.path.join(self.tmpdir, DAISY_OUTPUT_DIR_NAME)
        with open(ncc_path, encoding="utf-8") as fh:
            ncc = fh.read()
        self.assertIn('<meta name="ncc:multimediaType" content="audioFullText" />', ncc)

        text_dir = daisy_dir
        self.assertTrue(os.path.isfile(os.path.join(text_dir, "0001.html")))
        with open(os.path.join(text_dir, "0001.html"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn('<h1 id="seg_0001">Chapter 1</h1>', text)
        self.assertIn("<p>This is the text for chapter 1.</p>", text)

        with open(os.path.join(daisy_dir, "0001.smil"), encoding="utf-8") as fh:
            smil = fh.read()
        self.assertIn('<text src="0001.html#seg_0001" id="txt_0001" />', smil)
        self.assertIn('clip-begin="npt=0.000s" clip-end="npt=1.000s"', smil)

        # OPF lists the text documents (flat hrefs)
        with open(os.path.join(daisy_dir, "package.opf"), encoding="utf-8") as fh:
            opf = fh.read()
        self.assertIn('href="0001.html" media-type="application/xhtml+xml"', opf)
        self.assertIn('href="aud0001.wav" media-type="audio/wav"', opf)

    def test_build_daisy_no_ready_segments(self):
        """No recorded segments -> no book, empty path returned."""
        segments = self._make_segments(done=False)
        result = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Empty Book",
            segments=segments,
        )
        self.assertEqual(result, "")
        self.assertFalse(os.path.isdir(
            os.path.join(self.tmpdir, DAISY_OUTPUT_DIR_NAME)))


class TestExportDaisyZip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_book(self):
        segments = []
        for i in range(1, 3):
            saved = f"chapter{i:02d}.wav"
            _write_wav(os.path.join(self.tmpdir, saved), seconds=1)
            segments.append({
                "index": i,
                "title": f"Chapter {i}",
                "text": f"Text {i}",
                "saved": saved,
                "status": "done",
            })
        build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Test Book",
            segments=segments,
            audio_format="wav",
            include_text=True,
            language="en",
            publisher="Pub",
        )

    def test_export_creates_zip_with_daisy_files(self):
        """The ZIP contains the book under one top-level folder."""
        self._build_book()
        zip_path = os.path.join(self.tmpdir, "export.zip")
        result = export_daisy_zip(self.tmpdir, zip_path, book_folder="TestBook")
        self.assertTrue(os.path.isfile(result))
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
        self.assertIn("TestBook/ncc.html", names)
        self.assertIn("TestBook/package.opf", names)
        self.assertIn("TestBook/master.smil", names)
        self.assertIn("TestBook/0001.smil", names)
        self.assertIn("TestBook/aud0001.wav", names)
        self.assertIn("TestBook/0001.html", names)
        # Nothing outside the top-level folder
        for name in names:
            self.assertTrue(name.startswith("TestBook/"), name)

    def test_export_without_book_raises(self):
        """Exporting before any DAISY book was built raises FileNotFoundError."""
        zip_path = os.path.join(self.tmpdir, "empty.zip")
        with self.assertRaises(FileNotFoundError):
            export_daisy_zip(self.tmpdir, zip_path)

    def test_export_default_book_folder(self):
        """Without book_folder, the project folder name is used."""
        project = os.path.join(self.tmpdir, "My Nice Book")
        os.makedirs(project)
        segments = [{
            "index": 1, "title": "Ch1", "text": "x",
            "saved": "ch1.wav", "status": "done",
        }]
        _write_wav(os.path.join(project, "ch1.wav"), seconds=1)
        build_daisy_book(
            output_dir=project, project_name="My Nice Book", segments=segments,
        )
        zip_path = os.path.join(self.tmpdir, "out.zip")
        export_daisy_zip(project, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        self.assertIn("My Nice Book/ncc.html", names)


if __name__ == "__main__":
    unittest.main()
