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
            'clip-end="npt=1.250s" id="aud_0001" />',
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
        # Full-text docs are XHTML 1.0 Strict with bidirectional links:
        # every text block links back to its SMIL <text> element.
        self.assertIn('XHTML 1.0 Strict', text)
        self.assertIn('<h1 id="seg_0001"><a href="0001.smil#txt_0001">Chapter 1</a></h1>', text)
        self.assertIn(
            '<p id="p_0001_001"><a href="0001.smil#txt_0002">'
            'This is the text for chapter 1.</a></p>',
            text,
        )

        with open(os.path.join(daisy_dir, "0001.smil"), encoding="utf-8") as fh:
            smil = fh.read()
        # One <par> per text block (heading + paragraph), audio sliced
        # proportionally, last clip ending exactly at the segment duration.
        self.assertEqual(smil.count("<par "), 2)
        self.assertIn('<text src="0001.html#seg_0001" id="txt_0001" />', smil)
        self.assertIn('<text src="0001.html#p_0001_001" id="txt_0002" />', smil)
        self.assertIn('clip-begin="npt=0.000s"', smil)
        self.assertIn('clip-end="npt=1.250s"', smil)

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


# ---------------------------------------------------------------------------
# DAISY 3 (Z39.86-2005)
# ---------------------------------------------------------------------------
from ai_voice_studio.constants import DAISY3_OUTPUT_DIR_NAME  # noqa: E402
from ai_voice_studio.documents.daisy3_builder import (  # noqa: E402
    build_daisy3_book,
    export_daisy3_zip,
)


class TestDaisy3Builder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_segments(self, count=3, done=True):
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

    def test_build_daisy3_audio_text(self):
        """DAISY 3 books get DTBook, NCX, OPF, SMILs and copied audio."""
        segments = self._make_segments()
        opf_path = build_daisy3_book(
            output_dir=self.tmpdir,
            project_name="Test Book 3",
            segments=segments,
            audio_format="wav",
            language="en",
            publisher="Test Publisher",
        )
        daisy_dir = os.path.join(self.tmpdir, DAISY3_OUTPUT_DIR_NAME)
        self.assertTrue(os.path.isfile(opf_path))
        for name in ("book.xml", "ncx.xml", "package.opf"):
            self.assertTrue(os.path.isfile(os.path.join(daisy_dir, name)), name)
        daisy_files = sorted(os.listdir(daisy_dir))
        for i in range(1, len(segments) + 1):
            self.assertIn(f"{i:04d}.smil", daisy_files)
            self.assertIn(f"aud{i:04d}.wav", daisy_files)

        # DTBook: official doctype, frontmatter, bodymatter with level1/h1/p
        with open(os.path.join(daisy_dir, "book.xml"), encoding="utf-8") as fh:
            dtbook = fh.read()
        self.assertIn('PUBLIC "-//NISO//DTD dtbook 2005-3//EN"', dtbook)
        self.assertIn("<frontmatter>", dtbook)
        self.assertIn("<doctitle id=\"doctitle\">Test Book 3</doctitle>", dtbook)
        self.assertIn("<bodymatter>", dtbook)
        self.assertIn("<rearmatter>", dtbook)
        self.assertIn('<h1 id="seg_0001">Chapter 1</h1>', dtbook)
        self.assertIn('<p id="p_0001_001">This is the text for chapter 1.</p>', dtbook)

        # NCX: navMap with one navPoint per segment pointing into the SMILs
        with open(os.path.join(daisy_dir, "ncx.xml"), encoding="utf-8") as fh:
            ncx = fh.read()
        self.assertIn('xmlns="http://www.daisy.org/z3986/2005/ncx/"', ncx)
        self.assertIn('<navPoint id="nav_0001" playOrder="1">', ncx)
        self.assertIn('<content src="0001.smil#txt_0001" />', ncx)

        # SMIL 2.0: dtb namespace, clipBegin/clipEnd, text -> book.xml ids.
        # The <audio> sits directly inside <par> (no wrapping <seq>, which
        # DAISY 3 players do not play through).
        with open(os.path.join(daisy_dir, "0001.smil"), encoding="utf-8") as fh:
            smil = fh.read()
        self.assertIn('PUBLIC "-//NISO//DTD xml-smil 2005-1//EN"', smil)
        self.assertIn('xmlns:dtb="http://www.daisy.org/z3986/2005/dtbook/"', smil)
        self.assertIn('<text src="book.xml#seg_0001" id="txt_0001" />', smil)
        self.assertIn('clipBegin="0.000s"', smil)
        self.assertIn('clipEnd="1.250s"', smil)
        self.assertNotIn("<seq>\n          <audio", smil)
        # Audio directly inside <par>, sliced proportionally across the two
        # text blocks (heading + paragraph); the final clip carries a tail
        # beyond the measured duration so players never stop early.
        self.assertIn(
            '<audio src="aud0001.wav" clipBegin="0.000s" clipEnd="0.225s" />',
            smil,
        )
        self.assertIn(
            '<audio src="aud0001.wav" clipBegin="0.225s" clipEnd="1.250s" />',
            smil,
        )

        # OPF: spine references every SMIL, dtb:totalTime present
        with open(opf_path, encoding="utf-8") as fh:
            opf = fh.read()
        self.assertIn('<spine toc="ncx">', opf)
        self.assertIn('<itemref idref="smil_0001"/>', opf)
        self.assertIn('name="dtb:totalTime"', opf)

    def test_build_daisy3_no_ready_segments(self):
        """No recorded segments -> no book, empty path returned."""
        segments = self._make_segments(done=False)
        result = build_daisy3_book(
            output_dir=self.tmpdir, project_name="Empty", segments=segments)
        self.assertEqual(result, "")
        self.assertFalse(os.path.isdir(
            os.path.join(self.tmpdir, DAISY3_OUTPUT_DIR_NAME)))

    def test_export_daisy3_zip(self):
        """The ZIP contains the book under one top-level folder."""
        build_daisy3_book(
            output_dir=self.tmpdir,
            project_name="Test Book 3",
            segments=self._make_segments(count=2),
            audio_format="wav",
        )
        zip_path = os.path.join(self.tmpdir, "export3.zip")
        result = export_daisy3_zip(self.tmpdir, zip_path, book_folder="Book3")
        self.assertTrue(os.path.isfile(result))
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
        self.assertIn("Book3/package.opf", names)
        self.assertIn("Book3/book.xml", names)
        self.assertIn("Book3/ncx.xml", names)
        self.assertIn("Book3/0001.smil", names)
        self.assertIn("Book3/aud0001.wav", names)
        for name in names:
            self.assertTrue(name.startswith("Book3/"), name)

    def test_export_daisy3_without_book_raises(self):
        zip_path = os.path.join(self.tmpdir, "empty3.zip")
        with self.assertRaises(FileNotFoundError):
            export_daisy3_zip(self.tmpdir, zip_path)

    def test_build_daisy3_with_images(self):
        """Images from a DOCX source land in DAISY3/images and the DTBook."""
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        docx_path = os.path.join(self.tmpdir, "source.docx")
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/image1.png", png)
        segments = self._make_segments(count=1)
        opf_path = build_daisy3_book(
            output_dir=self.tmpdir,
            project_name="Image Book 3",
            segments=segments,
            audio_format="wav",
            source_file=docx_path,
        )
        daisy_dir = os.path.join(self.tmpdir, DAISY3_OUTPUT_DIR_NAME)
        self.assertTrue(os.path.isfile(
            os.path.join(daisy_dir, "images", "img0001.png")))
        with open(os.path.join(daisy_dir, "book.xml"), encoding="utf-8") as fh:
            dtbook = fh.read()
        self.assertIn('<img id="img_0001" src="images/img0001.png"', dtbook)
        self.assertIn("<imggroup", dtbook)
        with open(opf_path, encoding="utf-8") as fh:
            opf = fh.read()
        self.assertIn('href="images/img0001.png" media-type="image/png"', opf)


class TestDaisyMeta(unittest.TestCase):
    """DAISY book information entered in the wizard lands in the book metadata."""

    META = {
        "title": "My Custom Title",
        "creator": "Jane Writer",
        "date": "2026-09-01",
        "subject": "Testing, accessibility",
        "narrator": "Sam Voice",
        "producer": "Acme Productions",
        "show_software": True,
    }

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_segments(self, count=1):
        segments = []
        for i in range(1, count + 1):
            saved = f"chapter{i:02d}.wav"
            _write_wav(os.path.join(self.tmpdir, saved), seconds=i)
            segments.append({
                "index": i,
                "title": f"Chapter {i}",
                "text": f"Text for chapter {i}.",
                "saved": saved,
                "status": "done",
            })
        return segments

    def test_daisy2_meta_in_ncc_and_opf(self):
        """DAISY 2.02: title, creator, date, subject, narrator, producer land
        in ncc.html; the generator line is written when show_software is on."""
        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Fallback Title",
            segments=self._make_segments(),
            audio_format="wav",
            include_text=False,
            meta=dict(self.META),
        )
        with open(ncc_path, encoding="utf-8") as fh:
            ncc = fh.read()
        self.assertIn('dc:title" content="My Custom Title"', ncc)
        self.assertIn('dc:creator" content="Jane Writer"', ncc)
        self.assertIn('dc:date" content="2026-09-01"', ncc)
        self.assertIn('dc:subject" content="Testing, accessibility"', ncc)
        self.assertIn('ncc:narrator" content="Sam Voice"', ncc)
        self.assertIn('ncc:producer" content="Acme Productions"', ncc)
        self.assertIn('ncc:generator" content="AI Voice Studio"', ncc)
        # The NCC <title> and heading carry the custom title too.
        self.assertIn("<title>My Custom Title</title>", ncc)
        self.assertIn(">My Custom Title</a></h1>", ncc)

        daisy_dir = os.path.join(self.tmpdir, DAISY_OUTPUT_DIR_NAME)
        with open(os.path.join(daisy_dir, "package.opf"), encoding="utf-8") as fh:
            opf = fh.read()
        self.assertIn("<dc:creator>Jane Writer</dc:creator>", opf)
        self.assertIn("<dc:subject>Testing, accessibility</dc:subject>", opf)

    def test_daisy2_meta_software_hidden(self):
        """With show_software=False no generator/software metadata is written."""
        meta = dict(self.META, show_software=False, producer="")
        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="T",
            segments=self._make_segments(),
            audio_format="wav",
            meta=meta,
        )
        with open(ncc_path, encoding="utf-8") as fh:
            ncc = fh.read()
        self.assertNotIn("ncc:generator", ncc)
        self.assertNotIn("ncc:producer", ncc)

    def test_daisy3_meta_in_dtbook_ncx_opf(self):
        """DAISY 3: metadata lands in DTBook head, NCX docTitle/docAuthor and
        OPF; show_software adds "AI Voice Studio" as second producer."""
        opf_path = build_daisy3_book(
            output_dir=self.tmpdir,
            project_name="Fallback Title",
            segments=self._make_segments(),
            audio_format="wav",
            meta=dict(self.META),
        )
        daisy_dir = os.path.join(self.tmpdir, DAISY3_OUTPUT_DIR_NAME)
        with open(os.path.join(daisy_dir, "book.xml"), encoding="utf-8") as fh:
            dtbook = fh.read()
        self.assertIn("<doctitle id=\"doctitle\">My Custom Title</doctitle>", dtbook)
        self.assertIn("<docauthor id=\"docauthor\">Jane Writer</docauthor>", dtbook)
        self.assertIn('<meta name="dc:Creator" content="Jane Writer" />', dtbook)
        self.assertIn('<meta name="dc:Date" content="2026-09-01" />', dtbook)
        self.assertIn('<meta name="dc:Subject" content="Testing, accessibility" />', dtbook)
        self.assertIn('<meta name="dtb:narrator" content="Sam Voice" />', dtbook)
        self.assertIn('<meta name="dtb:producer" content="Acme Productions" />', dtbook)
        self.assertIn('<meta name="dtb:producer" content="AI Voice Studio" />', dtbook)

        with open(os.path.join(daisy_dir, "ncx.xml"), encoding="utf-8") as fh:
            ncx = fh.read()
        self.assertIn("<docTitle><text>My Custom Title</text></docTitle>", ncx)
        self.assertIn("<docAuthor><text>Jane Writer</text></docAuthor>", ncx)

        with open(opf_path, encoding="utf-8") as fh:
            opf = fh.read()
        self.assertIn("<dc:title>My Custom Title</dc:title>", opf)
        self.assertIn("<dc:creator>Jane Writer</dc:creator>", opf)
        self.assertIn("<dc:date>2026-09-01</dc:date>", opf)
        self.assertIn("<dc:subject>Testing, accessibility</dc:subject>", opf)
        self.assertIn('<meta name="dtb:narrator" content="Sam Voice" />', opf)
        self.assertIn('<meta name="dtb:producer" content="Acme Productions" />', opf)
        self.assertIn('<meta name="dtb:producer" content="AI Voice Studio" />', opf)

    def test_daisy3_meta_software_hidden(self):
        """With show_software=False only the user producer is written."""
        opf_path = build_daisy3_book(
            output_dir=self.tmpdir,
            project_name="T",
            segments=self._make_segments(),
            audio_format="wav",
            meta=dict(self.META, show_software=False),
        )
        daisy_dir = os.path.join(self.tmpdir, DAISY3_OUTPUT_DIR_NAME)
        with open(opf_path, encoding="utf-8") as fh:
            opf = fh.read()
        self.assertIn('dtb:producer" content="Acme Productions"', opf)
        self.assertNotIn('dtb:producer" content="AI Voice Studio"', opf)

    def test_daisy_meta_defaults(self):
        """Empty meta falls back: title from project name, date from today."""
        ncc_path = build_daisy_book(
            output_dir=self.tmpdir,
            project_name="Fallback Book",
            segments=self._make_segments(),
            audio_format="wav",
            meta={},
        )
        with open(ncc_path, encoding="utf-8") as fh:
            ncc = fh.read()
        self.assertIn('dc:title" content="Fallback Book"', ncc)
        self.assertNotIn("dc:creator", ncc)
        self.assertNotIn("ncc:narrator", ncc)
        # show_software defaults True -> generator present.
        self.assertIn('ncc:generator" content="AI Voice Studio"', ncc)


if __name__ == "__main__":
    unittest.main()
