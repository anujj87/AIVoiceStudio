"""Tests for document parsers (txt/md/html/docx/pdf)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.documents.parsers import (
    parse_clipboard,
    parse_document,
    parse_epub,
    parse_html,
    parse_markdown,
    parse_txt,
)


class TestTxt(unittest.TestCase):
    def test_paragraphs(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as fh:
            fh.write("First paragraph here.\n\nSecond paragraph.\n")
            path = fh.name
        try:
            doc = parse_txt(path)
            self.assertEqual(len(doc.blocks), 2)
            self.assertTrue(doc.blocks[0].text.startswith("First"))
        finally:
            os.remove(path)


class TestMarkdown(unittest.TestCase):
    def test_headings_and_text(self):
        md = (
            "# Chapter One\n\nSome **bold** text with [a link](https://x.y).\n\n"
            "## Section Two\n\n- item one\n- item two\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(md)
            path = fh.name
        try:
            doc = parse_markdown(path)
            kinds = [(b.kind, b.level) for b in doc.blocks]
            self.assertIn(("heading", 1), kinds)
            self.assertIn(("heading", 2), kinds)
            text = doc.text
            self.assertIn("bold", text)
            self.assertNotIn("**", text)
            self.assertIn("a link", text)
            self.assertNotIn("[a link]", text)
        finally:
            os.remove(path)


class TestHtml(unittest.TestCase):
    def test_headings(self):
        html = (
            "<html><body><h1>Title</h1><p>Some <b>bold</b> text.</p>"
            "<h2>Sub</h2><ul><li>One</li></ul></body></html>"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(html)
            path = fh.name
        try:
            doc = parse_html(path)
            self.assertTrue(any(b.kind == "heading" and b.level == 1 and b.text == "Title"
                                for b in doc.blocks))
            self.assertTrue(any(b.kind == "heading" and b.level == 2 and b.text == "Sub"
                                for b in doc.blocks))
        finally:
            os.remove(path)


class TestClipboard(unittest.TestCase):
    def test_blocks(self):
        doc = parse_clipboard("Line one.\n\nLine two.")
        self.assertEqual(len(doc.blocks), 2)


class TestEpub(unittest.TestCase):
    """Regression: EPUB parsing must not fail with "name 'ET' is not defined".

    The OPF/container helpers use xml.etree.ElementTree at module level, so the
    import must be module-wide (a function-local import inside parse_epub does
    not cover them).  Also covers Calibre-style heading classes and chapter
    charsets other than UTF-8.
    """

    def _make_epub(self, path: str, chapters: dict) -> None:
        """Build an EPUB. ``chapters`` maps file name -> bytes/str body html."""
        import zipfile

        container = (
            '<?xml version="1.0"?>'
            '<container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            "<rootfiles>"
            '<rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>"
        )
        items = "".join(
            f'<item id="c{i}" href="{name}" '
            'media-type="application/xhtml+xml"/>'
            for i, name in enumerate(chapters)
        )
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        opf = (
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
            'unique-identifier="bookid">'
            "<metadata/>"
            f"<manifest>{items}</manifest>"
            f"<spine>{spine}</spine>"
            "</package>"
        )
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("META-INF/container.xml", container)
            zf.writestr("OEBPS/content.opf", opf)
            for name, body in chapters.items():
                html = (
                    b"<html><body>" + (body if isinstance(body, bytes)
                                       else body.encode("utf-8")) + b"</body></html>"
                )
                zf.writestr(f"OEBPS/{name}", html)

    def test_parse_epub(self):
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as fh:
            path = fh.name
        try:
            self._make_epub(
                path,
                {"chapter1.xhtml": (
                    "<h1>Chapter One</h1><p>Hello EPUB world.</p>"
                    "<h2>Section</h2><p>More text.</p>"
                )},
            )
            doc = parse_document(path)
            self.assertEqual(doc.format, "epub")
            self.assertTrue(
                any(
                    b.kind == "heading" and b.level == 1 and b.text == "Chapter One"
                    for b in doc.blocks
                )
            )
            self.assertIn("Hello EPUB world.", doc.text)
        finally:
            os.remove(path)

    def test_parse_epub_direct(self):
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as fh:
            path = fh.name
        try:
            self._make_epub(
                path,
                {"c.xhtml": "<h1>One</h1><p>Text.</p>"},
            )
            doc = parse_epub(path)
            self.assertTrue(doc.blocks)
            self.assertTrue(any(b.kind == "heading" for b in doc.blocks))
        finally:
            os.remove(path)

    def test_calibre_style_headings(self):
        """EPUBs whose chapters use styled <p> blocks instead of h1-h6 must
        still yield headings with levels (cn/ct -> 1, cst -> 2, fmh -> 1)."""
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as fh:
            path = fh.name
        try:
            self._make_epub(
                path,
                {
                    "intro.xhtml": (
                        '<p class="fmh">Introduction</p>'
                        "<p>My bags were packed.</p>"
                    ),
                    "ch1.xhtml": (
                        '<p class="cn"><a href="x">Chapter 1</a></p>'
                        '<p class="ct">THE NETHERLANDS</p>'
                        '<p class="cst">Happiness Is a Number</p>'
                        "<p>It is a fact of human nature.</p>"
                    ),
                },
            )
            doc = parse_epub(path)
            heads = [(b.level, b.text) for b in doc.blocks if b.kind == "heading"]
            self.assertEqual(
                heads,
                [
                    (1, "Introduction"),
                    (1, "Chapter 1"),
                    (1, "THE NETHERLANDS"),
                    (2, "Happiness Is a Number"),
                ],
            )
            # Body text after the styled headings is still kept as plain text.
            self.assertIn("It is a fact of human nature.", doc.text)
        finally:
            os.remove(path)

    def test_cp1252_chapter_charset(self):
        """Chapter bytes in Windows-1252 (curly quotes) must not become U+FFFD."""
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as fh:
            path = fh.name
        try:
            chapter = b"<h1>Intro</h1><p>Caf\xe9s aren\x92t closed \x93yet\x94.</p>"
            self._make_epub(path, {"c.xhtml": chapter})
            doc = parse_epub(path)
            self.assertIn("aren\u2019t closed \u201cyet\u201d.", doc.text)
            self.assertIn("Caf\xe9s", doc.text)  # é preserved
            self.assertNotIn("\ufffd", doc.text)
        finally:
            os.remove(path)


class TestDispatcher(unittest.TestCase):
    def test_unsupported(self):
        with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as fh:
            path = fh.name
        try:
            from ai_voice_studio.documents.parsers import ParseError

            with self.assertRaises(ParseError):
                parse_document(path)
        finally:
            os.remove(path)

    def test_pdf_requires_pypdf(self):
        # pypdf is installed in the test env; just verify a fake pdf raises
        # a ParseError (not a crash).
        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as fh:
            fh.write(b"%PDF-1.4 junk")
            path = fh.name
        try:
            from ai_voice_studio.documents.parsers import ParseError

            with self.assertRaises(ParseError):
                parse_document(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
