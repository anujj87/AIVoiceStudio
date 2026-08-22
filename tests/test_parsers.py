"""Tests for document parsers (txt/md/html/docx/pdf)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.documents.parsers import (
    parse_clipboard,
    parse_document,
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
