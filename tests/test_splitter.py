"""Tests for the audio-file splitter (SPEC 3.4 naming rules)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.constants import (
    MODE_ALL_HEADINGS,
    MODE_H1_ONLY,
    MODE_PAGE_ONLY,
    MODE_PAGE_WITH_H1,
)
from ai_voice_studio.documents.parsers import Block, Document
from ai_voice_studio.documents.splitter import split_document


def doc_from(blocks):
    return Document(blocks=blocks, source="test", format="test")


class TestPageOnly(unittest.TestCase):
    def test_names_pages(self):
        doc = doc_from(
            [
                Block(kind="text", text="Page one content.", page=0),
                Block(kind="text", text="Page two content.", page=1),
                Block(kind="text", text="Page three content.", page=2),
            ]
        )
        segs = split_document(doc, MODE_PAGE_ONLY)
        self.assertEqual([s.title for s in segs], ["01 page 1", "02 page 2", "03 page 3"])
        self.assertEqual([s.text for s in segs],
                         ["Page one content.", "Page two content.", "Page three content."])


class TestPageWithH1(unittest.TestCase):
    def test_h1_splits_page(self):
        doc = doc_from(
            [
                Block(kind="text", text="Intro text.", page=0),
                Block(kind="heading", level=1, text="My Heading", page=0),
                Block(kind="text", text="Body under heading.", page=0),
                Block(kind="text", text="Next page text.", page=1),
            ]
        )
        segs = split_document(doc, MODE_PAGE_WITH_H1)
        titles = [s.title for s in segs]
        self.assertEqual(titles, ["01 page 1", "02 My Heading", "03 page 2"])
        self.assertEqual(segs[0].text, "Intro text.")
        self.assertIn("My Heading", segs[1].text)
        self.assertIn("Body under heading.", segs[1].text)

    def test_page_without_h1_stays_single(self):
        doc = doc_from(
            [
                Block(kind="text", text="A.", page=0),
                Block(kind="text", text="B.", page=0),
            ]
        )
        segs = split_document(doc, MODE_PAGE_WITH_H1)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].title, "01 page 1")


class TestH1Only(unittest.TestCase):
    def test_groups_by_h1(self):
        doc = doc_from(
            [
                Block(kind="heading", level=1, text="First", page=0),
                Block(kind="text", text="content one", page=0),
                Block(kind="heading", level=1, text="Second", page=1),
                Block(kind="text", text="content two", page=1),
            ]
        )
        segs = split_document(doc, MODE_H1_ONLY)
        self.assertEqual([s.title for s in segs], ["01 First", "02 Second"])
        self.assertIn("content one", segs[0].text)

    def test_leading_text_becomes_page_one(self):
        doc = doc_from(
            [
                Block(kind="text", text="Preamble.", page=0),
                Block(kind="heading", level=1, text="Chapter", page=0),
                Block(kind="text", text="Body.", page=0),
            ]
        )
        segs = split_document(doc, MODE_H1_ONLY)
        self.assertEqual([s.title for s in segs], ["01 page 1", "02 Chapter"])


class TestAllHeadings(unittest.TestCase):
    def test_every_heading(self):
        doc = doc_from(
            [
                Block(kind="text", text="plain", page=0),
                Block(kind="heading", level=1, text="One", page=0),
                Block(kind="heading", level=2, text="Two", page=0),
                Block(kind="heading", level=3, text="Three", page=1),
            ]
        )
        segs = split_document(doc, MODE_ALL_HEADINGS)
        self.assertEqual([s.title for s in segs], ["01 One", "02 Two", "03 Three"])
        self.assertEqual(segs[0].text, "One")


class TestUniqueTitles(unittest.TestCase):
    def test_duplicate_headings_get_suffix(self):
        doc = doc_from(
            [
                Block(kind="heading", level=1, text="Same", page=0),
                Block(kind="heading", level=1, text="Same", page=1),
            ]
        )
        segs = split_document(doc, MODE_H1_ONLY)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].title, "01 Same")
        self.assertNotEqual(segs[1].title, segs[0].title)


class TestFilenames(unittest.TestCase):
    def test_sanitized_names(self):
        doc = doc_from(
            [Block(kind="heading", level=1, text='Weird: "name" / with\\slash', page=0)]
        )
        segs = split_document(doc, MODE_ALL_HEADINGS)
        filename = segs[0].filename
        self.assertNotIn('"', filename)
        self.assertNotIn("/", filename)
        self.assertNotIn("\\", filename)


if __name__ == "__main__":
    unittest.main()
