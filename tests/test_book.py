"""Tests for the bundled HTML book (docs/book).

The book is opened from the Help menu (F1) and copied into the frozen build by
``packaging/ai_voice_studio.spec``.  When the folder is missing or a chapter is
renamed, nothing fails until a user presses F1 - the application starts, the
installer builds, and the book silently disappears.  These tests turn that into
a build-time failure:

* the index exists and every link in its table of contents resolves,
* every chapter is a complete document with the shared layout,
* cross-chapter links do not point at files that do not exist,
* the table of contents names every chapter that is actually on disk (and vice
  versa), so a new chapter cannot be added without listing it,
* the frozen build's own packaging entry still points at the book.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOOK = os.path.join(ROOT, "docs", "book")
INDEX = os.path.join(BOOK, "index.html")
CHAPTERS = os.path.join(BOOK, "chapters")

_CHAPTER_LINK = re.compile(r'href="chapters/([^"#]+\.html)"')
_RELATIVE_LINK = re.compile(r'href="([^"#:]+\.html)"')


class BookLayoutTest(unittest.TestCase):
    """The book must be a complete, internally consistent document set."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(INDEX):
            raise AssertionError(
                f"the book index is missing: {INDEX} (Help menu -> F1 opens it)"
            )
        with open(INDEX, encoding="utf-8") as handle:
            cls.index = handle.read()
        cls.chapter_names = sorted(
            name for name in os.listdir(CHAPTERS) if name.endswith(".html")
        )

    def test_index_lists_chapters(self):
        listed = list(dict.fromkeys(_CHAPTER_LINK.findall(self.index)))
        self.assertTrue(listed, "the index links no chapters")
        missing = [name for name in listed if name not in self.chapter_names]
        self.assertEqual(missing, [], f"index links missing chapters: {missing}")

    def test_every_chapter_is_listed_in_the_index(self):
        listed = set(_CHAPTER_LINK.findall(self.index))
        unlisted = [name for name in self.chapter_names if name not in listed]
        self.assertEqual(
            unlisted, [],
            f"chapters exist but are not in the table of contents: {unlisted}",
        )

    def test_every_chapter_is_a_complete_document(self):
        for name in self.chapter_names:
            with self.subTest(chapter=name), open(
                os.path.join(CHAPTERS, name), encoding="utf-8"
            ) as handle:
                text = handle.read()
            self.assertIn("<!DOCTYPE html>", text)
            self.assertIn('class="book-container"', text)
            self.assertIn('class="main-content"', text)
            self.assertIn('href="../css/style.css"', text)
            self.assertIn("<h1>", text)
            self.assertTrue(text.rstrip().endswith("</html>"), f"{name} is truncated")

    def test_chapter_links_resolve(self):
        for name in self.chapter_names:
            with open(os.path.join(CHAPTERS, name), encoding="utf-8") as handle:
                text = handle.read()
            for target in set(_RELATIVE_LINK.findall(text)):
                if target == "../css/style.css":
                    continue
                with self.subTest(chapter=name, target=target):
                    self.assertTrue(
                        os.path.isfile(os.path.join(CHAPTERS, target)),
                        f"{name} links {target}, which does not exist",
                    )

    def test_stylesheet_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(BOOK, "css", "style.css")))

    def test_latest_part_is_present(self):
        """The chapters that document the current subsystems must ship."""
        for name in (
            "ch76_voicelab_engines.html",
            "ch77_engine_environments.html",
            "ch78_compute_choice.html",
            "ch79_audio_io_shim.html",
            "ch80_start_selected_recording.html",
            "ch81_terms_dialog.html",
            "ch82_licensing.html",
        ):
            with self.subTest(chapter=name):
                self.assertIn(name, self.chapter_names)


class BookPackagingTest(unittest.TestCase):
    """The frozen build and the Help menu must keep pointing at the book."""

    def test_spec_ships_the_book(self):
        spec = os.path.join(ROOT, "packaging", "ai_voice_studio.spec")
        with open(spec, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("docs/book/index.html", text)
        self.assertIn("docs/book/chapters", text)

    def test_help_menu_opens_the_book(self):
        path = os.path.join(ROOT, "ai_voice_studio", "gui", "main_frame.py")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("docs", text)
        self.assertIn("book", text)
        self.assertIn("index.html", text)


if __name__ == "__main__":
    unittest.main()
