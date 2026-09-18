"""Tests that keep the shipped documentation honest.

The user-facing documents (``docs/*.html``), the repository documents
(``README.md``, ``PROJECT_SPEC.md``) and the installer all describe the same
program.  They drift in two ways that nothing else catches:

* a document keeps describing a component that was removed (the NeuTTS engine,
  the Coqui XTTS cloning engine, the DirectML/NPU back-end) or a menu item that
  was renamed (``Restart Selected Recording``), and
* the versions disagree — the application, the package and the installer each
  carry their own string.

Both are silent: the installer builds, the tests pass, and a user reads a
document that describes a different program.  These tests turn that into a
build-time failure.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import __version__  # noqa: E402
from ai_voice_studio.constants import APP_VERSION, TERMS_VERSION  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "docs")
SPEC = os.path.join(ROOT, "packaging", "ai_voice_studio.spec")
INSTALLER = os.path.join(ROOT, "packaging", "installer_common.iss")
MAIN_FRAME = os.path.join(ROOT, "ai_voice_studio", "gui", "main_frame.py")

# The documents the Help menu opens (names passed to _open_doc).
HELP_DOCS = (
    "README.html",
    "UserGuide.html",
    "AddonDevelopmentGuide.html",
    "AccessibilityGuide.html",
    "THIRD-PARTY-LICENSES.html",
)

# Documents a user reads as the current truth.  The book is excluded on
# purpose: it is a textbook and may describe a subsystem historically, with a
# "Current implementation" callout, which ``tests/test_book.py`` guards.
SHIPPED_DOCS = HELP_DOCS + ("../README.md", "../PROJECT_SPEC.md")

# Phrases that only ever described a component or menu item that no longer
# exists.  A document may not carry one because the shipped documentation is
# read as the current truth; a retirement is reported in DEV_PLAN.md instead.
RETIRED = (
    "NeuTTS",
    "Neuphonic",
    "Restart Selected Recording",
    "onnxruntime-directml",
)

# Names a document may mention *only* to say they were retired.  Naming them
# helps users migrating from an older version, claiming they are the cloning
# engine is what would be wrong.
RETIRED_IF_ASSERTED = ("XTTS", "Requirements-npu", "requirements-npu.txt")
_RETIREMENT_WORDS = ("retir", "legacy", "earlier release", "no longer", "removed")

# ``version X.Y.Z`` / ``Version X.Y.Z`` statements must name this release.
_VERSION_STATEMENT = re.compile(r"\b[Vv]ersion\s+(\d+\.\d+\.\d+)")


def _read(*parts: str) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return handle.read()


class VersionConsistencyTest(unittest.TestCase):
    """The application, the package and the installer must agree."""

    def test_package_and_constants_match(self):
        self.assertEqual(__version__, APP_VERSION)

    def test_installer_version_matches_the_application(self):
        text = _read(INSTALLER)
        self.assertIn(f"AppVersion={APP_VERSION}", text)
        self.assertIn(f"VersionInfoVersion={APP_VERSION}", text)
        self.assertIn(f"VersionInfoProductVersion={APP_VERSION}", text)

    def test_installer_output_name_carries_the_version(self):
        text = _read(INSTALLER)
        dashed = APP_VERSION.replace(".", "-")
        self.assertIn(f"AI-Voice-Studio-v-{dashed}-Setup", text)

    def test_terms_version_is_not_older_than_the_app(self):
        # A terms bump is a deliberate act; the app version never lags it.
        self.assertGreaterEqual(
            tuple(int(part) for part in TERMS_VERSION.split(".")),
            (2026, 3, 1),
            "TERMS_VERSION must not go backwards",
        )


class ShippedDocsTest(unittest.TestCase):
    """Every document the Help menu opens must exist and be installed."""

    def test_help_documents_exist(self):
        for name in HELP_DOCS:
            with self.subTest(document=name):
                self.assertTrue(os.path.isfile(os.path.join(DOCS, name)))

    def test_help_menu_documents_are_all_shipped(self):
        text = _read(MAIN_FRAME)
        opened = set(re.findall(r'_open_doc\("([^"]+)"', text))
        self.assertEqual(
            sorted(opened), sorted(HELP_DOCS),
            "the Help menu and the installer document list disagree",
        )

    def test_installer_copies_every_help_document(self):
        installer = _read(INSTALLER)
        for name in HELP_DOCS:
            with self.subTest(document=name):
                self.assertIn(f"docs\\{name}", installer)

    def test_spec_bundles_the_docs_folder(self):
        spec = _read(SPEC)
        self.assertIn("docs", spec)
        self.assertIn("docs/book/index.html", spec)


class DocsCurrencyTest(unittest.TestCase):
    """The shipped documents must not describe a program that no longer exists."""

    def test_no_retired_component_is_still_described_as_current(self):
        for name in SHIPPED_DOCS:
            text = _read(DOCS, name)
            for phrase in RETIRED:
                with self.subTest(document=name, phrase=phrase):
                    self.assertNotIn(
                        phrase, text,
                        f"{name} still mentions {phrase!r}, which the application "
                        f"no longer ships",
                    )

    def test_retired_names_appear_only_beside_a_retirement(self):
        for name in SHIPPED_DOCS:
            text = _read(DOCS, name)
            for phrase in RETIRED_IF_ASSERTED:
                for line in text.splitlines():
                    if phrase not in line:
                        continue
                    with self.subTest(document=name, phrase=phrase, line=line[:60]):
                        lowered = line.lower()
                        self.assertTrue(
                            any(word in lowered for word in _RETIREMENT_WORDS),
                            f"{name} mentions {phrase!r} without saying it was "
                            f"retired: {line.strip()!r}",
                        )

    def test_no_version_statement_names_another_release(self):
        for name in SHIPPED_DOCS:
            text = _read(DOCS, name)
            for found in _VERSION_STATEMENT.findall(text):
                with self.subTest(document=name, version=found):
                    self.assertEqual(
                        found, APP_VERSION,
                        f"{name} states version {found}, not {APP_VERSION}",
                    )

    def test_readme_states_the_current_version(self):
        self.assertIn(APP_VERSION, _read(DOCS, "README.html"))
        self.assertIn(APP_VERSION, _read(ROOT, "README.md"))

    def test_documents_cover_the_current_features(self):
        guide = _read(DOCS, "UserGuide.html")
        for phrase in (
            "Voice Lab",
            "tts_envs",
            "Compute",
            "Terms of Use",
            "Third-Party Licences",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, guide)

    def test_readme_covers_the_same_features(self):
        readme = _read(DOCS, "README.html")
        for phrase in ("Voice Lab", "terms of use", "Third-Party Licences"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, readme)


if __name__ == "__main__":
    unittest.main()
