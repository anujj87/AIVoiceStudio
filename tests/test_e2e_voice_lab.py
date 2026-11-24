"""Tests for the words check in ``tools/e2e_voice_lab_engine.py``.

The probe certifies a Voice Lab engine before it is documented: it speaks one
real sentence through the app's own path and reports what came back.  Its first
verdicts - length, level, and the application's own speech-versus-drone
detector - passed **every** broken take they were shown, because the detector
tells a boom from speech and not speech from *babble*: an engine that rendered
speech-shaped noise for the wrong words scored a pass (one clone engine at a
word error rate of 0.52 while it was loading without its text-embedding
weights, an ONNX export at 0.95-1.00, both reported as "OK: the engine
spoke").

The check that closes that hole transcribes the take with the application's own
Whisper large-v3 environment and compares what came back with the text the
engine was given.  These tests hold down what it must not get wrong:

* the comparison itself - Indic vowel marks stay inside their word, punctuation
  and case do not count, and the rate is the usual edit distance;
* the tri-state answer (the words came back / they did not / the check could
  not run), because a take nobody has listened to must never be a pass;
* the exit code the probe returns for each, since that is what a release or a
  developer's script reads.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import e2e_voice_lab_engine as probe


def _run(function, *args, **kwargs):
    """Call ``function`` and return what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


class WordsTest(unittest.TestCase):
    """What counts as a word, and what a word is compared as."""

    def test_indic_vowel_marks_stay_inside_the_word(self):
        """Half a Devanagari word is a combining mark, not punctuation."""
        self.assertEqual(probe._words("नमस्ते"), ["नमस्ते"])
        self.assertEqual(
            probe._words("यह वाक्य धीरे बोला गया"),
            ["यह", "वाक्य", "धीरे", "बोला", "गया"],
        )

    def test_punctuation_ends_a_word_and_is_not_part_of_it(self):
        self.assertEqual(probe._words("नमस्ते। यह!"), ["नमस्ते", "यह"])
        self.assertEqual(probe._words("Hello, world!"), ["hello", "world"])

    def test_case_and_spacing_do_not_count(self):
        self.assertEqual(probe._words("  Hello   WORLD  "), ["hello", "world"])

    def test_a_text_with_no_words_has_no_words(self):
        for text in ("", "   ", "...", "?!।"):
            with self.subTest(text=text):
                self.assertEqual(probe._words(text), [])


class WordErrorRateTest(unittest.TestCase):
    """The rate the limit is compared against."""

    def test_a_take_that_says_the_text_scores_zero(self):
        self.assertEqual(probe._word_error_rate("Hello, world!", "hello world"), 0.0)
        self.assertEqual(probe._word_error_rate("नमस्ते। यह है", "नमस्ते यह है"), 0.0)

    def test_one_word_in_four_wrong_scores_a_quarter(self):
        self.assertAlmostEqual(
            probe._word_error_rate("यह एक छोटा परीक्षण", "यह एक छोटा लक्षण"), 0.25
        )

    def test_a_take_that_says_something_else_entirely_scores_one(self):
        self.assertEqual(
            probe._word_error_rate("यह एक छोटा परीक्षण", "कुछ और बिल्कुल"), 1.0
        )

    def test_extra_words_score_above_one(self):
        self.assertGreater(probe._word_error_rate("यह है", "यह है और वह भी वहाँ"), 1.0)

    def test_an_empty_text_cannot_be_scored(self):
        self.assertNotEqual(probe._word_error_rate("", "कुछ भी"), 0.0)

    def test_the_limit_sits_between_the_engines_that_spoke_and_the_ones_that_did_not(self):
        """Measured, not chosen: 0.00-0.23 said the words, 0.52-1.00 did not."""
        self.assertGreater(probe.WORDS_LIMIT, 0.23)
        self.assertLess(probe.WORDS_LIMIT, 0.52)


class _Args:
    """The arguments ``_words_report`` reads."""

    def __init__(self, **overrides):
        self.wer = probe.WORDS_LIMIT
        self.no_words = False
        self.language = "hi"
        self.asr_accuracy = "balanced"
        for key, value in overrides.items():
            setattr(self, key, value)


class WordsReportTest(unittest.TestCase):
    """Spoke / did not speak / could not be checked."""

    def test_the_words_coming_back_is_a_pass(self):
        with mock.patch.object(probe, "_transcribe",
                               return_value=("यह एक छोटा परीक्षण", None)):
            result, printed = _run(probe._words_report, "take.wav",
                                   "यह एक छोटा परीक्षण", _Args())
        self.assertEqual(result, probe.WORDS_OK)
        self.assertIn("words      : 0.00", printed)

    def test_the_wrong_words_are_a_failure_that_says_what_was_heard(self):
        with mock.patch.object(probe, "_transcribe",
                               return_value=("कुछ और बिल्कुल", None)):
            result, printed = _run(probe._words_report, "take.wav",
                                   "यह एक छोटा परीक्षण", _Args())
        self.assertEqual(result, probe.WORDS_FAILED)
        self.assertIn("heard      : कुछ और बिल्कुल", printed)
        self.assertIn("words      : 1.00", printed)

    def test_the_limit_decides_a_take_that_is_partly_wrong(self):
        """0.25 is inside a 0.30 limit and outside a 0.10 one."""
        heard = ("यह एक छोटा लक्षण", None)
        with mock.patch.object(probe, "_transcribe", return_value=heard):
            generous, _ = _run(probe._words_report, "take.wav",
                               "यह एक छोटा परीक्षण", _Args(wer=0.30))
            strict, _ = _run(probe._words_report, "take.wav",
                             "यह एक छोटा परीक्षण", _Args(wer=0.10))
        self.assertEqual(generous, probe.WORDS_OK)
        self.assertEqual(strict, probe.WORDS_FAILED)

    def test_a_check_that_cannot_run_is_not_a_pass(self):
        with mock.patch.object(probe, "_transcribe",
                               return_value=("", "BackendMissing: no environment")):
            result, printed = _run(probe._words_report, "take.wav", "यह है", _Args())
        self.assertEqual(result, probe.WORDS_UNCHECKED)
        self.assertIn("NOT CHECKED", printed)
        self.assertIn("Settings > Manage the STT", printed)
        self.assertIn("--no-words", printed)

    def test_the_check_can_be_skipped_on_purpose(self):
        with mock.patch.object(probe, "_transcribe") as transcribe:
            result, printed = _run(probe._words_report, "take.wav", "यह है",
                                   _Args(no_words=True))
        self.assertEqual(result, probe.WORDS_SKIPPED)
        self.assertIn("not checked (--no-words)", printed)
        transcribe.assert_not_called()

    def test_a_text_with_no_words_is_not_compared(self):
        with mock.patch.object(probe, "_transcribe") as transcribe:
            result, printed = _run(probe._words_report, "take.wav", "।।।", _Args())
        self.assertEqual(result, probe.WORDS_SKIPPED)
        self.assertIn("no words", printed)
        transcribe.assert_not_called()


class _FakeEngine:
    """A Voice Lab engine that returns a tone, so no engine is installed."""

    def __init__(self, entry, device="cpu"):
        self.entry = entry
        self.device = device
        self.sample_rate = 24000

    def synthesize(self, text, speed=1.0):
        seconds = 1.0
        t = np.arange(int(self.sample_rate * seconds)) / self.sample_rate
        return (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2")


class ProbeExitCodeTest(unittest.TestCase):
    """What the probe returns, which is what a script reads."""

    def _patches(self, words):
        from ai_voice_studio import voicelab

        return (
            mock.patch.object(voicelab, "is_installed", return_value=True),
            mock.patch.object(voicelab, "installed_version", return_value="0.0.0"),
            mock.patch.object(voicelab.engines, "is_engine", return_value=True),
            mock.patch.object(voicelab.engines, "engine_name",
                              return_value="Fake engine"),
            mock.patch.object(voicelab, "VoicelabEngine", _FakeEngine),
            mock.patch.object(probe, "_judge", return_value=None),
            mock.patch.object(probe, "_words_report", return_value=words),
        )

    def _probe(self, words, tmpdir):
        argv = [
            "gptsovits",
            "--reference", __file__,
            "--transcript", "यह एक छोटा परीक्षण है",
            "--text", "यह एक छोटा परीक्षण है",
            "--out", tmpdir,
        ]
        patches = self._patches(words)
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            return _run(probe.main, argv)

    def test_it_reports_success_when_the_words_came_back(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            code, printed = self._probe(probe.WORDS_OK, folder)
        self.assertEqual(code, 0)
        self.assertIn("OK: the engine spoke the words.", printed)

    def test_it_fails_the_engine_when_the_words_did_not_come_back(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            code, printed = self._probe(probe.WORDS_FAILED, folder)
        self.assertEqual(code, 1)
        self.assertIn("did not speak the text it was given", printed)
        self.assertNotIn("OK:", printed)

    def test_a_take_that_could_not_be_checked_is_not_reported_as_a_pass(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            code, printed = self._probe(probe.WORDS_UNCHECKED, folder)
        self.assertEqual(code, 1)
        self.assertIn("not certified", printed)
        self.assertNotIn("OK:", printed)

    def test_a_skipped_check_is_reported_as_skipped_not_as_words(self):
        """--no-words is the caller's choice; the run must not claim more."""
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            code, printed = self._probe(probe.WORDS_SKIPPED, folder)
        self.assertEqual(code, 0)
        self.assertIn("OK: the engine spoke (the words were not checked).", printed)
        self.assertNotIn("spoke the words.", printed)


if __name__ == "__main__":
    unittest.main()
