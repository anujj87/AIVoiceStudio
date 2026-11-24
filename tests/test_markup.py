"""Expression markup in the text pipeline (``tts.markup``, ``process_punctuation``).

Kitten TTS 2.0 reads a leading ``[emotion]`` tag, inline ``<event>`` tags and
``(((emphasis)))`` spans as director's notes rather than words to pronounce, so
the punctuation modes - which expand every mark into a spoken word or delete it
- must not touch them.  The vocabulary is a mirror of the engine's own
(``kittenml.kittentts2.text``), and the last test here compares the two whenever
the engine is installed, so the mirror cannot drift unnoticed.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.constants import (
    PUNCTUATION_ALL,
    PUNCTUATION_DEFAULT,
    PUNCTUATION_MATH,
    PUNCTUATION_NONE,
)
from ai_voice_studio.documents.splitter import Segment
from ai_voice_studio.jobs.synthesizer import SynthesisWorker
from ai_voice_studio.tts import markup
from ai_voice_studio.tts.engine import process_punctuation
from ai_voice_studio.voicelab import engines as voice_lab

#: The engine that reads the markup, and one that does not.
KITTEN = "kitten_tts_2"
PLAIN = "piper"

TEXT = "[joyful] We won the grant <laugh> I can (((hardly))) believe it, really!"


def _engine_source() -> str:
    """The installed engine's ``text.py``, or ``""`` when it is not installed."""
    try:
        from ai_voice_studio.python_runtime import get_runtime

        env_dir = get_runtime(KITTEN).env_dir
    except Exception:
        return ""
    found = glob.glob(
        os.path.join(env_dir, "**", "kittenml", "kittentts2", "text.py"),
        recursive=True,
    )
    if not found:
        return ""
    with open(found[0], encoding="utf-8") as handle:
        return handle.read()


def _literal(node):
    """The value of ``frozenset({...})`` / ``re.compile("...")`` / a literal."""
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        if name == "compile":  # re.compile(<pattern>)
            return ast.literal_eval(node.args[0])
        if name in ("frozenset", "set", "tuple", "list"):
            return ast.literal_eval(node.args[0])
    return ast.literal_eval(node)


def _engine_constants(source: str) -> dict:
    """The vocabulary and patterns the installed engine declares."""
    found: dict = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in ("EXPRESSION_TAGS", "VOCAL_EVENT_TAGS",
                         "_EMPHASIS_SPAN"):
            try:
                found[target.id] = _literal(node.value)
            except Exception:
                continue
    return found


class MarkupVocabularyTest(unittest.TestCase):
    """What counts as a director's note, and for whom."""

    def test_the_three_kinds_of_span_are_recognised(self):
        self.assertTrue(markup.has_markup("[joyful] hello"))
        self.assertTrue(markup.has_markup("hello <laugh>"))
        self.assertTrue(markup.has_markup("I can (((hardly))) believe it"))
        self.assertTrue(markup.has_markup("hello [SAD] <sigh> (((well)))"))

    def test_ordinary_brackets_are_not_markup(self):
        # Neither is in the engine's vocabulary, so both stay ordinary text for
        # the mode and for the engine's own normalizer.
        for text in ("see section [3]", "x < 5 and y > 2", "[happy] cheer",
                     "(just a parenthesis)", "((( unbalanced ))"):
            self.assertFalse(markup.has_markup(text), text)

    def test_only_the_engine_that_reads_the_markup_is_protected(self):
        self.assertTrue(markup.supports(KITTEN))
        self.assertFalse(markup.supports(PLAIN))
        self.assertFalse(markup.supports(None))
        self.assertFalse(markup.supports(""))
        self.assertFalse(markup.supports("not_an_engine"))

    def test_the_registry_declares_it_for_kitten_tts_2_alone(self):
        declares = [engine_id for engine_id in voice_lab.engine_ids()
                    if voice_lab.expression_markup(engine_id)]
        self.assertEqual(declares, [KITTEN])

    def test_the_protection_of_a_plain_engine_is_a_no_op(self):
        text, spans = markup.protect_for(PLAIN, TEXT)
        self.assertEqual(text, TEXT)
        self.assertEqual(spans, [])

    def test_the_vocabulary_matches_the_installed_engine(self):
        source = _engine_source()
        if not source:
            self.skipTest("Kitten TTS 2.0 is not installed on this machine")
        constants = _engine_constants(source)
        self.assertEqual(set(constants),
                         {"EXPRESSION_TAGS", "VOCAL_EVENT_TAGS", "_EMPHASIS_SPAN"},
                         "the engine's text module changed shape")
        self.assertEqual(set(constants["EXPRESSION_TAGS"]),
                         set(markup.EXPRESSION_TAGS))
        self.assertEqual(set(constants["VOCAL_EVENT_TAGS"]),
                         set(markup.VOCAL_EVENT_TAGS))
        self.assertEqual(constants["_EMPHASIS_SPAN"],
                         markup._EMPHASIS_SPAN.pattern)


class ProtectionTest(unittest.TestCase):
    """No mode may expand, strip or reorder the markup of a markup engine."""

    def test_every_mode_keeps_the_markup(self):
        for mode in (PUNCTUATION_NONE, PUNCTUATION_MATH, PUNCTUATION_ALL):
            with self.subTest(mode=mode):
                out = process_punctuation(TEXT, mode, KITTEN)
                for span in ("[joyful]", "<laugh>", "(((hardly)))"):
                    self.assertIn(span, out)
                self.assertNotIn("\ue000", out)
                self.assertNotIn("\ue001", out)

    def test_the_markup_comes_back_exactly_as_it_was_typed(self):
        # The spans are the text the user typed, character for character; only
        # the punctuation *outside* them is expanded into words.
        self.assertEqual(
            process_punctuation(TEXT, PUNCTUATION_ALL, KITTEN),
            "[joyful] We won the grant <laugh> I can (((hardly))) believe it "
            "comma really exclamation mark",
        )

    def test_the_rest_of_the_text_is_still_processed(self):
        stripped = process_punctuation(TEXT, PUNCTUATION_NONE, KITTEN)
        self.assertNotIn(",", stripped)
        self.assertNotIn("!", stripped)
        spoken = process_punctuation(TEXT, PUNCTUATION_ALL, KITTEN)
        self.assertIn("comma", spoken)
        self.assertIn("exclamation mark", spoken)

    def test_the_default_mode_never_touches_the_text(self):
        self.assertEqual(process_punctuation(TEXT, PUNCTUATION_DEFAULT, KITTEN),
                         TEXT)

    def test_a_tag_inside_an_emphasis_span_survives(self):
        # The two-pass protection captures the emphasis span over the tag's own
        # placeholder, so the restore has to write the outer span back first -
        # otherwise a raw placeholder is left in the text.
        text = "(((really <laugh> loud)))"
        for mode in (PUNCTUATION_NONE, PUNCTUATION_ALL):
            with self.subTest(mode=mode):
                self.assertEqual(process_punctuation(text, mode, KITTEN), text)

    def test_several_spans_keep_their_order(self):
        text = "<laugh> (((one))) [sad] <sigh> (((two <gasp> three)))"
        self.assertEqual(process_punctuation(text, PUNCTUATION_NONE, KITTEN),
                         text)

    def test_a_span_that_looks_like_an_index_does_not_collide(self):
        # Eleven spans: the tenth placeholder must not match inside the first.
        text = " ".join(f"<laugh> (((span {i})))" for i in range(11))
        out = process_punctuation(text, PUNCTUATION_ALL, KITTEN)
        for i in range(11):
            self.assertIn(f"(((span {i})))", out)
        self.assertNotIn("\ue000", out)

    def test_a_non_tag_bracket_is_still_punctuation(self):
        # Only the engine's own vocabulary is protected: a citation keeps
        # behaving exactly as it did before.
        self.assertEqual(
            process_punctuation("see section [3], right?", PUNCTUATION_NONE,
                                KITTEN),
            "see section 3 right",
        )

    def test_a_plain_engine_is_completely_unchanged(self):
        for mode in (PUNCTUATION_NONE, PUNCTUATION_MATH, PUNCTUATION_ALL):
            with self.subTest(mode=mode):
                self.assertEqual(process_punctuation(TEXT, mode, PLAIN),
                                 process_punctuation(TEXT, mode))
        self.assertEqual(process_punctuation(TEXT, PUNCTUATION_NONE, PLAIN),
                         "joyful We won the grant laugh I can hardly believe it "
                         "really")

    def test_without_an_engine_nothing_changes(self):
        for mode in (PUNCTUATION_NONE, PUNCTUATION_MATH, PUNCTUATION_ALL):
            with self.subTest(mode=mode):
                self.assertEqual(process_punctuation(TEXT, mode, None),
                                 process_punctuation(TEXT, mode))


class _FakeEngine:
    sample_rate = 24000

    def __init__(self):
        self.texts: list = []

    def synthesize(self, text, sid=0, speed=1.0, pitch=1.0, volume=1.0):
        self.texts.append(text)
        return np.zeros(8, dtype=np.float32)


class TextPipelineTest(unittest.TestCase):
    """A recording hands the markup to the engine, not a mangled copy of it."""

    def _record(self, engine_id: str, punctuation: str, text: str):
        tmp = tempfile.mkdtemp(prefix="aivs_markup_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        engine = _FakeEngine()
        worker = SynthesisWorker(
            segments=[Segment(index=1, title="page 1", text=text)],
            voice_entry={"engine": engine_id},
            params={"compute": "cpu", "output_format": "wav",
                    "punctuation": punctuation},
            output_dir=tmp,
        )
        with mock.patch("ai_voice_studio.jobs.synthesizer.get_engine",
                        return_value=engine), \
                mock.patch("ai_voice_studio.jobs.synthesizer.planned_workers",
                           return_value=1), \
                mock.patch("ai_voice_studio.jobs.synthesizer.audio_output"
                           ".write_audio"):
            worker.run()
        return engine.texts

    def test_the_tags_reach_the_engine_through_a_recording(self):
        texts = self._record(KITTEN, PUNCTUATION_ALL, TEXT)
        self.assertEqual(len(texts), 1)
        for span in ("[joyful]", "<laugh>", "(((hardly)))"):
            self.assertIn(span, texts[0])
        # The mode still did its own work on the rest of the line.
        self.assertIn("exclamation mark", texts[0])

    def test_a_plain_engine_gets_the_mode_output_as_before(self):
        texts = self._record(PLAIN, PUNCTUATION_NONE, TEXT)
        self.assertEqual(len(texts), 1)
        self.assertNotIn("[", texts[0])
        self.assertNotIn("(", texts[0])

    def test_the_processed_text_log_holds_what_was_spoken(self):
        tmp = tempfile.mkdtemp(prefix="aivs_markup_log_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        engine = _FakeEngine()
        worker = SynthesisWorker(
            segments=[Segment(index=1, title="page 1", text=TEXT)],
            voice_entry={"engine": KITTEN},
            params={"compute": "cpu", "output_format": "wav",
                    "punctuation": PUNCTUATION_NONE},
            output_dir=tmp,
        )
        with mock.patch("ai_voice_studio.jobs.synthesizer.get_engine",
                        return_value=engine), \
                mock.patch("ai_voice_studio.jobs.synthesizer.planned_workers",
                           return_value=1), \
                mock.patch("ai_voice_studio.jobs.synthesizer.audio_output"
                           ".write_audio"):
            worker.run()
        with open(os.path.join(tmp, "processed_text.json"), encoding="utf-8") as fh:
            logged = json.load(fh)["processed_text"]["0"]["text"]
        self.assertIn("[joyful]", logged)
        self.assertEqual(logged, engine.texts[0])
