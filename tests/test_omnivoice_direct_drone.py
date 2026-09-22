"""Drone (``no speech``) recovery in the *direct* OmniVoice engine.

The same model as the HTTP server behind another front-end: the worker returns
WAV bytes, nobody looks at them, and a bad draw used to be written straight into
the recorded segment.  These tests drive ``OmniVoiceEngine`` with a fake worker
that returns a planned take per request, so the retry and the piece repair are
exercised without loading a model.

The property that must never break is the one the user asked for: however the
engine gets there, one segment's text comes back as one continuous array.
"""

from __future__ import annotations

import unittest

import numpy as np

from ai_voice_studio import omnivoice_quality as quality
from ai_voice_studio.omnivoice import OmniVoiceEngine
from ai_voice_studio.omnivoice_server import (
    DRONE_ATTEMPTS,
    DRONE_MAX_PIECES,
    DRONE_REPAIR_ATTEMPTS,
    DRONE_REPAIR_MAX_DRAWS,
)

SAMPLE_RATE = 24_000
VOICE = {"engine": "omnivoice", "tts": "omnivoice", "voice": "alloy"}


def _speech_like(seconds: float, amplitude: int = 8000) -> np.ndarray:
    """Noisy signal: a high zero-crossing rate, like voiced speech."""
    rng = np.random.default_rng(3)
    raw = rng.standard_normal(int(seconds * SAMPLE_RATE)) * amplitude
    return np.clip(raw, -32768, 32767).astype(np.int16)


def _drone(seconds: float, freq: float = 50.0, amplitude: int = 8000,
           phase: float = 0.0) -> np.ndarray:
    """Steady low-frequency tone: the sound the failure produces."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * freq * t + phase) * amplitude
    return tone.astype(np.int16)


class _FakeWorker:
    """Stands in for ``OmniVoiceWorker``: a plan decides each take."""

    def __init__(self, plan):
        self._plan = plan
        self.requests: list = []          # (text, seed) in order

    def _ensure_proc(self) -> None:
        """``OmniVoiceEngine`` starts the worker in its constructor."""

    def synthesize(self, text, *, speed=1.0, ref_audio="", ref_text="",
                   instruct="", language=None, num_step=None,
                   guidance_scale=None, class_temperature=None,
                   duration=None, seed=None) -> np.ndarray:
        self.requests.append((text, seed))
        return self._plan(len(self.requests), text)

    def close(self) -> None:
        pass

    def texts(self) -> list:
        return [text for text, _ in self.requests]

    def seeds(self) -> list:
        return [seed for _, seed in self.requests]


def _engine(plan, **kwargs) -> tuple:
    worker = _FakeWorker(plan)
    engine = OmniVoiceEngine(VOICE, worker=worker, **kwargs)
    return engine, worker


# ---------------------------------------------------------------------------
# The retry
# ---------------------------------------------------------------------------

class DirectDroneRetryTest(unittest.TestCase):
    def test_clean_first_draw_costs_one_request(self):
        engine, worker = _engine(lambda i, text: _speech_like(3.0))
        samples = engine.synthesize("some narration text")
        self.assertEqual(len(worker.requests), 1)
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)
        self.assertEqual(engine.last_warnings, [])
        self.assertEqual(engine.last_repairs, [])

    def test_second_draw_replaces_the_drone(self):
        engine, worker = _engine(
            lambda i, text: _drone(4.0) if i == 1 else _speech_like(3.0)
        )
        engine.seed = 5
        samples = engine.synthesize("some narration text")
        self.assertEqual(len(worker.requests), 2)
        # The caller's seed is honoured first; the retry is a different draw.
        self.assertEqual(worker.seeds(), [5, 5 + 7919])
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)
        # The delivered take is clean, so there is nothing to warn about.
        self.assertEqual(engine.last_warnings, [])

    def test_an_unseeded_retry_still_changes_the_roll(self):
        engine, worker = _engine(
            lambda i, text: _drone(4.0) if i == 1 else _speech_like(3.0)
        )
        engine.synthesize("some narration text")
        self.assertEqual(worker.seeds()[0], None)
        self.assertIsNotNone(worker.seeds()[1])

    def test_voice_knobs_and_speed_reach_every_draw(self):
        seen: list = []

        def plan(i, text):
            seen.append((i, text))
            return _drone(4.0) if i == 1 else _speech_like(3.0)

        worker = _FakeWorker(plan)
        engine = OmniVoiceEngine(VOICE, worker=worker)
        engine.instruct = "female, low pitch"
        engine.ref_audio = "C:/tmp/ref.wav"
        engine.language = "hi"
        # The speed is applied per draw, and the same voice is used throughout.
        calls: list = []
        original = worker.synthesize

        def spy(text, **kwargs):
            calls.append(kwargs)
            return original(text, **kwargs)

        worker.synthesize = spy
        engine.synthesize("some narration text", speed=1.2)
        self.assertEqual(len(calls), 2)
        for kwargs in calls:
            self.assertEqual(kwargs["speed"], 1.2)
            self.assertEqual(kwargs["instruct"], "female, low pitch")
            self.assertEqual(kwargs["ref_audio"], "C:/tmp/ref.wav")
            self.assertEqual(kwargs["language"], "hi")


# ---------------------------------------------------------------------------
# The piece repair
# ---------------------------------------------------------------------------

class DirectPieceRepairTest(unittest.TestCase):
    """When every draw of a segment is a drone, it is re-recorded piecewise."""

    TEXT = " ".join(f"Sentence number {i} of this paragraph." for i in range(40))

    def setUp(self):
        def plan(i, text):
            # The whole segment drones; every smaller piece comes back clean.
            if text == self.TEXT:
                return _drone(4.0)
            return _speech_like(2.0)

        self.engine, self.worker = _engine(plan)
        self.text = self.TEXT

    def test_segment_is_rebuilt_from_sentence_sized_pieces(self):
        samples = self.engine.synthesize(self.text)
        sent = self.worker.texts()
        whole = [t for t in sent if t == self.text]
        pieces = [t for t in sent if t != self.text]

        # Every draw of the whole segment was a drone, then the pieces (one
        # draw each, because each piece came back clean on the first try).
        self.assertEqual(len(whole), DRONE_ATTEMPTS)
        self.assertEqual(len(sent), DRONE_ATTEMPTS + len(pieces))
        self.assertGreater(len(pieces), 1)
        # Every word still reaches the engine, in order, exactly once.
        self.assertEqual(" ".join(pieces).split(), self.text.split())
        self.assertEqual(set(whole), {self.text})

        # One take, and it is the repaired one.
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(samples.size, len(pieces) * 2 * SAMPLE_RATE)
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)

        # The repair is reported, with nothing left unresolved.
        self.assertEqual(len(self.engine.last_repairs), 1)
        record = self.engine.last_repairs[0]
        self.assertEqual(record["pieces"], len(pieces))
        self.assertEqual(record["unrepaired"], 0)
        self.assertEqual(record["attempts"], DRONE_ATTEMPTS)
        self.assertEqual(len(self.engine.last_warnings), 1)
        self.assertIn("re-recorded automatically", self.engine.last_warnings[0])

    def test_one_segment_stays_one_array(self):
        # Whatever happened inside, the audio is a single continuous array.
        samples = self.engine.synthesize(self.text)
        self.assertEqual(np.asarray(samples).ndim, 1)
        self.assertEqual(samples.dtype, np.int16)

    def test_a_clean_segment_clears_the_previous_report(self):
        self.engine.synthesize(self.text)
        self.assertEqual(len(self.engine.last_repairs), 1)
        self.assertEqual(len(self.engine.last_warnings), 1)
        # A clean segment afterwards resets both lists, so a warning always
        # belongs to the segment that was just recorded.
        self.engine.synthesize("A short clean sentence.")
        self.assertEqual(self.engine.last_repairs, [])
        self.assertEqual(self.engine.last_warnings, [])


class DirectRepairFailureTest(unittest.TestCase):
    """A take that cannot be repaired must not fail the recording."""

    def test_short_text_keeps_the_best_take_and_warns(self):
        engine, worker = _engine(lambda i, text: _drone(4.0))
        samples = engine.synthesize("A short paragraph of narration.")
        # The segment's draws, plus the one last draw the repair is allowed
        # when the text is too short to be split any further.
        self.assertEqual(len(worker.requests), DRONE_ATTEMPTS + 1)
        self.assertTrue(np.asarray(samples).size)
        record = engine.last_repairs[0]
        self.assertEqual(record["unrepaired"], 1)
        self.assertEqual(record["pieces"], 1)
        self.assertIn("narration", record["text"])
        # Two lines: the repair could not fix it, and the whole-segment safety
        # net sees the same drone.  Both name the text so it can be re-recorded.
        self.assertEqual(len(engine.last_warnings), 2)
        self.assertIn("check the recording", engine.last_warnings[0])
        self.assertIn("Listen to it before publishing", engine.last_warnings[1])

    def test_repair_attempts_are_bounded(self):
        engine, worker = _engine(lambda i, text: _drone(4.0))
        engine.synthesize(DirectPieceRepairTest.TEXT)
        # The same bound the server path promises: the segment's own draws plus
        # the repair budget, and never more than DRONE_MAX_PIECES at one level.
        self.assertLessEqual(
            len(worker.requests),
            DRONE_ATTEMPTS + DRONE_REPAIR_MAX_DRAWS + DRONE_MAX_PIECES,
        )
        record = engine.last_repairs[0]
        self.assertLessEqual(record["pieces"], DRONE_MAX_PIECES)
        self.assertLessEqual(record["draws"], DRONE_REPAIR_MAX_DRAWS)

    def test_a_badly_behaved_model_still_delivers_speech(self):
        # Half the draws are drones; the delivered take must still be speech.
        state = {"n": 0}

        def plan(i, text):
            state["n"] += 1
            if state["n"] % 2 == 1:
                return _drone(4.0)
            return _speech_like(3.0)

        engine, _ = _engine(plan)
        samples = engine.synthesize(
            " ".join(f"Sentence {i} here." for i in range(60))
        )
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)


# ---------------------------------------------------------------------------
# Reporting, and the shared policy
# ---------------------------------------------------------------------------

class DirectReportingTest(unittest.TestCase):
    def test_whole_segment_safety_net_warns(self):
        engine, _ = _engine(lambda i, text: _speech_like(3.0))
        engine._report_quality(_drone(5.0))
        self.assertEqual(len(engine.last_warnings), 1)
        self.assertIn("noise rather than speech", engine.last_warnings[0])

    def test_clean_segment_warns_about_nothing(self):
        engine, _ = _engine(lambda i, text: _speech_like(3.0))
        engine._report_quality(_speech_like(5.0))
        self.assertEqual(engine.last_warnings, [])

    def test_repair_message_names_the_text(self):
        message = OmniVoiceEngine._repair_message(
            {"text": "यह एक वाक्य है।", "unrepaired": 0}
        )
        self.assertIn("यह एक वाक्य है।", message)

    def test_short_takes_are_never_judged(self):
        # A one-word preview must not be "repaired" into nonsense.
        engine, worker = _engine(lambda i, text: _drone(0.4))
        samples = engine.synthesize("Hi")
        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(engine.last_warnings, [])

    def test_both_engines_share_one_policy(self):
        from ai_voice_studio.omnivoice_server import (
            split_text_for_repair as server_split,
        )

        self.assertIs(server_split, quality.split_text_for_repair)
        self.assertEqual(DRONE_ATTEMPTS, quality.DRONE_ATTEMPTS)
        self.assertEqual(DRONE_REPAIR_ATTEMPTS, quality.DRONE_REPAIR_ATTEMPTS)
        self.assertEqual(DRONE_MAX_PIECES, quality.DRONE_MAX_PIECES)
        self.assertEqual(DRONE_REPAIR_MAX_DRAWS, quality.DRONE_REPAIR_MAX_DRAWS)

    def test_default_attempts_come_from_the_shared_policy(self):
        engine, _ = _engine(lambda i, text: _speech_like(3.0))
        self.assertEqual(engine.drone_attempts, quality.DRONE_ATTEMPTS)
        self.assertEqual(engine.repair_attempts, quality.DRONE_REPAIR_ATTEMPTS)

    def test_engine_accepts_tighter_attempt_budgets(self):
        engine, worker = _engine(
            lambda i, text: _drone(4.0), drone_attempts=1, repair_attempts=1
        )
        engine.synthesize("A short paragraph of narration.")
        self.assertEqual(len(worker.requests), 2)


if __name__ == "__main__":
    unittest.main()
