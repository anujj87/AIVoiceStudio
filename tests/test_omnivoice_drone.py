"""Drone repair tests for the OmniVoice server engine.

OmniVoice sometimes renders a chunk as a loud low-frequency drone instead of a
voice (upstream k2-fsa/OmniVoice issues #37 / #73 / #144).  The server flags it
with ``X-No-Speech-Detected`` but only reports it, so the drone used to be
written straight into the recorded segment.  These tests cover the client-side
judgement (quality.py) and the two repairs around it: re-drawing the chunk, and
re-recording it as sentence-sized pieces — always returning *one* take, so a
segment is still one file.

No test touches the GPU, torch or the server: the HTTP layer is faked and the
audio is synthetic.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.request
import wave
from unittest import mock

import numpy as np

from ai_voice_studio.omnivoice_server import (
    DRONE_ATTEMPTS,
    DRONE_MAX_PIECES,
    DRONE_REPAIR_ATTEMPTS,
    DRONE_REPAIR_MAX_DRAWS,
    OmniVoiceServerEngine,
    OmniVoiceServerManager,
    quality,
    split_text_for_repair,
)

SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------

def _speech_like(seconds: float, amplitude: int = 8000) -> np.ndarray:
    """Noisy signal: a high zero-crossing rate, like voiced speech."""
    rng = np.random.default_rng(3)
    raw = rng.standard_normal(int(seconds * SAMPLE_RATE)) * amplitude
    return np.clip(raw, -32768, 32767).astype(np.int16)


def _drone(seconds: float, freq: float = 50.0, amplitude: int = 8000,
           phase: float = 0.0) -> np.ndarray:
    """Steady low-frequency tone: the sound the failure produces."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = np.sin(2.0 * np.pi * freq * t + phase) * amplitude
    return wave.astype(np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)


def _wav_bytes(samples) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class _Resp:
    """Stand-in for an ``http.client.HTTPResponse`` (headers optional)."""

    status = 200

    def __init__(self, data: bytes, headers: dict | None = None):
        self._data = data
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# The detector itself
# ---------------------------------------------------------------------------

class DroneDetectionTest(unittest.TestCase):
    def test_speech_scores_high_and_drone_scores_zero(self):
        self.assertGreater(quality.speech_window_ratio(_speech_like(3.0)), 0.9)
        self.assertEqual(quality.speech_window_ratio(_drone(3.0)), 0.0)

    def test_silence_is_not_judged(self):
        # Nothing audible: 1.0, so "no signal to measure" is never a failure.
        self.assertEqual(quality.speech_window_ratio(_silence(5.0)), 1.0)
        self.assertFalse(quality.judge(_silence(5.0)).bad)

    def test_drone_is_bad_and_speech_is_not(self):
        drone = quality.judge(_drone(4.0))
        self.assertTrue(drone.bad)
        self.assertIn("sounds like speech", drone.reason)
        self.assertFalse(quality.judge(_speech_like(4.0)).bad)

    def test_short_output_is_never_judged(self):
        # Mirrors the server: too few windows to judge a one-word render.
        self.assertFalse(quality.judge(_drone(0.5)).bad)

    def test_server_flag_alone_marks_a_take_bad(self):
        verdict = quality.judge(_speech_like(3.0), server_flagged=True)
        self.assertTrue(verdict.bad)
        self.assertEqual(verdict.reason, "server reported no speech")

    def test_span_inside_a_chunk_is_located(self):
        signal = np.concatenate([_speech_like(5.0), _drone(6.0), _speech_like(5.0)])
        spans = quality.degenerate_spans(signal)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertAlmostEqual(start, 5.0, delta=0.3)
        self.assertAlmostEqual(end, 11.0, delta=0.3)
        # The whole-output ratio still looks healthy, which is exactly why the
        # span check exists: 6 seconds of buzzing inside 16 seconds of speech.
        self.assertGreater(quality.speech_window_ratio(signal), 0.6)
        self.assertTrue(quality.judge(signal).bad)

    def test_two_buzzes_count_as_one_stretch(self):
        signal = np.concatenate([
            _speech_like(4.0), _drone(4.0), _silence(0.25),
            _drone(4.0), _speech_like(4.0),
        ])
        spans = quality.degenerate_spans(signal)
        self.assertEqual(len(spans), 1, spans)
        self.assertAlmostEqual(spans[0][1] - spans[0][0], 8.25, delta=0.6)

    def test_a_short_blip_is_tolerated(self):
        # Under MIN_SPAN_SECONDS: not worth re-recording a whole chunk for.
        signal = np.concatenate([_speech_like(6.0), _drone(2.0), _speech_like(6.0)])
        self.assertEqual(quality.degenerate_spans(signal), [])
        self.assertFalse(quality.judge(signal).bad)

    def test_sustained_mid_tone_is_not_a_drone_span(self):
        # Calibration: real narration has windows below the server's 0.04 bar,
        # so the span rule uses 0.01.  A 5-second 300 Hz sound (ZCR 0.025) is
        # therefore left alone, while an 8-second 30 Hz stretch (0.002, the
        # range the real drones measured at) is not.
        mid = np.concatenate([_speech_like(8.0),
                              _drone(5.0, freq=300.0, phase=0.3),
                              _speech_like(8.0)])
        self.assertEqual(quality.degenerate_spans(mid), [])
        self.assertFalse(quality.judge(mid).bad)

        low = np.concatenate([_speech_like(8.0),
                              _drone(8.0, freq=30.0, phase=0.3),
                              _speech_like(8.0)])
        spans = quality.degenerate_spans(low)
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][1] - spans[0][0], 8.0, delta=0.5)
        self.assertTrue(quality.judge(low).bad)


# ---------------------------------------------------------------------------
# Repair inside the HTTP client
# ---------------------------------------------------------------------------

class RepairSplittingTest(unittest.TestCase):
    """The repair needs sentence-sized pieces even for short paragraphs."""

    def test_paragraph_is_split_into_its_sentences(self):
        first = ("रैंड ने देखा कि सफ़ेद झंडा दिखाने वाले दो आदमी सड़क के उस पार "
                 "खड़े थे और वे उसकी ओर देख रहे थे।")
        second = ("उनमें से एक की आँख काली थी और उसका जबड़ा सूजा हुआ था, जिससे वह "
                  "और भी डरावना लग रहा था।")
        third = "रैंड ने धीरे से सिर हिलाया और अपनी तलवार की मूठ पर हाथ रख दिया।"
        text = f"{first} {second} {third}"
        pieces = split_text_for_repair(text)
        # A 240-character paragraph is one server chunk but three sentences,
        # and the sentence is what the repair re-records.
        self.assertEqual(pieces, [first, second, third])
        self.assertEqual(" ".join(pieces), text)

    def test_one_long_sentence_is_cut_in_half_at_a_word_boundary(self):
        text = " ".join(["शब्द"] * 60)  # 299 chars, no sentence break at all
        pieces = split_text_for_repair(text)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(" ".join(pieces), text)
        self.assertTrue(all(len(piece) >= 60 for piece in pieces))
        self.assertTrue(all(" " not in piece[:1] + piece[-1:] for piece in pieces))

    def test_a_hopelessly_short_sentence_is_left_alone(self):
        text = "रैंड ने देखा।"
        self.assertEqual(split_text_for_repair(text), [text])

    def test_a_wall_of_text_is_hard_cut_below_the_limit(self):
        text = "अ" * 1200
        pieces = split_text_for_repair(text, target_chars=250)
        self.assertEqual(len(pieces), 5)
        self.assertTrue(all(len(piece) <= 250 for piece in pieces))
        self.assertEqual("".join(pieces), text)

    def test_no_more_than_the_piece_cap_at_one_level(self):
        text = " ".join(f"शब्द{i}" for i in range(400))  # ~2800 chars
        pieces = split_text_for_repair(text)
        self.assertLessEqual(len(pieces), DRONE_MAX_PIECES)
        self.assertEqual(" ".join(pieces), text)

    def test_empty_text_yields_no_pieces(self):
        self.assertEqual(split_text_for_repair(""), [])
        self.assertEqual(split_text_for_repair("   "), [])


class LongAudioTest(unittest.TestCase):
    """A recorded segment can be half an hour long; judging it must stay cheap."""

    def test_a_long_take_is_analysed_in_blocks(self):
        from ai_voice_studio.omnivoice_server import quality as q

        # 3 minutes of speech with a 6-second drone at 60 s.
        signal = np.concatenate([
            _speech_like(60.0), _drone(6.0), _speech_like(114.0),
        ])
        with mock.patch.object(q, "_BLOCK_FRAMES", 8):  # force many blocks
            spans = q.degenerate_spans(signal)
            verdict = q.judge(signal)
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0][0], 60.0, delta=0.3)
        self.assertAlmostEqual(spans[0][1], 66.0, delta=0.3)
        self.assertTrue(verdict.bad)
        # Block-wise analysis must give the same answer as one pass.
        with mock.patch.object(q, "_BLOCK_FRAMES", 10 ** 6):
            self.assertEqual(q.degenerate_spans(signal), spans)
            self.assertEqual(q.judge(signal).speech_ratio,
                             verdict.speech_ratio)


class _FakeServer:
    """A fake ``/v1/audio/speech`` that answers from a scripted plan.

    ``plan(index, payload)`` decides one response; tests usually base it on the
    text (``payload["input"]``) and the request number.
    """

    def __init__(self, plan):
        self.plan = plan
        self.requests: list = []

    def __call__(self, req, timeout=None):
        raw = req.data.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError:  # the clone endpoint posts multipart form data
            payload = {"input": _multipart_text(raw), "body": raw}
        self.requests.append(payload)
        return self.plan(len(self.requests), payload)

    def seeds(self) -> list:
        return [p.get("seed") for p in self.requests]


def _multipart_text(body: str) -> str:
    """The ``text`` form field of a multipart request body."""
    marker = 'name="text"\r\n\r\n'
    start = body.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = body.find("\r\n", start)
    return body[start:end if end >= 0 else None]


class DroneRetryTest(unittest.TestCase):
    """A drone is a bad draw, so the chunk is drawn again.

    Measured against the real server: a third of draws of a real Hindi chapter
    paragraph come back as a drone (2 of 6 draws of one paragraph, 4 of the 148
    chunks of a full chapter), and the rate does not depend on the text size
    (12 of 18 draws of its sentences failed).  Re-drawing is therefore the fix,
    and a high attempt count is nearly free: only a bad draw pays for the next.
    """

    def setUp(self):
        self.server = _FakeServer(
            lambda i, payload: _Resp(
                _wav_bytes(_drone(4.0) if self.n == i else _speech_like(4.0))
            )
        )
        self.n = 1
        self._patch = mock.patch.object(
            urllib.request, "urlopen", self.server
        )
        self._patch.start()
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8882)

    def tearDown(self):
        self._patch.stop()

    def test_second_draw_replaces_the_drone(self):
        samples = self.mgr.synthesize("some narration text", seed=5)
        self.assertEqual(len(self.server.requests), 2)
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)
        # The caller's seed is honoured first, the retry is a different draw.
        self.assertEqual(self.server.seeds(), [5, 5 + 7919])
        # Nothing to warn about: the delivered take is clean.
        self.assertEqual(self.mgr.last_repairs, [])

    def test_a_clean_first_draw_costs_one_request(self):
        self.n = 0
        samples = self.mgr.synthesize("some narration text")
        self.assertEqual(len(self.server.requests), 1)
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)

    def test_server_flag_is_honoured_even_when_the_audio_looks_fine(self):
        server = _FakeServer(lambda i, payload: _Resp(
            _wav_bytes(_speech_like(4.0)),
            {"X-No-Speech-Detected": "true"} if i == 1 else {},
        ))
        with mock.patch.object(urllib.request, "urlopen", server):
            self.mgr.synthesize("flagged text")
        self.assertEqual(len(server.requests), 2)


class DroneSentenceRepairTest(unittest.TestCase):
    """When every draw of a long chunk is a drone, it is re-recorded piecewise.

    Upstream issue #144 recovered by generating smaller; the pieces are joined
    back in order, so the caller still gets one continuous take.
    """

    TEXT = " ".join(f"Sentence number {i} of this paragraph." for i in range(40))

    def setUp(self):
        def plan(i, payload):
            # The whole chunk drones; every smaller piece of it comes back clean.
            if payload["input"] == self.TEXT:
                return _Resp(_wav_bytes(_drone(4.0)))
            return _Resp(_wav_bytes(_speech_like(2.0)))

        self.server = _FakeServer(plan)
        self._patch = mock.patch.object(urllib.request, "urlopen", self.server)
        self._patch.start()
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8883)

    def tearDown(self):
        self._patch.stop()

    def test_chunk_is_rebuilt_from_sentence_sized_pieces(self):
        text = self.TEXT
        samples = self.mgr.synthesize(text)

        sent = [p["input"] for p in self.server.requests]
        whole = [s for s in sent if s == text]
        pieces = [s for s in sent if s != text]
        # Every draw of the whole chunk was a drone, then the pieces (one draw
        # each, because each piece came back clean on the first try).
        self.assertEqual(len(whole), DRONE_ATTEMPTS)
        self.assertEqual(len(sent), DRONE_ATTEMPTS + len(pieces))
        self.assertGreater(len(pieces), 1)        # Every word of the text still reaches the engine, in order.
        self.assertEqual(whole[0].split(), text.split())
        self.assertEqual(set(whole), {whole[0]})
        self.assertEqual(
            " ".join(pieces).split(), text.split()
        )
        # One take, and it is the repaired one (all pieces were clean).
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(samples.size, len(pieces) * 2 * SAMPLE_RATE)
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)

        # The repair is reported, with nothing left unresolved.
        self.assertEqual(len(self.mgr.last_repairs), 1)
        record = self.mgr.last_repairs[0]
        self.assertEqual(record["pieces"], len(pieces))
        self.assertEqual(record["unrepaired"], 0)
        self.assertEqual(record["attempts"], DRONE_ATTEMPTS)

    def test_clone_path_repairs_the_same_way(self):
        text = self.TEXT
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as fh:
                fh.write(_wav_bytes(_speech_like(3.0)))

            def clone_plan(i, payload):
                # payload["input"] is the text field of the multipart body.
                if payload["input"] == text:
                    return _Resp(_wav_bytes(_drone(4.0)))
                return _Resp(_wav_bytes(_speech_like(2.0)))

            server = _FakeServer(clone_plan)
            with mock.patch.object(urllib.request, "urlopen", server):
                samples = self.mgr.synthesize_clone(text, ref_audio_path=ref)

        self.assertGreater(len(server.requests), DRONE_ATTEMPTS)
        pieces = [p["input"] for p in server.requests if p["input"] != text]
        self.assertGreater(len(pieces), 1)
        self.assertEqual(" ".join(pieces).split(), text.split())
        self.assertGreater(quality.speech_window_ratio(samples), 0.9)
        self.assertEqual(len(self.mgr.last_repairs), 1)


class DroneUnrepairableTest(unittest.TestCase):
    """A take that cannot be repaired must not fail the recording.

    The best-looking draw is kept, and the chunk is named so the user can find
    and re-record that part; the alternative — raising — would lose the rest of
    a 40-minute segment.
    """

    def setUp(self):
        self.server = _FakeServer(lambda i, payload: _Resp(_wav_bytes(_drone(4.0))))
        self._patch = mock.patch.object(urllib.request, "urlopen", self.server)
        self._patch.start()
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8884)

    def tearDown(self):
        self._patch.stop()

    def test_short_chunk_returns_the_best_take_and_reports_it(self):
        samples = self.mgr.synthesize("A short paragraph of narration.")
        # The chunk draws, plus the one last draw the repair is allowed when
        # the text is too short to be split any further.
        self.assertEqual(len(self.server.requests), DRONE_ATTEMPTS + 1)
        self.assertTrue(np.asarray(samples).size)
        record = self.mgr.last_repairs[0]
        self.assertEqual(record["unrepaired"], 1)
        self.assertEqual(record["pieces"], 1)
        self.assertIn("narration", record["text"])

    def test_long_chunk_reports_the_pieces_that_stayed_bad(self):
        text = " ".join(f"Sentence number {i} of this paragraph." for i in range(40))
        samples = self.mgr.synthesize(text)
        self.assertTrue(np.asarray(samples).size)
        record = self.mgr.last_repairs[0]
        self.assertGreater(record["pieces"], 1)
        # Every piece (and every piece a piece was split into) is named.
        self.assertGreaterEqual(record["unrepaired"], record["pieces"])
        self.assertEqual(len(record["unresolved"]), record["unrepaired"])
        self.assertTrue(all(u["text"] for u in record["unresolved"]))

    def test_repair_attempts_are_bounded(self):
        # However bad the model behaves, one chunk costs at most its own draws
        # plus the repair budget, and never more than DRONE_MAX_PIECES pieces
        # at one level.
        text = " ".join(f"Sentence number {i} of this paragraph." for i in range(40))
        self.mgr.synthesize(text)
        record = self.mgr.last_repairs[0]
        self.assertEqual(self.mgr.repair_attempts, DRONE_REPAIR_ATTEMPTS)
        # The budget can be overshot by one draw per piece of the level it ran
        # out on, which is the bound the constants promise.
        self.assertLessEqual(
            len(self.server.requests),
            DRONE_ATTEMPTS + DRONE_REPAIR_MAX_DRAWS + DRONE_MAX_PIECES,
        )
        self.assertGreater(record["draws"], 0)
        self.assertLessEqual(record["draws"], DRONE_REPAIR_MAX_DRAWS)
        self.assertLessEqual(record["pieces"], DRONE_REPAIR_MAX_DRAWS)
        self.assertLessEqual(record["pieces"], DRONE_MAX_PIECES)


# ---------------------------------------------------------------------------
# What the user is told
# ---------------------------------------------------------------------------

class DroneReportingTest(unittest.TestCase):
    def _engine(self, repairs):
        engine = OmniVoiceServerEngine.__new__(OmniVoiceServerEngine)
        engine._server = mock.Mock(last_repairs=repairs)
        engine.last_warnings = []
        return engine

    def test_repair_message_names_the_text(self):
        message = OmniVoiceServerEngine._repair_message(
            {"text": "यह एक वाक्य है।", "unrepaired": 0}
        )
        self.assertIn("यह एक वाक्य है।", message)
        self.assertIn("re-recorded automatically", message)

    def test_unrepaired_message_asks_the_user_to_check(self):
        message = OmniVoiceServerEngine._repair_message(
            {"text": "boom", "unrepaired": 1}
        )
        self.assertIn("check the recording", message)

    def test_clean_segment_warns_about_nothing(self):
        engine = self._engine([])
        engine._report_quality(_speech_like(5.0))
        self.assertEqual(engine.last_warnings, [])

    def test_repaired_segment_is_reported(self):
        engine = self._engine([{"text": "some text", "unrepaired": 0}])
        engine._report_quality(_speech_like(5.0))
        self.assertEqual(len(engine.last_warnings), 1)
        self.assertIn("some text", engine.last_warnings[0])

    def test_finished_take_that_is_still_noise_is_warned_about(self):
        engine = self._engine([])
        engine._report_quality(_drone(6.0))
        self.assertEqual(len(engine.last_warnings), 1)
        self.assertIn("noise rather than speech", engine.last_warnings[0])

    def test_a_take_inside_the_calibrated_margin_is_not_warned_about(self):
        # 3 seconds of drone: below MIN_SPAN_SECONDS, so it is not a fault.
        engine = self._engine([])
        engine._report_quality(np.concatenate([
            _speech_like(6.0), _drone(3.0), _speech_like(6.0),
        ]))
        self.assertEqual(engine.last_warnings, [])

    def test_drone_with_a_long_silence_is_not_a_warning(self):
        # A title page with long pauses has nothing to judge, not a fault.
        engine = self._engine([])
        engine._report_quality(np.concatenate([_silence(20.0), _speech_like(3.0)]))
        self.assertEqual(engine.last_warnings, [])


if __name__ == "__main__":
    unittest.main()
