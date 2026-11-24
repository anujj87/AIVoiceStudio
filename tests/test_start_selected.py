"""Start Selected Recording: which segments the worker actually records.

The picker dialog passes a start position and - for "only record selected
file" - an exclusive stop position.  These tests pin that contract down
without a real TTS engine.

The GPU pipeline (several segments in flight at once) is covered here too: the
segments may finish in any order, but they must be saved and reported in index
order, exactly like the one-segment-at-a-time loop.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.documents.splitter import Segment
from ai_voice_studio.jobs.synthesizer import SynthesisWorker


class _FakeEngine:
    sample_rate = 22050

    def __init__(self):
        self.texts: list[str] = []

    def synthesize(self, text, sid=0, speed=1.0, pitch=1.0, volume=1.0):
        self.texts.append(text)
        return np.zeros(8, dtype=np.float32)


class _FakePool:
    """Stands in for TtsEnginePool, with segments that finish out of order."""

    sample_rate = 22050

    def __init__(self, size=2, fail_on=None):
        self.size = size
        self.fail_on = fail_on
        self.seen: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def synthesize(self, text, sid=0, speed=1.0, pitch=1.0, volume=1.0):
        number = int(text.rsplit(" ", 1)[-1])
        if self.fail_on == number:
            raise RuntimeError("boom")
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        # Earlier segments take longer, so they *finish* last: the pipeline is
        # what puts the results back in order.
        time.sleep(0.01 * (5 - number))
        with self._lock:
            self.active -= 1
            self.seen.append(text)
        return np.full(8, number, dtype=np.float32)


class SynthesisPipelineTest(unittest.TestCase):
    """Several segments in flight, saved and reported in order."""

    def _run(self, pool, segments=4, cancel=False):
        tmp = tempfile.mkdtemp(prefix="aivs_pipe_")
        self.addCleanup(shutil.rmtree, tmp, True)
        cancel_event = threading.Event()
        if cancel:
            cancel_event.set()
        self.segments = [
            Segment(index=i + 1, title=f"chapter {i + 1}", text=f"text {i + 1}")
            for i in range(segments)
        ]
        statuses: list[str] = []
        errors: list[str] = []
        done: list[int] = []
        finished: list[bool] = []
        with mock.patch("ai_voice_studio.jobs.synthesizer.get_engine_pool",
                        return_value=pool), \
                mock.patch("ai_voice_studio.compute.synthesis_workers",
                           return_value=pool.size), \
                mock.patch("ai_voice_studio.jobs.synthesizer.audio_output.write_audio") as write:
            SynthesisWorker(
                segments=self.segments,
                voice_entry={"engine": "piper"},
                params={"compute": "cuda", "output_format": "wav"},
                output_dir=tmp,
                on_segment_done=lambda idx, title, path: done.append(idx),
                on_all_done=lambda: finished.append(True),
                on_error=errors.append,
                on_status=statuses.append,
                cancel_event=cancel_event,
            ).run()
        return {
            "write": write,
            "done": done,
            "finished": finished,
            "errors": errors,
            "statuses": statuses,
            "log": self._log(tmp),
            "samples": [call.args[0][0] for call in write.call_args_list],
            "paths": [os.path.basename(call.args[2]) for call in write.call_args_list],
        }

    def _log(self, tmp):
        path = os.path.join(tmp, "processed_text.json")
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)["processed_text"]

    def test_segments_are_reported_in_order_however_they_finish(self):
        pool = _FakePool(size=3)
        result = self._run(pool)
        self.assertGreaterEqual(pool.max_active, 2,
                               "segments must really be in flight together")
        self.assertNotEqual(pool.seen, [f"text {i}" for i in (1, 2, 3, 4)],
                            "the fake pool must finish out of order")
        self.assertEqual(sorted(pool.seen), [f"text {i}" for i in (1, 2, 3, 4)])
        self.assertEqual(result["done"], [0, 1, 2, 3])
        self.assertEqual(list(result["log"]), ["0", "1", "2", "3"])
        for index, entry in result["log"].items():
            self.assertEqual(entry["text"], f"text {int(index) + 1}")
            self.assertEqual(entry["file"], f"chapter {int(index) + 1}.wav")

    def test_each_audio_file_holds_its_own_segment(self):
        # The samples the fake pool returns carry the segment number, so the
        # file name and the audio written to it must agree.
        result = self._run(_FakePool(size=3))
        for number, path in zip(result["samples"], result["paths"]):
            self.assertEqual(path, f"chapter {int(number)}.wav")

    def test_a_failing_segment_stops_the_run_and_is_reported(self):
        result = self._run(_FakePool(size=3, fail_on=2))
        self.assertEqual(result["errors"], ["Segment 2 failed: boom"])
        self.assertEqual(result["done"], [0])
        self.assertEqual(result["finished"], [])
        self.assertEqual(list(result["log"]), ["0"])

    def test_a_cancelled_run_records_nothing_and_says_stopped(self):
        result = self._run(_FakePool(size=2), cancel=True)
        self.assertEqual(result["done"], [])
        self.assertEqual(result["finished"], [])
        self.assertIn("Stopped.", result["statuses"])


class SynthesisWorkerRangeTest(unittest.TestCase):
    """``start_index`` / ``end_index`` select the segments to record."""

    def _run(self, start_index=0, end_index=None):
        tmp = tempfile.mkdtemp(prefix="aivs_range_")
        self.addCleanup(shutil.rmtree, tmp, True)
        segments = [
            Segment(index=i + 1, title=f"chapter {i + 1}", text=f"text {i + 1}")
            for i in range(4)
        ]
        engine = _FakeEngine()
        done: list[bool] = []
        with mock.patch("ai_voice_studio.jobs.synthesizer.get_engine",
                        return_value=engine), \
                mock.patch("ai_voice_studio.jobs.synthesizer.audio_output.write_audio") as write:
            SynthesisWorker(
                segments=segments,
                voice_entry={"engine": "piper"},
                params={"compute": "cpu", "output_format": "wav"},
                output_dir=tmp,
                start_index=start_index,
                end_index=end_index,
                on_all_done=lambda: done.append(True),
            ).run()
        files = [call.args[2] for call in write.call_args_list]
        return engine.texts, files, done

    def test_no_bounds_records_every_segment(self):
        texts, files, done = self._run()
        self.assertEqual(texts, ["text 1", "text 2", "text 3", "text 4"])
        self.assertEqual(len(files), 4)
        self.assertEqual(done, [True])

    def test_only_record_selected_file_records_that_segment_and_stops(self):
        texts, files, done = self._run(start_index=2, end_index=3)
        self.assertEqual(texts, ["text 3"])
        self.assertEqual([os.path.basename(p) for p in files], ["chapter 3.wav"])
        self.assertEqual(done, [True])

    def test_record_all_files_from_here_runs_to_the_end(self):
        texts, _files, done = self._run(start_index=2)
        self.assertEqual(texts, ["text 3", "text 4"])
        self.assertEqual(done, [True])

    def test_a_past_the_end_stop_leaves_the_segment_alone(self):
        texts, files, done = self._run(start_index=3, end_index=9)
        self.assertEqual(texts, ["text 4"])
        self.assertEqual(len(files), 1)
        self.assertEqual(done, [True])


if __name__ == "__main__":
    unittest.main()
