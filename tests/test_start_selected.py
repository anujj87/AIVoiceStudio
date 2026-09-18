"""Start Selected Recording: which segments the worker actually records.

The picker dialog passes a start position and - for "only record selected
file" - an exclusive stop position.  These tests pin that contract down
without a real TTS engine.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.documents.splitter import Segment  # noqa: E402
from ai_voice_studio.jobs.synthesizer import SynthesisWorker  # noqa: E402


class _FakeEngine:
    sample_rate = 22050

    def __init__(self):
        self.texts: list[str] = []

    def synthesize(self, text, sid=0, speed=1.0, pitch=1.0, volume=1.0):
        self.texts.append(text)
        return np.zeros(8, dtype=np.float32)


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
