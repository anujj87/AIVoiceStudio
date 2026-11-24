"""Tests for the verdict in ``tools/measure_engine_vram.py``.

That tool measures what one GPU *slot* of a Voice Lab engine costs, and the
figures it prints are what ``engines._GPU_PARALLEL`` declares - so its
arithmetic is load-bearing and is checked here against the raw numbers the
reference measurement reported, without a card:

* the per-copy cost is the worst of the load peak and the generating peak over
  what the machine held before, divided by the copies that were loaded - a slot
  may never be charged less than the card was seen to use;
* the declared figure is the **dearest** run's cost rounded up, not the widest
  run's (one copy loading alone measured dearer than a copy beside another);
* the ceiling counts only copies that really spoke: a run the card could not
  finish bounds it from below and is reported as a failure, never as a pass;
* a run with no readable driver figure contributes nothing at all, and a
  measurement of only such runs declares nothing.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_PATH = os.path.join(ROOT, "tools", "measure_engine_vram.py")
_spec = importlib.util.spec_from_file_location("aivs_measure_vram_tool", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _run(slots, idle, load, loaded, peak, errors=(), wall=10.0, after=0):
    """One run as ``measure_run`` reports it: driver memory per phase."""
    return {
        "slots": slots, "idle": idle, "load": load, "loaded": loaded,
        "peak": peak, "after": after, "wall": wall, "load_wall": wall,
        "results": [], "errors": list(errors),
    }


#: The runs of the reference measurement (12 GB RTX 3060, Kitten TTS 2.0).
TWO_COPIES = [_run(1, 1574, 6440, 6334, 6544, wall=54.99),
              _run(2, 1723, 11031, 10823, 10943, wall=44.56)]


class SlotsCostTest(unittest.TestCase):
    """What one loaded copy cost a run."""

    def test_the_worst_phase_is_what_a_copy_costs(self):
        # Loading peaked above generating here (11031 against 10943), so the
        # loading peak is what the copies are charged for.
        run = _run(2, 1723, 11031, 10823, 10943)
        self.assertEqual(round(tool.slots_cost(run)), 4654)

    def test_the_copies_share_what_the_machine_gained(self):
        run = _run(3, 1000, 14000, 13000, 13500)
        self.assertEqual(round(tool.slots_cost(run)), 4333)  # (14000 - 1000) / 3

    def test_an_unreadable_driver_contributes_nothing(self):
        for idle, load, peak in ((0, 6440, 6544), (1574, 0, 0),
                                 (1574, 1000, 1200)):
            self.assertEqual(tool.slots_cost(_run(1, idle, load, 1000, peak)), 0,
                             (idle, load, peak))


class VerdictTest(unittest.TestCase):
    """What a set of runs says may be declared."""

    def test_the_dearest_run_decides_the_figure(self):
        verdict = tool.verdict(TWO_COPIES)
        self.assertEqual(verdict["vram_mb"], 5000)
        self.assertEqual(verdict["fine"]["slots"], 2, "the widest run that spoke")
        self.assertEqual(verdict["ceiling"], 2)
        self.assertEqual(verdict["failures"], [])

    def test_the_widest_run_is_not_automatically_the_dearest(self):
        # Three copies of a model that shares memory while loading can read
        # *cheaper* per copy than one on its own; the declaration still has to
        # cover the one on its own, or a lone slot would be under-charged.
        runs = [*TWO_COPIES, _run(3, 1700, 13200, 13100, 13200)]
        verdict = tool.verdict(runs)
        self.assertEqual(verdict["fine"]["slots"], 3)
        self.assertEqual(round(verdict["costs"][3]), 3833)
        self.assertEqual(verdict["vram_mb"], 5000, "4970 is the dearest")
        self.assertEqual(verdict["ceiling"], 3)

    def test_a_card_that_could_not_hold_a_copy_never_lifts_the_ceiling(self):
        failed = _run(3, 1723, 12288, 12288, 12288,
                      errors=["segment 1: RuntimeError: CUDA out of memory"])
        verdict = tool.verdict([*TWO_COPIES, failed])
        self.assertEqual(verdict["ceiling"], 2)
        self.assertEqual(verdict["vram_mb"], 5000)
        self.assertEqual(len(verdict["failures"]), 1)
        self.assertIn("3 copies", verdict["failures"][0])
        self.assertIn("out of memory", verdict["failures"][0])

    def test_nothing_readable_declares_nothing(self):
        for runs in ([], [_run(1, 0, 0, 0, 0)], [_run(2, 0, 0, 0, 0)]):
            verdict = tool.verdict(runs)
            self.assertEqual(verdict["vram_mb"], 0)
            self.assertEqual(verdict["ceiling"], 0)
            self.assertIsNone(verdict["fine"])

    def test_a_failed_run_still_shows_its_peaks_as_a_floor(self):
        # The widest run failed, but its peak was readable: it may be reported
        # as evidence, and the verdict still refuses to declare from it.
        failed = _run(3, 1000, 12000, 12000, 12000, errors=["boom"])
        verdict = tool.verdict([failed])
        self.assertEqual(verdict["vram_mb"], 0)
        self.assertEqual(verdict["ceiling"], 0)
        self.assertEqual(round(tool.slots_cost(verdict["readable"][0])), 3667)


class RoundingTest(unittest.TestCase):
    """A declared figure is rounded up, never down."""

    def test_up_to_the_next_hundred(self):
        self.assertEqual(tool.round_up(4970, tool.DECLARE_STEP_MB), 5000)
        self.assertEqual(tool.round_up(4654, tool.DECLARE_STEP_MB), 4700)
        self.assertEqual(tool.round_up(5000, tool.DECLARE_STEP_MB), 5000)
        self.assertEqual(tool.round_up(0, tool.DECLARE_STEP_MB), 0)

    def test_the_step_matches_the_declared_table(self):
        from ai_voice_studio.voicelab import engines

        for engine_id, declared in engines._GPU_PARALLEL.items():
            for variant, spec in declared["slots"].items():
                self.assertEqual(
                    spec["vram_mb"] % tool.DECLARE_STEP_MB, 0,
                    f"{engine_id}/{variant} is declared off the step: "
                    f"{spec['vram_mb']}",
                )


if __name__ == "__main__":
    unittest.main()
