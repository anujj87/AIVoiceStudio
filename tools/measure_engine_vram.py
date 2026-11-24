"""Measure one GPU slot of a Voice Lab engine, to declare its parallel plan.

Run it with the *application* interpreter, so the path measured is the one a
recording uses: the engine's own catalog voice entry, the app-facing
``voicelab.VoicelabEngine``, the shared worker pool, the engine's own
virtualenv and its worker subprocess.  While each run works, the card is
sampled through ``nvidia-smi`` - the driver's figure, not PyTorch's allocator,
because the card is what has to hold the copies - and the peak above idle is
what one slot costs (``engines._GPU_PARALLEL``).

    .venv\\Scripts\\python tools/measure_engine_vram.py kitten_tts_2
    .venv\\Scripts\\python tools/measure_engine_vram.py chatterbox --slots 2
    .venv\\Scripts\\python tools/measure_engine_vram.py kitten_tts_2 --text short

For one, two, three ... loaded copies it speaks the same set of segments through
the pool that many at a time, so the report shows both halves of a plan: the
peak the driver saw (what a slot costs) and the wall time (what the extra slots
buy).  Each run reads the card in four states - idle, loading, loaded and quiet,
then generating - because they answer different questions, and a copy is charged
the worst of the two peaks above idle, divided by the copies that were loaded.
The report ends with the ``_GPU_PARALLEL`` entry the numbers support (the
dearest per-copy cost, rounded up, so a slot is never charged less than the card
was seen to use, plus the copies this machine was seen to hold) and with the
plan the application settles on *now* (``voicelab.parallel_plan``), so the two
can be compared before and after the declaration is edited.

Exit codes: 0 measured (every run spoke), 1 a run failed (out of memory, no
audio), 2 the engine is not installed or the arguments are wrong.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: A sentence (KittenTTS-sized segment) and a paragraph: the paragraph is the
#: ~15 seconds of speech the declared figures are measured with, so a slot's
#: number is not the cost of a one-sentence segment only.
SHORT_TEXT = (
    "The quick brown fox jumps over the lazy dog every single morning."
)
PARA_TEXT = (
    "The library closed early that afternoon, so we walked down to the river "
    "instead. A few boats drifted past the bridge, and someone was playing a "
    "violin near the steps. We sat on the wall and watched the light change "
    "over the water until the street lamps came on. It was the kind of evening "
    "that makes a long week feel much shorter."
)


class Sampler:
    """Samples the driver's used video memory in the background."""

    def __init__(self, interval_ms=150):
        self.samples: list = []
        self.stop = threading.Event()
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"-lms={interval_ms}"],
            stdout=subprocess.PIPE, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for line in self.proc.stdout:
            if self.stop.is_set():
                break
            try:
                self.samples.append(int(float(line.strip().splitlines()[0])))
            except (ValueError, IndexError):
                continue

    def start(self):
        self.thread.start()
        time.sleep(0.4)
        return self

    def finish(self) -> None:
        self.stop.set()
        with contextlib.suppress(Exception):
            self.proc.kill()
        self.thread.join(timeout=2)

    def idle(self) -> int:
        """What the card held before this run: the desktop, the driver, us."""
        return int(min(self.samples)) if self.samples else 0

    def peak(self) -> int:
        return int(max(self.samples)) if self.samples else 0

    def median(self) -> int:
        return int(statistics.median(self.samples)) if self.samples else 0


def card_total_mb() -> int:
    """The card's own size, from the driver (0 when it cannot be read)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(float(result.stdout.strip().splitlines()[0]))
    except Exception:
        return 0


def voice_entry_for(engine_id: str, language: str, voice: str):
    """The voice entry the panels would hand to the engine."""
    from ai_voice_studio import voicelab

    entries = voicelab.builtin_voice_entries(engine_id)
    if voice:
        wanted = [e for e in entries if str(e.get("voice")) == voice]
        if not wanted:
            raise SystemExit(
                f"{engine_id} has no built-in voice called {voice!r}. "
                f"Known: {', '.join(sorted(str(e.get('voice')) for e in entries))}"
            )
        entries = wanted
    wanted = (language or "").lower()[:2]
    entries.sort(key=lambda e: str(e.get("language") or "").lower()[:2] != wanted)
    if not entries:
        raise SystemExit(
            f"{engine_id} offers no built-in voice to measure with. Pass a "
            "cloned voice instead, or measure the engine the way it is used."
        )
    entry = dict(entries[0])
    entry["device"] = "cuda"
    return entry


def round_up(value: float, step: int = 1) -> int:
    """``value`` rounded **up** to the next ``step``: a figure may over-state."""
    return int(-(-float(value) // step) * step)


#: The step a declared figure is rounded up to, so a slot is never charged
#: less than the card was seen to use (and a declaration reads like the table
#: it goes into: 4700, 8000, 5000 ...).
DECLARE_STEP_MB = 100


def slots_cost(run) -> float:
    """What one loaded copy cost this run, in MiB (0 when unreadable).

    The worst of the two peaks above what the machine held before - the load
    and the generating one - divided by the copies that were loaded.  A failed
    run (out of memory) has no figure of its own but still bounds the ceiling.
    """
    if not run["idle"] or not run["slots"]:
        return 0.0
    peak = max(run["load"], run["peak"])
    if not peak or peak <= run["idle"]:
        return 0.0
    return (peak - run["idle"]) / float(run["slots"])


def verdict(runs) -> dict:
    """What a set of runs measures: the figure to declare and the ceiling.

    ``vram_mb`` is the **dearest** per-copy cost measured - over every run that
    really spoke, not only the widest one - rounded up, because a declaration
    has to cover the worst a slot was seen to cost (a single copy loading one
    measured dearer than the same copy beside another).  ``fine`` is the widest
    run that spoke, the one several copies were resident for, which is the state
    a plan is built from.  ``ceiling`` is how many copies a measurement showed
    running at once: a run that failed still bounds it from below, never from
    above.  ``failures`` names what went wrong, run by run, so a number is never
    reported without its evidence.
    """
    readable = [run for run in runs if slots_cost(run)]
    spoke = [run for run in readable if not run.get("errors")]
    fine = max(spoke, key=lambda run: run["slots"]) if spoke else None
    dearest = max((slots_cost(run) for run in spoke), default=0.0)
    return {
        "vram_mb": round_up(dearest, DECLARE_STEP_MB) if spoke else 0,
        "costs": {run["slots"]: slots_cost(run) for run in spoke},
        "fine": fine,
        "ceiling": max((run["slots"] for run in spoke), default=0),
        "failures": [
            f"{run['slots']} copies: {(run.get('errors') or ['?'])[0]}"
            for run in runs if run.get("errors")
        ],
        "readable": readable,
    }


def quiet_window(seconds=1.2):
    """Sample the card while nothing happens: what it holds right now."""
    sampler = Sampler().start()
    try:
        time.sleep(seconds)
    finally:
        sampler.finish()
    return sampler


def speak_together(engine, texts, results, errors):
    """Speak every text at once; the pool decides how many run at a time."""
    sampler = Sampler().start()
    try:
        started = time.perf_counter()
        threads = []
        for index in range(len(texts)):
            def speak(i=index):
                try:
                    at = time.perf_counter()
                    samples = engine.synthesize(texts[i], speed=1.0)
                    spent = time.perf_counter() - at
                    samples = samples.reshape(-1)
                    results.append(
                        (i, spent, samples.size / float(engine.sample_rate))
                    )
                except Exception as exc:
                    errors.append(
                        f"segment {i + 1}: {type(exc).__name__}: {exc}"
                    )

            threads.append(threading.Thread(target=speak))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started
    finally:
        sampler.finish()
    return sampler, wall


def measure_run(engine_id, entry, slots, texts):
    """Load ``slots`` copies and speak ``texts`` through them.

    The pool is forced to ``instances`` with that many slots - exactly how the
    application builds it for an engine that speaks one text per call.  The card
    is read four times, because the four answers are different things:

    ``idle``    what the machine held before anything loaded;
    ``load``    the peak while every copy loaded and spoke its first segment;
    ``loaded``  what the copies cost while they sat there, loaded but quiet -
                this is what a *slot* costs, the model plus its runtime;
    ``peak``    the peak while several segments were generated at once.

    A slot is charged the worst of ``load`` and ``peak`` above ``idle``, so a
    declared figure is never smaller than what the card was seen to use.
    """
    from ai_voice_studio import voicelab

    idle = quiet_window()
    results: list = []
    errors: list = []

    # A fresh pool each time: that is what makes ``slots`` copies load.
    voicelab.close_workers(engine_id)
    voicelab.pool_for(engine_id, "cuda", slots=slots, mode="instances")
    engine = voicelab.VoicelabEngine(dict(entry), device="cuda")

    # Warm-up: one segment per copy, so every worker has loaded its own model
    # before the timed run (and the load cost lands in the ``load`` window).
    warm_results: list = []
    load_sampler, load_wall = speak_together(
        engine, [texts[0]] * slots, warm_results, errors
    )
    loaded = quiet_window()
    peak_sampler, wall = speak_together(engine, texts, results, errors)
    voicelab.close_workers(engine_id)
    after = quiet_window()
    return {
        "slots": slots,
        "idle": idle.idle(),
        "load": load_sampler.peak(),
        "loaded": loaded.median(),
        "peak": peak_sampler.peak(),
        "after": after.median(),
        "load_wall": load_wall,
        "wall": wall,
        "results": sorted(results),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure a Voice Lab engine's video memory per GPU slot."
    )
    parser.add_argument("engine", help="engine id, e.g. kitten_tts_2")
    parser.add_argument("--slots", type=int, default=3,
                        help="how many loaded copies to try (default 3)")
    parser.add_argument("--texts", type=int, default=4,
                        help="segments to speak per run (default 4)")
    parser.add_argument("--text", choices=("short", "para"), default="para",
                        help="segment length to measure with (default para, "
                             "~15 s of speech)")
    parser.add_argument("--variant", default="", help="variant id to measure")
    parser.add_argument("--voice", default="", help="built-in voice to speak")
    parser.add_argument("--language", default="en", help="language code")
    parser.add_argument("--keep-going", action="store_true",
                        help="carry on after a run fails (out of memory)")
    args = parser.parse_args()

    from ai_voice_studio import compute, voicelab
    from ai_voice_studio.voicelab import engines

    if not engines.is_engine(args.engine):
        print(f"error: {args.engine!r} is not a Voice Lab engine")
        return 2
    if not voicelab.is_installed(args.engine):
        print(f"error: {engines.engine_name(args.engine)} is not installed")
        return 2
    if not compute.has_nvidia_gpu():
        print("error: no NVIDIA GPU on this machine, nothing to measure")
        return 2

    entry = voice_entry_for(args.engine, args.language, args.voice)
    variant = args.variant or str(entry.get("variant") or "")
    text = SHORT_TEXT if args.text == "short" else PARA_TEXT
    texts = [text] * max(1, args.texts)
    total = card_total_mb()
    print(f"engine:  {engines.engine_name(args.engine)} ({args.engine})")
    print(f"card:    {compute.gpu_summary()} ({total} MiB), "
          f"{compute.free_vram_mb()} MiB free now")
    print(f"voice:   {entry.get('voice')} / {entry.get('language')} / {variant}")
    print(f"text:    {args.text} ({len(text)} chars)  segments/run: {len(texts)}")
    declared = engines.gpu_parallel(args.engine, variant)
    print(f"declared now: {declared if declared else 'no plan of its own'}")
    print()

    runs = []
    failures = []
    for slots in range(1, max(1, args.slots) + 1):
        print(f"--- {slots} loaded cop{'y' if slots == 1 else 'ies'} ---")
        try:
            run = measure_run(args.engine, entry, slots, texts)
        except Exception as exc:
            print(f"  run failed: {type(exc).__name__}: {exc}")
            failures.append(f"{slots} copies: {type(exc).__name__}: {exc}")
            if not args.keep_going:
                break
            continue
        runs.append(run)
        for message in run["errors"]:
            print(f"  ERROR {message}")
        if run["errors"]:
            failures.append(f"{slots} copies: {run['errors'][0]}")
        audio = sum(seconds for _i, _s, seconds in run["results"])
        spent = ", ".join(f"{s:5.2f}s" for _i, s, _a in run["results"])
        print(f"  segments {len(run['results'])}/{len(texts)}  per-segment "
              f"[{spent}]  audio {audio:5.2f}s  wall {run['wall']:6.2f}s")
        print(f"  load+first pass {run['load_wall']:6.2f}s after the pool was built")
        print(f"  driver: idle {run['idle']} MiB  load peak {run['load']} MiB  "
              f"loaded {run['loaded']} MiB  generating peak {run['peak']} MiB  "
              f"unloaded again {run['after']} MiB")
        if run["wall"] > 0 and audio:
            print(f"  throughput {audio / run['wall']:5.3f}x real time")
        cost = slots_cost(run)
        if cost:
            print(f"  {cost:6.0f} MiB per loaded copy  ->  "
                  f"{round_up(cost, DECLARE_STEP_MB)} MiB declared")
        if failures and not args.keep_going:
            break
        print()

    result = verdict(runs)
    available = verdict([run for run in runs if slots_cost(run)])
    print("=== verdict ===")
    for message in result["failures"]:
        print(f"  failed: {message}")
    best = result["fine"]
    if best is None:
        if available["fine"] is None:
            print("  nothing to declare: no run left a readable driver figure")
            return 1
        print("  only runs the card could not finish left a figure; the widest "
              "one is printed as a floor, and it may not be declared")
        best = available["fine"]
        declared_mb = round_up(slots_cost(best), DECLARE_STEP_MB)
    else:
        declared_mb = result["vram_mb"]
    ceiling = max(result["ceiling"], 1)

    print(f"  widest run: {best['slots']} copies -> idle {best['idle']} MiB, "
          f"load peak {best['load']} MiB, generating peak {best['peak']} MiB")
    costs = ", ".join(
        f"{slots} cop{'y' if slots == 1 else 'ies'}: {cost:6.0f} MiB"
        for slots, cost in sorted(result["costs"].items())
    )
    print(f"  per copy: {costs}")
    print(f"  {declared_mb} MiB declared (dearest, rounded up), verified with "
          f"{ceiling} cop{'y' if ceiling == 1 else 'ies'}")
    if best["slots"] == 1:
        print("  note: only one copy finished, so the figure is what a slot "
              "costs and nothing was shown about a second - ask for more "
              "slots (or a card with more free) to measure the ceiling")
    solo = next((run for run in result["readable"] if run["slots"] == 1), None)
    for run in sorted(result["readable"], key=lambda run: run["slots"]):
        if run["slots"] > 1 and solo and solo["wall"] > 0 and run["wall"] > 0:
            print(f"  throughput: {run['slots']} copies spoke their segments in "
                  f"{run['wall']:6.2f}s against {solo['wall']:6.2f}s with one "
                  f"({solo['wall'] / run['wall']:4.2f}x)")
    if total and best["peak"] > total:
        print(f"  note: the peak is above the card's {total} MiB - the driver "
              "was already swapping, so the figure is a floor, not a peak")
    free_now = compute.free_vram_mb() or 0
    print()
    print("declaration this measurement supports:")
    print(f'    "{args.engine}": {{')
    print('        "mode": "instances",')
    print(f'        "max_slots": {ceiling},')
    print('        "slots": {')
    print(f'            "{variant}": {{"vram_mb": {declared_mb}}},')
    print('        },')
    print('    },')
    print()
    affordable = 1 + max(0, (free_now - voicelab.GPU_VRAM_RESERVE_MB
                             - declared_mb)) // declared_mb
    print(f"plan the application settles on now: "
          f"{voicelab.parallel_plan(args.engine, 'cuda', variant)}")
    print(f"with this declaration, {free_now} MiB free would afford about "
          f"{affordable} copies")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
