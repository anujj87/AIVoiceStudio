"""Speed benchmark for the Voice Lab engines.

Whether an engine is worth offering on the CPU is a question about *this*
computer, not about the size of its model: a 100M model on a fast desktop can
beat a small transformer on a laptop, and a card changes the answer again.  So
the application measures it instead of guessing:

* a probe speaks one ordinary sentence and times it against the audio it
  produced; the figure reported is the **real-time factor** (RTF) - seconds of
  generation per second of audio, so 1.0 is exactly real time and 0.3 is three
  times faster than real time;
* the first call of a probe loads the model (and downloads its weights on a
  first run) and is **not** timed; only the second, short call is, so the
  figure is the engine's steady-state speed and never the download;
* a probe that does not finish inside :data:`PROBE_TIMEOUT_SECONDS` is stopped
  and recorded as *did not finish*, which is a result too - it is what "far too
  slow" looks like when the run would otherwise take an hour.

The CPU figure is the decision-relevant one, because the CPU is the device the
application offers by default: :func:`measure_engine` always takes it (unless
the CPU is what was asked for), stores it, and pushes the verdict into
``voicelab`` so every device list in the application - Settings, the Recording
window, the previews - agrees on what this machine can do.

Measurements are stored in Settings (``clone_engines.benchmark.<engine>``) and
reloaded at start-up by :func:`load_verdicts`.  Nothing here is imported at
start-up: the heavy pieces (the worker, the engine packages) are only touched
when a measurement is actually run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

#: The sentence a probe speaks.  One ordinary sentence is enough to measure a
#: real-time factor (the per-call overheads are what a whole recording pays
#: again and again, so a single sentence is the right unit), and short enough
#: to keep the probe quick on a slow CPU.
PROBE_TEXT = (
    "The quick brown fox jumps over the lazy dog, and the reader carries on "
    "at a steady pace."
)

#: How long one probe may take before it is stopped and called too slow.
PROBE_TIMEOUT_SECONDS = 300.0

#: Settings key prefix holding the measured figures of one engine:
#: ``{"cpu": {...}, "cuda": {...}}``.
SETTINGS_PREFIX = "clone_engines.benchmark"


class BenchmarkError(Exception):
    """Raised when a speed measurement cannot be taken at all."""


def settings_key(engine_id: str) -> str:
    """Settings key holding one engine's measured figures."""
    return f"{SETTINGS_PREFIX}.{engine_id}"


def stored(settings, engine_id: str) -> dict[str, Any]:
    """The stored measurements of one engine (``{}`` when there are none)."""
    if settings is None:
        return {}
    try:
        value = settings.get(settings_key(engine_id), {})
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def timed_out_for(settings, engine_id: str, device: str) -> bool:
    """True when that device's probe was stopped for taking too long."""
    entry = stored(settings, engine_id).get(str(device)) or {}
    return bool(entry.get("timed_out"))


def load_verdicts(settings) -> None:
    """Push every stored CPU verdict into the registry (application start).

    The device lists consult the in-memory verdicts, so this is what makes a
    measurement from an earlier session apply to this one.  Called with the
    application's ``Settings``; also safe to call again after the file changed.
    """
    from . import clear_cpu_rtf, set_cpu_rtf
    from . import engines as engines_mod

    clear_cpu_rtf()
    for engine_id in engines_mod.engine_ids():
        entry = stored(settings, engine_id).get("cpu") or {}
        if not isinstance(entry, dict) or not entry:
            continue
        rtf = entry.get("rtf")
        if entry.get("timed_out") or rtf is None:
            # Measured and too slow to finish: the CPU is not offered.
            set_cpu_rtf(engine_id, None)
            continue
        try:
            set_cpu_rtf(engine_id, float(rtf))
        except (TypeError, ValueError):
            continue


# ---------------------------------------------------------------------------
# Running one probe
# ---------------------------------------------------------------------------
def _normalise(device: str | None) -> str:
    return "cuda" if str(device or "").strip().lower() == "cuda" else "cpu"


def _probe_voice(engine_id: str) -> dict[str, Any]:
    """A built-in voice to measure with (the first one the engine offers)."""
    from . import builtin_voice_entries
    from . import engines as engines_mod

    voices = builtin_voice_entries(engine_id)
    if not voices:
        raise BenchmarkError(
            f"{engines_mod.engine_name(engine_id)} has no built-in voice to "
            "measure with. Clone a voice for it first, then measure again."
        )
    return voices[0]


def _run_with_timeout(worker, request: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one worker request, killing the worker when it takes too long.

    The worker protocol has no per-request deadline - a synthesis owns its
    process until it answers - so the wait happens here and the process is
    killed once the deadline passes.  The measurement's model is loaded afresh
    by the next probe, which is exactly what "did not finish in time" means.
    """
    answers: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            answers.append(worker._request(request))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=run, name=f"voicelab-benchmark-{request.get('engine', '')}", daemon=True
    )
    thread.start()
    thread.join(max(1.0, float(timeout)))
    if thread.is_alive():
        worker.kill()
        return {
            "ok": False,
            "timed_out": True,
            "error": (
                f"did not finish in {int(timeout)} seconds - this device runs "
                "it far slower than real time"
            ),
        }
    if errors:
        raise BenchmarkError(str(errors[0]))
    if not answers:
        return {"ok": False, "error": "the engine's worker gave no answer"}
    return answers[0]


def probe(
    engine_id: str,
    device: str,
    voice_entry: dict[str, Any] | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Measure one engine on one device; returns the worker's answer.

    The answer is either ``{"ok": True, "rtf": ..., "seconds": ...,
    "audio_seconds": ...}`` or ``{"ok": False, "error": ..., "timed_out":
    bool}`` - a measurement that cannot be taken is a result too.  Only a
    worker that cannot be *started* raises, because that is an installation
    problem rather than a speed figure.
    """
    from . import CloneWorker, VoicelabError, build_synthesize_request

    resolved = _normalise(device)
    try:
        worker = CloneWorker(engine_id, resolved)
        # A worker that was just constructed has **no process yet**: every
        # public method starts it through ``_ensure_proc`` (which is also what
        # makes it answer a ``ping``), and ``_request`` deliberately refuses to
        # talk to a worker that is not running.  Without this call the probe
        # raised "The Voice Lab worker is not running." for every engine - the
        # measurement never happened, so no CPU verdict was ever recorded and
        # the CPU stayed out of the device list however fast the engine was.
        worker._ensure_proc()
    except VoicelabError as exc:
        raise BenchmarkError(str(exc)) from exc
    try:
        request = build_synthesize_request(
            engine_id=engine_id,
            device=resolved,
            text=PROBE_TEXT,
            voice_entry=voice_entry,
        )
        # The request names the device this probe is *for*.  The builder resolves
        # a device the way synthesis needs it - a GPU-only engine's "cpu"
        # becomes the GPU, because a stored CPU preference must not send such an
        # engine there before it has been measured - and that is exactly wrong
        # for a measurement: the figure filed as "this computer's CPU" would be
        # the card's, and an engine that never ran on a CPU could be offered it
        # (or kept from it) for the wrong reason.
        request["device"] = resolved
        request["cmd"] = "benchmark"
        request["timeout"] = float(timeout)
        answer = _run_with_timeout(worker, request, timeout)
        return _measured_as_asked(answer, resolved)
    except VoicelabError as exc:
        raise BenchmarkError(str(exc)) from exc
    finally:
        # A benchmark worker exists for its measurement only: its model is
        # loaded again by whatever speaks next (through the shared pool), and
        # leaving it running would hold a card's memory for nothing.
        try:
            worker.kill()
        except Exception:
            log.debug("Ignoring a failure in probe", exc_info=True)


# ---------------------------------------------------------------------------
# Measuring one engine (the entry point the GUI uses)
# ---------------------------------------------------------------------------
def _measured_as_asked(answer: dict[str, Any], device: str) -> dict[str, Any]:
    """Refuse a figure that belongs to a different device than the one asked.

    The worker reports the device the backend actually ran on, and a backend
    that cannot use the card falls back to the CPU by itself (Pocket TTS does).
    Filing that figure under "cuda" - or a card's figure under "cpu" - would
    turn the wrong number into a device verdict, so it is reported as a
    measurement that did not happen instead.
    """
    used = answer.get("device_used")
    if not answer.get("ok") or not used or str(used) == str(device):
        return answer
    return {
        "ok": False,
        "error": (
            f"the engine ran on {used} instead of {device}, so this is not its "
            f"speed on {device}"
        ),
    }


def _progress_message(engine_id: str, device: str) -> str:
    from . import engines as engines_mod

    where = (
        "the NVIDIA GPU" if device == "cuda" else "this computer's CPU"
    )
    return (
        f"Measuring {engines_mod.engine_name(engine_id)} on {where}... "
        "the first call loads the model and is not timed."
    )


def measure_engine(
    engine_id: str,
    device: str = "cpu",
    settings=None,
    voice_entry: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Measure an engine on the device asked for **and** on the CPU.

    The CPU figure is the one that decides whether the CPU is offered for the
    engine, so it is always taken unless the CPU is what was asked for.  The
    results are stored in ``settings`` (when one is given) and pushed into
    ``voicelab`` immediately, so the device lists update without a restart.

    Returns ``{"engine", "device", "probes", "cpu_rtf", "cpu_allowed"}``.
    """
    from . import engines as engines_mod
    from . import is_installed, set_cpu_rtf

    if not engines_mod.is_engine(engine_id):
        raise BenchmarkError(f"'{engine_id}' is not a Voice Lab engine.")
    if not is_installed(engine_id):
        raise BenchmarkError(
            f"{engines_mod.engine_name(engine_id)} is not installed yet - "
            "install it first (Settings > Voice Clone)."
        )
    if voice_entry is None:
        voice_entry = _probe_voice(engine_id)
    wanted = _normalise(device)
    targets = [wanted] if wanted == "cpu" else [wanted, "cpu"]
    probes: dict[str, Any] = {}
    for target in targets:
        if progress is not None:
            try:
                progress(_progress_message(engine_id, target))
            except Exception:
                log.debug("Ignoring a failure in measure_engine", exc_info=True)
        try:
            result = probe(engine_id, target, voice_entry=voice_entry, timeout=timeout)
        except BenchmarkError as exc:
            result = {"ok": False, "error": str(exc)}
        probes[target] = result
        if target != "cpu":
            continue
        if result.get("ok"):
            set_cpu_rtf(engine_id, result.get("rtf"))
        elif result.get("timed_out"):
            # Stopped for taking too long: above the limit by an unknown
            # margin, which is a verdict of "not offered".
            set_cpu_rtf(engine_id, None)
        # Any other failure (a missing reference voice, a broken environment)
        # says nothing about the CPU's speed: the previous verdict stands.
    if settings is not None:
        record(settings, engine_id, probes)
    return {
        "engine": engine_id,
        "device": wanted,
        "probes": probes,
        "cpu_rtf": _cpu_rtf(engine_id),
        "cpu_allowed": _cpu_allowed(engine_id),
    }


def _cpu_rtf(engine_id: str) -> float | None:
    """The live measured CPU figure (read through the registry)."""
    from . import cpu_rtf

    return cpu_rtf(engine_id)


def _cpu_allowed(engine_id: str) -> bool:
    """Whether the CPU is offered for the engine right now."""
    from . import cpu_allowed

    return cpu_allowed(engine_id)


def record(settings, engine_id: str, probes: dict[str, Any]) -> dict[str, Any]:
    """Store measured probes in Settings (whether they succeeded or not)."""
    data = stored(settings, engine_id)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for device, result in (probes or {}).items():
        if not isinstance(result, dict):
            continue
        if result.get("ok"):
            data[str(device)] = {
                "rtf": float(result.get("rtf") or 0.0),
                "seconds": float(result.get("seconds") or 0.0),
                "audio_seconds": float(result.get("audio_seconds") or 0.0),
                "sample_rate": int(result.get("sample_rate") or 0),
                "measured_at": stamp,
            }
        else:
            data[str(device)] = {
                "timed_out": bool(result.get("timed_out")),
                "error": str(result.get("error") or "")[:200],
                "measured_at": stamp,
            }
    if settings is not None:
        settings.set(settings_key(engine_id), data)
        # Written out at once.  A measurement takes minutes, and the Settings
        # dialog saves only when it is applied - so closing it without Apply
        # (or the application stopping first) used to throw away a figure that
        # is expensive to take and that decides the device lists.
        try:
            settings.save()
        except Exception:
            log.warning("Could not save the measured speed", exc_info=True)
    return data


def summary_text(engine_id: str, summary: dict[str, Any]) -> str:
    """Plain-language account of a measurement, for the message box/log."""
    from . import CPU_RTF_LIMIT, cpu_verdict, format_rtf, has_cuda
    from . import engines as engines_mod

    lines: list[str] = []
    for device in ("cuda", "cpu"):
        result = (summary.get("probes") or {}).get(device)
        if not isinstance(result, dict):
            continue
        where = "GPU" if device == "cuda" else "CPU"
        if result.get("ok"):
            lines.append(
                f"{where}: {format_rtf(result.get('rtf'))} "
                f"({float(result.get('audio_seconds') or 0):.1f} s of audio in "
                f"{float(result.get('seconds') or 0):.1f} s)."
            )
        else:
            lines.append(f"{where}: {result.get('error') or 'failed'}.")
    verdict = cpu_verdict(engine_id)
    name = engines_mod.engine_name(engine_id)
    if verdict is True:
        lines.append(
            f"That is under the {CPU_RTF_LIMIT:g}x real time limit, so the CPU "
            f"is now offered for {name}."
        )
    elif verdict is False and has_cuda():
        lines.append(
            f"That is over the {CPU_RTF_LIMIT:g}x real time limit, so the CPU "
            f"is not offered for {name} while this computer has a GPU."
        )
    elif verdict is False:
        lines.append(
            f"That is over the {CPU_RTF_LIMIT:g}x real time limit. No NVIDIA "
            "GPU was found, so the CPU stays available - it is the only device "
            "this computer has."
        )
    return "\n".join(lines)
