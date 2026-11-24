"""Pure planning logic for an audio description run.

Everything in this module is a plain function over numbers and lists: no wx, no
subprocess, no model.  That is deliberate - it is the part of the feature that
decides *how* a video is cut up, *how long* the run will take and *where* a
description may be spoken, and it is therefore the part that can be tested
exactly.  The GUI, the job and the model worker all call into it.

The three ideas here:

* **Chunking.** A single call can only hold a few minutes of video, so a long
  video is cut into half-open ``[start, end)`` windows that tile it with no gap
  and no overlap, and the remainder becomes a shorter last chunk.
* **Estimating.** Describing video is orders of magnitude more expensive than
  transcribing it, which surprises anyone used to Whisper's speed, so the cost is
  estimated before the run and shown as a human string.
* **Gap fitting.** A description has to be spoken *in* the silence between
  dialogue.  :func:`free_gaps` finds those silences and :func:`fit_into_gaps`
  places descriptions into them, dropping (not truncating into dialogue) whatever
  does not fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..constants import (
    AD_CHUNK_SECONDS,
    AD_DEFAULT_FPS,
    AD_MAX_DESCRIPTION_SECONDS,
    AD_MAX_WINDOW_SECONDS,
    AD_MIN_GAP_SECONDS,
    AD_WARN_SECONDS,
    AD_WINDOW_SECONDS,
)
from . import catalog

#: Hard cap on the frames sampled from one chunk.  A 10-minute chunk at 1 fps is
#: 600 frames, which no small model can take; the cap keeps a mis-set frame rate
#: from turning into a memory error.
MAX_FRAMES_PER_CHUNK = 120

#: Seconds of one-off cost per source file (loading the model into memory).
MODEL_LOAD_SECONDS = 45.0

#: Average speaking rate used to bound a description by time (words per second).
WORDS_PER_SECOND = 2.6


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_count(duration_seconds: float, chunk_seconds: float = AD_CHUNK_SECONDS) -> int:
    """How many chunks a video of ``duration_seconds`` is cut into."""
    duration = _positive(duration_seconds)
    if duration <= 0:
        return 0
    step = _chunk_step(chunk_seconds)
    return math.ceil(round(duration / step, 9))


def plan_chunks(duration_seconds: float,
                chunk_seconds: float = AD_CHUNK_SECONDS) -> list[dict[str, Any]]:
    """Cut a video into described-in-isolation windows.

    Returns one entry per chunk: ``{"index", "start", "end", "duration"}``.  The
    windows are half-open and tile ``[0, duration)`` exactly: ``end`` of one is
    ``start`` of the next, and the last chunk ends at the video's duration even
    when that is shorter than the requested length.  A video shorter than one
    chunk therefore yields exactly one chunk covering all of it, which is why the
    "chunk" option is harmless for a short clip.
    """
    duration = _positive(duration_seconds)
    if duration <= 0:
        return []
    step = _chunk_step(chunk_seconds)
    chunks: list[dict[str, Any]] = []
    index = 0
    start = 0.0
    while start < duration - 1e-9:
        end = min(start + step, duration)
        chunks.append({
            "index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })
        index += 1
        start = end
    return chunks


def plan_windows(chunk: dict[str, Any],
                 window_seconds: float = AD_WINDOW_SECONDS) -> list[dict[str, Any]]:
    """Cut one chunk into the pieces that are each described separately.

    A chunk exists to bound memory and to give a long run somewhere to stop and
    resume; it is far too long to be the unit a listener is told about.  One
    description per two minutes is a summary - measured on a real lecture, it
    said "a man sits in a chair beside a whiteboard" and nothing about the line
    of code that had appeared - so every chunk is described window by window,
    and *this* is what decides how much is said: about one description every
    twenty seconds, the cadence broadcast audio description uses.

    Returns one entry per window: ``{"index", "start", "end", "duration",
    "chunk"}``.  The windows tile the chunk exactly and none is longer than
    :data:`AD_MAX_WINDOW_SECONDS`, so a wildly large window setting still yields
    descriptions rather than one summary.
    """
    start = _positive(chunk.get("start"))
    end = _positive(chunk.get("end"))
    if end <= start:
        return []
    size = _positive(window_seconds) or AD_WINDOW_SECONDS
    size = min(size, AD_MAX_WINDOW_SECONDS)
    size = max(1.0, size)
    chunk_index = chunk.get("index")
    windows: list[dict[str, Any]] = []
    index = 0
    cursor = start
    while cursor < end - 1e-9:
        stop = min(cursor + size, end)
        windows.append({
            "index": index,
            "start": round(cursor, 3),
            "end": round(stop, 3),
            "duration": round(stop - cursor, 3),
            "chunk": chunk_index,
        })
        index += 1
        cursor = stop
    return windows


def plan_windows_for(chunks: Sequence[dict[str, Any]],
                     window_seconds: float = AD_WINDOW_SECONDS) -> list[dict[str, Any]]:
    """Every window of every chunk of a plan, in order.

    The describing worker is handed these; the GUI counts them, because the
    number of *descriptions* a run will produce is the number of windows, not of
    chunks.
    """
    windows: list[dict[str, Any]] = []
    for chunk in chunks or []:
        windows.extend(plan_windows(chunk, window_seconds))
    return windows


def window_count(chunks: Sequence[dict[str, Any]],
                 window_seconds: float = AD_WINDOW_SECONDS) -> int:
    """How many descriptions a chunk plan stands for.

    A caller that already carries ``windows`` per chunk (the worker's request
    does) gets those counted, so nothing is planned twice.
    """
    total = 0
    for chunk in chunks or []:
        windows = chunk.get("windows")
        total += len(windows) if windows else len(plan_windows(
            chunk, window_seconds))
    return total


def chunk_at(chunks: Sequence[dict[str, Any]], seconds: float) -> dict[str, Any] | None:
    """The chunk holding ``seconds`` (``None`` when nothing does)."""
    for chunk in chunks:
        if chunk["start"] <= seconds < chunk["end"]:
            return chunk
    if chunks and abs(seconds - chunks[-1]["end"]) < 1e-6:
        return chunks[-1]
    return None


def sample_times(start: float, end: float, fps: float = AD_DEFAULT_FPS,
                 *, cap: int = MAX_FRAMES_PER_CHUNK) -> list[float]:
    """Frame timestamps to extract from one window, in increasing order.

    A frame is taken at the window's start and then every ``1 / fps`` seconds
    while the window lasts.  A non-positive rate means "one frame at the start";
    the result is always non-empty for a non-empty window so a chunk is never
    described from nothing.
    """
    start = _positive(start) or 0.0
    end = _positive(end)
    if end <= start:
        return []
    rate = _positive(fps)
    if rate <= 0:
        return [round(start, 3)]
    count = math.ceil((end - start) * rate)
    count = max(1, min(int(cap), count))
    return [round(min(start + i / rate, end - 1e-3), 3) for i in range(count)]


def total_frames(duration_seconds: float, fps: float = AD_DEFAULT_FPS,
                 chunk_seconds: float = AD_CHUNK_SECONDS) -> int:
    """Frames a whole run will sample (used by the estimate and the warning)."""
    return sum(
        len(sample_times(chunk["start"], chunk["end"], fps))
        for chunk in plan_chunks(duration_seconds, chunk_seconds)
    )


# ---------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------
def estimate(duration_seconds: float,
             *,
             model_id: str,
             variant: str = "",
             fps: float = AD_DEFAULT_FPS,
             device: str = "cpu",
             chunk_seconds: float = AD_CHUNK_SECONDS,
             window_seconds: float = AD_WINDOW_SECONDS,
             speech_seconds: float = 0.0) -> dict[str, Any]:
    """Estimate how long describing a video will take.

    The model is: compute per minute of video scales linearly with the frame
    rate, plus a one-off model-load cost and a small per-chunk overhead.  It is
    an estimate for warning, not a promise; it is deliberately conservative.

    Describing a chunk window by window does not change this figure: a call
    costs mostly its frames and its answer, both of which are already counted
    per minute of video, and the per-call share of that cost is what a run of
    ten twenty-second windows was measured to take on a 12 GB card.  What the
    extra calls change is *how much* is said, and that is reported as
    ``windows``.

    ``speech_seconds`` adds the cost of the optional speech-timing pass, which is
    what makes gap fitting possible (it uses the speech-to-text model, which is
    fast next to the describer).
    """
    duration = _positive(duration_seconds)
    rate = _positive(fps)
    chunks = chunk_count(duration, chunk_seconds)
    windows = window_count(plan_chunks(duration, chunk_seconds), window_seconds)
    per_minute = catalog.cost_per_video_minute(model_id, variant, device)
    # Frame rate scales the frame work directly; the model's own cost figure is
    # quoted at the catalog's reference rate.
    reference = _positive(catalog.REFERENCE_FPS) or AD_DEFAULT_FPS
    per_minute *= (rate / reference) if reference else 1.0
    if duration <= 0:
        return {
            "seconds": 0.0, "chunks": 0, "windows": 0, "per_minute": per_minute,
            "fps": rate, "device": device, "frames": 0, "descriptions": 0,
            "speech_seconds": 0.0,
        }
    base = (duration / 60.0) * per_minute
    overhead = MODEL_LOAD_SECONDS + 5.0 * chunks
    total = base + overhead + max(0.0, float(speech_seconds))
    return {
        "seconds": total,
        "chunks": chunks,
        "windows": windows,
        "per_minute": per_minute,
        "fps": rate,
        "device": device,
        "frames": total_frames(duration, fps, chunk_seconds),
        "descriptions": windows,
        "speech_seconds": max(0.0, float(speech_seconds)),
    }


def needs_warning(duration_seconds: float,
                  threshold: float = AD_WARN_SECONDS) -> bool:
    """Whether a video is long enough that the user must be warned first."""
    return _positive(duration_seconds) > _positive(threshold)


def format_duration(seconds: float) -> str:
    """A short human label for a length of time ("45 s", "12 min", "1 h 40 min")."""
    total = round(max(0.0, float(seconds or 0.0)))
    if total < 60:
        return f"{total} s"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if minutes == 0:
        return f"{hours} h"
    return f"{hours} h {minutes:02d} min"


def warning_message(duration_seconds: float, estimate_info: dict[str, Any],
                    *, chunked: bool) -> str:
    """The long-video warning shown before a run starts."""
    video = format_duration(duration_seconds)
    taken = format_duration(estimate_info.get("seconds", 0.0))
    chunks = int(estimate_info.get("chunks", 0) or 0)
    lines = [
        f"This video is {video} long.",
        f"Describing it on {_device_words(estimate_info.get('device'))} is "
        f"estimated to take about {taken}"
        + (f", in {chunks} chunk(s)." if chunks else "."),
    ]
    if not chunked and chunks > 1:
        lines.append(
            "Describe it in chunks to keep memory use low and to be able to "
            "stop and resume between chunks."
        )
    lines.append(
        "Describing video is far slower than transcribing audio. You can close "
        "this window at any time; chunks already described are kept."
    )
    return "\n".join(lines)


def _device_words(device: str | None) -> str:
    return "the GPU" if str(device) == "cuda" else "the CPU"


# ---------------------------------------------------------------------------
# Gap fitting
# ---------------------------------------------------------------------------
def free_gaps(speech: Sequence[dict[str, Any]], duration: float,
              min_gap: float = AD_MIN_GAP_SECONDS) -> list[tuple[float, float]]:
    """The silent windows of ``duration`` where a description could be spoken.

    ``speech`` is a list of ``{"start", "end"}`` intervals (from speech to text);
    overlapping and touching items are merged first, so the result is a sorted
    list of disjoint, true silences with no gap shorter than ``min_gap``.
    """
    span = _positive(duration)
    if span <= 0:
        return []
    intervals: list[tuple[float, float]] = []
    for item in speech or []:
        raw_start, raw_end = item.get("start"), item.get("end")
        if raw_start is None or raw_end is None:
            # A phrase with no timing tells us nothing about where the silence
            # is, so it cannot carve a hole in the timeline.
            continue
        start = max(0.0, min(_positive(raw_start), span))
        end = max(0.0, min(_positive(raw_end), span))
        if end > start:
            intervals.append((start, end))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor >= _positive(min_gap):
            gaps.append((round(cursor, 3), round(start, 3)))
        cursor = max(cursor, end)
    if span - cursor >= _positive(min_gap):
        gaps.append((round(cursor, 3), round(span, 3)))
    return gaps


def fit_into_gaps(descriptions: Sequence[dict[str, Any]],
                  speech: Sequence[dict[str, Any]],
                  *,
                  duration: float,
                  min_gap: float = AD_MIN_GAP_SECONDS,
                  max_seconds: float = AD_MAX_DESCRIPTION_SECONDS) -> dict[str, Any]:
    """Place chunk descriptions into the silences that follow them.

    Each description is placed into the **earliest pause that can hold it**: a
    gap counts when it has not already ended before the description's own start,
    and the description begins at the later of the gap's start and the
    description's start.  So a description whose chunk begins in the middle of a
    pause uses the rest of that pause instead of jumping to a distant one, which
    is what keeps the narration close to the action it describes.

    "Can hold it" is judged against the description's own length, not against a
    fixed budget: a sentence that needs three seconds skips a two-second pause in
    favour of a slightly later one, instead of being chopped to a fragment just
    because the fragment would fit.  Only when no pause can hold the description
    in full does it fall back to the earliest pause with any room, and shorten
    there; and a description with no room anywhere is dropped rather than spoken
    over speech.

    Returns ``{"placed": [...], "dropped": n}`` where each placed item is
    ``{"start", "end", "text"}``.
    """
    gaps = free_gaps(speech, duration, min_gap)
    placed: list[dict[str, Any]] = []
    used: list[int] = []
    dropped = 0
    for item in descriptions or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        wanted_at = _positive(item.get("start")) or 0.0
        floor = _positive(min_gap)
        ceiling = _positive(max_seconds)
        # What the description needs to be said in full (capped by the caller's
        # own limit, so one very long paragraph cannot demand a whole minute).
        need = max(floor, min(speaking_seconds(text), ceiling)) if ceiling else \
            max(floor, speaking_seconds(text))

        eligible: list[tuple[int, float, float]] = []
        for index, (gap_start, gap_end) in enumerate(gaps):
            if index in used:
                continue
            # The pause must still have room once the description's own start
            # has passed; a pause that is already over cannot hold anything.
            if gap_end - wanted_at < floor:
                continue
            starts_at = max(gap_start, wanted_at)
            available = gap_end - starts_at
            if available < floor:
                continue
            eligible.append((index, starts_at, available))

        chosen = next((item for item in eligible if item[2] >= need), None)
        if chosen is None and eligible:
            chosen = eligible[0]
        if chosen is None:
            dropped += 1
            continue
        index, starts_at, available = chosen
        budget = min(max(floor, ceiling), available)
        fitted = shorten_to_seconds(text, budget)
        if not fitted:
            dropped += 1
            continue
        used.append(index)
        placed.append({
            "start": round(starts_at, 3),
            "end": round(starts_at + budget, 3),
            "text": fitted,
            "source_start": item.get("start"),
            "source_end": item.get("end"),
        })
    placed.sort(key=lambda item: item["start"])
    return {"placed": placed, "dropped": dropped, "gaps": len(gaps)}


def speaking_seconds(text: str) -> float:
    """How long ``text`` takes to say at :data:`WORDS_PER_SECOND`."""
    words = len(str(text or "").split())
    return words / WORDS_PER_SECOND if words else 0.0


def shorten_to_seconds(text: str, seconds: float) -> str:
    """Trim ``text`` to the words that fit ``seconds`` of speech.

    Whole words only, ending at the last sentence-ish boundary when one is
    available, so a shortened description still reads as a sentence.
    """
    limit = int(max(1.0, _positive(seconds) * WORDS_PER_SECOND))
    words = str(text or "").split()
    if len(words) <= limit:
        return str(text or "").strip()
    kept = words[:limit]
    # Prefer ending on a sentence boundary inside the budget.
    for index in range(len(kept) - 1, max(-1, len(kept) - 12), -1):
        if kept[index].endswith((".", "!", "?", ";")):
            kept = kept[: index + 1]
            break
    return " ".join(kept).strip()


def _chunk_step(chunk_seconds: float) -> float:
    step = _positive(chunk_seconds)
    if not step:
        step = AD_CHUNK_SECONDS
    return max(1.0, step)


def _positive(value: Any) -> float:
    """``value`` as a non-negative float (``0`` for anything unusable).

    ``None`` is a real state here - a model may not know when a phrase was
    spoken, and a probe may fail - so it is answered with 0 rather than raising.
    """
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, number)
