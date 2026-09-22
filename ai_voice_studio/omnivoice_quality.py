"""Drone ("no speech") detection and recovery for the OmniVoice engines.

OmniVoice sometimes renders a chunk as a steady low-frequency drone instead of
a voice — a buzz with no speech in it (upstream k2-fsa/OmniVoice issues #37
``#73`` and #144).  The app has two front-ends for the same model, and both get
the same bad draw at the same rate:

* :mod:`ai_voice_studio.omnivoice` — the direct engine (``omnivoice-triton`` in
  a worker subprocess);
* :mod:`ai_voice_studio.omnivoice_server` — the ``omnivoice-server`` HTTP API,
  which *reports* the drone with an ``X-No-Speech-Detected`` header but does
  not repair it, and whose header used to be dropped on the floor, so a drone
  was written straight into the recorded segment.

Everything both engines need lives here, so the policy cannot drift between
them: judge the audio that was actually produced, draw a bad chunk again (a
drone is a bad roll, not a bad setting — measured on this machine at 28% of the
chunks of a real Hindi chapter, 2 of 6 draws of one paragraph), and only when
every draw is a drone split the chunk into sentence-sized pieces and re-record
those.  Pieces are always joined back in order, so one segment stays exactly
one file.

Detection
---------
The detector mirrors the server's own (250 ms frames, zero-crossing rate) with
thresholds calibrated here on a real 35-minute chapter that came out of
OmniVoice with audible buzzes: 30 stretches of 4-51 seconds with a mean
zero-crossing rate of 0.0001-0.011 and RMS far above silence — 6.8 minutes of
drone in total.  On top of the server's whole-output ratio it locates *where*
inside a take the drone sits (:func:`degenerate_spans`): a chapter paragraph can
hold 20-60 seconds of drone and still score a healthy overall ratio, and the
whole-output ratio cannot express that.
"""

from __future__ import annotations

import random
from typing import Any, Callable, NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Recovery policy
# ---------------------------------------------------------------------------
# The failure is drawn per request at a roughly constant rate whatever the text
# size (2/6 whole paragraphs and 2/3 sentences of the same text failed, so
# smaller text is no safer).  With a third of draws bad, four attempted draws
# leave about 1% of chunks buzzing and the *expected* cost stays near one draw
# per chunk — only the unlucky third pays for a second.  A chunk that cannot be
# repaired at all keeps the best-looking take and is reported, so a segment
# that may still buzz is named in the app log instead of being shipped
# silently.

#: How many times one chunk may be drawn before the sentence-level repair runs.
DRONE_ATTEMPTS = 4

#: Attempts per piece while the chunk is being re-recorded piece by piece.
DRONE_REPAIR_ATTEMPTS = 3

#: Cap on how many pieces one chunk may be re-recorded in, so a very long chunk
#: that keeps droning cannot turn the repair into hundreds of attempts.
DRONE_MAX_PIECES = 8

#: Levels of splitting the repair may use: the whole chunk into sentences, and
#: a stubborn sentence into clause-sized halves.
DRONE_REPAIR_MAX_DEPTH = 2

#: Never re-record a piece shorter than this (below it the audio is disjoint).
DRONE_REPAIR_MIN_CHARS = 60

#: Draws the repair of one chunk may spend in total (splitting included) before
#: it settles for the best take it has.  Only reached by a chunk this voice
#: keeps choking on.
DRONE_REPAIR_MAX_DRAWS = 20

#: Piece size the repair aims for at the top level.  Short generations are what
#: upstream issue #144 recovered with; here they are the last resort after the
#: whole chunk has already been drawn ``DRONE_ATTEMPTS`` times.
DRONE_REPAIR_CHARS = 250

#: A retry rotates the seed by this much, so a seeded project stays
#: reproducible on its first draw while every retry is a different roll.
_SEED_STRIDE = 7919

#: A retry of one *request* moves the seed by this much instead.  A different
#: stride from ``_SEED_STRIDE``, so a request re-sent inside one attempt never
#: lands on the seed the next drone attempt is about to use.
RETRY_SEED_STRIDE = 104729


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

#: OmniVoice output sample rate.
SAMPLE_RATE = 24_000

#: Analysis window, in milliseconds (the server's ``DEGENERATE_FRAME_MS``).
FRAME_MS = 250

#: Below this zero-crossing rate a window has no speech in it (server value).
SPEECH_ZCR_THRESHOLD = 0.04

#: Below *this* zero-crossing rate a window is a drone, not a slow sentence.
#: Calibrated against a real chapter: audible drones sit at 0.0001-0.011, while
#: ordinary narration still has windows under the server's own 0.04 bar.
DRONE_ZCR_THRESHOLD = 0.01

#: Windows quieter than this are silence and are not judged at all.
SILENCE_RMS = 0.005

#: Minimum fraction of audible windows that must look like speech.
MIN_SPEECH_RATIO = 0.15

#: Takes shorter than this are never judged (mirrors the server).
MIN_JUDGE_SECONDS = 1.0

#: A drone stretch must last this long to count as a fault.  The measured
#: failures were 4-51 s; 3 s only added false alarms on slow narration.
MIN_SPAN_SECONDS = 4.0

#: Audible drone windows closer than this belong to the same stretch.
SPAN_GAP_SECONDS = 0.5

#: Frames analysed per pass.  A recorded segment can be half an hour long (7.2
#: million samples), and turning all of it into float64 at once would need
#: hundreds of megabytes; a block of frames is a few tens.
_BLOCK_FRAMES = 1024


class Verdict(NamedTuple):
    """What one synthesis result looks like."""

    #: Fraction of audible windows that look like speech (1.0 = nothing to judge).
    speech_ratio: float
    #: Longest audible non-speech stretch, in seconds.
    longest_drone_s: float
    #: Total audible non-speech time, in seconds.
    drone_s: float
    #: True when the server flagged this response itself.
    server_flagged: bool
    #: True when the audio must not be written into a recording as-is.
    bad: bool

    @property
    def reason(self) -> str:
        """Short human-readable reason, for logs and warning messages."""
        if self.server_flagged:
            return "server reported no speech"
        if self.speech_ratio < MIN_SPEECH_RATIO:
            return f"only {self.speech_ratio:.0%} of the audio sounds like speech"
        if self.longest_drone_s >= MIN_SPAN_SECONDS:
            return f"{self.longest_drone_s:.1f}s of drone inside the audio"
        return ""


def _flat(samples) -> tuple:
    """1-D samples plus the divisor that puts them on the [-1, 1] scale.

    The engine hands us int16 PCM at full scale; a float array is already
    normalised, so it is taken as it is.
    """
    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        return arr, 32768.0
    return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0,
                         neginf=0.0), 1.0


def frame_stats(samples, frame_ms: int = FRAME_MS) -> tuple:
    """Per-frame ``(audible, speech, drone)`` boolean arrays.

    ``speech`` uses the server's threshold, ``drone`` the stricter one; all
    three are empty when the audio is shorter than a single frame.  Frames are
    analysed in blocks, so judging a whole 30-minute segment costs a few tens
    of megabytes rather than a gigabyte.
    """
    signal, scale = _flat(samples)
    frame_len = max(1, int(SAMPLE_RATE * frame_ms / 1000))
    n_frames = signal.shape[0] // frame_len
    if n_frames == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty

    audible = np.zeros(n_frames, dtype=bool)
    speech = np.zeros(n_frames, dtype=bool)
    drone = np.zeros(n_frames, dtype=bool)
    for first in range(0, n_frames, _BLOCK_FRAMES):
        last = min(first + _BLOCK_FRAMES, n_frames)
        block = signal[first * frame_len:last * frame_len].astype(np.float64)
        block = block.reshape(last - first, frame_len) / scale
        rms = np.sqrt(np.mean(block ** 2, axis=1))
        crossings = np.count_nonzero(
            np.diff(np.sign(block), axis=1) != 0, axis=1
        )
        zcr = crossings / frame_len
        block_audible = rms >= SILENCE_RMS
        audible[first:last] = block_audible
        speech[first:last] = block_audible & (zcr >= SPEECH_ZCR_THRESHOLD)
        drone[first:last] = block_audible & (zcr < DRONE_ZCR_THRESHOLD)
    return audible, speech, drone


def _ratio(audible, speech) -> float:
    """Speech windows over audible windows (1.0 when nothing is audible)."""
    audible_count = int(np.count_nonzero(audible))
    if audible_count == 0:
        return 1.0
    return float(np.count_nonzero(speech) / audible_count)


def _spans(drone, frame_ms: int, min_span_s: float, gap_s: float) -> list:
    """Merge drone-frame runs into ``(start_s, end_s)`` stretches."""
    if drone.size == 0:
        return []
    frame_s = frame_ms / 1000.0
    runs: list = []
    start = None
    for i, is_drone in enumerate(drone):
        if is_drone and start is None:
            start = i
        elif not is_drone and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, int(drone.shape[0])])

    merged: list = []
    for start_i, end_i in runs:
        if merged and (start_i - merged[-1][1]) * frame_s <= gap_s:
            merged[-1][1] = end_i
        else:
            merged.append([start_i, end_i])
    return [
        (round(start_i * frame_s, 2), round(end_i * frame_s, 2))
        for start_i, end_i in merged
        if (end_i - start_i) * frame_s >= min_span_s
    ]


def speech_window_ratio(samples, frame_ms: int = FRAME_MS) -> float:
    """Fraction of audible windows whose zero-crossing rate looks like speech.

    Mirrors the server's ``speech_window_ratio``.  Returns ``1.0`` when there
    is nothing audible to judge, so "no signal to measure" is never treated as
    a failure.
    """
    audible, speech, _ = frame_stats(samples, frame_ms)
    return _ratio(audible, speech)


def degenerate_spans(
    samples,
    frame_ms: int = FRAME_MS,
    min_span_s: float = MIN_SPAN_SECONDS,
    gap_s: float = SPAN_GAP_SECONDS,
) -> list:
    """Audible drone stretches of ``samples`` as ``(start_s, end_s)``.

    Only stretches at least ``min_span_s`` long are returned; drone windows
    separated by less than ``gap_s`` (a moment of near-silence inside one
    buzz) count as one stretch.
    """
    _, _, drone = frame_stats(samples, frame_ms)
    return _spans(drone, frame_ms, min_span_s, gap_s)


def judge(samples, server_flagged: bool = False) -> Verdict:
    """Judge one synthesis result.

    ``bad`` is what an engine acts on: the server said the response holds no
    speech, the whole output is a drone, or it contains a drone stretch long
    enough to be heard as a fault in the recorded file.  One pass over the
    audio measures all of it.
    """
    audible, speech, drone = frame_stats(samples)
    ratio = _ratio(audible, speech)
    spans = _spans(drone, FRAME_MS, MIN_SPAN_SECONDS, SPAN_GAP_SECONDS)
    drone_s = round(sum(end - start for start, end in spans), 2)
    longest = max((end - start for start, end in spans), default=0.0)
    too_short = int(np.asarray(samples).size) < MIN_JUDGE_SECONDS * SAMPLE_RATE
    bad = bool(server_flagged)
    if not bad and not too_short:
        bad = ratio < MIN_SPEECH_RATIO or longest >= MIN_SPAN_SECONDS
    return Verdict(
        speech_ratio=round(float(ratio), 4),
        longest_drone_s=round(float(longest), 2),
        drone_s=drone_s,
        server_flagged=bool(server_flagged),
        bad=bad,
    )


def describe(samples) -> str:
    """One-line description of a synthesis result, for the log."""
    verdict = judge(samples)
    return (
        f"speech ratio {verdict.speech_ratio:.2f}, "
        f"drone {verdict.drone_s:.1f}s (longest {verdict.longest_drone_s:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Text splitting for the repair
# ---------------------------------------------------------------------------
_SENTENCE_END = ".!?\u0964\u0965\u3002\uff01\uff1f\u2026\"\u201d\u2019\u00bb"


def split_sentences(paragraph: str) -> list:
    """Sentences of ``paragraph``, terminators included.

    Understands the Devanagari danda and CJK full stops through
    ``_SENTENCE_END``, so a Hindi or Chinese chapter splits at its own
    punctuation rather than being hard-cut.
    """
    sentences: list = []
    start = 0
    for i, ch in enumerate(paragraph):
        if ch in _SENTENCE_END:
            sentences.append(paragraph[start:i + 1].strip())
            start = i + 1
    if start < len(paragraph):
        sentences.append(paragraph[start:].strip())
    return [s for s in sentences if s]


def split_at_words(fragment: str, limit: int) -> list:
    """Cut ``fragment`` into equal pieces of about ``limit`` characters.

    Balanced rather than greedy: filling each piece to the limit leaves a
    stubby remainder, and a stub is exactly what the drone repair must not
    send on its own.  A sentence with no spaces at all (Chinese, or a wall of
    Devanagari) is hard cut, which is all that can be done with it.
    """
    if len(fragment) <= limit:
        return [fragment]
    want = max(2, -(-len(fragment) // limit))
    words = fragment.split(" ")
    if len(words) <= 1:
        return [fragment[i:i + limit].strip()
                for i in range(0, len(fragment), limit)
                if fragment[i:i + limit].strip()]
    target = len(fragment) / want
    pieces: list = []
    buf: list = []
    buf_len = 0
    for word in words:
        buf.append(word)
        buf_len += len(word) + 1
        if buf_len >= target and len(pieces) < want - 1:
            pieces.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def split_text_for_repair(
    text: str,
    target_chars: int = DRONE_REPAIR_CHARS,
    max_pieces: int = DRONE_MAX_PIECES,
    min_chars: int = DRONE_REPAIR_MIN_CHARS,
) -> list:
    """Split ``text`` into pieces for a drone repair.

    Unlike the server engine's :func:`split_text_for_server`, *short* text is
    split too: a 270-character paragraph is one server chunk but the repair
    wants it in pieces, and the piece is the unit that gets re-recorded.  The
    number of pieces follows from ``target_chars`` (capped at ``max_pieces``),
    and a sentence with no sentence break in it is cut at a word boundary, so
    the repair always ends up asking the model for something it has not already
    refused.  Same text, same pieces, whichever engine asks.

    Pieces are returned in order and reassemble to the original text (bar
    whitespace), which is what keeps a repaired segment one continuous file.
    """
    text = (text or "").strip()
    if not text:
        return []
    ceiling = max(min_chars, target_chars)
    pieces: list = []
    for sentence in split_sentences(text) or [text]:
        pieces.extend(split_at_words(sentence, ceiling))
    # Sentences are the natural granularity of the repair; only a text with
    # more of them than allowed is merged down to ``max_pieces`` (smallest
    # neighbours first, so no piece grows much beyond the target).
    while len(pieces) > max(1, max_pieces):
        i = min(range(len(pieces) - 1),
                key=lambda k: len(pieces[k]) + len(pieces[k + 1]))
        pieces[i:i + 2] = [pieces[i] + " " + pieces[i + 1]]
    return pieces


# ---------------------------------------------------------------------------
# Drawing and repair
# ---------------------------------------------------------------------------
#: ``draw(text, seed) -> (samples, server_flagged)``: one generation request.
Draw = Callable[[str, "int | None"], "tuple"]


def attempt_seed(seed: int | None, attempt: int) -> int | None:
    """Seed for one draw of a chunk.

    The first attempt keeps the caller's seed (or ``None``, leaving the engine
    to pick its own), so an explicitly seeded project stays reproducible.
    Every later attempt is a *different* draw: upstream measured that no
    generation parameter avoids the drone, so the retry changes nothing but the
    roll of the dice.
    """
    if attempt == 0:
        return int(seed) if seed is not None else None
    if seed is None:
        return random.randint(0, 2 ** 31 - 1)
    return int(seed) + attempt * _SEED_STRIDE


def retry_seed(seed: int | None, retry: int) -> int | None:
    """Seed for re-sending the *same request* (``retry`` 0 = the first send).

    A request can fail without the text being at fault: the server answers
    HTTP 500 when one generation comes back empty (its own log names the text:
    "Generation returned no audio for text: '629'"), which is a bad roll of the
    dice rather than a bad request, and the same text drawn again comes back
    clean.  So the request is sent again with a different draw — but the first
    send keeps the caller's seed, and an *unseeded* request stays unseeded
    (it already samples freely, so pinning it to a seed would be wrong).
    """
    if retry <= 0:
        return int(seed) if seed is not None else None
    if seed is None:
        return None
    return int(seed) + retry * RETRY_SEED_STRIDE


def best_take(
    text: str,
    draw: Draw,
    *,
    seed: int | None = None,
    attempts: int = DRONE_ATTEMPTS,
) -> tuple:
    """Draw ``text`` until a take passes the drone check, or attempts run out.

    Returns ``(samples, verdict, attempts_used)`` — the best-scoring take when
    every attempt was a drone (highest speech ratio, least drone time), so the
    caller always has audio to fall back on.
    """
    best: Any = None
    best_verdict: Any = None
    best_score: Any = None
    used = 0
    for attempt in range(max(1, int(attempts))):
        used = attempt + 1
        samples, flagged = draw(text, attempt_seed(seed, attempt))
        verdict = judge(samples, server_flagged=flagged)
        score = (0 if verdict.bad else 1, verdict.speech_ratio, -verdict.drone_s)
        if best_score is None or score > best_score:
            best, best_verdict, best_score = samples, verdict, score
        if not verdict.bad:
            break
    return best, best_verdict, used


def _repair_limit(depth: int, max_depth: int = DRONE_REPAIR_MAX_DEPTH) -> int:
    """Piece size the repair aims for at one level (smaller when deeper)."""
    if depth >= max_depth:
        return DRONE_REPAIR_CHARS
    return DRONE_REPAIR_MIN_CHARS * 2


def repair_by_pieces(
    text: str,
    draw: Draw,
    *,
    seed: int | None = None,
    depth: int = DRONE_REPAIR_MAX_DEPTH,
    budget: int = DRONE_REPAIR_MAX_DRAWS,
    repair_attempts: int = DRONE_REPAIR_ATTEMPTS,
    max_depth: int = DRONE_REPAIR_MAX_DEPTH,
) -> tuple:
    """Re-record ``text`` in smaller pieces and join them in order.

    Returns ``(samples, unresolved, pieces, draws)`` where ``unresolved`` holds
    ``(text, reason)`` of the pieces that stayed drones and ``pieces`` counts
    the pieces this level produced.  ``depth`` is how many splitting levels are
    still allowed, ``budget`` how many draws may be spent, and the *sum* over
    this level and the levels below it never exceeds the budget (what one piece
    is given for a deeper split is taken out of a pool the siblings no longer
    need).
    """
    # Never split into more pieces than the budget can draw twice over: one
    # round of attempts for each piece, and a pool left for the pieces that
    # need splitting again.
    affordable = max(1, min(DRONE_MAX_PIECES,
                            budget // max(1, repair_attempts * 2)))
    pieces = split_text_for_repair(
        text,
        target_chars=_repair_limit(depth, max_depth),
        max_pieces=affordable,
    )
    if len(pieces) <= 1:
        # Nothing left to split (a single short sentence): one last draw to
        # report on, then the caller keeps the take it already has.
        samples, verdict, used = best_take(
            text, draw, seed=seed, attempts=1
        )
        unresolved = [(text, verdict.reason)] if verdict.bad else []
        return samples, unresolved, 1, used

    pool = max(0, budget - len(pieces) * repair_attempts)
    parts: list = []
    unresolved: list = []
    draws = 0
    for piece in pieces:
        samples, verdict, used = best_take(
            piece, draw, seed=seed, attempts=repair_attempts
        )
        draws += used
        if verdict.bad and depth > 1 and len(piece) > DRONE_REPAIR_MIN_CHARS \
                and pool >= repair_attempts:
            samples, deeper, _, deeper_draws = repair_by_pieces(
                piece, draw, seed=seed, depth=depth - 1, budget=pool,
                repair_attempts=repair_attempts, max_depth=max_depth,
            )
            draws += deeper_draws
            pool = max(0, pool - deeper_draws)
            unresolved.extend(deeper)
        elif verdict.bad:
            unresolved.append((piece, verdict.reason))
        parts.append(samples)

    joined = np.concatenate(parts) if parts else None
    return joined, unresolved, len(pieces), draws


def make_repair_record(
    chunk: str,
    verdict: Verdict,
    *,
    attempts: int,
    pieces: int,
    draws: int,
    unresolved: list,
) -> dict:
    """What had to be done to one chunk, for the log and for the user."""
    return {
        "text": chunk[:120],
        "reason": verdict.reason,
        "attempts": attempts,
        "speech_ratio": verdict.speech_ratio,
        "drone_s": verdict.drone_s,
        "pieces": max(1, pieces),
        "draws": draws,
        "unrepaired": len(unresolved),
        "unresolved": [
            {"text": text[:80], "reason": reason}
            for text, reason in unresolved
        ],
    }


def repair_log(
    chunk: str,
    verdict: Verdict,
    *,
    attempts: int,
    pieces: int,
    draws: int,
    unresolved: list,
) -> str:
    """The log line for a chunk that had to be repaired (or could not be)."""
    if pieces <= 1:
        return (
            "OmniVoice kept returning noise instead of speech (%s) for a "
            "%d-character chunk after %d attempts, and the text cannot be "
            "split further. Listen to this part of the recording. "
            "Text: %.120s" % (verdict.reason, len(chunk), attempts, chunk)
        )
    return (
        "OmniVoice returned noise instead of speech (%s) for a "
        "%d-character chunk; it was re-recorded automatically as %d "
        "pieces (%d extra draws).%s"
        % (
            verdict.reason, len(chunk), pieces, draws,
            "" if not unresolved else (
                f" {len(unresolved)} piece(s) still look like noise: "
                + "; ".join(text for text, _ in unresolved)
            ),
        )
    )


def repair_message(record: dict) -> str:
    """One user-readable line about a repaired (or unrepaired) drone."""
    where = (record.get("text") or "").strip()[:80]
    if record.get("unrepaired"):
        return (
            "OmniVoice kept returning noise instead of speech for "
            f"\"{where}\". That part of this segment may still sound "
            "wrong - check the recording."
        )
    return (
        "OmniVoice returned noise instead of speech for "
        f"\"{where}\"; it was re-recorded automatically."
    )


def recover(
    text: str,
    draw: Draw,
    *,
    seed: int | None = None,
    attempts: int = DRONE_ATTEMPTS,
    repair_attempts: int = DRONE_REPAIR_ATTEMPTS,
    budget: int = DRONE_REPAIR_MAX_DRAWS,
    depth: int = DRONE_REPAIR_MAX_DEPTH,
) -> tuple:
    """One chunk of text as samples, without the OmniVoice drone.

    Returns ``(samples, repair_record_or_None)``; the record describes what had
    to be done, and is ``None`` when the first draw was clean.  A bad draw is
    repaired by drawing the chunk again; when every draw of a chunk is a drone
    — the sign of text this voice keeps choking on — the chunk is re-recorded
    in pieces instead, and a piece that droned too is split again.  Only
    ``samples`` is returned, joined in order, so one segment stays one file.  A
    chunk that stays a drone keeps its best-looking take and is reported
    instead of failing the whole recording.
    """
    import logging  # noqa: PLC0415

    log = logging.getLogger(__name__)

    samples, verdict, used = best_take(
        text, draw, seed=seed, attempts=attempts
    )
    if not verdict.bad:
        if used > 1:
            log.info(
                "OmniVoice drew a drone for a %d-character chunk; "
                "attempt %d came back clean.", len(text), used,
            )
        return samples, None

    repaired, unresolved, pieces, draws = repair_by_pieces(
        text,
        draw,
        seed=seed,
        depth=depth,
        budget=budget,
        repair_attempts=repair_attempts,
    )
    if repaired is None:
        repaired = samples
    log.warning(repair_log(
        text, verdict, attempts=used, pieces=pieces, draws=draws,
        unresolved=unresolved,
    ))
    return repaired, make_repair_record(
        text, verdict, attempts=used, pieces=pieces, draws=draws,
        unresolved=unresolved,
    )
