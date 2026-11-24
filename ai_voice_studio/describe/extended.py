"""Extended audio description: stop the film, speak, go on.

Standard audio description is spoken *into* the pauses of the soundtrack, which
is why :func:`ai_voice_studio.describe.pipeline.fit_into_gaps` exists.  A film
whose dialogue is wall to wall has no such pauses: every description either
lands on top of the dialogue or is dropped.  **Extended** audio description
answers that by stopping the film - the picture holds on the frame it had
reached and the soundtrack goes quiet - speaking the description, and then going
on.  The film gets longer; nobody misses a word of it.

What this module produces, all of it from plain numbers:

* :func:`plan_extended` - where each description stops the film, how long the
  film is held, and what every one of those moments is called on the **extended**
  timeline (the timeline of the described film, pauses included).
* :func:`extended_segments` - the same descriptions on that extended clock, with
  the held time attached, so the script and the subtitle track match the
  described film rather than the original one.
* :func:`render_command` - the FFmpeg invocation that really does it: each stop
  holds the picture (``tpad=stop_mode=clone`` / ``start_mode=clone``), the film's
  own soundtrack is silenced over the same stretch (``apad``), and the spoken
  narration - written on this same extended timeline - is mixed on top.

Only the last of those needs anything the application cannot do in pure Python,
and it is the only part that cannot be verified without FFmpeg; the plan, the
timings and the documents are exact.

This module also writes the **standard** described film - the other half of the
"three outputs" an audio description project produces (the text, the spoken
track, and the film with the description in it).  There the film is not stopped
at all: its picture is copied untouched and its own soundtrack is mixed with the
narration, which is already laid out on the film's own clock.  Which of the two
tracks gives way is the **track priority**: the film can be mixed as it is,
brought down for its whole length, or - *Duck, don't lower* - brought down only
while a description is actually being spoken, which is written as a gain
trapezoid around each spoken description (:func:`ducking_expression`).  See
:func:`mix`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from typing import Any

from ..constants import (
    AD_DEFAULT_TRACK_PRIORITY,
    AD_DUCK_EDGE_SECONDS,
    AD_DUCK_LEVEL,
    AD_DUCK_MAX_EXPRESSION,
    AD_DUCK_MERGE_SECONDS,
    AD_MIN_PAUSE_SECONDS,
    AD_PAUSE_PADDING,
    AD_PLACEMENT_EXTENDED,
    AD_PLACEMENT_STANDARD,
    AD_TRACK_PRIORITY_DUCK,
    AD_TRACK_PRIORITY_GAIN,
)
from . import output as output_module
from . import pipeline

log = logging.getLogger(__name__)

Segment = dict[str, Any]

#: Added after a source's own name for the described film: ``My Film - described``.
RENDER_SUFFIX = " - described"

#: The spoken track of a described film is written on the *extended* timeline, so
#: it is named apart from a standard ``- narration`` track: the two have the same
#: descriptions but different clocks, and confusing them would put the voice in
#: the wrong place.
NARRATION_SUFFIX = " - extended narration"

#: Container the described film is written in (H.264/AAC in MP4 plays anywhere).
RENDER_EXTENSION = ".mp4"


class ExtendedRenderError(RuntimeError):
    """The described film could not be rendered."""


def is_extended(placement: Any) -> bool:
    """True when ``placement`` asks for the film to be stopped."""
    return str(placement or "").strip().lower() == AD_PLACEMENT_EXTENDED


def normalise_placement(placement: Any) -> str:
    """``placement`` as one of the two known values (standard when unsure)."""
    return (AD_PLACEMENT_EXTENDED if is_extended(placement)
            else AD_PLACEMENT_STANDARD)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def pause_seconds(text: str, measured: float | None = None,
                  padding: float = AD_PAUSE_PADDING,
                  minimum: float = AD_MIN_PAUSE_SECONDS) -> float:
    """How long the film is held for one description.

    ``measured`` is the real length of the spoken audio when the caller has
    already synthesized it - the honest number, because the pause exists exactly
    as long as the description takes to say.  Without it the text is estimated
    from :func:`pipeline.speaking_seconds`, plus a short breath before the
    picture moves again so a description is never clipped by the film restarting.
    """
    if measured is None:
        spoken = pipeline.speaking_seconds(text)
    else:
        try:
            spoken = max(0.0, float(measured))
        except (TypeError, ValueError):
            spoken = pipeline.speaking_seconds(text)
    return round(max(float(minimum), spoken + max(0.0, float(padding))), 3)


def plan_extended(descriptions: Sequence[Segment],
                  duration: float = 0.0,
                  *,
                  clip_seconds: Sequence[float | None] | None = None,
                  padding: float = AD_PAUSE_PADDING,
                  minimum: float = AD_MIN_PAUSE_SECONDS) -> dict[str, Any]:
    """Where the described film stops, and what the extended clock says.

    Each description stops the film at its own place in the video (its chunk
    boundary, or the moment the describer chose for it) and holds it while the
    description is spoken.  Two rules keep the result playable:

    * stop times are made non-decreasing, so the film is never wound backwards
      even if a description is timed before the one before it; and
    * a stop time is clamped into ``[0, duration]``, so a description the model
      timed past the end of the film only stops it at the end.

    ``clip_seconds[i]`` is the measured spoken length of description ``i`` once
    it has been synthesized (``None`` estimates it from the text).

    Returns ``{"stops", "count", "original_duration", "added_seconds",
    "extended_duration", "added_ratio"}``.  Each stop is ``{"index",
    "video_time", "pause", "extended_start", "text", "chunk"}``, where
    ``extended_start`` is the moment that description is heard in the described
    film.
    """
    span = max(0.0, _number(duration))
    stops: list[dict[str, Any]] = []
    cumulative = 0.0
    previous = 0.0
    for item in descriptions or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        wanted = _number(item.get("start"))
        if span:
            wanted = min(wanted, span)
        video_time = max(previous, wanted)
        index = len(stops)
        measured = None
        if clip_seconds is not None and index < len(clip_seconds):
            measured = clip_seconds[index]
        hold = pause_seconds(text, measured, padding, minimum)
        stops.append({
            "index": index,
            "video_time": round(video_time, 3),
            "pause": hold,
            "extended_start": round(video_time + cumulative, 3),
            "text": text,
            "chunk": item.get("chunk"),
        })
        cumulative += hold
        previous = video_time
    added = round(cumulative, 3)
    return {
        "stops": stops,
        "count": len(stops),
        "original_duration": round(span, 3),
        "added_seconds": added,
        "extended_duration": round(span + added, 3),
        "added_ratio": round(added / span, 4) if span else 0.0,
    }


def extended_segments(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The descriptions on the extended clock, with their held time attached.

    This is what the documents are written from, so an SRT for a described film
    lines up with the described film: a description starts when the film has
    stopped for it and lasts as long as that stop.
    """
    segments: list[dict[str, Any]] = []
    for stop in plan.get("stops", []) or []:
        start = _number(stop.get("extended_start"))
        pause = _number(stop.get("pause"))
        segments.append({
            "start": round(start, 3),
            "end": round(start + pause, 3),
            "text": stop.get("text", ""),
            "pause": round(pause, 3),
            "video_start": _number(stop.get("video_time")),
            "chunk": stop.get("chunk"),
        })
    return segments


def placement_summary(plan: dict[str, Any], films: Sequence[str] = ()) -> str:
    """One line the completion dialog can show about an extended run."""
    count = int(plan.get("count", 0) or 0)
    if not count:
        return "No description needed the film to be stopped."
    added = pipeline.format_duration(plan.get("added_seconds", 0.0))
    total = pipeline.format_duration(plan.get("extended_duration", 0.0))
    percent = round(float(plan.get("added_ratio", 0.0) or 0.0) * 100)
    line = (
        f"The film was stopped {count} time(s) for {added} of description; "
        f"the described film is {total} long"
    )
    line += f" ({percent}% longer)." if percent else "."
    if films:
        line += " Described film: " + ", ".join(films) + "."
    return line


# ---------------------------------------------------------------------------
# The render
# ---------------------------------------------------------------------------
def render_stem(source_path: str) -> str:
    """``My Film - described`` for ``My Film.mp4``."""
    return output_module.output_stem(source_path) + RENDER_SUFFIX


def described_path(source_path: str, output_dir: str) -> str:
    """Where the described film is written for one source video."""
    return os.path.join(output_dir, render_stem(source_path) + RENDER_EXTENSION)


def narration_stem(source_path: str) -> str:
    """The name of the spoken track that goes with a described film."""
    return output_module.output_stem(source_path) + NARRATION_SUFFIX


def render_parts(plan: dict[str, Any], duration: float = 0.0) -> list[dict[str, float]]:
    """The stretches of film in the described version, and the holds between them.

    Read the described film as a sentence: *play* ``[0, t1)``, *hold* ``p1``,
    *play* ``[t1, t2)``, *hold* ``p2``, ... , *play* ``[tk, end)``.  Each entry
    here is one ``play`` stretch, with ``hold_before`` for the hold that comes
    just before it (the first frame frozen backwards) and ``hold_after`` for a
    hold that has no film left to play after it (the last frame frozen on).

    A stretch with no film in it is dropped and its hold joins the next stretch,
    so two descriptions at the same moment hold the picture once for both, and a
    stop at the very end of the film holds the final frame instead of producing
    an empty part.
    """
    span = max(0.0, _number(duration) or _number(plan.get("original_duration")))
    if span <= 0:
        # Without a length there is no way to say where the film stops; the plan
        # and the documents are still usable, the render is not.
        raise ExtendedRenderError(
            "The length of this video could not be read, so it cannot be stopped "
            "for a description."
        )
    parts: list[dict[str, float]] = []
    pending = 0.0
    start = 0.0
    for stop in plan.get("stops", []) or []:
        time = _number(stop.get("video_time"))
        if span:
            time = min(time, span)
        time = max(start, time)
        hold = _number(stop.get("pause"))
        if time > start:
            parts.append({
                "start": round(start, 3), "end": round(time, 3),
                "hold_before": round(pending, 3), "hold_after": 0.0,
            })
            pending = 0.0
            start = time
        pending += hold
    if span > start:
        parts.append({
            "start": round(start, 3), "end": round(span, 3),
            "hold_before": round(pending, 3), "hold_after": 0.0,
        })
        pending = 0.0
    if pending:
        # Nothing is left to play: the last description holds the final frame.
        parts[-1]["hold_after"] = round(pending, 3)
    return parts


def render_graph(plan: dict[str, Any], duration: float = 0.0) -> str:
    """The ``-filter_complex`` graph that stops the film and speaks into the stop.

    One video part and one audio part per stretch of film.  A hold before a part
    clones that part's first frame backwards (``tpad=start_mode=clone``) and a
    hold after the last part clones its last frame on (``stop_mode=clone``); the
    soundtrack is padded with silence over exactly the same stretches
    (``apad``), so the film is quiet while the description speaks.  The parts are
    concatenated into one film, and the spoken narration - input ``1``, written
    on this same extended timeline - is mixed on top with ``normalize=0`` so
    nothing is quietly attenuated.

    The narration is silent except while the film is held, and the film is silent
    over exactly those stretches, so the two never overlap: adding them is the
    right operation rather than a compromise.
    """
    if not plan.get("stops"):
        raise ExtendedRenderError(
            "There is nothing to stop the film for: no descriptions."
        )
    parts = render_parts(plan, duration)
    chain: list[str] = []
    links: list[str] = []
    for index, part in enumerate(parts):
        before = _number(part.get("hold_before"))
        after = _number(part.get("hold_after"))
        pads = ""
        if before:
            pads += f",tpad=start_mode=clone:start_duration={before:.3f}"
        if after:
            pads += f",tpad=stop_mode=clone:stop_duration={after:.3f}"
        chain.append(
            f"[0:v]trim=start={part['start']:.3f}:end={part['end']:.3f},"
            f"setpts=PTS-STARTPTS{pads},format=yuv420p[v{index}]"
        )
        # The soundtrack is quiet for the held time: the film is stopped, so its
        # own audio must not play on underneath the description.  A hold before
        # the stretch is silence pushed in front of it (``adelay``), a hold after
        # it is silence appended to it (``apad``).
        audio = (
            f"[0:a]atrim=start={part['start']:.3f}:end={part['end']:.3f},"
            f"asetpts=PTS-STARTPTS"
        )
        if after:
            audio += f",apad=pad_dur={after:.3f}"
        if before:
            # ``all=1``: one delay value for however many channels the film has.
            audio += f",adelay={round(before * 1000)}:all=1"
        audio += (
            ",aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:"
            f"channel_layouts=stereo[a{index}]"
        )
        chain.append(audio)
        links.append(f"[v{index}][a{index}]")
    chain.append("".join(links) + f"concat=n={len(links)}:v=1:a=1[filmv][filma]")
    chain.append(
        "[1:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:"
        "channel_layouts=stereo[spoke]"
    )
    chain.append("[filma][spoke]amix=inputs=2:duration=first:normalize=0[outa]")
    return ";".join(chain)


def render_command(ffmpeg_exe: str, source: str, narration_path: str,
                   out_path: str, plan: dict[str, Any],
                   duration: float = 0.0) -> list[str]:
    """The FFmpeg argv that writes ``out_path`` for an extended description."""
    return [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", source,
        "-i", narration_path,
        "-filter_complex", render_graph(plan, duration),
        "-map", "[filmv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]


def render(ffmpeg_exe: str | None, source: str, narration_path: str,
           out_path: str, plan: dict[str, Any], duration: float = 0.0,
           timeout: float = 7200.0) -> str:
    """Render the described film; returns ``out_path``.

    Raises :class:`ExtendedRenderError` when FFmpeg is missing or fails, with the
    tail of FFmpeg's own complaint - the one thing that makes a failed render
    fixable.
    """
    if not ffmpeg_exe:
        raise ExtendedRenderError(
            "FFmpeg is required to write a described video and is not installed."
        )
    if not os.path.isfile(source):
        raise ExtendedRenderError(f"Source video is gone: {source}")
    if not os.path.isfile(narration_path):
        raise ExtendedRenderError(
            f"The narration track is missing: {narration_path}"
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    args = render_command(ffmpeg_exe, source, narration_path, out_path, plan,
                          duration)
    log.info("Rendering the described film: %s", out_path)
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtendedRenderError(
            "Writing the described video ran out of time "
            f"({int(timeout)} s). Try a shorter video or fewer descriptions."
        ) from exc
    except OSError as exc:
        raise ExtendedRenderError(f"Could not run FFmpeg: {exc}") from exc
    if result.returncode != 0 or not os.path.isfile(out_path):
        raise ExtendedRenderError(
            "FFmpeg could not write the described video: "
            + (result.stderr or "").strip()[-400:]
        )
    return out_path


# ---------------------------------------------------------------------------
# The standard described film: the film, with the description mixed over it
# ---------------------------------------------------------------------------
def track_priority_gain(priority: Any) -> float:
    """The film soundtrack's gain for a *track priority* choice.

    ``"same"`` (and anything unrecognised, including a missing value) is 1.0 -
    the film's own track untouched, which is what every run before this choice
    existed produced.  ``"higher25"`` and ``"higher50"`` play the film's track at
    0.75 and 0.5 while the description keeps the level the narration voice was
    given in Settings.  ``"duck"`` is 1.0 as well, because it lowers nothing: it
    dips the film only while a description is heard, which is
    :func:`track_priority_duck`.  A raw number is accepted too, so a caller with
    an already computed gain does not have to invent a choice name for it.
    """
    if isinstance(priority, (int, float)) and not isinstance(priority, bool):
        number = float(priority)
    else:
        key = str(priority or "")
        if key not in AD_TRACK_PRIORITY_GAIN:
            key = AD_DEFAULT_TRACK_PRIORITY
        number = float(AD_TRACK_PRIORITY_GAIN[key])
    if number != number:  # NaN
        return 1.0
    return max(0.0, min(number, 1.0))


def track_priority_duck(priority: Any) -> float:
    """The level the film's own sound dips to while a description speaks.

    ``"duck"`` is the one choice that leaves the film's own level alone
    everywhere and brings it down *only* while a description is actually being
    heard, so its resting gain is 1.0 (:func:`track_priority_gain`) and the dip
    lives here: ``AD_DUCK_LEVEL``, a quarter of the film's own level, about
    -12 dB.

    Every other choice - and a missing or unrecognised value - returns 1.0, which
    means "this film never dips", so a run that did not ask for it is mixed
    exactly as it always was.  A raw number is accepted too, so a caller with an
    already computed level does not have to invent a choice name for it.
    """
    if isinstance(priority, (int, float)) and not isinstance(priority, bool):
        number = float(priority)
    elif str(priority or "") == AD_TRACK_PRIORITY_DUCK:
        number = float(AD_DUCK_LEVEL)
    else:
        return 1.0
    if number != number:  # NaN
        return 1.0
    return max(0.0, min(number, 1.0))


def ducking_spans(turns: Sequence[Any],
                  gap: float = AD_DUCK_MERGE_SECONDS) -> list[tuple[float, float]]:
    """The stretches of film to dip, from the moments the voice is heard in.

    ``turns`` is what the narration track recorded while it was written
    (``narration.write_track``): one ``[start, end]`` pair per spoken
    description, in seconds on the film's own clock.  Anything unusable is
    dropped and the rest is sorted, because a dip that began at the wrong moment
    would sit on the film's own sound rather than on the voice.

    Two descriptions closer together than ``gap`` become **one** stretch: the
    film coming back up for a fraction of a second between two back-to-back
    descriptions would be heard as its own sound flapping, which is worse than
    either extreme.
    """
    spans: list[tuple[float, float]] = []
    for turn in turns or ():
        try:
            start = float(turn[0])
            end = float(turn[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if start != start or end != end:  # NaN
            continue
        start = max(0.0, start)
        end = max(0.0, end)
        if end > start:
            spans.append((start, end))
    spans.sort()
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= max(0.0, float(gap)):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def ducking_expression(spans: Sequence[tuple[float, float]], level: Any,
                       gain: float = 1.0, edge: float = AD_DUCK_EDGE_SECONDS,
                       limit: int = AD_DUCK_MAX_EXPRESSION) -> str:
    """The ``volume`` expression that dips the film under the voice, or ``""``.

    One term per stretch.  ``clip((t-a)/e,0,1)-clip((t-b)/e,0,1)`` is 0 before a
    description, ramps to 1 over the first ``e`` seconds of it, stays 1 while it
    is spoken, and ramps back to 0 over the ``e`` seconds after it - a trapezoid,
    so the film is brought down and let back up instead of clicking.  The terms
    cannot overlap (see :func:`ducking_spans`), so their sum is "how much of the
    dip is in effect right now", and the expression is

    ``gain * (1 - (1 - level) * min(1, sum))``

    ``gain`` is the film's resting level (1.0 for the ducking choice, which does
    not lower the film at all), so asking for both a lower film and a dip gets
    both.  ``""`` is returned when there is nothing to dip - no descriptions at
    all, or a level of 1.0 - or when the expression would not fit on one command
    line (``limit``); the caller then mixes the two tracks as they are rather
    than failing to write the film.
    """
    depth = track_priority_duck(level)
    resting = track_priority_gain(gain)
    if depth >= 1.0:
        return ""
    seconds = max(0.001, float(edge))
    terms = [
        f"clip((t-{start:.3f})/{seconds:g},0,1)"
        f"-clip((t-{end:.3f})/{seconds:g},0,1)"
        for start, end in spans if end > start
    ]
    if not terms:
        return ""
    body = f"(1-{1.0 - depth:g}*min(1,{'+'.join(terms)}))"
    expression = body if resting >= 1.0 else f"{resting:g}*{body}"
    return expression if len(expression) <= int(limit) else ""


def mix_graph(film_volume: float = 1.0, duck_level: Any = 1.0,
              duck_turns: Sequence[Any] = ()) -> str:
    """The ``-filter_complex`` graph that mixes the description over the film.

    Standard audio description is spoken *into* the pauses of the existing
    soundtrack, so the described film is the film itself: nothing is stopped, no
    frame is held, and the video is passed through **untouched**.  All that
    happens is audio - the film's own soundtrack, mixed with the spoken track,
    which is already written on the film's own clock and is silent wherever it
    has no description to speak.

    ``normalize=0`` deliberately: FFmpeg's mixer divides the sum of its inputs by
    their number by default, which would make both the film and the description
    half as loud.  Nothing here overlaps, so nothing needs attenuating.

    ``film_volume`` is the *track priority* the user chose (see
    :func:`track_priority_gain`): it is the gain put on the film's own
    soundtrack - 1.0 leaves it alone, and a lower value brings the film down so
    the description sits above it.  The description's own input is never
    touched, so "25 percent higher" means exactly that: the film is quieter by a
    quarter while the voice stays at the level the user tuned.  With the default
    of 1.0 no ``volume`` stage is emitted at all, so the graph of a run that did
    not ask for this is byte for byte what it always was.

    ``duck_level`` and ``duck_turns`` are *Duck, don't lower*: the level the
    film's own sound dips to (``AD_DUCK_LEVEL``, or 1.0 for a film that never
    dips) and the moments the voice is actually heard in, which the narration
    track recorded as it was written.  The two become a single ``volume`` stage
    on the film's input whose gain is a trapezoid around every spoken
    description (:func:`ducking_expression`), so the film plays at its own level
    between descriptions and steps out of the way under them - and with nothing
    to dip no stage is emitted at all.
    """
    film = ("[0:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:"
            "channel_layouts=stereo")
    gain = track_priority_gain(film_volume)
    dip = ducking_expression(ducking_spans(duck_turns), duck_level, gain)
    if dip:
        chain = [film + "[rawfilm]",
                 f"[rawfilm]volume='{dip}':eval=frame[film]"]
    elif gain < 1.0:
        chain = [film + "[rawfilm]",
                 f"[rawfilm]volume={gain:g}[film]"]
    else:
        chain = [film + "[film]"]
    chain += [
        "[1:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:"
        "channel_layouts=stereo[voice]",
        "[film][voice]amix=inputs=2:duration=first:normalize=0[outa]",
    ]
    return ";".join(chain)


def mix_attempts(source: str, narration_path: str, out_path: str,
                 film_volume: float = 1.0, duck_level: Any = 1.0,
                 duck_turns: Sequence[Any] = ()) -> list[list[str]]:
    """The FFmpeg invocations that produce a standard described film, in order.

    The first is the honest one - the film's picture copied as it is and its
    soundtrack mixed with the description.  What follows are the ways that can
    legitimately not work, each one a film that still plays:

    * a video whose codec cannot be stored in an MP4 as it is (an old AVI, a
      VP9 WebM) is re-encoded rather than refused;
    * a video with **no soundtrack at all** (a screen recording) has nothing to
      mix with, so the description becomes the whole soundtrack.

    Every attempt keeps the video's own frames unless re-encoding is the only way
    to write them, which is why the copy comes first.

    When *Duck, don't lower* was chosen the ducking mix comes first and the
    plain mix - both tracks as they are - is tried next, before the narration is
    ever used on its own: a dip that could not be applied costs the listener the
    dip, never the film's own sound.

    ``film_volume``, ``duck_level`` and ``duck_turns`` are carried into the mix,
    and only into the mix: a video with no soundtrack has no film track to bring
    down, so the attempts that map the narration alone are identical at any
    track priority.
    """
    attempts: list[list[str]] = []

    def build(graph: str | None, video_args: list[str], audio_map: str) -> list[str]:
        args = ["-i", source, "-i", narration_path]
        if graph:
            args += ["-filter_complex", graph]
        args += ["-map", "0:v:0", "-map", audio_map]
        args += video_args
        args += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                 out_path]
        return args

    copy_video = ["-c:v", "copy"]
    encode_video = ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p"]
    ducked = mix_graph(film_volume, duck_level, duck_turns)
    plain = mix_graph(film_volume)
    graphs = [ducked]
    if ducked != plain:
        # The dip is a refinement: if FFmpeg cannot apply it (an expression too
        # long for one command line, a build without the filter), the film is
        # still written with both of its tracks rather than losing its own sound.
        graphs.append(plain)
    # Both mixes are tried before dropping the film's own soundtrack: a video
    # whose *codec* cannot go into an MP4 must be re-encoded, not have its sound
    # taken away.
    for graph, audio_map in ([(g, "[outa]") for g in graphs]
                             + [(None, "1:a:0")]):
        for video_args in (copy_video, encode_video):
            attempts.append(build(graph, video_args, audio_map))
    return attempts


def mix(ffmpeg_exe: str | None, source: str, narration_path: str,
        out_path: str, timeout: float = 7200.0,
        film_volume: float = 1.0, duck_level: Any = 1.0,
        duck_turns: Sequence[Any] = ()) -> str:
    """Write the standard described film (the film with its description); returns
    its path.

    ``film_volume`` is the track priority - the gain on the film's own soundtrack
    while the description is left where the narration settings put it - and
    ``duck_level`` with ``duck_turns`` is *Duck, don't lower*, which brings the
    film's own soundtrack down only while a description is spoken (see
    :func:`mix_graph`).

    Raises :class:`ExtendedRenderError` when FFmpeg is missing or every way of
    writing the file failed, with the tail of FFmpeg's own complaint.
    """
    if not ffmpeg_exe:
        raise ExtendedRenderError(
            "FFmpeg is required to write a described video and is not installed."
        )
    if not os.path.isfile(source):
        raise ExtendedRenderError(f"Source video is gone: {source}")
    if not os.path.isfile(narration_path):
        raise ExtendedRenderError(
            f"The narration track is missing: {narration_path}"
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    last = ""
    for args in mix_attempts(source, narration_path, out_path, film_volume,
                             duck_level, duck_turns):
        argv = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", *args]
        log.debug("Mixing the described film: %s", out_path)
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExtendedRenderError(
                "Writing the described video ran out of time "
                f"({int(timeout)} s). Try a shorter video."
            ) from exc
        except OSError as exc:
            raise ExtendedRenderError(f"Could not run FFmpeg: {exc}") from exc
        if result.returncode == 0 and os.path.isfile(out_path):
            return out_path
        last = (result.stderr or "").strip()[-400:]
    raise ExtendedRenderError(
        "FFmpeg could not write the described video: " + (last or "no reason "
                                                          "given")
    )


def _number(value: Any) -> float:
    """A finite, non-negative float (0 for anything unusable)."""
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, number)
