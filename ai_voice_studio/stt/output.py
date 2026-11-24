"""Speech-to-text output documents.

Turning a list of transcribed segments into a document is independent of which
model produced them, so it lives here: one function per output type
(:func:`format_document`), the time-based split used by the "time by time
separation" file-creation mode (:func:`split_by_time`) and the small
path/extension helpers shared by the transcribe job and the wizard.

A *segment* is a plain dict with ``start`` and ``end`` (seconds, floats; may be
``None`` when a model returns no timing) and ``text`` (the spoken words).
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Sequence
from typing import Any

from ..constants import (
    STT_FORMAT_CSV,
    STT_FORMAT_HTML,
    STT_FORMAT_JSON,
    STT_FORMAT_MARKDOWN,
    STT_FORMAT_SRT,
    STT_FORMAT_TXT,
    STT_FORMAT_VTT,
)
from ..util import sanitize_filename

Segment = dict[str, Any]

#: Audio and video containers the source pickers accept.  FFmpeg (which the
#: application already downloads for MP3/FLAC) decodes the video formats too,
#: so a lecture recorded as .mp4 can be transcribed without a separate step.
MEDIA_EXTENSIONS = {
    # audio
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".amr", ".mp4a",
    # video
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg",
    ".mpeg", ".3gp",
}

#: What each output format's writer expects / produces.
FORMAT_EXTENSIONS = {
    STT_FORMAT_TXT: ".txt",
    STT_FORMAT_HTML: ".html",
    STT_FORMAT_MARKDOWN: ".md",
    STT_FORMAT_SRT: ".srt",
    STT_FORMAT_VTT: ".vtt",
    STT_FORMAT_JSON: ".json",
    STT_FORMAT_CSV: ".csv",
}


def is_media_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS


def scan_source(path: str, recursive: bool = True) -> list[str]:
    """The media files a source entry stands for.

    A file maps to itself (when it is media); a folder maps to every media
    file inside it, sorted, optionally descending into sub-folders.  A folder
    that contains no media files maps to an empty list.
    """
    if os.path.isfile(path):
        return [path] if is_media_file(path) else []
    if not os.path.isdir(path):
        return []
    found: list[str] = []
    if recursive:
        for root, dirs, names in os.walk(path):
            dirs.sort()
            for name in sorted(names):
                candidate = os.path.join(root, name)
                if is_media_file(candidate):
                    found.append(candidate)
    else:
        for name in sorted(os.listdir(path)):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate) and is_media_file(candidate):
                found.append(candidate)
    return found


def output_stem(source_path: str) -> str:
    """A safe document name for a source file (its name without extension)."""
    base = os.path.splitext(os.path.basename(source_path))[0]
    return sanitize_filename(base) or "transcript"


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float | None, *, millisecond_sep: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (WebVTT); empty when None."""
    if seconds is None:
        return ""
    try:
        total = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return ""
    hours, rem = divmod(total, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    whole = int(secs)
    millis = round((secs - whole) * 1000)
    if millis == 1000:  # rounding carried over
        whole += 1
        millis = 0
        if whole == 60:
            whole = 0
            minutes += 1
    return f"{int(hours):02d}:{int(minutes):02d}:{whole:02d}{millisecond_sep}{millis:03d}"


def _clock(seconds: float | None) -> str:
    """A short ``H:MM:SS`` / ``M:SS`` label for plain-text transcripts."""
    if seconds is None:
        return ""
    try:
        total = int(max(0.0, float(seconds)))
    except (TypeError, ValueError):
        return ""
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _as_txt(segments: Sequence[Segment], include_timestamp: bool, title: str) -> str:
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("")
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if include_timestamp:
            lines.append(f"[{_clock(seg.get('start'))}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines).strip() + "\n"


def _as_markdown(segments: Sequence[Segment], include_timestamp: bool,
                 title: str) -> str:
    out: list[str] = []
    if title:
        out.extend([f"# {title}", ""])
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if include_timestamp:
            out.append(f"**{_clock(seg.get('start'))}** {text}")
        else:
            out.append(text)
        out.append("")
    return "\n".join(out).strip() + "\n"


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _as_html(segments: Sequence[Segment], include_timestamp: bool,
             title: str) -> str:
    heading = _html_escape(title) if title else "Transcript"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{heading}</title>",
        "<style>body{font-family:sans-serif;line-height:1.5;max-width:48em;"
        "margin:2em auto}.ts{color:#666;font-family:monospace;"
        "margin-right:.5em}</style>",
        "</head>",
        "<body>",
        f"<h1>{heading}</h1>",
    ]
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if include_timestamp:
            parts.append(
                f'<p><span class="ts">[{_clock(seg.get("start"))}]</span>'
                f"{_html_escape(text)}</p>"
            )
        else:
            parts.append(f"<p>{_html_escape(text)}</p>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


def _as_srt(segments: Sequence[Segment]) -> str:
    blocks: list[str] = []
    index = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        index += 1
        start = format_timestamp(seg.get("start"), millisecond_sep=",")
        end = format_timestamp(seg.get("end"), millisecond_sep=",")
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks).strip() + "\n" if blocks else ""


def _as_vtt(segments: Sequence[Segment]) -> str:
    blocks: list[str] = ["WEBVTT", ""]
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = format_timestamp(seg.get("start"), millisecond_sep=".")
        end = format_timestamp(seg.get("end"), millisecond_sep=".")
        blocks.append(f"{start} --> {end}")
        blocks.append(text)
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def _as_json(segments: Sequence[Segment], include_timestamp: bool,
             title: str) -> str:
    payload = {
        "title": title,
        "include_timestamp": include_timestamp,
        "segments": [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": str(seg.get("text", "")).strip(),
            }
            for seg in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _as_csv(segments: Sequence[Segment], include_timestamp: bool,
            title: str) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if include_timestamp:
        writer.writerow(["start", "end", "text"])
        for seg in segments:
            writer.writerow([
                format_timestamp(seg.get("start"), millisecond_sep=","),
                format_timestamp(seg.get("end"), millisecond_sep=","),
                str(seg.get("text", "")).strip(),
            ])
    else:
        writer.writerow(["text"])
        for seg in segments:
            writer.writerow([str(seg.get("text", "")).strip()])
    return buffer.getvalue()


_WRITERS = {
    STT_FORMAT_TXT: _as_txt,
    STT_FORMAT_MARKDOWN: _as_markdown,
    STT_FORMAT_HTML: _as_html,
    STT_FORMAT_JSON: _as_json,
    STT_FORMAT_CSV: _as_csv,
}


def format_document(segments: Sequence[Segment], output_format: str,
                    include_timestamp: bool = False, title: str = "") -> str:
    """Render ``segments`` in the chosen output format.

    The subtitle formats (SRT/VTT) always carry their timing - it is what makes
    them subtitles - so ``include_timestamp`` only affects the text formats.
    """
    if output_format == STT_FORMAT_SRT:
        return _as_srt(segments)
    if output_format == STT_FORMAT_VTT:
        return _as_vtt(segments)
    writer = _WRITERS.get(output_format, _as_txt)
    return writer(segments, include_timestamp, title)


# ---------------------------------------------------------------------------
# Time-by-time splitting
# ---------------------------------------------------------------------------
def total_duration(segments: Sequence[Segment]) -> float:
    """The span from the first start to the last end (0 when unknown)."""
    starts = [s.get("start") for s in segments if s.get("start") is not None]
    ends = [s.get("end") for s in segments if s.get("end") is not None]
    if not starts or not ends:
        return 0.0
    return max(0.0, float(max(ends)) - float(min(starts)))


def split_by_time(segments: Sequence[Segment], minutes: int) -> list[list[Segment]]:
    """Group segments into chunks of at most ``minutes`` minutes each.

    A segment is never cut in half: a chunk is closed when the *next* segment
    would push it past the limit, and a single segment longer than the limit
    becomes its own chunk.  Segments with no timing are all collected into one
    chunk (there is nothing to measure against).
    """
    limit = max(1.0, float(minutes) * 60.0)
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        if not current:
            current.append(seg)
            continue
        start = current[0].get("start")
        end = seg.get("end")
        if start is None or end is None:
            current.append(seg)
            continue
        if float(end) - float(start) > limit:
            chunks.append(current)
            current = [seg]
        else:
            current.append(seg)
    if current:
        chunks.append(current)
    return chunks


def plan_documents(
    source_paths: Sequence[str],
    split_mode: str,
    minutes: int = 30,
    *,
    segments_by_source: dict[str, Sequence[Segment]] | None = None,
) -> list[dict[str, Any]]:
    """Describe the documents a job will write.

    Returns one entry per document: ``{"source", "stem", "part", "segments"}``.
    ``part`` is 0 for the one-document-per-source mode and 1-based for the
    time-by-time mode (used to name the file "lecture 1 part 2").
    """
    plans: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    for source in source_paths:
        stem = output_stem(source)
        segments = list((segments_by_source or {}).get(source, []))
        parts = (
            split_by_time(segments, minutes)
            if split_mode == "by_time" and segments
            else [segments]
        )
        for index, part_segments in enumerate(parts, start=1):
            name = stem if len(parts) == 1 and split_mode != "by_time" else (
                f"{stem} part {index}" if len(parts) > 1 else stem
            )
            # Disambiguate two source files with the same base name.
            key = name.lower()
            if key in used:
                used[key] += 1
                name = f"{name} ({used[key]})"
            else:
                used[key] = 1
            plans.append({
                "source": source,
                "stem": name,
                "part": 0 if split_mode != "by_time" else index,
                "segments": part_segments,
            })
    return plans
