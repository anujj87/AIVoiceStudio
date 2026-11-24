"""Audio description output documents.

A description run produces **text**, and text is what the rest of this
application is already good at turning into audio.  Two kinds of document come
out of here and they serve two different people:

* the **description script** - a human-readable, reviewable page (plain text or
  HTML) that a volunteer describer or an editor can read, correct and hand on.
  It is always written, even when no audio is produced, because a draft script
  is genuinely useful on its own; and
* the **subtitle track** (SRT/WebVTT) - what a player needs in order to speak
  the description at the right moment.

A *description segment* is a plain dict with ``start``, ``end`` (seconds; may be
``None``) and ``text`` - the same shape the speech-to-text side uses, so the
subtitle and JSON writers are the ones already tested in
:mod:`ai_voice_studio.stt.output` rather than a second implementation.
"""

from __future__ import annotations

import html as _html
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

from ..constants import (
    AD_FORMAT_JSON,
    AD_FORMAT_SCRIPT,
    AD_FORMAT_SCRIPT_HTML,
    AD_FORMAT_SRT,
    AD_FORMAT_VTT,
)
from ..stt.output import format_document as _stt_format_document
from ..stt.output import format_timestamp as _format_timestamp
from ..util import sanitize_filename

_log = logging.getLogger(__name__)

Segment = dict[str, Any]

#: Where a run records the narration tracks it spoke (see``describe.narration``).
#: Each entry names the source video, the file and the voice that spoke it.
NARRATION_KEY = "describe_narration"

#: Which file extension each output format writes.
FORMAT_EXTENSIONS: dict[str, str] = {
    AD_FORMAT_SCRIPT: ".txt",
    AD_FORMAT_SCRIPT_HTML: ".html",
    AD_FORMAT_SRT: ".srt",
    AD_FORMAT_VTT: ".vtt",
    AD_FORMAT_JSON: ".json",
}

#: Plain-text formats an audio description understands.
OUTPUT_FORMATS = tuple(FORMAT_EXTENSIONS)


def output_stem(source_path: str, fallback: str = "description") -> str:
    """A safe document name for a source video (its name without extension).

    A source with no usable name - an empty path, or one that only carries
    punctuation - falls back to ``fallback`` rather than to
    ``sanitize_filename``'s own "untitled", so the file reads as what it is.
    """
    base = os.path.splitext(os.path.basename(source_path))[0]
    if not base.strip():
        return fallback
    return sanitize_filename(base, 80) or fallback


def _clock(seconds: float | None) -> str:
    """A short ``H:MM:SS`` / ``M:SS`` label for the script."""
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


def script_paragraphs(segments: Sequence[Segment]) -> list[dict[str, str]]:
    """Group descriptions into ``{"time", "text"}`` paragraphs for the script.

    A description that stopped the film (extended audio description) says so:
    the time is the moment in the *described* film, and the reader of the script
    is told how long the picture was held there.
    """
    paragraphs: list[dict[str, str]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start, end = seg.get("start"), seg.get("end")
        if start is None and end is None:
            span = ""
        else:
            span = f"[{_clock(start)} - {_clock(end)}]".replace(" - ]", "]")
        held = _held_seconds(seg)
        if held:
            note = f"picture held {_clock(held) if held >= 60 else f'{held:g} s'}"
            span = f"{span} ({note})" if span else f"({note})"
        paragraphs.append({"time": span, "text": text})
    return paragraphs


def _held_seconds(segment: Segment) -> float:
    """How long this description held the picture (0 for a normal one)."""
    try:
        return max(0.0, float(segment.get("pause") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def format_script(segments: Sequence[Segment], title: str = "",
                  source: str = "", html: bool = False) -> str:
    """Render the reviewable description script (plain text or a web page)."""
    paragraphs = script_paragraphs(segments)
    heading = title or "Audio description script"
    held = any(_held_seconds(seg) for seg in segments)
    if html:
        escape = _html.escape
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{escape(heading)}</title>",
            "<style>body{font-family:sans-serif;line-height:1.6;max-width:48em;"
            "margin:2em auto}.ts{color:#666;font-family:monospace;"
            "margin-right:.5em}h2{font-size:1.1em;margin-bottom:.2em}</style>",
            "</head>",
            "<body>",
            f"<h1>{escape(heading)}</h1>",
        ]
        if source:
            parts.append(f"<p>Source: {escape(os.path.basename(source))}</p>")
        parts.append(
            f"<p>{len(paragraphs)} description(s). Review and edit as needed "
            "- this is an AI draft.</p>"
        )
        if held:
            parts.append(
                "<p>Extended audio description: the film is stopped for each "
                "description and the times below are in the described film.</p>"
            )
        for paragraph in paragraphs:
            parts.append(f"<h2>{escape(paragraph['time'])}</h2>")
            parts.append(f"<p>{escape(paragraph['text'])}</p>")
        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts) + "\n"

    lines: list[str] = [heading]
    if source:
        lines.append(f"Source: {os.path.basename(source)}")
    lines.append(
        f"{len(paragraphs)} description(s). This is an AI draft - review and "
        "edit as needed."
    )
    if held:
        lines.append(
            "Extended audio description: the film is stopped for each "
            "description, and the times below are those of the described film."
        )
    lines.append("")
    for paragraph in paragraphs:
        if paragraph["time"]:
            lines.append(paragraph["time"])
        lines.append(paragraph["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_output(segments: Sequence[Segment], output_format: str,
                  title: str = "", source: str = "") -> str:
    """Render ``segments`` as the chosen document type."""
    if output_format == AD_FORMAT_SCRIPT:
        return format_script(segments, title=title, source=source, html=False)
    if output_format == AD_FORMAT_SCRIPT_HTML:
        return format_script(segments, title=title, source=source, html=True)
    if output_format == AD_FORMAT_SRT:
        return _stt_format_document(segments, AD_FORMAT_SRT)
    if output_format == AD_FORMAT_VTT:
        return _stt_format_document(segments, AD_FORMAT_VTT)
    if output_format == AD_FORMAT_JSON:
        return _as_json(segments, title)
    return format_script(segments, title=title, source=source, html=False)


def _as_json(segments: Sequence[Segment], title: str) -> str:
    """The machine-readable description, in the shape the transcribe side uses.

    The same ``{"start", "end", "text"}`` keys, because a downstream tool should
    not have to care which half of the application produced them, plus the
    ``pause`` an extended description held the picture for - the one datum a
    player needs in order to know that this film is not the original cut.
    """
    payload = {
        "title": title,
        "segments": [
            dict(
                {
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": str(seg.get("text", "")).strip(),
                },
                **({"pause": _held_seconds(seg)} if _held_seconds(seg) else {}),
            )
            for seg in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_document(project_dir: str, stem: str, segments: Sequence[Segment],
                   output_format: str, title: str = "",
                   source: str = "") -> str:
    """Write one description document into the project folder; return its path."""
    os.makedirs(project_dir, exist_ok=True)
    text = format_output(segments, output_format, title=title, source=source)
    ext = FORMAT_EXTENSIONS.get(output_format, ".txt")
    path = os.path.join(project_dir, stem + ext)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def timestamp(seconds: float | None) -> str:
    """``HH:MM:SS,mmm`` (re-exported so callers need one import)."""
    return _format_timestamp(seconds, millisecond_sep=",")


# ---------------------------------------------------------------------------
# Removing what a run wrote (Edit > Restart All Descriptions)
# ---------------------------------------------------------------------------
def remove_outputs(project_dir: str, data: dict[str, Any]) -> int:
    """Delete the documents a description run wrote; return how many went.

    ``Restart all`` on a recording project deletes its audio and starts again;
    an audio description project has to do the same for its *scripts, subtitle
    tracks and narration tracks*, otherwise the menu entry would say "restart
    all" and leave the old descriptions behind to confuse the next run.  Every
    segment is marked pending again and the recorded output list is cleared.
    """
    removed = 0
    names: list[str] = []
    for item in data.get("describe_outputs", []) or []:
        for name in item.get("files", []) or []:
            names.append(os.path.basename(str(name)))
    for name in names:
        if not name:
            continue
        path = os.path.join(project_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError:
            _log.warning("Could not delete %s", path, exc_info=True)
    data["describe_outputs"] = []
    # The narration tracks went with them: a restart that left the old spoken
    # script behind would play the description the user just deleted.
    data[NARRATION_KEY] = []
    for segment in data.get("segments", []) or []:
        segment["status"] = "pending"
        segment["saved"] = None
    try:
        from .. import project as project_module

        project_module.save_project(project_dir, data)
    except (OSError, ValueError) as exc:
        _log.warning("Could not update the project after removing outputs: %s", exc)
    return removed
