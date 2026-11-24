"""Video helpers for audio description.

The application already downloads and manages FFmpeg (for MP3/FLAC output and
for decoding video in the transcribe flow), so nothing new is pulled in here:
this module only asks FFmpeg what it already knows about a media file.  The one
question the describe flow must answer before it can plan anything is **how long
is the video**, and that is read from FFmpeg's own ``-i`` banner.

FFmpeg's probe is used rather than a Python media library because it is the same
binary the rest of the application depends on and it needs no extra package in
the GUI process.  When FFmpeg is missing the duration is reported as ``None`` and
the caller falls back to describing the file as a single chunk.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess

from ..audio.ffmpeg import find_ffmpeg

log = logging.getLogger(__name__)

#: Video containers the source pickers accept (audio-only files are handled by
#: the speech-to-text project base, not here).
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".mpg",
    ".mpeg", ".3gp", ".ts", ".m2ts", ".ogv",
}

#: ``Duration: 00:12:34.56`` inside FFmpeg's banner.
_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)"
)


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def scan_videos(path: str, recursive: bool = True) -> list[str]:
    """The video files a source entry stands for (a file maps to itself)."""
    if os.path.isfile(path):
        return [path] if is_video_file(path) else []
    if not os.path.isdir(path):
        return []
    found: list[str] = []
    if recursive:
        for root, dirs, names in os.walk(path):
            dirs.sort()
            for name in sorted(names):
                candidate = os.path.join(root, name)
                if is_video_file(candidate):
                    found.append(candidate)
    else:
        for name in sorted(os.listdir(path)):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate) and is_video_file(candidate):
                found.append(candidate)
    return found


def parse_duration(text: str) -> float | None:
    """The duration in seconds from FFmpeg's probe output (``None`` if absent)."""
    match = _DURATION_RE.search(text or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    try:
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):  # pragma: no cover - regex guarantees digits
        return None


def probe_duration(path: str, ffmpeg_exe: str | None = None,
                   timeout: float = 30.0) -> float | None:
    """Length of a media file in seconds, or ``None`` when it cannot be read.

    A failure is not an error: the caller describes the file as one chunk and the
    estimate is skipped, which is exactly what a still-unfinished download needs.
    """
    if not path or not os.path.isfile(path):
        return None
    exe = ffmpeg_exe or find_ffmpeg()
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-hide_banner", "-i", path],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except Exception:
        log.debug("Could not probe %s", path, exc_info=True)
        return None
    return parse_duration(result.stderr or result.stdout or "")


def probe_durations(paths: list[str],
                    ffmpeg_exe: str | None = None) -> list[float | None]:
    """Probe several files in order (one ``None`` entry per unreadable file)."""
    return [probe_duration(path, ffmpeg_exe) for path in paths]


def total_duration(paths: list[str],
                   ffmpeg_exe: str | None = None) -> float:
    """Sum of the known durations; an unreadable file contributes nothing."""
    return sum(value for value in probe_durations(paths, ffmpeg_exe)
               if value)
