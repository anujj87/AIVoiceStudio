"""Project persistence (SPEC 3.7).

Each project folder contains ``project.json`` describing the document, the
chosen audio mode, the split segments, the per-project TTS settings and which
segments have already been synthesized (used for resume after a crash/quit).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

from .constants import PROJECT_FILE_NAME

log = logging.getLogger(__name__)

_lock = threading.RLock()


def project_file(project_dir: str) -> str:
    return os.path.join(project_dir, PROJECT_FILE_NAME)


def create_project(
    project_dir: str,
    name: str,
    source_file: str,
    audio_mode: str,
    segments: List[Dict[str, Any]],
    tts_settings: Optional[Dict[str, Any]] = None,
    project_type: Optional[str] = None,
    daisy_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = {
        "name": name,
        "source_file": source_file,
        "created": datetime.datetime.now().isoformat(),
        "audio_mode": audio_mode,
        "project_type": project_type or "audio_playlist",
        "segments": [
            {
                "index": seg.get("index", i + 1),
                "title": seg.get("title", f"segment {i + 1}"),
                "text": seg.get("text", ""),
                "status": "pending",
                "saved": None,
            }
            for i, seg in enumerate(segments)
        ],
        "tts": tts_settings or {},
        "daisy": daisy_settings or {},
    }
    save_project(project_dir, data)
    return data


def load_project(project_dir: str) -> Dict[str, Any]:
    path = project_file(project_dir)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_project(project_dir: str, data: Dict[str, Any]) -> None:
    os.makedirs(project_dir, exist_ok=True)
    path = project_file(project_dir)
    with _lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


def mark_segment_done(project_dir: str, index: int, saved_path: str) -> None:
    """Atomically mark a segment as done (called from the worker thread)."""
    try:
        data = load_project(project_dir)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not update project: %s", exc)
        return
    for seg in data.get("segments", []):
        if seg.get("index") == index:
            seg["status"] = "done"
            seg["saved"] = os.path.basename(saved_path)
            break
    save_project(project_dir, data)


def first_pending_index(project_dir: str) -> int:
    """Resume logic: first segment not yet done (SPEC 3.6)."""
    try:
        data = load_project(project_dir)
    except (OSError, json.JSONDecodeError):
        return 0
    for seg in data.get("segments", []):
        if seg.get("status") != "done":
            return seg.get("index", 1) - 1
    return len(data.get("segments", []))


# ---------------------------------------------------------------------------
# Project / recording lifecycle (Edit menu actions)
# ---------------------------------------------------------------------------
_AUDIO_EXTS = (".wav", ".mp3", ".flac")


def recorded_files(project_dir: str) -> List[str]:
    """Audio files actually present in the project folder (sorted).

    Uses the files on disk rather than the ``saved`` field in project.json so
    it stays correct even when that metadata is missing or out of date.
    """
    try:
        names = os.listdir(project_dir)
    except OSError:
        return []
    files = [
        n for n in names
        if os.path.splitext(n)[1].lower() in _AUDIO_EXTS
        and os.path.isfile(os.path.join(project_dir, n))
    ]
    return sorted(files)


def remove_project(project_dir: str) -> None:
    """Delete a project folder and everything in it (recordings + metadata)."""
    shutil.rmtree(project_dir, ignore_errors=True)


def reset_recordings(project_dir: str) -> int:
    """Delete all recorded audio and mark every segment pending again.

    The project itself (name, document, audio mode, segments) is kept.
    Returns the number of audio files removed.
    """
    data = None
    try:
        data = load_project(project_dir)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not reset recordings: %s", exc)
        return 0
    removed = 0
    for seg in data.get("segments", []):
        saved = seg.get("saved")
        if saved:
            path = os.path.join(project_dir, os.path.basename(saved))
            removed += _delete_if_audio(path)
        seg["status"] = "pending"
        seg["saved"] = None
    # Also remove stray audio files that are no longer referenced.
    try:
        names = os.listdir(project_dir)
    except OSError:
        names = []
    for name in names:
        if os.path.splitext(name)[1].lower() in _AUDIO_EXTS:
            removed += _delete_if_audio(os.path.join(project_dir, name))
    save_project(project_dir, data)
    return removed


def remove_recording(project_dir: str, filename: str) -> None:
    """Delete one recorded audio file and mark its segment pending again."""
    base = os.path.basename(filename)
    _delete_if_audio(os.path.join(project_dir, base))
    try:
        data = load_project(project_dir)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not update project after removing %s: %s", base, exc)
        return
    for seg in data.get("segments", []):
        if seg.get("saved") == base:
            seg["status"] = "pending"
            seg["saved"] = None
            break
    save_project(project_dir, data)


def _delete_if_audio(path: str) -> int:
    """Delete ``path`` if it is an audio file; returns 1 on success."""
    try:
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in _AUDIO_EXTS:
            os.remove(path)
            return 1
    except OSError as exc:
        log.warning("Could not delete %s: %s", path, exc)
    return 0
