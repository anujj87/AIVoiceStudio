"""FFmpeg detection and on-demand download (SPEC 3.6).

FFmpeg is needed only for MP3/FLAC output. When missing, the app offers to
download it into ``%APPDATA%\\AIVoiceStudio\\ffmpeg`` (official gyan.dev
"essentials" build, Windows x86_64).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import zipfile
from typing import Callable, Optional

from .. import paths
from ..constants import FFMPEG_DOWNLOAD_URL, FFMPEG_EXE_NAME
from ..tts.downloader import DownloadCancelled, download_file

log = logging.getLogger(__name__)


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg.exe: app-folder copy first, then system PATH."""
    app_copy = os.path.join(paths.ffmpeg_dir(), FFMPEG_EXE_NAME)
    if os.path.isfile(app_copy):
        return app_copy
    found = shutil.which("ffmpeg")
    return found


def ensure_ffmpeg() -> Optional[str]:
    """Return a working ffmpeg path or None (used to gate MP3/FLAC output)."""
    candidate = find_ffmpeg()
    if candidate and _works(candidate):
        return candidate
    return None


def _works(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def download_ffmpeg(
    progress: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Download and unpack FFmpeg into the app's user folder; return exe path."""
    ffdir = paths.ffmpeg_dir()
    archive = os.path.join(ffdir, "ffmpeg-release-essentials.zip")

    def cb(name: str, done: int, total: int):
        if progress:
            progress("ffmpeg-release-essentials.zip", done, total)

    download_file(
        FFMPEG_DOWNLOAD_URL,
        ffdir,
        filename="ffmpeg-release-essentials.zip",
        progress=cb,
        cancel_event=cancel_event,
    )
    if cancel_event and cancel_event.is_set():
        raise DownloadCancelled()
    with zipfile.ZipFile(archive) as zf:
        # Extract only ffmpeg.exe (inside bin/), skipping the rest of the build.
        exe_members = [
            m for m in zf.infolist()
            if m.filename.endswith("/bin/ffmpeg.exe") and not m.is_dir()
        ]
        if not exe_members:
            raise RuntimeError("ffmpeg.exe not found inside the downloaded archive")
        member = exe_members[0]
        with zf.open(member) as src, open(os.path.join(ffdir, FFMPEG_EXE_NAME), "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.remove(archive)
    exe = os.path.join(ffdir, FFMPEG_EXE_NAME)
    if not _works(exe):
        raise RuntimeError("Downloaded FFmpeg failed its version check")
    return exe


def convert_wav(src_wav: str, dest_path: str, fmt: str, ffmpeg_exe: str) -> None:
    """Convert a WAV file to MP3 or FLAC via FFmpeg."""
    if fmt == "wav":
        shutil.copyfile(src_wav, dest_path)
        return
    codec_args = {"mp3": ["-codec:a", "libmp3lame", "-q:a", "2"], "flac": ["-codec:a", "flac"]}
    args = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", "-i", src_wav]
    args += codec_args.get(fmt, codec_args["mp3"])
    args += [dest_path]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion to {fmt} failed: {result.stderr[-400:]}")


def needs_mp3_normalization(path: str) -> bool:
    """True when an MP3 file is not DAISY-player friendly.

    DAISY 2.02/3 players reliably decode MPEG-1 Layer III; the TTS engines
    emit MPEG-2 LSF MP3s at the engine's native sample rate (e.g. 24 kHz),
    which several DAISY players truncate or refuse ("audio cuts out" while
    VLC plays the same file fine).  Probing via FFmpeg is cheap and safe.
    """
    exe = find_ffmpeg()
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "-i", path],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        import re  # noqa: PLC0415

        stream = re.search(
            r"Stream #\d+:\d+.*?: Audio: mp3.*?, (\d+) Hz", result.stderr or "")
        if not stream:
            return False
        return int(stream.group(1)) < 44100
    except Exception:  # noqa: BLE001
        return False


def normalize_mp3_for_daisy(src: str, dst: str, ffmpeg_exe: Optional[str] = None) -> None:
    """Transcode an MP3 to DAISY-friendly MPEG-1 Layer III 44.1 kHz mono.

    Used by the DAISY builders when staging recorded audio: a segment saved
    as a low-rate MPEG-2 LSF MP3 is re-encoded so DAISY players decode it
    fully.  ``dst`` must already point inside the book folder (same folder
    as ``src`` is fine: the source is read first, then the output replaces
    it).
    """
    exe = ffmpeg_exe or find_ffmpeg()
    if not exe:
        raise RuntimeError("FFmpeg is required to normalize DAISY audio")
    tmp_out = dst + ".norm.tmp.mp3"
    args = [
        exe, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
        "-codec:a", "libmp3lame", "-ar", "44100", "-ac", "1",
        "-q:a", "2", tmp_out,
    ]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0 or not os.path.isfile(tmp_out):
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        raise RuntimeError(
            f"FFmpeg normalization failed for {src}: {result.stderr[-400:]}")
    os.replace(tmp_out, dst)
