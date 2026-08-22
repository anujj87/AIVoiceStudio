"""Audio file writing.

WAV is written with the standard library (``wave`` module). MP3/FLAC go through
FFmpeg (``ffmpeg.py``); callers must ensure FFmpeg is available first and fall
back to WAV otherwise.
"""

from __future__ import annotations

import logging
import os
import wave
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def write_wav(samples: "np.ndarray", sample_rate: int, path: str) -> None:
    """Write int16 samples as a 16-bit PCM mono WAV file."""
    data = np.asarray(samples, dtype=np.int16)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    channels = data.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data.tobytes())


def write_audio(
    samples: "np.ndarray",
    sample_rate: int,
    path: str,
    fmt: str = "wav",
    ffmpeg_exe: Optional[str] = None,
) -> None:
    """Write samples to ``path`` in the requested format.

    * fmt == "wav": direct write.
    * fmt in ("mp3", "flac"): write a temp WAV, then convert with FFmpeg.
      Raises RuntimeError when FFmpeg is missing so the caller can prompt.
    """
    fmt = (fmt or "wav").lower()
    if fmt == "wav":
        write_wav(samples, sample_rate, path)
        return
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg is required for MP3/FLAC output and is not installed")
    from .ffmpeg import convert_wav  # local import to avoid cycles

    wav_tmp = path + ".tmp.wav"
    try:
        write_wav(samples, sample_rate, wav_tmp)
        convert_wav(wav_tmp, path, fmt, ffmpeg_exe)
    finally:
        if os.path.exists(wav_tmp):
            os.remove(wav_tmp)
