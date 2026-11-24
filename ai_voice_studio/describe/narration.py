"""Turn a description run's text into a spoken narration track.

The rest of this application already knows how to speak written text offline: a
voice is a downloaded TTS model, :mod:`ai_voice_studio.tts.engine` loads it and
:mod:`ai_voice_studio.jobs.synthesizer` is the loop the Recording window runs.
This module is the small bridge between the two halves of the feature.  It takes
the *description segments* a run produced (``start``, ``end`` and ``text``) and
lays them on one timeline as a single audio file that sits next to the
description script and the subtitle track in the project folder - so an audio
description project exports both the text a human can review and the audio a
player can play.

Two rules decide where a description is heard:

* a description that carries a timing is spoken at that time - which, with gap
  fitting on, is exactly the pause the describer chose for it; and
* no description ever starts before the previous one has finished.  Two
  descriptions can therefore never be mixed into an unintelligible overlap, and
  a word is never swallowed.  When the timings do not fit (a voice reading more
  slowly than the estimate behind the gap), the track runs slightly long rather
  than talking over the next description.

Nothing here needs a GUI, a video or a model download in order to be tested: the
timeline assembly is pure numpy and the engine is created once, lazily, and can
be injected.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..audio import output as audio_output
from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation
from . import output as output_module

log = logging.getLogger(__name__)

Segment = dict[str, Any]

#: Added after a source's own name, before the extension: ``My Film - narration``.
NARRATION_SUFFIX = " - narration"

#: Silence kept after the last description, so it is never cut off mid-word.
TAIL_SECONDS = 0.5

#: Formats ``audio.output`` can write without FFmpeg.
DIRECT_FORMATS = ("wav",)

#: Fallback sample rate for an injected engine that does not report one.
DEFAULT_SAMPLE_RATE = 22050


class NarrationError(RuntimeError):
    """The narration track could not be produced."""


def narration_stem(source_path: str) -> str:
    """The companion document name for a source video.

    It matches the script and subtitle names (``clip.txt``, ``clip.srt``), so
    everything a run wrote for one video sorts together.
    """
    return output_module.output_stem(source_path) + NARRATION_SUFFIX


def spoken_segments(segments: Sequence[Segment]) -> list[Segment]:
    """The descriptions that have text to speak, in the order they are given."""
    return [
        segment for segment in segments
        if str(segment.get("text", "")).strip()
    ]


def _start_seconds(segment: Segment) -> float | None:
    """A description's start time in seconds, or ``None`` when it has none."""
    try:
        value = segment.get("start")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assemble(
    clips: Sequence[np.ndarray],
    sample_rate: int,
    starts: Sequence[float | None] = (),
    tail_seconds: float = TAIL_SECONDS,
) -> tuple[np.ndarray, list[float]]:
    """Lay spoken clips onto one silent timeline.

    ``starts[i]`` is the moment ``clips[i]`` wants to be heard (``None`` means
    "straight after the previous one").  A clip is never placed before the
    previous clip has finished, so clips never overlap and nothing is mixed.

    Returns the int16 mono samples and the time each clip really landed at, so a
    caller can report the difference between what was asked for and what fits.
    """
    rate = int(sample_rate) or DEFAULT_SAMPLE_RATE
    pieces: list[tuple[int, np.ndarray]] = []
    placed: list[float] = []
    cursor = 0.0
    for index, clip in enumerate(clips):
        data = np.asarray(clip).reshape(-1)
        wanted = starts[index] if index < len(starts) else None
        at = cursor if wanted is None else max(cursor, float(wanted))
        offset = round(at * rate)
        placed.append(at)
        pieces.append((offset, data))
        # The end of this clip is where the next one may start at the earliest.
        cursor = (offset + data.shape[0]) / rate

    length = max((offset + data.shape[0] for offset, data in pieces), default=0)
    tail = round(max(0.0, float(tail_seconds)) * rate)
    out = np.zeros(length + tail, dtype=np.int16)
    for offset, data in pieces:
        if data.shape[0]:
            out[offset:offset + data.shape[0]] = data.astype(np.int16, copy=False)
    return out, placed


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class Narrator:
    """Speak a description run with one installed voice.

    The engine is built on the first description and then reused for every one
    after it: loading a TTS model costs seconds, and a description run has
    hundreds of descriptions.  ``engine_factory`` is there so the timeline can
    be exercised without a downloaded voice.
    """

    def __init__(
        self,
        voice_entry: dict[str, Any],
        params: dict[str, Any] | None = None,
        *,
        output_format: str = "wav",
        engine_factory: Callable[[], Any] | None = None,
        ffmpeg_exe: str | None = None,
    ):
        if not voice_entry:
            raise NarrationError(
                "No voice was chosen for the narration. Download one in "
                "Settings > Download and remove ONNX TTS, or turn the narration track off."
            )
        self.voice_entry = dict(voice_entry)
        self.params = dict(params or {})
        self.output_format = str(output_format or "wav").lower()
        self._engine_factory = engine_factory
        self._ffmpeg_exe = ffmpeg_exe
        self._engine = None

    # -- the voice ---------------------------------------------------------
    @property
    def voice_label(self) -> str:
        """``Piper - Amy``: what the UI calls the voice that is speaking."""
        entry = self.voice_entry
        engine = entry.get("tts_name") or entry.get("tts") or "voice"
        name = entry.get("voice_name") or entry.get("voice") or ""
        return f"{engine} - {name}" if name else str(engine)

    @property
    def engine(self):
        """The TTS engine, loaded on first use and kept for the whole run."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def _build_engine(self):
        if self._engine_factory is not None:
            try:
                return self._engine_factory()
            except Exception as exc:
                raise NarrationError(
                    f"Could not load the narration voice: {exc}"
                ) from exc
        from .. import compute

        try:
            return get_engine(
                self.voice_entry,
                provider=compute.provider_for(
                    str(self.params.get("compute", "cpu"))
                ),
                num_threads=compute.cpu_threads(),
            )
        except EngineUnavailableError as exc:
            raise NarrationError(
                f"Could not load the narration voice: {exc}"
            ) from exc
        except Exception as exc:
            raise NarrationError(
                f"Could not start the narration voice: {exc}"
            ) from exc

    def close(self) -> None:
        """Drop the loaded engine (the cache in ``tts.engine`` keeps the model)."""
        self._engine = None

    def speak(self, text: str) -> np.ndarray:
        """One description, spoken with this voice's punctuation and tuning."""
        prepared = process_punctuation(
            str(text),
            str(self.params.get("punctuation", "default")),
            # An engine that reads expression markup keeps it (``tts.markup``).
            self.voice_entry.get("engine"),
        )
        if not prepared.strip():
            prepared = str(text)
        return self.engine.synthesize(
            prepared,
            sid=int(self.voice_entry.get("sid", 0) or 0),
            speed=_as_float(self.params.get("rate"), 1.0),
            pitch=_as_float(self.params.get("pitch"), 1.0),
            volume=_as_float(self.params.get("volume"), 1.0),
        )

    # -- the track ---------------------------------------------------------
    def _resolve_format(self) -> tuple[str, str | None, str]:
        """The format to write, the FFmpeg to use, and a note when it changed.

        MP3/FLAC need FFmpeg.  A missing FFmpeg must not cost the user the
        narration they asked for, so the file is written as WAV instead and the
        change is reported rather than raised.
        """
        fmt = self.output_format or "wav"
        if fmt in DIRECT_FORMATS:
            return "wav", None, ""
        exe = self._ffmpeg_exe
        if exe is None:
            try:
                from ..audio.ffmpeg import find_ffmpeg

                exe = find_ffmpeg()
            except Exception:
                exe = None
        if exe:
            return fmt, exe, ""
        return "wav", None, (
            f"FFmpeg is not installed, so the narration was written as WAV "
            f"instead of {fmt.upper()}."
        )

    def narrate(
        self,
        segments: Sequence[Segment],
        output_dir: str,
        *,
        stem: str | None = None,
        source: str = "",
        on_progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """Speak ``segments`` into one narration file inside ``output_dir``.

        Raises :class:`NarrationError` when there is nothing to speak or the
        voice cannot be loaded; the caller decides whether that ends the run
        (the text documents are already written and worth keeping).
        """
        clips, starts = self.speak_all(segments, on_progress=on_progress)
        return self.write_track(
            clips, starts, output_dir, stem=stem, source=source,
            descriptions=len(segments),
        )

    # -- the two halves, kept apart for extended audio description ----------
    def speak_all(
        self,
        segments: Sequence[Segment],
        *,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> tuple[list[np.ndarray], list[float | None]]:
        """Speak every description; return the clips and the times they were timed at.

        Extended audio description needs the *measured* length of each clip
        before it can say how long the film is stopped, so the speaking and the
        laying-out are separate steps: this one speaks, and the caller decides
        where each clip ends up.
        """
        pieces = spoken_segments(segments)
        if not pieces:
            raise NarrationError(
                "There is nothing to narrate yet: the description has no text."
            )
        # Load the voice before any file is created, so a run that cannot
        # possibly narrate says so instead of leaving an empty file behind.
        # The value itself is unused; reading the property is the point.
        _ = self.engine
        clips: list[np.ndarray] = []
        starts: list[float | None] = []
        total = len(pieces)
        for index, segment in enumerate(pieces, start=1):
            if on_progress is not None:
                on_progress(
                    f"Narrating description {index} of {total}...",
                    index / total,
                )
            clips.append(self.speak(str(segment.get("text", ""))))
            starts.append(_start_seconds(segment))
        return clips, starts

    @property
    def sample_rate(self) -> int:
        """The rate the loaded voice speaks at (used to measure a pause)."""
        return int(getattr(self.engine, "sample_rate", 0) or DEFAULT_SAMPLE_RATE)

    def write_track(
        self,
        clips: Sequence[np.ndarray],
        starts: Sequence[float | None],
        output_dir: str,
        *,
        stem: str | None = None,
        source: str = "",
        descriptions: int | None = None,
        tail_seconds: float = TAIL_SECONDS,
    ) -> dict[str, Any]:
        """Lay already-spoken clips on a timeline and write the audio file.

        The dictionary that comes back describes the file that was written, and
        includes ``turns``: when each description is actually heard in it, on the
        film's own clock.  A described film dips its own soundtrack under exactly
        those stretches and nowhere else (see
        ``extended.ducking_expression``), so the moment the voice speaks is
        recorded where it is measured rather than guessed at again in the mixer.
        """
        if not clips:
            raise NarrationError(
                "There is nothing to narrate yet: the description has no text."
            )
        rate = self.sample_rate
        samples, placed = assemble(clips, rate, starts,
                                   tail_seconds=tail_seconds)
        fmt, ffmpeg_exe, note = self._resolve_format()
        name = (stem or narration_stem(source)) + "." + fmt
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name)
        try:
            audio_output.write_audio(
                samples, rate, path, fmt, ffmpeg_exe=ffmpeg_exe,
            )
        except OSError as exc:
            raise NarrationError(f"Could not write {name}: {exc}") from exc
        except RuntimeError as exc:
            raise NarrationError(f"Could not write {name}: {exc}") from exc
        total = len(clips)
        log.info(
            "Narrated %d description(s) into %s (%.1f s) with %s",
            total, path, len(samples) / rate if rate else 0.0, self.voice_label,
        )

        return {
            "file": name,
            "format": fmt,
            "seconds": round(len(samples) / rate, 3) if rate else 0.0,
            "spoken": total,
            "descriptions": total if descriptions is None else int(descriptions),
            "turns": [
                [round(at, 3), round(at + seconds, 3)]
                for at, seconds in zip(placed, self.clip_seconds(clips), strict=False)
            ],
            "voice": str(self.voice_entry.get("voice", "") or ""),
            "voice_name": str(self.voice_entry.get("voice_name", "") or ""),
            "engine": str(self.voice_entry.get("engine", "") or ""),
            "note": note,
        }

    def clip_seconds(self, clips: Sequence[np.ndarray]) -> list[float]:
        """How long each spoken clip lasts - the honest length of a pause."""
        rate = self.sample_rate or DEFAULT_SAMPLE_RATE
        return [len(np.asarray(clip).reshape(-1)) / rate for clip in clips]


def resolve_voice(
    voices: Sequence[dict[str, Any]],
    remembered: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Pick the voice the user used last out of the installed ones.

    ``remembered`` is the flat ``{tts, language, variant, voice}`` block kept in
    Settings.  A remembered voice that has since been removed must not leave the
    box empty, so the search loosens one field at a time and finally falls back
    to the first installed voice.
    """
    if not voices:
        return None
    remembered = remembered or {}
    wanted = {
        key: remembered.get(key)
        for key in ("tts", "language", "variant", "voice")
    }
    if not any(value for value in wanted.values()):
        return None
    for keys in (
        ("tts", "language", "variant", "voice"),
        ("tts", "variant", "voice"),
        ("tts", "voice"),
        ("tts",),
    ):
        for entry in voices:
            if all(entry.get(key) == wanted[key] for key in keys):
                return dict(entry)
    return None
