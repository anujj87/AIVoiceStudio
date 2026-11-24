"""The background audio description job.

The GUI starts one :class:`DescribeJob` per *Start audio description* press.  It
runs on a worker thread, describes every source video and writes each source's
documents into the project folder **as it goes**: the worker hands back every
finished chunk, so the script on disk is refreshed after each one and a Close, a
crash or a long night never loses the chunks that were already described.

When the choice was to speak the descriptions too, the same job then narrates
that source with an installed offline voice (``describe.narration``) and reports
the narration file with the documents.  A voice that will not load is a warning,
not a failed run: the descriptions are already written and are worth keeping.

Progress and completion are delivered through plain callbacks; the dialog
marshals them onto the GUI thread.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .. import project as project_module
from ..audio.ffmpeg import find_ffmpeg
from ..constants import (
    AD_CHUNK_SECONDS,
    AD_DEFAULT_FPS,
    AD_DEFAULT_TRACK_PRIORITY,
    AD_FORMAT_SCRIPT,
    AD_FORMAT_SCRIPT_HTML,
    AD_MAX_DESCRIPTION_SECONDS,
    AD_MIN_GAP_SECONDS,
    AD_PLACEMENT_EXTENDED,
    AD_WINDOW_SECONDS,
)
from . import engine, extended, narration
from . import output as output_module
from .engine import DescribeEngineError

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]
DoneCallback = Callable[[list[dict[str, Any]]], None]
ErrorCallback = Callable[[BaseException], None]
#: ``(segment, described so far, chunks in the source)`` - one finished chunk.
ChunkCallback = Callable[[dict[str, Any], int, int], None]

#: Formats that already are a reviewable script.
_SCRIPT_FORMATS = (AD_FORMAT_SCRIPT, AD_FORMAT_SCRIPT_HTML)

#: Share of a source's progress bar the spoken narration takes, when it is on.
_NARRATION_BAND = 0.2

#: Extended audio description has three phases per source - describe it, speak
#: every description, then write the documents and the described film - so the
#: bar is split three ways instead.
_EXTENDED_DESCRIBE_BAND = 0.5
_EXTENDED_SPEAK_BAND = 0.4


class DescribeJob(threading.Thread):
    """Describe a list of source videos into description documents."""

    def __init__(
        self,
        project_dir: str,
        source_paths: Sequence[str],
        options: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
        on_chunk: ChunkCallback | None = None,
        on_done: DoneCallback | None = None,
        on_error: ErrorCallback | None = None,
    ):
        super().__init__(daemon=True)
        self.project_dir = project_dir
        self.source_paths = list(source_paths)
        self.options = dict(options)
        self._on_progress = on_progress
        self._on_chunk = on_chunk
        self._on_done = on_done
        self._on_error = on_error
        self._cancel = threading.Event()
        self.outputs: list[dict[str, Any]] = []
        #: Things that did not stop the run but must be said out loud, such as a
        #: narration track that could not be spoken.
        self.warnings: list[str] = []
        self._narrator_obj: narration.Narrator | None = None

    # -- control -----------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- helpers -----------------------------------------------------------
    def _report(self, message: str, fraction: float) -> None:
        if self._on_progress is not None:
            try:
                self._on_progress(message, max(0.0, min(1.0, fraction)))
            except Exception:
                log.debug("progress callback failed", exc_info=True)

    def _chunk_ready(self, segment: dict[str, Any], done: int,
                     total: int) -> None:
        """Report one finished chunk, so a dialog can show it as it happens."""
        if self._on_chunk is None:
            return
        try:
            self._on_chunk(dict(segment), done, total)
        except Exception:
            log.debug("chunk callback failed", exc_info=True)

    def _window_seconds(self) -> float:
        """How much video one description covers (see AD_WINDOW_SECONDS)."""
        try:
            return float(self.options.get("window_seconds", AD_WINDOW_SECONDS))
        except (TypeError, ValueError):
            return AD_WINDOW_SECONDS

    def _plan(self, source: str) -> dict[str, Any]:
        return engine.plan_for_video(
            source,
            float(self.options.get("chunk_seconds", AD_CHUNK_SECONDS)),
            self._window_seconds(),
        )

    def _speech_segments(self, source: str) -> list[dict[str, Any]]:
        """Spoken phrases of ``source`` for gap fitting, or ``[]``.

        Gap fitting needs to know where the silence is, and that is the
        speech-to-text model's job - the application already has one.  When it
        is not installed (or fails) the run simply places every description at
        its chunk boundary instead, which is a worse but valid audio description
        track; saying so is more useful than failing the whole run.
        """
        model_id = str(self.options.get("speech_model") or "")
        if not model_id:
            return []
        try:
            from ..stt import engine as stt_engine

            return list(stt_engine.transcribe_audio(
                source,
                model_id=model_id,
                variant=self.options.get("speech_variant") or None,
                language=self.options.get("language", "auto"),
                compute=self.options.get("compute", "cpu"),
            ))
        except Exception as exc:
            log.info("Speech timings unavailable for gap fitting: %s", exc)
            self._report(
                "Could not get speech timings, so descriptions go at their "
                "chunk boundaries.", 0.0,
            )
            return []

    def _write(self, stem: str, segments: Sequence[dict[str, Any]],
               fmt: str, source: str) -> str:
        title = stem
        return output_module.write_document(
            self.project_dir, stem, segments, fmt, title=title, source=source,
        )

    # -- narration ---------------------------------------------------------
    def _narrator(self) -> narration.Narrator | None:
        """The narrator this run speaks with, or ``None`` when it is off.

        Extended audio description *is* a spoken description - the film is
        stopped so the voice can be heard - so the voice is required there too.
        Either way it is checked before any describing starts: a run that cannot
        possibly narrate has to say so up front, not after an hour of work.
        """
        stopping_the_film = extended.is_extended(self.options.get("placement"))
        if not self.options.get("narration") and not stopping_the_film:
            return None
        if self._narrator_obj is None:
            voice = self.options.get("voice") or {}
            if not voice:
                if stopping_the_film:
                    raise DescribeEngineError(
                        "Extended audio description speaks every description "
                        "while the film is stopped, so it needs an offline voice "
                        "and none is installed. Download one in Settings > "
                        "Download and remove ONNX TTS, or choose standard audio "
                        "description and speak into the pauses instead."
                    )
                raise DescribeEngineError(
                    "A narration track was requested but no voice is installed. "
                    "Download a voice in Settings > Download and remove ONNX TTS, or turn "
                    "the narration track off."
                )
            self._narrator_obj = narration.Narrator(
                voice,
                self.options.get("tts") or {},
                output_format=str(self.options.get("narration_format", "wav")),
            )
        return self._narrator_obj

    def _narrate(
        self,
        narrator: narration.Narrator,
        source: str,
        segments: Sequence[dict[str, Any]],
        base: float,
        span: float,
    ) -> dict[str, Any] | None:
        """Speak one source's descriptions; report a failure instead of raising.

        The written descriptions are the product of the run and are already on
        disk by the time this happens, so a voice that will not load must not
        throw them away.  It becomes a warning the dialog shows.
        """
        name = os.path.basename(source)
        start = base + span * (1.0 - _NARRATION_BAND)
        band = span * _NARRATION_BAND
        try:
            info = narrator.narrate(
                segments,
                self.project_dir,
                stem=narration.narration_stem(source),
                source=source,
                on_progress=lambda message, fraction: self._report(
                    message, start + (band * fraction)
                ),
            )
        except Exception as exc:
            log.exception("Narrating %s failed", source)
            self.warnings.append(f"Could not narrate {name}: {exc}")
            self._report(f"Could not narrate {name}: {exc}", base + span)
            return None
        if info.get("note"):
            self.warnings.append(f"{name}: {info['note']}")
        self._report(
            f"Spoke {info['spoken']} description(s) of {name} into "
            f"{info['file']} ({info['seconds']:.0f} s).",
            base + span,
        )
        return info

    def _save_partial(self, source: str, stem: str,
                      segments: list[dict[str, Any]]) -> None:
        """Refresh the reviewable script on disk after every finished chunk."""
        try:
            self._write(stem, segments, AD_FORMAT_SCRIPT, source)
        except OSError as exc:
            log.warning("Could not refresh the partial script: %s", exc)

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            log.exception("Audio description failed")
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    log.debug("error callback failed", exc_info=True)

    def _run(self) -> None:
        sources = self.source_paths
        if not sources:
            raise DescribeEngineError("No video files were selected.")
        os.makedirs(self.project_dir, exist_ok=True)
        # Nothing to narrate about is worth failing the whole run over, but a
        # narration track that cannot be spoken at all is worth refusing now.
        narrator = self._narrator()
        stopping_the_film = extended.is_extended(self.options.get("placement"))

        fmt = str(self.options.get("output_format", AD_FORMAT_SCRIPT))
        fps = float(self.options.get("fps", AD_DEFAULT_FPS))
        # Fitting descriptions into the pauses is exactly what stopping the film
        # replaces: there is nothing to fit into when the film waits for them.
        gap_fitting = bool(self.options.get("gap_fitting", False)) and \
            not stopping_the_film
        min_gap = float(self.options.get("min_gap", AD_MIN_GAP_SECONDS))
        max_seconds = float(self.options.get(
            "max_description", AD_MAX_DESCRIPTION_SECONDS))

        for index, source in enumerate(sources, start=1):
            if self._cancel.is_set():
                return
            base = (index - 1) / len(sources)
            span = 1.0 / len(sources)
            name = os.path.basename(source)
            self._report(
                f"Preparing {name} ({index} of {len(sources)})...", base)
            plan = self._plan(source)
            duration = float(plan["duration"])
            stem = output_module.output_stem(source)
            speech = self._speech_segments(source) if gap_fitting else []
            collected: list[dict[str, Any]] = []

            # One description per *window*, not per chunk: the number the
            # progress report counts is how many descriptions are coming.
            chunk_total = int(plan.get("windows")
                              or len(list(plan.get("chunks") or [])))

            def _segment_ready(segment: dict[str, Any],
                               _stem=stem, _source=source,
                               _list=collected,
                               _total=chunk_total) -> None:
                _list.append(segment)
                self._save_partial(_source, _stem, list(_list))
                self._chunk_ready(segment, len(_list), _total)

            # Describing a video is the slow half by far; when a narration track
            # is written afterwards it gets the tail of this source's bar, and an
            # extended run shares the bar out three ways instead.
            if stopping_the_film:
                describe_span = span * _EXTENDED_DESCRIBE_BAND
            else:
                describe_span = span * (1.0 - _NARRATION_BAND) if narrator else span
            segments = engine.describe_video(
                source,
                model_id=str(self.options["model"]),
                variant=str(self.options.get("variant") or ""),
                fps=fps,
                chunk_seconds=float(self.options.get(
                    "chunk_seconds", AD_CHUNK_SECONDS)),
                window_seconds=self._window_seconds(),
                device=str(self.options.get("compute", "cpu")),
                language=str(self.options.get("language", "en")),
                gap_fitting=gap_fitting,
                speech_segments=speech,
                max_description_seconds=max_seconds,
                min_gap=min_gap,
                on_progress=lambda message, fraction=None, _b=base,
                _s=describe_span: (
                    self._report(message, _b + (_s * (fraction or 0.0)))
                ),
                on_segment=_segment_ready,
            )
            if self._cancel.is_set():
                return

            if stopping_the_film:
                record = self._extended_source(
                    narrator, source, stem, segments, duration, fmt, base, span)
            else:
                record = self._standard_source(
                    narrator, source, stem, segments, duration, plan, fmt,
                    base, span)
            written = [os.path.join(self.project_dir, name)
                       for name in record["files"]]
            self.outputs.append(record)
            self._report(
                f"Wrote {', '.join(os.path.basename(p) for p in written)}"
                f" ({len(segments)} description(s)).",
                base + span,
            )

        self._record_outputs()
        self._report("Audio description finished.", 1.0)
        if self._on_done is not None:
            try:
                self._on_done(self.outputs)
            except Exception:
                log.debug("done callback failed", exc_info=True)

    # -- the two ways of describing one source -----------------------------
    def _write_documents(self, stem: str, segments: Sequence[dict[str, Any]],
                         fmt: str, source: str) -> list[str]:
        """Write the chosen document, and always the reviewable script."""
        written: list[str] = []
        try:
            written.append(self._write(stem, segments, fmt, source))
        except OSError as exc:
            raise DescribeEngineError(
                f"Could not write the description for "
                f"{os.path.basename(source)}: {exc}"
            ) from exc
        # The script is the reviewable draft, so it is always produced - a
        # description without a script a human can edit is only half useful.
        if fmt not in _SCRIPT_FORMATS:
            try:
                written.append(
                    self._write(stem, segments, AD_FORMAT_SCRIPT, source))
            except OSError as exc:
                log.warning("Could not write the description script: %s", exc)
        return written

    def _standard_source(self, narrator, source: str, stem: str,
                         segments: Sequence[dict[str, Any]], duration: float,
                         plan: dict[str, Any], fmt: str, base: float,
                         span: float) -> dict[str, Any]:
        """Describe one source into documents, and speak it into the pauses."""
        written = self._write_documents(stem, segments, fmt, source)
        record: dict[str, Any] = {
            "source": source,
            "title": stem,
            "duration": duration,
            "chunks": len(plan["chunks"]),
            "windows": int(plan.get("windows") or 0),
            "described": len(segments),
            "files": [os.path.basename(path) for path in written],
        }
        # ...and the spoken track, when one was asked for: the same
        # descriptions, at the times the describer chose for them.
        if narrator is not None and not self._cancel.is_set():
            info = self._narrate(narrator, source, segments, base, span)
            if info is not None:
                record["files"].append(info["file"])
                record["narration"] = info
                # ...and the film itself, which is the point of the audio track:
                # the original picture with the original sound and the
                # description heard together, written as a new file so the
                # source is never touched.
                self._render_mix(record, source, info)
        if self.options.get("gap_fitting"):
            self._hint_about_extended(source, segments, plan)
        return record

    def _track_priority(self) -> str:
        """The track priority this run was started with (see constants).

        It only ever moves the film's own soundtrack: *25%* and *50% higher*
        bring it down for the whole film (``extended.track_priority_gain``) and
        *Duck, don't lower* brings it down only while a description is heard
        (``extended.track_priority_duck``).  A run that did not choose one, or
        one from an older project file, is ``same`` - both tracks as they are.
        """
        return str(self.options.get("track_priority") or AD_DEFAULT_TRACK_PRIORITY)

    def _render_mix(self, record: dict[str, Any], source: str,
                    info: dict[str, Any]) -> None:
        """Write the standard described film: the film, with the description in
        the pauses of its own soundtrack.

        The picture is copied, not re-encoded, so what the viewer sees is
        bit-for-bit the source video; the description is mixed into the pauses
        the narration track was laid into.  The *track priority* is what decides
        which of the two tracks gives way: the description keeps the level the
        narration settings gave it and the film's own soundtrack is brought down
        by the chosen amount (nothing at all for ``same``, or only while a
        description is heard for *Duck, don't lower*).  A failure is a warning,
        not a lost run: the script and the spoken track are already on disk.
        """
        name = os.path.basename(source)
        path = extended.described_path(source, self.project_dir)
        priority = self._track_priority()
        gain = extended.track_priority_gain(priority)
        # *Duck, don't lower* dips the film only while the voice is heard, so the
        # mixer needs the moments the voice is heard in - recorded by the
        # narration track as it was written.  Nothing else has anything to dip.
        dip = extended.track_priority_duck(priority)
        turns = info.get("turns") if dip < 1.0 else ()
        try:
            out_path = extended.mix(
                find_ffmpeg(), source,
                os.path.join(self.project_dir, str(info.get("file") or "")),
                path,
                film_volume=gain,
                duck_level=dip,
                duck_turns=turns or (),
            )
        except extended.ExtendedRenderError as exc:
            log.warning("Could not write the described film for %s: %s",
                        source, exc)
            self.warnings.append(
                f"Could not write the described video for {name}: {exc}"
            )
            return
        record["film"] = os.path.basename(out_path)
        record["narration"]["mixed"] = True
        record["track_priority"] = priority
        if dip < 1.0:
            # The dip is the one part of the mix that leaves no trace in the
            # settings, so it is recorded next to the file it shaped.
            record["duck_level"] = round(dip, 3)
        record["files"].append(os.path.basename(out_path))
        if dip < 1.0:
            note = (", the film's own sound dipping to "
                    f"{round(dip * 100):d}% while the description is spoken.")
        elif gain < 1.0:
            note = (", the film's own sound brought down by "
                    f"{round((1.0 - gain) * 100):d}%.")
        else:
            note = "."
        self._report(
            f"Wrote the described film {os.path.basename(out_path)}: the "
            "original picture and sound with the description over the pauses"
            + note,
            1.0,
        )

    def _hint_about_extended(self, source: str,
                             segments: Sequence[dict[str, Any]],
                             plan: dict[str, Any]) -> None:
        """Say so when a wall-to-wall film has nothing left to describe into.

        Chunks that produced no description because there was no pause to speak
        in are usually a film whose dialogue never stops - the case extended
        audio description exists for - so the user is told which door to open
        rather than just seeing fewer descriptions than they expected.  A chunk
        the describer skipped because nothing in its picture had changed since
        the last description is counted here too, and named as what it is.
        """
        chunks = int(plan.get("windows")) if plan.get("windows") \
            else len(plan.get("chunks", []) or [])
        missing = chunks - len(segments)
        if chunks < 2 or missing <= 0:
            return
        self.warnings.append(
            f"{os.path.basename(source)}: {missing} of {chunks} "
            "piece(s) produced no description to speak - nothing in the "
            "picture had changed, or no pause was long enough to hold one. "
            "If this video's dialogue never leaves a pause, choose extended "
            "audio description, which stops the film for every description."
        )

    def _extended_source(self, narrator, source: str, stem: str,
                         segments: Sequence[dict[str, Any]], duration: float,
                         fmt: str, base: float, span: float) -> dict[str, Any]:
        """Stop the film for every description, then write what that produced.

        The order is forced by the feature itself: a description's spoken length
        *is* the length of its pause, so the descriptions are spoken first, the
        plan is built from those measured lengths, and only then are the spoken
        track and the documents written - both on the described film's clock, so
        the subtitle track matches the film it belongs to.
        """
        name = os.path.basename(source)
        words = narration.spoken_segments(segments)
        record: dict[str, Any] = {
            "source": source,
            "title": stem,
            "duration": duration,
            "described": len(segments),
            "files": [],
            "placement": AD_PLACEMENT_EXTENDED,
        }
        speak_at = base + span * _EXTENDED_DESCRIBE_BAND
        band = span * _EXTENDED_SPEAK_BAND
        try:
            clips, _wanted = narrator.speak_all(
                segments,
                on_progress=lambda message, fraction: self._report(
                    message, speak_at + (band * fraction)
                ),
            )
        except Exception as exc:
            log.exception("Speaking the descriptions of %s failed", source)
            # Without a voice there is no pause, so there is no described film;
            # the descriptions are still written, on the original clock.
            self.warnings.append(
                f"Could not speak the descriptions of {name}, so it was not "
                f"described in extended mode: {exc}"
            )
            record["files"] = [os.path.basename(path) for path in
                               self._write_documents(stem, segments, fmt, source)]
            self._report(f"Could not speak {name}: {exc}", base + span)
            return record

        plan = extended.plan_extended(
            words, duration, clip_seconds=narrator.clip_seconds(clips),
        )
        record["extended"] = {
            "count": plan["count"],
            "added_seconds": plan["added_seconds"],
            "extended_duration": plan["extended_duration"],
            "original_duration": plan["original_duration"],
            "stops": plan["stops"],
        }
        if not plan["stops"]:
            self.warnings.append(
                f"No descriptions were written for {name}, so there was nothing "
                "to stop the film for."
            )
            record["files"] = [os.path.basename(path) for path in
                               self._write_documents(stem, segments, fmt, source)]
            return record

        # The spoken track lives on the described film's clock: each description
        # is heard at the moment its pause begins.
        info = narrator.write_track(
            clips,
            [stop["extended_start"] for stop in plan["stops"]],
            self.project_dir,
            stem=extended.narration_stem(source),
            descriptions=len(segments),
        )
        record["narration"] = dict(info, placement=AD_PLACEMENT_EXTENDED)
        record["files"].append(info["file"])
        self._report(
            extended.placement_summary(plan, [extended.render_stem(source)]),
            speak_at + band,
        )

        # The documents describe the *described* film, pause for pause.
        for path in self._write_documents(stem, extended.extended_segments(plan),
                                          fmt, source):
            record["files"].append(os.path.basename(path))

        self._render_film(record, source, plan, duration)
        record["files"] = list(dict.fromkeys(record["files"]))
        return record

    def _render_film(self, record: dict[str, Any], source: str,
                     plan: dict[str, Any], duration: float) -> None:
        """Write the described film itself: the film with the pauses baked in.

        A failure here is a warning, not a lost run: the pause points, the spoken
        track and the documents are all still there, and the user is told exactly
        which part could not be written.
        """
        info = record.get("narration") or {}
        name = os.path.basename(source)
        if not info.get("file"):
            return
        try:
            out_path = extended.render(
                find_ffmpeg(),
                source,
                os.path.join(self.project_dir, info["file"]),
                extended.described_path(source, self.project_dir),
                plan,
                duration,
            )
        except extended.ExtendedRenderError as exc:
            log.warning("Could not write the described film for %s: %s",
                        source, exc)
            self.warnings.append(
                f"Could not write the described video for {name}: {exc}"
            )
            return
        record["film"] = os.path.basename(out_path)
        record["files"].append(os.path.basename(out_path))
        self._report(f"Wrote the described film {os.path.basename(out_path)}.", 1.0)

    def _record_outputs(self) -> None:
        """Store the written documents in ``project.json`` for the recent list."""
        try:
            data = project_module.load_project(self.project_dir)
        except (OSError, ValueError) as exc:
            log.warning("Could not record describe outputs: %s", exc)
            return
        data["describe_outputs"] = self.outputs
        # The narration tracks have their own key: the recent-project list and a
        # future player want to know which voice spoke which file.
        data[output_module.NARRATION_KEY] = [
            dict(item["narration"], source=item.get("source", ""))
            for item in self.outputs
            if item.get("narration")
        ]
        done = {os.path.basename(item["source"]) for item in self.outputs}
        for seg in data.get("segments", []):
            if os.path.basename(seg.get("source", seg.get("title", ""))) in done:
                seg["status"] = "done"
        project_module.save_project(self.project_dir, data)
