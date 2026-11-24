"""The background transcription job.

The GUI starts one :class:`TranscriptionJob` per "Start transcribe" press.  It
runs on a worker thread, transcribes every source file and writes the chosen
output documents into the project folder immediately, so a crash or a Close
never loses a finished document.  Progress and completion are delivered through
plain callbacks; the dialog marshals them onto the GUI thread.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .. import project as project_module
from . import output as output_module
from .engine import STTEngineError, transcribe_audio

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]
DoneCallback = Callable[[list[dict[str, Any]]], None]
ErrorCallback = Callable[[BaseException], None]


class TranscriptionJob(threading.Thread):
    """Transcribe a list of source files into output documents."""

    def __init__(
        self,
        project_dir: str,
        source_paths: Sequence[str],
        options: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
        on_done: DoneCallback | None = None,
        on_error: ErrorCallback | None = None,
    ):
        super().__init__(daemon=True)
        self.project_dir = project_dir
        self.source_paths = list(source_paths)
        self.options = dict(options)
        self._on_progress = on_progress
        self._on_done = on_done
        self._on_error = on_error
        self._cancel = threading.Event()
        self.outputs: list[dict[str, Any]] = []

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

    def _write_document(self, stem: str, segments, title: str) -> str:
        fmt = self.options.get("output_format", "txt")
        text = output_module.format_document(
            segments,
            fmt,
            include_timestamp=bool(self.options.get("include_timestamp", False)),
            title=title,
        )
        ext = output_module.FORMAT_EXTENSIONS.get(fmt, ".txt")
        path = os.path.join(self.project_dir, stem + ext)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            log.exception("Transcription failed")
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    log.debug("error callback failed", exc_info=True)

    def _run(self) -> None:
        sources = self.source_paths
        if not sources:
            raise STTEngineError("No source files were selected.")
        os.makedirs(self.project_dir, exist_ok=True)
        split_mode = self.options.get("split_mode", "per_source")
        minutes = int(self.options.get("minutes", 30) or 30)

        for index, source in enumerate(sources, start=1):
            if self._cancel.is_set():
                return
            base = (index - 1) / len(sources)
            span = 1.0 / len(sources)
            name = os.path.basename(source)
            self._report(f"Transcribing {name} ({index} of {len(sources)})...", base)

            segments = transcribe_audio(
                source,
                model_id=self.options["model"],
                variant=self.options.get("variant"),
                language=self.options.get("language", "auto"),
                translate_to=self.options.get("translate_to", ""),
                accuracy=self.options.get("accuracy", "balanced"),
                compute=self.options.get("compute", "cpu"),
                on_progress=lambda message, fraction=None, _b=base, _s=span: (
                    self._report(message, _b + (_s * (fraction or 0.0)))
                ),
            )
            if self._cancel.is_set():
                return

            stem = output_module.output_stem(source)
            parts = (
                output_module.split_by_time(segments, minutes)
                if split_mode == "by_time"
                else [segments]
            )
            for part_index, part in enumerate(parts, start=1):
                if self._cancel.is_set():
                    return
                doc_stem = f"{stem} part {part_index}" if len(parts) > 1 else stem
                path = self._write_document(doc_stem, part, doc_stem)
                self.outputs.append({
                    "source": source,
                    "title": doc_stem,
                    "file": os.path.basename(path),
                    "segments": len(part),
                })
                self._report(
                    f"Saved {os.path.basename(path)}",
                    base + span * (part_index / max(1, len(parts))),
                )

        self._record_outputs()
        self._report("Transcription finished.", 1.0)
        if self._on_done is not None:
            try:
                self._on_done(self.outputs)
            except Exception:
                log.debug("done callback failed", exc_info=True)

    def _record_outputs(self) -> None:
        """Store the written documents in ``project.json`` for the recent list."""
        try:
            data = project_module.load_project(self.project_dir)
        except (OSError, ValueError) as exc:
            log.warning("Could not record STT outputs: %s", exc)
            return
        data["stt_outputs"] = self.outputs
        done = {os.path.basename(o["source"]) for o in self.outputs}
        for seg in data.get("segments", []):
            if os.path.basename(seg.get("source", seg.get("title", ""))) in done:
                seg["status"] = "done"
        project_module.save_project(self.project_dir, data)
