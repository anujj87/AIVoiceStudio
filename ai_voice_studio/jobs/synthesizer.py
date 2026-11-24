"""Background synthesis worker.

Each segment is synthesized and **saved to disk immediately** after synthesis
(SPEC 3.6) so a crash or forced quit never loses finished segments. The worker
polls a cancel event between segments; a pause event suspends it.

On the GPU back-end several segments are synthesized at the same time
(``tts.engine.planned_workers``).  For a sherpa-onnx model that is
``compute.synthesis_workers``: the engine works one sentence at a time and the
Kokoro implementation ignores ``max_num_sentences``, so a single engine leaves
the card waiting for the CPU to phonemise the next sentence.  A Voice Lab
engine that can fill a GPU (Chatterbox, Qwen3-TTS, VibeVoice) brings its own
number - as
many loaded models, or as many texts in one batched call, as the card has room
for.  Whichever it is, the segments are still *saved and reported* in order,
and each one is produced by the same model at the same precision it would get
from a single engine; only the number of segments in flight changes.

The worker is GUI-framework agnostic: it calls plain callbacks, and the GUI
forwards them to the UI thread via wx.PostEvent.

Every processed segment is also logged to ``processed_text.json`` in the
project folder: the exact text that was sent to the TTS engine, keyed to the
audio file that was created for it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from .. import compute
from ..audio import output as audio_output
from ..constants import PROCESSED_TEXT_FILE_NAME
from ..documents.splitter import Segment
from ..tts.engine import (
    EngineUnavailableError,
    get_engine,
    get_engine_pool,
    planned_workers,
    process_punctuation,
)
from ..util import sanitize_filename

log = logging.getLogger(__name__)


class SynthesisWorker(threading.Thread):
    def __init__(
        self,
        segments: list[Segment],
        voice_entry: dict,
        params: dict,
        output_dir: str,
        ffmpeg_exe: str | None = None,
        start_index: int = 0,
        end_index: int | None = None,
        on_segment_done: Callable[[int, str, str], None] | None = None,
        on_all_done: Callable[[], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        super().__init__(name="SynthesisWorker", daemon=True)
        self.segments = segments
        self.voice_entry = voice_entry
        self.params = params
        self.output_dir = output_dir
        self.ffmpeg_exe = ffmpeg_exe
        self.start_index = max(0, start_index)
        # Exclusive stop position.  None means "to the end of the project"
        # (the plain Start recording behaviour); start_index + 1 records one
        # single segment, which is what "only record selected file" needs.
        self.end_index = end_index
        self.on_segment_done = on_segment_done
        self.on_all_done = on_all_done
        self.on_error = on_error
        self.on_status = on_status
        self.cancel_event = cancel_event or threading.Event()
        self.pause_event = pause_event
        self._engine = None
        self._workers = 1

    # -- public -------------------------------------------------------------
    def cancel(self) -> None:
        self.cancel_event.set()

    def pause(self) -> None:
        if self.pause_event is not None:
            self.pause_event.set()

    def resume(self) -> None:
        if self.pause_event is not None:
            self.pause_event.clear()

    @property
    def done_count(self) -> int:
        return self.start_index

    # -- thread body ---------------------------------------------------------
    def run(self) -> None:
        compute_choice = self.params.get("compute", "cpu")
        provider = compute.provider_for(compute_choice)
        total = len(self.segments)
        stop = total
        if self.end_index is not None:
            stop = max(self.start_index, min(self.end_index, total))
        # How many segments may be in flight at once: one for a CPU run, or for
        # a run of a single segment; more for a GPU run, whose card would
        # otherwise wait on the CPU for every sentence.  A Voice Lab engine
        # that can fill a card brings its own number (several loaded models, or
        # several texts in one batched call - see tts.engine.planned_workers).
        self._workers = planned_workers(
            self.voice_entry, compute_choice, total=stop - self.start_index
        )
        log.info(
            "Synthesizing %d/%d segments with %d engine(s) on %s",
            stop - self.start_index, total, self._workers, provider,
        )
        if self.on_status:
            self.on_status(
                "Loading TTS engine..."
                if self._workers <= 1
                else f"Loading {self._workers} TTS engines..."
            )
        try:
            # Both take the 80-95% band of the machine's logical CPUs
            # (compute.cpu_threads) for the CPU-side pre/post work.
            if self._workers > 1:
                self._engine = get_engine_pool(
                    self.voice_entry,
                    provider=provider,
                    size=self._workers,
                    num_threads=compute.cpu_threads(),
                )
                # The pool may have loaded fewer engines than asked for.
                self._workers = max(1, getattr(self._engine, "size", 1))
            else:
                self._engine = get_engine(
                    self.voice_entry,
                    provider=provider,
                    num_threads=compute.cpu_threads(),
                )
        except EngineUnavailableError as exc:
            if self.on_error:
                self.on_error(str(exc))
            return
        except Exception as exc:
            if self.on_error:
                self.on_error(f"Could not start the TTS engine: {exc}")
            return

        if self.on_status:
            self.on_status("TTS engine ready. Starting synthesis...")
        processed: dict[int, dict[str, str]] = {}
        if self._synthesize_all(processed, total, stop) and self.on_all_done:
            self.on_all_done()

    # -- synthesis pipeline --------------------------------------------------
    def _synthesize_all(
        self,
        processed: dict[int, dict[str, str]],
        total: int,
        stop: int,
    ) -> bool:
        """Synthesize the selected segments, saving each one in order.

        Up to ``self._workers`` segments are synthesized at the same time, but
        the results are *consumed* in index order, so the audio files, the
        ``processed_text.json`` log and the ``on_segment_done`` callbacks are
        exactly the ones the old one-segment-at-a-time loop produced.

        Returns True when the whole range was recorded (the caller then reports
        "all done"), False after a cancel or a failure.
        """
        concurrency = max(1, min(self._workers, stop - self.start_index))
        failed = False
        indices = iter(range(self.start_index, stop))
        nxt = next(indices, None)
        pending: list[tuple[int, Future]] = []
        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="aivs-tts"
        ) as pool:
            while (nxt is not None or pending) and not failed:
                # Keep the pipeline full - unless the run was cancelled or is
                # paused, in which case nothing new is started.
                while nxt is not None and len(pending) < concurrency:
                    self._wait_while_paused()
                    if self.cancel_event.is_set():
                        break
                    if self.on_status:
                        segment = self.segments[nxt]
                        self.on_status(
                            f"Segment {nxt + 1} of {total}: {segment.title}"
                        )
                    pending.append((nxt, pool.submit(self._synthesize_segment, nxt)))
                    nxt = next(indices, None)
                if not pending:
                    break
                idx, future = pending.pop(0)
                try:
                    title, out_path, text = future.result()
                except Exception as exc:
                    if self.on_error:
                        self.on_error(f"Segment {idx + 1} failed: {exc}")
                    failed = True
                    break
                processed[idx] = {
                    "title": title,
                    "file": os.path.basename(out_path),
                    "text": text,
                }
                write_processed_text(self.output_dir, processed)
                if self.on_segment_done:
                    self.on_segment_done(idx, title, out_path)
        # A cancel with segments still unstarted is a stop; one that arrives
        # once everything has been recorded changes nothing - the run is
        # complete and the caller may report it as such.
        if self.cancel_event.is_set() and nxt is not None:
            if self.on_status:
                self.on_status("Stopped.")
            return False
        return not failed

    def _wait_while_paused(self) -> None:
        """Block while the run is paused (the old loop did this per segment)."""
        if self.pause_event is None:
            return
        while self.pause_event.is_set():
            if self.cancel_event.is_set():
                return
            threading.Event().wait(0.2)

    def _synthesize_segment(self, idx: int) -> tuple[str, str, str]:
        """Synthesize one segment and write its audio file.

        Runs on a pool thread when more than one segment is in flight; the
        ordered bookkeeping stays in :meth:`_synthesize_all`.  Returns
        ``(title, audio path, text that was spoken)``.
        """
        segment = self.segments[idx]
        text = process_punctuation(
            segment.text,
            self.params.get("punctuation", "default"),
            # The engine decides whether the text may contain expression markup
            # that the mode must not expand or strip (``tts.markup``).
            self.voice_entry.get("engine"),
        )
        if not text.strip():
            text = segment.text
        samples = self._engine.synthesize(
            text,
            sid=self.voice_entry.get("sid", 0),
            speed=self.params.get("rate", 1.0),
            pitch=self.params.get("pitch", 1.0),
            volume=self.params.get("volume", 1.0),
        )
        stem = sanitize_filename(segment.title, 80)
        fmt = self.params.get("output_format", "wav")
        if fmt == "wav":
            out_path = os.path.join(self.output_dir, stem + ".wav")
            audio_output.write_audio(
                samples, self._engine.sample_rate, out_path, "wav"
            )
        else:
            if not self.ffmpeg_exe:
                raise RuntimeError(
                    "FFmpeg is required for MP3/FLAC output and is not installed"
                )
            out_path = os.path.join(self.output_dir, stem + "." + fmt)
            audio_output.write_audio(
                samples,
                self._engine.sample_rate,
                out_path,
                fmt,
                ffmpeg_exe=self.ffmpeg_exe,
            )
        return segment.title, out_path, text


def write_processed_text(output_dir: str, entries: dict[int, dict[str, str]]) -> None:
    """Atomically write the processed-text log next to the audio files.

    ``entries`` maps a 0-based segment index to ``{"title", "file", "text"}``;
    the file it produced (e.g. ``01 page 1.wav``) is stored so each entry is
    keyed to the audio file the user chose to create. Only segments that were
    actually synthesized are written, so a crashed recording leaves a complete
    log of everything finished so far.
    """
    try:
        path = os.path.join(output_dir, PROCESSED_TEXT_FILE_NAME)
        tmp = path + ".tmp"
        payload = {
            "processed_text": {
                str(index): entry for index, entry in sorted(entries.items())
            }
        }
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Could not write processed text log: %s", exc)
