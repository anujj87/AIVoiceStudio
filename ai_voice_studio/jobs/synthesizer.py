"""Background synthesis worker.

Each segment is synthesized and **saved to disk immediately** after synthesis
(SPEC 3.6) so a crash or forced quit never loses finished segments. The worker
polls a cancel event between segments; a pause event suspends it.

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
from typing import Callable, Dict, List, Optional

from .. import compute
from ..audio import output as audio_output
from ..constants import PROCESSED_TEXT_FILE_NAME
from ..documents.splitter import Segment
from ..tts.engine import EngineUnavailableError, get_engine, process_punctuation
from ..util import sanitize_filename

log = logging.getLogger(__name__)


class SynthesisWorker(threading.Thread):
    def __init__(
        self,
        segments: List[Segment],
        voice_entry: Dict,
        params: Dict,
        output_dir: str,
        ffmpeg_exe: Optional[str] = None,
        start_index: int = 0,
        on_segment_done: Optional[Callable[[int, str, str], None]] = None,
        on_all_done: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(name="SynthesisWorker", daemon=True)
        self.segments = segments
        self.voice_entry = voice_entry
        self.params = params
        self.output_dir = output_dir
        self.ffmpeg_exe = ffmpeg_exe
        self.start_index = max(0, start_index)
        self.on_segment_done = on_segment_done
        self.on_all_done = on_all_done
        self.on_error = on_error
        self.on_status = on_status
        self.cancel_event = cancel_event or threading.Event()
        self.pause_event = pause_event
        self._engine = None

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
    def run(self) -> None:  # noqa: C901
        provider = compute.provider_for(self.params.get("compute", "cpu"))
        fmt = self.params.get("output_format", "wav")
        try:
            self._engine = get_engine(
                self.voice_entry, provider=provider, num_threads=2
            )
        except EngineUnavailableError as exc:
            if self.on_error:
                self.on_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            if self.on_error:
                self.on_error(f"Could not start the TTS engine: {exc}")
            return

        total = len(self.segments)
        processed: Dict[int, Dict[str, str]] = {}
        for idx in range(self.start_index, total):
            if self.cancel_event.is_set():
                if self.on_status:
                    self.on_status("Stopped.")
                break
            if self.pause_event is not None:
                while self.pause_event.is_set():
                    if self.cancel_event.is_set():
                        break
                    threading.Event().wait(0.2)

            segment = self.segments[idx]
            if self.on_status:
                self.on_status(f"Segment {idx + 1} of {total}: {segment.title}")
            try:
                text = process_punctuation(segment.text, self.params.get("punctuation", "default"))
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
                if fmt == "wav":
                    out_path = os.path.join(self.output_dir, stem + ".wav")
                    audio_output.write_audio(samples, self._engine.sample_rate, out_path, "wav")
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
                if self.on_segment_done:
                    self.on_segment_done(idx, segment.title, out_path)
                processed[idx] = {
                    "title": segment.title,
                    "file": os.path.basename(out_path),
                    "text": text,
                }
                write_processed_text(self.output_dir, processed)
            except Exception as exc:  # noqa: BLE001
                if self.on_error:
                    self.on_error(f"Segment {idx + 1} failed: {exc}")
                break
        else:
            if self.on_all_done:
                self.on_all_done()


def write_processed_text(output_dir: str, entries: Dict[int, Dict[str, str]]) -> None:
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
