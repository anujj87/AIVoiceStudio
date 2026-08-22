"""Voice cloning (XTTS v2).

The app's sherpa-onnx engines cannot clone a voice from a short recording, so
cloning is delegated to Coqui XTTS v2 - the standard offline model that turns
a 4-5 second WAV sample into a new voice (17 languages). It runs in a separate
worker subprocess using the ``coqui-tts`` PyTorch runtime, which is an
optional download (``CloneEngine``) exactly like the GPU runtime: it is never
part of the installer, and it never shares a process with sherpa-onnx so the
two ONNX runtimes cannot conflict.

Flow (Settings -> Voice clone):
1. Download the cloning engine (pip installs ``coqui-tts`` + CPU torch into
   the user folder; first use also fetches the ~2 GB XTTS model automatically).
2. Choose a name, a language and a 4-5 second WAV sample -> ``create_cloned_voice``
   copies the sample into the models folder and registers the voice.
3. The cloned voice appears in Available TTS and the Recording window and is
   synthesised through ``XtTsCloneEngine`` (a persistent worker process).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading

import numpy as np

from .. import paths
from ..util import find_python, sanitize_filename

log = logging.getLogger(__name__)

#: XTTS v2 supported languages (code, English name) - used in the clone tab.
XTTS_LANGUAGES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("pl", "Polish"),
    ("tr", "Turkish"),
    ("ru", "Russian"),
    ("nl", "Dutch"),
    ("cs", "Czech"),
    ("ar", "Arabic"),
    ("zh-cn", "Chinese"),
    ("ja", "Japanese"),
    ("hu", "Hungarian"),
    ("ko", "Korean"),
    ("hi", "Hindi"),
]

_SAMPLE_RATE = 24000


class CloneEngineError(Exception):
    """Raised when the XTTS cloning engine is not installed or fails."""


def engine_dir() -> str:
    return os.path.join(paths.user_data_dir(), "runtime", "clone")


def engine_installed() -> bool:
    """True when the ``coqui_tts`` package can be imported from the clone
    runtime folder (the app never imports it into its own process)."""
    pkg = os.path.join(engine_dir(), "coqui_tts")
    return os.path.isdir(pkg) or os.path.isfile(pkg + ".py")


def ensure_engine(progress=None, cancel_event: threading.Event | None = None) -> None:
    """Install the cloning engine into the user folder with pip --target.

    Works from any Python that ships pip (the dev venv and a packaged build
    with bundled pip). ``coqui-tts`` pulls its own PyTorch (CPU) build.
    """
    target = engine_dir()
    os.makedirs(target, exist_ok=True)
    if progress:
        progress("Preparing pip install of the cloning engine", 0, 1)
    import tempfile  # noqa: PLC0415
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = [
        find_python(), "-m", "pip", "install",
        "--target", target, "--upgrade",
        "coqui-tts",
    ]
    if progress:
        progress("Installing coqui-tts and PyTorch (about 2.5 GB)", 0, 0)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            cwd=tempfile.gettempdir(), creationflags=creationflags,
            timeout=7200,
        )
    except subprocess.TimeoutExpired as exc:
        raise CloneEngineError("The pip install timed out after 2 hours.") from exc
    except OSError as exc:
        raise CloneEngineError(
            f"Could not start pip to install the cloning engine: {exc}"
        ) from exc
    if cancel_event and cancel_event.is_set():
        raise CloneEngineError("Installation cancelled.")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise CloneEngineError(
            "Installing the cloning engine failed:\n" + "\n".join(tail)
        )
    if progress:
        progress("Cloning engine installed.", 1, 1)


def remove_engine() -> None:
    shutil.rmtree(engine_dir(), ignore_errors=True)


class CloneClient:
    """Manages the persistent XTTS worker subprocess.

    The worker loads the (large) XTTS model once and answers ``synthesize``
    requests over a JSON-lines pipe, so cloning stays fast after the first
    sentence. The subprocess uses the same interpreter as the app but a
    separate process, keeping its onnxruntime (via torch) isolated from
    sherpa-onnx's.
    """

    def __init__(self, timeout: float = 600.0):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._timeout = timeout
        self._next_id = 0

    # -- lifecycle ----------------------------------------------------------
    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [sys.executable, "-m", "ai_voice_studio.clone.worker"]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        # Bring the worker up and verify the runtime is importable.
        resp = self._request({"cmd": "ping"})
        if not resp.get("ok"):
            raise CloneEngineError(
                resp.get("error", "The cloning worker failed to start.")
            )

    def close(self):
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    self._proc.kill()
                self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- protocol -----------------------------------------------------------
    def _request(self, req: dict) -> dict:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise CloneEngineError("The cloning worker is not running.")
            req = dict(req)
            req["id"] = self._next_id
            self._next_id += 1
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise CloneEngineError(f"The cloning worker failed: {exc}") from exc
            if not line:
                raise CloneEngineError(
                    "The cloning worker stopped. Is the cloning engine downloaded?"
                )
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CloneEngineError(
                    "Invalid response from the cloning worker."
                ) from exc
            if resp.get("id") != req["id"]:
                raise CloneEngineError(
                    "The cloning worker returned an out-of-order response."
                )
            return resp

    # -- public API ---------------------------------------------------------
    def synthesize(
        self,
        text: str,
        sample_wav: str,
        language: str = "en",
        speed: float = 1.0,
    ) -> np.ndarray:
        self._ensure_proc()
        resp = self._request(
            {
                "cmd": "synthesize",
                "text": text,
                "sample_wav": sample_wav,
                "language": language,
                "speed": float(speed),
            }
        )
        if not resp.get("ok"):
            raise CloneEngineError(resp.get("error", "Synthesis failed."))
        raw = base64.b64decode(resp["wav"])
        samples = np.frombuffer(raw, dtype=np.int16).copy()
        return samples


class XtTsCloneEngine:
    """Drops into the app's engine interface (``get_engine``) for cloned voices.

    Implements the same ``synthesize`` contract as ``TtsEngine`` so the
    recording worker and preview buttons need no changes: returns int16
    samples at ``sample_rate`` (24 kHz, XTTS v2's output rate).
    """

    sample_rate = _SAMPLE_RATE

    def __init__(self, voice_entry: dict, client: CloneClient | None = None):
        self.voice_entry = voice_entry
        self._client = client or CloneClient()
        self.sample_wav = voice_entry.get("sample") or voice_entry.get("dir")
        if not self.sample_wav or not os.path.isfile(self.sample_wav):
            raise CloneEngineError(
                "The sample recording for "
                f"'{voice_entry.get('voice', '')}' is missing. Re-create the "
                "cloned voice."
            )
        self.language = voice_entry.get("xtts_lang") or "en"

    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        samples = self._client.synthesize(
            text=text, sample_wav=self.sample_wav,
            language=self.language, speed=float(speed),
        )
        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415

            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415

            samples = _apply_volume(samples, volume)
        return samples

    def close(self):
        self._client.close()


def create_cloned_voice(
    name: str,
    sample_path: str,
    language: str,
    store,
) -> dict:
    """Copy ``sample_path`` into the models folder and register a cloned voice.

    ``store`` is a ``ModelStore``; returns the new custom-voice entry.
    """
    name = sanitize_filename(name.strip(), 40) or "cloned_voice"
    dest = store.custom_dir(name)
    sample = os.path.join(dest, "sample.wav")
    shutil.copy2(sample_path, sample)
    store.add_custom_voice(
        name, "xtts", dest, "xtts",
        extra={"sample": sample, "language": language},
    )
    return {"name": name, "tts": "xtts", "dir": dest, "kind": "xtts",
            "sample": sample, "language": language}
