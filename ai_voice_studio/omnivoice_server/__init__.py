"""OmniVoice Server TTS engine (omnivoice-server).

``omnivoice-server`` wraps the OmniVoice model with an OpenAI-compatible
HTTP API.  Other applications on the same network can also use the server.

Features (beyond the triton/hybrid worker):
- OpenAI-compatible ``/v1/audio/speech`` endpoint
- HTTP streaming (sentence-level chunked transfer)
- Voice profile storage (CRUD for cloned voices)
- Bearer token authentication
- Concurrent request handling (configurable thread pool)
- Prometheus metrics at ``/metrics``

Server configuration (Settings -> OmniVoice Server):
- Host / port binding
- API key protection
- CORS origins for browser frontends
- Inference steps (1-64)
- Max concurrent requests
- Auto-start on app launch

The ``omnivoice-server`` package must be installed separately::

    pip install omnivoice-server

Flow (Settings -> Compute -> Install OmniVoice Server):
1. Check that ``omnivoice_server`` is importable from the managed venv.
2. Install via pip (includes PyTorch, omnivoice, etc.).
3. Server starts on-demand or auto-starts with the app.
4. Synthesis via HTTP POST to ``http://host:port/v1/audio/speech``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .. import omnivoice_quality as quality
from ..omnivoice_quality import split_sentences

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # OmniVoice output sample rate

#: This engine's id.  It resolves to the environment the *direct* OmniVoice
#: engine uses (``%APPDATA%/AIVoiceStudio/tts_envs/omnivoice``): the two are
#: the same model with two front-ends and want the same base package and the
#: same CUDA PyTorch, so they deliberately share one environment.  Every other
#: TTS engine has an environment of its own.
ENGINE_ID = "omnivoice_server"

# The omnivoice-server HTTP API validates ``input`` (JSON) and ``text``
# (multipart) with ``max_length=10_000`` — a longer text is rejected before
# inference even starts with a bare "HTTP 422: Request validation failed".
# A whole book chapter easily exceeds that, so long texts are split into
# server-sized chunks at sentence/paragraph boundaries, synthesized one by
# one and concatenated; the audio the caller receives is identical to what
# one giant request would produce (if the server accepted it).
_TEXT_CHUNK_TARGET = 9_500

# ---------------------------------------------------------------------------
# Drone ("no speech") handling
# ---------------------------------------------------------------------------
# OmniVoice sometimes renders a chunk as a loud low-frequency drone instead of
# a voice (upstream issues #37 / #73 / #144; measured on this machine at 28% of
# the chunks of a real Hindi chapter — and 2 of 6 draws of one paragraph).  The
# server detects it and answers with the ``X-No-Speech-Detected`` header, but
# only *reports* it — the drone used to go straight into the recorded file.
#
# The detector, the splitting and the repair policy live in
# ``ai_voice_studio/omnivoice_quality.py`` because the *direct* OmniVoice
# engine needs exactly the same treatment; this module re-exports the policy
# so existing callers (and the dev probe) keep importing it from here.
# ``_synthesize_chunk`` below is the whole of this engine's part: judge what the
# server returned, draw again while it is a drone, and only then re-record the
# chunk in sentence-sized pieces.  Either way the caller receives one
# continuous take per text, so one segment stays one file.
DRONE_ATTEMPTS = quality.DRONE_ATTEMPTS
DRONE_REPAIR_ATTEMPTS = quality.DRONE_REPAIR_ATTEMPTS
DRONE_MAX_PIECES = quality.DRONE_MAX_PIECES
DRONE_REPAIR_MAX_DEPTH = quality.DRONE_REPAIR_MAX_DEPTH
DRONE_REPAIR_MIN_CHARS = quality.DRONE_REPAIR_MIN_CHARS
DRONE_REPAIR_MAX_DRAWS = quality.DRONE_REPAIR_MAX_DRAWS
DRONE_REPAIR_CHARS = quality.DRONE_REPAIR_CHARS
split_text_for_repair = quality.split_text_for_repair

# ---------------------------------------------------------------------------
# Failed requests
# ---------------------------------------------------------------------------
# A request can come back as an HTTP 500 with a generic "Internal Server Error"
# body even though the server is healthy: the generation for that one request
# returned an empty tensor, and ``tensors_to_wav_bytes`` refuses to build a WAV
# from it ("tensors_to_wav_bytes[0]: tensor is empty (size=0)").  The server's
# own log names the text ("Generation returned no audio for text: '629'").
# Measured while recording a real Hindi chapter over the clone endpoint: 3 such
# answers among 152 requests (2%) — rare per request, routine over a chapter,
# and two of the three landed minutes apart during one recording.  One of them
# ended a segment with "Segment 19 failed: Clone synthesis failed: server
# returned HTTP 500" and threw away everything recorded after it.  It is a bad
# draw, not a bad request, so the text is sent again.

#: Sends of one piece of text before its failure is reported to the caller.
#: Small on purpose: the failure it absorbs is transient (a second draw comes
#: back clean), while a dead server must still surface quickly.
REQUEST_ATTEMPTS = 3

#: Failures worth sending the same text again.  An HTTP 5xx is the server
#: itself failing on a request it accepted; a 4xx (422 validation, 413 too
#: large) is the server *rejecting* the request, so repeating it can only fail
#: the same way.  The rest are the connection going away mid-synthesis.
_RETRYABLE_HINTS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "no connection could be made",
    "remote end closed",
    "incompleteread",
    "broken pipe",
    "temporarily unavailable",
)

#: Log levels the server writes into its own structured log for a problem.
_PROBLEM_LEVELS = ("[WARNING", "[ERROR", "[CRITICAL")

_HTTP_STATUS_MARKER = "server returned http "

#: Name of the log file ``start()`` sends the server's stdout to.
_LOG_NAME = "omnivoice_server.log"


def _tail_of(path: str, lines: int) -> str:
    """Last ``lines`` lines of a text file, or ``""`` when unreadable."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except Exception:  # noqa: BLE001
        return ""


def _request_status_code(message: str) -> int | None:
    """The HTTP status named in an error message, or ``None``.

    ``_http_error_detail`` renders a request failure as
    "server returned HTTP 500: ..."; a message that names no status (a
    connection error, a timeout) has nothing to parse.
    """
    low = (message or "").lower()
    start = low.find(_HTTP_STATUS_MARKER)
    if start < 0:
        return None
    digits = ""
    for char in low[start + len(_HTTP_STATUS_MARKER):]:
        if not char.isdigit():
            break
        digits += char
    return int(digits) if digits else None


def _retryable_request_failure(message: str) -> bool:
    """True when re-sending the same text can plausibly succeed."""
    status = _request_status_code(message)
    if status is not None:
        return status >= 500
    low = (message or "").lower()
    return any(hint in low for hint in _RETRYABLE_HINTS)


def _read_server_config() -> dict:
    """Read OmniVoice Server settings from the app's settings.json.

    Returns the ``omnivoice_server`` section as a dict.  Falls back to
    empty dict so callers can use ``.get()`` with defaults.
    """
    try:
        from ..settings import Settings  # noqa: PLC0415
        s = Settings()
        return s.get("omnivoice_server", {})
    except Exception:  # noqa: BLE001
        return {}

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8881
DEFAULT_DEVICE = "cuda"
DEFAULT_NUM_STEPS = 32
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5001,http://127.0.0.1:5001,"
    "http://localhost:5173,http://127.0.0.1:5173"
)


# ---------------------------------------------------------------------------
# Long-text splitting (server rejects > 10,000 characters with HTTP 422)
# ---------------------------------------------------------------------------
def _response_headers(resp) -> dict:
    """Response headers as a lower-cased ``{name: value}`` dict.

    The server reports a drone with ``X-No-Speech-Detected: true`` and the
    audio duration with ``X-Audio-Duration-S``.  Test doubles and some
    transports expose no headers at all, hence the guard.
    """
    headers = getattr(resp, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except Exception:  # noqa: BLE001
        return {}


def _group_fragments(fragments: list, limit: int) -> list:
    """Join ``fragments`` into groups of at most ``limit`` characters.

    A fragment longer than the limit is hard-cut (a "sentence" with no
    punctuation at all would otherwise never fit the request cap).
    """
    groups: list = []
    buf: list = []
    buf_len = 0
    for fragment in fragments:
        if len(fragment) > limit:
            if buf:
                groups.append(" ".join(buf))
                buf, buf_len = [], 0
            for j in range(0, len(fragment), limit):
                piece = fragment[j:j + limit].strip()
                if piece:
                    groups.append(piece)
            continue
        if buf_len + len(fragment) + 1 > limit and buf:
            groups.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(fragment)
        buf_len += len(fragment) + 1
    if buf:
        groups.append(" ".join(buf))
    return groups


def split_text_for_server(text: str, limit: int = _TEXT_CHUNK_TARGET) -> list:
    """Split ``text`` into chunks the omnivoice-server API accepts.

    The server's pydantic schema caps ``input``/``text`` at 10,000 characters
    and answers anything longer with HTTP 422 "Request validation failed"
    before synthesis starts.  Book chapters are routinely longer, so this
    splits at paragraph breaks first, then sentence ends (including Devanagari
    danda and CJK full stops), then hard-cuts as a last resort.  Chunks are
    kept in order and none exceeds ``limit`` characters (unless a single
    "sentence" alone is longer, which the hard cut then splits).
    """
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []

    chunks: list = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            chunks.append(paragraph)
            continue
        # Paragraph itself too long: split at sentence boundaries.
        chunks.extend(_group_fragments(split_sentences(paragraph), limit))
    return chunks if chunks else [text[:limit]]


# ---------------------------------------------------------------------------
# Package detection (managed venv)
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """True when ``omnivoice_server`` is importable from its own venv."""
    try:
        from ..python_runtime import engine_runtime  # noqa: PLC0415
        rt = engine_runtime(ENGINE_ID)
        if not rt.is_created:
            return False
        result = rt.run_in_env(
            "import importlib.metadata; "
            "print(importlib.metadata.version('omnivoice-server'))"
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


def installed_version() -> str | None:
    """Return the installed omnivoice-server version, or None."""
    try:
        from ..python_runtime import engine_runtime  # noqa: PLC0415
        rt = engine_runtime(ENGINE_ID)
        if not rt.is_created:
            return None
        result = rt.run_in_env(
            "import importlib.metadata; "
            "print(importlib.metadata.version('omnivoice-server'))"
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Server subprocess management
# ---------------------------------------------------------------------------
class OmniVoiceServerError(Exception):
    """Raised when the OmniVoice server fails to start or respond."""


class OmniVoiceServerManager:
    """Manages the omnivoice-server HTTP subprocess.

    The server loads the OmniVoice model once and answers synthesis
    requests over HTTP.  Other apps on the same network can also
    use it if the host is set to ``0.0.0.0``.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        device: str = DEFAULT_DEVICE,
        num_steps: int = DEFAULT_NUM_STEPS,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        api_key: str = "",
        cors_origins: str = DEFAULT_CORS_ORIGINS,
        model_id: str = "k2-fsa/OmniVoice",
        drone_attempts: int = DRONE_ATTEMPTS,
        repair_attempts: int = DRONE_REPAIR_ATTEMPTS,
    ):
        self._host = host
        self._port = port
        self._device = device
        self._num_steps = num_steps
        self._max_concurrent = max_concurrent
        self._api_key = api_key
        self._cors_origins = cors_origins
        self._model_id = model_id
        #: Draws allowed per chunk before the sentence-level repair kicks in,
        #: and attempts per sentence-sized piece while repairing.
        self.drone_attempts = max(1, int(drone_attempts))
        self.repair_attempts = max(1, int(repair_attempts))
        #: What the last ``synthesize`` / ``synthesize_clone`` call had to
        #: repair: one dict per affected chunk (see ``_synthesize_chunk``).
        self.last_repairs: list = []
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._ready = threading.Event()
        # Server subprocess log (see start()): the server writes its log to
        # stdout, which must go to a real file — an undrained PIPE fills up
        # and makes every server-side write fail with [Errno 22].
        self._log_fh = None
        self._log_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def process_alive(self) -> bool:
        """True when the subprocess this manager started is still running."""
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_running(self) -> bool:
        """True when we started a subprocess that is still alive,
        OR when a server is reachable via health check (started externally)."""
        if self.process_alive:
            return True
        # Another manager instance may have started the server, or it
        # was started externally.  Verify with a health check.
        return self.health_check()

    def start(self, timeout: float = 120.0) -> None:
        """Start the server subprocess and wait until it's ready.

        If a server is already running on the configured host:port
        (started by another manager instance or externally), we detect
        it via health check and skip starting a new subprocess.
        """
        with self._lock:
            if self.is_running:
                return
            self._ready.clear()

        # Before starting a new process, check if a server is already
        # listening on this host:port (e.g. started from Settings panel
        # or another thread).  This avoids port conflicts.
        if self.health_check():
            self._ready.set()
            log.info(
                "OmniVoice server already running at %s (detected via health check)",
                self.base_url,
            )
            return

        from ..python_runtime import engine_runtime  # noqa: PLC0415
        rt = engine_runtime(ENGINE_ID)

        if not rt.is_created:
            raise OmniVoiceServerError(
                "The managed Python environment is not ready. "
                "Install the OmniVoice Server dependency first."
            )

        cmd = [
            rt.python_exe, "-m", "omnivoice_server",
            "--host", self._host,
            "--port", str(self._port),
            "--device", self._device,
            "--num-step", str(self._num_steps),
            "--max-concurrent", str(self._max_concurrent),
            "--model", self._model_id,
        ]
        if self._api_key:
            cmd.extend(["--api-key", self._api_key])
        if self._cors_origins:
            cmd.extend(["--cors-origins", self._cors_origins])

        env = dict(os.environ)
        # Ensure the managed venv's site-packages are on PYTHONPATH
        # so omnivoice_server can be found.
        env["PYTHONPATH"] = rt.env_dir + os.pathsep + env.get("PYTHONPATH", "")

        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        # The server logs to stdout.  It used to run with stdout=PIPE, but
        # nothing drained that pipe: once the OS pipe buffer filled, every
        # log write inside the server failed with "[Errno 22] Invalid
        # argument" and each synthesis request returned HTTP 500 — the
        # server looked broken while it was perfectly healthy.  Send the
        # output to a log file (truncated on each start) instead.
        from ..paths import logs_dir  # noqa: PLC0415
        self._log_path = os.path.join(logs_dir(), _LOG_NAME)
        try:
            self._log_fh = open(
                self._log_path, "w", encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            self._log_fh = None
            self._log_path = None
            log.warning("Could not open OmniVoice server log file: %s", exc)

        log.info(
            "Starting OmniVoice server: %s (log: %s)",
            " ".join(cmd),
            self._log_path or "<discarded>",
        )
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=self._log_fh if self._log_fh else subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                env=env,
            )
        except OSError as exc:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None
            raise OmniVoiceServerError(f"Failed to start server: {exc}") from exc

        # Wait for the server to become ready (health check).
        self._wait_ready(timeout)

    def _wait_ready(self, timeout: float) -> None:
        """Poll the health endpoint until the server responds or timeout."""
        import urllib.request  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415

        url = f"{self.base_url}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running:
                # Server exited before becoming ready
                output = self._log_tail()
                raise OmniVoiceServerError(
                    f"Server exited prematurely. Output:\n{output}"
                )
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        self._ready.set()
                        log.info("OmniVoice server ready at %s", self.base_url)
                        return
            except (urllib.error.URLError, OSError, ConnectionError):
                pass
            time.sleep(1.0)
        raise OmniVoiceServerError(
            f"Server did not become ready within {timeout}s."
        )

    def stop(self) -> None:
        """Stop the server subprocess."""
        with self._lock:
            if self._proc is None:
                return
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None
            self._ready.clear()
            if self._log_fh:
                try:
                    self._log_fh.close()
                except Exception:  # noqa: BLE001
                    pass
                self._log_fh = None
            log.info("OmniVoice server stopped.")

    def _last_server_problem(self) -> str:
        """The last problem line the server itself logged, or ``""``.

        A request-level 500 is answered with a generic body ("Internal Server
        Error") that says nothing about the cause; the reason is in the
        server's own log ("Generation returned no audio for text: '629'"),
        which is where the actionable half of the story lives.  Quoting that
        one line turns an opaque HTTP failure into a named one.  The long
        engine-side traceback is deliberately not quoted: it is not the reason
        for anything, and the log file has it anyway.
        """
        text = self._log_tail(80)
        if not text:
            # We did not start this server (the app started it in another
            # process, or this manager is a late arrival), so fall back to the
            # one place the server ever writes.
            from ..paths import logs_dir  # noqa: PLC0415
            text = _tail_of(os.path.join(logs_dir(), _LOG_NAME), 80)
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line or not any(level in line for level in _PROBLEM_LEVELS):
                continue
            # "2026-09-21T21:34:47Z [WARNING] [module] reason" -> "reason"
            _, _, reason = line.rpartition("] ")
            return (reason or line).strip()
        return ""

    def _request_error_detail(self, exc: Exception) -> str:
        """``_http_error_detail``, plus the server's own reason for a 5xx."""
        detail = self._http_error_detail(exc)
        status = _request_status_code(detail)
        if status is not None and status >= 500:
            reason = self._last_server_problem()
            if reason and reason not in detail:
                detail = f"{detail} (server log: {reason})"
        return detail

    def _log_tail(self, lines: int = 40) -> str:
        """Last lines of the server log file (for error reporting)."""
        return _tail_of(self._log_path or "", lines)

    def restart(self, timeout: float = 120.0) -> None:
        """Restart the server (stop then start)."""
        self.stop()
        self.start(timeout=timeout)

    def health_check(self, retries: int = 3, delay: float = 1.0,
                     timeout: float = 10.0) -> bool:
        """Return True if the server responds to /health.

        Retries a few times with short delays: while the server is busy
        (cloning reference-audio preprocessing, long synthesis), the health
        endpoint can be slow to answer, and a single 5s probe used to
        mis-report a perfectly healthy server as "not responding".
        """
        import urllib.request  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415

        for attempt in range(max(1, retries)):
            try:
                url = f"{self.base_url}/health"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status == 200
            except Exception:  # noqa: BLE001
                if attempt < retries - 1:
                    time.sleep(delay)
        return False

    # -- Synthesis -----------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "alloy",
        instructions: str = "",
        speed: float = 1.0,
        stream: bool = False,
        num_step: int | None = None,
        guidance_scale: float | None = 3.0,
        denoise: bool | None = True,
        t_shift: float | None = None,
        position_temperature: float | None = None,
        class_temperature: float | None = None,
        duration: float | None = None,
        language: str | None = None,
        layer_penalty_factor: float | None = None,
        preprocess_prompt: bool | None = None,
        postprocess_output: bool | None = None,
        audio_chunk_duration: float | None = None,
        audio_chunk_threshold: float | None = None,
        request_timeout_s: int | None = None,
        seed: int | None = None,
        response_format: str = "wav",
    ) -> np.ndarray:
        """Synthesize text via the server; returns int16 samples at 24 kHz.

        Every generation parameter the omnivoice-server HTTP API documents is
        exposed here; parameters left as ``None`` are omitted from the request
        and the server applies its own defaults.  ``instructions`` (strongest),
        ``voice`` / ``speaker`` presets and ``ref_audio``-based cloning follow
        the server's documented precedence rules.
        """
        import urllib.request  # noqa: PLC0415

        if not text.strip():
            raise ValueError("Nothing to synthesize")

        url = f"{self.base_url}/v1/audio/speech"
        optional = {
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
            "t_shift": t_shift,
            "position_temperature": position_temperature,
            "class_temperature": class_temperature,
            "duration": duration,
            "language": language,
            "layer_penalty_factor": layer_penalty_factor,
            "preprocess_prompt": preprocess_prompt,
            "postprocess_output": postprocess_output,
            "audio_chunk_duration": audio_chunk_duration,
            "audio_chunk_threshold": audio_chunk_threshold,
            "request_timeout_s": request_timeout_s,
            "seed": seed,
        }
        # The server caps ``input`` at 10,000 characters (HTTP 422 above
        # that), so long texts are synthesized chunk by chunk and joined.
        # Chunks inherit the same voice/instructions/parameters; the audio is
        # concatenated in order, which is exactly what one giant request would
        # have produced if the API accepted it.  A chunk that comes back as a
        # drone is drawn again (and repaired sentence by sentence if needed),
        # so the caller still receives one continuous take per text.
        def _send(piece: str, piece_seed: Optional[int]):
            return self._post_speech(
                piece,
                url=url,
                voice=voice,
                instructions=instructions,
                speed=speed,
                stream=stream,
                response_format=response_format,
                optional=optional,
                seed=piece_seed,
                request_timeout_s=request_timeout_s,
            )

        samples = None
        repairs: list = []
        for chunk in split_text_for_server(text):
            part, repair = self._synthesize_chunk(chunk, seed=seed, send=_send)
            if repair:
                repairs.append(repair)
            samples = part if samples is None else np.concatenate([samples, part])

        self.last_repairs = repairs
        return samples

    def _post_speech(
        self,
        chunk: str,
        *,
        url: str,
        voice: str,
        instructions: str,
        speed: float,
        stream: bool,
        response_format: str,
        optional: Dict[str, Any],
        seed: Optional[int],
        request_timeout_s: int | None,
    ) -> tuple:
        """One ``/v1/audio/speech`` request; returns ``(wav_bytes, headers)``."""
        import urllib.request  # noqa: PLC0415

        payload: Dict[str, Any] = {
            "model": "omnivoice",
            "input": chunk,
            "voice": voice,
            "response_format": response_format or "wav",
            "speed": speed,
            "stream": bool(stream),
        }
        for name, value in optional.items():
            if name == "seed":
                continue
            if value is not None:
                payload[name] = value
        if seed is not None:
            payload["seed"] = seed
        # Voice design instructions are the strongest control and are
        # only sent when the caller provided a real description.
        if instructions:
            payload["instructions"] = instructions

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        # Long texts legitimately take minutes on CPU; the socket timeout
        # must scale with the text or long recordings die mid-flight with
        # "Synthesis failed: timed out".
        request_timeout = self._request_timeout(request_timeout_s, chunk)
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                wav_bytes = resp.read()
                headers = _response_headers(resp)
        except Exception as exc:
            raise OmniVoiceServerError(
                f"Synthesis failed: {self._request_error_detail(exc)}"
            ) from exc
        return wav_bytes, headers

    def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio_path: str,
        ref_text: str = "",
        speed: float = 1.0,
        num_step: int | None = None,
        guidance_scale: float | None = None,
        denoise: bool | None = None,
        t_shift: float | None = None,
        position_temperature: float | None = None,
        class_temperature: float | None = None,
        duration: float | None = None,
        language: str | None = None,
        layer_penalty_factor: float | None = None,
        preprocess_prompt: bool | None = None,
        postprocess_output: bool | None = None,
        audio_chunk_duration: float | None = None,
        audio_chunk_threshold: float | None = None,
        request_timeout_s: int | None = None,
        seed: int | None = None,
        response_format: str = "wav",
    ) -> np.ndarray:
        """Synthesize text with voice cloning via the server (multipart)."""
        import urllib.request  # noqa: PLC0415
        import io  # noqa: PLC0415

        if not text.strip():
            raise ValueError("Nothing to synthesize")

        url = f"{self.base_url}/v1/audio/speech/clone"
        # The server caps ``text`` at 10,000 characters (HTTP 422 above
        # that), so long texts are synthesized chunk by chunk against the
        # same uploaded reference sample and joined in order.  A drone chunk
        # is drawn again (and repaired sentence by sentence if needed), so the
        # caller still receives one continuous take per text.
        def _send(piece: str, piece_seed: Optional[int]):
            return self._post_clone(
                piece,
                url=url,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                speed=speed,
                response_format=response_format,
                num_step=num_step,
                guidance_scale=guidance_scale,
                denoise=denoise,
                t_shift=t_shift,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                duration=duration,
                language=language,
                layer_penalty_factor=layer_penalty_factor,
                preprocess_prompt=preprocess_prompt,
                postprocess_output=postprocess_output,
                audio_chunk_duration=audio_chunk_duration,
                audio_chunk_threshold=audio_chunk_threshold,
                request_timeout_s=request_timeout_s,
                seed=piece_seed,
            )

        chunks = split_text_for_server(text)
        samples = None
        repairs: list = []
        for chunk in chunks:
            part, repair = self._synthesize_chunk(chunk, seed=seed, send=_send)
            if repair:
                repairs.append(repair)
            samples = part if samples is None else np.concatenate([samples, part])

        self.last_repairs = repairs
        return samples

    def _post_clone(
        self,
        chunk: str,
        *,
        url: str,
        ref_audio_path: str,
        ref_text: str,
        speed: float,
        response_format: str,
        num_step: int | None,
        guidance_scale: float | None,
        denoise: bool | None,
        t_shift: float | None,
        position_temperature: float | None,
        class_temperature: float | None,
        duration: float | None,
        language: str | None,
        layer_penalty_factor: float | None,
        preprocess_prompt: bool | None,
        postprocess_output: bool | None,
        audio_chunk_duration: float | None,
        audio_chunk_threshold: float | None,
        request_timeout_s: int | None,
        seed: int | None,
    ) -> tuple:
        """One ``/v1/audio/speech/clone`` request; ``(wav_bytes, headers)``."""
        import urllib.request  # noqa: PLC0415

        body = self._clone_multipart_body(
            chunk,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            speed=speed,
            response_format=response_format,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            t_shift=t_shift,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
            duration=duration,
            language=language,
            layer_penalty_factor=layer_penalty_factor,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            audio_chunk_duration=audio_chunk_duration,
            audio_chunk_threshold=audio_chunk_threshold,
            request_timeout_s=request_timeout_s,
            seed=seed,
        )

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "multipart/form-data; boundary=----AIVoiceStudioBoundary",
            },
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        # Cloning adds reference-audio preprocessing on top of generation,
        # so its timeout gets the same text-aware scale plus headroom.
        request_timeout = self._request_timeout(request_timeout_s, chunk)
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                wav_bytes = resp.read()
                headers = _response_headers(resp)
        except Exception as exc:
            raise OmniVoiceServerError(
                f"Clone synthesis failed: {self._request_error_detail(exc)}"
            ) from exc
        return wav_bytes, headers

    @staticmethod
    def _clone_multipart_body(
        text: str,
        *,
        ref_audio_path: str,
        ref_text: str,
        speed: float,
        response_format: str,
        num_step: int | None,
        guidance_scale: float | None,
        denoise: bool | None,
        t_shift: float | None,
        position_temperature: float | None,
        class_temperature: float | None,
        duration: float | None,
        language: str | None,
        layer_penalty_factor: float | None,
        preprocess_prompt: bool | None,
        postprocess_output: bool | None,
        audio_chunk_duration: float | None,
        audio_chunk_threshold: float | None,
        request_timeout_s: int | None,
        seed: int | None,
    ) -> bytes:
        """Multipart body for one ``/v1/audio/speech/clone`` request."""
        # Build multipart form data
        boundary = "----AIVoiceStudioBoundary"
        body = b""

        def _add_field(name: str, value: Any) -> None:
            nonlocal body
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            body += f"{value}\r\n".encode()

        def _add_form(name: str, value: Any) -> None:
            """Add an optional scalar form field (skips None)."""
            if value is not None and value != "":
                _add_field(name, value)

        _add_field("text", text)
        _add_form("ref_text", ref_text)
        _add_form("speed", speed)
        _add_form("num_step", num_step)
        _add_form("guidance_scale", guidance_scale)
        _add_form("denoise", denoise)
        _add_form("t_shift", t_shift)
        _add_form("position_temperature", position_temperature)
        _add_form("class_temperature", class_temperature)
        _add_form("duration", duration)
        _add_form("language", language)
        _add_form("layer_penalty_factor", layer_penalty_factor)
        _add_form("preprocess_prompt", preprocess_prompt)
        _add_form("postprocess_output", postprocess_output)
        _add_form("audio_chunk_duration", audio_chunk_duration)
        _add_form("audio_chunk_threshold", audio_chunk_threshold)
        _add_form("request_timeout_s", request_timeout_s)
        _add_form("seed", seed)
        _add_form("response_format", response_format if response_format != "wav" else None)

        # Add reference audio file
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="ref_audio"; '
            f'filename="{os.path.basename(ref_audio_path)}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        with open(ref_audio_path, "rb") as fh:
            body += fh.read()
        body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return body

    @staticmethod
    def _http_error_detail(exc: Exception) -> str:
        """Best readable message from a failed HTTP request.

        The server puts the real reason (e.g. the inference traceback line)
        in the JSON error body; ``str(HTTPError)`` only says
        "HTTP Error 500", which hid the cause and made every request-level
        failure look like a dead server.
        """
        import json  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415

        if not isinstance(exc, urllib.error.HTTPError):
            return str(exc)
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return str(exc)
        detail = raw.strip()
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict) and err.get("message"):
                    detail = str(err["message"])
                elif payload.get("detail"):
                    detail = str(payload["detail"])
        except Exception:  # noqa: BLE001
            pass
        return f"server returned HTTP {exc.code}: {detail}"

    @staticmethod
    def _request_timeout(request_timeout_s: int | None, text: str) -> float:
        """Socket timeout for one synthesis request, in seconds.

        ``request_timeout_s`` (the server's per-request limit) wins when set.
        Otherwise the timeout scales with the text length: generation runs at
        roughly real-time or slower on CPU, so a long paragraph can need
        several minutes.  The previous fixed 120s timeout killed long
        recordings mid-synthesis; the server kept working.
        """
        if request_timeout_s:
            return float(request_timeout_s)
        chars = max(1, len(text or ""))
        # 30s base for connection/startup + 4s of headroom per 100 chars,
        # bounded to at least 120s and at most 30 minutes.
        scaled = 30.0 + chars * 0.04
        return float(min(max(scaled, 120.0), 1800.0))

    # -- Drone handling ------------------------------------------------------

    @staticmethod
    def _attempt_seed(seed: int | None, attempt: int) -> int | None:
        """Seed for one draw of a chunk (see ``omnivoice_quality``)."""
        return quality.attempt_seed(seed, attempt)

    def _draw(self, send):
        """``draw(text, seed) -> (samples, server_flagged)`` over ``send``.

        ``send(piece, seed)`` performs one request and returns
        ``(wav_bytes, headers)``.  The drone recovery only wants samples plus
        the server's own verdict, so this adapter is the whole of this
        engine's part in the shared policy.

        It is also where a failed *request* is absorbed, because that is a bad
        draw rather than a bad request: the server answers HTTP 500 when one
        generation came back empty (see ``REQUEST_ATTEMPTS`` above), so the
        same text is sent again — with ``quality.retry_seed`` giving it a
        different roll, while the seed of the next drone attempt stays what
        the shared policy computed.  The take that comes back is then judged
        exactly as a first take would be.  A failure that survives every
        attempt is raised unchanged, so a dead server still looks like one.
        """
        def draw(text: str, seed):
            failure: Exception | None = None
            for retry in range(REQUEST_ATTEMPTS):
                if failure is not None:
                    log.info(
                        "OmniVoice server request failed (%s); sending the "
                        "%d-character chunk again (attempt %d of %d).",
                        failure, len(text), retry + 1, REQUEST_ATTEMPTS,
                    )
                try:
                    wav_bytes, headers = send(
                        text, quality.retry_seed(seed, retry)
                    )
                except OmniVoiceServerError as exc:
                    if not _retryable_request_failure(str(exc)):
                        raise
                    failure = exc
                    continue
                return (
                    self._wav_bytes_to_samples(wav_bytes),
                    headers.get("x-no-speech-detected") == "true",
                )
            assert failure is not None  # REQUEST_ATTEMPTS >= 1
            raise failure

        return draw

    def _best_attempt(self, text: str, *, send, seed, attempts: int) -> tuple:
        """Draw ``text`` until a take passes the drone check, or attempts run out.

        Returns ``(samples, verdict, attempts_used)`` — the best-scoring take
        when every attempt was a drone, so the caller always has audio to fall
        back on.
        """
        return quality.best_take(
            text, self._draw(send), seed=seed, attempts=attempts
        )

    def _synthesize_chunk(self, chunk: str, *, seed, send) -> tuple:
        """One server-sized text chunk as samples, without the OmniVoice drone.

        Returns ``(samples, repair_record_or_None)``; the record describes what
        had to be done, and is ``None`` when the first draw was clean.

        The policy itself is shared with the direct OmniVoice engine
        (``omnivoice_quality``): a bad draw is repaired by drawing the chunk
        again — the failure is drawn per request at a roughly constant rate
        whatever the text size, so most chunks are fixed by a second (rarely
        third) draw and the retry costs nothing until it is needed — and when
        every draw of a chunk is a drone, the chunk is re-recorded in pieces
        instead.  Only ``samples`` is returned, joined in order, so one segment
        stays one file.  A chunk that stays a drone keeps its best-looking take
        and is reported instead of failing the whole recording.
        """
        return quality.recover(
            chunk,
            self._draw(send),
            seed=seed,
            attempts=self.drone_attempts,
            repair_attempts=self.repair_attempts,
        )

    def _repair_by_pieces(self, text: str, *, send, seed, depth: int,
                          budget: int) -> tuple:
        """Re-record ``text`` in smaller pieces and join them in order.

        Returns ``(samples, unresolved, pieces, draws)``; the splitting rules,
        the budget and the deep-split accounting all live in the shared
        ``omnivoice_quality.repair_by_pieces``.
        """
        return quality.repair_by_pieces(
            text,
            self._draw(send),
            seed=seed,
            depth=depth,
            budget=budget,
            repair_attempts=self.repair_attempts,
            max_depth=DRONE_REPAIR_MAX_DEPTH,
        )

    @staticmethod
    def _wav_bytes_to_samples(wav_bytes: bytes) -> np.ndarray:
        """Convert raw WAV bytes to int16 samples."""
        import wave  # noqa: PLC0415
        import io  # noqa: PLC0415

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                return np.frombuffer(frames, dtype=np.int16).copy()
        except Exception:
            # Fallback: assume raw PCM int16 at 24kHz
            return np.frombuffer(wav_bytes, dtype=np.int16).copy()

    def get_voices(self) -> list:
        """Fetch available voices from the server."""
        return self.get_voice_info().get("voices", [])

    def get_voice_info(self) -> dict:
        """Fetch the full ``/v1/voices`` response (voices + design
        attributes vocabulary), or ``{}`` when the server is unreachable."""
        import urllib.request  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415

        url = f"{self.base_url}/v1/voices"
        req = urllib.request.Request(url, method="GET")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def get_models(self) -> list:
        """List models advertised by the server (OpenAI-compatible)."""
        import urllib.request  # noqa: PLC0415

        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url, method="GET")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict):
                    return data.get("data") or data.get("models") or []
        except Exception:  # noqa: BLE001
            pass
        return []

    # -- Voice profiles (server-stored clones) -----------------------------

    def _build_profile_forms(
        self, *, profile_id: str, ref_audio_path: str, ref_text: str, overwrite: bool,
    ) -> bytes:
        """Multipart body for ``POST /v1/voices/profiles``."""
        boundary = "----AIVoiceStudioProfilesBoundary"
        body = b""

        def _add_field(name: str, value: str) -> None:
            nonlocal body
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            body += f"{value}\r\n".encode()

        _add_field("profile_id", profile_id)
        if ref_text:
            _add_field("ref_text", ref_text)
        if overwrite:
            _add_field("overwrite", "true")
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="ref_audio"; '
            f'filename="{os.path.basename(ref_audio_path)}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        with open(ref_audio_path, "rb") as fh:
            body += fh.read()
        body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return body

    def save_profile(
        self,
        profile_id: str,
        ref_audio_path: str,
        ref_text: str = "",
        overwrite: bool = False,
    ) -> dict:
        """Store a voice-cloning profile on the server (reusable)."""
        import urllib.request  # noqa: PLC0415

        url = f"{self.base_url}/v1/voices/profiles"
        body = self._build_profile_forms(
            profile_id=profile_id,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            overwrite=overwrite,
        )
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "multipart/form-data; "
                "boundary=----AIVoiceStudioProfilesBoundary",
            },
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise OmniVoiceServerError(
                f"Could not save voice profile: {exc}"
            ) from exc

    def list_profiles(self) -> list:
        """List server-stored clone profiles (from ``/v1/voices``)."""
        voices = self.get_voices()
        profiles = []
        for voice in voices:
            if isinstance(voice, dict) and voice.get("type") == "clone":
                profiles.append(
                    {
                        "profile_id": voice.get("profile_id"),
                        "description": voice.get("description", ""),
                    }
                )
        return [p for p in profiles if p.get("profile_id")]

    def get_profile(self, profile_id: str) -> dict | None:
        """Fetch a single server-stored voice profile."""
        import urllib.request  # noqa: PLC0415
        import urllib.parse  # noqa: PLC0415

        url = f"{self.base_url}/v1/voices/profiles/{urllib.parse.quote(profile_id)}"
        req = urllib.request.Request(url, method="GET")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a server-stored voice profile."""
        import urllib.request  # noqa: PLC0415
        import urllib.parse  # noqa: PLC0415

        url = f"{self.base_url}/v1/voices/profiles/{urllib.parse.quote(profile_id)}"
        req = urllib.request.Request(url, method="DELETE")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# Singleton server manager
# ---------------------------------------------------------------------------
_server_manager: OmniVoiceServerManager | None = None
_server_lock = threading.Lock()


def get_server_manager(**kwargs) -> OmniVoiceServerManager:
    """Get or create the singleton OmniVoiceServerManager."""
    global _server_manager  # noqa: PLW0603
    with _server_lock:
        if _server_manager is None:
            _server_manager = OmniVoiceServerManager(**kwargs)
        return _server_manager


# ---------------------------------------------------------------------------
# Engine wrapper (plugs into get_engine interface)
# ---------------------------------------------------------------------------
class OmniVoiceServerEngine:
    """Wraps the OmniVoice HTTP server so it plugs into the TTS engine cache.

    Implements the same ``synthesize`` contract as ``TtsEngine``: returns
    int16 samples at ``sample_rate`` (24 kHz).
    """

    sample_rate = _SAMPLE_RATE

    def __init__(self, voice_entry: dict):
        self.voice_entry = voice_entry
        # Read the user's configured server settings so we connect
        # to the same server the user started from Settings.
        cfg = _read_server_config()
        self._server = get_server_manager(
            host=cfg.get("host", DEFAULT_HOST),
            port=cfg.get("port", DEFAULT_PORT),
        )
        if not self._server.is_running:
            self._server.start()

        # Per-project OmniVoice options (see omnivoice/spec.apply_omni_to_voice).
        from ..omnivoice import spec  # noqa: PLC0415
        self._omni = voice_entry.get("omni") or {}
        self._instruct = spec.resolve_instruct(
            "omnivoice_server",
            voice_entry.get("voice"),
            self._omni.get("instruct") or voice_entry.get("instruct"),
        )
        self._ref_audio = (
            self._omni.get("ref_audio") or voice_entry.get("ref_audio") or ""
        ).strip()
        self._ref_text = (
            self._omni.get("ref_text") or voice_entry.get("ref_text") or ""
        ).strip()
        self._language = spec.clean_language(
            self._omni.get("language") or voice_entry.get("language")
        )
        #: What the last segment needed fixing (see ``_repair_message``), so a
        #: caller can warn the user instead of shipping a droning take.
        self.last_warnings: list = []

    def _omni_kwargs(self) -> Dict[str, Any]:
        """Advanced generation knobs configured for this voice.

        Client-level defaults keep the exact request behaviour the studio had
        before these knobs existed (guidance 3.0, denoise on); anything the
        user configured overrides them.  Values stay ``None`` only for
        parameters that were never sent historically, so the server's own
        defaults apply.
        """
        omni = self._omni

        def _or(value: Any, default: Any) -> Any:
            return value if value is not None else default

        return {
            "num_step": omni.get("num_step"),
            "guidance_scale": _or(omni.get("guidance_scale"), 3.0),
            "denoise": _or(omni.get("denoise"), True),
            "class_temperature": omni.get("class_temperature"),
            "position_temperature": omni.get("position_temperature"),
            "t_shift": omni.get("t_shift"),
            "duration": omni.get("duration"),
            "seed": omni.get("seed"),
            "language": self._language,
        }

    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> np.ndarray:
        """Synthesize text; returns int16 samples at 24 kHz."""
        if not text.strip():
            raise ValueError("Nothing to synthesize")

        # Quick health check with retries — avoid hanging on a dead server
        # without mis-judging a busy one (cloning preprocessing can stall
        # /health for several seconds).  If the probe still fails we do NOT
        # abort here: the synthesis request itself is the real test, and it
        # carries a text-aware timeout plus its own error reporting.  A
        # single missed probe used to kill clone requests with "server not
        # responding" while the server was merely busy.
        if not self._server.health_check(retries=3, delay=1.0):
            log.warning(
                "OmniVoice server health probe failed before synthesis; "
                "attempting the request anyway"
            )

        kwargs = self._omni_kwargs()
        if self._ref_audio and os.path.isfile(self._ref_audio):
            samples = self._server.synthesize_clone(
                text,
                ref_audio_path=self._ref_audio,
                ref_text=self._ref_text,
                speed=speed,
                **kwargs,
            )
        else:
            # Voice design (instructions) or an OpenAI preset name.
            voice_id = self.voice_entry.get("voice", "alloy")
            samples = self._server.synthesize(
                text,
                voice=voice_id,
                instructions=self._instruct,
                speed=speed,
                **kwargs,
            )

        self._report_quality(samples)

        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415
            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415
            samples = _apply_volume(samples, volume)
        return samples

    def _report_quality(self, samples) -> None:
        """Turn the server pipeline's repairs into log lines and warnings.

        Every drone the client had to re-draw or re-record is named here, so a
        segment that needed fixing can be found in the app log; a take that
        still looks like noise is warned about loudly because the recording
        finished "successfully" and nothing else would tell the user.
        """
        records = list(getattr(self._server, "last_repairs", []) or [])
        self.last_warnings = [self._repair_message(r) for r in records]
        for message in self.last_warnings:
            log.warning(message)
        # Safety net over the finished take: a boundary between two repaired
        # chunks, or a drone shorter than one chunk's verdict, would only be
        # visible on the whole segment.
        verdict = quality.judge(samples)
        if verdict.bad:
            message = (
                "OmniVoice thinks part of this segment is noise rather than "
                f"speech ({verdict.reason}). Listen to it before publishing."
            )
            self.last_warnings.append(message)
            log.warning(message)

    @staticmethod
    def _repair_message(record: dict) -> str:
        """One user-readable line about a repaired (or unrepaired) drone."""
        return quality.repair_message(record)

    def close(self) -> None:
        """Stop the server (only if we started it)."""
        if self._server and self._server.is_running:
            self._server.stop()
