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

log = logging.getLogger(__name__)

_SAMPLE_RATE = 24000  # OmniVoice output sample rate

# The omnivoice-server HTTP API validates ``input`` (JSON) and ``text``
# (multipart) with ``max_length=10_000`` — a longer text is rejected before
# inference even starts with a bare "HTTP 422: Request validation failed".
# A whole book chapter easily exceeds that, so long texts are split into
# server-sized chunks at sentence/paragraph boundaries, synthesized one by
# one and concatenated; the audio the caller receives is identical to what
# one giant request would produce (if the server accepted it).
_TEXT_CHUNK_TARGET = 9_500


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
_SENTENCE_END = ".!?\u0964\u0965\u3002\uff01\uff1f\u2026\"\u201d\u2019\u00bb"


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
        sentences: list = []
        start = 0
        for i, ch in enumerate(paragraph):
            if ch in _SENTENCE_END:
                # Include trailing quotes/brackets after the terminator.
                sentences.append(paragraph[start:i + 1])
                start = i + 1
        if start < len(paragraph):
            sentences.append(paragraph[start:])
        buf: list = []
        buf_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > limit:
                # A single "sentence" longer than the cap (no punctuation at
                # all): hard-cut so the request still validates.
                if buf:
                    chunks.append(" ".join(buf))
                    buf, buf_len = [], 0
                for j in range(0, len(sentence), limit):
                    piece = sentence[j:j + limit].strip()
                    if piece:
                        chunks.append(piece)
                continue
            if buf_len + len(sentence) + 1 > limit and buf:
                chunks.append(" ".join(buf))
                buf, buf_len = [], 0
            buf.append(sentence)
            buf_len += len(sentence) + 1
        if buf:
            chunks.append(" ".join(buf))
    return chunks if chunks else [text[:limit]]


# ---------------------------------------------------------------------------
# Package detection (managed venv)
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """True when ``omnivoice_server`` is importable from the managed venv."""
    try:
        from ..python_runtime import get_runtime  # noqa: PLC0415
        rt = get_runtime()
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
        from ..python_runtime import get_runtime  # noqa: PLC0415
        rt = get_runtime()
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
    ):
        self._host = host
        self._port = port
        self._device = device
        self._num_steps = num_steps
        self._max_concurrent = max_concurrent
        self._api_key = api_key
        self._cors_origins = cors_origins
        self._model_id = model_id
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

        from ..python_runtime import get_runtime  # noqa: PLC0415
        rt = get_runtime()

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
        self._log_path = os.path.join(logs_dir(), "omnivoice_server.log")
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

    def _log_tail(self, lines: int = 40) -> str:
        """Last lines of the server log file (for error reporting)."""
        if not self._log_path or not os.path.isfile(self._log_path):
            return ""
        try:
            with open(self._log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except Exception:  # noqa: BLE001
            return ""

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
        # have produced if the API accepted it.
        samples = None
        for chunk in split_text_for_server(text):
            payload: Dict[str, Any] = {
                "model": "omnivoice",
                "input": chunk,
                "voice": voice,
                "response_format": response_format or "wav",
                "speed": speed,
                "stream": bool(stream),
            }
            for name, value in optional.items():
                if value is not None:
                    payload[name] = value
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
            except Exception as exc:
                raise OmniVoiceServerError(
                    f"Synthesis failed: {self._http_error_detail(exc)}"
                ) from exc

            part = self._wav_bytes_to_samples(wav_bytes)
            samples = part if samples is None else np.concatenate([samples, part])

        return samples

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
        # same uploaded reference sample and joined in order.
        chunks = split_text_for_server(text)
        samples = None
        for chunk in chunks:
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
            except Exception as exc:
                raise OmniVoiceServerError(
                    f"Clone synthesis failed: {self._http_error_detail(exc)}"
                ) from exc

            part = self._wav_bytes_to_samples(wav_bytes)
            samples = part if samples is None else np.concatenate([samples, part])

        return samples

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

        if pitch != 1.0:
            from ..tts.engine import _shift_pitch  # noqa: PLC0415
            samples = _shift_pitch(samples, pitch)
        if volume != 1.0:
            from ..tts.engine import _apply_volume  # noqa: PLC0415
            samples = _apply_volume(samples, volume)
        return samples

    def close(self) -> None:
        """Stop the server (only if we started it)."""
        if self._server and self._server.is_running:
            self._server.stop()
