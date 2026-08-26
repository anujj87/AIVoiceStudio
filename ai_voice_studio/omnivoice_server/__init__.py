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

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def is_running(self) -> bool:
        """True when we started a subprocess that is still alive,
        OR when a server is reachable via health check (started externally)."""
        if self._proc is not None and self._proc.poll() is None:
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

        log.info("Starting OmniVoice server: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=flags,
                env=env,
            )
        except OSError as exc:
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
                output = ""
                if self._proc and self._proc.stdout:
                    try:
                        output = self._proc.stdout.read(4096)
                    except Exception:  # noqa: BLE001
                        pass
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
            log.info("OmniVoice server stopped.")

    def restart(self, timeout: float = 120.0) -> None:
        """Restart the server (stop then start)."""
        self.stop()
        self.start(timeout=timeout)

    def health_check(self) -> bool:
        """Return True if the server responds to /health."""
        import urllib.request  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415

        try:
            url = f"{self.base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:  # noqa: BLE001
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
        guidance_scale: float = 3.0,
        denoise: bool = True,
    ) -> np.ndarray:
        """Synthesize text via the server; returns int16 samples at 24 kHz."""
        import urllib.request  # noqa: PLC0415

        if not text.strip():
            raise ValueError("Nothing to synthesize")

        url = f"{self.base_url}/v1/audio/speech"
        payload: Dict[str, Any] = {
            "model": "omnivoice",
            "input": text,
            "voice": voice,
            "response_format": "wav",
            "speed": speed,
            "stream": stream,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
        }
        if instructions:
            payload["instructions"] = instructions
        if num_step is not None:
            payload["num_step"] = num_step

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                wav_bytes = resp.read()
        except Exception as exc:
            raise OmniVoiceServerError(
                f"Synthesis failed: {exc}"
            ) from exc

        return self._wav_bytes_to_samples(wav_bytes)

    def synthesize_clone(
        self,
        text: str,
        *,
        ref_audio_path: str,
        ref_text: str = "",
        speed: float = 1.0,
        num_step: int | None = None,
    ) -> np.ndarray:
        """Synthesize text with voice cloning via the server."""
        import urllib.request  # noqa: PLC0415
        import io  # noqa: PLC0415

        if not text.strip():
            raise ValueError("Nothing to synthesize")

        url = f"{self.base_url}/v1/audio/speech/clone"

        # Build multipart form data
        boundary = "----AIVoiceStudioBoundary"
        body = b""

        def _add_field(name: str, value: str) -> None:
            nonlocal body
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            body += f"{value}\r\n".encode()

        _add_field("text", text)
        if ref_text:
            _add_field("ref_text", ref_text)
        _add_field("speed", str(speed))
        if num_step is not None:
            _add_field("num_step", str(num_step))

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

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                wav_bytes = resp.read()
        except Exception as exc:
            raise OmniVoiceServerError(
                f"Clone synthesis failed: {exc}"
            ) from exc

        return self._wav_bytes_to_samples(wav_bytes)

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
        import urllib.request  # noqa: PLC0415
        import urllib.error  # noqa: PLC0415

        url = f"{self.base_url}/v1/voices"
        req = urllib.request.Request(url, method="GET")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("voices", [])
        except Exception:  # noqa: BLE001
            return []


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

        # Quick health check — avoid hanging on a stuck server.
        if not self._server.health_check():
            raise OmniVoiceServerError(
                f"OmniVoice server at {self._server.base_url} is not responding. "
                "Try restarting it from Settings > OmniVoice Server."
            )

        instruct = self.voice_entry.get("instruct", "")
        ref_audio = self.voice_entry.get("ref_audio", "")
        ref_text = self.voice_entry.get("ref_text", "")

        if ref_audio and os.path.isfile(ref_audio):
            samples = self._server.synthesize_clone(
                text,
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                speed=speed,
            )
        else:
            # Use voice design via instructions
            voice_id = self.voice_entry.get("voice", "alloy")
            samples = self._server.synthesize(
                text,
                voice=voice_id,
                instructions=instruct,
                speed=speed,
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
