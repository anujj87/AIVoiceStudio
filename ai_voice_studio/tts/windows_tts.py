"""Windows system voices: SAPI5 and Windows Core (OneCore).

The built-in sherpa-onnx engines need a downloaded model.  Windows itself
ships voices that are always available, and this module makes them usable as
ordinary TTS engines everywhere a voice is chosen (Available TTS, Punctuation,
Recording settings and the Recording window):

* ``sapi5``        - the classic "Desktop" voices of the Speech API
  (Microsoft David Desktop, Zira Desktop, Hazel Desktop, ...), reached through
  ``System.Speech.Synthesis.SpeechSynthesizer``.
* ``windows_core`` - the modern Windows voices used by Narrator (Microsoft
  David, Mark, Zira, Hazel, Heera, Kalpana, ...), reachable only through the
  Windows Runtime ``Windows.Media.SpeechSynthesis`` API.

Neither engine needs a download, a package or a model file, so both are listed
as ready to use.  Synthesis is performed by Windows PowerShell (present on
every supported Windows), which writes a WAV file that :class:`WindowsVoiceEngine`
reads back - no pywin32 / pyttsx3 / COM registration is required.

Enumerating the voices costs one PowerShell start-up per engine, so results
are cached in memory and refreshed in the background (the settings panels keep
drawing while the probe runs and fill themselves in when it finishes).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

SAPI5 = "sapi5"
WINDOWS_CORE = "windows_core"
ENGINE_IDS: Tuple[str, ...] = (SAPI5, WINDOWS_CORE)

#: Display names, kept in step with the catalog entries of the same id.
ENGINE_NAMES: Dict[str, str] = {
    SAPI5: "SAPI5 (Windows voices)",
    WINDOWS_CORE: "Windows Core voices",
}

#: Catalog variant every built-in voice belongs to.
VARIANT = "installed"

_CACHE_TTL = 300.0
_cache: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}
_loading: set[str] = set()
_listeners: Dict[str, List[Callable[[List[Dict[str, str]]], None]]] = {}
_lock = threading.RLock()

_SYNTH_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """True when the Windows speech APIs can be used on this machine."""
    return sys.platform == "win32"


def is_system_engine(engine_id: str | None) -> bool:
    return (engine_id or "") in ENGINE_IDS


def _powershell() -> str:
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"


def engine_name(engine_id: str) -> str:
    return ENGINE_NAMES.get(engine_id, engine_id)


# ---------------------------------------------------------------------------
# PowerShell helpers
# ---------------------------------------------------------------------------
_LIST_SCRIPT: Dict[str, str] = {
    SAPI5: r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $synth.GetInstalledVoices()) {
  if ($v.Enabled) {
    $info = $v.VoiceInfo
    Write-Output ($info.Name + "`t" + $info.Gender + "`t" + $info.Culture.Name)
  }
}
$synth.Dispose()
""",
    WINDOWS_CORE: r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime] | Out-Null
foreach ($v in [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices) {
  Write-Output ($v.DisplayName + "`t" + $v.Gender + "`t" + $v.Language)
}
""",
}

_SYNTH_SCRIPT: Dict[str, str] = {
    SAPI5: r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$text = [System.IO.File]::ReadAllText($env:AIVS_TEXT_FILE, [System.Text.Encoding]::UTF8)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:AIVS_VOICE) { $synth.SelectVoice($env:AIVS_VOICE) }
$synth.Rate = [int]$env:AIVS_SAPI_RATE
$synth.SetOutputToWaveFile($env:AIVS_OUT_FILE)
$synth.Speak($text)
$synth.SetOutputToNull()
$synth.Dispose()
""",
    WINDOWS_CORE: r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime] | Out-Null
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
  $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$text = [System.IO.File]::ReadAllText($env:AIVS_TEXT_FILE, [System.Text.Encoding]::UTF8)
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
if ($env:AIVS_VOICE) {
  foreach ($v in [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices) {
    if ($v.DisplayName -eq $env:AIVS_VOICE) { $synth.Voice = $v; break }
  }
}
$synth.Options.SpeakingRate = [double]$env:AIVS_SPEAKING_RATE
$task = $synth.SynthesizeTextToStreamAsync($text)
$netTask = $asTask.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]).Invoke($null, @($task))
$netTask.Wait(-1) | Out-Null
$stream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($netTask.Result)
$out = [System.IO.File]::Create($env:AIVS_OUT_FILE)
$stream.CopyTo($out)
$out.Close()
$synth.Dispose()
""",
}


def _run_powershell(script: str, env: dict | None = None, timeout: int = 60):
    from ..util import run_tracked  # noqa: PLC0415

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    # run_tracked keeps the child reachable by the shutdown hook: the voice
    # enumeration runs on a daemon thread, and a PowerShell child still alive
    # while Python finalises used to crash the process on exit.
    return run_tracked(
        [_powershell(), "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-Command", script],
        encoding="utf-8", errors="replace",
        timeout=timeout, creationflags=creationflags, env=env,
        cwd=tempfile.gettempdir(),
    )


# ---------------------------------------------------------------------------
# Voice enumeration (cached, never blocking the caller)
# ---------------------------------------------------------------------------
def _enumerate(engine_id: str) -> List[Dict[str, str]]:
    script = _LIST_SCRIPT.get(engine_id)
    if not script:
        return []
    result = _run_powershell(script, timeout=60)
    voices: List[Dict[str, str]] = []
    for row in (result.stdout or "").splitlines():
        cells = [cell.strip() for cell in row.split("\t")]
        if len(cells) < 3 or not cells[0]:
            continue
        name, gender, locale = cells[0], cells[1], cells[2]
        voices.append({"id": name, "name": name, "gender": gender, "language": locale})
    return voices


def _notify(engine_id: str, voices: List[Dict[str, str]]) -> None:
    with _lock:
        listeners = _listeners.pop(engine_id, [])
    for callback in listeners:
        try:
            callback(list(voices))
        except Exception:  # noqa: BLE001
            log.debug("Windows voice listener failed", exc_info=True)


def _load_job(engine_id: str) -> None:
    try:
        found = _enumerate(engine_id)
    except Exception:  # noqa: BLE001
        log.debug("Could not enumerate %s voices", engine_id, exc_info=True)
        found = []
    with _lock:
        _cache[engine_id] = (time.time(), found)
        _loading.discard(engine_id)
    _notify(engine_id, found)


def refresh(engine_id: str, on_ready=None, force: bool = False) -> None:
    """(Re)enumerate ``engine_id``'s voices.

    ``on_ready(voices)`` runs on the worker thread (the GUI wraps it in
    ``wx.CallAfter``).  A cached, still-fresh list is handed back immediately
    instead of starting PowerShell again.
    """
    if on_ready is not None:
        with _lock:
            _listeners.setdefault(engine_id, []).append(on_ready)
    if not is_available():
        _notify(engine_id, [])
        return
    entries: Optional[List[Dict[str, str]]] = None
    with _lock:
        cached = _cache.get(engine_id)
        fresh = cached is not None and (time.time() - cached[0]) < _CACHE_TTL
        if engine_id in _loading:
            return
        if fresh and not force:
            entries = list(cached[1])  # type: ignore[index]
        else:
            _loading.add(engine_id)
    if entries is not None:
        _notify(engine_id, entries)
        return
    threading.Thread(
        target=_load_job, args=(engine_id,), daemon=True,
        name=f"aivs-win-voices-{engine_id}",
    ).start()


def _is_fresh(engine_id: str) -> bool:
    with _lock:
        cached = _cache.get(engine_id)
    return cached is not None and (time.time() - cached[0]) < _CACHE_TTL


def voices(engine_id: str) -> List[Dict[str, str]]:
    """Cached voice descriptors; starts a background refresh when stale."""
    if not is_available():
        return []
    with _lock:
        cached = _cache.get(engine_id)
    if cached is None:
        refresh(engine_id)
        return []
    if (time.time() - cached[0]) >= _CACHE_TTL:
        refresh(engine_id)
    return list(cached[1])


def voice_entries(engine_ids: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Consumer voice entries (the shape the GUI cascades expect)."""
    ids = tuple(engine_ids) if engine_ids else ENGINE_IDS
    entries: List[Dict[str, Any]] = []
    for engine_id in ids:
        for index, voice in enumerate(voices(engine_id)):
            entries.append(_entry(engine_id, index, voice))
    return entries


def _entry(engine_id: str, index: int, voice: Dict[str, str]) -> Dict[str, Any]:
    gender = (voice.get("gender") or "").strip()
    locale = (voice.get("language") or "").strip()
    label = voice.get("name") or voice.get("id") or f"Voice {index}"
    details = ", ".join(bit for bit in (gender.title(), locale) if bit)
    if details:
        label = f"{label} ({details})"
    return {
        "tts": engine_id,
        "tts_name": engine_name(engine_id),
        "language": locale or "en",
        "variant": VARIANT,
        "voice": voice.get("id", ""),
        "voice_name": label,
        "sid": index,
        "engine": engine_id,
        "dir": "",
        "builtin": True,
        "requires_gpu": False,
    }


def add_installed_voices(target: List[Dict[str, Any]], on_ready=None) -> None:
    """Append the built-in Windows voices to ``target`` (never blocking).

    ``on_ready`` is called (worker thread) when a fresh enumeration lands, so
    callers can refresh their lists.  Cached results are appended right away.
    """
    if not is_available():
        return
    for engine_id in ENGINE_IDS:
        voices(engine_id)  # cached, or kicks off one background probe
        # Only listen while a probe is actually pending: a listener attached
        # to an already-cached list would fire immediately and a callback
        # that rebuilds the list would then recurse.
        if on_ready is not None and not _is_fresh(engine_id):
            refresh(engine_id, on_ready)
    target.extend(voice_entries())


# ---------------------------------------------------------------------------
# WAV reading
# ---------------------------------------------------------------------------
def _read_wav(path: str) -> Tuple["np.ndarray", int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if not frames:
        raise RuntimeError("The Windows voice produced no audio.")
    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2")
    elif width == 4:
        from .engine import samples_to_int16  # noqa: PLC0415

        samples = samples_to_int16(np.frombuffer(frames, dtype="<f4"))
    elif width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        samples = np.clip((raw - 128.0) * 256.0, -32768, 32767).astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported WAV sample width ({width} bytes).")
    if channels > 1:
        samples = samples.reshape(-1, channels)[:, 0]
    return samples, rate


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class WindowsVoiceEngine:
    """Duck-types :class:`ai_voice_studio.tts.engine.TtsEngine` for Windows voices.

    ``synthesize`` returns int16 mono samples and sets ``sample_rate`` to the
    rate Windows produced (SAPI5: usually 22050 Hz, Windows Core: 16000 Hz).
    """

    def __init__(self, voice_entry: Dict[str, Any]):
        self.engine = str(voice_entry.get("engine", ""))
        if self.engine not in ENGINE_IDS:
            raise ValueError(f"Unknown Windows voice engine '{self.engine}'.")
        if not is_available():
            raise ValueError(
                "SAPI5 and Windows Core voices are only available on Windows."
            )
        self.voice_entry = voice_entry
        self.voice_id = str(voice_entry.get("voice") or "")
        self.sample_rate = 22050

    # -- helpers ------------------------------------------------------------
    def _select_voice(self, available: List[Dict[str, str]]) -> str:
        """Resolve the configured voice against a fresh enumeration."""
        if self.voice_id and any(v.get("id") == self.voice_id for v in available):
            return self.voice_id
        index = int(self.voice_entry.get("sid", 0) or 0)
        if 0 <= index < len(available):
            return str(available[index].get("id", ""))
        return ""

    def _speak_to_wav(self, text: str, speed: float) -> str:
        engine_id = self.engine
        script = _SYNTH_SCRIPT[engine_id]
        fd, out_path = tempfile.mkstemp(prefix="aivs_win_", suffix=".wav")
        os.close(fd)
        fd, text_path = tempfile.mkstemp(prefix="aivs_win_", suffix=".txt")
        os.close(fd)
        with open(text_path, "w", encoding="utf-8") as fh:
            fh.write(text)

        try:
            speed = float(speed or 1.0)
            env = dict(os.environ)
            env["AIVS_TEXT_FILE"] = text_path
            env["AIVS_OUT_FILE"] = out_path
            env["AIVS_VOICE"] = self._select_voice(voices(engine_id))
            # SAPI5 wants an integer -10..10, Windows Core a 0.5..6.0 float.
            env["AIVS_SAPI_RATE"] = str(
                max(-10, min(10, int(round((speed - 1.0) * 10.0))))
            )
            env["AIVS_SPEAKING_RATE"] = str(max(0.5, min(6.0, speed)))
            result = _run_powershell(script, env=env, timeout=_SYNTH_TIMEOUT)
        except Exception:
            try:
                os.remove(out_path)
            except OSError:
                pass
            raise
        finally:
            try:
                os.remove(text_path)
            except OSError:
                pass

        ok = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        if result.returncode != 0 or not ok:
            try:
                os.remove(out_path)
            except OSError:
                pass
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            reason = detail[-1] if detail else f"PowerShell exit code {result.returncode}"
            raise RuntimeError(
                f"The {engine_name(engine_id)} voice could not be synthesized: {reason}"
            )
        return out_path

    # -- engine interface ---------------------------------------------------
    def synthesize(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
    ) -> "np.ndarray":
        if not text.strip():
            raise ValueError("Nothing to synthesize")
        out_path = self._speak_to_wav(text, speed)
        try:
            samples, rate = _read_wav(out_path)
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass
        self.sample_rate = rate
        if pitch != 1.0 or volume != 1.0:
            from .engine import _apply_volume, _shift_pitch  # noqa: PLC0415

            if pitch != 1.0:
                samples = _shift_pitch(samples, pitch)
            if volume != 1.0:
                samples = _apply_volume(samples, volume)
        return samples
