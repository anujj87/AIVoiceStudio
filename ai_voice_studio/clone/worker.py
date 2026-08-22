"""XTTS v2 cloning worker subprocess.

Runs ``python -m ai_voice_studio.clone.worker`` and speaks JSON lines on
stdin/stdout (one request per line, one response per line). The worker runs in
its own process so its PyTorch/onnxruntime can never conflict with the
sherpa-onnx runtime in the main app. The ``coqui-tts`` package is imported
from the clone runtime folder (``%APPDATA%/AIVoiceStudio/runtime/clone``);
the first ``synthesize`` call also downloads the ~2 GB XTTS model
automatically (Coqui CPML non-commercial licence).

Protocol (requests from the app, responses ``{"id": ..., "ok": ...}``):
* ``{"cmd": "ping"}``                     -- engine availability probe
* ``{"cmd": "synthesize", "text", "sample_wav", "language", "speed"}``
* ``{"cmd": "quit"}``                     -- graceful shutdown
"""

from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np


def _runtime_dir() -> str:
    try:
        from .. import paths  # noqa: PLC0415

        return os.path.join(paths.user_data_dir(), "runtime", "clone")
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    runtime_dir = _runtime_dir()
    if runtime_dir and os.path.isdir(runtime_dir):
        sys.path.insert(0, runtime_dir)

    out = sys.stdout
    err = sys.stderr
    tts = None

    def respond(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            respond({"ok": False, "error": "Invalid request."})
            continue
        cmd = req.get("cmd")

        if cmd == "quit":
            break

        if cmd == "ping":
            try:
                import coqui_tts  # noqa: PLC0415, F401

                respond({"ok": True, "engine": True})
            except Exception as exc:  # noqa: BLE001
                respond({
                    "ok": True, "engine": False,
                    "error": f"Cloning engine not installed: {exc}",
                })
            continue

        if cmd == "synthesize":
            try:
                if tts is None:
                    # Lazy: first request pays the torch import + model download.
                    try:
                        from coqui_tts.api import TTS  # noqa: PLC0415
                    except ImportError as exc:
                        raise RuntimeError(
                            f"Cloning engine not installed: {exc}"
                        ) from exc
                    tts = TTS(
                        "tts_models/multilingual/multi-dataset/xtts_v2"
                    ).to("cpu")
                text = req["text"]
                sample = req["sample_wav"]
                language = req.get("language", "en")
                speed = float(req.get("speed", 1.0))
                # Keep library prints off the protocol pipe (stdout). tqdm /
                # HF progress bars already go to stderr; this catches the rest.
                saved_stdout = sys.stdout
                sys.stdout = err
                try:
                    wav = tts.tts(
                        text=text, speaker_wav=sample,
                        language=language, speed=speed,
                    )
                finally:
                    sys.stdout = saved_stdout
                arr = np.asarray(wav, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    raise ValueError("The engine returned no audio.")
                samples = np.clip(arr, -1.0, 1.0) * 32767.0
                wav_b64 = base64.b64encode(
                    samples.astype(np.int16).tobytes()
                ).decode("ascii")
                respond({"ok": True, "wav": wav_b64})
            except Exception as exc:  # noqa: BLE001
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        respond({"ok": False, "error": f"Unknown command: {cmd}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
