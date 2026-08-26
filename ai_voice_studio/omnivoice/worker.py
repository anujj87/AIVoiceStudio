"""OmniVoice TTS worker subprocess.

Runs ``python -m ai_voice_studio.omnivoice.worker`` and speaks JSON lines on
stdin/stdout (one request per line, one response per line).  The worker runs
in its own process so its PyTorch/CUDA runtime can never conflict with the
sherpa-onnx runtime in the main app.

The ``omnivoice-triton`` package must be installed in the Python environment
(``pip install omnivoice-triton``).  The first ``synthesize`` call also
downloads the ~2 GB OmniVoice model automatically from HuggingFace.

Protocol (requests from the app, responses ``{"id": ..., "ok": ...}``):
* ``{"cmd": "ping"}``                    -- engine availability probe
* ``{"cmd": "synthesize", "text", ...}`` -- synthesize speech
* ``{"cmd": "quit"}``                    -- graceful shutdown
"""

from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np


def main() -> int:
    out = sys.stdout
    err = sys.stderr
    runner = None

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
                from omnivoice_triton import create_runner  # noqa: PLC0415, F401
                respond({"ok": True, "engine": True})
            except ImportError as exc:
                respond({
                    "ok": True, "engine": False,
                    "error": (
                        "OmniVoice engine not installed. "
                        "Install with: pip install omnivoice-triton\n"
                        f"Details: {exc}"
                    ),
                })
            continue

        if cmd == "synthesize":
            try:
                if runner is None:
                    # Lazy: first request pays the torch import + model download.
                    try:
                        from omnivoice_triton import create_runner  # noqa: PLC0415
                    except ImportError as exc:
                        raise RuntimeError(
                            "OmniVoice engine not installed. "
                            "Install with: pip install omnivoice-triton"
                        ) from exc

                    mode = req.get("mode", "triton")
                    runner = create_runner(mode)
                    runner.load_model()

                text = req["text"]
                speed = float(req.get("speed", 1.0))
                language = req.get("language", "auto")
                num_steps = int(req.get("num_steps", 32))
                ref_audio = req.get("ref_audio", "")
                ref_text = req.get("ref_text", "")
                instruct = req.get("instruct", "")

                # Redirect stdout to suppress model loading / tqdm output
                saved_stdout = sys.stdout
                sys.stdout = err
                try:
                    if ref_audio:
                        # Voice cloning mode
                        result = runner.generate_voice_clone(
                            text=text,
                            ref_audio=ref_audio,
                            ref_text=ref_text,
                        )
                    elif instruct:
                        # Voice design mode
                        result = runner.generate_voice_design(
                            text=text,
                            instruct=instruct,
                        )
                    else:
                        # Auto TTS mode
                        result = runner.generate(
                            text=text,
                            language=language,
                        )
                finally:
                    sys.stdout = saved_stdout

                audio = result["audio"]
                sample_rate = result.get("sample_rate", 24000)
                time_ms = result.get("time_ms", 0)
                vram_gb = result.get("peak_vram_gb", 0)

                # Convert float32 samples to int16
                arr = np.asarray(audio, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    raise ValueError("The engine returned no audio.")
                clipped = np.clip(arr, -1.0, 1.0) * 32767.0
                wav_b64 = base64.b64encode(
                    clipped.astype(np.int16).tobytes()
                ).decode("ascii")
                respond({
                    "ok": True,
                    "wav": wav_b64,
                    "sample_rate": sample_rate,
                    "time_ms": time_ms,
                    "peak_vram_gb": vram_gb,
                })
            except Exception as exc:  # noqa: BLE001
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        respond({"ok": False, "error": f"Unknown command: {cmd}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
