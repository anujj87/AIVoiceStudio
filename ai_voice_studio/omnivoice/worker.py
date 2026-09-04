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

Synthesize requests carry the full OmniVoice feature surface.  Which of the
generation knobs are actually applied depends on the installed runner:

* ``language``      -- optional language hint (``"auto"`` / ``""`` / ``null``
                       let the model detect it)
* ``num_step``      -- diffusion steps 1-64 (legacy alias ``num_steps``)
* ``guidance_scale`` -- CFG strength 0-10
* ``class_temperature`` -- token sampling temperature 0-2
* ``duration``      -- fixed output duration (if the runner supports it)
* ``seed``          -- RNG seed for reproducible output (if supported)
* ``speed``         -- speaking speed; applied natively when the runner
                       accepts it, otherwise approximated by resampling the
                       generated audio (reported in ``warnings``)
* ``ref_audio`` / ``ref_text`` -- voice cloning
* ``instruct``      -- voice design (natural-language description)

Any requested parameter the installed runner cannot accept is reported in the
response's ``skipped`` list instead of silently changing behaviour or
crashing the worker.
"""

from __future__ import annotations

import base64
import inspect
import json
import sys

import numpy as np


def _resample_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """Approximate ``speed`` by linear resampling (pitch changes slightly).

    Used only when the installed runner cannot apply ``speed`` natively.
    ``speed > 1`` makes the audio shorter / faster, ``speed < 1`` longer.
    """
    if speed <= 0.01:
        return samples
    n_out = max(1, int(round(len(samples) / speed)))
    if n_out == len(samples):
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16)


def _supported_kwargs(method, kwargs: dict) -> tuple[dict, list]:
    """Split kwargs into those the callable accepts and those it skips."""
    if not kwargs:
        return {}, []
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return dict(kwargs), []
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    supported = {}
    skipped = []
    for name, value in kwargs.items():
        if name in params:
            supported[name] = value
        else:
            skipped.append(name)
    return supported, skipped


def _language(value):
    """Normalise a language hint; auto/empty -> None (native auto-detect)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("auto", "automatic", "any"):
        return None
    return text


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
                language = _language(req.get("language"))
                num_step = req.get("num_step", req.get("num_steps"))
                instruct = req.get("instruct", "")
                ref_audio = req.get("ref_audio", "")
                ref_text = req.get("ref_text", "")

                # Optional advanced generation knobs (only sent when present).
                options = {}
                if language is not None:
                    options["language"] = language
                if num_step is not None:
                    options["num_step"] = int(num_step)
                if req.get("guidance_scale") is not None:
                    options["guidance_scale"] = float(req["guidance_scale"])
                if req.get("class_temperature") is not None:
                    options["class_temperature"] = float(req["class_temperature"])
                if req.get("duration") is not None:
                    options["duration"] = float(req["duration"])
                if req.get("seed") is not None:
                    options["seed"] = int(req["seed"])
                if speed != 1.0:
                    options["speed"] = speed

                # Choose the generation mode and its base arguments.
                if ref_audio:
                    method = getattr(runner, "generate_voice_clone", None)
                    base = {"text": text, "ref_audio": ref_audio}
                    if ref_text:
                        base["ref_text"] = ref_text
                    mode_name = "voice clone"
                elif instruct:
                    method = getattr(runner, "generate_voice_design", None)
                    base = {"text": text, "instruct": instruct}
                    mode_name = "voice design"
                else:
                    method = getattr(runner, "generate", None)
                    base = {"text": text}
                    mode_name = "auto voice"
                if method is None:
                    raise RuntimeError(
                        f"The installed OmniVoice runner has no {mode_name} "
                        "generation method."
                    )

                # Forward only the parameters this runner version accepts.
                supported, skipped = _supported_kwargs(
                    method, {**base, **options}
                )

                # Redirect stdout to suppress model loading / tqdm output
                saved_stdout = sys.stdout
                sys.stdout = err
                try:
                    result = method(**supported)
                finally:
                    sys.stdout = saved_stdout

                audio = result["audio"]
                sample_rate = result.get("sample_rate", 24000)
                peak_vram_gb = result.get("peak_vram_gb", 0)

                # Convert float32 samples to int16
                arr = np.asarray(audio, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    raise ValueError("The engine returned no audio.")
                samples = np.clip(arr, -1.0, 1.0) * 32767.0
                samples = samples.astype(np.int16)

                warnings = []
                if speed != 1.0 and "speed" not in supported:
                    # The runner cannot take speed natively: approximate it by
                    # resampling so the Recording window's rate slider still
                    # has an effect on OmniVoice-direct.
                    samples = _resample_speed(samples, speed)
                    warnings.append(
                        "speed applied approximately by resampling "
                        "(the installed runner has no native speed control)"
                    )
                if "duration" in skipped:
                    warnings.append(
                        "fixed duration is not supported by the installed runner; ignored"
                    )

                wav_b64 = base64.b64encode(samples.tobytes()).decode("ascii")
                time_s = result.get("time_s")
                if time_s is None:
                    time_ms = result.get("time_ms", 0)
                else:
                    time_ms = int(round(float(time_s) * 1000.0))
                respond({
                    "ok": True,
                    "wav": wav_b64,
                    "sample_rate": sample_rate,
                    "time_ms": time_ms,
                    "peak_vram_gb": peak_vram_gb,
                    "mode_used": mode_name,
                    "skipped": sorted(set(skipped)),
                    "warnings": warnings,
                })
            except Exception as exc:  # noqa: BLE001
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        respond({"ok": False, "error": f"Unknown command: {cmd}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
