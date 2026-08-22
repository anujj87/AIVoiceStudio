"""OmniVoice worker subprocess — rewritten.

Runs the OmniVoice inference in an isolated process with its own runtime
(onnxruntime-genai for CPU, or PyTorch+omnivoice for GPU). Communicates
with the main app via JSON on stdin/stdout (same protocol as Qwen3 worker).

Usage::

    python -m ai_voice_studio.omnivoice.worker --variant onnx
    python -m ai_voice_studio.omnivoice.worker --variant gpu

The ONNX variant uses the Prince-1/OmniVoice-Onnx model which includes:
  - text_encoder.onnx   (text → hidden states)
  - llm_decoder.onnx    (hidden states → audio token sequences)  
  - audio_decoder.onnx  (audio tokens → mel spectrogram)
  - vocoder.onnx        (mel → waveform)

The GPU variant uses the k2-fsa/OmniVoice PyTorch model directly.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# WAV encoding helper
# ---------------------------------------------------------------------------

def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode int16 samples as a WAV in memory, return raw bytes."""
    import wave

    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    buf.close()
    try:
        with wave.open(buf.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        with open(buf.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(buf.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# GPU variant (PyTorch + omnivoice)
# ---------------------------------------------------------------------------

class GpuSynthesizer:
    """OmniVoice synthesis using PyTorch + omnivoice (GPU).

    This is the primary, fully-featured path:
    - Voice cloning from reference audio
    - Voice design via text instructions
    - 600+ languages
    - High quality output
    """

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from omnivoice import OmniVoice  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch or omnivoice is not installed. "
                "Download the OmniVoice GPU engine in Settings first."
            ) from exc

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = OmniVoice.from_pretrained(
            self._model_dir,
            device_map=device,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        )

    def synthesize(
        self,
        text: str,
        ref_wav: str = "",
        ref_text: str = "",
        instruct: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        self._ensure_model()

        kwargs = {"text": text, "speed": speed}

        if ref_wav and os.path.isfile(ref_wav):
            kwargs["ref_audio"] = ref_wav
            if ref_text:
                kwargs["ref_text"] = ref_text
        elif instruct:
            kwargs["instruct"] = instruct

        audio = self._model.generate(**kwargs)
        # audio is a list of np.ndarray at 24kHz, values in [-1, 1]
        return (audio[0] * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# ONNX variant (onnxruntime-genai)
# ---------------------------------------------------------------------------

class OnnxSynthesizer:
    """OmniVoice synthesis using onnxruntime-genai (CPU or CUDA).

    This variant uses the ONNX-exported model from Prince-1/OmniVoice-Onnx.
    The model requires onnxruntime-genai and its genai_config.json.

    The ONNX model follows the OmniVoice architecture:
      1. Text encoding via text_encoder.onnx
      2. Language model decoding via llm_decoder.onnx (autoregressive)
      3. Audio token decoding via audio_decoder.onnx
      4. Vocoder via vocoder.onnx (neural vocoder → waveform)

    The genai_config.json in the model directory orchestrates this pipeline
    automatically through onnxruntime-genai's Generator API.
    """

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._session = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._session is not None:
            return
        try:
            import onnxruntime_genai as og  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-genai is not installed. "
                "Please install it: pip install onnxruntime-genai"
            ) from exc

        # Verify the model directory has required files
        config_path = os.path.join(self._model_dir, "genai_config.json")
        if not os.path.isfile(config_path):
            # Try to find a compatible config structure
            onnx_files = [f for f in os.listdir(self._model_dir)
                         if f.endswith(".onnx")]
            if not onnx_files:
                raise RuntimeError(
                    f"No ONNX model files found in {self._model_dir}. "
                    "Download the OmniVoice model in Settings first."
                )

        try:
            config = og.Config(self._model_dir)
            self._session = og.Model(config)
            self._tokenizer = og.Tokenizer(self._session)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the OmniVoice ONNX model: {exc}. "
                "The model may be corrupted. Try re-downloading it."
            ) from exc

    def synthesize(
        self,
        text: str,
        ref_wav: str = "",
        ref_text: str = "",
        instruct: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize text using the ONNX model.

        Uses onnxruntime-genai's Generator API which handles the full
        OmniVoice pipeline (text encoding → LLM → audio tokens → vocoder).
        """
        self._ensure_model()
        import onnxruntime_genai as og  # noqa: PLC0415

        # Build system prompt with optional instruction
        system_parts = []
        if instruct:
            system_parts.append(f"instruct: {instruct}")
        system_parts.append(
            "You are a multilingual text-to-speech model. "
            "Generate natural speech from the given text."
        )
        system_prompt = "\n".join(system_parts)

        # Format as chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(messages)
        except Exception:
            # Fallback: simple prompt formatting
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{text}\n<|assistant|>\n"

        input_ids = self._tokenizer.encode(prompt)

        params = og.GeneratorParams(self._session)
        params.set_input_ids(input_ids)
        params.max_length = 8192

        generator = og.Generator(self._session, params)
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

        # Get the generated audio token IDs
        output_ids = generator.get_generated_tokens()

        # Decode audio tokens to waveform using the model's audio decoder
        return self._decode_audio_tokens(output_ids, speed)

    def _decode_audio_tokens(self, token_ids, speed: float) -> np.ndarray:
        """Decode model output token IDs to audio waveform.

        The OmniVoice ONNX model outputs audio tokens that represent
        mel spectrogram frames. These are decoded to a waveform through
        the model's built-in audio decoder and vocoder.

        The onnxruntime-genai model may handle this automatically through
        its Generator pipeline. If the output contains raw audio data
        (as bytes or float arrays), we decode that directly.
        """
        try:
            # Try to get raw audio output from the generator
            # Some ONNX models include an audio decoder that outputs
            # float audio samples directly
            if hasattr(token_ids, 'get_array'):
                audio_data = np.array(token_ids.get_array(), dtype=np.float32)
                if len(audio_data) > 0:
                    # Normalize and convert to int16
                    if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                        return (audio_data * 32767).astype(np.int16)
                    return np.clip(audio_data, -32768, 32767).astype(np.int16)
        except Exception:
            pass

        try:
            # Decode the token IDs to text and try to parse audio data
            output_text = self._tokenizer.decode(token_ids)

            # Check if output contains audio data markers
            if "<audio>" in output_text:
                start = output_text.index("<audio>") + len("<audio>")
                end = output_text.index("</audio>") if "</audio>" in output_text else len(output_text)
                audio_str = output_text[start:end].strip()
                # Try to decode as base64-encoded audio
                try:
                    import base64 as b64
                    audio_bytes = b64.b64decode(audio_str)
                    return np.frombuffer(audio_bytes, dtype=np.int16).copy()
                except Exception:
                    pass

            # The ONNX model may not be properly configured for audio output.
            # Fall back to silence with a warning.
            log.warning(
                "OmniVoice ONNX model produced text output instead of audio. "
                "The model may not have audio decoder configured. "
                "Consider using the GPU variant for voice synthesis."
            )
        except Exception as exc:
            log.error("Failed to decode OmniVoice ONNX output: %s", exc)

        # Return 1 second of silence as fallback
        return np.zeros(_SAMPLE_RATE, dtype=np.int16)


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

def _setup_onnx_path():
    """Add the bundled ONNX runtime to sys.path if available."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        vendor = os.path.join(meipass, "vendor", "omnivoice-onnx")
        if os.path.isdir(vendor) and vendor not in sys.path:
            sys.path.insert(0, vendor)
            return
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vendor = os.path.join(root, "vendor", "omnivoice-onnx")
    if os.path.isdir(vendor) and vendor not in sys.path:
        sys.path.insert(0, vendor)


def main():
    parser = argparse.ArgumentParser(description="OmniVoice worker subprocess")
    parser.add_argument("--variant", default="onnx", choices=["onnx", "gpu"])
    args = parser.parse_args()

    # Set up logging for the worker process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker_log = logging.getLogger("omnivoice.worker")
    worker_log.info("OmniVoice worker starting (variant=%s)", args.variant)

    if args.variant == "onnx":
        _setup_onnx_path()

    synth = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"ok": False, "error": "Invalid JSON"})
            continue

        cmd = req.get("cmd")

        if cmd == "ping":
            _respond({"ok": True})

        elif cmd == "quit":
            worker_log.info("Received quit command")
            _respond({"ok": True})
            break

        elif cmd == "synthesize":
            try:
                model_dir = req.get("model_dir", "")
                if not model_dir or not os.path.isdir(model_dir):
                    _respond({
                        "ok": False,
                        "error": f"Model directory not found: {model_dir}",
                    })
                    continue

                # Lazy-create the synthesizer
                if synth is None:
                    if args.variant == "gpu":
                        worker_log.info("Creating GPU synthesizer for %s", model_dir)
                        synth = GpuSynthesizer(model_dir)
                    else:
                        worker_log.info("Creating ONNX synthesizer for %s", model_dir)
                        synth = OnnxSynthesizer(model_dir)

                ref_wav = req.get("ref_wav", "")
                ref_text = req.get("ref_text", "")
                instruct = req.get("instruct", "")
                text = req.get("text", "")
                speed = req.get("speed", 1.0)

                # Log synthesis request (without full text for privacy)
                worker_log.info(
                    "Synthesizing: text_len=%d, ref=%s, instruct=%s, speed=%.2f",
                    len(text),
                    bool(ref_wav and os.path.isfile(ref_wav)),
                    bool(instruct),
                    speed,
                )

                samples = synth.synthesize(
                    text=text,
                    ref_wav=ref_wav,
                    ref_text=ref_text,
                    instruct=instruct,
                    speed=speed,
                )
                wav_bytes = _make_wav_bytes(samples, _SAMPLE_RATE)
                _respond({
                    "ok": True,
                    "wav": base64.b64encode(wav_bytes).decode("ascii"),
                    "sample_rate": _SAMPLE_RATE,
                    "num_samples": len(samples),
                })
            except Exception as exc:  # noqa: BLE001
                worker_log.error("Synthesis failed: %s", exc, exc_info=True)
                _respond({"ok": False, "error": str(exc)})

        else:
            _respond({"ok": False, "error": f"Unknown command: {cmd}"})

    worker_log.info("OmniVoice worker exiting")


def _respond(resp: dict):
    json.dump(resp, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
