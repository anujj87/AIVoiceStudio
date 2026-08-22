"""Qwen3-TTS worker subprocess.

Runs the Qwen3-TTS ONNX inference in an isolated process with its own runtime
(onnxruntime, librosa, soundfile, tokenizers).  Communicates with the main app
via JSON on stdin/stdout (same protocol as the OmniVoice worker).

Usage::

    python -m ai_voice_studio.qwen.worker

The worker uses ``Qwen3TTSONNXInference`` from ``tts_engine.py`` which
implements the full Qwen3-TTS inference pipeline:

  1. Text tokenization (BPE tokenizer)
  2. Speaker encoder (reference audio → x-vector)
  3. Speech tokenizer (reference audio → codec codes)
  4. Autoregressive talker (text → audio tokens)
  5. Code predictor (interleaved codebook prediction)
  6. Speech decoder (audio tokens → waveform)
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
# Qwen3 ONNX Synthesizer (wraps tts_engine.py)
# ---------------------------------------------------------------------------

class Qwen3Synthesizer:
    """Qwen3-TTS synthesis using pure ONNX inference.

    Loads the full Qwen3-TTS model pipeline from a model directory that
    contains:
      - fp16/ or onnx/ subdirectories with the ONNX model files
      - voice_clone_config.json / voice_design_config.json
      - tokenizer/ directory

    Supports voice cloning from a reference recording.
    """

    def __init__(self, model_dir: str, use_gpu: bool = False):
        self._model_dir = model_dir
        self._use_gpu = use_gpu
        self._engine = None
        self._speaker_embed_cache: dict[str, np.ndarray] = {}
        self._ref_codes_cache: dict[str, np.ndarray] = {}

    def _ensure_engine(self):
        if self._engine is not None:
            return

        try:
            from .tts_engine import Qwen3TTSONNXInference  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-TTS engine modules are not available. "
                "Ensure onnxruntime, librosa, soundfile, and tokenizers "
                "are installed in the worker's Python environment."
            ) from exc

        # Determine quantization: fp16 or onnx (fp32)
        fp16_dir = os.path.join(self._model_dir, "fp16")
        onnx_dir = os.path.join(self._model_dir, "onnx")
        quantize = "fp16" if os.path.isdir(fp16_dir) else None

        try:
            self._engine = Qwen3TTSONNXInference(
                dual_model_dir=self._model_dir,
                use_gpu=self._use_gpu,
                quantize=quantize,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the Qwen3-TTS model from {self._model_dir}: {exc}. "
                "The model may be corrupted. Try re-downloading it."
            ) from exc

    def synthesize(
        self,
        text: str,
        ref_wav: str = "",
        ref_text: str = "",
        language: str = "english",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech with voice cloning support.

        Parameters
        ----------
        text:
            The text to synthesize.
        ref_wav:
            Path to a reference WAV file for voice cloning. If empty, uses
            a default/predefined speaker.
        ref_text:
            Transcript of the reference audio (for ICL mode).
        language:
            Language identifier (e.g. "english", "chinese").
        speed:
            Playback speed multiplier.

        Returns
        -------
        (samples, sample_rate):
            int16 audio samples and the sample rate (24000).
        """
        self._ensure_engine()
        engine = self._engine

        sample_rate = engine.speaker_encoder_sr  # 24000
        speaker_embedding = None
        ref_codes = None
        ref_ids_content = None
        use_icl = False

        if ref_wav and os.path.isfile(ref_wav):
            try:
                import librosa  # noqa: PLC0415
                ref_audio, ref_sr = librosa.load(ref_wav, sr=None, mono=True)

                # Extract speaker embedding
                cache_key = os.path.abspath(ref_wav)
                if cache_key in self._speaker_embed_cache:
                    speaker_embedding = self._speaker_embed_cache[cache_key]
                else:
                    speaker_embedding = engine.extract_speaker_embedding(
                        ref_audio, ref_sr
                    )
                    self._speaker_embed_cache[cache_key] = speaker_embedding

                # Encode reference audio to codec codes (for ICL mode)
                if ref_text and engine.tokenizer is not None:
                    if cache_key in self._ref_codes_cache:
                        ref_codes = self._ref_codes_cache[cache_key]
                    else:
                        ref_codes = engine.encode_audio(ref_audio, ref_sr)
                        self._ref_codes_cache[cache_key] = ref_codes

                    # Encode reference text
                    ref_formatted = engine.build_ref_text(ref_text)
                    ref_ids = engine._encode_text(ref_formatted)
                    ref_ids_content = np.array(ref_ids[3:-2], dtype=np.int64)
                    use_icl = True

            except Exception as exc:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "Could not process reference audio %s: %s", ref_wav, exc
                )

        # Generate audio
        audio, _ = engine.generate(
            text=text,
            speaker_embedding=speaker_embedding,
            language=language,
            ref_text=ref_text if use_icl else None,
            ref_codes=ref_codes,
            ref_ids_content=ref_ids_content,
        )

        # Apply speed adjustment if needed
        if speed != 1.0 and speed > 0:
            import librosa  # noqa: PLC0415
            audio = librosa.effects.time_stretch(audio, rate=speed)

        # Convert float32 [-1, 1] to int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        return audio_int16, sample_rate


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS worker subprocess")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for inference if available")
    args = parser.parse_args()

    # Set up logging for the worker process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker_log = logging.getLogger("qwen.worker")
    worker_log.info("Qwen3-TTS worker starting (gpu=%s)", args.gpu)

    synth = None
    current_model_dir = None

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
            if synth is not None:
                try:
                    synth.close()
                except Exception:  # noqa: BLE001
                    pass
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

                # Re-create synthesizer if model dir changed
                if synth is None or current_model_dir != model_dir:
                    worker_log.info(
                        "Creating Qwen3 synthesizer for %s", model_dir
                    )
                    synth = Qwen3Synthesizer(
                        model_dir=model_dir,
                        use_gpu=args.gpu,
                    )
                    current_model_dir = model_dir

                ref_wav = req.get("ref_wav", "")
                ref_text = req.get("ref_text", "")
                text = req.get("text", "")
                language = req.get("language", "english")
                speed = req.get("speed", 1.0)

                # Log synthesis request (without full text for privacy)
                worker_log.info(
                    "Synthesizing: text_len=%d, ref=%s, lang=%s, speed=%.2f",
                    len(text),
                    bool(ref_wav and os.path.isfile(ref_wav)),
                    language,
                    speed,
                )

                samples, sample_rate = synth.synthesize(
                    text=text,
                    ref_wav=ref_wav,
                    ref_text=ref_text,
                    language=language,
                    speed=speed,
                )
                wav_bytes = _make_wav_bytes(samples, sample_rate)
                _respond({
                    "ok": True,
                    "wav": base64.b64encode(wav_bytes).decode("ascii"),
                    "sample_rate": sample_rate,
                    "num_samples": len(samples),
                })
            except Exception as exc:  # noqa: BLE001
                worker_log.error("Synthesis failed: %s", exc, exc_info=True)
                _respond({"ok": False, "error": str(exc)})

        else:
            _respond({"ok": False, "error": f"Unknown command: {cmd}"})

    worker_log.info("Qwen3-TTS worker exiting")


def _respond(resp: dict):
    json.dump(resp, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
