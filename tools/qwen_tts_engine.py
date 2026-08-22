#!/usr/bin/env python3
"""
Qwen3-TTS ONNX Inference Engine

Pure ONNX inference for Qwen3-TTS. No PyTorch dependency required at runtime.
Supports Voice Clone (reference audio) and Voice Design (text description).

This module is the shared library used by:
  - generate_cache.py  (Step 1: export constant embedding cache)
  - create_speaker.py  (Step 2: create speaker profile)
  - synthesize.py      (Step 3: synthesize speech)

Copyright 2026 Alibaba Cloud (original Qwen3-TTS)
Licensed under the Apache License, Version 2.0
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Simple context-manager timer for measuring stage durations."""

    def __init__(self, name: str = "", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        if self.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled and self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            if self.name:
                print(f"  [Timer] {self.name}: {self.elapsed:.3f}s")

    @staticmethod
    def format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

try:
    import onnxruntime as ort
except ImportError:
    print("Error: pip install onnxruntime")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Error: pip install librosa soundfile")
    sys.exit(1)

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


# ---------------------------------------------------------------------------
# BPE Tokenizer (fallback when neither tokenizers nor tiktoken is available)
# ---------------------------------------------------------------------------

class Qwen2Tokenizer:
    """Simplified Qwen2 BPE tokenizer. Loads special tokens from tokenizer_config.json."""

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[str],
        special_tokens: Optional[Dict[str, int]] = None,
        backend: str = "regex",
    ):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.backend = backend
        self.special_tokens = special_tokens or {}

        for token, token_id in self.special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = token_id
            if token_id not in self.vocab_inv:
                self.vocab_inv[token_id] = token

        self.bpe_ranks = {}
        for i, merge in enumerate(merges):
            parts = merge.split()
            if len(parts) == 2:
                self.bpe_ranks[tuple(parts)] = i

        # Compile regex pattern (Qwen2 BPE pattern)
        self.pat = None
        self._use_regex = False
        try:
            import regex
            self.pat = regex.compile(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
            )
            self._use_regex = True
        except Exception:
            pass

        if self.pat is None:
            self.pat = re.compile(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
                re.UNICODE,
            )
            self._use_regex = False

    @classmethod
    def from_pretrained(cls, path: str) -> "Qwen2Tokenizer":
        """Load tokenizer from a directory containing vocab.json, merges.txt, tokenizer_config.json."""
        vocab_path = os.path.join(path, "vocab.json")
        merges_path = os.path.join(path, "merges.txt")
        config_path = os.path.join(path, "tokenizer_config.json")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json not found at {vocab_path}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"merges.txt not found at {merges_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        special_tokens = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for token_id_str, token_info in config.get("added_tokens_decoder", {}).items():
                token_content = token_info.get("content", "")
                if token_content:
                    special_tokens[token_content] = int(token_id_str)
            print(f"  Loaded {len(special_tokens)} special tokens from tokenizer_config.json")
        else:
            print(f"  Warning: tokenizer_config.json not found, using default special tokens")

        return cls(vocab, merges, special_tokens=special_tokens)

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe(self, token: str) -> List[str]:
        if len(token) <= 1:
            return [token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return [token]
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        return list(word)

    def _tokenize(self, text: str) -> List[str]:
        bpe_tokens: list = []
        for match in self.pat.finditer(text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            token_str = "".join(
                chr(b) if b < 256 and chr(b).isprintable() and chr(b) != " " else f"\\x{b:02x}"
                for b in token_bytes
            )
            bpe_tokens.extend(self._bpe(token_str))
        return bpe_tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Check for special tokens in the text
        special_pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
        if special_pattern and re.search(special_pattern, text):
            return self._encode_normal(text)
        tokens = self._tokenize(text)
        return [self.vocab.get(t, 0) for t in tokens]

    def _encode_normal(self, text: str) -> List[int]:
        special_pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
        if not special_pattern:
            return self.encode(text)
        ids: list = []
        for part in re.split(f"({special_pattern})", text):
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode(part))
        return ids

    def _byte_to_token(self, b: int) -> str:
        if b < 256 and chr(b).isprintable() and chr(b) != " ":
            return chr(b)
        return f"\\x{b:02x}"

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.vocab_inv.get(tid, "") for tid in token_ids]
        text = "".join(tokens)
        byte_pattern = re.compile(r"\\x([0-9a-fA-F]{2})")
        result: list = []
        i = 0
        while i < len(text):
            m = byte_pattern.match(text, i)
            if m:
                result.append(int(m.group(1), 16))
                i = m.end()
            else:
                result.extend(text[i].encode("utf-8"))
                i += 1
        return bytes(result).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# ONNX Session Wrapper
# ---------------------------------------------------------------------------

class ONNXInferenceSession:
    """ONNX model session wrapper with automatic FP16 <-> FP32 conversion."""

    def __init__(self, onnx_path: str, use_gpu: bool = False):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._input_dtypes: Dict[str, np.dtype] = {}
        for inp in self.session.get_inputs():
            if "float16" in inp.type:
                self._input_dtypes[inp.name] = np.float16
            elif "float" in inp.type:
                self._input_dtypes[inp.name] = np.float32
        self._has_fp16_inputs = any(dt == np.float16 for dt in self._input_dtypes.values())

    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if self._has_fp16_inputs:
            converted = {}
            for name, arr in inputs.items():
                expected = self._input_dtypes.get(name)
                converted[name] = arr.astype(expected) if expected is not None and arr.dtype != expected else arr
            outputs = self.session.run(self.output_names, converted)
        else:
            outputs = self.session.run(self.output_names, inputs)
        return [
            o.astype(np.float32) if isinstance(o, np.ndarray) and o.dtype == np.float16 else o
            for o in outputs
        ]


# ---------------------------------------------------------------------------
# Main Inference Class
# ---------------------------------------------------------------------------

class Qwen3TTSONNXInference:
    """
    Qwen3-TTS pure-ONNX inference engine (dual-model mode).

    Loads shared components (speaker encoder, speech tokenizer) plus two
    independent talker sets (voice_clone and voice_design) from the model
    directory.  Supports Voice Clone and Voice Design synthesis.
    """

    def __init__(
        self,
        dual_model_dir: str,
        use_gpu: bool = False,
        quantize: Optional[str] = None,
    ):
        self.use_gpu = use_gpu
        self.dual_mode = True
        self.dual_model_dir = dual_model_dir

        # Determine model subdirectory: "fp16" or "onnx" (FP32)
        if quantize == "fp16":
            model_subdir = "fp16"
        else:
            model_subdir = "onnx"

        self.onnx_dir = os.path.join(dual_model_dir, model_subdir, "voice_clone")
        self.onnx_shared_dir = os.path.join(dual_model_dir, model_subdir, "shared")
        self.onnx_vc_dir = os.path.join(dual_model_dir, model_subdir, "voice_clone")
        self.onnx_vd_dir = os.path.join(dual_model_dir, model_subdir, "voice_design")

        # Load voice_clone config as primary config
        config_path = os.path.join(dual_model_dir, "voice_clone_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Try loading voice_design config
        vd_config_path = os.path.join(dual_model_dir, "voice_design_config.json")
        if os.path.exists(vd_config_path):
            with open(vd_config_path, "r") as f:
                self.vd_config = json.load(f)
            print(f"  [OK] VoiceDesign config loaded")
        else:
            self.vd_config = None

        self.talker_config = self.config.get("talker_config", {})
        self.speaker_encoder_config = self.config.get("speaker_encoder_config", {})

        # Token IDs
        self.tts_bos_token_id = self.config.get("tts_bos_token_id", 151672)
        self.tts_eos_token_id = self.config.get("tts_eos_token_id", 151673)
        self.tts_pad_token_id = self.config.get("tts_pad_token_id", 151671)
        self.im_start_token_id = self.config.get("im_start_token_id", 151644)
        self.im_end_token_id = self.config.get("im_end_token_id", 151645)
        self.assistant_token_id = self.config.get("assistant_token_id", 77091)

        self.codec_bos_id = self.talker_config.get("codec_bos_id", 2149)
        self.codec_eos_id = self.talker_config.get("codec_eos_token_id", 2150)
        self.codec_pad_id = self.talker_config.get("codec_pad_id", 2148)
        self.codec_think_id = self.talker_config.get("codec_think_id", 2154)
        self.codec_nothink_id = self.talker_config.get("codec_nothink_id", 2155)
        self.codec_think_bos_id = self.talker_config.get("codec_think_bos_id", 2156)
        self.codec_think_eos_id = self.talker_config.get("codec_think_eos_id", 2157)
        self.codec_language_id = self.talker_config.get("codec_language_id", {})

        self.spk_id = self.talker_config.get("spk_id", {})
        self.spk_is_dialect = self.talker_config.get("spk_is_dialect", {})
        self.tts_model_type = self.config.get("tts_model_type", "base")

        # CustomVoice speaker support
        self.custom_voice_speakers = None
        if self.tts_model_type == "custom_voice":
            self.custom_voice_speakers = self.config.get("custom_voice_speakers", None)

        # Talker model parameters
        self.hidden_size = self.talker_config.get("hidden_size", 2048)
        self.num_layers = self.talker_config.get("num_hidden_layers", 28)
        self.num_kv_heads = self.talker_config.get("num_key_value_heads", 8)
        self.head_dim = self.talker_config.get("head_dim", 128)
        self.num_code_groups = self.talker_config.get("num_code_groups", 16)

        # Code Predictor parameters
        self.code_predictor_config = self.talker_config.get("code_predictor_config", {})
        self.cp_hidden_size = self.code_predictor_config.get("hidden_size", 1024)
        self.cp_num_layers = self.code_predictor_config.get("num_hidden_layers", 5)
        self.cp_num_kv_heads = self.code_predictor_config.get("num_key_value_heads", 8)
        self.cp_head_dim = self.code_predictor_config.get("head_dim", 128)

        # Speaker Encoder / Codec parameters
        self.speaker_encoder_sr = self.speaker_encoder_config.get("sample_rate", 24000)
        self.enc_dim = self.speaker_encoder_config.get("enc_dim", 2048)
        self.tokenizer_type = self.config.get("tokenizer_type", "12hz")
        self.codec_vocab_size = self.talker_config.get("vocab_size", 2048)
        self.codec_offset = self.config.get("codec_offset", 2)

        # Load tokenizer (auto-detect from dual_model_dir/tokenizer)
        tokenizer_path = os.path.join(dual_model_dir, "tokenizer")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = None

        self.tokenizer = None
        if tokenizer_path:
            # Prefer tokenizers (Rust) for speed
            tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
            if HAS_TOKENIZERS and os.path.exists(tokenizer_json):
                try:
                    from tokenizers import Tokenizer as RustTokenizer
                    self.tokenizer = RustTokenizer.from_file(tokenizer_json)
                    print(f"  [OK] tokenizers (Rust) loaded from {tokenizer_json}")
                except Exception as e:
                    print(f"  [--] tokenizers (Rust) failed: {e}")

            # Fallback to custom BPE tokenizer
            if self.tokenizer is None:
                try:
                    self.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
                    print(f"  [OK] custom tokenizer loaded from {tokenizer_path}")
                except Exception as e:
                    print(f"  [--] custom tokenizer not loaded: {e}")
        else:
            print(f"  [--] Tokenizer directory not found, ICL mode unavailable")

        # Load ONNX models
        self._load_models()

        # Initialize embedding cache
        self._cached_embeds: Dict[str, np.ndarray] = {}
        self._init_cached_embeds()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_talker_set(self, model_dir: str, label: str = "") -> dict:
        """Load a complete talker set (talker + code_predictor + embeddings) from a directory."""
        prefix = f"[{label}] " if label else ""
        result: dict = {}

        for name, filename in [
            ("text_embedding", "text_embedding.onnx"),
            ("codec_embedding", "codec_embedding.onnx"),
            ("talker", "talker_decode.onnx"),
        ]:
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                result[name] = ONNXInferenceSession(path, self.use_gpu)
                print(f"  {prefix}[OK] {name}")
            else:
                result[name] = None
                print(f"  {prefix}[--] {name} not found")

        # Code Predictor (prefer KV-cache version)
        path_kv = os.path.join(model_dir, "code_predictor_kv.onnx")
        path_no_kv = os.path.join(model_dir, "code_predictor.onnx")
        result["code_predictor_has_kv"] = False
        if os.path.exists(path_kv):
            result["code_predictor"] = ONNXInferenceSession(path_kv, self.use_gpu)
            result["code_predictor_has_kv"] = True
            print(f"  {prefix}[OK] code_predictor_kv")
        elif os.path.exists(path_no_kv):
            result["code_predictor"] = ONNXInferenceSession(path_no_kv, self.use_gpu)
            print(f"  {prefix}[OK] code_predictor")
        else:
            result["code_predictor"] = None
            print(f"  {prefix}[--] code_predictor not found")

        # Code Predictor Embeddings (g0 .. g{num_code_groups-1})
        result["code_predictor_embeddings"] = []
        for i in range(self.num_code_groups):
            path = os.path.join(model_dir, f"code_predictor_embed_g{i}.onnx")
            if os.path.exists(path):
                result["code_predictor_embeddings"].append(ONNXInferenceSession(path, self.use_gpu))
            else:
                result["code_predictor_embeddings"].append(None)
        loaded = sum(1 for p in result["code_predictor_embeddings"] if p is not None)
        if loaded > 0:
            print(f"  {prefix}[OK] code_predictor_embeddings: {loaded}/{self.num_code_groups}")
        else:
            print(f"  {prefix}[--] code_predictor_embeddings: not found")

        return result

    def _load_models(self):
        """Load all ONNX models from the dual-model directory."""
        print("Loading ONNX models...")

        # Shared: Speaker Encoder + Speech Tokenizer
        for name, filename in [
            ("speaker_encoder", "speaker_encoder.onnx"),
            ("speech_encoder", "speech_tokenizer_encoder.onnx"),
            ("speech_decoder", "speech_tokenizer_decoder.onnx"),
        ]:
            path = os.path.join(self.onnx_shared_dir, filename)
            if os.path.exists(path):
                setattr(self, name, ONNXInferenceSession(path, self.use_gpu))
                print(f"  [shared] [OK] {name}")
            else:
                setattr(self, name, None)
                print(f"  [shared] [--] {name} not found")

        # Voice Clone talker set
        print(f"\n  Loading voice_clone talker set...")
        self._vc_models = self._load_talker_set(self.onnx_vc_dir, "voice_clone")

        # Voice Design talker set
        print(f"\n  Loading voice_design talker set...")
        self._vd_models = self._load_talker_set(self.onnx_vd_dir, "voice_design")

        # Default to voice_clone as the active talker
        self._set_active_talker("voice_clone")

    def _set_active_talker(self, talker_name: str):
        """Switch the active talker model set ('voice_clone' or 'voice_design')."""
        if talker_name == "voice_clone":
            models = self._vc_models
        elif talker_name == "voice_design":
            models = self._vd_models
        else:
            raise ValueError(f"Unknown talker: {talker_name}")

        self.text_embedding = models["text_embedding"]
        self.codec_embedding = models["codec_embedding"]
        self.talker = models["talker"]
        self.code_predictor = models["code_predictor"]
        self.code_predictor_has_kv = models["code_predictor_has_kv"]
        self.code_predictor_embeddings = models["code_predictor_embeddings"]
        self._active_talker = talker_name

        # Recompute cached embeddings (different talkers have different weights)
        self._cached_embeds = {}
        self._init_cached_embeds()

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def _init_cached_embeds(self):
        """Load constant embeddings from model_cache.npz, or compute them on the fly."""
        active = getattr(self, "_active_talker", "voice_clone")
        cache_dir = self.onnx_vd_dir if active == "voice_design" else (self.onnx_vc_dir or self.onnx_dir)
        cache_path = os.path.join(cache_dir, "model_cache.npz")

        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                required = ["tts_bos_embed", "tts_eos_embed", "tts_pad_embed",
                            "codec_pad_bos_embed", "codec_bos_embed", "role_embed"]
                if all(k in data for k in required):
                    for k in data.files:
                        self._cached_embeds[k] = data[k]
                    print(f"  [OK] Loaded {len(self._cached_embeds)} cached embeddings from {cache_path}")
                    return
                else:
                    missing = [k for k in required if k not in data]
                    print(f"  [WARN] model_cache.npz missing keys: {missing}, will recompute")
            except Exception as e:
                print(f"  [WARN] Failed to load model_cache.npz: {e}, will recompute")

        if self.text_embedding is None or self.codec_embedding is None:
            print("  [--] text_embedding or codec_embedding not loaded, skipping embed cache")
            return

        print("Computing global constant embeddings (run 'generate_cache.py' to persist)...")
        self._compute_cached_embeds()
        print(f"  [OK] Computed {len(self._cached_embeds)} constant embeddings")

    def _compute_cached_embeds(self):
        """Compute all model-level constant embeddings via ONNX models."""
        # TTS special token embeddings
        tts_special_ids = np.array(
            [self.tts_bos_token_id, self.tts_eos_token_id, self.tts_pad_token_id], dtype=np.int64
        )
        tts_special_embeds = self.get_text_embedding(tts_special_ids)
        self._cached_embeds["tts_bos_embed"] = tts_special_embeds[:, 0:1, :]
        self._cached_embeds["tts_eos_embed"] = tts_special_embeds[:, 1:2, :]
        self._cached_embeds["tts_pad_embed"] = tts_special_embeds[:, 2:3, :]

        # codec_pad + codec_bos
        codec_pad_bos = np.array([self.codec_pad_id, self.codec_bos_id], dtype=np.int64)
        self._cached_embeds["codec_pad_bos_embed"] = self.get_codec_embedding(codec_pad_bos)
        self._cached_embeds["codec_bos_embed"] = self.get_codec_embedding(
            np.array([self.codec_bos_id], dtype=np.int64)
        )
        self._cached_embeds["codec_pad_embed"] = self.get_codec_embedding(
            np.array([self.codec_pad_id], dtype=np.int64)
        )

        # <|im_start|>assistant\n embedding
        if self.tokenizer:
            role_ids = self._encode_text("<|im_start|>assistant\n")
        else:
            role_ids = [self.im_start_token_id, self.assistant_token_id, 198]
        self._cached_embeds["role_embed"] = self.get_text_embedding(
            np.array(role_ids[:3], dtype=np.int64)
        )

        # Per-language codec prefix embeddings
        for lang_name, lid in self.codec_language_id.items():
            prefix_ids = np.array([
                self.codec_think_id, self.codec_think_bos_id, lid, self.codec_think_eos_id,
            ], dtype=np.int64)
            self._cached_embeds[f"codec_prefix_embed_{lang_name}"] = self.get_codec_embedding(prefix_ids)

        # Voice Clone tts_prefix (codec_len=7)
        self._cached_embeds["tts_prefix_embed_vc"] = np.concatenate([
            np.tile(self._cached_embeds["tts_pad_embed"], (1, 5, 1)),
            self._cached_embeds["tts_bos_embed"],
        ], axis=1)

        # Voice Design tts_prefix (codec_len=6, no spk)
        self._cached_embeds["tts_prefix_embed_vd"] = np.concatenate([
            np.tile(self._cached_embeds["tts_pad_embed"], (1, 4, 1)),
            self._cached_embeds["tts_bos_embed"],
        ], axis=1)

    def export_cached_embeds(self, output_path: Optional[str] = None) -> str:
        """Export constant embeddings to model_cache.npz."""
        if not self._cached_embeds:
            if self.text_embedding is None or self.codec_embedding is None:
                raise ValueError("text_embedding and codec_embedding are required to compute cache.")
            self._compute_cached_embeds()

        if output_path is None:
            active = getattr(self, "_active_talker", "voice_clone")
            if active == "voice_design" and self.onnx_vd_dir:
                output_path = os.path.join(self.onnx_vd_dir, "model_cache.npz")
            else:
                output_path = os.path.join(self.onnx_vc_dir or self.onnx_dir, "model_cache.npz")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(output_path, **self._cached_embeds)

        file_size = os.path.getsize(output_path)
        print(f"  Exported model cache to {output_path} ({file_size / 1024:.1f} KB)")
        for k, v in self._cached_embeds.items():
            print(f"    {k}: {v.shape} ({v.dtype})")
        return output_path

    # ------------------------------------------------------------------
    # Pre-computation helpers (for speaker profile)
    # ------------------------------------------------------------------

    def compute_ref_codec_embed_sum(self, ref_codes: np.ndarray) -> np.ndarray:
        """Pre-compute the sum of all codebook embeddings for ref_codes. [1, codes_len, hidden]"""
        first_code = ref_codes[:, 0:1].flatten()
        codec_embed_sum = self.get_codec_embedding(first_code)
        for i in range(1, min(ref_codes.shape[1], self.num_code_groups)):
            code_i = ref_codes[:, i:i + 1].flatten()
            if i < len(self.code_predictor_embeddings) and self.code_predictor_embeddings[i] is not None:
                embed_i = self.get_code_predictor_embedding(i, code_i)
            else:
                embed_i = self.get_codec_embedding(code_i)
            codec_embed_sum = codec_embed_sum + embed_i
        return codec_embed_sum

    def compute_ref_text_embed(self, ref_ids_content: np.ndarray) -> np.ndarray:
        """Pre-compute text embedding for the reference text. [1, ref_len, hidden]"""
        return self.get_text_embedding(ref_ids_content)

    def compute_ref_codec_embed(self, ref_codec_embed_sum: np.ndarray) -> np.ndarray:
        """Compute full ICL codec embedding: codec_bos + ref_codec_embed_sum. [1, codes_len+1, hidden]"""
        codec_bos_embed = self._cached_embeds.get("codec_bos_embed")
        if codec_bos_embed is None:
            codec_bos_embed = self.get_codec_embedding(np.array([self.codec_bos_id], dtype=np.int64))
        return np.concatenate([codec_bos_embed, ref_codec_embed_sum], axis=1)

    def compute_talker_input_embed_base(
        self, speaker_embedding: np.ndarray, language: str = "chinese",
    ) -> np.ndarray:
        """Pre-compute base talker input embedding (role + combined). [1, 9, hidden]"""
        lang_id = self.codec_language_id.get(language.lower(), self.codec_language_id.get("chinese", 2055))

        cache_key = f"codec_prefix_embed_{language.lower()}"
        codec_prefix_embed = self._cached_embeds.get(cache_key)
        if codec_prefix_embed is None:
            codec_prefix_ids = np.array([
                self.codec_think_id, self.codec_think_bos_id, lang_id, self.codec_think_eos_id,
            ], dtype=np.int64)
            codec_prefix_embed = self.get_codec_embedding(codec_prefix_ids)

        spk_embed = speaker_embedding.reshape(1, 1, -1)
        codec_pad_bos_embed = self._cached_embeds.get("codec_pad_bos_embed")
        if codec_pad_bos_embed is None:
            codec_pad_bos_embed = self.get_codec_embedding(
                np.array([self.codec_pad_id, self.codec_bos_id], dtype=np.int64)
            )

        codec_input_embed = np.concatenate([codec_prefix_embed, spk_embed, codec_pad_bos_embed], axis=1)

        tts_prefix_embed = self._cached_embeds.get("tts_prefix_embed_vc")
        if tts_prefix_embed is None:
            tts_bos = self._cached_embeds.get("tts_bos_embed", self.get_text_embedding(
                np.array([self.tts_bos_token_id], dtype=np.int64)))
            tts_pad = self._cached_embeds.get("tts_pad_embed", self.get_text_embedding(
                np.array([self.tts_pad_token_id], dtype=np.int64)))
            tts_prefix_embed = np.concatenate([np.tile(tts_pad, (1, 5, 1)), tts_bos], axis=1)

        combined_embed = tts_prefix_embed + codec_input_embed[:, :-1, :]
        role_embed = self._cached_embeds.get("role_embed")
        if role_embed is None:
            if self.tokenizer:
                role_ids = self._encode_text("<|im_start|>assistant\n")
            else:
                role_ids = [self.im_start_token_id, self.assistant_token_id, 198]
            role_embed = self.get_text_embedding(np.array(role_ids[:3], dtype=np.int64))

        return np.concatenate([role_embed, combined_embed], axis=1)

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def compute_mel_spectrogram(
        self, audio: np.ndarray, sr: int,
        n_mels: int = 128, n_fft: int = 1024, hop_length: int = 256,
        win_length: int = 1024, fmin: float = 0, fmax: float = 12000,
    ) -> np.ndarray:
        """Compute mel spectrogram (matching the original PyTorch implementation)."""
        if sr != self.speaker_encoder_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.speaker_encoder_sr)
        padding = (n_fft - hop_length) // 2
        audio_padded = np.pad(audio, (padding, padding), mode="reflect")
        stft = librosa.stft(y=audio_padded, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window="hann", center=False)
        magnitudes = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
        mel_basis = librosa.filters.mel(sr=self.speaker_encoder_sr, n_fft=n_fft,
                                        n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_spec = np.dot(mel_basis, magnitudes)
        mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        return mel_spec.T.astype(np.float32)

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract speaker embedding (x-vector) from audio. Returns [enc_dim]."""
        if self.speaker_encoder is None:
            raise ValueError("Speaker encoder not loaded")
        mel = self.compute_mel_spectrogram(audio, sr)
        mel = mel[np.newaxis, ...]
        outputs = self.speaker_encoder.run({"mel_spectrogram": mel})
        return outputs[0][0]

    def encode_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Encode audio to codec codes. Returns [seq_len, num_quantizers]."""
        if self.speech_encoder is None:
            raise ValueError("Speech encoder not loaded")
        target_sr = 24000
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        audio = audio.reshape(1, 1, -1).astype(np.float32)
        outputs = self.speech_encoder.run({"input_values": audio})
        return outputs[0][0].transpose(1, 0)

    def decode_audio(self, codes: np.ndarray, chunk_size: int = 300, left_context: int = 25) -> np.ndarray:
        """Decode codec codes to audio waveform."""
        if self.speech_decoder is None:
            raise ValueError("Speech decoder not loaded")
        seq_len = codes.shape[0]
        total_upsample = 1920  # 24000 Hz / 12.5 fps
        try:
            codes_input = codes.reshape(1, seq_len, -1).astype(np.int64)
            outputs = self.speech_decoder.run({"codes": codes_input})
            return outputs[0][0]
        except Exception as e:
            print(f"  [Vocoder] ERROR: {e}")
            return np.zeros(seq_len * total_upsample, dtype=np.float32)

    # ------------------------------------------------------------------
    # Tokenizer / Embedding helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs (supports multiple tokenizer backends)."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        result = self.tokenizer.encode(text)
        return result.ids if hasattr(result, "ids") else result

    def tokenize_text(self, text: str) -> np.ndarray:
        """Tokenize text to int64 numpy array."""
        return np.array(self._encode_text(text), dtype=np.int64)

    def build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def get_text_embedding(self, token_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings. Returns [1, seq_len, hidden_size]."""
        if self.text_embedding is None:
            raise ValueError("Text embedding not loaded")
        return self.text_embedding.run({"input_ids": token_ids.reshape(1, -1)})[0]

    def get_codec_embedding(self, codec_ids: np.ndarray) -> np.ndarray:
        """Get codec embeddings. Returns [1, seq_len, hidden_size]."""
        if self.codec_embedding is None:
            raise ValueError("Codec embedding not loaded")
        return self.codec_embedding.run({"codec_ids": codec_ids.reshape(1, -1)})[0]

    def get_code_predictor_embedding(self, group_idx: int, codec_ids: np.ndarray) -> np.ndarray:
        """Get code predictor embedding for a specific codebook group. Returns [1, seq_len, hidden]."""
        if group_idx >= len(self.code_predictor_embeddings) or self.code_predictor_embeddings[group_idx] is None:
            return self.get_codec_embedding(codec_ids)
        return self.code_predictor_embeddings[group_idx].run({"code_ids": codec_ids.reshape(1, -1)})[0]

    # ------------------------------------------------------------------
    # ICL prompt building
    # ------------------------------------------------------------------

    def generate_icl_prompt(
        self,
        text_ids: np.ndarray,
        ref_ids: np.ndarray,
        ref_codes: np.ndarray,
        tts_pad_embed: np.ndarray,
        tts_eos_embed: np.ndarray,
        non_streaming_mode: bool = False,
        ref_codec_embed_sum: Optional[np.ndarray] = None,
        ref_text_embed: Optional[np.ndarray] = None,
        ref_codec_embed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build ICL (in-context learning) prompt embeddings for voice clone."""
        # Text embedding
        if ref_text_embed is not None:
            text_embed = np.concatenate([ref_text_embed, self.get_text_embedding(text_ids)], axis=1)
        else:
            text_embed = self.get_text_embedding(np.concatenate([ref_ids, text_ids]))
        text_embed = np.concatenate([text_embed, tts_eos_embed], axis=1)

        # Codec embedding
        if ref_codec_embed is not None:
            codec_embed = ref_codec_embed
        else:
            if ref_codec_embed_sum is not None:
                embed_sum = ref_codec_embed_sum
            else:
                embed_sum = self.compute_ref_codec_embed_sum(ref_codes)
            codec_bos = self._cached_embeds.get("codec_bos_embed")
            if codec_bos is None:
                codec_bos = self.get_codec_embedding(np.array([self.codec_bos_id]))
            codec_embed = np.concatenate([codec_bos, embed_sum], axis=1)

        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]

        if non_streaming_mode:
            codec_pad_embed = self.get_codec_embedding(
                np.array([self.codec_pad_id] * text_lens, dtype=np.int64)
            )
            icl = text_embed + codec_pad_embed
            icl = np.concatenate([icl, codec_embed + np.tile(tts_pad_embed, (1, codec_lens, 1))], axis=1)
            return icl, tts_pad_embed
        else:
            if text_lens > codec_lens:
                icl = text_embed[:, :codec_lens] + codec_embed
                return icl, text_embed[:, codec_lens:]
            else:
                if codec_lens > text_lens:
                    text_embed = np.concatenate(
                        [text_embed, np.tile(tts_pad_embed, (1, codec_lens - text_lens, 1))], axis=1
                    )
                return text_embed + codec_embed, tts_pad_embed

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def init_kv_cache(self, batch_size: int = 1) -> np.ndarray:
        return np.zeros(
            (self.num_layers, 2, batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32
        )

    def sample_token(
        self, logits: np.ndarray, temperature: float = 0.9, top_k: int = 50,
        top_p: float = 1.0, suppress_tokens: list = None,
        generated_tokens: list = None, repetition_penalty: float = 1.0,
    ) -> int:
        """Sample a token from logits (pure numpy, no PyTorch dependency)."""
        logits = logits.astype(np.float64).copy()

        # Suppress tokens
        if suppress_tokens is not None:
            logits[suppress_tokens] = -np.inf

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for tid in set(generated_tokens):
                if 0 <= tid < len(logits):
                    if logits[tid] > 0:
                        logits[tid] /= repetition_penalty
                    else:
                        logits[tid] *= repetition_penalty

        # Temperature
        logits = logits / temperature

        # Top-K
        if top_k > 0:
            top_k = min(top_k, len(logits))
            threshold = np.partition(logits, -top_k)[-top_k]
            logits[logits < threshold] = -np.inf

        # Top-P (nucleus)
        if top_p < 1.0:
            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            shifted = sorted_logits - np.max(sorted_logits)
            probs = np.exp(shifted) / np.sum(np.exp(shifted))
            cum = np.cumsum(probs)
            mask = cum > top_p
            mask[1:] = mask[:-1].copy()
            mask[0] = False
            logits[sorted_idx[mask]] = -np.inf

        # Softmax + sample
        shifted = logits - np.max(logits)
        probs = np.exp(shifted)
        probs = probs / probs.sum()
        if np.isnan(probs).any() or probs.sum() == 0:
            valid = np.isfinite(logits)
            if valid.any():
                probs = valid.astype(np.float64)
                probs /= probs.sum()
            else:
                return 0
        return int(np.random.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # TTS generation (core autoregressive loop)
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        language: str = "chinese",
        ref_text: Optional[str] = None,
        ref_codes: Optional[np.ndarray] = None,
        instruct: Optional[str] = None,
        speaker_id: Optional[int] = None,
        x_vector_only: bool = False,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        seed: Optional[int] = None,
        ref_codec_embed_sum: Optional[np.ndarray] = None,
        ref_ids_content: Optional[np.ndarray] = None,
        ref_text_embed: Optional[np.ndarray] = None,
        ref_codec_embed: Optional[np.ndarray] = None,
        talker_input_embed_base: Optional[np.ndarray] = None,
        _voice_design_mode: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from text.

        Supports three speaker modes:
          1. speaker_id (CustomVoice): use a predefined speaker token ID
          2. x_vector_only: use only speaker embedding (simpler, lower quality)
          3. ICL (recommended): use ref_text + ref_codes + speaker_embedding

        Returns (audio_waveform, sample_rate).
        """
        if self.talker is None:
            raise ValueError("talker_decode.onnx not loaded, cannot generate")

        use_predefined_speaker = speaker_id is not None
        if not _voice_design_mode and not use_predefined_speaker and speaker_embedding is None:
            raise ValueError("Either speaker_embedding or speaker_id must be provided")

        # Determine ICL mode
        use_icl = not x_vector_only and not use_predefined_speaker and not _voice_design_mode
        if use_icl:
            if ref_text is None or ref_codes is None:
                print("Warning: ICL mode requires ref_text and ref_codes. Falling back to x_vector_only.")
                use_icl = False
            elif self.tokenizer is None:
                print("Warning: ICL mode requires tokenizer. Falling back to x_vector_only.")
                use_icl = False
            elif self.text_embedding is None or self.codec_embedding is None:
                print("Warning: ICL mode requires text/codec embedding models. Falling back to x_vector_only.")
                use_icl = False

        if not use_icl:
            if self.text_embedding is None or self.codec_embedding is None:
                missing = []
                if self.text_embedding is None:
                    missing.append("text_embedding.onnx")
                if self.codec_embedding is None:
                    missing.append("codec_embedding.onnx")
                if missing:
                    raise ValueError(f"Missing required models: {', '.join(missing)}")

        lang_id = self.codec_language_id.get(language.lower(), self.codec_language_id.get("chinese", 2055))

        if _voice_design_mode:
            mode_str = "voice_design"
        elif use_predefined_speaker:
            mode_str = "predefined_speaker"
        elif use_icl:
            mode_str = "ICL"
        else:
            mode_str = "x_vector_only"
        print(f"Generating: '{text}' | Language: {language} | Mode: {mode_str}")

        total_start = time.perf_counter()
        timings: dict = {}

        # === Build talker input embedding ===
        embed_start = time.perf_counter()

        if self._cached_embeds:
            tts_bos_embed = self._cached_embeds["tts_bos_embed"]
            tts_eos_embed = self._cached_embeds["tts_eos_embed"]
            tts_pad_embed = self._cached_embeds["tts_pad_embed"]
            codec_pad_bos_embed = self._cached_embeds["codec_pad_bos_embed"]
            role_embed = self._cached_embeds["role_embed"]
        else:
            tts_ids = np.array([self.tts_bos_token_id, self.tts_eos_token_id, self.tts_pad_token_id], dtype=np.int64)
            tts_embeds = self.get_text_embedding(tts_ids)
            tts_bos_embed = tts_embeds[:, 0:1, :]
            tts_eos_embed = tts_embeds[:, 1:2, :]
            tts_pad_embed = tts_embeds[:, 2:3, :]
            codec_pad_bos_embed = self.get_codec_embedding(
                np.array([self.codec_pad_id, self.codec_bos_id], dtype=np.int64))
            role_ids = (self._encode_text("<|im_start|>assistant\n") if self.tokenizer
                        else [self.im_start_token_id, self.assistant_token_id, 198])
            role_embed = self.get_text_embedding(np.array(role_ids[:3], dtype=np.int64))

        # Use pre-computed base embedding if available
        if talker_input_embed_base is not None and not _voice_design_mode and not use_predefined_speaker:
            talker_input_embed = talker_input_embed_base.copy()
            codec_input_embed_last = codec_pad_bos_embed[:, 1:2, :]
        else:
            cache_key = f"codec_prefix_embed_{language.lower()}"
            codec_prefix_embed = self._cached_embeds.get(cache_key)
            if codec_prefix_embed is None:
                codec_prefix_embed = self.get_codec_embedding(np.array([
                    self.codec_think_id, self.codec_think_bos_id, lang_id, self.codec_think_eos_id,
                ], dtype=np.int64))

            if _voice_design_mode:
                codec_input_embed = np.concatenate([codec_prefix_embed, codec_pad_bos_embed], axis=1)
            elif use_predefined_speaker:
                spk_embed = self.get_codec_embedding(np.array([speaker_id], dtype=np.int64))
                codec_input_embed = np.concatenate([codec_prefix_embed, spk_embed, codec_pad_bos_embed], axis=1)
            else:
                spk_embed = speaker_embedding.reshape(1, 1, -1)
                codec_input_embed = np.concatenate([codec_prefix_embed, spk_embed, codec_pad_bos_embed], axis=1)

            codec_input_embed_last = codec_input_embed[:, -1:, :]
            codec_len = codec_input_embed.shape[1]

            if _voice_design_mode:
                tts_prefix = self._cached_embeds.get("tts_prefix_embed_vd")
            else:
                tts_prefix = self._cached_embeds.get("tts_prefix_embed_vc")
            if tts_prefix is None:
                tts_prefix = np.concatenate([
                    np.tile(tts_pad_embed, (1, codec_len - 2, 1)), tts_bos_embed
                ], axis=1)

            combined_embed = tts_prefix + codec_input_embed[:, :-1, :]
            talker_input_embed = np.concatenate([role_embed, combined_embed], axis=1)

        # === Instruction embedding ===
        if instruct and instruct.strip():
            if self.tokenizer is None:
                print("  [WARN] Instruct requires tokenizer")
            else:
                instruct_ids = np.array(
                    self._encode_text(self.build_instruct_text(instruct.strip())), dtype=np.int64
                )
                instruct_embed = self.get_text_embedding(instruct_ids)
                talker_input_embed = np.concatenate([instruct_embed, talker_input_embed], axis=1)

        timings["embed_base"] = time.perf_counter() - embed_start

        # === ICL prompt ===
        trailing_text_hidden = None
        if use_icl:
            icl_start = time.perf_counter()
            if ref_ids_content is None:
                ref_formatted = self.build_ref_text(ref_text)
                ref_ids = self._encode_text(ref_formatted)
                ref_ids_content = np.array(ref_ids[3:-2], dtype=np.int64)

            text_formatted = self.build_assistant_text(text)
            text_ids = self._encode_text(text_formatted)
            text_ids_content = np.array(text_ids[3:-5], dtype=np.int64)

            icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                text_ids=text_ids_content, ref_ids=ref_ids_content, ref_codes=ref_codes,
                tts_pad_embed=tts_pad_embed, tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
                ref_codec_embed_sum=ref_codec_embed_sum,
                ref_text_embed=ref_text_embed, ref_codec_embed=ref_codec_embed,
            )
            talker_input_embed = np.concatenate([talker_input_embed, icl_input_embed], axis=1)
            timings["icl_prompt"] = time.perf_counter() - icl_start
        else:
            # Non-ICL: add first text token
            if self.tokenizer:
                text_formatted = self.build_assistant_text(text)
                text_ids = self._encode_text(text_formatted)
                first_text_embed = self.get_text_embedding(np.array([text_ids[3]], dtype=np.int64))
                first_combined = first_text_embed + codec_input_embed_last
                talker_input_embed = np.concatenate([talker_input_embed, first_combined], axis=1)
                if len(text_ids) > 9:
                    remaining_ids = np.array(text_ids[4:-5], dtype=np.int64)
                    trailing_text_hidden = np.concatenate(
                        [self.get_text_embedding(remaining_ids), tts_eos_embed], axis=1
                    )
                else:
                    trailing_text_hidden = tts_eos_embed

        # === Autoregressive decode ===
        kv_cache = self.init_kv_cache(1)
        generated_codes: list = []

        # Prefill
        seq_len = talker_input_embed.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.float32)
        position_ids = np.stack([np.arange(seq_len, dtype=np.int64)] * 3, axis=0)[:, np.newaxis, :]

        has_hidden_states = "hidden_states" in self.talker.output_names

        prefill_start = time.perf_counter()
        talker_outputs = self.talker.run({
            "inputs_embeds": talker_input_embed.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
        })

        hidden_states = None
        if len(talker_outputs) >= 3:
            logits, kv_cache, hidden_states = talker_outputs[0], talker_outputs[1], talker_outputs[2]
        elif len(talker_outputs) >= 2:
            logits, kv_cache = talker_outputs[0], talker_outputs[1]
        else:
            logits = talker_outputs[0]
            kv_cache = self.init_kv_cache(1)

        timings["prefill"] = time.perf_counter() - prefill_start

        trailing_idx = 0
        trailing_len = trailing_text_hidden.shape[1] if trailing_text_hidden is not None else 0

        if seed is not None:
            np.random.seed(seed)

        # Build suppress_tokens list (special tokens except EOS)
        talker_vocab_size = self.talker_config.get("vocab_size", 3072)
        base_suppress = [i for i in range(talker_vocab_size - 1024, talker_vocab_size) if i != self.codec_eos_id]
        all_generated_codes: list = []

        decode_start = time.perf_counter()
        for step in range(max_new_tokens):
            suppress = base_suppress + ([self.codec_eos_id] if step < min_new_tokens else [])

            first_code = self.sample_token(
                logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                suppress_tokens=suppress, generated_tokens=all_generated_codes,
                repetition_penalty=repetition_penalty,
            )

            if first_code == self.codec_eos_id:
                break

            # Generate remaining codebook codes via code predictor
            step_codes = [first_code]
            if hidden_states is not None and self.code_predictor is not None and len(self.code_predictor_embeddings) > 0:
                talker_hidden = hidden_states[:, -1:, :]
                first_code_embed = self.get_codec_embedding(np.array([first_code], dtype=np.int64))

                if self.code_predictor_has_kv:
                    prefill_embed = np.concatenate([talker_hidden, first_code_embed], axis=1).astype(np.float32)
                    cp_kv = np.zeros(
                        (self.cp_num_layers, 2, 1, self.cp_num_kv_heads, 0, self.cp_head_dim), dtype=np.float32
                    )
                    cp_out = self.code_predictor.run({
                        "inputs_embeds": prefill_embed, "past_key_values": cp_kv,
                        "generation_step": np.array([0], dtype=np.int64),
                    })
                    pred_logits, cp_kv = cp_out[0], cp_out[1]
                    if pred_logits.ndim == 3:
                        pred_logits = pred_logits[:, -1, :]
                    pred_code = self.sample_token(pred_logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
                    step_codes.append(pred_code)

                    for g in range(1, self.num_code_groups - 1):
                        sub_embed = self.get_code_predictor_embedding(g, np.array([pred_code], dtype=np.int64))
                        cp_out = self.code_predictor.run({
                            "inputs_embeds": sub_embed.astype(np.float32), "past_key_values": cp_kv,
                            "generation_step": np.array([g], dtype=np.int64),
                        })
                        pred_logits, cp_kv = cp_out[0], cp_out[1]
                        if pred_logits.ndim == 3:
                            pred_logits = pred_logits[:, -1, :]
                        pred_code = self.sample_token(pred_logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
                        step_codes.append(pred_code)
                else:
                    # No KV cache: recompute full sequence each step
                    embed_seq = [talker_hidden.astype(np.float32), first_code_embed.astype(np.float32)]
                    for g in range(self.num_code_groups - 1):
                        inputs_embed = np.concatenate(embed_seq, axis=1)
                        pred_logits = self.code_predictor.run({
                            "inputs_embeds": inputs_embed.astype(np.float32),
                            "generation_step": np.array([g], dtype=np.int64),
                        })[0]
                        pred_code = self.sample_token(pred_logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
                        step_codes.append(pred_code)
                        idx = g + 1
                        if idx < len(self.code_predictor_embeddings) and self.code_predictor_embeddings[idx] is not None:
                            sub = self.code_predictor_embeddings[idx].run(
                                {"code_ids": np.array([[pred_code]], dtype=np.int64)}
                            )[0]
                        else:
                            sub = self.get_codec_embedding(np.array([pred_code], dtype=np.int64))
                        embed_seq.append(sub.astype(np.float32))
            else:
                # Fallback: sample from ref_codes if available
                for g in range(1, self.num_code_groups):
                    if ref_codes is not None and ref_codes.shape[0] > 0:
                        step_codes.append(int(ref_codes[np.random.randint(0, ref_codes.shape[0]), g]))
                    else:
                        step_codes.append(0)

            generated_codes.append(step_codes)
            all_generated_codes.append(first_code)

            # Prepare next-step input embedding
            next_codec_embed = self.get_codec_embedding(np.array([step_codes[0]], dtype=np.int64))
            for g in range(1, len(step_codes)):
                if g < len(self.code_predictor_embeddings) and self.code_predictor_embeddings[g] is not None:
                    embed_g = self.get_code_predictor_embedding(g, np.array([step_codes[g]], dtype=np.int64))
                else:
                    embed_g = self.get_codec_embedding(np.array([step_codes[g]], dtype=np.int64))
                next_codec_embed = next_codec_embed + embed_g

            # Add trailing text (streaming mode)
            if trailing_idx < trailing_len:
                next_text_embed = trailing_text_hidden[:, trailing_idx:trailing_idx + 1, :]
                trailing_idx += 1
            else:
                next_text_embed = tts_pad_embed

            next_embed = next_text_embed + next_codec_embed
            curr_len = kv_cache.shape[4] + 1
            attention_mask = np.ones((1, curr_len), dtype=np.float32)
            position_ids = np.array([[[curr_len - 1]]] * 3, dtype=np.int64)

            talker_outputs = self.talker.run({
                "inputs_embeds": next_embed.astype(np.float32),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": kv_cache,
            })

            if len(talker_outputs) >= 3:
                logits, kv_cache, hidden_states = talker_outputs[0], talker_outputs[1], talker_outputs[2]
            elif len(talker_outputs) >= 2:
                logits, kv_cache = talker_outputs[0], talker_outputs[1]
                hidden_states = None
            else:
                logits = talker_outputs[0]
                hidden_states = None

            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - decode_start
                print(f"  Generated {step + 1} tokens... ({(step + 1) / elapsed:.1f} tokens/s)")

        timings["decode"] = time.perf_counter() - decode_start
        num_generated = len(generated_codes)

        if num_generated == 0:
            print("Warning: No codes generated")
            return np.array([]), 24000

        # Decode audio
        codes = np.array(generated_codes, dtype=np.int64)
        print(f"  Generated {num_generated} code frames")

        ref_codes_len = 0
        if use_icl and ref_codes is not None:
            ref_codes_len = ref_codes.shape[0]
            codes = np.concatenate([ref_codes, codes], axis=0)

        if self.speech_decoder is not None:
            vocoder_start = time.perf_counter()
            audio = self.decode_audio(codes)
            timings["vocoder"] = time.perf_counter() - vocoder_start

            # Trim reference audio portion for ICL mode
            if ref_codes_len > 0:
                cut_samples = int(ref_codes_len / len(codes) * len(audio))
                audio = audio[cut_samples:]

            total_time = time.perf_counter() - total_start
            audio_duration = len(audio) / 24000
            rtf = total_time / audio_duration if audio_duration > 0 else 0

            print(f"  Timing: prefill={Timer.format_duration(timings.get('prefill', 0))}, "
                  f"decode={Timer.format_duration(timings.get('decode', 0))}, "
                  f"vocoder={Timer.format_duration(timings.get('vocoder', 0))}, "
                  f"total={Timer.format_duration(total_time)}")
            print(f"  Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x")

            return audio, 24000
        else:
            print("Warning: Speech decoder not loaded, returning codes")
            return codes, 0

    # ------------------------------------------------------------------
    # Voice Design
    # ------------------------------------------------------------------

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "auto",
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech via Voice Design mode (no reference audio needed).

        Uses a natural-language instruct description to control the voice timbre.
        Automatically switches to the voice_design talker and switches back afterwards.
        """
        if not instruct or not instruct.strip():
            raise ValueError("instruct is required for voice_design mode")

        prev_talker = getattr(self, "_active_talker", "voice_clone")
        if prev_talker != "voice_design":
            print("  Switching to voice_design talker...")
            self._set_active_talker("voice_design")

        try:
            audio, sr = self.generate(
                text=text, speaker_embedding=None, speaker_id=None,
                language=language, ref_text=None, ref_codes=None,
                instruct=instruct, x_vector_only=False,
                non_streaming_mode=non_streaming_mode,
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, seed=seed,
                _voice_design_mode=True,
            )
            return audio, sr
        finally:
            if prev_talker != "voice_design":
                self._set_active_talker(prev_talker)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """Load audio file. Returns (waveform, sample_rate)."""
    audio, file_sr = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32), file_sr

