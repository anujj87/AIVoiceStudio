"""Installed-model state.

State lives in ``%APPDATA%/AIVoiceStudio/models.json`` so it survives restarts
and can be inspected by the GUI (Download/Remove, Available TTS, Voice clone).

Layout::

    {
      "variants": {
        "piper": {"en_US": {"medium": {"dir": "...", "complete": true}}}
      },
      "custom_voices": [
        {"name": "my_voice", "tts": "piper", "dir": "...", "kind": "tarball"},
        {"name": "narrator", "tts": "omnivoice", "dir": "...",
         "kind": "omni_design", "engine": "omnivoice", "instruct": "male"}
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from typing import Any, Dict, List, Optional

from .. import paths
from ..util import sanitize_filename
from . import catalog

log = logging.getLogger(__name__)

# A "complete" marker written after all artifacts of a variant are extracted.
_MARKER = ".aivs_complete"


class ModelStore:
    def __init__(self, state_file: str | None = None):
        self._file = state_file or paths.models_state_file()
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {"variants": {}, "custom_voices": []}
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self._file, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                self._data.update(loaded)
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read model state %s: %s", self._file, exc)

    def save(self) -> None:
        with self._lock:
            try:
                with open(self._file, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, ensure_ascii=False)
            except OSError as exc:
                log.error("Could not save model state %s: %s", self._file, exc)

    # -- directories --------------------------------------------------------
    def variant_dir(self, tts_id: str, lang_code: str, variant_id: str) -> str:
        safe = [sanitize_filename(x, 40) for x in (tts_id, lang_code, variant_id)]
        d = os.path.join(paths.models_dir(), *safe)
        os.makedirs(d, exist_ok=True)
        return d

    def custom_dir(self, name: str) -> str:
        d = os.path.join(paths.models_dir(), "custom", sanitize_filename(name, 60))
        os.makedirs(d, exist_ok=True)
        return d

    # -- variant status -----------------------------------------------------
    def is_variant_installed(self, tts_id: str, lang_code: str, variant_id: str) -> bool:
        variant = self._data["variants"].get(tts_id, {}).get(lang_code, {}).get(variant_id)
        if not variant:
            return False
        if not variant.get("complete"):
            return False
        return os.path.isdir(variant["dir"])

    def variant_info(self, tts_id: str, lang_code: str, variant_id: str) -> Optional[Dict[str, Any]]:
        return self._data["variants"].get(tts_id, {}).get(lang_code, {}).get(variant_id)

    def mark_installed(self, tts_id: str, lang_code: str, variant_id: str, directory: str) -> None:
        with self._lock:
            self._data["variants"].setdefault(tts_id, {}).setdefault(lang_code, {})[variant_id] = {
                "dir": directory,
                "complete": True,
            }
        marker = os.path.join(directory, _MARKER)
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("complete")
        except OSError:
            pass
        self.save()

    def remove_variant(self, tts_id: str, lang_code: str, variant_id: str) -> bool:
        info = self.variant_info(tts_id, lang_code, variant_id)
        with self._lock:
            if tts_id in self._data["variants"] and lang_code in self._data["variants"][tts_id]:
                self._data["variants"][tts_id][lang_code].pop(variant_id, None)
        removed = False
        if info and os.path.isdir(info["dir"]):
            shutil.rmtree(info["dir"], ignore_errors=True)
            removed = True
        self.save()
        return removed

    def installed_variants(self) -> List[Dict[str, str]]:
        result = []
        for tts_id, langs in self._data["variants"].items():
            for lang_code, variants in langs.items():
                for variant_id in variants:
                    if self.is_variant_installed(tts_id, lang_code, variant_id):
                        result.append(
                            {"tts": tts_id, "language": lang_code, "variant": variant_id}
                        )
        return result

    def installed_voices(self) -> List[Dict[str, Any]]:
        """Flatten installed variants into per-voice entries usable by the UI.

        Piper/vits variants list their catalog voices; kokoro variants expose
        the voices stored inside the single model artifact.
        """
        voices = []
        for entry in self.installed_variants():
            tts = catalog.find_tts(entry["tts"])
            if not tts:
                continue
            variant = catalog.find_variant(tts, entry["language"], entry["variant"])
            if not variant:
                continue
            engine = tts.get("engine")
            # Prefer the directory recorded in the state file: it stays valid
            # even after the model location was changed in Settings.
            info = self.variant_info(entry["tts"], entry["language"], entry["variant"])
            if info and info.get("dir"):
                variant_dir = info["dir"]
            else:
                variant_dir = self.variant_dir(entry["tts"], entry["language"], entry["variant"])
            if engine == "pocket":
                # Pocket TTS (sherpa-onnx): each voice uses a reference WAV
                # stored in the variant folder (test_wavs/<voice>.wav).
                for voice in variant.get("voices", []):
                    voices.append(
                        {
                            "tts": entry["tts"],
                            "tts_name": tts["name"],
                            "language": entry["language"],
                            "variant": entry["variant"],
                            "voice": voice["id"],
                            "voice_name": voice.get("name", voice["id"]),
                            "sid": 0,
                            "engine": engine,
                            "dir": variant_dir,
                            "reference": os.path.join(
                                variant_dir, voice.get("reference", "")
                            ),
                        }
                    )
                continue
            if engine in ("kokoro", "kitten"):
                for voice in variant.get("voices", []):
                    voices.append(
                        {
                            "tts": entry["tts"],
                            "tts_name": tts["name"],
                            "language": entry["language"],
                            "variant": entry["variant"],
                            "voice": voice["id"],
                            "voice_name": voice.get("name", voice["id"]),
                            "sid": voice.get("sid", 0),
                            "engine": engine,
                            "dir": variant_dir,
                        }
                    )
            elif engine == "omnivoice":
                # OmniVoice: GPU-required, no local model files to scan.
                # The model is downloaded on first use by the worker.
                for voice in variant.get("voices", []):
                    voices.append(
                        {
                            "tts": entry["tts"],
                            "tts_name": tts["name"],
                            "language": entry["language"],
                            "variant": entry["variant"],
                            "voice": voice["id"],
                            "voice_name": voice.get("name", voice["id"]),
                            "sid": voice.get("sid", 0),
                            "engine": engine,
                            "dir": "",
                            "requires_gpu": True,
                            "requires_package": "omnivoice-triton",
                        }
                    )
            elif engine == "omnivoice_server":
                # OmniVoice Server: HTTP API, GPU-required, no local files.
                for voice in variant.get("voices", []):
                    voices.append(
                        {
                            "tts": entry["tts"],
                            "tts_name": tts["name"],
                            "language": entry["language"],
                            "variant": entry["variant"],
                            "voice": voice["id"],
                            "voice_name": voice.get("name", voice["id"]),
                            "sid": voice.get("sid", 0),
                            "engine": engine,
                            "dir": "",
                            "requires_gpu": True,
                            "requires_package": "omnivoice-server",
                        }
                    )
            else:
                for voice in variant.get("voices", []):
                    voices.append(
                        {
                            "tts": entry["tts"],
                            "tts_name": tts["name"],
                            "language": entry["language"],
                            "variant": entry["variant"],
                            "voice": voice["id"],
                            "voice_name": voice.get("name", voice["id"]),
                            "sid": voice.get("sid", 0),
                            "engine": engine,
                            "dir": os.path.join(variant_dir, voice["id"]),
                            "artifact": voice.get("artifact"),
                            "frontend": tts.get("frontend", ""),
                        }
                    )
        return voices

    # -- custom (user-created) voices ---------------------------------------
    def add_custom_voice(self, name: str, tts_id: str, directory: str, kind: str,
                         extra: Dict[str, Any] | None = None) -> None:
        """Register a custom (user-created) voice.

        ``kind`` is a sherpa voice kind (``tarball``/``onnx``) for standalone
        model folders, or an OmniVoice voice-library kind
        (``omni_clone``/``omni_design``) whose engine, reference sample and
        design instructions travel in ``extra``.
        """
        entry: Dict[str, Any] = {
            "name": name,
            "tts": tts_id,
            "dir": directory,
            "kind": kind,
            "engine": "vits",
        }
        if extra:
            entry.update(extra)
        with self._lock:
            self._data["custom_voices"] = [
                v for v in self._data["custom_voices"] if v["name"] != name
            ]
            self._data["custom_voices"].append(entry)
        self.save()

    def rename_custom_voice(self, name: str, new_name: str) -> bool:
        """Rename a custom voice in place (the voice folder keeps its name
        because it is keyed by an id, not by the voice name).

        Returns ``True`` on success; ``False`` when ``name`` is not found or
        ``new_name`` collides with an existing custom voice.
        """
        with self._lock:
            voices = self._data.get("custom_voices", [])
            if any(v["name"] == new_name for v in voices):
                return False
            for v in voices:
                if v["name"] == name:
                    v["name"] = new_name
                    self.save()
                    return True
        return False

    def custom_voices(self, tts_id: str | None = None) -> List[Dict[str, Any]]:
        voices = self._data.get("custom_voices", [])
        if tts_id:
            voices = [v for v in voices if v.get("tts") == tts_id]
        return voices

    def remove_custom_voice(self, name: str) -> bool:
        voice = next((v for v in self._data.get("custom_voices", []) if v["name"] == name), None)
        with self._lock:
            self._data["custom_voices"] = [
                v for v in self._data["custom_voices"] if v["name"] != name
            ]
        removed = False
        if voice and os.path.isdir(voice["dir"]):
            # Pocket TTS clones point at the shared downloaded model folder
            # (only the reference WAV is private to the voice); never delete
            # the shared model files when removing one cloned voice.
            if voice.get("kind") == "pocket_reference":
                reference = voice.get("reference") or voice.get("sample")
                if reference and os.path.isfile(reference):
                    try:
                        os.remove(reference)
                    except OSError:
                        pass
                    removed = True
            else:
                shutil.rmtree(voice["dir"], ignore_errors=True)
                removed = True
        self.save()
        return removed


def resolve_voice_files(voice_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Map a voice entry (from installed_voices or custom voices) to files the
    engine needs: model path, tokens path, optional espeak-ng-data dir,
    optional kokoro voices file and sid."""
    directory = voice_entry["dir"]
    engine = voice_entry.get("engine", "vits")
    files: Dict[str, Any] = {"engine": engine, "dir": directory}

    # OmniVoice has no local model files; the model is downloaded on first use
    # by the worker subprocess via omnivoice-triton / HuggingFace.
    if engine == "omnivoice":
        files.update({
            "model": None,
            "tokens": None,
            "sid": voice_entry.get("sid", 0),
        })
        return files

    # OmniVoice Server: HTTP API, no local model files.
    if engine == "omnivoice_server":
        files.update({
            "model": None,
            "tokens": None,
            "sid": voice_entry.get("sid", 0),
        })
        return files

    def find(*names: str) -> Optional[str]:
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        return None

    if engine == "pocket":
        # Pocket TTS (sherpa-onnx): six ONNX parts + two JSON tables + a
        # reference (voice) WAV per voice.
        files.update(
            {
                "decoder": find("decoder.onnx", "decoder.int8.onnx"),
                "encoder": find("encoder.onnx"),
                "lm_flow": find("lm_flow.onnx", "lm_flow.int8.onnx"),
                "lm_main": find("lm_main.onnx", "lm_main.int8.onnx"),
                "text_conditioner": find("text_conditioner.onnx"),
                "vocab_json": find("vocab.json"),
                "token_scores_json": find("token_scores.json"),
                "reference": voice_entry.get("reference", ""),
                "sid": voice_entry.get("sid", 0),
            }
        )
    elif engine in ("kokoro", "kitten"):
        if engine == "kitten":
            # The three Kitten v0.8 sizes use different file names in the
            # sherpa-onnx release: micro is ``model.onnx``, nano-int8 is
            # ``model.int8.onnx`` and nano-fp32 is ``model.fp32.onnx``.
            # Looking only for model.onnx/model.int8.onnx made the fp32
            # voice report "Model files are incomplete (missing model)"
            # right after a successful download.
            model = (
                find("model.onnx", "model.fp32.onnx", "model.int8.onnx")
                or _find_onnx(directory)
            )
        else:
            model = find("model.onnx", "model.int8.onnx")
        tokens = find("tokens.txt")
        voices = find("voices.bin", "voices.txt")
        # Kokoro multi-lang (v1.0+) bundles espeak-ng-data and a set of
        # lexicon-*.txt files; pass all lexicons comma-separated like the
        # official sherpa-onnx examples do. Kitten bundles its own
        # espeak-ng-data and has no lexicon files.
        data_dir = os.path.join(directory, "espeak-ng-data")
        if not os.path.isdir(data_dir):
            data_dir = ""
        lexicons = []
        try:
            lexicons = sorted(
                f for f in os.listdir(directory)
                if f.startswith("lexicon-") and f.endswith(".txt")
            )
        except OSError:
            pass
        lexicon = ",".join(os.path.join(directory, f) for f in lexicons)
        files.update(
            {
                "model": model,
                "tokens": tokens,
                "voices_file": voices,
                "data_dir": data_dir,
                "lexicon": lexicon,
                "sid": voice_entry.get("sid", 0),
            }
        )
    elif engine == "matcha":
        # Matcha: acoustic model (model-steps-*.onnx) + shared neural vocoder.
        model = None
        try:
            entries = os.listdir(directory)
            steps = sorted(e for e in entries if _STEP_ONNX.match(e))
        except OSError:
            steps = []
        if steps:
            model = os.path.join(directory, steps[0])
        else:
            model = _find_onnx(directory)
        tokens = find("tokens.txt")
        lexicon = find("lexicon.txt")
        # Lexicon-based models (zh-baker etc.) must NOT get espeak-ng-data:
        # sherpa-onnx would switch to the piper frontend and crash on their
        # tokens.txt. Only fall back to the shared data when no lexicon exists.
        data_dir = os.path.join(directory, "espeak-ng-data")
        if not os.path.isdir(data_dir):
            if not lexicon:
                shared = os.path.join(paths.models_dir(), "shared", "espeak-ng-data")
                if os.path.isdir(shared):
                    data_dir = shared
                else:
                    data_dir = ""
            else:
                data_dir = ""
        vocoder = find("vocos-22khz-univ.onnx")
        if not vocoder:
            shared_vocoder = os.path.join(
                paths.models_dir(), "shared", "vocoders", "vocos-22khz-univ.onnx"
            )
            if os.path.isfile(shared_vocoder):
                vocoder = shared_vocoder
        files.update(
            {
                "model": model,
                "tokens": tokens,
                "lexicon": lexicon or "",
                "data_dir": data_dir,
                "vocoder": vocoder or "",
                "sid": voice_entry.get("sid", 0),
            }
        )
    else:
        model = _find_onnx(directory)
        tokens = find("tokens.txt")
        lexicon = find("lexicon.txt")
        # Character-frontend models (e.g. Coqui VITS) take plain text with
        # model + tokens only; the espeak-ng piper frontend must NOT be
        # attached to them.
        data_dir = os.path.join(directory, "espeak-ng-data")
        if voice_entry.get("frontend") != "char":
            if not os.path.isdir(data_dir):
                # Same rule as the vits branch: lexicon-based models must not
                # receive the shared espeak-ng-data (sherpa-onnx would use the
                # piper frontend).
                if not lexicon:
                    shared = os.path.join(
                        paths.models_dir(), "shared", "espeak-ng-data"
                    )
                    if os.path.isdir(shared):
                        data_dir = shared
                    else:
                        data_dir = ""
                else:
                    data_dir = ""
        files.update(
            {
                "model": model,
                "tokens": tokens,
                "lexicon": lexicon or "",
                "data_dir": data_dir if os.path.isdir(data_dir) else "",
                "sid": voice_entry.get("sid", 0),
            }
        )
    return files


_STEP_ONNX = re.compile(r"model-steps-\d+\.onnx$")


def _find_onnx(directory: str) -> Optional[str]:
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    onnx = sorted(e for e in entries if e.lower().endswith(".onnx"))
    # Prefer a model named model.onnx, then the largest .onnx file.
    if "model.onnx" in onnx:
        return os.path.join(directory, "model.onnx")
    best, best_size = None, -1
    for name in onnx:
        size = os.path.getsize(os.path.join(directory, name))
        if size > best_size:
            best, best_size = name, size
    return os.path.join(directory, best) if best else None
