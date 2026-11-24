"""Tests for the Voice Lab (Pocket TTS / F5-TTS / KokoClone / ...).

No engine package and no GPU is required: the registry, the device selection,
the request builder, the clone store integration and the worker protocol are
all exercised with the standard-library interpreter.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import wave
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import voicelab
from ai_voice_studio.tts import catalog
from ai_voice_studio.tts.models import ModelStore, resolve_voice_files
from ai_voice_studio.voicelab import engines as voice_lab
from ai_voice_studio.voicelab import options as tuning

WORKER = os.path.join(
    os.path.dirname(os.path.abspath(voicelab.__file__)), "worker.py"
)


def _write_silence(path, frames=480, sample_rate=24000):
    """A tiny silent WAV, written with the standard library.

    ``soundfile`` is an engine-level dependency and is deliberately *not*
    needed to run this suite, so the fake engine and its reference recordings
    are built from ``wave`` instead.
    """
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def _load_worker():
    """The worker module, loaded standalone (it must not import the app)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("aivs_voicelab_worker", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


class RegistryTest(unittest.TestCase):
    def test_the_engines_are_registered(self):
        self.assertEqual(
            voice_lab.engine_ids(),
            ("pocket_tts", "f5tts", "kokoclone", "chatterbox",
             "qwen3tts", "vibevoice", "moss_tts", "gptsovits",
             "indic_parler", "kitten_tts_2"),
        )
        # NeuTTS was dropped: it needs a HuggingFace login to fetch its model.
        self.assertNotIn("neutts", voice_lab.engine_ids())
        # Every engine here clones a voice from a recording but one: Indic
        # Parler-TTS speaks the speakers its model card names and is steered
        # by a description of the voice, with no reference-audio cloning at
        # all (its own ``voice_cloning`` flag says so).
        for engine_id in voice_lab.engine_ids():
            info = voice_lab.engine(engine_id)
            self.assertTrue(info["name"], engine_id)
            self.assertEqual(bool(info["voice_cloning"]),
                             engine_id != "indic_parler", engine_id)
            # Every engine runs on the GPU; the smaller ones can also run on
            # a CPU, the transformer engines cannot (``gpu`` and no ``cpu``).
            self.assertIn(info["devices"], ("cpu+gpu", "gpu"), engine_id)
            self.assertTrue(voice_lab.gpu_supported(engine_id), engine_id)
            self.assertTrue(voice_lab.packages(engine_id), engine_id)

    def test_the_transformer_engines_are_gpu_only(self):
        """Chatterbox, Qwen3-TTS, VibeVoice, MOSS-TTS, GPT-SoVITS and Indic
        Parler-TTS are too slow on a CPU to offer it."""
        for engine_id in ("chatterbox", "qwen3tts", "vibevoice", "moss_tts",
                          "gptsovits", "indic_parler"):
            self.assertTrue(voice_lab.gpu_only(engine_id), engine_id)
            self.assertFalse(voice_lab.cpu_supported(engine_id), engine_id)
            for device, _label in voicelab.device_options(engine_id):
                self.assertEqual(device, "cuda", engine_id)
        # The F5-TTS-based engines, KokoClone and Kitten TTS 2.0 are offered on
        # the CPU as well: F5-TTS and the Kokoro ONNX model it speaks through
        # are small enough to run there, and KittenTTS 2 documents a CPU run
        # (about 6 GB of RAM) as a mode of its own, even though its PyTorch
        # path is the slower one.
        for engine_id in ("pocket_tts", "f5tts", "kokoclone", "kitten_tts_2"):
            self.assertFalse(voice_lab.gpu_only(engine_id), engine_id)
            self.assertTrue(voice_lab.cpu_supported(engine_id), engine_id)

    def test_catalog_exposes_the_engines(self):
        ids = {tts["id"] for tts in catalog.get_tts_list()}
        for engine_id in voice_lab.engine_ids():
            self.assertIn(engine_id, ids)
        cloning = {tts["id"] for tts in catalog.cloning_capable_tts()}
        for engine_id in voice_lab.engine_ids():
            # The Voice Clone category (and the catalog helper behind it) lists
            # the engines that clone a voice from a recording; Indic Parler-TTS
            # is the one that does not.
            self.assertEqual(engine_id in cloning, engine_id != "indic_parler",
                             engine_id)
        f5tts = catalog.find_tts("f5tts")
        self.assertIn("torch", catalog.requires_packages(f5tts))
        self.assertIn("torchaudio", catalog.requires_packages(f5tts))

    def test_builtin_voices_match_the_declared_languages(self):
        """Every built-in voice must sit inside a declared language/variant."""
        for engine_id in voice_lab.engine_ids():
            info = voice_lab.engine(engine_id)
            declared = {
                (lang["code"], variant["id"])
                for lang in info["languages"]
                for variant in lang["variants"]
            }
            voices = voice_lab.builtin_voices(engine_id)
            # GPT-SoVITS is clone-only: it speaks from a reference recording,
            # so its built-in list is empty by design and its voices appear
            # only once the user clones one.
            if engine_id not in ("gptsovits",):
                self.assertTrue(voices, engine_id)
            for voice in voices:
                self.assertEqual(voice["engine"], engine_id)
                self.assertTrue(voice["voice"])
                self.assertTrue(voice["voice_name"], voice["voice"])
                self.assertIn((voice["language"], voice["variant"]), declared,
                              (engine_id, voice["voice"]))
                self.assertNotIn("device", voice)

    def test_builtin_voice_counts(self):
        self.assertEqual(voice_lab.count_builtin_voices("pocket_tts"), 26)
        self.assertEqual(voice_lab.count_builtin_voices("f5tts"), 2)
        # One Kokoro voice for each language KokoClone speaks out of the box.
        self.assertEqual(voice_lab.count_builtin_voices("kokoclone"), 7)
        # Chatterbox has no named voices: one "default voice" entry per model
        # (two English ones) and one for each of the multilingual model's 23
        # languages.
        self.assertEqual(voice_lab.count_builtin_voices("chatterbox"), 25)
        # Nine named voices on each of the two CustomVoice models.
        self.assertEqual(voice_lab.count_builtin_voices("qwen3tts"), 18)
        # Seven VibeVoice voice presets on each of its two model sizes.
        self.assertEqual(voice_lab.count_builtin_voices("vibevoice"), 14)
        # MOSS-TTS has one "default voice" per language on each variant: 31
        # languages on both Local Transformer checkpoints (31 + 31).
        self.assertEqual(voice_lab.count_builtin_voices("moss_tts"), 62)
        # GPT-SoVITS ships no voices at all: it always clones one.
        self.assertEqual(voice_lab.count_builtin_voices("gptsovits"), 0)
        self.assertEqual(voice_lab.GPTSOVITS_BUILTIN_VOICES, ())
        # Indic Parler-TTS: one entry per speaker name its model card
        # documents (the unofficial languages are deliberately absent).
        self.assertEqual(voice_lab.count_builtin_voices("indic_parler"), 65)
        # Kitten TTS 2.0: its 47 built-in voices - thirty-eight English ones
        # (the eight from KittenTTS 0.8 plus thirty persona voices) and nine
        # named after the language their reference clip carries.  This is the
        # engine's own ``available_voices`` count, and the worker keeps its own
        # copy of the list (see ``KittenTts2Test``).
        self.assertEqual(voice_lab.count_builtin_voices("kitten_tts_2"), 47)


    def test_every_engine_names_the_variant_a_clone_belongs_on(self):
        """A cloned voice must land on a model that can actually speak it."""
        declared = {
            engine_id: {
                variant["id"]
                for lang in voice_lab.engine(engine_id)["languages"]
                for variant in lang["variants"]
            }
            for engine_id in voice_lab.engine_ids()
        }
        for engine_id in voice_lab.engine_ids():
            self.assertIn(
                voice_lab.clone_variant(engine_id), declared[engine_id], engine_id
            )
        # Chatterbox's English models cannot speak another language, so a
        # clone goes on the multilingual one; Qwen3-TTS's CustomVoice models
        # cannot clone at all, so a clone goes on a Base model.
        self.assertEqual(voice_lab.clone_variant("chatterbox"),
                         voice_lab.CHATTERBOX_MULTILINGUAL_VARIANT)
        self.assertEqual(voice_lab.clone_variant("qwen3tts"), "qwen3_clone_1_7b")
        self.assertIsNone(voice_lab.qwen_model("chatterbox_mtl"))
        self.assertEqual(voice_lab.qwen_model("qwen3_clone_0_6b")["kind"], "clone")

    def test_vibevoice_declares_a_model_per_size_and_seven_presets(self):
        """VibeVoice is a separate entry per model size, GPU-only, in its own
        environment, with the fork's voice presets as its built-in voices."""
        self.assertEqual(
            [variant for variant, _repo, _label in voice_lab._VIBEVOICE_MODELS],
            ["vibevoice_1_5b", "vibevoice_7b"],
        )
        for variant, repo, _label in voice_lab._VIBEVOICE_MODELS:
            self.assertTrue(repo.startswith("vibevoice/VibeVoice-"), repo)
            self.assertEqual(voice_lab.vibevoice_model(variant)["repo"], repo)
        # A CPU run of an autoregressive 1.5-7B model is many times slower
        # than real time, so the CPU is not offered.
        self.assertTrue(voice_lab.gpu_only("vibevoice"))
        self.assertFalse(voice_lab.cpu_supported("vibevoice"))
        # Its pinned transformers cannot share an environment with Qwen3-TTS's.
        self.assertTrue(voice_lab.own_environment("vibevoice"))
        self.assertEqual(
            {lang["code"] for lang in voice_lab.engine("vibevoice")["languages"]},
            {"en", "zh"},
        )
        voices = voice_lab.builtin_voices("vibevoice")
        self.assertEqual(len(voices), 14)
        self.assertEqual(
            {voice["voice"] for voice in voices},
            {"Alice", "Maya", "Carter", "Frank", "Samuel", "Bowen",
             "Xinran"},
        )
        for voice in voices:
            self.assertTrue(voice["ref_resource"].startswith("vibevoice:"))
            self.assertIn(voice["variant"],
                          ("vibevoice_1_5b", "vibevoice_7b"))

    def test_the_qwen_speakers_sit_in_a_model_that_has_them(self):
        voices = voice_lab.builtin_voices("qwen3tts")
        self.assertEqual(sorted({v["voice"] for v in voices}), [
            "Aiden", "Dylan", "Eric", "Ono_Anna", "Ryan", "Serena",
            "Sohee", "Uncle_Fu", "Vivian",
        ])
        for voice in voices:
            self.assertEqual(voice_lab.qwen_model(voice["variant"])["kind"],
                             "custom")
            self.assertIn(voice["language"], ("zh", "en", "ja", "ko"))

    def test_f5_voices_point_at_package_references(self):
        for voice in voice_lab.builtin_voices("f5tts"):
            self.assertTrue(voice["ref_resource"].startswith("f5_tts:"))
            self.assertTrue(voice["ref_text"])

    def test_kokoclone_builtin_voices_are_kokoro_voices(self):
        """KokoClone speaks with Kokoro's voices until a sample is cloned."""
        voices = voice_lab.builtin_voices("kokoclone")
        self.assertEqual(
            [v["voice"] for v in voices],
            ["af_bella", "hf_alpha", "ff_siwis", "em_alex", "im_nicola",
             "pf_dora", "zf_001"],
        )
        self.assertEqual(
            [v["language"] for v in voices],
            ["en", "hi", "fr", "es", "it", "pt", "zh"],
        )
        # A built-in voice is not a clone: it carries neither a reference
        # recording nor the clone marker.
        for voice in voices:
            self.assertNotIn("ref_audio", voice)
            self.assertNotEqual(voice["voice"], voice_lab.CLONE_VOICE_ID)


class DeviceTest(unittest.TestCase):
    def setUp(self):
        voicelab.has_cuda(force=True)
        self.addCleanup(voicelab.has_cuda, True)

    def test_cpu_is_always_offered_first(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            options = voicelab.device_options()
        self.assertEqual(options[0][0], "cpu")
        self.assertEqual([value for value, _ in options], ["cpu"])

    def test_gpu_is_offered_in_addition_to_the_cpu(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            options = voicelab.device_options()
        values = [value for value, _ in options]
        self.assertEqual(values, ["cpu", "cuda", "auto"])

    def test_resolving_a_device(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(voicelab.resolve_device("auto"), "cuda")
            self.assertEqual(voicelab.resolve_device("cuda"), "cuda")
            self.assertEqual(voicelab.resolve_device("cpu"), "cpu")
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            self.assertEqual(voicelab.resolve_device("auto"), "cpu")
            # A GPU that is not there silently falls back to the CPU.
            self.assertEqual(voicelab.resolve_device("cuda"), "cpu")
        self.assertEqual(voicelab.resolve_device(None), "cpu")
        self.assertEqual(voicelab.resolve_device("nonsense"), "cpu")

    def test_request_carries_the_resolved_device(self):
        voice = voice_lab.builtin_voices("pocket_tts")[0]
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            request = voicelab.build_synthesize_request(
                engine_id="pocket_tts", device="auto", text="Hello", voice_entry=voice
            )
        self.assertEqual(request["device"], "cuda")
        self.assertEqual(request["engine"], "pocket_tts")
        self.assertEqual(request["voice"], voice["voice"])
        self.assertEqual(request["text"], "Hello")

    def test_every_builtin_voice_builds_a_serializable_request(self):
        """A typo in the registry must not reach the worker as garbage."""
        for engine_id in voice_lab.engine_ids():
            for voice in voice_lab.builtin_voices(engine_id):
                request = voicelab.build_synthesize_request(
                    engine_id=engine_id, device="cpu", text="Hello",
                    voice_entry=voice, speed=1.0,
                )
                payload = json.dumps(request)
                self.assertEqual(json.loads(payload)["cmd"], "synthesize")
                self.assertEqual(request["engine"], engine_id)
                # The device is resolved per engine: a GPU-only engine turns
                # a stored CPU choice into the only device it has.
                self.assertEqual(
                    request["device"], voicelab.resolve_device("cpu", engine_id)
                )
                self.assertEqual(request["device"],
                                 "cuda" if voice_lab.gpu_only(engine_id) else "cpu")
                self.assertTrue(request["voice"], voice)
                self.assertNotIn("None", payload)


class GpuSupportTest(unittest.TestCase):
    """Every Voice Lab engine can additionally run on an NVIDIA GPU."""

    def test_every_engine_supports_the_gpu(self):
        for engine_id in voice_lab.engine_ids():
            self.assertTrue(voice_lab.gpu_supported(engine_id), engine_id)
            cuda = voice_lab.cuda_packages(engine_id)
            # Every engine's GPU support is PyTorch's, so every engine names it
            # in its CUDA packages; the entry may carry the pin the engine asks
            # for, but the distribution is what matters.
            self.assertTrue(cuda, engine_id)
            self.assertIn(
                "torch", {voicelab.requirement_name(p) for p in cuda},
                engine_id,
            )

    def test_the_gpu_torch_comes_from_a_pytorch_wheel_index(self):
        for engine_id in voice_lab.engine_ids():
            index = voice_lab.cuda_index_url(engine_id)
            self.assertTrue(voice_lab.cuda_packages(engine_id), engine_id)
            self.assertIn(index, (voice_lab.PYTORCH_CUDA_INDEX,
                                 voice_lab.CHATTERBOX_CUDA_INDEX), engine_id)

    def test_the_cuda_index_carries_wheels_for_this_python(self):
        # ``cu121`` stops at Python 3.12: on Python 3.13 pip answered the
        # engine install with "No matching distribution found for torch" and
        # the install failed outright.  Keep the index in step with the
        # interpreter the application runs on.
        self.assertNotIn("cu121", voice_lab.PYTORCH_CUDA_INDEX)
        self.assertTrue(voice_lab.PYTORCH_CUDA_INDEX.endswith("/cu128"),
                        voice_lab.PYTORCH_CUDA_INDEX)

    def test_no_module_hard_codes_the_cuda_index(self):
        # One source of truth (``engines.PYTORCH_CUDA_INDEX``): a second,
        # stale literal is exactly how the OmniVoice installers and the Voice
        # Lab installers drifted apart.
        import pathlib
        import re

        import ai_voice_studio

        root = pathlib.Path(ai_voice_studio.__file__).parent
        offenders = []
        declared = set()
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for found in re.findall(
                r"https?://download\.pytorch\.org/whl/cu\w+", text
            ):
                if path.name == "engines.py":
                    declared.add(found)
                else:
                    offenders.append((path.name, found))
        self.assertEqual(offenders, [],
                         "hard-code the index in engines.py instead")
        # ... and the only indices the application knows are the declared ones
        # (the shared cu128 and the cu126 Chatterbox needs for its torch pin).
        self.assertEqual(
            declared,
            {voice_lab.PYTORCH_CUDA_INDEX, voice_lab.CHATTERBOX_CUDA_INDEX},
        )

    def test_f5tts_keeps_torch_and_torchaudio_on_the_same_index(self):
        # Mixing a CUDA torch with a PyPI torchaudio breaks the pair; both
        # wheels must come from the same index.
        self.assertEqual(tuple(sorted(voice_lab.cuda_packages("f5tts"))),
                         ("torch", "torchaudio"))

    def test_a_gpu_machine_installs_cuda_torch_first(self):
        from ai_voice_studio import venv_packages

        for engine_id in voice_lab.engine_ids():
            with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                    mock.patch.object(venv_packages, "installed",
                                      return_value=False):
                plan = voicelab.install_plan(engine_id)
            self.assertEqual(plan[0][1], voice_lab.cuda_index_url(engine_id),
                             engine_id)
            self.assertIn(
                "torch",
                {voicelab.requirement_name(p) for p in plan[0][0]},
                engine_id,
            )
            # Everything else is still installed - from PyPI, or from the
            # source archive an engine declares instead (KokoClone's own
            # package and its Kanade model are not on PyPI).
            installed = {pkg for step, _index in plan for pkg in step}
            # By distribution name: a CUDA step names the pinned release
            # (``torch==2.6.0``) while the engine's uninstall list holds the
            # plain name.
            installed_names = {voicelab.requirement_name(p) for p in installed}
            elsewhere = set(voice_lab.packaged_elsewhere(engine_id))
            declared_steps = [list(arguments) for arguments, _index, _names
                              in voice_lab.extra_requirements(engine_id)]
            plan_steps = [list(step) for step, _index in plan]
            for package in voice_lab.packages(engine_id):
                if package in elsewhere and package not in installed:
                    # Not installed from an index by name: it comes from a
                    # source archive or a pinned wheel an extra step names
                    # itself (KokoClone's tree, Parler-TTS's wheel).  Every
                    # step the engine declares must be in the
                    # plan that installs it.
                    self.assertTrue(declared_steps, (engine_id, package))
                    for declared in declared_steps:
                        self.assertIn(declared, plan_steps,
                                      (engine_id, package))
                    continue
                self.assertIn(voicelab.requirement_name(package),
                              installed_names, (engine_id, package))

    def test_a_cpu_machine_installs_the_plain_wheels(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            plan = voicelab.install_plan("f5tts")
        self.assertEqual(plan,
                         [(list(voice_lab.packages("f5tts")), None)])

    def test_the_cpu_budget_travels_with_the_request(self):
        from ai_voice_studio import compute

        voice = dict(voice_lab.builtin_voices("pocket_tts")[0])
        request = voicelab.build_synthesize_request(
            engine_id="pocket_tts", device="cpu", text="Hi", voice_entry=voice,
        )
        self.assertEqual(request["threads"], compute.cpu_threads())
        self.assertGreaterEqual(request["threads"], 1)


class InstallPlanTest(unittest.TestCase):
    """What ``pip`` is asked to do when an engine is installed."""

    def test_installed_package_names_never_carry_a_version_specifier(self):
        # ``packages_for`` feeds both the install line and ``pip uninstall``,
        # and a requirement like "torchao<0.18" would make Remove fail.
        for engine_id in voice_lab.engine_ids():
            for package in voicelab.packages_for(engine_id):
                for symbol in ("<", ">", "=", ";", " "):
                    self.assertNotIn(symbol, package, (engine_id, package))

    @staticmethod
    def _plain_pypi_engines():
        """Engines installed from PyPI by name alone, in one step.  KokoClone
        additionally installs two source archives and the GPU ONNX runtime."""
        return [e for e in voice_lab.engine_ids()
                if not voice_lab.extra_requirements(e)
                and not voice_lab.cuda_extra_packages(e)]

    def test_a_cpu_machine_installs_every_package_in_one_step(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            for engine_id in self._plain_pypi_engines():
                plan = voicelab.install_plan(engine_id)
                self.assertEqual(
                    plan, [(list(voicelab.packages_for(engine_id)), None)],
                    engine_id,
                )

    def test_a_gpu_machine_installs_cuda_torch_first(self):
        for engine_id in voice_lab.engine_ids():
            with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                    mock.patch("ai_voice_studio.venv_packages.installed",
                               return_value=False):
                plan = voicelab.install_plan(engine_id)
            cuda = list(voice_lab.cuda_packages(engine_id))
            # Each engine's CUDA wheels come from its own index: the pinned
            # releases Chatterbox needs are not on the application-wide one.
            self.assertEqual(plan[0],
                             (cuda, voice_lab.cuda_index_url(engine_id)))
            # ... and the same packages are not requested twice.  (The extra
            # steps only ever *add* their own requirements, so the CUDA build
            # of PyTorch is never replaced by a later step.)  A plain name for
            # a package the CUDA step installed would make pip replace it with
            # PyPI's CPU-only wheel, so the by-name steps must leave it alone -
            # recognised by distribution name, because the CUDA entry may be
            # pinned (``torch==2.6.0``) while the rest of the list is not.
            cuda_names = {voicelab.requirement_name(package) for package in cuda}
            for arguments, _index in plan[1:]:
                for package in voicelab.packages_for(engine_id):
                    if voicelab.requirement_name(package) not in cuda_names:
                        continue
                    self.assertNotIn(package, arguments, (engine_id, package))

    def test_chatterbox_installs_its_pinned_cuda_pytorch(self):
        """Chatterbox pins torch==2.6.0; its CUDA build must be installed
        *first*, from the index that still carries that release."""
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            plan = voicelab.install_plan("chatterbox")
        cuda, index = plan[0]
        self.assertEqual(cuda, ["torch==2.6.0", "torchaudio==2.6.0"])
        self.assertEqual(index, voice_lab.CHATTERBOX_CUDA_INDEX)
        self.assertIn("cu126", index)
        # The plain names are gone from the by-name step, so pip keeps the
        # CUDA build (a local version `2.6.0+cu126` satisfies `==2.6.0`).
        for arguments, _index in plan[1:]:
            self.assertNotIn("torch", arguments)
            self.assertNotIn("torchaudio", arguments)
        self.assertIn("chatterbox-tts", plan[1][0])

    def test_a_plain_cpu_install_gets_every_name_from_pypi(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            for engine_id in voice_lab.engine_ids():
                plan = voicelab.install_plan(engine_id)
                by_name = [p for step, _index in plan for p in step
                           if not p.startswith("-")]
                for package in voicelab.packages_for(engine_id):
                    if package in voice_lab.packaged_elsewhere(engine_id):
                        continue
                    self.assertIn(package, by_name, (engine_id, package))

    def test_kokoclone_installs_its_source_archives_and_cuda_onnx_runtime(self):
        """KokoClone and Kanade are source trees, not PyPI packages."""
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            plan = voicelab.install_plan("kokoclone")
        arguments = [step for step, _index in plan]
        self.assertTrue(any("https://" in arg for arg_list in arguments
                            for arg in arg_list), "the source archives are missing")
        # The source archives are installed with --no-deps, so their own
        # dependency lists cannot swap the CUDA PyTorch for the CPU wheel.
        source_step = next(args for args in arguments
                           if any("kokoclone/archive" in arg for arg in args))
        self.assertIn("--no-deps", source_step)
        self.assertTrue(any(voice_lab.KOKOCLONE_SOURCE_URL in args
                            for args in arguments))
        self.assertTrue(any(voice_lab.KANADE_SOURCE_URL in args
                            for args in arguments))
        # Both are still uninstalled by name.
        for name in ("kokoclone", "kanade-tokenizer"):
            self.assertIn(name, voicelab.packages_for("kokoclone"))
        # The GPU ONNX runtime is added *after* the PyPI step that installs the
        # CPU one, so the accelerated build is the last one to land (both
        # distributions ship the same module and the last one wins).
        cpu_step = next(index for index, args in enumerate(arguments)
                        if "onnxruntime" in args)
        gpu_step = next(index for index, args in enumerate(arguments)
                        if "onnxruntime-gpu" in args)
        self.assertGreater(gpu_step, cpu_step, plan)

    def test_kokoclone_installs_misaki_despite_its_python_pin(self):
        # ``misaki`` declares Requires-Python <3.13 while the application runs
        # on 3.13, so the flag is what makes the install work at all.
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            plan = voicelab.install_plan("kokoclone")
        misaki = [args for args, _index in plan if "misaki" in args]
        self.assertEqual(len(misaki), 1, plan)
        self.assertIn("--ignore-requires-python", misaki[0])
        # ... and it is not also asked for by name from the index.
        for args, _index in plan:
            if args is misaki[0]:
                continue
            self.assertNotIn("misaki", args)

    def test_the_worker_and_the_app_agree_on_what_each_engine_imports(self):
        worker = _load_worker()
        for engine_id in voice_lab.engine_ids():
            self.assertEqual(
                tuple(worker._ENGINE_MODULES.get(engine_id, ())),
                voice_lab.import_modules(engine_id),
                engine_id,
            )

    def test_the_worker_and_the_app_agree_on_the_kokoclone_voices(self):
        """The worker answers "is this built-in voice usable?" from its own
        copy of the list; it must not drift from the catalogue."""
        worker = _load_worker()
        self.assertEqual(
            list(worker._KOKOCLONE_VOICES),
            [(v["voice"], v["language"])
             for v in voice_lab.builtin_voices("kokoclone")],
        )
        self.assertTrue(all(isinstance(backend, type)
                            for backend in worker._BACKENDS.values()))
        self.assertIn("kokoclone", worker._BACKENDS)

    def test_the_worker_and_the_app_agree_on_the_chatterbox_models(self):
        worker = _load_worker()
        declared = {
            variant["id"]
            for lang in voice_lab.engine("chatterbox")["languages"]
            for variant in lang["variants"]
        }
        self.assertEqual({spec[0] for spec in worker._CHATTERBOX_MODELS},
                         declared)
        # The languages the multilingual model claims to speak are the ones
        # the catalogue offers voices for (plus English, from all three).
        multilingual = next(spec for spec in worker._CHATTERBOX_MODELS
                             if spec[0] == "chatterbox_mtl")
        self.assertEqual(
            set(multilingual[3]),
            {lang["code"] for lang in voice_lab.engine("chatterbox")["languages"]},
        )

    def test_vibevoice_installs_its_source_archive(self):
        """VibeVoice is a GitHub source tree, not a PyPI package, so it is
        installed from a pinned archive - like KokoClone."""
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            plan = voicelab.install_plan("vibevoice")
        arguments = [step for step, _index in plan]
        self.assertEqual(plan[0], (["torch"], voice_lab.PYTORCH_CUDA_INDEX))
        self.assertTrue(
            any(voice_lab.VIBEVOICE_SOURCE_URL in args for args in arguments),
            plan,
        )
        self.assertIn("/archive/", voice_lab.VIBEVOICE_SOURCE_URL)
        # It is still uninstalled by name, and PyTorch is not asked for twice.
        self.assertIn("vibevoice", voicelab.packages_for("vibevoice"))
        for args, _index in plan[1:]:
            self.assertNotIn("torch", args)

    def test_the_worker_and_the_app_agree_on_the_vibevoice_models(self):
        worker = _load_worker()
        self.assertEqual(
            {spec[0] for spec in worker._VIBEVOICE_MODELS},
            {variant for variant, _repo, _label in voice_lab._VIBEVOICE_MODELS},
        )
        for variant, repo, _label in voice_lab._VIBEVOICE_MODELS:
            self.assertEqual(voice_lab.vibevoice_model(variant)["repo"], repo)
        # The presets the worker answers with are exactly the catalogue's.
        self.assertEqual(
            sorted(worker._VIBEVOICE_VOICES),
            sorted({v["voice"] for v in voice_lab.builtin_voices("vibevoice")}),
        )

    def test_the_worker_and_the_app_agree_on_the_qwen_models(self):
        worker = _load_worker()
        self.assertEqual(
            sorted(spec[:3] for spec in worker._QWEN_MODELS),
            sorted(
                (variant, voice_lab.qwen_model(variant)["repo"],
                 voice_lab.qwen_model(variant)["kind"])
                for variant, _repo, _kind, _label in voice_lab._QWEN_MODELS
            ),
        )
        # The voices the worker offers are exactly the catalogue's speakers.
        self.assertEqual(
            sorted(worker._QWEN_SPEAKERS),
            sorted({v["voice"] for v in voice_lab.builtin_voices("qwen3tts")}),
        )
        self.assertEqual(
            set(worker._QWEN_LANGUAGE_NAMES), set(voice_lab._QWEN_LANGUAGES)
        )

    def test_the_moss_variants_all_fit_under_twelve_gb(self):
        """Every MOSS-TTS variant offered must fit a 12 GB card, and the 8B
        checkpoint (which does not) must be absent."""
        self.assertEqual(
            [variant for variant, *_rest in voice_lab._MOSS_MODELS],
            ["moss_tts_local_v15_4b", "moss_tts_local_1_7b"],
        )
        for variant, repo, _label, vram, _devices in voice_lab._MOSS_MODELS:
            self.assertLess(vram, 12 * 1024, variant)
            self.assertTrue(repo.startswith("OpenMOSS-Team/MOSS-TTS"), repo)
            self.assertEqual(voice_lab.moss_model(variant)["repo"], repo)
        # The 8B v1.5 checkpoint needs ~19 GB, so it is deliberately not here.
        repos = {repo for _v, repo, *_r in voice_lab._MOSS_MODELS}
        self.assertNotIn("OpenMOSS-Team/MOSS-TTS-v1.5", repos)
        # Both checkpoints are GPU-only.
        self.assertTrue(voice_lab.gpu_only("moss_tts"))
        self.assertFalse(voice_lab.cpu_supported("moss_tts"))
        # Its pinned transformers cannot share an environment with another
        # engine's.
        self.assertTrue(voice_lab.own_environment("moss_tts"))

    def test_moss_declares_the_family_languages_and_default_voices(self):
        """Each MOSS variant has one default voice per language it speaks."""
        self.assertEqual(len(voice_lab._MOSS_LANGUAGES), 31)
        voices = voice_lab.builtin_voices("moss_tts")
        self.assertEqual(len(voices), 62)
        for voice in voices:
            self.assertEqual(voice["voice"], voice_lab.MOSS_DEFAULT_VOICE)
            self.assertEqual(voice["engine"], "moss_tts")
        # Both checkpoints speak every one of the family's 31 languages.
        languages = {voice["language"] for voice in voices}
        self.assertEqual(
            languages, {code for code, _name in voice_lab._MOSS_LANGUAGES}
        )

    def test_gptsovits_offers_every_release_under_twelve_gb(self):
        """All six GPT-SoVITS releases fit a card, and it ships no voices."""
        self.assertEqual(
            [variant for variant, *_rest in voice_lab._GPTSOVITS_MODELS],
            ["gptsovits_v2proplus", "gptsovits_v2pro", "gptsovits_v4",
             "gptsovits_v3", "gptsovits_v2", "gptsovits_v1"],
        )
        for variant, version, _label, vram in voice_lab._GPTSOVITS_MODELS:
            self.assertLess(vram, 12 * 1024, variant)
            self.assertIn(version, ("v1", "v2", "v3", "v4", "v2Pro",
                                    "v2ProPlus"))
            self.assertEqual(voice_lab.gptsovits_model(variant)["version"],
                             version)
        self.assertTrue(voice_lab.gpu_only("gptsovits"))
        self.assertFalse(voice_lab.cpu_supported("gptsovits"))
        self.assertTrue(voice_lab.own_environment("gptsovits"))
        # It is clone-only: no voices of its own, so the ones that appear are
        # the voices the user creates.
        self.assertEqual(voice_lab.builtin_voices("gptsovits"), [])
        self.assertEqual(voice_lab.GPTSOVITS_BUILTIN_VOICES, ())
        # Japanese is absent on purpose: its front-end needs ``pyopenjtalk``,
        # which has no wheel for this Python and cannot be built here, so the
        # engine is offered with the languages it can really speak.
        self.assertEqual(
            {lang["code"] for lang in voice_lab.engine("gptsovits")["languages"]},
            {"en", "zh", "ko", "yue"},
        )
        # ... and its Japanese front-end is not asked for either, because no
        # buildable release of it exists for this Python.
        self.assertNotIn("pyopenjtalk", voicelab.packages_for("gptsovits"))

    def test_the_worker_and_the_app_agree_on_the_moss_models(self):
        worker = _load_worker()
        self.assertEqual(
            sorted(worker._MOSS_MODELS),
            sorted(
                (variant, voice_lab.moss_model(variant)["repo"])
                for variant, _repo, _label, _vram, _devices
                in voice_lab._MOSS_MODELS
            ),
        )
        self.assertEqual(
            worker._MOSS_LANGUAGE_NAMES, voice_lab._MOSS_LANGUAGE_NAMES
        )

    def test_the_worker_and_the_app_agree_on_the_gptsovits_releases(self):
        worker = _load_worker()
        self.assertEqual(
            sorted(worker._GPTSOVITS_MODELS),
            sorted(
                (variant, version)
                for variant, version, _label, _vram in voice_lab._GPTSOVITS_MODELS
            ),
        )
        self.assertEqual(
            set(worker._GPTSOVITS_LANGUAGES),
            {lang["code"] for lang in voice_lab.engine("gptsovits")["languages"]},
        )
        self.assertEqual(worker._GPTSOVITS_SOURCE_URL,
                         voice_lab.GPT_SOVITS_SOURCE_URL)
        self.assertEqual(worker._GPTSOVITS_WEIGHTS_REPO,
                         voice_lab.GPT_SOVITS_WEIGHTS_REPO)

    def test_moss_installs_its_source_archive(self):
        """MOSS-TTS is a GitHub source tree, not a PyPI package."""
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            plan = voicelab.install_plan("moss_tts")
        arguments = [step for step, _index in plan]
        self.assertEqual(plan[0], (["torch", "torchaudio"],
                                   voice_lab.PYTORCH_CUDA_INDEX))
        self.assertTrue(
            any(voice_lab.MOSS_SOURCE_URL in args for args in arguments), plan
        )
        self.assertIn("/archive/", voice_lab.MOSS_SOURCE_URL)
        self.assertIn("moss-tts", voicelab.packages_for("moss_tts"))
        # ``moss-tts`` declares its ``transformers`` pin only in an extra the
        # bare-URL install does not select, so it is pinned here - and *after*
        # the archive, because pip's by-name step runs before it and would
        # otherwise install the newest release at the same time.
        self.assertEqual(
            arguments[-1], ["transformers==5.0.0"],
            "the transformers pin must be the last step",
        )
        # It is installed once, at the pinned release: the by-name step must
        # not also ask for it, or pip would see two versions.
        by_name = [p for args, _index in plan[1:-1] for p in args]
        self.assertNotIn("transformers", by_name)
        for args, _index in plan[1:]:
            self.assertNotIn("torch", args)

    def test_gptsovits_is_probed_by_a_distinctive_package(self):
        """GPT-SoVITS has no distribution of its own to probe for."""
        self.assertEqual(voice_lab.probe_package("gptsovits"),
                         "rotary_embedding_torch")
        self.assertIn("rotary_embedding_torch",
                      voicelab.packages_for("gptsovits"))
        # Its packages come from PyPI (and PyTorch from the CUDA index).
        self.assertEqual(voice_lab.extra_requirements("gptsovits"), ())
        self.assertEqual(voice_lab.cuda_packages("gptsovits"),
                         ("torch", "torchaudio"))

    def test_gptsovits_installs_the_dependencies_its_imports_need(self):
        """Every module on GPT-SoVITS's import path is installed by name.

        The engine has no packaging metadata, so pip cannot resolve its
        dependencies from it: they are listed in ``engines.py`` instead.  The
        list has to cover what ``TTS_infer_pack/TTS.py`` imports, or the first
        preview fails with a missing module - ``ffmpeg`` did exactly that.
        """
        packages = voicelab.packages_for("gptsovits")
        for package in ("ffmpeg-python", "pytorch-lightning", "peft",
                        "matplotlib", "jieba"):
            self.assertIn(package, packages)
        # ``jieba_fast`` is a C extension with no wheel for this Python and
        # needs a C++ compiler to build, so asking for it failed the whole
        # install with "pip install failed (exit 1)".  The worker stands in
        # plain ``jieba`` for it under the ``jieba_fast`` name, so it is
        # deliberately not installed here.
        self.assertNotIn("jieba_fast", packages)
        # The import list the worker is checked against names the same modules
        # those packages provide, so the post-install check catches this class
        # of failure too.
        self.assertEqual(
            voice_lab.import_modules("gptsovits"),
            ("torch", "torchaudio", "librosa", "ffmpeg", "pytorch_lightning"),
        )

    def test_moss_installs_the_transformers_release_its_code_targets(self):
        """The archive pins transformers only in an extra pip will not select.

        Without the pin pip takes the newest release, and the checkpoint's own
        remote code then fails against it - a missing ``sentencepiece``, then
        an internal that changed shape, then a refused import over
        ``safetensors``.  The pin is what makes the engine load at all.
        """
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            plan = voicelab.install_plan("moss_tts")
        pinned = [args for args, _index in plan
                  if any("transformers==" in a for a in args)]
        self.assertEqual(pinned, [["transformers==5.0.0"]])
        # ``sentencepiece`` is what the checkpoint's tokenizer asks for.
        self.assertIn("sentencepiece", voicelab.packages_for("moss_tts"))

    def test_gptsovits_imports_its_source_tree_the_way_upstream_does(self):
        """The tree needs two ``sys.path`` entries, not one.

        The repository root holds the ``GPT_SoVITS`` package; the folder
        *inside* it holds the modules its code imports as top level (``AR``,
        ``module``, ``text``, ``tools``) - upstream's ``api_v2.py`` adds both.
        Only the root was added before, which is what produced
        ``No module named 'AR'`` on the first preview.
        """
        worker = _load_worker()
        tree = os.path.join(tempfile.gettempdir(), "aivs_test_gptsovits")
        os.makedirs(os.path.join(tree, "GPT_SoVITS"), exist_ok=True)
        self.addCleanup(shutil.rmtree, tree, True)
        saved = list(sys.path)
        self.addCleanup(sys.path.__setitem__, slice(None), saved)
        worker._gptsovits_import_tree(tree)
        self.assertIn(tree, sys.path)
        self.assertIn(os.path.join(tree, "GPT_SoVITS"), sys.path)

    def test_gptsovits_stands_in_plain_jieba_for_jieba_fast(self):
        """``jieba_fast`` cannot be built on this Python; ``jieba`` is enough.

        Three of the engine's text front-ends do ``import jieba_fast``, and
        ``jieba_fast``'s ``setup.py`` imports the removed
        ``pkgutil.ImpImporter``, so it fails to build at all on Python 3.13.
        The worker registers plain ``jieba`` under that name, which is what
        makes the engine importable here.
        """
        worker = _load_worker()
        fake = types.ModuleType("jieba")
        fake.posseg = types.ModuleType("jieba.posseg")
        # The test interpreter has no ``jieba``, so it is supplied: what is
        # asserted is that the shim registers it under the ``jieba_fast`` name
        # and reports itself done, whatever interpreter runs the tests.
        with mock.patch.object(worker, "_module_available",
                               return_value=False), \
                mock.patch.dict(sys.modules, {"jieba": fake,
                                             "jieba.posseg": fake.posseg}):
            worker._GPTSOVITS_JIEBA_PATCHED = False
            worker._ensure_gptsovits_jieba_fast()
            self.assertTrue(worker._GPTSOVITS_JIEBA_PATCHED)
            self.assertIs(sys.modules.get("jieba_fast"), fake)
            self.assertIs(sys.modules.get("jieba_fast.posseg"), fake.posseg)

    def test_the_source_tree_check_uses_the_trees_own_import_path(self):
        """A source-tree engine is verified through the worker's own path.

        A plain ``import torch`` says nothing about whether the fetched tree
        can be imported, which is the failure that reached the user as a
        first-preview error rather than an install-time one.
        """
        tree = os.path.join(tempfile.gettempdir(), "aivs_probe_test")
        script = voicelab._source_tree_probe_script(
            ("torch", "ffmpeg"), tree
        )
        # The path is embedded with ``repr``, so it is the value it stands for
        # that matters, not its spelling in the script.
        self.assertIn(repr(os.path.join(tree, "GPT_SoVITS")), script)
        self.assertIn("GPT_SoVITS.TTS_infer_pack.TTS", script)
        self.assertIn("jieba_fast", script)
        # Only GPT-SoVITS runs this way: every other engine is a package.
        self.assertIsNone(voicelab._engine_source_tree("moss_tts"))

    def test_the_new_engines_fill_a_card(self):
        """Both engines declare how they use a GPU, one text per call."""
        moss = voice_lab.gpu_parallel("moss_tts", "moss_tts_local_1_7b")
        self.assertEqual(moss["mode"], "instances")
        self.assertGreater(moss["vram_mb"], 0)
        sovits = voice_lab.gpu_parallel("gptsovits", "gptsovits_v2proplus")
        self.assertEqual(sovits["mode"], "instances")
        self.assertGreater(sovits["vram_mb"], 0)


class EngineVerificationTest(unittest.TestCase):
    """An installed engine is imported once, so a broken env says so."""

    class _Runtime:
        def __init__(self, python_exe, created=True):
            self.python_exe = python_exe
            self.is_created = created

        def usable_python_exe(self):
            """The engine's interpreter, as the real runtime hands it over.

            This stand-in is built by the test rather than by ``venv``, so
            there is no recorded base interpreter that could have gone.
            """
            return self.python_exe

    def _patch_runtime(self, python_exe, created=True):
        runtime = self._Runtime(python_exe, created)
        patcher = mock.patch(
            "ai_voice_studio.python_runtime.get_runtime", return_value=runtime
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return runtime

    def test_a_working_engine_reports_no_problem(self):
        # ``json`` stands in for an engine package that really does import.
        self._patch_runtime(sys.executable)
        with mock.patch.object(voice_lab, "import_modules",
                               return_value=("json",)):
            self.assertIsNone(voicelab.verify_engine_import("f5tts"))

    def test_a_broken_engine_reports_the_import_error(self):
        # A module that cannot be imported stands in for an environment pip
        # "installed" into but which the engine cannot load.
        self._patch_runtime(sys.executable)
        with mock.patch.object(voice_lab, "import_modules",
                               return_value=("aivs_no_such_module",)):
            problem = voicelab.verify_engine_import("f5tts")
        self.assertIsNotNone(problem)
        self.assertIn("aivs_no_such_module", problem)
        self.assertIn(voice_lab.engine_name("f5tts"), problem)

    def test_a_missing_environment_is_not_a_problem(self):
        self._patch_runtime(sys.executable, created=False)
        self.assertIsNone(voicelab.verify_engine_import("f5tts"))


class CloneStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch(
            "ai_voice_studio.paths.user_data_dir", return_value=self.tmp.name
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ModelStore(
            state_file=os.path.join(self.tmp.name, "models.json")
        )
        self.sample = os.path.join(self.tmp.name, "my_voice.wav")
        with open(self.sample, "wb") as fh:
            fh.write(b"RIFFfake")

    def _installed_engine(self, engine_id):
        return [
            v for v in self.store.installed_voices()
            if v.get("engine") == engine_id
        ]

    def test_create_clone_copies_the_recording_and_registers_it(self):
        entry = voicelab.create_clone(
            self.store, "f5tts", "My Voice", self.sample,
            ref_text="hello there", language="en",
        )
        self.assertEqual(entry["name"], "My Voice")
        self.assertEqual(entry["engine"], "f5tts")
        self.assertTrue(os.path.isfile(entry["ref_audio"]))
        self.assertNotEqual(entry["ref_audio"], self.sample)
        self.assertTrue(entry["ref_audio"].startswith(self.tmp.name))

        voices = self._installed_engine("f5tts")
        self.assertEqual(len(voices), 1)
        voice = voices[0]
        self.assertTrue(voice["cloned"])
        self.assertEqual(voice["voice_name"], "My Voice")
        self.assertEqual(voice["ref_text"], "hello there")
        self.assertEqual(voice["tts_name"], voice_lab.engine_name("f5tts"))

    def test_create_clone_validates_its_input(self):
        with self.assertRaises(ValueError):
            voicelab.create_clone(self.store, "f5tts", "", self.sample)
        with self.assertRaises(ValueError):
            voicelab.create_clone(self.store, "f5tts", "Name", "")
        with self.assertRaises(ValueError):
            voicelab.create_clone(
                self.store, "f5tts", "Name", os.path.join(self.tmp.name, "nope.wav")
            )
        with self.assertRaises(ValueError):
            voicelab.create_clone(self.store, "piper", "Name", self.sample)

    def test_delete_clone_removes_it_everywhere(self):
        voicelab.create_clone(self.store, "pocket_tts", "Temp", self.sample)
        self.assertTrue(self._installed_engine("pocket_tts"))
        self.assertTrue(voicelab.delete_clone(self.store, "Temp"))
        self.assertEqual(self._installed_engine("pocket_tts"), [])

    def test_clone_file_survives_a_missing_source(self):
        entry = voicelab.create_clone(self.store, "f5tts", "Keeper", self.sample)
        os.remove(self.sample)
        self.assertTrue(os.path.isfile(entry["ref_audio"]))
        self.assertTrue(voicelab.clone_voices(self.store, "f5tts"))

    def test_voice_lab_engines_need_no_local_model_files(self):
        files = resolve_voice_files(
            {"dir": self.tmp.name, "engine": "f5tts", "sid": 0}
        )
        self.assertEqual(files["engine"], "f5tts")
        self.assertIsNone(files["model"])
        for engine_id in voice_lab.engine_ids():
            files = resolve_voice_files(
                {"dir": self.tmp.name, "engine": engine_id, "sid": 0}
            )
            self.assertIsNone(files["model"], engine_id)


class MeasuredDevicePolicyTest(unittest.TestCase):
    """What a *measured* CPU speed changes about the device list.

    The declared devices are a guess about the model's size; a measurement is
    what this computer actually does with it (``voicelab.benchmark``).
    """

    def setUp(self):
        self.addCleanup(voicelab.clear_cpu_rtf)
        voicelab.clear_cpu_rtf()

    def test_an_unmeasured_engine_keeps_its_declared_devices(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(
                [v for v, _l in voicelab.device_options("chatterbox")],
                ["cuda"],
            )
            self.assertEqual(
                [v for v, _l in voicelab.device_options("f5tts")],
                ["cpu", "cuda", "auto"],
            )
        self.assertIsNone(voicelab.cpu_verdict("f5tts"))

    def test_a_fast_cpu_adds_the_cpu_to_a_gpu_only_engine(self):
        voicelab.set_cpu_rtf("qwen3tts", 1.4)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            values = [v for v, _l in voicelab.device_options("qwen3tts")]
            # The GPU stays first (that is what the engine is for), but the
            # measured CPU is now a real option.
            self.assertEqual(values, ["cuda", "cpu"])
            self.assertEqual(voicelab.resolve_device("cpu", "qwen3tts"), "cpu")
        self.assertIn("measured 1.40x real time",
                      voicelab.cpu_option_label("qwen3tts"))
        self.assertIn("the CPU is offered",
                      voicelab.cpu_measurement_note("qwen3tts"))

    def test_a_cpu_over_the_limit_is_not_added(self):
        voicelab.set_cpu_rtf("qwen3tts", voicelab.CPU_RTF_LIMIT + 0.01)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(
                [v for v, _l in voicelab.device_options("qwen3tts")], ["cuda"]
            )
            self.assertEqual(voicelab.resolve_device("cpu", "qwen3tts"), "cuda")

    def test_a_slow_cpu_takes_the_entry_away_while_a_card_is_present(self):
        voicelab.set_cpu_rtf("f5tts", 4.2)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(
                [v for v, _l in voicelab.device_options("f5tts")],
                ["cuda", "auto"],
            )
            # A stored "cpu" must not start the very run the measurement
            # ruled out.
            self.assertEqual(voicelab.resolve_device("cpu", "f5tts"), "cuda")
            self.assertEqual(voicelab.resolve_device("auto", "f5tts"), "cuda")
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            # No card to use instead: the slow CPU is still all there is.
            self.assertEqual(
                [v for v, _l in voicelab.device_options("f5tts")], ["cpu"]
            )
            self.assertEqual(voicelab.resolve_device("cpu", "f5tts"), "cpu")

    def test_a_measurement_that_did_not_finish_counts_as_too_slow(self):
        voicelab.set_cpu_rtf("f5tts", None)
        self.assertTrue(voicelab.cpu_measured("f5tts"))
        self.assertIs(voicelab.cpu_verdict("f5tts"), False)
        self.assertIn("did not finish", voicelab.cpu_measurement_note("f5tts"))
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertNotIn("cpu", [v for v, _l in voicelab.device_options("f5tts")])

    def test_the_measured_figure_is_written_the_way_it_reads(self):
        self.assertEqual(voicelab.format_rtf(0.42), "0.42x real time")
        self.assertEqual(voicelab.format_rtf(1.44), "1.44x real time")
        self.assertEqual(voicelab.format_rtf(12.5), "12x real time")
        self.assertEqual(voicelab.format_rtf(None), "did not finish")
        self.assertEqual(voicelab.cpu_measurement_note("f5tts"), "")


class _BenchmarkSettings:
    """A ``Settings`` stand-in: the benchmark only gets, sets and saves."""

    def __init__(self, values=None):
        self.values = dict(values or {})
        self.saved = 0
        #: Simulates a settings file that cannot be written.
        self.fail_save = False

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value

    def save(self):
        if self.fail_save:
            raise OSError("the settings file is read-only")
        self.saved += 1


class _ScriptedWorker:
    """A ``CloneWorker`` stand-in that answers from a script of results.

    The script is shared - and consumed - by every worker a measurement
    creates, because the probes of one measurement run one after another.
    """

    def __init__(self, answers):
        self.answers = answers
        self.requests = []
        self.killed = False
        self.started = False
        self.engine_id = ""
        self.device = ""

    def _ensure_proc(self):
        """Start the worker, as ``CloneWorker._ensure_proc`` does.

        Modelled rather than stubbed: a freshly made worker has no process,
        and ``_request`` refuses to talk to one that was never started - so a
        probe that forgets to start it fails here exactly as it failed in the
        application ("The Voice Lab worker is not running."), instead of
        appearing to measure an engine that never spoke.
        """
        self.started = True

    def _request(self, request):
        if not self.started:
            raise voicelab.VoicelabError(
                "The Voice Lab worker is not running."
            )
        self.requests.append(request)
        if not self.answers:
            return {"ok": False, "error": "no scripted answer"}
        answer = self.answers.pop(0)
        if callable(answer):
            return answer()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def kill(self):
        self.killed = True


class SpeedBenchmarkTest(unittest.TestCase):
    """The speed benchmark: one sentence timed, stored, and turned into a
    device verdict (``voicelab.benchmark``)."""

    def setUp(self):
        self.addCleanup(voicelab.clear_cpu_rtf)
        voicelab.clear_cpu_rtf()
        self.settings = _BenchmarkSettings()
        self.voice = dict(voice_lab.builtin_voices("f5tts")[0])

    @contextlib.contextmanager
    def _patched(self, answers, installed=True):
        """Patch the worker factory and the install check for one run."""
        self.workers = []

        def _worker(engine_id, device):
            worker = _ScriptedWorker(answers)
            worker.engine_id = engine_id
            worker.device = device
            self.workers.append(worker)
            return worker

        with mock.patch.object(voicelab, "CloneWorker", _worker), \
                mock.patch.object(voicelab, "is_installed",
                                  return_value=installed):
            yield

    @staticmethod
    def _ok(rtf, audio_seconds=3.0, sample_rate=24000, device_used=None):
        answer = {
            "ok": True,
            "rtf": rtf,
            "seconds": rtf * audio_seconds,
            "audio_seconds": audio_seconds,
            "sample_rate": sample_rate,
        }
        if device_used is not None:
            answer["device_used"] = device_used
        return answer

    def test_a_probe_asks_the_worker_to_measure_and_is_then_closed(self):
        from ai_voice_studio.voicelab import benchmark

        with self._patched([self._ok(0.5)]):
            result = benchmark.probe("f5tts", "cpu", voice_entry=self.voice)
        self.assertTrue(result["ok"])
        request = self.workers[0].requests[0]
        self.assertEqual(request["cmd"], "benchmark")
        self.assertEqual(request["engine"], "f5tts")
        self.assertEqual(request["device"], "cpu")
        self.assertEqual(request["text"], benchmark.PROBE_TEXT)
        self.assertTrue(self.workers[0].started, "the probe starts the worker")
        self.assertTrue(self.workers[0].killed, "the probe worker is released")

    def test_measuring_probes_the_cpu_and_enables_it_when_it_is_fast(self):
        from ai_voice_studio.voicelab import benchmark

        with self._patched([self._ok(0.2), self._ok(1.4)]):
            summary = benchmark.measure_engine(
                "chatterbox", "cuda", settings=self.settings,
                voice_entry=self.voice,
            )
        self.assertEqual([w.device for w in self.workers], ["cuda", "cpu"])
        self.assertTrue(summary["cpu_allowed"])
        self.assertAlmostEqual(voicelab.cpu_rtf("chatterbox"), 1.4)
        stored = self.settings.get(benchmark.settings_key("chatterbox"))
        self.assertAlmostEqual(stored["cpu"]["rtf"], 1.4)
        self.assertAlmostEqual(stored["cuda"]["rtf"], 0.2)
        self.assertTrue(stored["cpu"]["measured_at"])
        # The device list follows without a restart.
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertIn("cpu",
                          [v for v, _l in voicelab.device_options("chatterbox")])

    def test_the_cpu_probe_really_asks_for_the_cpu_on_a_gpu_only_engine(self):
        """A measurement must never be quietly taken on the other device.

        The request builder resolves a GPU-only engine's "cpu" to the GPU, so
        that a stored CPU choice cannot start the very run a declaration (or an
        earlier measurement) ruled out.  For a *probe* that resolution is
        backwards: the "CPU" figure would be the card's, so the CPU could only
        ever be offered when the GPU was fast - and it was recorded as a CPU
        figure in Settings.
        """
        from ai_voice_studio.voicelab import benchmark

        self.assertTrue(voice_lab.gpu_only("chatterbox"))
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                self._patched([self._ok(0.9, device_used="cpu")]):
            benchmark.probe("chatterbox", "cpu", voice_entry=self.voice)
        self.assertEqual(self.workers[0].device, "cpu")
        self.assertEqual(self.workers[0].requests[0]["device"], "cpu")

    def test_a_figure_from_the_other_device_is_not_filed_under_this_one(self):
        """``device_used`` is what the probe is filed under, so it must match.

        An engine that cannot use the card falls back to the CPU inside the
        worker; that figure is not this engine's GPU speed and must not become
        a GPU verdict.
        """
        from ai_voice_studio.voicelab import benchmark

        with self._patched([self._ok(0.4, device_used="cpu")]):
            result = benchmark.probe("f5tts", "cuda", voice_entry=self.voice)
        self.assertFalse(result["ok"])
        self.assertIn("ran on cpu instead of cuda", result["error"])

    def test_the_measured_figure_is_written_out_straight_away(self):
        """A minute-long measurement must not wait for the dialog's Apply.

        The Settings dialog saves when it is applied, so a figure that only
        lived in memory was lost by closing it - along with the CPU entry the
        measurement had just decided.
        """
        from ai_voice_studio.voicelab import benchmark

        with self._patched([self._ok(0.3)]):
            benchmark.measure_engine("f5tts", "cpu", settings=self.settings,
                                     voice_entry=self.voice)
        self.assertEqual(self.settings.saved, 1)
        self.assertAlmostEqual(
            self.settings.values[benchmark.settings_key("f5tts")]["cpu"]["rtf"],
            0.3,
        )

    def test_a_settings_file_that_cannot_be_written_still_measures(self):
        from ai_voice_studio.voicelab import benchmark

        self.settings.fail_save = True
        with self._patched([self._ok(0.3)]):
            summary = benchmark.measure_engine(
                "f5tts", "cpu", settings=self.settings, voice_entry=self.voice
            )
        self.assertTrue(summary["probes"]["cpu"]["ok"])
        self.assertAlmostEqual(summary["cpu_rtf"], 0.3)

    def test_a_cpu_probe_that_times_out_is_recorded_as_not_offered(self):
        from ai_voice_studio.voicelab import benchmark

        timed_out = {"ok": False, "timed_out": True,
                     "error": "did not finish in 300 seconds"}
        with self._patched([self._ok(0.2), timed_out]):
            summary = benchmark.measure_engine(
                "chatterbox", "cuda", settings=self.settings,
                voice_entry=self.voice,
            )
        self.assertFalse(summary["cpu_allowed"])
        self.assertIsNone(voicelab.cpu_rtf("chatterbox"))
        self.assertTrue(voicelab.cpu_measured("chatterbox"))
        self.assertTrue(benchmark.timed_out_for(
            self.settings, "chatterbox", "cpu"))
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(
                [v for v, _l in voicelab.device_options("chatterbox")],
                ["cuda"],
            )

    def test_the_cpu_is_probed_once_when_it_is_the_device_asked_for(self):
        from ai_voice_studio.voicelab import benchmark

        with self._patched([self._ok(0.9)]):
            benchmark.measure_engine("f5tts", "cpu", settings=self.settings,
                                     voice_entry=self.voice)
        self.assertEqual([w.device for w in self.workers], ["cpu"])
        self.assertAlmostEqual(voicelab.cpu_rtf("f5tts"), 0.9)

    def test_a_probe_that_never_answers_is_stopped(self):
        from ai_voice_studio.voicelab import benchmark

        release = threading.Event()

        def _blocking():
            release.wait(10)
            return self._ok(1.0)

        with self._patched([_blocking]):
            result = benchmark.probe("f5tts", "cpu", voice_entry=self.voice,
                                     timeout=0.2)
        release.set()
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertIn("did not finish", result["error"])
        self.assertTrue(self.workers[0].killed)

    def test_an_engine_that_is_not_installed_cannot_be_measured(self):
        from ai_voice_studio.voicelab import benchmark

        with self._patched([], installed=False):
            with self.assertRaises(benchmark.BenchmarkError) as caught:
                benchmark.measure_engine("f5tts", "cpu", settings=self.settings,
                                         voice_entry=self.voice)
        self.assertIn("not installed", str(caught.exception))

    def test_stored_verdicts_are_loaded_at_start_up(self):
        from ai_voice_studio.voicelab import benchmark

        self.settings.set(benchmark.settings_key("f5tts"),
                          {"cpu": {"rtf": 6.0, "seconds": 18.0,
                                    "audio_seconds": 3.0, "sample_rate": 24000}})
        self.settings.set(benchmark.settings_key("kokoclone"),
                          {"cpu": {"timed_out": True,
                                    "error": "did not finish in 300 seconds"}})
        benchmark.load_verdicts(self.settings)
        self.assertAlmostEqual(voicelab.cpu_rtf("f5tts"), 6.0)
        self.assertIs(voicelab.cpu_verdict("f5tts"), False)
        self.assertTrue(voicelab.cpu_measured("kokoclone"))
        self.assertIsNone(voicelab.cpu_rtf("kokoclone"))
        self.assertIs(voicelab.cpu_verdict("pocket_tts"), None)

    def test_the_summary_reads_as_an_answer(self):
        from ai_voice_studio.voicelab import benchmark

        voicelab.set_cpu_rtf("chatterbox", 5.0)
        text = benchmark.summary_text("chatterbox", {
            "probes": {
                "cuda": self._ok(0.31),
                "cpu": self._ok(5.0),
            }
        })
        self.assertIn("GPU: 0.31x real time", text)
        self.assertIn("CPU: 5.00x real time", text)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            text = benchmark.summary_text("chatterbox", {"probes": {}})
        self.assertIn("not offered", text)


class EngineWiringTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(voicelab.close_workers)

    def test_get_engine_returns_the_voice_lab_wrapper(self):
        from ai_voice_studio.tts.engine import get_engine

        voice = dict(voice_lab.builtin_voices("f5tts")[0])
        voice["engine"] = "f5tts"
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            engine = get_engine(voice, provider="cpu")
        self.assertIsInstance(engine, voicelab.VoicelabEngine)
        self.assertEqual(engine.engine_id, "f5tts")
        self.assertEqual(engine.device, "cpu")

    def test_get_engine_honours_the_gpu_provider(self):
        from ai_voice_studio.tts.engine import get_engine

        voice = dict(voice_lab.builtin_voices("pocket_tts")[0])
        voice["engine"] = "pocket_tts"
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            engine = get_engine(voice, provider="cuda")
            self.assertEqual(engine.device, "cuda")
            # A stored per-voice device wins over the provider.
            voice["device"] = "cpu"
            engine = get_engine(voice, provider="cuda")
        self.assertEqual(engine.device, "cpu")


class TuningOptionsTest(unittest.TestCase):
    """The per-engine tuning vocabulary (``voicelab.options``)."""

    def test_every_engine_offers_tuning(self):
        self.assertEqual(tuning.engine_ids(), voice_lab.engine_ids())
        for engine_id in voice_lab.engine_ids():
            specs = tuning.specs(engine_id)
            self.assertTrue(specs, engine_id)
            keys = [option.key for option in specs]
            self.assertEqual(len(keys), len(set(keys)), engine_id)
            for option in specs:
                self.assertIn(
                    option.kind,
                    (tuning.KIND_INT, tuning.KIND_FLOAT, tuning.KIND_BOOL,
                     tuning.KIND_TEXT, tuning.KIND_CHOICE, tuning.KIND_SEED),
                    option.key,
                )
                self.assertTrue(option.label, option.key)

    def test_numeric_options_have_a_sane_range(self):
        for engine_id in voice_lab.engine_ids():
            for option in tuning.specs(engine_id):
                if option.kind not in (tuning.KIND_INT, tuning.KIND_FLOAT):
                    continue
                self.assertIsNotNone(option.minimum, option.key)
                self.assertIsNotNone(option.maximum, option.key)
                self.assertLess(option.minimum, option.maximum, option.key)
                self.assertGreaterEqual(
                    option.default, option.minimum, option.key
                )
                self.assertLessEqual(option.default, option.maximum, option.key)
                if option.kind == tuning.KIND_CHOICE:
                    self.assertIn(option.default, option.choices, option.key)

    def test_the_engines_documented_defaults_are_used(self):
        f5 = {option.key: option.default for option in tuning.specs("f5tts")}
        self.assertEqual(f5["nfe_step"], 32)
        self.assertAlmostEqual(f5["cfg_strength"], 2.0)
        self.assertAlmostEqual(f5["sway_sampling_coef"], -1.0)
        pocket = {o.key: o.default for o in tuning.specs("pocket_tts")}
        self.assertFalse(pocket["quantize"])
        # The quantized model is a different model: it must reload.
        self.assertTrue(
            tuning.spec("pocket_tts", "quantize").load_time
        )

    def test_defaults_are_never_stored(self):
        for engine_id in voice_lab.engine_ids():
            self.assertEqual(tuning.clean(engine_id, tuning.defaults(engine_id)), {})
            self.assertEqual(tuning.describe_overrides(
                engine_id, tuning.defaults(engine_id)), "")

    def test_clean_keeps_only_real_overrides(self):
        self.assertEqual(
            tuning.clean("f5tts", {"nfe_step": 16, "cfg_strength": 2.0}),
            {"nfe_step": 16},
        )
        # Unknown keys, blanks and non-numbers are dropped or clamped.
        self.assertEqual(
            tuning.clean("f5tts", {"nope": 1, "seed": "", "nfe_step": "999"}),
            {"nfe_step": 64},
        )
        self.assertEqual(tuning.clean("f5tts", {"seed": "7"}), {"seed": 7})
        self.assertEqual(
            tuning.clean("pocket_tts", {"quantize": "true"}),
            {"quantize": True},
        )
        with self.assertRaises(ValueError):
            tuning.clean("f5tts", {"nfe_step": "many"})
        with self.assertRaises(ValueError):
            tuning.clean("f5tts", {"seed": "later"})

    def test_describe_overrides_reads_like_a_sentence_fragment(self):
        text = tuning.describe_overrides(
            "f5tts", {"nfe_step": 16, "cfg_strength": 3.5, "seed": 7}
        )
        self.assertIn("16", text)
        self.assertIn("cfg", text)  # case-insensitive label prefix
        self.assertIn("seed 7", text)
        self.assertIn("quantize",
                      tuning.describe_overrides("pocket_tts", {"quantize": True}))

    def test_resolution_prefers_the_project_over_the_saved_defaults(self):
        self.assertEqual(
            tuning.resolve("f5tts", {"nfe_step": 8}, {"nfe_step": 64}),
            {"nfe_step": 8},
        )
        self.assertEqual(
            tuning.resolve("f5tts", {}, {"nfe_step": 64}), {"nfe_step": 64}
        )
        self.assertEqual(tuning.resolve("f5tts", {}, {}), {})
        self.assertEqual(
            tuning.resolve("f5tts", {"nfe_step": 32}, {"nfe_step": 16}),
            {"nfe_step": 16},
        )

    def test_apply_to_voice_copies_and_never_mutates(self):
        voice = dict(voice_lab.builtin_voices("f5tts")[0])
        original = dict(voice)
        enriched = tuning.apply_to_voice(voice, {"nfe_step": 16})
        self.assertEqual(voice, original, "the original entry must not change")
        self.assertEqual(enriched["options"], {"nfe_step": 16})
        self.assertEqual(enriched["ref_resource"], voice["ref_resource"])
        # Defaults (and an empty set) must not leave an ``options`` key behind.
        self.assertNotIn("options", tuning.apply_to_voice(voice, {"nfe_step": 32}))
        self.assertNotIn("options", tuning.apply_to_voice(enriched, {}))

    def test_problems_are_reported_but_never_fatal(self):
        notes = tuning.problems("pocket_tts", {"quantize": True}, "cuda")
        self.assertTrue(notes)
        self.assertIn("int8", " ".join(notes))
        self.assertEqual(tuning.problems("pocket_tts", {"quantize": True}, "cpu"), [])
        self.assertEqual(tuning.problems("f5tts", {"nfe_step": 16}, "gpu"), [])

    def test_settings_key_is_stable(self):
        self.assertEqual(
            tuning.settings_key("f5tts"), "clone_engines.options.f5tts"
        )

    def test_the_summary_hint_mentions_the_runtime_knob(self):
        self.assertIn(
            "quantiz", tuning.max_seconds_note("pocket_tts").lower()
        )
        self.assertIn("Diffusion", tuning.max_seconds_note("f5tts"))


class TuningRequestTest(unittest.TestCase):
    """The tuning overrides must reach the worker and nothing else."""

    def test_tuning_is_cleaned_before_it_is_sent(self):
        request = voicelab.build_synthesize_request(
            engine_id="f5tts", device="cpu", text="Hi",
            voice_entry=dict(voice_lab.builtin_voices("f5tts")[0]),
            tuning={"nfe_step": 16, "cfg_strength": 2.0, "junk": 1},
        )
        self.assertEqual(request["options"], {"nfe_step": 16})

    def test_no_options_key_without_overrides(self):
        request = voicelab.build_synthesize_request(
            engine_id="f5tts", device="cpu", text="Hi",
            voice_entry=dict(voice_lab.builtin_voices("f5tts")[0]),
            tuning=tuning.defaults("f5tts"),
        )
        self.assertNotIn("options", request)

    def test_a_voice_entry_can_carry_the_overrides(self):
        entry = tuning.apply_to_voice(
            dict(voice_lab.builtin_voices("pocket_tts")[0]),
            {"quantize": True},
        )
        request = voicelab.build_synthesize_request(
            engine_id="pocket_tts", device="cpu", text="Hi", voice_entry=entry,
        )
        self.assertEqual(request["options"], {"quantize": True})

    def test_an_explicit_override_list_beats_the_voice_entry(self):
        entry = tuning.apply_to_voice(
            dict(voice_lab.builtin_voices("f5tts")[0]), {"nfe_step": 16}
        )
        request = voicelab.build_synthesize_request(
            engine_id="f5tts", device="cpu", text="Hi", voice_entry=entry,
            tuning={"cfg_strength": 3.0},
        )
        self.assertEqual(request["options"], {"cfg_strength": 3.0})

    def test_the_engine_wrapper_carries_the_overrides(self):
        self.addCleanup(voicelab.close_workers)
        entry = tuning.apply_to_voice(
            dict(voice_lab.builtin_voices("f5tts")[0]), {"nfe_step": 16}
        )
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            engine = voicelab.VoicelabEngine(entry, device="cpu")
        self.assertEqual(engine.tuning, {"nfe_step": 16})
        captured = {}

        def fake_request(request):
            captured.update(request)
            import base64

            import numpy as np

            return {
                "ok": True,
                "wav": base64.b64encode(
                    np.zeros(4, dtype=np.int16).tobytes()
                ).decode("ascii"),
                "sample_rate": 24000,
                "applied": ["nfe_step"],
                "skipped": [],
            }

        with mock.patch.object(engine._worker, "_ensure_proc"), \
                mock.patch.object(engine._worker, "_request",
                                  side_effect=fake_request):
            engine.synthesize("Hello")
        self.assertEqual(captured["options"], {"nfe_step": 16})

    def test_the_card_peak_the_worker_reports_reaches_the_log(self):
        """Regression: the peak was looked up under a class name that does not
        exist, so every synthesis of a GPU engine failed with a ``NameError``
        *after* the engine had already done the work."""
        self.addCleanup(voicelab.close_workers)
        voicelab.VoicelabEngine._vram_seen.clear()
        voice = dict(voice_lab.builtin_voices("f5tts")[0]) | {"engine": "f5tts"}
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            engine = voicelab.VoicelabEngine(voice, device="cpu")
        payload = base64.b64encode(np.zeros(4, dtype=np.int16).tobytes()).decode(
            "ascii"
        )
        response = {"ok": True, "wav": payload, "sample_rate": 24000,
                    "vram_peak_mb": 7800}
        with mock.patch.object(engine._worker, "_ensure_proc"), \
                mock.patch.object(engine._worker, "_request",
                                  return_value=response):
            with self.assertLogs("ai_voice_studio.voicelab", level="INFO") as logs:
                samples = engine.synthesize("Hello")
                # A second call at the same peak is not worth another line.
                engine.synthesize("Hello")
        self.assertEqual(list(samples), [0, 0, 0, 0])
        mentions = [line for line in logs.output if "video memory" in line]
        self.assertEqual(len(mentions), 1, logs.output)
        self.assertIn("7.6 GB", mentions[0])
        self.assertEqual(voicelab.VoicelabEngine._vram_seen["f5tts"], 7800)

    def test_a_skipped_option_is_only_logged(self):
        with self.assertLogs("ai_voice_studio.voicelab", level="WARNING") as logs:
            voicelab.VoicelabEngine._report_unused_options(
                {"applied": [], "skipped": ["sway_sampling_coef"]}
            )
        self.assertIn("sway_sampling_coef", " ".join(logs.output))
        # A clean report must stay silent.
        voicelab.VoicelabEngine._report_unused_options(
            {"applied": ["nfe_step"], "skipped": []}
        )
        voicelab.VoicelabEngine._report_unused_options(None)


class EngineContractTest(unittest.TestCase):
    """``VoicelabEngine`` must behave like every other TTS engine.

    The worker is stubbed out, so no engine package, GPU or subprocess is
    needed: the contract (request shape, int16 samples, sample rate) is what
    the recording worker and the preview buttons depend on.
    """

    def _engine(self, engine_id="f5tts"):
        self.addCleanup(voicelab.close_workers)
        voice = dict(voice_lab.builtin_voices(engine_id)[0])
        voice["engine"] = engine_id
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            return voicelab.VoicelabEngine(voice, device="cpu")

    @staticmethod
    def _respond(samples):
        import base64

        import numpy as np

        payload = base64.b64encode(np.asarray(samples, dtype=np.int16).tobytes())
        return {
            "ok": True,
            "wav": payload.decode("ascii"),
            "sample_rate": 24000,
            "device_used": "cpu",
        }

    def test_synthesize_sends_the_voice_and_returns_int16(self):
        import numpy as np

        engine = self._engine("f5tts")
        captured = {}

        def fake_request(request):
            captured.update(request)
            return self._respond([0, 1000, -1000, 32767])

        with mock.patch.object(engine._worker, "_ensure_proc"), \
                mock.patch.object(engine._worker, "_request", side_effect=fake_request):
            samples = engine.synthesize("Hello there", speed=1.5)

        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(list(samples), [0, 1000, -1000, 32767])
        self.assertEqual(engine.sample_rate, 24000)
        self.assertEqual(captured["cmd"], "synthesize")
        self.assertEqual(captured["engine"], "f5tts")
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(captured["text"], "Hello there")
        self.assertEqual(captured["speed"], 1.5)
        # The reference voice travels with the request.
        self.assertIn("ref_resource", captured)
        self.assertIsNone(engine._worker._proc, "no worker may be started")

    def test_a_failed_response_becomes_a_voicelab_error(self):
        engine = self._engine("pocket_tts")
        with mock.patch.object(engine._worker, "_ensure_proc"), \
                mock.patch.object(
                    engine._worker, "_request",
                    return_value={"ok": False, "error": "RuntimeError: boom"},
                ), self.assertRaises(voicelab.VoicelabError) as ctx:
            engine.synthesize("Hello")
        self.assertIn("boom", str(ctx.exception))

    def test_nothing_is_synthesized_for_blank_text(self):
        engine = self._engine("f5tts")
        with self.assertRaises(ValueError):
            engine.synthesize("   ")


class WorkerProtocolTest(unittest.TestCase):
    """Drive the real worker script with this interpreter (no engine needed).

    Requests are piped in one batch and the answer is read with
    ``communicate(timeout=...)``: the worker always gets a ``quit``, the pipes
    are closed and the child is reaped, so a wedged worker can never leave a
    test (or the test run) hanging.
    """

    def _run(self, *requests):
        payload = "".join(json.dumps(r) + "\n" for r in requests)
        payload += json.dumps({"cmd": "quit"}) + "\n"
        proc = subprocess.Popen(
            [sys.executable, WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
        )
        try:
            out, err = proc.communicate(payload, timeout=120)
        except subprocess.TimeoutExpired:  # pragma: no cover - failure path
            proc.kill()
            proc.communicate()
            self.fail("the Voice Lab worker did not answer in time")
        self.assertEqual(proc.returncode, 0, err)
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    def test_ping_reports_the_four_engines(self):
        response, = self._run({"cmd": "ping"})
        self.assertTrue(response["ok"])
        self.assertEqual(
            sorted(response["engines"]),
            sorted(voice_lab.engine_ids()),
        )
        self.assertTrue(all(isinstance(v, bool)
                           for v in response["engines"].values()))

    def test_unknown_engine_reports_a_readable_error(self):
        response, = self._run({
            "cmd": "synthesize", "engine": "nope", "device": "cpu", "text": "Hi",
        })
        self.assertFalse(response["ok"])
        self.assertIn("Unknown Voice Lab engine", response["error"])

    def test_empty_text_is_rejected(self):
        response, = self._run({
            "cmd": "synthesize", "engine": "f5tts", "device": "cpu", "text": "  ",
        })
        self.assertFalse(response["ok"])
        self.assertIn("Nothing to synthesize", response["error"])

    def test_an_engine_that_is_not_installed_says_so(self):
        responses = self._run(
            {"cmd": "ping"},
            {"cmd": "voices", "engine": "nope"},
        )
        self.assertTrue(responses[0]["ok"])
        self.assertFalse(responses[1]["ok"])
        self.assertIn("Unknown engine", responses[1]["error"])

    def test_quit_stops_the_worker(self):
        # ``_run`` appends the quit and asserts the process exits cleanly.
        self.assertEqual(self._run({"cmd": "ping"})[0]["ok"], True)


class KokoCloneBackendTest(unittest.TestCase):
    """KokoClone's worker backend, driven with a fake engine.

    The engine's own packages are gigabytes of models and are not installed
    here, so ``core.cloner.KokoClone`` is replaced by a stand-in that records
    how it was called.  What is checked is everything the application is
    responsible for: which call is made for a built-in voice and which for a
    clone, the language that travels with it, the working directory the engine
    needs and the per-voice availability answer.
    """

    class _FakeKokoro:
        def __init__(self):
            self.calls = []

        def create(self, text, voice=None, speed=1.0, is_phonemes=False,
                   lang=None):
            self.calls.append({"text": text, "voice": voice, "speed": speed,
                               "is_phonemes": is_phonemes, "lang": lang})
            return np.zeros(240, dtype=np.float32), 24000

    class _FakeCloner:
        device = None

        def __init__(self, sample_rate=24000):
            self.sample_rate = sample_rate
            self.router_calls = []
            self.generate_calls = []
            self.kokoro = KokoCloneBackendTest._FakeKokoro()
            self.kokoro_model = None

        # -- the engine's own language routing ------------------------------
        def _get_config(self, lang):
            self.router_calls.append(lang)
            return (f"model-{lang}.onnx", "voices.bin", None,
                    _G2P(), f"default_{lang}", None)

        def _patch_kokoro_compat(self, kokoro):
            return kokoro

        # -- the engine's own text-to-cloned-speech call --------------------
        def generate(self, text, lang, reference_audio, output_path="out.wav"):
            self.generate_calls.append(
                {"text": text, "lang": lang, "ref": reference_audio,
                 "out": output_path}
            )
            _write_silence(output_path, frames=240)

    def setUp(self):
        self.worker = _load_worker()
        # The engine compatibility repairs are once-per-process, so every test
        # starts where a fresh worker process would: unpatched.
        self.worker._MISAKI_G2P_PATCHED = False
        self.worker._CLONER_ROUTER_PATCHED = False
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # ``KokoCloneBackend`` moves the worker into its own folder: put the
        # test process back where it was when the test ends.
        here = os.getcwd()
        self.addCleanup(os.chdir, here)

    def _backend(self, device="cpu", **request):
        payload = {"workspace": self.tmp.name, "text": "Hello"}
        payload.update(request)
        return self.worker.KokoCloneBackend(device, payload)

    def _patch_import(self, backend, cloner):
        backend._cloner = cloner
        backend._kokoro_for = lambda *_a, **_k: cloner.kokoro
        return backend

    # -- language -----------------------------------------------------------
    def test_the_language_comes_from_the_voice_entry(self):
        language = self.worker._kokoclone_language
        self.assertEqual(language("en"), "en")
        self.assertEqual(language("hi"), "hi")
        self.assertEqual(language("zh"), "zh")
        # Long forms, casing and the Japanese alias all fold onto a code the
        # engine knows; anything else falls back to English rather than
        # failing a whole recording.
        self.assertEqual(language("pt-BR"), "pt")
        self.assertEqual(language("JP"), "ja")
        self.assertEqual(language("ja"), "ja")
        self.assertEqual(language("klingon"), "en")
        self.assertEqual(language(None), "en")

    # -- working directory --------------------------------------------------
    def test_the_engine_runs_in_a_folder_it_may_write_to(self):
        backend = self._backend(device="cpu")
        self.assertTrue(os.path.isdir(backend.workspace))
        self.assertTrue(backend.workspace.startswith(self.tmp.name))
        self.assertEqual(os.path.abspath(os.getcwd()),
                         os.path.abspath(backend.workspace))
        # Each device gets its own folder, so a CPU and a GPU worker running
        # at the same time never download the model over each other.
        other = self._backend(device="cuda")
        self.assertNotEqual(other.workspace, backend.workspace)

    # -- built-in voices ----------------------------------------------------
    def test_a_builtin_voice_asks_the_engine_for_the_language_model(self):
        backend = self._patch_import(self._backend(), self._FakeCloner())
        audio, rate = backend._speak_builtin("Hello there", "hi", "hf_alpha")
        self.assertEqual(len(audio), 240)
        self.assertEqual(rate, 24000)
        self.assertEqual(backend._cloner.router_calls, ["hi"])
        call, = backend._cloner.kokoro.calls
        self.assertEqual(call["voice"], "hf_alpha")
        self.assertTrue(call["is_phonemes"])
        # The phonemiser already produced the phonemes, so Kokoro is not asked
        # to phonemise anything itself.
        self.assertIsNone(call["lang"])
        self.assertEqual(backend._cloner.kokoro.calls[0]["text"],
                         "phonemes:Hello there")

    def test_a_builtin_english_voice_uses_kokoro_directly(self):
        class _NoG2P(self._FakeCloner):
            def _get_config(self, lang):
                self.router_calls.append(lang)
                return f"model-{lang}.onnx", "voices.bin", None, None, "af_bella", None

        backend = self._patch_import(self._backend(), _NoG2P())
        backend._speak_builtin("Hello", "en", "af_bella")
        call, = backend._cloner.kokoro.calls
        self.assertFalse(call["is_phonemes"])
        self.assertEqual(call["lang"], "en-us")
        self.assertEqual(call["voice"], "af_bella")

    def test_a_phonemiser_that_returns_a_bare_string_still_speaks(self):
        """Regression: current ``misaki`` returns the phonemes on their own.

        Both KokoClone's own ``generate`` and this worker unpack the
        phonemiser's answer as ``(phonemes, tokens)``; a bare string raised
        ``ValueError: too many values to unpack (expected 2)`` for every
        eSpeak language (Hindi, French, ...), while English - which has no
        phonemiser - kept working.  That is the "first preview plays, the
        next one fails" report.
        """
        class _BareStringCloner(self._FakeCloner):
            def _get_config(self, lang):
                self.router_calls.append(lang)
                return (f"model-{lang}.onnx", "voices.bin", None,
                        _BareG2P(), f"default_{lang}", None)

        backend = self._patch_import(self._backend(), _BareStringCloner())
        audio, rate = backend._speak_builtin("नमस्ते", "hi", "hf_alpha")
        call, = backend._cloner.kokoro.calls
        self.assertTrue(call["is_phonemes"])
        self.assertEqual(call["text"], "phonemes:नमस्ते")
        self.assertEqual(np.asarray(audio).size, 240)
        self.assertEqual(rate, 24000)

    def test_the_engine_phonemiser_is_wrapped_to_return_a_pair(self):
        """The installed ``misaki`` class is repaired where it is imported.

        KokoClone's ``core.cloner`` binds ``EspeakG2P`` at import time, so the
        class itself - not a copy in this module - has to gain the pair.
        """
        espeak, misaki = _fake_misaki_espeak(returns_pair=False)
        with mock.patch.dict(sys.modules, {
            "misaki": misaki, "misaki.espeak": espeak, "core.cloner": None,
        }):
            self.worker._ensure_misaki_g2p_compat()
            phonemes, tokens = misaki.espeak.EspeakG2P()("Hello")
        self.assertEqual(phonemes, "phonemes:Hello")
        self.assertIsNone(tokens)

    def test_the_phonemiser_shim_leaves_a_pair_returning_g2p_alone(self):
        espeak, misaki = _fake_misaki_espeak(returns_pair=True)
        with mock.patch.dict(sys.modules, {
            "misaki": misaki, "misaki.espeak": espeak, "core.cloner": None,
        }):
            self.worker._ensure_misaki_g2p_compat()
            result = misaki.espeak.EspeakG2P()("Hi")
        self.assertEqual(result, ("phonemes:Hi", 2))

    def test_the_engines_language_router_hands_out_a_pair(self):
        """A phonemiser built inside ``_get_config`` escapes a class patch.

        KokoClone's Mandarin front-end is exactly that - a closure over its
        own phonemiser - and the engine's ``generate`` unpacks whatever the
        router hands out as ``(phonemes, tokens)``, so the router itself has
        to be repaired before a Mandarin voice is cloned.
        """
        espeak, misaki = _fake_misaki_espeak(returns_pair=False)
        cloner = _fake_cloner_module(_BareG2P())
        with mock.patch.dict(sys.modules, {
            "misaki": misaki, "misaki.espeak": espeak, "core.cloner": cloner,
        }):
            self.worker._ensure_misaki_g2p_compat()
            # ... as the engine's own ``generate`` unpacks it.
            phonemes, tokens = cloner.KokoClone()._get_config("zh")[3]("\u4f60\u597d")
        self.assertEqual(phonemes, "phonemes:\u4f60\u597d")
        self.assertIsNone(tokens)

    def test_a_changed_engine_says_so_instead_of_crashing(self):
        class _Changed(self._FakeCloner):
            _get_config = None

        backend = self._patch_import(self._backend(), _Changed())
        with self.assertRaises(RuntimeError) as caught:
            backend._speak_builtin("Hello", "en", "af_bella")
        self.assertIn("cloned voice", str(caught.exception))

    # -- cloning ------------------------------------------------------------
    def test_a_clone_is_spoken_by_the_engine_with_the_reference(self):
        reference = _write_silence(os.path.join(self.tmp.name, "reference.wav"))
        backend = self._patch_import(self._backend(), self._FakeCloner())
        audio, rate = backend._speak_clone("Hi", "fr", reference)
        call, = backend._cloner.generate_calls
        self.assertEqual(call["text"], "Hi")
        self.assertEqual(call["lang"], "fr")
        self.assertEqual(call["ref"], reference)
        # The temporary output is read back and removed again.
        self.assertFalse(os.path.exists(call["out"]))
        self.assertEqual(rate, 24000)
        self.assertEqual(np.asarray(audio).size, 240)

    def test_a_missing_reference_is_reported_clearly(self):
        backend = self._patch_import(self._backend(), self._FakeCloner())
        with self.assertRaises(FileNotFoundError) as caught:
            backend._speak_clone("Hi", "en", os.path.join(self.tmp.name, "nope.wav"))
        self.assertIn("reference recording", str(caught.exception))

    def test_a_cloned_voice_picks_the_clone_path_and_a_builtin_the_other(self):
        reference = _write_silence(os.path.join(self.tmp.name, "reference.wav"))
        cloner = self._FakeCloner()
        backend = self._patch_import(self._backend(), cloner)
        backend.synthesize("Clone me", {"language": "en", "voice": "__clone__",
                                         "ref_audio": reference})
        self.assertEqual(len(cloner.generate_calls), 1)
        self.assertEqual(cloner.kokoro.calls, [])
        backend.synthesize("Built-in", {"language": "en", "voice": "af_bella"})
        self.assertEqual(len(cloner.kokoro.calls), 1)
        # ... and neither call resamples the audio (the engine owns the speed).
        self.assertEqual(backend.sample_rate, 24000)

    # -- eSpeak NG ----------------------------------------------------------
    def _fake_phonemizer(self):
        """A stand-in for the engine's phonemiser stack.

        ``espeakng_loader`` (which carries the eSpeak NG shared library and its
        data) and phonemizer's ``EspeakWrapper`` are the two things the worker
        talks to; both are faked here so the bootstrap can be driven without
        either package installed.
        """
        site = os.path.join(self.tmp.name, "site-packages")
        data = os.path.join(site, "espeak-ng-data")
        os.makedirs(data, exist_ok=True)
        library = os.path.join(site, "espeak-ng.dll")
        with open(library, "wb") as handle:
            handle.write(b"MZ")
        loader = types.ModuleType("espeakng_loader")
        loader.get_library_path = lambda: library
        loader.get_data_path = lambda: data
        wrapper = types.SimpleNamespace(
            set_library=mock.MagicMock(),
            set_data_path=mock.MagicMock(),
        )
        wrapper_module = types.ModuleType("phonemizer.backend.espeak.wrapper")
        wrapper_module.EspeakWrapper = wrapper
        modules = {
            "espeakng_loader": loader,
            "phonemizer": types.ModuleType("phonemizer"),
            "phonemizer.backend": types.ModuleType("phonemizer.backend"),
            "phonemizer.backend.espeak": types.ModuleType(
                "phonemizer.backend.espeak"
            ),
            "phonemizer.backend.espeak.wrapper": wrapper_module,
        }
        return library, data, wrapper, modules

    def test_the_phonemiser_is_pointed_at_the_bundled_espeak(self):
        """Cloning must not need an eSpeak NG installed on the computer.

        KokoClone phonemises every language through ``misaki.espeak``, which
        only ever looks at a hard-coded system path - the copy inside the
        ``espeakng_loader`` wheel is what this application installs, so the
        worker has to hand phonemizer its path.  Without this the first clone
        fails with "RuntimeError: espeak not installed on your system".
        """
        library, data, wrapper, modules = self._fake_phonemizer()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.worker._prepare_espeak_phonemizer())
            wrapper.set_library.assert_called_once_with(library)
            wrapper.set_data_path.assert_called_once_with(data)
            # kokoro-onnx (the engine's Kokoro stage) reads the variables.
            self.assertEqual(os.environ["PHONEMIZER_ESPEAK_LIBRARY"], library)
            self.assertEqual(os.environ["PHONEMIZER_ESPEAK_DATA_PATH"], data)

    def test_an_espeak_the_user_pointed_at_is_left_alone(self):
        library, _data, wrapper, modules = self._fake_phonemizer()
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(
            os.environ, {"PHONEMIZER_ESPEAK_LIBRARY": library}
        ):
            self.assertIsNone(self.worker._prepare_espeak_phonemizer())
            self.assertEqual(os.environ["PHONEMIZER_ESPEAK_LIBRARY"], library)
        wrapper.set_library.assert_not_called()

    def test_a_missing_espeak_is_explained_instead_of_failing_later(self):
        with mock.patch.dict(sys.modules, {
            "espeakng_loader": None,
            "phonemizer.backend.espeak.wrapper": None,
        }), mock.patch.dict(os.environ, {}, clear=True):
            note = self.worker._prepare_espeak_phonemizer()
        self.assertIn("eSpeak NG", note)

    def test_the_backend_prepares_espeak_before_the_engine_is_imported(self):
        """The bootstrap runs for every request, not only the first clone."""
        with mock.patch.object(self.worker, "_prepare_espeak_phonemizer",
                               return_value=None) as prepare:
            self._backend()
        prepare.assert_called_once_with()

    def test_a_missing_espeak_reaches_the_user_with_the_response(self):
        with mock.patch.object(self.worker, "_prepare_espeak_phonemizer",
                               return_value="install eSpeak NG"):
            backend = self._backend()
        self.assertEqual(backend.notes, ["install eSpeak NG"])

    # -- availability -------------------------------------------------------
    def test_every_builtin_voice_is_probed_for_its_modules(self):
        backend = self._backend()
        with mock.patch.object(self.worker, "_module_available",
                               return_value=True):
            available = backend.available_voices()
        self.assertEqual(sorted(available), sorted(v for v, _l in
                                                   self.worker._KOKOCLONE_VOICES))
        self.assertTrue(all(available.values()))

    def test_a_voice_whose_extra_is_missing_is_reported_unavailable(self):
        backend = self._backend()
        missing = set(self.worker._KOKOCLONE_LANGUAGE_MODULES["zh"])
        with mock.patch.object(
            self.worker, "_module_available",
            side_effect=lambda name: name not in missing,
        ):
            available = backend.available_voices()
        self.assertFalse(available["zf_001"])
        self.assertTrue(available["af_bella"])

    def test_an_engine_without_its_own_modules_reports_nothing_usable(self):
        backend = self._backend()
        base = set(self.worker._KOKOCLONE_BASE_MODULES)
        with mock.patch.object(
            self.worker, "_module_available",
            side_effect=lambda name: name not in base,
        ):
            self.assertFalse(any(backend.available_voices().values()))


class _G2P:
    """Stand-in for the engine's phonemiser callable."""

    def __call__(self, text):
        return f"phonemes:{text}", None


class _BareG2P:
    """A phonemiser that returns the phoneme string on its own.

    This is what the current ``misaki``'s ``EspeakG2P.__call__`` does.
    """

    def __call__(self, text):
        return f"phonemes:{text}"


def _fake_cloner_module(phonemiser):
    """A fake ``core.cloner`` whose router always hands out ``phonemiser``."""
    module = types.ModuleType("core.cloner")

    class KokoClone:
        def _get_config(self, language):
            return (f"model-{language}.onnx", "voices.bin", None,
                    phonemiser, f"default_{language}", None)

    module.KokoClone = KokoClone
    return module


def _fake_misaki_espeak(returns_pair):
    """A fake ``misaki`` package holding an ``EspeakG2P`` stand-in.

    Returns ``(espeak_module, misaki_module)`` so a test can put both in
    ``sys.modules`` and let the worker's own ``from misaki.espeak import ...``
    find them.
    """
    espeak = types.ModuleType("misaki.espeak")

    class _EspeakG2P:
        def __call__(self, text):
            if returns_pair:
                return f"phonemes:{text}", 2
            return f"phonemes:{text}"

    espeak.EspeakG2P = _EspeakG2P
    misaki = types.ModuleType("misaki")
    misaki.__path__ = []
    misaki.espeak = espeak
    return espeak, misaki


class WorkerTuningTest(unittest.TestCase):
    """How the worker applies and reports the tuning options."""

    def setUp(self):
        self.worker = _load_worker()

    def test_option_report_deduplicates(self):
        report = self.worker.OptionReport()
        report.mark("seed")
        report.record(applied=["nfe_step"], skipped=["sway_sampling_coef"])
        report.record(applied=["nfe_step"], skipped=["sway_sampling_coef", "x"])
        self.assertEqual(report.as_dict(), {
            "applied": ["seed", "nfe_step"],
            "skipped": ["sway_sampling_coef", "x"],
        })
        report.reset()
        self.assertEqual(report.as_dict(), {"applied": [], "skipped": []})

    def test_seeding(self):
        self.assertTrue(self.worker._apply_seed(7))
        self.assertTrue(self.worker._apply_seed("7"))
        self.assertFalse(self.worker._apply_seed(None))
        self.assertFalse(self.worker._apply_seed(""))
        self.assertFalse(self.worker._apply_seed("later"))
        # Two draws with the same seed must agree.
        import numpy as np

        self.worker._apply_seed(11)
        first = np.random.random()
        self.worker._apply_seed(11)
        self.assertEqual(first, np.random.random())

    def test_the_backend_cache_key_follows_the_loaded_model(self):
        key = self.worker._backend_key
        self.assertEqual(
            key("qwen3tts", "cuda", {"variant": "qwen3_custom_1_7b"}),
            key("qwen3tts", "cuda", {"variant": "qwen3_custom_1_7b"}),
        )
        self.assertNotEqual(
            key("qwen3tts", "cuda", {"variant": "qwen3_custom_1_7b"}),
            key("qwen3tts", "cuda", {"variant": "qwen3_custom_0_6b"}),
        )
        self.assertNotEqual(
            key("pocket_tts", "cpu", {"options": {"quantize": True}}),
            key("pocket_tts", "cpu", {"options": {}}),
        )
        self.assertNotEqual(
            key("pocket_tts", "cpu", {"language": "fr"}),
            key("pocket_tts", "cpu", {"language": "en"}),
        )
        self.assertNotEqual(
            key("f5tts", "cpu", {}), key("f5tts", "cuda", {})
        )

    def test_backends_only_forward_supported_options(self):
        base = self.worker.Backend

        class Fake(base):
            own_options = ("quantize",)

            @staticmethod
            def infer(nfe_step=32, cfg_strength=2.0):
                return nfe_step, cfg_strength

        request = {
            "options": {
                "nfe_step": 16,          # supported
                "quantize": True,        # handled by the backend itself
                "sway_sampling_coef": -1.0,  # not in this fake signature
            }
        }
        backend = Fake("cpu", request)
        kwargs = backend.option_kwargs(Fake.infer)
        self.assertEqual(kwargs, {"nfe_step": 16})
        self.assertEqual(backend.report.as_dict()["applied"], ["nfe_step"])
        self.assertEqual(
            backend.report.as_dict()["skipped"], ["sway_sampling_coef"]
        )

    def test_a_backend_without_own_options_gets_everything(self):
        backend = self.worker.Backend("cpu", {"options": {"nfe_step": 16}})
        self.assertEqual(backend.request_options(), {"nfe_step": 16})
        # A None / non-dict options payload must not crash the worker.
        self.assertEqual(
            self.worker.Backend("cpu", {"options": None}).request_options(), {}
        )
        self.assertEqual(
            self.worker.Backend("cpu", {"options": [1]}).request_options(), {}
        )

    def test_synthesize_reports_the_options_over_the_protocol(self):
        """End to end through ``main()`` with a fake engine registered."""
        import contextlib
        import io

        worker = self.worker

        class Fake(worker.Backend):
            name = "fake"
            sample_rate = 16000

            @staticmethod
            def infer(nfe_step=32, cfg_strength=2.0):
                return nfe_step

            def synthesize(self, text, request):
                self.option_kwargs(Fake.infer)
                return worker.np.zeros(8, dtype=worker.np.int16), 16000, False

        request = {
            "cmd": "synthesize",
            "engine": "fake",
            "device": "cpu",
            "text": "Hello",
            "options": {
                "nfe_step": 16,
                "waveform_temp": 0.5,  # the fake engine has no such argument
                "seed": 7,
            },
        }
        out = io.StringIO()
        with mock.patch.dict(worker._BACKENDS, {"fake": Fake}), \
                mock.patch.object(worker, "_protocol_stream", lambda: out), \
                mock.patch.object(sys, "stdin",
                                  io.StringIO(json.dumps(request) + "\n")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(worker.main(), 0)
        response = json.loads(out.getvalue().strip().splitlines()[0])
        self.assertTrue(response["ok"], response)
        self.assertIn("nfe_step", response["applied"])
        self.assertIn("seed", response["applied"])
        self.assertEqual(response["skipped"], ["waveform_temp"])
        self.assertEqual(response["sample_rate"], 16000)
        self.assertEqual(response["device_used"], "cpu")


    def test_benchmark_times_only_the_second_call(self):
        """The speed probe speaks a warm-up first, and times the sentence.

        The first call is what loads the model (and downloads its weights on
        a first run), so timing it would report the download rather than the
        engine's speed.
        """
        worker = self.worker
        calls = []

        class Fake(worker.Backend):
            name = "fake"
            sample_rate = 1000

            def synthesize(self, text, request):
                calls.append(text)
                return worker.np.zeros(1000, dtype=worker.np.int16), 1000, False

        request = {"cmd": "benchmark", "engine": "fake", "device": "cpu",
                   "text": "Measure this sentence."}
        out = io.StringIO()
        with mock.patch.dict(worker._BACKENDS, {"fake": Fake}), \
                mock.patch.object(worker, "_protocol_stream", lambda: out), \
                mock.patch.object(sys, "stdin",
                                  io.StringIO(json.dumps(request) + "\n")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(worker.main(), 0)
        response = json.loads(out.getvalue().strip().splitlines()[0])
        self.assertTrue(response["ok"], response)
        self.assertEqual(calls,
                         [worker._BENCHMARK_WARMUP_TEXT,
                          "Measure this sentence."])
        # One second of audio (1000 samples at 1 kHz), so the factor is the
        # seconds of generation per second of audio and nothing else.
        self.assertAlmostEqual(response["audio_seconds"], 1.0, places=3)
        self.assertAlmostEqual(
            response["rtf"],
            response["seconds"] / response["audio_seconds"], places=6,
        )
        self.assertEqual(response["sample_rate"], 1000)
        self.assertEqual(response["device_used"], "cpu")

    def test_benchmark_uses_its_own_text_when_none_is_sent(self):
        worker = self.worker
        calls = []

        class Fake(worker.Backend):
            def synthesize(self, text, request):
                calls.append(text)
                return worker.np.zeros(4, dtype=worker.np.int16), 8000, False

        request = {"cmd": "benchmark", "engine": "fake", "device": "cpu"}
        out = io.StringIO()
        with mock.patch.dict(worker._BACKENDS, {"fake": Fake}), \
                mock.patch.object(worker, "_protocol_stream", lambda: out), \
                mock.patch.object(sys, "stdin",
                                  io.StringIO(json.dumps(request) + "\n")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(worker.main(), 0)
        response = json.loads(out.getvalue().strip().splitlines()[0])
        self.assertTrue(response["ok"], response)
        self.assertEqual(calls[-1], worker._BENCHMARK_TEXT)
        self.assertEqual(response["text"], worker._BENCHMARK_TEXT)

    def test_a_benchmark_that_fails_reports_it_readably(self):
        worker = self.worker

        class Fake(worker.Backend):
            def synthesize(self, text, request):
                raise RuntimeError("the model could not be loaded")

        request = {"cmd": "benchmark", "engine": "fake", "device": "cpu"}
        out = io.StringIO()
        with mock.patch.dict(worker._BACKENDS, {"fake": Fake}), \
                mock.patch.object(worker, "_protocol_stream", lambda: out), \
                mock.patch.object(sys, "stdin",
                                  io.StringIO(json.dumps(request) + "\n")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(worker.main(), 0)
        response = json.loads(out.getvalue().strip().splitlines()[0])
        self.assertFalse(response["ok"])
        self.assertIn("the model could not be loaded", response["error"])


class WorkerHelpersTest(unittest.TestCase):
    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("aivs_voicelab_worker", WORKER)
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

    def test_int16_conversion(self):
        np = self.worker.np
        samples = self.worker._to_int16(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32))
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(list(samples), [0, 32767, -32767, 32767])
        already = np.array([1, 2, 3], dtype=np.int16)
        self.assertEqual(list(self.worker._to_int16(already)), [1, 2, 3])

    def test_int16_conversion_widens_a_bfloat16_tensor(self):
        """VibeVoice returns a bfloat16 waveform, which numpy cannot hold at
        all - ``.numpy()`` raises "Got unsupported ScalarType BFloat16" - so
        the conversion has to widen it to float32 first."""
        np = self.worker.np

        class _BfloatTensor:
            """A torch-like tensor whose dtype only numpy can't handle."""

            def __init__(self, values):
                self._values = list(values)
                self.widened = False

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                self.widened = True
                tensor = self

                class _Widened:
                    def numpy(inner):
                        return np.array(tensor._values, dtype=np.float32)

                return _Widened()

            def numpy(self):
                raise TypeError("Got unsupported ScalarType BFloat16")

        tensor = _BfloatTensor([0.0, 1.0, -1.0])
        samples = self.worker._to_int16(tensor)
        self.assertEqual(samples.dtype, np.int16)
        self.assertEqual(list(samples), [0, 32767, -32767])
        self.assertTrue(tensor.widened)

    def test_speed_resampling_changes_the_length(self):
        np = self.worker.np
        samples = np.arange(1000, dtype=np.int16)
        faster = self.worker._resample_speed(samples, 2.0)
        self.assertLess(len(faster), len(samples))
        self.assertEqual(len(self.worker._resample_speed(samples, 1.0)), 1000)

    def test_argument_aliasing_follows_the_signature(self):
        def old_api(ref_audio, gen_text, speed=1.0):
            return ref_audio, gen_text, speed

        def new_api(ref_file, ref_text, gen_text, nfe_step=32):
            return ref_file, ref_text, gen_text, nfe_step

        self.assertEqual(
            self.worker._pick_param(old_api, ("ref_file", "ref_audio")), "ref_audio"
        )
        self.assertEqual(
            self.worker._pick_param(new_api, ("ref_file", "ref_audio")), "ref_file"
        )
        supported, skipped = self.worker._supported_kwargs(
            new_api, {"ref_file": "a", "ref_audio": "b"}
        )
        self.assertEqual(supported, {"ref_file": "a"})
        self.assertEqual(skipped, ["ref_audio"])
        self.assertTrue(self.worker._accepts(new_api, "gen_text"))
        self.assertFalse(self.worker._accepts(new_api, "progress"))

    def test_resource_resolution_is_forgiving(self):
        self.assertIsNone(self.worker._resolve_resource(""))
        self.assertIsNone(self.worker._resolve_resource("no_such_pkg:file.wav"))
        self.assertEqual(self.worker._read_text_resource(""), "")


class WorkerComputeSettingsTest(unittest.TestCase):
    """The worker applies the application's compute choice.

    GPU: use the whole available card.  CPU: use the share the application
    computed (80-95% of the logical CPUs) - so both sides always agree.
    """

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("aivs_voicelab_worker", WORKER)
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

    def test_the_cpu_budget_is_applied_to_torch(self):
        seen = {}

        class FakeTorch:
            @staticmethod
            def set_num_threads(value):
                seen["threads"] = value

            @staticmethod
            def set_num_interop_threads(value):
                seen["interop"] = value

        self.assertTrue(self.worker._limit_cpu(FakeTorch, 7))
        self.assertEqual(seen, {"threads": 7, "interop": 3})
        # No budget, or a nonsense one: leave torch alone.
        self.assertFalse(self.worker._limit_cpu(FakeTorch, None))
        self.assertFalse(self.worker._limit_cpu(FakeTorch, 0))
        self.assertFalse(self.worker._limit_cpu(FakeTorch, "many"))

    def test_a_gpu_request_maximizes_the_card(self):
        seen = {}

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def set_device(index):
                seen["device"] = index

        class FakeTorch:
            cuda = FakeCuda
            backends = types.SimpleNamespace(
                cudnn=types.SimpleNamespace(benchmark=False, allow_tf32=False),
                cuda=types.SimpleNamespace(
                    matmul=types.SimpleNamespace(allow_tf32=False)
                ),
            )

            @staticmethod
            def set_float32_matmul_precision(value):
                seen["precision"] = value

            @staticmethod
            def set_grad_enabled(value):
                seen["grad"] = value

        self.assertTrue(self.worker._maximize_cuda(FakeTorch))
        # Whichever card is there is used - no model / VRAM requirement.
        self.assertEqual(seen["device"], 0)
        self.assertEqual(seen["precision"], "high")
        self.assertFalse(seen["grad"])
        self.assertTrue(FakeTorch.backends.cudnn.benchmark)
        self.assertTrue(FakeTorch.backends.cuda.matmul.allow_tf32)

    def test_a_gpu_request_lets_torch_pick_the_fastest_attention(self):
        """An autoregressive engine spends most of its time in attention, so a
        slow kernel is what leaves a card two thirds idle."""
        seen = []

        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def set_device(index):
                    pass

            class backends:
                class cudnn:
                    benchmark = False
                    allow_tf32 = False

                class cuda:
                    class matmul:
                        allow_tf32 = False

                    @staticmethod
                    def enable_flash_sdp(value):
                        seen.append(("flash", value))

                    @staticmethod
                    def enable_mem_efficient_sdp(value):
                        seen.append(("memory efficient", value))

                    @staticmethod
                    def enable_math_sdp(value):
                        seen.append(("math", value))

            @staticmethod
            def set_float32_matmul_precision(value):
                pass

            @staticmethod
            def set_grad_enabled(value):
                pass

        self.assertTrue(self.worker._maximize_cuda(FakeTorch))
        self.assertEqual(seen, [("flash", True), ("memory efficient", True),
                                ("math", True)])

    def test_a_torch_without_those_knobs_is_not_an_error(self):
        class Older:
            class cuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def set_device(index):
                    pass

            class backends:
                class cudnn:
                    benchmark = False
                    allow_tf32 = False

                class cuda:
                    class matmul:
                        allow_tf32 = False

            @staticmethod
            def set_float32_matmul_precision(value):
                pass

            @staticmethod
            def set_grad_enabled(value):
                pass

        self.assertTrue(self.worker._maximize_cuda(Older))

    def test_the_cuda_allocator_is_configured_before_any_engine_loads_torch(self):
        """The allocator reads this when it first touches the card, which is
        after the worker module is imported but before an engine loads torch.

        Without it, a card that has just run a long segment fragments into
        holes too small for the next model copy or batch, and the engine says
        "out of memory" where it could have used the whole card.
        """
        code = (
            "import importlib.util, os, sys;\n"
            "spec = importlib.util.spec_from_file_location('probe', sys.argv[1]);\n"
            "module = importlib.util.module_from_spec(spec);\n"
            "spec.loader.exec_module(module);\n"
            "print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>'))\n"
        )
        env = {key: value for key, value in os.environ.items()
               if key != "PYTORCH_CUDA_ALLOC_CONF"}
        done = subprocess.run(
            [sys.executable, "-c", code, WORKER], capture_output=True,
            text=True, env=env, timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("expandable_segments:True", done.stdout)

    def test_a_chosen_allocator_setting_is_left_alone(self):
        code = (
            "import importlib.util, os, sys;\n"
            "spec = importlib.util.spec_from_file_location('probe', sys.argv[1]);\n"
            "module = importlib.util.module_from_spec(spec);\n"
            "spec.loader.exec_module(module);\n"
            "print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>'))\n"
        )
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512")
        done = subprocess.run(
            [sys.executable, "-c", code, WORKER], capture_output=True,
            text=True, env=env, timeout=120,
        )
        self.assertIn("max_split_size_mb:512", done.stdout)

    def test_a_machine_without_cuda_stays_on_the_cpu(self):
        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available():
                    return False

        self.assertFalse(self.worker._maximize_cuda(FakeTorch))

    def test_the_chosen_backend_decides_what_is_applied(self):
        calls = []
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda index: calls.append(("device", index)),
        )
        fake.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(benchmark=False, allow_tf32=False),
            cuda=types.SimpleNamespace(
                matmul=types.SimpleNamespace(allow_tf32=False)
            ),
        )
        fake.set_num_threads = lambda value: calls.append(("threads", value))
        fake.set_num_interop_threads = lambda value: calls.append(("interop", value))
        fake.set_float32_matmul_precision = (
            lambda value: calls.append(("precision", value))
        )
        fake.set_grad_enabled = lambda value: calls.append(("grad", value))
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.worker._apply_compute_settings("cpu", 6)
            self.assertIn(("threads", 6), calls)
            self.assertNotIn("device", [name for name, _ in calls])
        calls.clear()
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.worker._apply_compute_settings("cuda", 6)
        self.assertIn(("device", 0), calls)
        self.assertIn(("precision", "high"), calls)
        self.assertNotIn(("threads", 6), calls)


class WorkerEnvironmentTest(unittest.TestCase):
    """Which interpreter the worker is started with.

    Regression: the choice used to be made with the *cached* package probe,
    which answers "not installed" until its background probe has finished.  On
    the first preview after an install - exactly when the cache was just
    invalidated - the worker was therefore started in the shared addon
    environment and F5-TTS died with "ModuleNotFoundError: No module named
    'f5_tts'", a module its own environment does have.
    """

    def setUp(self):
        from ai_voice_studio import python_runtime as pr
        from ai_voice_studio import venv_packages as vp

        self.pr = pr
        self.vp = vp
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        #: Distinguishes the environments of several workers in one test.
        self._count = 0
        with vp._lock:
            vp._cache.clear()
            vp._disk_cache.clear()
            vp._listeners.clear()
            vp._loading.clear()
        self.addCleanup(self._clear)

    def _clear(self):
        with self.vp._lock:
            self.vp._cache.clear()
            self.vp._disk_cache.clear()
            self.vp._listeners.clear()
            self.vp._loading.clear()

    def _env(self, name: str, packages, created: bool = True):
        """A virtualenv on disk holding ``packages`` (as site-packages dirs)."""
        env_dir = os.path.join(self.tmp.name, name)
        if not created:
            return self.pr.PythonRuntime(env_dir)
        site = os.path.join(env_dir, "Lib", "site-packages")
        os.makedirs(site)
        os.makedirs(os.path.join(env_dir, "Scripts"))
        # An environment is its interpreter *and* the ``pyvenv.cfg`` venv
        # writes next to it (see ``PythonRuntime.is_created``): without the
        # marker the interpreter cannot even start.
        with open(os.path.join(env_dir, "Scripts", "python.exe"), "w") as fh:
            fh.write("")
        with open(os.path.join(env_dir, "pyvenv.cfg"), "w") as fh:
            fh.write("")
        for package in packages:
            stem = package.replace("-", "_")
            os.makedirs(os.path.join(site, stem))
            os.makedirs(os.path.join(site, f"{stem}-1.0.dist-info"))
        return self.pr.PythonRuntime(env_dir)

    def _worker(self, engine_id: str, own_packages, shared_packages):
        """A worker plus its two candidate environments.

        An empty ``own_packages`` also means "this engine has no environment of
        its own yet" (an install from before the per-TTS environments).
        """
        run = self._count
        self._count += 1
        own = self._env(f"tts_envs_{engine_id}_{run}", own_packages,
                        created=bool(own_packages))
        shared = self._env(f"addon_env_{run}", shared_packages)

        def fake_get_runtime(env_id=None):
            return own if env_id == engine_id else shared

        patcher = mock.patch("ai_voice_studio.python_runtime.get_runtime",
                             side_effect=fake_get_runtime)
        patcher.start()
        self.addCleanup(patcher.stop)
        return voicelab.CloneWorker(engine_id, "cpu"), own, shared

    def test_the_engines_own_environment_wins_before_the_probe_answers(self):
        worker, own, _shared = self._worker(
            "f5tts", ["f5_tts"], ["numpy"],
        )
        # The cache really is cold: this is the state right after an install.
        self.assertFalse(self.vp.is_known("f5_tts", engine="f5tts"))
        self.assertIs(worker._runtime(), own)

    def test_an_engine_installed_in_the_shared_environment_still_works(self):
        worker, own, shared = self._worker(
            "f5tts", [], ["f5_tts"],
        )
        self.assertFalse(own.is_created)  # never created: a pre-per-TTS install
        self.assertIs(worker._runtime(), shared)

    def test_a_partly_installed_engine_says_so_instead_of_failing_later(self):
        worker, _own, _shared = self._worker(
            # ``torch`` and ``torchaudio`` are missing.
            "chatterbox", ["chatterbox"], ["numpy"],
        )
        with self.assertRaises(voicelab.VoicelabError) as caught:
            worker._runtime()
        self.assertIn(voice_lab.engine_name("chatterbox"),
                      str(caught.exception))

    def test_an_own_environment_engine_never_runs_in_the_shared_one(self):
        """KokoClone must not start in the addon/OmniVoice environment.

        Everything it needs happens to be there - OmniVoice installed a
        different build of the same packages - and borrowing them is exactly
        what would break both engines.
        """
        modules = list(voice_lab.import_modules("kokoclone"))
        worker, own, shared = self._worker("kokoclone", [], modules)
        self.assertFalse(own.is_created)
        self.assertTrue(shared.is_created)
        with self.assertRaises(voicelab.VoicelabError) as caught:
            worker._runtime()
        self.assertIn("its own Python environment", str(caught.exception))
        self.assertIn("Settings > Voice Clone", str(caught.exception))

    def test_an_own_environment_engine_uses_its_own_environment(self):
        modules = list(voice_lab.import_modules("kokoclone"))
        worker, own, _shared = self._worker("kokoclone", modules, modules)
        self.assertIs(worker._runtime(), own)

    def test_an_own_environment_engine_with_a_half_installed_env_says_so(self):
        # ``core`` is there, ``kanade_tokenizer`` (the model package) is not.
        worker, _own, _shared = self._worker("kokoclone", ["core"], ["core"])
        with self.assertRaises(voicelab.VoicelabError) as caught:
            worker._runtime()
        self.assertIn("Settings > Voice Clone", str(caught.exception))

    def test_every_engine_is_checked_with_its_own_import_modules(self):
        for engine_id in voice_lab.engine_ids():
            modules = list(voice_lab.import_modules(engine_id))
            worker, own, _shared = self._worker(engine_id, modules, ["numpy"])
            self.assertIs(worker._runtime(), own, engine_id)

    def test_the_worker_process_is_started_in_python_utf8_mode(self):
        """A bare ``open()`` in an engine must not decode with Windows-1252.

        ``kittenml`` reads its ``voices.json`` without an encoding, and that
        file contains typographic quotes - so every synthesis failed with
        ``UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d``.  The
        mode cannot be turned on inside the process (the file-encoding default
        is fixed when the interpreter starts), so the child is started with
        it; the parent's protocol pipe is written in UTF-8 either way.
        """
        modules = list(voice_lab.import_modules("kitten_tts_2"))
        worker, _own, _shared = self._worker("kitten_tts_2", modules, ["numpy"])
        seen: dict = {}

        def _popen(command, **kwargs):
            seen.update(kwargs)
            return mock.Mock(poll=mock.Mock(return_value=None))

        os.environ["AIVS_TEST_MARKER"] = "1"
        self.addCleanup(os.environ.pop, "AIVS_TEST_MARKER", None)
        with mock.patch.object(worker, "_request", return_value={"ok": True}), \
                mock.patch.object(voicelab.subprocess, "Popen", side_effect=_popen):
            worker._ensure_proc()
        env = seen.get("env") or {}
        self.assertEqual(env.get("PYTHONUTF8"), "1")
        # ... and the rest of the environment still reaches the process.
        self.assertEqual(env.get("AIVS_TEST_MARKER"), "1")

    def test_the_worker_environment_is_the_hugging_face_one_plus_utf8(self):
        from ai_voice_studio import hf_access

        with mock.patch.object(hf_access, "worker_environment",
                               return_value={"HF_TOKEN": "hf_x"}) as base:
            env = voicelab._worker_environment("kitten_tts_2")
        self.assertEqual(env, {"HF_TOKEN": "hf_x", "PYTHONUTF8": "1"})
        base.assert_called_once_with()


class OwnEnvironmentTest(unittest.TestCase):
    """Which engines may fall back to the old shared addon environment."""

    @staticmethod
    def _scope_aware_installed(shared_only=()):
        """``venv_packages.installed`` that only knows the shared env.

        ``shared_only`` names the packages that are in the *shared* addon
        environment (where OmniVoice lives) and nowhere else.
        """

        def fake(package, engine=None):
            return engine is None and package in shared_only

        return fake

    def test_kokoclone_requires_its_own_environment(self):
        # The engines whose dependencies cannot live beside another engine's
        # (KokoClone's and Chatterbox's pinned PyTorch, Qwen's pinned
        # transformers) must never be reported installed because of anything
        # another engine put in the shared environment.
        own = ("kokoclone", "chatterbox", "qwen3tts", "vibevoice",
               "moss_tts", "gptsovits", "indic_parler", "kitten_tts_2")
        for engine_id in own:
            self.assertTrue(voice_lab.own_environment(engine_id), engine_id)
        for engine_id in voice_lab.engine_ids():
            if engine_id not in own:
                self.assertFalse(voice_lab.own_environment(engine_id), engine_id)

    def test_a_shared_only_install_does_not_count_for_kokoclone(self):
        installed = self._scope_aware_installed({"kokoclone", "kanade-tokenizer"})
        with mock.patch("ai_voice_studio.venv_packages.installed",
                        side_effect=installed), \
                mock.patch("ai_voice_studio.venv_packages.version",
                           side_effect=lambda package, engine=None:
                           "0.1.0" if installed(package, engine) else None):
            self.assertFalse(voicelab.is_installed("kokoclone"))
            self.assertIsNone(voicelab.installed_version("kokoclone"))

    def test_a_legacy_shared_install_still_counts_for_the_other_engines(self):
        installed = self._scope_aware_installed({"pocket-tts"})
        with mock.patch("ai_voice_studio.venv_packages.installed",
                        side_effect=installed), \
                mock.patch("ai_voice_studio.venv_packages.version",
                           side_effect=lambda package, engine=None:
                           "1.0" if installed(package, engine) else None):
            self.assertTrue(voicelab.is_installed("pocket_tts"))
            self.assertEqual(voicelab.installed_version("pocket_tts"), "1.0")

    def test_the_engine_environment_folder_is_its_own(self):
        """The environment is per engine, and KokoClone's is not OmniVoice's."""
        from ai_voice_studio import python_runtime as pr

        self.assertEqual(pr.environment_id("kokoclone"), "kokoclone")
        self.assertNotEqual(pr.environment_id("kokoclone"),
                            pr.environment_id("omnivoice"))
        self.assertNotEqual(pr.environment_id("kokoclone"),
                            pr.environment_id("omnivoice_server"))
        self.assertNotEqual(pr.environment_id("kokoclone"), "default")


# ---------------------------------------------------------------------------
# The GPU-only transformer engines
# ---------------------------------------------------------------------------
def _fake_torch(available=True):
    """A ``torch`` stand-in: enough for the engine backends' CUDA check."""
    module = types.ModuleType("torch")
    module.bfloat16 = "bfloat16"
    module.float32 = "float32"

    class _Cuda:
        @staticmethod
        def is_available():
            return available

    module.cuda = _Cuda()
    return module


def _fake_chatterbox_modules(record, sample_rate=24000):
    """Fake ``chatterbox`` models that record how they were spoken to."""

    class _Model:
        sr = sample_rate

        @classmethod
        def from_pretrained(cls, device="cuda", **kwargs):
            record.append(("load", cls.__name__, device))
            return cls()

        def generate(self, text, audio_prompt_path=None, **kwargs):
            record.append(("generate", type(self).__name__, text,
                           audio_prompt_path, kwargs))
            return np.zeros(240, dtype=np.float32)

    class ChatterboxTTS(_Model):
        pass

    class ChatterboxTurboTTS(_Model):
        pass

    class ChatterboxMultilingualTTS(_Model):
        def generate(self, text, language_id=None, audio_prompt_path=None,
                     **kwargs):
            record.append(("generate", type(self).__name__, text,
                           audio_prompt_path, language_id, kwargs))
            return np.zeros(240, dtype=np.float32)

    modules = {"chatterbox": types.ModuleType("chatterbox")}
    modules["chatterbox"].__path__ = []
    for name, cls in (
        ("chatterbox.tts", ChatterboxTTS),
        ("chatterbox.tts_turbo", ChatterboxTurboTTS),
        ("chatterbox.mtl_tts", ChatterboxMultilingualTTS),
    ):
        module = types.ModuleType(name)
        setattr(module, cls.__name__, cls)
        modules[name] = module
    return modules


def _one_per_text(texts, sample):
    """The list of recordings the real package returns for a text/text list."""
    if isinstance(texts, (list, tuple)):
        return [sample for _ in texts]
    return [sample]


def _fake_qwen_module(record, sample_rate=24000):
    """A fake ``qwen_tts`` whose model records every call it receives."""
    module = types.ModuleType("qwen_tts")

    class Qwen3TTSModel:
        @classmethod
        def from_pretrained(cls, repo, device_map=None, dtype=None, **kwargs):
            record.append(("load", repo, device_map, dtype, kwargs))
            return cls()

        def create_voice_clone_prompt(self, ref_audio=None, ref_text=None,
                                      x_vector_only_mode=False, **kwargs):
            record.append(("prompt", ref_audio, ref_text, x_vector_only_mode))
            return {"prompt_for": ref_audio}

        def generate_custom_voice(self, text, language=None, speaker=None,
                                  **kwargs):
            record.append(("custom", text, language, speaker, kwargs))
            # One recording per text, like the real package (a list of texts
            # is one batched call, and every one of them comes back).
            return _one_per_text(text, np.zeros(240, dtype=np.float32)), sample_rate

        def generate_voice_clone(self, text, language=None,
                                 voice_clone_prompt=None, ref_audio=None,
                                 ref_text=None, **kwargs):
            record.append(("clone", text, language, voice_clone_prompt,
                           ref_audio, ref_text, kwargs))
            return _one_per_text(text, np.zeros(240, dtype=np.float32)), sample_rate

    module.Qwen3TTSModel = Qwen3TTSModel
    return module


def _fake_vibevoice_modules(record, sample_rate=24000):
    """Fake ``vibevoice`` package recording what the worker asked of it."""

    class _Tensor:
        def to(self, device, *args, **kwargs):
            return self

    class VibeVoiceForConditionalGenerationInference:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            record.append(("load", repo, kwargs))
            return cls()

        def eval(self):
            record.append(("eval",))

        def set_ddpm_inference_steps(self, num_steps=10):
            record.append(("steps", num_steps))

        def generate(self, **kwargs):
            record.append(("generate", kwargs))

            class _Out:
                speech_outputs = [np.zeros(240, dtype=np.float32)]

            return _Out()

    class VibeVoiceProcessor:
        tokenizer = "tokenizer"

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            record.append(("processor", repo))
            return cls()

        def __call__(self, text=None, voice_samples=None, **kwargs):
            record.append(("inputs", text, voice_samples, kwargs))
            return {"input_ids": _Tensor(), "attention_mask": _Tensor()}

    package = types.ModuleType("vibevoice")
    package.__path__ = []
    modular = types.ModuleType("vibevoice.modular")
    modular.__path__ = []
    processor_package = types.ModuleType("vibevoice.processor")
    processor_package.__path__ = []
    model_module = types.ModuleType("vibevoice.modular.modeling_vibevoice_inference")
    model_module.VibeVoiceForConditionalGenerationInference = (
        VibeVoiceForConditionalGenerationInference
    )
    processor_module = types.ModuleType("vibevoice.processor.vibevoice_processor")
    processor_module.VibeVoiceProcessor = VibeVoiceProcessor
    return {
        "vibevoice": package,
        "vibevoice.modular": modular,
        "vibevoice.modular.modeling_vibevoice_inference": model_module,
        "vibevoice.processor": processor_package,
        "vibevoice.processor.vibevoice_processor": processor_module,
    }


class VibeVoiceBackendTest(unittest.TestCase):
    """VibeVoice's worker backend, driven with fake models.

    The two models are gigabytes, need a GPU and are downloaded from Hugging
    Face, so they are replaced by stand-ins that record what the worker asked
    them for.  What is checked is the application's own responsibility: the
    variant picking the model, the script's speaker label, the reference
    recording (a clone, or a preset fetched once), the tuning knobs, and the
    plain report when no GPU is usable.
    """

    def setUp(self):
        self.worker = _load_worker()
        self.record = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _backend(self, variant, device="cuda", **request):
        payload = {"text": "Hello there.", "variant": variant,
                   "language": "en"}
        payload.update(request)
        return self.worker.VibeVoiceBackend(device, payload)

    @contextlib.contextmanager
    def _engine(self, available=True):
        modules = _fake_vibevoice_modules(self.record)
        modules["torch"] = _fake_torch(available)
        with mock.patch.dict(sys.modules, modules):
            yield

    def _sample(self):
        path = os.path.join(self.tmp.name, "reference.wav")
        _write_silence(path, frames=240)
        return path

    @staticmethod
    def _speak(backend):
        return backend.synthesize(backend.request["text"], backend.request)

    def _generate_kwargs(self):
        return next(entry[1] for entry in self.record if entry[0] == "generate")

    def test_the_variant_picks_the_model(self):
        for variant, repo in (
            ("vibevoice_1_5b", "vibevoice/VibeVoice-1.5B"),
            ("vibevoice_7b", "vibevoice/VibeVoice-7B"),
        ):
            self.record.clear()
            backend = self._backend(variant, ref_audio=self._sample())
            with self._engine():
                self._speak(backend)
            self.assertEqual(self.record[0][0], "load")
            self.assertEqual(self.record[0][1], repo)
            # Both models are loaded on the card in bfloat16.
            self.assertEqual(self.record[0][2]["torch_dtype"], "bfloat16")
            self.assertEqual(self.record[0][2]["device_map"], "cuda")

    def test_a_plain_paragraph_gets_the_single_speaker_label(self):
        backend = self._backend("vibevoice_1_5b", ref_audio=self._sample())
        with self._engine():
            self._speak(backend)
        inputs = next(entry for entry in self.record if entry[0] == "inputs")
        self.assertEqual(inputs[1], ["Speaker 1: Hello there."])
        self.assertEqual(inputs[2], [[backend.request["ref_audio"]]])

    def test_a_labelled_script_is_passed_through(self):
        backend = self._backend(
            "vibevoice_1_5b", ref_audio=self._sample(),
            text="Speaker 1: Hi\nSpeaker 2: Hello",
        )
        with self._engine():
            self._speak(backend)
        inputs = next(entry for entry in self.record if entry[0] == "inputs")
        self.assertEqual(inputs[1], ["Speaker 1: Hi\nSpeaker 2: Hello"])

    def test_a_built_in_preset_is_fetched_once_and_then_reused(self):
        """The fork's presets are not inside the pip package: the worker
        downloads the one a voice asks for and caches it in its own folder."""
        workspace = os.path.join(self.tmp.name, "engine_home")
        backend = self._backend(
            "vibevoice_1_5b", workspace=workspace,
            ref_resource="vibevoice:demo/voices/en-Alice_woman.wav",
        )
        self.record.clear()
        downloaded = []

        def _urlopen(url, timeout=None):
            downloaded.append(url)

            class _Response(contextlib.AbstractContextManager):
                def __enter__(self_inner):
                    return io.BytesIO(b"RIFF....")

                def __exit__(self_inner, *exc):
                    return False

            return _Response()

        with mock.patch.object(self.worker.urllib.request, "urlopen", _urlopen), self._engine():
            self._speak(backend)
            self._speak(backend)
        self.assertEqual(len(downloaded), 1, downloaded)
        self.assertIn("en-Alice_woman.wav", downloaded[0])
        self.assertEqual(len([e for e in self.record if e[0] == "inputs"]), 2)

    def test_a_preset_that_cannot_be_downloaded_says_so(self):
        backend = self._backend(
            "vibevoice_1_5b", workspace=self.tmp.name,
            ref_resource="vibevoice:demo/voices/en-Maya_woman.wav",
        )

        def _urlopen(url, timeout=None):
            raise OSError("no network")

        with mock.patch.object(self.worker.urllib.request, "urlopen", _urlopen), self._engine():
            with self.assertRaises(RuntimeError) as caught:
                self._speak(backend)
        message = str(caught.exception)
        self.assertIn("en-Maya_woman.wav", message)
        self.assertIn("internet connection", message)

    def test_the_tuning_options_reach_the_right_calls(self):
        backend = self._backend(
            "vibevoice_1_5b", ref_audio=self._sample(),
            options={"cfg_scale": 2.0, "ddpm_steps": 20, "seed": 7},
        )
        with self._engine():
            self._speak(backend)
        # The sampler's step count goes through the model's own setter...
        self.assertIn(("steps", 20), self.record)
        # ... and never into the generating call.
        kwargs = self._generate_kwargs()
        self.assertNotIn("ddpm_steps", kwargs)
        self.assertEqual(kwargs["cfg_scale"], 2.0)
        self.assertEqual(kwargs["is_prefill"], True)
        # ``seed`` is applied by the worker's request loop, not by the
        # backend, so only the engine's own two knobs are reported here.
        self.assertEqual(sorted(backend.report.applied),
                         ["cfg_scale", "ddpm_steps"])

    def test_the_engine_defaults_are_left_alone(self):
        backend = self._backend("vibevoice_1_5b", ref_audio=self._sample())
        with self._engine():
            self._speak(backend)
        kwargs = self._generate_kwargs()
        self.assertNotIn("cfg_scale", kwargs)
        self.assertEqual([e for e in self.record if e[0] == "steps"], [])

    def test_a_missing_reference_recording_says_so(self):
        backend = self._backend("vibevoice_1_5b", ref_audio="nope.wav")
        with self._engine(), self.assertRaises(FileNotFoundError):
            self._speak(backend)

    def test_a_voice_with_no_reference_says_what_to_do(self):
        backend = self._backend("vibevoice_1_5b")
        with self._engine(), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        self.assertIn("built-in voices", str(caught.exception))

    def test_a_missing_gpu_is_reported_in_plain_words(self):
        backend = self._backend("vibevoice_1_5b", ref_audio=self._sample())
        with self._engine(available=False), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        message = str(caught.exception)
        self.assertIn("NVIDIA GPU", message)
        self.assertIn("VibeVoice", message)

    def test_the_built_in_voice_inventory_needs_no_model(self):
        """Answering "which voices are usable?" must not load anything."""
        backend = self._backend("vibevoice_7b")
        self.assertEqual(sorted(backend.available_voices()),
                         sorted(self.worker._VIBEVOICE_VOICES))
        self.assertEqual(self.record, [])


class ChatterboxBackendTest(unittest.TestCase):
    """Chatterbox's worker backend, driven with fake models.

    The three models are gigabytes and need a GPU, so they are replaced by
    stand-ins that record what the worker asked them for.  What is checked is
    the application's own responsibility: the variant choosing the model, the
    reference clip and language travelling with a request, and the plain
    report when no GPU is usable.
    """

    def setUp(self):
        self.worker = _load_worker()
        self.record = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _backend(self, variant, device="cuda", **request):
        payload = {"text": "Hello", "variant": variant, "language": "en"}
        payload.update(request)
        return self.worker.ChatterboxBackend(device, payload)

    @contextlib.contextmanager
    def _engine(self, available=True):
        modules = _fake_chatterbox_modules(self.record)
        modules["torch"] = _fake_torch(available)
        with mock.patch.dict(sys.modules, modules):
            yield

    def _sample(self):
        path = os.path.join(self.tmp.name, "reference.wav")
        _write_silence(path, frames=240)
        return path

    @staticmethod
    def _speak(backend):
        """Synthesize with the very request the backend was built for.

        The worker builds a backend from the request and then hands the same
        request to ``synthesize``; the options and the voice live on it.
        """
        return backend.synthesize(backend.request["text"], backend.request)

    def test_the_variant_picks_the_model(self):
        for variant, expected in (
            ("chatterbox_v1", "ChatterboxTTS"),
            ("chatterbox_turbo", "ChatterboxTurboTTS"),
            ("chatterbox_mtl", "ChatterboxMultilingualTTS"),
        ):
            self.record.clear()
            backend = self._backend(variant)
            with self._engine():
                self._speak(backend)
            self.assertEqual(self.record[0], ("load", expected, "cuda"))
            # The rate comes from the model itself, not from a guess.
            self.assertEqual(backend.sample_rate, 24000)

    def test_only_the_multilingual_model_is_told_a_language(self):
        with self._engine():
            multilingual = self._backend("chatterbox_mtl", text="Bonjour",
                                         language="fr")
            self._speak(multilingual)
            english = self._backend("chatterbox_v1")
            self._speak(english)
        # ("generate", class, text, audio_prompt_path, language_id, kwargs)
        self.assertEqual(self.record[1][2], "Bonjour")
        self.assertEqual(self.record[1][4], "fr")
        # The English model's generate takes no language at all.
        self.assertEqual(self.record[3][2], "Hello")
        self.assertEqual(self.record[3][3], None)

    def test_an_english_model_refuses_another_language(self):
        backend = self._backend("chatterbox_turbo", text="Hallo",
                                language="de")
        with self._engine(), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        self.assertIn("only speaks English", str(caught.exception))

    def test_a_clone_travels_as_a_reference_clip(self):
        sample = self._sample()
        backend = self._backend("chatterbox_mtl", text="Hallo",
                                language="de", ref_audio=sample)
        with self._engine():
            self._speak(backend)
        call = self.record[1]
        self.assertEqual(call[2], "Hallo")
        self.assertEqual(call[3], sample)
        self.assertEqual(call[4], "de")

    def test_a_missing_reference_recording_says_so(self):
        backend = self._backend("chatterbox_mtl", ref_audio="nope.wav")
        with self._engine(), self.assertRaises(FileNotFoundError):
            self._speak(backend)

    def test_a_missing_gpu_is_reported_in_plain_words(self):
        backend = self._backend("chatterbox_v1")
        with self._engine(available=False), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        message = str(caught.exception)
        self.assertIn("NVIDIA GPU", message)
        self.assertIn("Chatterbox", message)

    def test_the_built_in_voice_inventory_needs_no_model(self):
        """Answering "which voices are usable?" must not load anything."""
        backend = self._backend("chatterbox_mtl")
        self.assertEqual(backend.available_voices(), {"default": True})
        self.assertEqual(self.record, [])


class QwenTtsBackendTest(unittest.TestCase):
    """Qwen3-TTS's worker backend, driven with a fake model."""

    def setUp(self):
        self.worker = _load_worker()
        self.record = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _backend(self, variant, device="cuda", **request):
        payload = {"text": "Hello", "variant": variant, "language": "en"}
        payload.update(request)
        return self.worker.QwenTtsBackend(device, payload)

    @contextlib.contextmanager
    def _engine(self, available=True):
        modules = {"qwen_tts": _fake_qwen_module(self.record),
                   "torch": _fake_torch(available)}
        with mock.patch.dict(sys.modules, modules):
            yield

    def _sample(self):
        path = os.path.join(self.tmp.name, "reference.wav")
        _write_silence(path, frames=240)
        return path

    def _speak(self, backend, text=None):
        """Synthesize with the very request the backend was built for."""
        request = dict(backend.request)
        if text is not None:
            request["text"] = text
            backend.request = request
        return backend.synthesize(request["text"], request)

    def test_each_variant_loads_its_own_model(self):
        for variant, repo in (
            ("qwen3_custom_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            ("qwen3_custom_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            ("qwen3_clone_1_7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            ("qwen3_clone_0_6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        ):
            self.record.clear()
            extra = ({"voice": "Ryan"} if "custom" in variant
                     else {"ref_audio": self._sample()})
            backend = self._backend(variant, **extra)
            with self._engine():
                self._speak(backend)
            self.assertEqual(self.record[0][1], repo, variant)
            self.assertEqual(self.record[0][2], "cuda:0", variant)

    def test_a_custom_voice_model_speaks_its_named_voice(self):
        backend = self._backend("qwen3_custom_1_7b", voice="Ryan")
        with self._engine():
            self._speak(backend)
        call = self.record[1]
        self.assertEqual(call[:4], ("custom", "Hello", "English", "Ryan"))

    def test_the_language_travels_as_the_packages_own_name(self):
        with self._engine():
            german = self._backend("qwen3_custom_0_6b", text="Hallo",
                                   language="de-DE", voice="Vivian")
            self._speak(german)
            unknown = self._backend("qwen3_custom_0_6b", text="Ciao",
                                    language="xx", voice="Vivian")
            self._speak(unknown)
        customs = [entry for entry in self.record if entry[0] == "custom"]
        self.assertEqual(customs[0][2], "German")
        # An unknown code lets the model decide rather than failing the run.
        self.assertEqual(customs[1][2], "Auto")

    def test_a_base_model_clones_from_the_recording_and_reuses_the_prompt(self):
        sample = self._sample()
        backend = self._backend("qwen3_clone_1_7b", ref_audio=sample,
                                ref_text="Hello there")
        with self._engine():
            self._speak(backend)
            self._speak(backend, "Again")
        prompts = [entry for entry in self.record if entry[0] == "prompt"]
        # Extracting the speaker's features is the expensive part: it is done
        # once per reference clip, not once per segment.
        self.assertEqual(len(prompts), 1, self.record)
        self.assertEqual(prompts[0][1:], (sample, "Hello there", False))
        clones = [entry for entry in self.record if entry[0] == "clone"]
        self.assertEqual(len(clones), 2)
        self.assertEqual(clones[0][3], {"prompt_for": sample})
        self.assertEqual([c[1] for c in clones], ["Hello", "Again"])

    def test_a_reference_without_a_transcript_copies_the_timbre_only(self):
        backend = self._backend("qwen3_clone_0_6b", ref_audio=self._sample())
        with self._engine():
            self._speak(backend)
        prompt = next(entry for entry in self.record if entry[0] == "prompt")
        self.assertTrue(prompt[3])

    def test_a_base_model_needs_a_recording(self):
        backend = self._backend("qwen3_clone_1_7b")
        with self._engine(), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        self.assertIn("reference recording", str(caught.exception))

    def test_a_custom_voice_model_refuses_a_recording(self):
        backend = self._backend("qwen3_custom_1_7b", voice="Ryan",
                                ref_audio=self._sample())
        with self._engine(), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        self.assertIn("cannot clone", str(caught.exception))

    def test_a_missing_gpu_is_reported_in_plain_words(self):
        backend = self._backend("qwen3_custom_1_7b", voice="Ryan")
        with self._engine(available=False), self.assertRaises(RuntimeError) as caught:
            self._speak(backend)
        self.assertIn("NVIDIA GPU", str(caught.exception))

    def test_only_the_custom_models_offer_named_voices(self):
        self.assertEqual(self._backend("qwen3_custom_0_6b").available_voices(),
                         dict.fromkeys(self.worker._QWEN_SPEAKERS, True))
        self.assertEqual(self._backend("qwen3_clone_1_7b").available_voices(), {})
        self.assertEqual(self.record, [], "no model may be loaded to answer this")

    def test_several_texts_are_generated_in_one_batched_call(self):
        """One call for several segments: that is where the card's parallelism
        comes from (measured: four sentences in one call take 52 s where four
        calls take 183 s)."""
        backend = self._backend("qwen3_custom_1_7b", voice="Ryan")
        request = dict(backend.request)
        with self._engine():
            results = backend.synthesize_many(
                ["First one", "Second one", "Third one"], request
            )
        calls = [entry for entry in self.record if entry[0] == "custom"]
        self.assertEqual(len(calls), 1, self.record)
        self.assertEqual(calls[0][1], ["First one", "Second one", "Third one"])
        self.assertEqual(len(results), 3)
        for audio, rate, handled in results:
            self.assertFalse(handled)
            self.assertEqual(rate, 24000)
            self.assertGreater(np.asarray(audio).size, 0)

    def test_a_base_model_batches_its_cloned_lines_too(self):
        sample = self._sample()
        backend = self._backend("qwen3_clone_1_7b", ref_audio=sample,
                                ref_text="Hello there")
        request = dict(backend.request)
        with self._engine():
            results = backend.synthesize_many(["One", "Two"], request)
        clones = [entry for entry in self.record if entry[0] == "clone"]
        self.assertEqual(len(clones), 1)
        self.assertEqual(clones[0][1], ["One", "Two"])
        # The speaker's features are still extracted once for the whole clip.
        self.assertEqual(len([e for e in self.record if e[0] == "prompt"]), 1)
        self.assertEqual(len(results), 2)

    def test_one_text_is_never_turned_into_a_batch(self):
        backend = self._backend("qwen3_custom_1_7b", voice="Ryan")
        request = dict(backend.request)
        with self._engine():
            backend.synthesize_many(["Just the one"], request)
        call = self.record[1]
        self.assertEqual(call[1], "Just the one")
        self.assertEqual(len([e for e in self.record if e[0] == "custom"]), 1)

    def test_more_texts_than_the_engine_takes_is_refused(self):
        backend = self._backend("qwen3_custom_1_7b", voice="Ryan")
        request = dict(backend.request)
        too_many = ["text"] * (backend.batch_limit + 1)
        with self._engine(), self.assertRaises(RuntimeError) as caught:
            backend.synthesize_many(too_many, request)
        self.assertIn("at most", str(caught.exception))
        self.assertEqual(self.record, [], "nothing may be loaded for this")

    def test_the_tuning_options_reach_only_the_generating_call(self):
        backend = self._backend(
            "qwen3_custom_1_7b", voice="Ryan",
            options={"temperature": 0.3, "instruct": "Whisper it.",
                     "seed": 7},
        )
        with self._engine():
            self._speak(backend)
        call = self.record[1]
        self.assertEqual(call[4], {"temperature": 0.3,
                                  "instruct": "Whisper it."})
        self.assertEqual(sorted(backend.report.applied),
                         ["instruct", "temperature"])


class GpuFillTest(unittest.TestCase):
    """How many segments one card is asked to hold at once.

    The cost of a slot is the engine's own measured peak video memory
    (``engines._GPU_PARALLEL``); how many slots a *machine* can afford is
    worked out from the memory free on its card at that moment, so the driver
    is stubbed here and the arithmetic is what is checked.
    """

    RTX_3060 = 11241  # what was free on the 12 GB card the figures come from

    def _plan(self, engine_id, variant="", free_mb=RTX_3060, device="cuda"):
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.compute.free_vram_mb",
                           return_value=free_mb):
            return voicelab.parallel_plan(engine_id, device, variant)

    def test_a_cpu_run_speaks_one_segment_at_a_time(self):
        for engine_id in ("pocket_tts", "f5tts", "kokoclone"):
            self.assertEqual(self._plan(engine_id, device="cpu"), (1, "solo"),
                             engine_id)

    def test_an_engine_with_no_plan_of_its_own_stays_solo(self):
        # These engines keep working exactly as they always did: one segment,
        # one loaded model, however much the card could have held.
        for engine_id in ("pocket_tts", "f5tts", "kokoclone"):
            self.assertEqual(self._plan(engine_id), (1, "solo"), engine_id)

    def test_a_small_card_keeps_to_one_loaded_model(self):
        # Chatterbox speaks one text per call, so every extra segment needs
        # another 4.7 GB copy of the weights.  A 4 GB or 6 GB card holds one.
        for card in (4096, 6144):
            self.assertEqual(
                self._plan("chatterbox", "chatterbox_v1", free_mb=card),
                (1, "instances"), card,
            )

    def test_a_bigger_card_holds_more_segments(self):
        self.assertEqual(self._plan("chatterbox", "chatterbox_v1"),
                         (2, "instances"))
        self.assertEqual(
            self._plan("chatterbox", "chatterbox_v1", free_mb=24576),
            (3, "instances"),  # the engine's own ceiling, whatever the card
        )

    def test_a_batched_engine_fills_the_card_without_a_second_model(self):
        # Qwen3-TTS generates a list of texts in one call, so the extra
        # segments only add their own share of the batch to one loaded model.
        for card in (4096, 6144, 8192):
            self.assertEqual(
                self._plan("qwen3tts", "qwen3_custom_1_7b", free_mb=card),
                (1, "batch"), card,
            )
        self.assertEqual(self._plan("qwen3tts", "qwen3_custom_1_7b"),
                         (5, "batch"))
        self.assertEqual(
            self._plan("qwen3tts", "qwen3_custom_1_7b", free_mb=24576),
            (6, "batch"),
        )
        # The 0.6B model is smaller, so the same card holds more of it.
        self.assertGreater(self._plan("qwen3tts", "qwen3_custom_0_6b")[0],
                           self._plan("qwen3tts", "qwen3_custom_1_7b")[0])

    def test_the_reserve_is_left_to_the_rest_of_the_machine(self):
        # 6200 MB free is one 4.7 GB model and nothing like two: the second
        # copy must not eat the memory the desktop is using.
        self.assertEqual(self._plan("chatterbox", "chatterbox_v1",
                                    free_mb=6200),
                         (1, "instances"))
        self.assertEqual(voicelab.GPU_VRAM_RESERVE_MB, 1500)

    def test_a_card_the_driver_cannot_describe_stays_solo(self):
        for free in (None, 0):
            self.assertEqual(
                self._plan("chatterbox", "chatterbox_v1", free_mb=free),
                (1, "solo"), free,
            )

    def test_the_lighter_model_fits_more_often(self):
        turbo = self._plan("chatterbox", "chatterbox_turbo", free_mb=10240)
        english = self._plan("chatterbox", "chatterbox_v1", free_mb=10240)
        self.assertEqual(turbo, (2, "instances"))
        self.assertEqual(english, (1, "instances"))

    def test_an_unknown_variant_is_charged_the_dearest_model(self):
        for engine_id in ("chatterbox", "qwen3tts", "vibevoice"):
            declared = voice_lab._GPU_PARALLEL[engine_id]["slots"]
            dearest = max(declared.values(), key=lambda s: s["vram_mb"])
            self.assertEqual(
                voice_lab.gpu_parallel(engine_id, "a_future_model")["vram_mb"],
                dearest["vram_mb"], engine_id,
            )

    def test_the_declared_variants_are_the_catalogue_variants(self):
        for engine_id, declared in voice_lab._GPU_PARALLEL.items():
            offered = {
                variant["id"]
                for lang in voice_lab.engine(engine_id)["languages"]
                for variant in lang["variants"]
            }
            self.assertEqual(set(declared["slots"]), offered, engine_id)
            self.assertIn(declared["mode"], ("instances", "batch"))
        # An engine nobody declared a plan for answers "no plan".
        self.assertIsNone(voice_lab.gpu_parallel("f5tts"))
        self.assertIsNone(voice_lab.gpu_parallel("nonexistent"))

    def test_the_worker_and_the_app_agree_on_what_can_batch(self):
        worker = _load_worker()
        for engine_id in voice_lab.engine_ids():
            limit = worker._BACKENDS[engine_id].batch_limit
            declared = voice_lab.gpu_parallel(engine_id)
            if declared and declared["mode"] == "batch":
                # The worker must accept what the plan may hand it.
                self.assertGreater(limit, 1, engine_id)
                self.assertGreaterEqual(limit, declared["max_slots"], engine_id)
            else:
                self.assertEqual(limit, 1, engine_id)


class PlannedWorkersTest(unittest.TestCase):
    """What the recording pipeline asks its engine for."""

    def _planned(self, voice_entry, compute_choice="cpu", total=5):
        from ai_voice_studio.tts.engine import planned_workers

        with self._driver(GpuFillTest.RTX_3060):
            return planned_workers(voice_entry, compute_choice, total=total)

    @staticmethod
    @contextlib.contextmanager
    def _driver(free_mb):
        """Pin the card's free memory, so the plan does not depend on it."""
        with mock.patch("ai_voice_studio.compute.free_vram_mb",
                        return_value=free_mb), \
                mock.patch.object(voicelab, "has_cuda", return_value=True):
            yield

    def test_a_local_sherpa_model_keeps_the_compute_back_end_rule(self):
        self.assertEqual(self._planned({"engine": "vits"}, "cpu"), 1)
        self.assertGreaterEqual(self._planned({"engine": "vits"}, "cuda"), 1)

    def test_a_voice_lab_engine_brings_its_own_number(self):
        self.assertEqual(
            self._planned({"engine": "chatterbox", "variant": "chatterbox_v1",
                           "device": "cuda"}),
            2,
        )
        # Without the variant on purpose: the engine's worker pool is shared by
        # every voice of that engine, so the pipeline asks for what the pool was
        # built for - the dearest model the engine declares.
        self.assertEqual(
            self._planned({"engine": "qwen3tts",
                           "variant": "qwen3_custom_1_7b",
                           "device": "cuda"}),
            4,
        )
        with self._driver(GpuFillTest.RTX_3060):
            per_variant = voicelab.parallel_plan(
                "qwen3tts", "cuda", "qwen3_custom_1_7b"
            )
        self.assertEqual(per_variant[0], 5)
        # A Voice Lab engine with no plan of its own - and one on the CPU -
        # stays one segment at a time.
        self.assertEqual(self._planned({"engine": "f5tts", "device": "cuda"}), 1)
        self.assertEqual(self._planned({"engine": "pocket_tts"}, "cpu"), 1)

    def test_a_gpu_only_engine_ignores_a_stored_cpu_choice(self):
        # The engine has no CPU mode at all (``resolve_device`` sends a
        # GPU-only engine to the card whatever is stored for it), so the plan
        # still comes from the card.
        self.assertEqual(
            self._planned({"engine": "qwen3tts",
                           "variant": "qwen3_custom_1_7b",
                           "device": "cpu"}, "cpu"),
            4,
        )

    def test_other_runtimes_stay_one_at_a_time(self):
        for engine_id in ("omnivoice", "omnivoice_server", "sapi5",
                          "windows_core"):
            self.assertEqual(self._planned({"engine": engine_id}, "cuda"), 1,
                             engine_id)

    def test_a_single_segment_run_asks_for_one(self):
        self.assertEqual(self._planned({"engine": "qwen3tts"}, "cuda", total=1), 1)
        self.assertEqual(self._planned({"engine": "vits"}, "cuda", total=0), 1)


class _NoProcessWorker:
    """A worker stand-in: records what the pool asked it for."""

    def __init__(self, name, calls, barrier=None, fail_batch=False):
        self.name = name
        self.calls = calls
        self.barrier = barrier
        self.fail_batch = fail_batch
        self.closed = False

    @staticmethod
    def _result(value=0):
        return (
            np.full(4, value, dtype=np.int16),
            24000,
            {"applied": [], "skipped": [], "device_used": "cuda",
             "vram_peak_mb": 4096, "notes": []},
        )

    def synthesize(self, request):
        self.calls.append(("single", request.get("text"), self.name))
        if self.barrier is not None:
            # Only a *second* worker, entered at the same time, releases this:
            # a pool that serialized the segments would time out here.
            self.barrier.wait(timeout=5)
        return self._result(0)

    def synthesize_batch(self, requests):
        self.calls.append(
            ("batch", [request.get("text") for request in requests], self.name)
        )
        if self.fail_batch:
            raise voicelab.VoicelabError("CUDA out of memory")
        return [self._result(index + 1) for index, _request in enumerate(requests)]

    def close(self):
        self.closed = True

    def kill(self):
        self.closed = True


class WorkerPoolTest(unittest.TestCase):
    """The pool: one worker process per segment, or one batched call."""

    def _pool(self, slots, mode, workers, engine_id="qwen3tts"):
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            pool = voicelab.WorkerPool(engine_id, "cuda", slots=slots, mode=mode)
        for worker in pool._workers:
            worker.close()  # no real process is ever started by these tests
        pool._workers = list(workers)
        pool._idle = queue.Queue()
        for worker in pool._workers:
            pool._idle.put(worker)
        self.addCleanup(pool.close)
        return pool

    @staticmethod
    def _request(text, **extra):
        request = {"cmd": "synthesize", "engine": "qwen3tts",
                   "device": "cuda", "text": text, "voice": "Vivian",
                   "options": {}}
        request.update(extra)
        return request

    def test_instances_mode_speaks_two_segments_at_once(self):
        calls = []
        barrier = threading.Barrier(2)
        pool = self._pool(2, "instances", [
            _NoProcessWorker("w0", calls, barrier=barrier),
            _NoProcessWorker("w1", calls, barrier=barrier),
        ], engine_id="chatterbox")
        self.assertEqual(pool.slots, 2)
        results = {}

        def run(index):
            results[index] = pool.synthesize(self._request(f"segment {index}"))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        # Both segments were in flight together, on a worker each.
        self.assertEqual(sorted(name for _k, _t, name in calls), ["w0", "w1"])
        self.assertEqual(len(results), 2)

    def test_instances_mode_speaks_each_text_through_the_pool(self):
        calls = []
        pool = self._pool(2, "instances", [
            _NoProcessWorker("w0", calls), _NoProcessWorker("w1", calls),
        ], engine_id="chatterbox")
        results = pool.synthesize_batch([self._request("a"),
                                         self._request("b")])
        self.assertEqual(len(results), 2)
        self.assertEqual([kind for kind, _t, _w in calls],
                         ["single", "single"])

    def test_batch_mode_asks_for_every_text_in_one_call(self):
        calls = []
        pool = self._pool(3, "batch", [_NoProcessWorker("w0", calls)])
        results = {}

        def run(index):
            results[index] = pool.synthesize(self._request(f"text {index}"))

        with mock.patch.object(voicelab, "BATCH_GRACE_SECONDS", 0.3):
            threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][0], "batch")
        self.assertEqual(sorted(calls[0][1]), ["text 0", "text 1", "text 2"])
        # Every caller got its *own* recording, not one caller's.
        self.assertEqual(
            sorted(int(np.asarray(result[0])[0]) for result in results.values()),
            [1, 2, 3],
        )

    def test_batch_mode_still_speaks_a_lone_segment_on_its_own(self):
        calls = []
        pool = self._pool(3, "batch", [_NoProcessWorker("w0", calls)])
        pool.synthesize(self._request("only one"))
        self.assertEqual(calls, [("single", "only one", "w0")])

    def test_a_different_voice_is_never_batched_with_another(self):
        calls = []
        pool = self._pool(3, "batch", [_NoProcessWorker("w0", calls)])
        results = {}

        def run(index, voice):
            results[index] = pool.synthesize(self._request(f"t{index}", voice=voice))

        with mock.patch.object(voicelab, "BATCH_GRACE_SECONDS", 0.4):
            threads = [threading.Thread(target=run, args=(0, "Vivian")),
                       threading.Thread(target=run, args=(1, "Ryan"))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        # Neither call is lost, and they are spoken separately: the second is
        # carried over to its own call rather than dropped.
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual([kind for kind, _t, _w in calls], ["single", "single"])
        self.assertEqual(len(results), 2)

    def test_a_batch_the_card_could_not_take_falls_back_to_one_by_one(self):
        calls = []
        fake = _NoProcessWorker("w0", calls, fail_batch=True)
        pool = self._pool(3, "batch", [fake])
        results = {}

        def run(index):
            results[index] = pool.synthesize(self._request(f"text {index}"))

        with self.assertLogs("ai_voice_studio.voicelab", level="WARNING"):
            threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(calls[0][0], "batch")
        self.assertEqual([kind for kind, _t, _w in calls[1:]],
                         ["single"] * 3)
        # A single bad segment must not take the ones beside it down with it.
        self.assertEqual(len(results), 3)

    def test_a_failure_that_is_the_texts_own_reaches_its_caller(self):
        calls = []

        class Picky(_NoProcessWorker):
            def synthesize_batch(self, requests):
                for request in requests:
                    if request.get("text") == "bad":
                        raise voicelab.VoicelabError(
                            "RuntimeError: that text is not supported"
                        )
                return super().synthesize_batch(requests)

            def synthesize(self, request):
                if request.get("text") == "bad":
                    raise voicelab.VoicelabError(
                        "RuntimeError: that text is not supported"
                    )
                return super().synthesize(request)

        pool = self._pool(3, "batch", [Picky("w0", calls)])
        with self.assertRaises(voicelab.VoicelabError) as caught:
            pool.synthesize_batch([self._request("a"), self._request("bad")])
        self.assertIn("not supported", str(caught.exception))

    def test_a_dispatcher_failure_never_leaves_a_caller_waiting(self):
        calls = []
        pool = self._pool(3, "batch", [_NoProcessWorker("w0", calls)])
        with mock.patch.object(pool, "_text_length", side_effect=RuntimeError(
            "a broken request"
        )), self.assertLogs("ai_voice_studio.voicelab", level="ERROR"):
            with self.assertRaises(voicelab.VoicelabError) as caught:
                pool.synthesize(self._request("Hello"))
        self.assertIn("gather the segments", str(caught.exception))
        self.assertEqual(calls, [])

    def test_a_closed_pool_refuses_more_work(self):
        pool = self._pool(2, "batch", [_NoProcessWorker("w0", [])])
        pool.close()
        with self.assertRaises(voicelab.VoicelabError):
            pool.synthesize(self._request("Hello"))

    def test_the_pool_says_what_it_will_do_with_the_card(self):
        self.assertIn("one segment at a time",
                      self._pool(1, "solo", [_NoProcessWorker("w0", [])],
                                 engine_id="f5tts").describe())
        self.assertIn("2 segments are spoken at once",
                      self._pool(2, "instances",
                                 [_NoProcessWorker("w0", []),
                                  _NoProcessWorker("w1", [])],
                                 engine_id="chatterbox").describe())
        self.assertIn("2 segments are spoken in one batched call",
                      self._pool(2, "batch", [_NoProcessWorker("w0", [])],
                                 engine_id="qwen3tts").describe())

    def test_a_solo_pool_never_batches(self):
        calls = []
        pool = self._pool(1, "solo", [_NoProcessWorker("w0", calls)],
                          engine_id="f5tts")
        self.assertEqual((pool.slots, pool.mode), (1, "solo"))
        pool.synthesize_batch([self._request("a"), self._request("b")])
        self.assertEqual([kind for kind, _t, _w in calls],
                         ["single", "single"])

    def test_a_batch_stays_within_what_a_slot_was_measured_with(self):
        """A batch of whole paragraphs would ask the card for several times
        the memory a slot's figure covers, so long segments are spoken in
        smaller batches instead - in the *same* order, one piece each."""
        calls = []
        budget = voicelab.batch_chars("qwen3tts", 4)
        pool = self._pool(4, "batch", [_NoProcessWorker("w0", calls)],
                          engine_id="qwen3tts")
        self.assertEqual(pool.chars_per_batch, budget)
        self.assertGreater(budget, 0)

        def speak(texts):
            results = {}

            def run(index):
                results[index] = pool.synthesize(self._request(texts[index]))

            with mock.patch.object(voicelab, "BATCH_GRACE_SECONDS", 0.3):
                threads = [threading.Thread(target=run, args=(i,))
                           for i in range(len(texts))]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
            self.assertEqual(len(results), len(texts))

        # Four sentences fit in one call ...
        speak(["x" * (budget // 4)] * 4)
        self.assertEqual(len(calls), 1, calls)
        calls.clear()
        # ... two paragraphs do not.
        speak(["y" * 900, "z" * 900])
        self.assertEqual([kind for kind, _t, _w in calls],
                         ["single", "single"], calls)

    def test_an_engine_without_a_measured_length_has_no_budget(self):
        self.assertEqual(voicelab.batch_chars("chatterbox", 2), 0)
        pool = self._pool(2, "instances",
                          [_NoProcessWorker("w0", []),
                           _NoProcessWorker("w1", [])],
                          engine_id="chatterbox")
        self.assertEqual(pool.chars_per_batch, 0)

    def test_the_engine_tells_the_pipeline_how_many_segments_fit(self):
        """The recording pipeline keeps ``engine.size`` segments in flight; an
        engine that did not answer would silently lose the parallelism, so the
        property is part of the contract with it."""
        self.addCleanup(voicelab.close_workers)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            voicelab.pool_for("chatterbox", "cuda", slots=3,
                              mode="instances")
            voice = dict(voice_lab.builtin_voices("chatterbox")[0])
            voice.update(engine="chatterbox", device="cuda")
            engine = voicelab.VoicelabEngine(voice, device="cuda")
        self.assertEqual(engine.size, 3)
        self.assertEqual(engine.slots, 3)
        self.assertEqual(engine.parallel_mode, "instances")

    def test_two_calls_are_only_batched_when_they_really_are_the_same(self):
        same = self._request("a")
        other = self._request("b")
        self.assertTrue(voicelab.WorkerPool._same_call(same, other))
        self.assertFalse(
            voicelab.WorkerPool._same_call(same, self._request("a", voice="Ryan"))
        )
        self.assertFalse(
            voicelab.WorkerPool._same_call(same, self._request("a", speed=0.9))
        )


class WorkerBatchRequestTest(unittest.TestCase):
    """``CloneWorker``'s batch call: one request for several texts."""

    @staticmethod
    def _request(text, **extra):
        request = {"cmd": "synthesize", "engine": "qwen3tts",
                   "device": "cuda", "text": text, "voice": "Vivian",
                   "options": {}}
        request.update(extra)
        return request

    def _payload(self, values):
        return [base64.b64encode(np.full(4, value, dtype=np.int16).tobytes())
                .decode("ascii") for value in values]

    def test_a_batch_asks_the_worker_for_the_whole_list_at_once(self):
        worker = voicelab.CloneWorker("qwen3tts", "cuda")
        captured = {}
        payloads = self._payload([1, 2, 3])

        def fake_request(request):
            captured.update(request)
            return {"ok": True, "wav": payloads[0], "wavs": payloads,
                    "sample_rate": 22050, "device_used": "cuda",
                    "vram_peak_mb": 9000}

        with mock.patch.object(worker, "_ensure_proc"), \
                mock.patch.object(worker, "_request", side_effect=fake_request):
            results = worker.synthesize_batch(
                [self._request("one"), self._request("two"),
                 self._request("three")]
            )
        # The texts travel as one list; the single-text field is gone.
        self.assertEqual(captured["texts"], ["one", "two", "three"])
        self.assertNotIn("text", captured)
        self.assertEqual(captured["voice"], "Vivian")
        # ... and every caller gets its own recording, in order.
        self.assertEqual([int(result[0][0]) for result in results], [1, 2, 3])
        self.assertEqual({result[1] for result in results}, {22050})
        self.assertEqual(results[0][2]["vram_peak_mb"], 9000)

    def test_an_empty_batch_never_reaches_the_worker(self):
        worker = voicelab.CloneWorker("qwen3tts", "cuda")
        with mock.patch.object(worker, "_ensure_proc") as started:
            self.assertEqual(worker.synthesize_batch([]), [])
        started.assert_not_called()

    def test_a_short_answer_is_an_error_not_a_silent_lie(self):
        worker = voicelab.CloneWorker("qwen3tts", "cuda")
        payloads = self._payload([1, 2])
        with mock.patch.object(worker, "_ensure_proc"), \
                mock.patch.object(worker, "_request",
                                  return_value={"ok": True, "wavs": payloads,
                                                "sample_rate": 24000}):
            with self.assertRaises(voicelab.VoicelabError) as caught:
                worker.synthesize_batch([self._request("a"),
                                         self._request("b"),
                                         self._request("c")])
        self.assertIn("2 recording(s) for 3 segment(s)", str(caught.exception))

    def test_an_engine_error_is_raised_to_every_caller(self):
        worker = voicelab.CloneWorker("qwen3tts", "cuda")
        with mock.patch.object(worker, "_ensure_proc"), \
                mock.patch.object(worker, "_request",
                                  return_value={"ok": False,
                                                "error": "CUDA out of memory"}):
            with self.assertRaises(voicelab.VoicelabError) as caught:
                worker.synthesize_batch([self._request("a"),
                                         self._request("b")])
        self.assertIn("out of memory", str(caught.exception))

    def test_the_report_says_what_the_card_needed(self):
        payload = base64.b64encode(np.zeros(4, dtype=np.int16).tobytes()).decode(
            "ascii"
        )
        samples, rate, report = voicelab.CloneWorker._read_audio(
            {"ok": True, "wav": payload, "sample_rate": 24000,
             "applied": ["temperature"], "vram_peak_mb": 7800}
        )
        self.assertEqual(report["vram_peak_mb"], 7800)
        self.assertEqual(report["applied"], ["temperature"])
        self.assertEqual((rate, int(samples[0])), (24000, 0))
        # An engine that could not report a peak (a CPU run) says zero.
        _samples, _rate, silent = voicelab.CloneWorker._read_audio(
            {"ok": True, "wav": payload, "sample_rate": 24000}
        )
        self.assertEqual(silent["vram_peak_mb"], 0)


class WorkerProtocolStreamTest(unittest.TestCase):
    """The answer stream must survive engines that write where they should not.

    Regression: the first batched Qwen3-TTS call failed with "Invalid response
    from the Voice Lab worker" because importing the package makes its SoX
    wrapper print a banner - and importing SoX's subprocess output goes to the
    process's descriptor 1, which ``sys.stdout`` redirection cannot catch.  One
    stray line shifted every answer after it by one request.
    """

    def test_a_child_writing_to_the_standard_output_misses_the_protocol(self):
        code = (
            "import importlib.util, subprocess, sys;\n"
            "spec = importlib.util.spec_from_file_location('probe', sys.argv[1]);\n"
            "module = importlib.util.module_from_spec(spec);\n"
            "spec.loader.exec_module(module);\n"
            "out = module._protocol_stream();\n"
            "subprocess.run([sys.executable, '-c', \"print('noise from a child')\"]);\n"
            "print('noise from the worker');\n"
            "out.write('{\\\"ok\\\": true}\\n');\n"
            "out.flush()\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code, WORKER], capture_output=True,
            text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        # Only the protocol reaches the application's pipe ...
        self.assertEqual(done.stdout, '{"ok": true}\n')
        # ... and everything else lands on the stderr the app discards.
        self.assertIn("noise from a child", done.stderr)
        self.assertIn("noise from the worker", done.stderr)

    def test_the_request_loop_uses_it(self):
        source = open(WORKER, encoding="utf-8").read()
        self.assertIn("out = _protocol_stream()", source)

    def test_a_blank_line_from_the_worker_is_not_an_answer(self):
        worker = voicelab.CloneWorker("f5tts", "cpu")
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.stdout.readline.side_effect = ["\n", "  \n",
                                           '{"ok": true, "engines": {}}\n']
        worker._proc = proc
        self.assertEqual(worker._request({"cmd": "ping"}),
                         {"ok": True, "engines": {}})


class WorkerBatchLoopTest(unittest.TestCase):
    """The worker's request loop with a batching engine behind it."""

    def setUp(self):
        self.worker = _load_worker()

    def _run(self, request, fake):
        """Drive ``main()`` with the answer stream pointed at this test.

        The worker writes its protocol to a descriptor of its own (see
        ``_protocol_stream``), which a redirected ``sys.stdout`` cannot capture,
        so the stream itself is what is stood in for here - the separation of
        engine noise from it has its own test (``WorkerProtocolStreamTest``).
        """
        out = io.StringIO()
        with mock.patch.dict(self.worker._BACKENDS, {"fake": fake}), \
                mock.patch.object(self.worker, "_protocol_stream",
                                  lambda: out), \
                mock.patch.object(sys, "stdin",
                                  io.StringIO(json.dumps(request) + "\n")):
            self.assertEqual(self.worker.main(), 0)
        return json.loads(out.getvalue().strip().splitlines()[0])

    def test_a_list_of_texts_comes_back_as_a_list_of_recordings(self):
        worker = self.worker
        seen = {}

        class Fake(worker.Backend):
            name = "fake"
            batch_limit = 4

            def synthesize(self, text, request):
                seen["single"] = text
                return np.full(4, 9, dtype=np.int16), 16000, False

            def synthesize_many(self, texts, request):
                seen["batch"] = list(texts)
                return [(np.full(4, index, dtype=np.int16), 16000, False)
                        for index, _text in enumerate(texts)]

        response = self._run(
            {"cmd": "synthesize", "engine": "fake", "device": "cpu",
             "texts": ["one", "two", "three"]},
            Fake,
        )
        self.assertTrue(response["ok"], response)
        self.assertEqual(seen["batch"], ["one", "two", "three"])
        self.assertNotIn("single", seen)
        self.assertEqual(len(response["wavs"]), 3)
        self.assertEqual(response["wav"], response["wavs"][0])
        decoded = [np.frombuffer(base64.b64decode(wav), dtype=np.int16)
                   for wav in response["wavs"]]
        self.assertEqual([int(item[0]) for item in decoded], [0, 1, 2])
        self.assertEqual(response["sample_rate"], 16000)

    def test_a_single_text_still_takes_the_single_call(self):
        worker = self.worker
        seen = {}

        class Fake(worker.Backend):
            name = "fake"

            def synthesize(self, text, request):
                seen["single"] = text
                return np.full(4, 1, dtype=np.int16), 16000, False

        response = self._run(
            {"cmd": "synthesize", "engine": "fake", "device": "cpu",
             "text": "just one"},
            Fake,
        )
        self.assertTrue(response["ok"], response)
        self.assertEqual(seen["single"], "just one")
        self.assertEqual(len(response["wavs"]), 1)

    def test_a_blank_text_in_a_batch_is_refused(self):
        worker = self.worker

        class Fake(worker.Backend):
            name = "fake"

            def synthesize_many(self, texts, request):
                raise AssertionError("nothing may be spoken")

        response = self._run(
            {"cmd": "synthesize", "engine": "fake", "device": "cpu",
             "texts": ["fine", "  "]},
            Fake,
        )
        self.assertFalse(response["ok"])
        self.assertIn("Nothing to synthesize", response["error"])

    def test_the_request_texts_helper_understands_both_shapes(self):
        self.assertEqual(self.worker._request_texts({"text": "one"}), ["one"])
        self.assertEqual(self.worker._request_texts({"texts": ["a", "b"]}),
                         ["a", "b"])
        self.assertEqual(self.worker._request_texts({"texts": "one"}), ["one"])
        for request in ({"text": " "}, {"texts": []}, {}):
            with self.assertRaises(ValueError):
                self.worker._request_texts(request)


class TransformerGpuOnlyWiringTest(unittest.TestCase):
    """The application half of the GPU-only engines."""

    def test_the_worker_and_the_app_agree_on_the_cpu_only_error(self):
        worker = _load_worker()
        self.assertIn("chatterbox", worker._BACKENDS)
        self.assertIn("qwen3tts", worker._BACKENDS)
        for engine_id in ("chatterbox", "qwen3tts", "vibevoice"):
            self.assertTrue(voice_lab.gpu_only(engine_id))

    def test_two_variants_never_share_one_loaded_model(self):
        worker = _load_worker()
        first = worker._backend_key("chatterbox", "cuda",
                                   {"variant": "chatterbox_v1"})
        second = worker._backend_key("chatterbox", "cuda",
                                    {"variant": "chatterbox_mtl"})
        self.assertNotEqual(first, second)
        self.assertEqual(
            worker._backend_key("qwen3tts", "cuda", {"variant": "a"}),
            ("qwen3tts", "cuda", "a"),
        )

    def test_a_gpu_only_engine_always_resolves_to_the_gpu(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            for engine_id in ("chatterbox", "qwen3tts", "vibevoice"):
                self.assertEqual(voicelab.resolve_device("cpu", engine_id),
                                 "cuda")
                self.assertEqual(voicelab.resolve_device("auto", engine_id),
                                 "cuda")
            # The other engines keep their CPU fallback when there is no card.
            self.assertEqual(voicelab.resolve_device("cuda", "f5tts"), "cpu")

    def test_the_cloned_voice_lands_on_the_model_that_can_speak_it(self):
        store = _FakeVoiceStore()
        sample = os.path.join(tempfile.mkdtemp(), "reference.wav")
        self.addCleanup(shutil.rmtree, os.path.dirname(sample),
                        ignore_errors=True)
        _write_silence(sample, frames=240)
        chatterbox = voicelab.create_clone(
            store, "chatterbox", "Reader", sample, language="de"
        )
        self.assertEqual(chatterbox["variant"],
                         voice_lab.CHATTERBOX_MULTILINGUAL_VARIANT)
        qwen = voicelab.create_clone(store, "qwen3tts", "Narrator", sample)
        self.assertEqual(qwen["variant"], "qwen3_clone_1_7b")

    def test_a_gpu_only_voice_asks_the_worker_for_the_gpu(self):
        voice = dict(voice_lab.builtin_voices("qwen3tts")[0])
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            request = voicelab.build_synthesize_request(
                engine_id="qwen3tts", device="cpu", text="Hello",
                voice_entry=voice,
            )
        self.assertEqual(request["device"], "cuda")
        self.assertEqual(request["engine"], "qwen3tts")


class _FakeVoiceStore:
    """The few ``ModelStore`` calls ``create_clone`` makes."""

    def __init__(self):
        self.voices = {}

    def custom_dir(self, name):
        folder = tempfile.mkdtemp(prefix="aivs_clone_")
        self._dirs = getattr(self, "_dirs", [])
        self._dirs.append(folder)
        return folder

    def add_custom_voice(self, name, tts, directory, kind, extra=None):
        self.voices[name] = {"name": name, "tts": tts, "dir": directory,
                             "kind": kind, **(extra or {})}

    def custom_voices(self, engine_id=None):
        return [v for v in self.voices.values()
                if engine_id is None or v.get("tts") == engine_id]


class PocketVoiceCloningAccessTest(unittest.TestCase):
    """Pocket TTS's *voice-cloning* weights are gated on Hugging Face.

    Kyutai's library does not fail when they cannot be downloaded: it quietly
    loads the cloning-free weights instead, and only then - if a clone is asked
    for - answers with advice a person who installed the setup file cannot
    follow ("make sure you're logged in locally with ``uvx hf auth login``").
    The pre-made voices keep working, which is exactly why the user believes the
    engine is fine and looks for the fault in the clone page.  These tests hold
    the worker and the application to the two steps that do work: accept the
    licence on the repository's page, and give the application a read token.
    """

    #: Kyutai's own sentence (``pocket_tts/models/tts_model.py``), which is what
    #: the user actually read.
    LIBRARY_WORDING = (
        "If you want access to the model with voice cloning, go to "
        "https://huggingface.co/kyutai/pocket-tts and accept the terms, then "
        "make sure you're logged in locally with `uvx hf auth login`."
    )

    def _backend(self, worker, *, has_voice_cloning=False):
        """A PocketBackend with no pocket-tts import and no weights loaded."""
        backend = worker.PocketBackend.__new__(worker.PocketBackend)
        backend.device = "cpu"
        backend.request = {}
        backend.report = worker.OptionReport()
        backend.notes = []
        backend.has_voice_cloning = has_voice_cloning
        backend._states = {}
        backend.sample_rate = 24000
        return backend

    def test_the_engine_declares_the_gated_repository_it_clones_from(self):
        self.assertEqual(voice_lab.gated_repo("pocket_tts"),
                         "kyutai/pocket-tts")
        self.assertEqual(voicelab.gated_repo("pocket_tts"),
                         "kyutai/pocket-tts")
        # The engines that download from a gated repository name it, and every
        # other engine names nothing at all.
        gated = {
            "pocket_tts": "kyutai/pocket-tts",
            "indic_parler": "ai4bharat/indic-parler-tts",
        }
        for engine_id in voice_lab.engine_ids():
            self.assertEqual(voice_lab.gated_repo(engine_id),
                             gated.get(engine_id, ""), engine_id)

    def test_a_pre_made_voice_is_not_a_clone(self):
        worker = _load_worker()
        backend = self._backend(worker)
        self.assertFalse(backend._wants_cloned_voice({"voice": "alba"}))
        self.assertTrue(backend._wants_cloned_voice({"ref_audio": "C:/v.wav"}))
        self.assertTrue(backend._wants_cloned_voice({"ref_resource": "pocket:v"}))

    def test_a_clone_without_the_gated_weights_names_the_page_and_the_panel(self):
        worker = _load_worker()
        backend = self._backend(worker)
        with self.assertRaises(RuntimeError) as caught:
            backend.synthesize("Hello", {"ref_audio": "C:/voice.wav"})
        message = str(caught.exception)
        self.assertIn("https://huggingface.co/kyutai/pocket-tts", message)
        self.assertIn("Settings > Hugging Face account", message)
        self.assertIn("licence", message)
        # Not the library's advice: this installation has no `hf` command.
        self.assertNotIn("hf auth login", message)

    def test_the_librarys_own_refusal_becomes_the_two_steps(self):
        worker = _load_worker()

        wording = self.LIBRARY_WORDING

        class _Model:
            has_voice_cloning = True

            def get_state_for_audio_prompt(self, _prompt):
                raise RuntimeError(wording)

        backend = self._backend(worker, has_voice_cloning=True)
        backend.model = _Model()
        with self.assertRaises(RuntimeError) as caught:
            backend._state("C:/voice.wav")
        self.assertIn("https://huggingface.co/kyutai/pocket-tts",
                      str(caught.exception))

    def test_a_pre_made_voice_still_speaks_after_the_fallback_load(self):
        """Loading without cloning is not a failure: only clones are refused."""
        worker = _load_worker()
        backend = self._backend(worker)
        spoken = {}

        def _state(prompt):
            spoken["prompt"] = prompt
            return "state"

        backend._state = _state
        backend.model = mock.Mock()
        backend.model.generate_audio.return_value = "audio"
        audio, rate, handled = backend.synthesize("Hello", {"voice": "alba"})
        self.assertEqual((audio, rate, handled), ("audio", 24000, False))
        self.assertEqual(spoken["prompt"], "alba")

    def test_loading_without_the_gated_weights_is_noted(self):
        worker = _load_worker()
        fake = types.ModuleType("pocket_tts")

        class _Model:
            has_voice_cloning = False
            sample_rate = 24000

        fake.TTSModel = types.SimpleNamespace(load_model=lambda **kw: _Model())
        with mock.patch.dict(sys.modules, {"pocket_tts": fake}):
            backend = worker.PocketBackend("cpu", {"language": "en"})
        self.assertFalse(backend.has_voice_cloning)
        self.assertTrue(any("without voice cloning" in note
                            for note in backend.notes))

    def test_the_workers_own_wording_repeats_the_same_two_steps(self):
        # The worker is a standalone script that imports no part of the
        # application, so its copy has to be checked on its own.
        worker = _load_worker()
        for text in (worker._POCKET_NO_CLONING_HELP, worker._POCKET_NO_CLONING_NOTE):
            self.assertIn("https://huggingface.co/kyutai/pocket-tts", text)
        self.assertIn("Settings > Hugging Face account",
                      worker._POCKET_NO_CLONING_HELP)

    def test_a_gated_failure_from_the_worker_is_translated_for_the_user(self):
        message = voicelab._error_text(
            "pocket_tts",
            "RuntimeError: 401 Client Error. Cannot access gated repo for "
            "https://huggingface.co/kyutai/pocket-tts/resolve/abc. "
            "Please log in.")
        self.assertIn("https://huggingface.co/kyutai/pocket-tts", message)
        self.assertIn("Settings > Hugging Face account", message)
        # The engine is named too: the panel is the same for every gated model.
        self.assertIn("Pocket TTS", message)

    def test_an_unrelated_engine_failure_is_left_exactly_as_it_was(self):
        self.assertEqual(
            voicelab._error_text("pocket_tts", "ValueError: bad zip file"),
            "ValueError: bad zip file")
        self.assertEqual(voicelab._error_text("pocket_tts", ""),
                         "Synthesis failed.")

    def test_the_application_names_the_engine_in_a_one_off_failure(self):
        with self.assertRaises(voicelab.VoicelabError) as caught:
            voicelab.CloneWorker._read_audio(
                {"ok": False, "error": f"RuntimeError: {self.LIBRARY_WORDING}"},
                "pocket_tts")
        message = str(caught.exception)
        self.assertIn("https://huggingface.co/kyutai/pocket-tts", message)
        self.assertIn("Settings > Hugging Face account", message)


if __name__ == "__main__":
    unittest.main()
