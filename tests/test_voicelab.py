"""Tests for the Voice Lab (Pocket TTS / Bark / F5-TTS).

No engine package and no GPU is required: the registry, the device selection,
the request builder, the clone store integration and the worker protocol are
all exercised with the standard-library interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import voicelab  # noqa: E402
from ai_voice_studio.tts import catalog  # noqa: E402
from ai_voice_studio.tts.models import ModelStore, resolve_voice_files  # noqa: E402
from ai_voice_studio.voicelab import engines as voice_lab  # noqa: E402
from ai_voice_studio.voicelab import options as tuning  # noqa: E402

WORKER = os.path.join(
    os.path.dirname(os.path.abspath(voicelab.__file__)), "worker.py"
)


class RegistryTest(unittest.TestCase):
    def test_the_engines_are_registered(self):
        self.assertEqual(
            voice_lab.engine_ids(),
            ("pocket_tts", "bark", "f5tts"),
        )
        # NeuTTS was dropped: it needs a HuggingFace login to fetch its model.
        self.assertNotIn("neutts", voice_lab.engine_ids())
        for engine_id in voice_lab.engine_ids():
            info = voice_lab.engine(engine_id)
            self.assertTrue(info["name"], engine_id)
            self.assertTrue(info["voice_cloning"], engine_id)
            self.assertEqual(info["devices"], "cpu+gpu", engine_id)
            self.assertTrue(voice_lab.packages(engine_id), engine_id)

    def test_catalog_exposes_the_engines(self):
        ids = {tts["id"] for tts in catalog.get_tts_list()}
        for engine_id in voice_lab.engine_ids():
            self.assertIn(engine_id, ids)
        cloning = {tts["id"] for tts in catalog.cloning_capable_tts()}
        for engine_id in voice_lab.engine_ids():
            self.assertIn(engine_id, cloning)
        bark = catalog.find_tts("bark")
        self.assertIn("transformers", catalog.requires_packages(bark))
        self.assertIn("torch", catalog.requires_packages(bark))

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
        # Ten speaker presets for each of the twelve languages, plus the ten
        # English speakers of suno/bark-small.
        self.assertEqual(voice_lab.count_builtin_voices("bark"), 130)

    def test_bark_presets_cover_every_language(self):
        voices = voice_lab.builtin_voices("bark")
        presets = {v["voice"] for v in voices}
        self.assertIn("v2/en_speaker_0", presets)
        self.assertIn("v2/hi_speaker_9", presets)
        self.assertIn("v2/zh_speaker_4", presets)
        small = [v for v in voices if v["variant"] == "bark_small"]
        self.assertEqual(len(small), 10)
        self.assertTrue(all(v.get("repo") == "suno/bark-small" for v in small))

    def test_f5_voices_point_at_package_references(self):
        for voice in voice_lab.builtin_voices("f5tts"):
            self.assertTrue(voice["ref_resource"].startswith("f5_tts:"))
            self.assertTrue(voice["ref_text"])


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
                self.assertEqual(request["device"], "cpu")
                self.assertTrue(request["voice"], voice)
                self.assertNotIn("None", payload)

    def test_bark_request_picks_the_model_repo(self):
        small = next(v for v in voice_lab.builtin_voices("bark")
                     if v["variant"] == "bark_small")
        request = voicelab.build_synthesize_request(
            engine_id="bark", device="cpu", text="Hi", voice_entry=small
        )
        self.assertEqual(request["repo"], "suno/bark-small")
        big = voice_lab.builtin_voices("bark")[0]
        request = voicelab.build_synthesize_request(
            engine_id="bark", device="cpu", text="Hi", voice_entry=big
        )
        self.assertEqual(request["repo"], "suno/bark")


class GpuSupportTest(unittest.TestCase):
    """Every Voice Lab engine can additionally run on an NVIDIA GPU."""

    def test_every_engine_supports_the_gpu(self):
        for engine_id in voice_lab.engine_ids():
            self.assertTrue(voice_lab.gpu_supported(engine_id), engine_id)
            cuda = voice_lab.cuda_packages(engine_id)
            self.assertTrue(cuda, engine_id)
            self.assertIn("torch", cuda, engine_id)

    def test_the_gpu_torch_comes_from_the_pytorch_wheel_index(self):
        for engine_id in voice_lab.engine_ids():
            self.assertEqual(voice_lab.cuda_index_url(engine_id),
                             voice_lab.PYTORCH_CUDA_INDEX, engine_id)

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
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for found in re.findall(
                r"https?://download\.pytorch\.org/whl/cu\w+", text
            ):
                if found != voice_lab.PYTORCH_CUDA_INDEX:
                    offenders.append((path.name, found))
        self.assertEqual(offenders, [],
                         "hard-code the index via PYTORCH_CUDA_INDEX instead")

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
            self.assertEqual(plan[0][1], voice_lab.PYTORCH_CUDA_INDEX, engine_id)
            self.assertIn("torch", plan[0][0], engine_id)
            # Everything else is still installed, from PyPI.
            installed = {pkg for step, _index in plan for pkg in step}
            for package in voice_lab.packages(engine_id):
                self.assertIn(package, installed, (engine_id, package))

    def test_a_cpu_machine_installs_the_plain_wheels(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            plan = voicelab.install_plan("f5tts")
        self.assertEqual(plan,
                         [(list(voice_lab.packages("f5tts")), None)])

    def test_the_cpu_budget_travels_with_the_request(self):
        from ai_voice_studio import compute

        voice = dict(voice_lab.builtin_voices("bark")[0])
        request = voicelab.build_synthesize_request(
            engine_id="bark", device="cpu", text="Hi", voice_entry=voice,
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

    def test_a_cpu_machine_installs_every_package_in_one_step(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            for engine_id in voice_lab.engine_ids():
                plan = voicelab.install_plan(engine_id)
                self.assertEqual(
                    plan, [(list(voicelab.packages_for(engine_id)), None)],
                    engine_id,
                )

    def test_a_gpu_machine_installs_cuda_torch_first(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            for engine_id in voice_lab.engine_ids():
                plan = voicelab.install_plan(engine_id)
                cuda = list(voice_lab.cuda_packages(engine_id))
                self.assertTrue(cuda, engine_id)
                self.assertEqual(plan[0], (cuda, voice_lab.PYTORCH_CUDA_INDEX))
                # ... and the same packages are not requested twice.
                for package in cuda:
                    self.assertNotIn(package, plan[1][0], engine_id)

    def test_the_worker_and_the_app_agree_on_what_each_engine_imports(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("aivs_voicelab_worker", WORKER)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        for engine_id in voice_lab.engine_ids():
            self.assertEqual(
                tuple(worker._ENGINE_MODULES.get(engine_id, ())),
                voice_lab.import_modules(engine_id),
                engine_id,
            )


class EngineVerificationTest(unittest.TestCase):
    """An installed engine is imported once, so a broken env says so."""

    class _Runtime:
        def __init__(self, python_exe, created=True):
            self.python_exe = python_exe
            self.is_created = created

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
            self.assertIsNone(voicelab.verify_engine_import("bark"))

    def test_a_broken_engine_reports_the_import_error(self):
        # A module that cannot be imported stands in for an environment pip
        # "installed" into but which the engine cannot load.
        self._patch_runtime(sys.executable)
        with mock.patch.object(voice_lab, "import_modules",
                               return_value=("aivs_no_such_module",)):
            problem = voicelab.verify_engine_import("bark")
        self.assertIsNotNone(problem)
        self.assertIn("aivs_no_such_module", problem)
        self.assertIn(voice_lab.engine_name("bark"), problem)

    def test_a_missing_environment_is_not_a_problem(self):
        self._patch_runtime(sys.executable, created=False)
        self.assertIsNone(voicelab.verify_engine_import("bark"))


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

    def test_bark_only_clones_from_a_speaker_embedding(self):
        with self.assertRaises(ValueError) as ctx:
            voicelab.create_clone(self.store, "bark", "Bark clone", self.sample)
        self.assertIn(".npz", str(ctx.exception))
        embedding = os.path.join(self.tmp.name, "speaker.npz")
        with open(embedding, "wb") as fh:
            fh.write(b"npz")
        entry = voicelab.create_clone(self.store, "bark", "Bark clone", embedding)
        self.assertTrue(os.path.isfile(entry["ref_audio"]))

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


class EngineWiringTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(voicelab.close_workers)

    def test_get_engine_returns_the_voice_lab_wrapper(self):
        from ai_voice_studio.tts.engine import get_engine

        voice = dict(voice_lab.builtin_voices("bark")[0])
        voice["engine"] = "bark"
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            engine = get_engine(voice, provider="cpu")
        self.assertIsInstance(engine, voicelab.VoicelabEngine)
        self.assertEqual(engine.engine_id, "bark")
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
        bark = {option.key: option.default for option in tuning.specs("bark")}
        self.assertAlmostEqual(bark["text_temp"], 0.7)
        self.assertAlmostEqual(bark["waveform_temp"], 0.7)
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
        self.assertEqual(
            tuning.clean("bark", {"text_temp": "0"}), {"text_temp": 0.0}
        )
        self.assertEqual(
            tuning.clean("pocket_tts", {"quantize": "true"}),
            {"quantize": True},
        )
        with self.assertRaises(ValueError):
            tuning.clean("f5tts", {"nfe_step": "many"})
        with self.assertRaises(ValueError):
            tuning.clean("bark", {"text_temp": "hot"})

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
            engine_id="bark", device="cpu", text="Hi",
            voice_entry=dict(voice_lab.builtin_voices("bark")[0]),
            tuning=tuning.defaults("bark"),
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
                ):
            with self.assertRaises(voicelab.VoicelabError) as ctx:
                engine.synthesize("Hello")
        self.assertIn("boom", str(ctx.exception))

    def test_nothing_is_synthesized_for_blank_text(self):
        engine = self._engine("bark")
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
            "cmd": "synthesize", "engine": "bark", "device": "cpu", "text": "  ",
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


class WorkerTuningTest(unittest.TestCase):
    """How the worker applies and reports the tuning options."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("aivs_voicelab_worker", WORKER)
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)

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
            key("bark", "cpu", {"repo": "suno/bark"}),
            key("bark", "cpu", {"repo": "suno/bark"}),
        )
        self.assertNotEqual(
            key("bark", "cpu", {"repo": "suno/bark"}),
            key("bark", "cpu", {"repo": "suno/bark-small"}),
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
    environment and Bark died with "ModuleNotFoundError: No module named
    'torch'", a module its own environment does have.
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
        with open(os.path.join(env_dir, "Scripts", "python.exe"), "w") as fh:
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
            "bark", ["torch", "transformers"], ["numpy"],
        )
        # The cache really is cold: this is the state right after an install.
        self.assertFalse(self.vp.is_known("torch", engine="bark"))
        self.assertIs(worker._runtime(), own)

    def test_an_engine_installed_in_the_shared_environment_still_works(self):
        worker, own, shared = self._worker(
            "bark", [], ["torch", "transformers"],
        )
        self.assertFalse(own.is_created)  # never created: a pre-per-TTS install
        self.assertIs(worker._runtime(), shared)

    def test_a_partly_installed_engine_says_so_instead_of_failing_later(self):
        worker, _own, _shared = self._worker(
            "bark", ["transformers"], ["numpy"],  # torch is missing
        )
        with self.assertRaises(voicelab.VoicelabError) as caught:
            worker._runtime()
        self.assertIn(voice_lab.engine_name("bark"), str(caught.exception))

    def test_every_engine_is_checked_with_its_own_import_modules(self):
        for engine_id in voice_lab.engine_ids():
            modules = list(voice_lab.import_modules(engine_id))
            worker, own, _shared = self._worker(engine_id, modules, ["numpy"])
            self.assertIs(worker._runtime(), own, engine_id)


if __name__ == "__main__":
    unittest.main()
