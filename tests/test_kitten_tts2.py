"""Kitten TTS 2.0 (KittenML), the 1.7B speech-language Voice Lab engine.

**Kitten TTS 2.0** (``KittenML/kitten-tts-2``, the package ``kittenml``) is the
successor to the small ONNX voices the catalogue already ships under the id
``kitten``.  It is deliberately a *second* entry rather than a new variant of
that one: it is a different kind of engine (a 1.7B speech language model driven
from a Python package, not a downloadable ONNX artifact), it has its own voices,
its own languages and its own Python environment, and the ONNX entry keeps
working exactly as before.

What these tests hold down, in the order they would break:

* the registry the application reads (packages, CUDA wheels, import modules and
  the environment it installs into) and the worker's own copy of the voice list
  and module list - the worker runs in its own interpreter and cannot import the
  application, so the two are duplicated and must not drift;
* the ONNX Kitten entry is *not* disturbed by adding this engine;
* the request a voice builds and the arguments that reach the engine, checked
  against a fake package rather than a real one (no 1.7B checkpoint is
  downloaded to run a test);
* the device policy: the GPU is what the engine is offered on, and only a
  measurement on this computer can add the CPU.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import voicelab
from ai_voice_studio.voicelab import engines as voice_lab
from ai_voice_studio.voicelab import options as tuning

WORKER = os.path.join(
    os.path.dirname(os.path.abspath(voicelab.__file__)), "worker.py"
)

ENGINE = "kitten_tts_2"

#: The eight voices KittenTTS 0.8 already had.  The project keeps their names
#: stable on purpose, so code written against the old model keeps working.
LEGACY_VOICES = ("Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki",
                 "Leo")

#: The nine voices that are named after the language their clip carries.
LANGUAGE_VOICES = ("Arabic", "Chinese", "French", "German", "Hindi",
                   "Italian", "Portuguese", "Russian", "Spanish")


def _load_worker():
    """The worker module, loaded standalone (it must not import the app)."""
    spec = importlib.util.spec_from_file_location("aivs_kitten_worker", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


class RegistryTest(unittest.TestCase):
    """What the application reads about the engine."""

    def test_the_engine_is_registered_and_reachable(self):
        from ai_voice_studio.tts import catalog

        listed = {tts["id"] for tts in catalog.get_tts_list()}
        self.assertTrue(voice_lab.is_engine(ENGINE))
        self.assertIn(ENGINE, listed)
        info = voice_lab.engine(ENGINE)
        self.assertEqual(info["name"], "Kitten TTS 2.0 (KittenML)")
        self.assertTrue(info["note"], "a note explains the engine")
        self.assertTrue(info["license"])
        self.assertTrue(info["homepage"].startswith("https://"))
        self.assertTrue(info["size_hint"])
        self.assertTrue(voice_lab.packages(ENGINE))
        self.assertTrue(tuning.specs(ENGINE), "it needs a tuning dialog")
        # It is a cloning engine: a voice comes from a recording as well as
        # from the package's own reference clips.
        self.assertTrue(info["voice_cloning"])
        self.assertIn(ENGINE, {tts["id"] for tts in catalog.cloning_capable_tts()})

    def test_it_declares_the_expression_markup_it_reads(self):
        """The text pipeline asks the registry, so the engine must say so.

        Its own module (``kittenml.kittentts2.text``) treats a leading
        ``[emotion]`` tag, inline ``<event>`` tags and ``(((emphasis)))`` spans
        as director's notes rather than words to pronounce; the application
        keeps them out of the punctuation modes because of this flag.
        """
        self.assertTrue(voice_lab.expression_markup(ENGINE))
        self.assertFalse(voice_lab.expression_markup("f5tts"))
        self.assertFalse(voice_lab.expression_markup("kitten"))
        # The note a user reads names them too.
        note = voice_lab.engine(ENGINE)["note"]
        for mention in ("[emotion]", "<event>", "(((emphasis)))"):
            self.assertIn(mention, note)

    def test_it_gets_its_own_python_environment(self):
        """Two TTS engines, two environments: nothing is shared."""
        from ai_voice_studio import python_runtime as pr

        self.assertTrue(voice_lab.own_environment(ENGINE))
        self.assertEqual(pr.environment_id(ENGINE), ENGINE)
        folder = pr.engine_env_dir(ENGINE)
        self.assertIn("tts_envs", folder)
        self.assertEqual(os.path.basename(folder), ENGINE)
        # ... and it is not the ONNX engine's, the shared addon environment's,
        # or another Voice Lab engine's.
        others = {
            pr.engine_env_dir(other)
            for other in ("kitten", "f5tts", "kokoclone", "indic_parler", "")
        }
        self.assertNotIn(folder, others)

    def test_the_onnx_kitten_entry_is_not_touched(self):
        """The new engine is a *second* entry; the 0.8 ONNX one stays as it is."""
        from ai_voice_studio.tts import catalog

        onnx = catalog.find_tts("kitten")
        self.assertIsNotNone(onnx)
        self.assertEqual(onnx["engine"], "kitten")
        self.assertIsNone(catalog.engine_env_id(onnx),
                          "the ONNX engine is not pip-installed")
        variants = onnx["languages"][0]["variants"]
        self.assertEqual(
            [v["id"] for v in variants],
            ["v0_8_nano_int8", "v0_8_nano_fp32", "v0_8_micro", "v0_8_mini"],
        )
        new = catalog.find_tts(ENGINE)
        self.assertNotEqual(new["engine"], onnx["engine"])
        self.assertEqual(catalog.engine_env_id(new), ENGINE)

    def test_nothing_here_needs_a_hugging_face_account(self):
        """Its weights are public; the token panel must not list it."""
        from ai_voice_studio import hf_access

        self.assertEqual(voice_lab.gated_repo(ENGINE), "")
        for row in hf_access.gated_repos():
            self.assertNotIn("kitten", row["repo"].lower(), row)
        # The two engines that *are* gated still are: adding a public engine
        # must not have disturbed the registry.
        self.assertEqual(voice_lab.gated_repo("pocket_tts"),
                         "kyutai/pocket-tts")
        self.assertEqual(voice_lab.gated_repo("indic_parler"),
                         "ai4bharat/indic-parler-tts")


class VoicesTest(unittest.TestCase):
    """The 47 built-in voices and the ten languages they cover."""

    def test_the_engine_ships_its_47_builtin_voices(self):
        voices = voice_lab.builtin_voices(ENGINE)
        self.assertEqual(len(voices), 47)
        self.assertEqual(voice_lab.count_builtin_voices(ENGINE), 47)
        names = {voice["voice"] for voice in voices}
        for legacy in LEGACY_VOICES:
            self.assertIn(legacy, names, legacy)
        for language_voice in LANGUAGE_VOICES:
            self.assertIn(language_voice, names, language_voice)
        # Every voice sits in a language/variant the engine declares, on the
        # one variant there is.
        declared = {
            (lang["code"], variant["id"])
            for lang in voice_lab.engine(ENGINE)["languages"]
            for variant in lang["variants"]
        }
        for voice in voices:
            self.assertEqual(voice["engine"], ENGINE)
            self.assertEqual(voice["variant"], voice_lab.KITTEN2_VARIANT)
            self.assertIn((voice["language"], voice["variant"]), declared,
                          voice["voice"])
            self.assertTrue(voice["voice_name"], voice["voice"])
            self.assertNotIn("device", voice)

    def test_each_language_voice_carries_its_language(self):
        by_language = {
            voice["language"]: voice["voice"]
            for voice in voice_lab.builtin_voices(ENGINE)
            if voice["language"] != "en"
        }
        self.assertEqual(
            by_language,
            {"ar": "Arabic", "zh": "Chinese", "fr": "French", "de": "German",
             "hi": "Hindi", "it": "Italian", "pt": "Portuguese",
             "ru": "Russian", "es": "Spanish"},
        )
        # English is the thirty-eight-voice set, so the ten languages are the
        # nine language voices plus it - not the twenty the project advertises.
        self.assertEqual(
            {lang["code"] for lang in voice_lab.engine(ENGINE)["languages"]},
            {"en", *by_language},
        )
        self.assertEqual(
            len([v for v in voice_lab.builtin_voices(ENGINE)
                 if v["language"] == "en"]),
            38,
        )

    def test_every_voice_builds_a_request_the_worker_accepts(self):
        for voice in voice_lab.builtin_voices(ENGINE):
            request = voicelab.build_synthesize_request(
                engine_id=ENGINE, device="cuda", text="Hello",
                voice_entry=voice,
            )
            self.assertEqual(request["engine"], ENGINE)
            self.assertEqual(request["voice"], voice["voice"])
            self.assertEqual(request["language"], voice["language"])
            self.assertEqual(request["variant"], voice_lab.KITTEN2_VARIANT)
            self.assertNotIn("None", repr(request))

    def test_a_clone_request_carries_the_reference_and_its_transcript(self):
        entry = {
            "voice": voice_lab.CLONE_VOICE_ID,
            "language": "de",
            "variant": voice_lab.clone_variant(ENGINE),
            "engine": ENGINE,
            "ref_audio": r"C:\voices\clone\reference.wav",
            "ref_text": "Guten Morgen.",
        }
        request = voicelab.build_synthesize_request(
            engine_id=ENGINE, device="cpu", text="Hello", voice_entry=entry,
        )
        self.assertEqual(request["ref_audio"], entry["ref_audio"])
        self.assertEqual(request["ref_text"], entry["ref_text"])

    def test_a_clone_lands_on_the_one_variant_there_is(self):
        variants = {
            variant["id"]
            for lang in voice_lab.engine(ENGINE)["languages"]
            for variant in lang["variants"]
        }
        self.assertEqual(variants, {voice_lab.KITTEN2_VARIANT})
        self.assertEqual(voice_lab.clone_variant(ENGINE),
                         voice_lab.KITTEN2_VARIANT)


class DevicePolicyTest(unittest.TestCase):
    """The GPU is the engine; the CPU has to earn its place."""

    def setUp(self):
        self.addCleanup(voicelab.clear_cpu_rtf)
        voicelab.clear_cpu_rtf()

    def test_the_engine_is_gpu_only_until_it_is_measured(self):
        self.assertTrue(voice_lab.gpu_only(ENGINE))
        self.assertFalse(voice_lab.cpu_supported(ENGINE))
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(
                [value for value, _label in voicelab.device_options(ENGINE)],
                ["cuda"],
            )
            self.assertEqual(voicelab.resolve_device("cpu", ENGINE), "cuda")
        self.assertIsNone(voicelab.cpu_verdict(ENGINE))

    def test_a_measured_fast_cpu_is_offered_as_well(self):
        voicelab.set_cpu_rtf(ENGINE, 1.5)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            values = [value for value, _label in voicelab.device_options(ENGINE)]
            self.assertEqual(values, ["cuda", "cpu"])
            self.assertEqual(voicelab.resolve_device("cpu", ENGINE), "cpu")
            labels = dict(voicelab.device_options(ENGINE))
        self.assertIn("measured 1.50x real time", labels["cpu"])

    def test_a_measured_slow_cpu_stays_out_when_a_card_is_present(self):
        voicelab.set_cpu_rtf(ENGINE, 3.4)
        with mock.patch.object(voicelab, "has_cuda", return_value=True):
            self.assertEqual(
                [value for value, _label in voicelab.device_options(ENGINE)],
                ["cuda"],
            )
            self.assertEqual(voicelab.resolve_device("cpu", ENGINE), "cuda")

    def test_the_gpu_wheels_come_from_the_cuda_index(self):
        self.assertEqual(tuple(voice_lab.cuda_packages(ENGINE)),
                         ("torch", "torchaudio"))
        self.assertEqual(voice_lab.cuda_index_url(ENGINE),
                         voice_lab.PYTORCH_CUDA_INDEX)
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.venv_packages.installed",
                           return_value=False):
            plan = voicelab.install_plan(ENGINE)
        cuda, index = plan[0]
        self.assertEqual(index, voice_lab.PYTORCH_CUDA_INDEX)
        self.assertEqual(tuple(cuda), ("torch", "torchaudio"))
        # The by-name step must not ask for them again, or pip would replace
        # the CUDA build with PyPI's CPU-only wheel.
        for arguments, _index in plan[1:]:
            self.assertNotIn("torch", arguments)
            self.assertNotIn("torchaudio", arguments)
        self.assertIn("kittenml", plan[1][0])

    def test_a_cpu_machine_installs_everything_from_pypi_in_one_step(self):
        with mock.patch.object(voicelab, "has_cuda", return_value=False):
            plan = voicelab.install_plan(ENGINE)
        self.assertEqual(plan, [(list(voicelab.packages_for(ENGINE)), None)])
        self.assertIn("kittenml", plan[0][0])

    def test_the_engine_has_tuning_and_its_weights_reload(self):
        keys = {option.key for option in tuning.specs(ENGINE)}
        self.assertIn("preset", keys)
        self.assertIn("normalize", keys)
        self.assertIn("weights", keys)
        self.assertIn("seed", keys)
        # The packing and the decoder pick *files*: a change has to reload.
        self.assertTrue(tuning.spec(ENGINE, "weights").load_time)
        self.assertTrue(tuning.spec(ENGINE, "decoder").load_time)
        self.assertIn("Kitten TTS 2.0", tuning.max_seconds_note(ENGINE))
        self.assertEqual(tuning.clean(ENGINE, tuning.defaults(ENGINE)), {})
        self.assertEqual(
            tuning.clean(ENGINE, {"weights": "packed", "preset": "stable"}),
            {},
            "the engine's own defaults are never stored or sent",
        )


#: The runs ``tools/measure_engine_vram.py kitten_tts_2`` reported on the 12 GB
#: RTX 3060 (2026-10-16) after a warm-up, over 65-character segments: the driver's
#: used video memory, in MiB, in the four phases the tool samples.  They are kept
#: here because the declaration below is *derived* from them - a figure that stops
#: matching its measurement fails ``GpuPlanTest``.
MEASURED_RUNS = (
    {"slots": 1, "idle": 1207, "load": 6115, "loaded": 5818, "peak": 5943,
     "after": 1216, "errors": [], "wall": 87.79, "load_wall": 105.39,
     "results": []},
    {"slots": 2, "idle": 1216, "load": 10597, "loaded": 10415, "peak": 10645,
     "after": 1214, "errors": [], "wall": 45.75, "load_wall": 113.01,
     "results": []},
)


def _measure_tool():
    """The measurement tool, loaded from ``tools/``."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools", "measure_engine_vram.py",
    )
    spec = importlib.util.spec_from_file_location("aivs_measure_vram", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GpuPlanTest(unittest.TestCase):
    """How many segments this engine may speak at once, and where that comes from.

    The plan itself is the application's (``voicelab.parallel_plan``): the card
    decides how many *slots* are affordable, the declaration says what one costs
    and what the engine can do with them.  What is checked here is the half that
    belongs to this engine - the figure, the mode, and the fact that the figure
    still follows from its own measurement.
    """

    #: What was free on the card the figures were measured on.
    RTX_3060_FREE = 10639

    def _plan(self, free_mb, variant=None, device="cuda"):
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.compute.free_vram_mb",
                           return_value=free_mb):
            return voicelab.parallel_plan(
                ENGINE, device, voice_lab.KITTEN2_VARIANT if variant is None
                else variant
            )

    def test_the_engine_declares_one_model_copy_per_segment(self):
        """``generate`` takes one text, so a second segment needs a second copy."""
        declared = voice_lab.gpu_parallel(ENGINE, voice_lab.KITTEN2_VARIANT)
        self.assertIsNotNone(declared, "the engine has a plan of its own")
        self.assertEqual(declared["mode"], "instances")
        self.assertGreater(declared["vram_mb"], 0)
        # It is not a batched engine: the worker's own call takes one text.
        self.assertEqual(declared["per_text_mb"], 0)
        self.assertEqual(declared["chars_per_slot"], 0)
        self.assertEqual(voice_lab.gpu_chars_per_slot(ENGINE), 0)
        self.assertEqual(voicelab.batch_chars(ENGINE, 3), 0)
        self.assertEqual(voice_lab._GPU_PARALLEL[ENGINE]["slots"], {
            voice_lab.KITTEN2_VARIANT: {"vram_mb": declared["vram_mb"]},
        })

    def test_the_declared_figure_is_the_dearest_its_measurement_found(self):
        """The number in the table comes from the runs, not from a guess.

        One copy peaked higher to the driver than the same copy beside a second
        one (4970 MiB against 4654 MiB), so the declaration has to cover the
        dearest of the two, rounded up - a slot must never be charged less than
        the card was seen to use.
        """
        tool = _measure_tool()
        verdict = tool.verdict(MEASURED_RUNS)
        # The per-copy costs the four phases give, spelled out, so the
        # arithmetic between the raw MiB and the declared figure is not hidden:
        self.assertEqual(round(verdict["costs"][1]), 4908)
        self.assertEqual(round(verdict["costs"][2]), 4714)
        self.assertEqual(verdict["ceiling"], 2, "this card held two copies")
        self.assertEqual(verdict["vram_mb"], 5000)
        # What the second copy bought, from the same two runs (1.9x):
        self.assertAlmostEqual(
            MEASURED_RUNS[0]["wall"] / MEASURED_RUNS[1]["wall"], 1.92, places=2
        )
        self.assertEqual(verdict["vram_mb"],
                         voice_lab.gpu_parallel(ENGINE)["vram_mb"])
        self.assertEqual(verdict["failures"], [])
        # A declaration is never cheaper than the dearest measurement.
        self.assertGreaterEqual(verdict["vram_mb"],
                                max(verdict["costs"].values()))

    def test_a_small_card_keeps_one_copy_and_a_big_one_fills_up(self):
        """What the measured figure buys, card by card."""
        # The 12 GB card the figures come from: two copies fit the card but not
        # the budget a recording leaves (free, minus the desktop reserve).
        self.assertEqual(self._plan(self.RTX_3060_FREE), (1, "instances"))
        self.assertEqual(self._plan(12000), (2, "instances"))
        self.assertEqual(self._plan(16000), (2, "instances"))
        # A 24 GB card reaches the ceiling the other one-text-per-call engines
        # declare, and nothing goes past it however large the card is.
        self.assertEqual(self._plan(24576), (3, "instances"))
        self.assertEqual(self._plan(98304), (3, "instances"))
        self.assertEqual(voice_lab._GPU_PARALLEL[ENGINE]["max_slots"], 3)

    def test_a_recording_asks_the_engine_for_what_the_plan_affords(self):
        """The pipeline reads the same number, so several segments stay in flight.

        Deliberately without the variant: the worker pool is shared by every
        voice of one engine, so the recording asks for what that pool was built
        for.
        """
        from ai_voice_studio.tts.engine import planned_workers

        entry = {"engine": ENGINE, "variant": voice_lab.KITTEN2_VARIANT,
                 "device": "cuda"}
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.compute.free_vram_mb",
                           return_value=24576):
            self.assertEqual(planned_workers(entry, "cuda", total=6), 3)
            self.assertEqual(planned_workers(entry, "cuda", total=2), 2)
        with mock.patch.object(voicelab, "has_cuda", return_value=True), \
                mock.patch("ai_voice_studio.compute.free_vram_mb",
                           return_value=self.RTX_3060_FREE):
            self.assertEqual(planned_workers(entry, "cuda", total=6), 1)
        # A GPU-only engine sent to the CPU by a stored setting still runs on
        # the card (it has no CPU mode), so the plan has to answer for the card.
        self.assertEqual(self._plan(24576, device="cpu"), (3, "instances"))

    def test_a_machine_without_a_card_speaks_one_segment_at_a_time(self):
        """No driver to ask means no plan: the single copy it always was."""
        with mock.patch.object(voicelab, "has_cuda", return_value=False), \
                mock.patch("ai_voice_studio.compute.free_vram_mb",
                           return_value=None):
            slots, mode = voicelab.parallel_plan(
                ENGINE, "cpu", voice_lab.KITTEN2_VARIANT
            )
        self.assertEqual((slots, mode), (1, "solo"))


class WorkerAgreementTest(unittest.TestCase):
    """The worker keeps its own copy of the registry; the two must not drift."""

    def test_the_worker_knows_the_engine_and_its_modules(self):
        worker = _load_worker()
        self.assertIn(ENGINE, worker._BACKENDS)
        self.assertEqual(
            tuple(worker._ENGINE_MODULES.get(ENGINE, ())),
            voice_lab.import_modules(ENGINE),
        )
        self.assertEqual(worker._BACKENDS[ENGINE].batch_limit, 1)

    def test_the_worker_and_the_app_agree_on_the_voices(self):
        worker = _load_worker()
        self.assertEqual(
            set(worker._KITTEN2_VOICES),
            {voice["voice"] for voice in voice_lab.builtin_voices(ENGINE)},
        )
        self.assertEqual(len(worker._KITTEN2_VOICES), 47)
        self.assertEqual(
            tuple(worker._KITTEN2_LANGUAGE_VOICES), LANGUAGE_VOICES
        )

    def test_the_backend_cache_key_follows_the_loaded_weights(self):
        worker = _load_worker()
        first = worker._backend_key(
            ENGINE, "cuda", {"options": {"weights": "packed"}}
        )
        second = worker._backend_key(
            ENGINE, "cuda", {"options": {"weights": "emb4"}}
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            first,
            worker._backend_key(ENGINE, "cuda", {"options": {"weights": "packed"}}),
        )
        # The decoder is a second reason to load again.
        self.assertNotEqual(
            worker._backend_key(ENGINE, "cuda", {"options": {"decoder": "student_w4"}}),
            worker._backend_key(ENGINE, "cuda", {"options": {}}),
        )
        # A different *device* is always its own backend.
        self.assertNotEqual(
            worker._backend_key(ENGINE, "cpu", {}),
            worker._backend_key(ENGINE, "cuda", {}),
        )


class _FakeModel:
    """Stands in for ``kittenml.KittenTTS(...)``'s model object."""

    sample_rate = 24000

    def __init__(self, seen):
        self.seen = seen

    def generate(self, text, **kwargs):
        self.seen.append(("generate", text, kwargs))
        return np.zeros(2400, dtype=np.float32)


def _fake_kittenml(seen):
    """A ``kittenml`` package whose ``KittenTTS`` records how it was loaded."""

    @contextlib.contextmanager
    def fake():
        module = types.ModuleType("kittenml")

        class KittenTTS:
            def __init__(self, model_name, **kwargs):
                seen.append(("load", model_name, kwargs))
                self.model = _FakeModel(seen)

            def generate(self, text, **kwargs):
                return self.model.generate(text, **kwargs)

        module.KittenTTS = KittenTTS
        with mock.patch.dict(sys.modules, {"kittenml": module}):
            yield module

    return fake()


class BackendTest(unittest.TestCase):
    """What actually reaches the engine for each kind of voice."""

    def _backend(self, worker, request=None, device="cpu"):
        return worker.KittenTts2Backend(device, request or {})

    def test_available_voices_answer_without_loading_the_model(self):
        worker = _load_worker()
        backend = self._backend(worker)
        # No ``kittenml`` is importable here: answering must not need one.
        with mock.patch.dict(sys.modules, {"kittenml": None}):
            available = backend.available_voices()
        self.assertEqual(len(available), 47)
        self.assertTrue(all(available.values()))

    def test_a_built_in_voice_is_passed_by_name(self):
        worker = _load_worker()
        seen = []
        request = {"voice": "Bruno", "language": "en", "variant": "x"}
        backend = self._backend(worker, request)
        with _fake_kittenml(seen):
            audio, rate, handled = backend.synthesize("Hello there.", request)
        self.assertEqual(seen[0], ("load", "KittenML/kitten-tts-2",
                                   {"device": "cpu"}))
        _kind, text, kwargs = seen[1]
        self.assertEqual(text, "Hello there.")
        self.assertEqual(kwargs["voice"], "Bruno")
        self.assertNotIn("reference", kwargs)
        self.assertTrue(kwargs["normalize"], "English keeps the normalizer")
        self.assertEqual(rate, 24000)
        self.assertFalse(handled)
        self.assertEqual(audio.shape, (2400,))

    def test_a_language_voice_turns_the_english_normalizer_off(self):
        """Its normalizer mangles numbers and dates in every other language."""
        worker = _load_worker()
        seen = []
        request = {"voice": "German", "language": "de"}
        backend = self._backend(worker, request)
        with _fake_kittenml(seen):
            backend.synthesize("Guten Morgen.", request)
        self.assertFalse(seen[1][2]["normalize"])
        # ... unless the user asked for it in Engine tuning.
        seen.clear()
        asked = dict(request, options={"normalize": True})
        backend = self._backend(worker, asked)
        with _fake_kittenml(seen):
            backend.synthesize("Guten Morgen.", asked)
        self.assertTrue(seen[1][2]["normalize"])

    def test_the_expression_markup_reaches_the_engine_verbatim(self):
        """Director's notes go over as typed - the engine parses them itself."""
        worker = _load_worker()
        seen = []
        text = ("[joyful] We won the grant <laugh> I can (((hardly))) "
                "believe it, really!")
        request = {"voice": "Bella", "language": "en"}
        backend = self._backend(worker, request)
        with _fake_kittenml(seen):
            backend.synthesize(text, request)
        self.assertEqual(seen[1][1], text)

    def test_no_punctuation_mode_can_mangle_it_on_the_way_there(self):
        """The whole path: the mode runs, then the backend hands it over."""
        from ai_voice_studio.constants import PUNCTUATION_ALL, PUNCTUATION_NONE
        from ai_voice_studio.tts.engine import process_punctuation

        worker = _load_worker()
        text = "[sad] I am <sigh> (((very))) tired, truly!"
        request = {"voice": "Bella", "language": "en"}
        for mode in (PUNCTUATION_NONE, PUNCTUATION_ALL):
            with self.subTest(mode=mode):
                seen = []
                backend = self._backend(worker, request)
                with _fake_kittenml(seen):
                    backend.synthesize(
                        process_punctuation(text, mode, ENGINE), request
                    )
                spoken = seen[1][1]
                for span in ("[sad]", "<sigh>", "(((very)))"):
                    self.assertIn(span, spoken)

    def test_a_clone_passes_the_reference_and_its_transcript(self):
        worker = _load_worker()
        seen = []
        audio_file = os.path.abspath(__file__)
        request = {
            "voice": voice_lab.CLONE_VOICE_ID,
            "language": "en",
            "ref_audio": audio_file,
            "ref_text": "This is the transcript.",
        }
        backend = self._backend(worker, request)
        with _fake_kittenml(seen):
            backend.synthesize("Hello.", request)
        kwargs = seen[1][2]
        self.assertEqual(kwargs["reference"], audio_file)
        self.assertEqual(kwargs["reference_text"], "This is the transcript.")
        self.assertNotIn("voice", kwargs)

    def test_a_missing_reference_recording_is_reported_in_plain_words(self):
        worker = _load_worker()
        seen = []
        request = {"ref_audio": os.path.join(os.path.dirname(__file__),
                                             "no-such-clip.wav")}
        backend = self._backend(worker, request)
        with _fake_kittenml(seen), self.assertRaises(FileNotFoundError) as caught:
            backend.synthesize("Hello.", request)
        self.assertIn("reference recording", str(caught.exception))

    def test_the_weight_packing_and_decoder_choose_what_is_loaded(self):
        worker = _load_worker()
        seen = []
        request = {"voice": "Luna", "options": {
            "weights": "emb4", "decoder": "student_w4",
            "preset": "expressive",
        }}
        backend = self._backend(worker, request)
        with _fake_kittenml(seen):
            backend.synthesize("Hello.", request)
        _kind, _model_name, load_kwargs = seen[0]
        self.assertEqual(load_kwargs["weights"], "emb4")
        self.assertEqual(load_kwargs["decoder"], "student_w4")
        # The decoding preset is an argument of ``generate``, not of the load.
        self.assertEqual(seen[1][2]["preset"], "expressive")
        self.assertIn("preset", backend.report.applied)
        self.assertIn("normalize", backend.report.applied)
        # The load-time options are handled here, so nothing reports them as
        # an argument the engine could not take.
        self.assertNotIn("weights", backend.report.skipped)
        self.assertNotIn("decoder", backend.report.skipped)

    def test_the_default_packing_is_not_asked_for_by_name(self):
        """``packed`` and ``default`` are the package's own defaults."""
        worker = _load_worker()
        seen = []
        request = {"voice": "Bella", "options": {}}
        backend = self._backend(worker, request)
        with _fake_kittenml(seen):
            backend.synthesize("Hello.", request)
        self.assertEqual(seen[0][2], {"device": "cpu"})

    def test_a_gpu_run_loads_on_the_card(self):
        worker = _load_worker()
        seen = []
        backend = self._backend(worker, {"voice": "Bella"}, device="cuda")
        with mock.patch.object(worker, "_cuda_or_error") as guard, \
                _fake_kittenml(seen):
            backend.synthesize("Hello.", {"voice": "Bella"})
        guard.assert_called_once()
        self.assertEqual(seen[0][2], {"device": "cuda"})

    def test_a_cpu_run_never_insists_on_a_card(self):
        worker = _load_worker()
        seen = []
        backend = self._backend(worker, {"voice": "Bella"}, device="cpu")
        with mock.patch.object(worker, "_cuda_or_error") as guard, \
                _fake_kittenml(seen):
            backend.synthesize("Hello.", {"voice": "Bella"})
        guard.assert_not_called()

    def test_a_missing_package_names_the_engine_and_the_panel(self):
        worker = _load_worker()
        backend = self._backend(worker, {"voice": "Bella"})
        with mock.patch.dict(sys.modules, {"kittenml": None}):
            with self.assertRaises(RuntimeError) as caught:
                backend.synthesize("Hello.", {"voice": "Bella"})
        message = str(caught.exception)
        self.assertIn("kittenml", message)
        self.assertIn("Kitten TTS 2.0", message)
        self.assertIn("Settings >", message)


if __name__ == "__main__":
    unittest.main()
