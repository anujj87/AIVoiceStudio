"""Audio description unit tests: catalog, planning, output and the job.

These are pure-Python (no wx and no model): the part of the feature that decides
*how* a video is cut up, *how long* the run will take and *where* a description
may be spoken is deliberately free of both, so it can be tested exactly.  The
GUI side is covered by the smoke tests, which build the wizard page, the
describe dialog and the settings category.

The model inference itself cannot run here (it needs a downloaded checkpoint in
a per-model virtualenv and, for the good tiers, a GPU), so the job tests drive
``engine.describe_video`` through a stub and check everything around it.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
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

from ai_voice_studio import project
from ai_voice_studio.audio.ffmpeg import find_ffmpeg
from ai_voice_studio.constants import (
    AD_CHUNK_SECONDS,
    AD_DEFAULT_FPS,
    AD_FORMAT_SCRIPT,
    AD_FORMAT_SRT,
    PROJECT_TYPE_AD,
)
from ai_voice_studio.describe import (
    catalog,
    engine,
    environments,
    extended,
    job,
    media,
    narration,
    output,
    pipeline,
    weights,
    worker,
)

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
#: The models the worker drives through their own ``chat`` method, and the ones
#: whose checkpoint registers ``AutoModel`` rather than a task-specific class.
_MINICPM_CHAT_MODELS = ("minicpm_o", "minicpm_o_4_5", "minicpm_v_4")

#: The models driven through their own single-image question method, because
#: they have neither a processor nor a chat template for the generic path.
_QUERY_MODELS = ("moondream2",)


class CatalogTest(unittest.TestCase):
    def test_ids_are_unique_and_every_model_is_named(self):
        ids = catalog.model_ids()
        self.assertEqual(len(ids), len(set(ids)))
        for model_id in ids:
            entry = catalog.model(model_id)
            self.assertTrue(entry["name"], model_id)
            self.assertTrue(entry["license"], model_id)
            self.assertTrue(entry["notes"], model_id)
            self.assertTrue(entry["languages"], model_id)

    def test_every_variant_declares_its_size_and_video_memory(self):
        # The whole point of the settings table: a user picks by hardware.
        for model_id in catalog.model_ids():
            for entry in catalog.variant_entries(model_id):
                with self.subTest(model=model_id, variant=entry["value"]):
                    self.assertGreater(entry["size_gb"], 0)
                    self.assertGreater(entry["vram_gb"], 0)
                    self.assertIn(entry["fit"], ("cpu", "gpu"))
                    self.assertIn("value", entry)
                    self.assertTrue(entry["label"])

    def test_every_variant_has_a_downloadable_repository(self):
        # A variant with no repository cannot be described (or downloaded), so
        # the catalog must not offer one.
        for model_id in catalog.model_ids():
            for value, _label in catalog.variants(model_id):
                self.assertTrue(catalog.hf_model(model_id, value),
                                f"{model_id}/{value}")

    def test_default_variants_exist(self):
        for model_id in catalog.model_ids():
            self.assertIn(catalog.default_variant(model_id),
                          [v for v, _ in catalog.variants(model_id)])

    def test_variant_rows_line_up_with_the_variants(self):
        for model_id in catalog.model_ids():
            rows = catalog.variant_rows(model_id)
            self.assertEqual(len(rows), len(catalog.variants(model_id)))
            self.assertEqual([r[0] for r in rows],
                             [v for v, _ in catalog.variants(model_id)])
            for _value, _label, size, vram, fit_label, _note in rows:
                self.assertTrue(size.endswith("GB"))
                self.assertTrue(vram.endswith("GB"))
                self.assertIn(fit_label, ("CPU or GPU", "GPU"))

    def test_hallucination_is_stated_for_every_model(self):
        # A blind listener cannot check a description against the picture, so
        # every model says plainly how much it tends to invent.
        for model_id in catalog.model_ids():
            self.assertTrue(catalog.hallucination_note(model_id), model_id)

    def test_no_model_needs_a_login(self):
        for model_id in catalog.model_ids():
            self.assertTrue(catalog.backend_modules(model_id), model_id)
            self.assertEqual(catalog.model(model_id)["backend"], "transformers")

    def test_a_gpu_only_variant_is_offered_the_gpu_alone(self):
        for model_id in catalog.model_ids():
            for value, _label in catalog.variants(model_id):
                if not catalog.gpu_only(model_id, value):
                    continue
                self.assertEqual(
                    [v for v, _ in catalog.device_options(model_id, value, True)],
                    ["cuda"],
                )

    def test_a_cpu_capable_variant_offers_both_devices(self):
        for model_id in catalog.model_ids():
            for value, _label in catalog.variants(model_id):
                if catalog.gpu_only(model_id, value):
                    continue
                with_cuda = [v for v, _ in catalog.device_options(
                    model_id, value, True)]
                self.assertIn("cpu", with_cuda, f"{model_id}/{value}")
                without_cuda = [v for v, _ in catalog.device_options(
                    model_id, value, False)]
                self.assertEqual(without_cuda, ["cpu"], f"{model_id}/{value}")

    def test_the_preselect_is_cpu_capable_and_small(self):
        # The wizard and the dialog must not open on a 37 GB GPU-only model.
        model_id = catalog.default_model_id()
        self.assertIn(model_id, catalog.model_ids())
        variant = catalog.default_variant(model_id)
        self.assertTrue(catalog.cpu_ok(model_id, variant))
        self.assertLessEqual(catalog.variant_vram_gb(model_id, variant), 12.0)

    def test_gemma3_is_offered_with_its_licence_stated_plainly(self):
        """Gemma 3 was added on request: its licence is named, not glossed over.

        It is the one catalog entry whose licence is not OSI-approved, and its
        download is gated, so both facts have to be readable *before* the user
        picks it rather than only in a licence file.
        """
        entry = catalog.model("gemma3")
        self.assertEqual(entry["name"], "Gemma 3")
        self.assertIn("not OSI-approved", entry["license"])
        self.assertIn("Gemma Terms of Use", entry["license"])
        self.assertIn("gated", entry["notes"])
        self.assertIn("terms", entry["notes"].lower())
        self.assertIn("vision", entry["notes"].lower())
        self.assertTrue(catalog.hallucination_note("gemma3"))

    def test_gemma3_offers_only_the_checkpoints_that_can_see(self):
        # Gemma 3 1B is text-only, so it cannot describe a video and must not be
        # in a video-description catalog at all.
        variants = [value for value, _label in catalog.variants("gemma3")]
        self.assertEqual(variants, ["4b", "12b", "27b"])
        self.assertNotIn("1b", variants)
        self.assertIn("1B is text-only", catalog.model("gemma3")["notes"])

    def test_gemma3_has_a_cpu_capable_variant_and_no_fake_grounding(self):
        # The app promises a first run with no NVIDIA card, and Gemma 3 does not
        # ground to a second, so it must not claim to.
        self.assertTrue(catalog.cpu_ok("gemma3", "4b"))
        self.assertGreater(catalog.variant_vram_gb("gemma3", "4b"), 0)
        self.assertFalse(catalog.supports_timestamps("gemma3"))
        self.assertEqual(catalog.default_variant("gemma3"), "4b")
        self.assertIn("cpu", [v for v, _l in catalog.device_options(
            "gemma3", "4b", True)])

    def test_cost_falls_when_the_gpu_is_used(self):
        for model_id in catalog.model_ids():
            variant = catalog.default_variant(model_id)
            cpu = catalog.cost_per_video_minute(model_id, variant, "cpu")
            gpu = catalog.cost_per_video_minute(model_id, variant, "cuda")
            self.assertGreater(cpu, 0)
            self.assertGreater(gpu, 0)
            self.assertLess(gpu, cpu, model_id)

    def test_minicpm_o_installs_the_packages_its_own_code_imports(self):
        """MiniCPM-o 2.6 runs its *own* modelling code, and that code imports
        more than a stock describing model needs.

        The failure without them is a live one, seen on a real install:
        "Could not load the model: This modeling file requires the following
        packages that were not found in your environment: torchaudio.  Run
        `pip install torchaudio`".  ``transformers`` refuses to load
        ``processing_minicpmo.py`` without ``torchaudio``, and
        ``modeling_minicpmo.py`` pulls ``utils.py``, which imports ``librosa``.
        """
        packages = catalog.pip_packages("minicpm_o")
        self.assertIn("torchaudio", packages)
        self.assertIn("librosa", packages)
        # ``soxr`` is deliberately *not* declared: transformers imports it
        # whenever librosa is importable, and librosa requires it, so declaring
        # the one package carries the other.
        self.assertNotIn("soxr", packages)
        self.assertIn("soundfile", packages)

    def test_the_minicpm_models_keep_transformers_in_the_4_52_series(self):
        # transformers 4.53.0 removed WHISPER_ATTENTION_CLASSES, which
        # MiniCPM-o's own modelling code imports (its config.json records
        # transformers_version 4.44.2), so a newer environment cannot import this
        # model at all - verified against 5.19.0 and 4.57.6, which both fail on
        # that one name.  MiniCPM-V 4 shares the pin for its own reason: its
        # config.json records 4.51.0 and its files import the deprecated
        # transformers.modeling_attn_mask_utils.  The pin is scoped to these
        # models' own environments.
        package = "transformers<4.53"
        pinned = {"minicpm_o", "minicpm_o_4_5", "minicpm_v_4"}
        for model_id in sorted(pinned):
            self.assertIn(package, catalog.pip_packages(model_id))
        for model_id in catalog.model_ids():
            if model_id in pinned:
                continue
            with self.subTest(model=model_id):
                self.assertNotIn(package, catalog.pip_packages(model_id))

    def test_a_checkpoint_that_names_its_own_class_says_so(self):
        """A checkpoint's ``auto_map`` decides which class loads it.

        MiniCPM-o 2.6 and MiniCPM-V 4 register ``AutoModel`` and register
        *nothing* for the task-specific classes, so a run that asks for the
        task-specific one is refused before a weight is read: "Unrecognized
        configuration class <class '...MiniCPMOConfig'> for this kind of
        AutoModel: AutoModelForImageTextToText".  That error was reported from a
        correct environment, which is why the catalogue - not the worker - has to
        answer with the class the checkpoint registered.
        """
        for model_id in _MINICPM_CHAT_MODELS:
            with self.subTest(model=model_id):
                self.assertEqual(catalog.auto_class(model_id), "AutoModel")
                self.assertTrue(catalog.needs_remote_code(model_id))
        # Every model transformers supports natively keeps the task-specific
        # class the worker already asks for.
        for model_id in catalog.model_ids():
            if catalog.needs_remote_code(model_id):
                continue
            with self.subTest(model=model_id):
                self.assertEqual(catalog.auto_class(model_id), "")

    def test_the_minicpm_family_is_driven_through_its_own_chat(self):
        # Its processors ship no chat template that carries the standard
        # multimodal message and its own ``generate`` wants inputs_embeds, so
        # ``model.chat`` is the only documented way in.
        for model_id in _MINICPM_CHAT_MODELS:
            with self.subTest(model=model_id):
                self.assertEqual(catalog.driver(model_id), catalog.DRIVER_CHAT)
        for model_id in catalog.model_ids():
            if model_id in _MINICPM_CHAT_MODELS or model_id in _QUERY_MODELS:
                continue
            with self.subTest(model=model_id):
                self.assertEqual(catalog.driver(model_id),
                                 catalog.DRIVER_GENERATE)

    def test_moondream_is_driven_through_its_own_skills(self):
        """Moondream 2 cannot be reached through a multimodal message at all.

        Its repository carries a tokenizer and its own modelling code - no
        processor, no chat template - so there is no template to apply and its
        ``generate`` wants ``inputs_embeds`` rather than the ``input_ids`` a
        processor produces.  The catalogue therefore has to answer with the
        class its ``auto_map`` registers, with its own ``query`` skill, and with
        the dtype its forward pass is written in.
        """
        entry = catalog.model("moondream2")
        self.assertEqual(catalog.auto_class("moondream2"),
                         "AutoModelForCausalLM")
        self.assertEqual(catalog.driver("moondream2"), catalog.DRIVER_QUERY)
        self.assertEqual(catalog.dtype("moondream2"), "auto")
        self.assertTrue(catalog.needs_remote_code("moondream2"))
        self.assertIn("one picture at a time", entry["notes"])
        self.assertIn("query", entry["variants"][0]["note"])

    def test_only_a_checkpoint_whose_code_fixes_it_names_a_dtype(self):
        # Everything else is loaded in the run's own choice - float32 on a CPU,
        # bfloat16 on a card - which is what the catalogue's sizes are quoted in.
        for model_id in catalog.model_ids():
            if model_id in _QUERY_MODELS:
                continue
            with self.subTest(model=model_id):
                self.assertEqual(catalog.dtype(model_id), "")

    def test_minicpm_o_is_loaded_vision_only(self):
        """An omni model asked for a description should not build its speech.

        Its config builds a whisper encoder and a TTS module beside the vision
        stack, and its own code asserts the TTS packages when they are wanted.
        Audio description reads the picture, so both halves are switched off at
        load - the model card's own advice for vision-only use - and the audio
        and speech packages stay out of what a description needs.
        """
        for model_id in ("minicpm_o", "minicpm_o_4_5"):
            with self.subTest(model=model_id):
                self.assertEqual(catalog.load_kwargs(model_id),
                                 {"init_audio": False, "init_tts": False})
        # MiniCPM-V 4 has no audio or speech half at all, so it asks for nothing.
        self.assertEqual(catalog.load_kwargs("minicpm_v_4"), {})

    def test_minicpm_o_4_5_is_the_family_s_current_flagship(self):
        """Added on request: the 9B generation above MiniCPM-o 2.6.

        It is an entry of its own rather than a second variant of 2.6, because
        its environment carries a package 2.6 has never needed - so the two must
        not share the one MiniCPM-o virtualenv.
        """
        entry = catalog.model("minicpm_o_4_5")
        self.assertEqual(entry["name"], "MiniCPM-o 4.5")
        self.assertEqual(entry["license"], "Apache 2.0")
        self.assertEqual(entry["languages_label"], "30+")
        self.assertEqual(catalog.hf_model("minicpm_o_4_5", "9b"),
                         "openbmb/MiniCPM-o-4_5")
        self.assertEqual(catalog.default_variant("minicpm_o_4_5"), "9b")
        # 9.4B parameters in bf16 are about 18.7 GB of weights, which is what the
        # download is; the video memory has to cover the weights themselves.
        self.assertAlmostEqual(catalog.variant_size_gb("minicpm_o_4_5", "9b"),
                               18.7, places=1)
        self.assertGreater(catalog.variant_vram_gb("minicpm_o_4_5", "9b"),
                           catalog.variant_size_gb("minicpm_o_4_5", "9b"))
        self.assertTrue(catalog.gpu_only("minicpm_o_4_5", "9b"))
        self.assertIn("4-bit", catalog.model("minicpm_o_4_5")["fit_label"])
        # Its own environment, and the packages its card installs for inference
        # without TTS included.
        self.assertNotEqual(engine.env_dir("minicpm_o_4_5"),
                            engine.env_dir("minicpm_o"))
        packages = catalog.pip_packages("minicpm_o_4_5")
        self.assertIn("torchaudio", packages)
        self.assertIn("minicpmo-utils>=1.0.5", packages)
        self.assertIn("transformers<4.53", packages)
        # ``minicpmo-utils`` is not a convenience: the checkpoint's own
        # ``utils.py`` imports ``minicpmo.utils`` outside any guard, and
        # transformers' remote-code check reads imports inside functions, so the
        # load is refused before a weight is read without it.  The entry says so
        # rather than leaving the package to look optional.
        self.assertIn("minicpmo.utils", entry["notes"])
        # ``librosa`` is in that same import list; the pin is the other half,
        # because ``minicpmo-utils`` pins ``librosa==0.9.0`` and 0.9.0 imports
        # ``pkg_resources``, which setuptools 81 removed.  Installed-but-
        # unimportable is the one failure the panel cannot see: the checkpoint
        # reports librosa as missing and Repair finds nothing to install.
        self.assertIn("librosa", packages)
        self.assertIn("setuptools<81", packages)

    def test_a_model_is_not_offered_a_cpu_run_it_cannot_survive(self):
        # 18.7 GB of weights widened to float32 on a CPU is not a run, and the
        # device list must not offer it either.
        self.assertEqual(
            [value for value, _label in catalog.device_options(
                "minicpm_o_4_5", "9b", True)],
            ["cuda"],
        )

    def test_the_4b_minicpm_answers_the_question_plainly(self):
        """The 4B MiniCPM is MiniCPM-V 4, and the entry says so.

        It was added on request ("MiniCPM-o also has a 4B model with video"):
        the -o line is 8B and up, and this is the 4B checkpoint of the same
        family, which does read video and does run on Windows.
        """
        entry = catalog.model("minicpm_v_4")
        self.assertEqual(entry["name"], "MiniCPM-V 4")
        self.assertIn("no 4B MiniCPM-o", entry["notes"])
        self.assertIn("video", entry["notes"].lower())
        self.assertEqual(entry["license"], "Apache 2.0")
        self.assertEqual(catalog.default_variant("minicpm_v_4"), "4.1b")
        self.assertEqual(catalog.hf_model("minicpm_v_4", "4.1b"),
                         "openbmb/MiniCPM-V-4")
        # It fits a 12 GB card and, unlike MiniCPM-o, a CPU as well.
        self.assertTrue(catalog.cpu_ok("minicpm_v_4", "4.1b"))
        self.assertLessEqual(catalog.variant_vram_gb("minicpm_v_4", "4.1b"), 12.0)
        self.assertNotIn("torchaudio", catalog.pip_packages("minicpm_v_4"))
        self.assertNotIn("librosa", catalog.pip_packages("minicpm_v_4"))

    def test_a_pin_does_not_leak_into_the_model_s_other_packages(self):
        # ``environments`` plans installs from the catalogue and uninstalls from
        # its plain names, so a pinned requirement must strip to its name.
        self.assertIn("transformers", environments.packages_for("minicpm_o"))
        self.assertNotIn("transformers<4.53",
                         environments.packages_for("minicpm_o"))


# ---------------------------------------------------------------------------
# One environment per describing model
# ---------------------------------------------------------------------------
class DescribeEnvironmentPlanTest(unittest.TestCase):
    """The pip plan an *Install/Repair* press turns into commands.

    A describing model's environment is built from its catalogue entry alone, so
    a package that is missing from the plan is missing from the environment - and
    the model then fails to load in the middle of a run rather than at install
    time.  These tests pin the two properties that matter: PyTorch's family comes
    from one wheel index together, and the model's own extra packages survive the
    split.
    """

    def test_torchaudio_comes_from_the_same_index_as_torch(self):
        # A CUDA torch beside a PyPI (CPU-only on Windows) torchaudio is a
        # runtime crash, not a warning, so the three PyTorch packages must be
        # installed in one step from the CUDA index.
        with mock.patch.object(environments, "_has_cuda", return_value=True):
            plan = environments.install_plan("minicpm_o")
        self.assertEqual(len(plan), 2)
        cuda_packages, index_url = plan[0]
        self.assertEqual(sorted(cuda_packages),
                         ["torch", "torchaudio", "torchvision"])
        self.assertIn("pytorch.org", index_url or "")
        rest, rest_index = plan[1]
        self.assertIsNone(rest_index)
        for package in ("torch", "torchaudio", "torchvision"):
            self.assertNotIn(package, rest)
        self.assertIn("transformers<4.53", rest)
        self.assertIn("librosa", rest)
        self.assertIn("bitsandbytes", rest)

    def test_a_cpu_only_machine_installs_everything_in_one_step(self):
        with mock.patch.object(environments, "_has_cuda", return_value=False):
            plan = environments.install_plan("minicpm_o")
        self.assertEqual(len(plan), 1)
        packages, index_url = plan[0]
        self.assertIsNone(index_url)
        for package in ("transformers<4.53", "torch", "torchaudio", "librosa",
                        "soundfile"):
            self.assertIn(package, packages)
        # No card, so the quantiser is not part of the install at all.
        self.assertNotIn("bitsandbytes", packages)

    def test_missing_packages_names_what_the_catalogue_declares(self):
        # ``is_installed`` only proves the *back-end* imports.  This is the
        # question the Repair button asks instead, and MiniCPM-o 2.6 is why it
        # exists: its environment reported itself installed (transformers was
        # there) while torchaudio and librosa - the packages its own files
        # import - were absent, and the only way back was to delete the whole
        # environment and build it again.
        complete = {"transformers": "5.19.0", "torch": "2.11.0",
                    "torchvision": "0.26.0", "accelerate": "1.0",
                    "pillow": "12.3.0", "numpy": "2.5.2",
                    "soundfile": "0.13.1"}
        with mock.patch.object(environments, "is_created", return_value=True), \
                mock.patch.object(environments, "_has_cuda",
                                  return_value=False), \
                mock.patch.object(environments, "_distributions",
                                  return_value=dict(complete)):
            self.assertEqual(environments.missing_packages("minicpm_o"),
                             ["torchaudio", "librosa"])

    def test_a_complete_environment_is_missing_nothing(self):
        complete = {"transformers": "4.52.4", "torch": "2.11.0",
                    "torchvision": "0.26.0", "torchaudio": "2.11.0",
                    "accelerate": "1.0", "pillow": "12.3.0",
                    "numpy": "2.5.2", "librosa": "1.0.0",
                    "soundfile": "0.13.1", "bitsandbytes": "0.50.2"}
        with mock.patch.object(environments, "is_created", return_value=True), \
                mock.patch.object(environments, "_has_cuda",
                                  return_value=True), \
                mock.patch.object(environments, "_distributions",
                                  return_value=dict(complete)):
            self.assertEqual(environments.missing_packages("minicpm_o"), [])

    def test_the_cuda_only_quantiser_counts_as_missing_on_a_gpu_machine(self):
        without_quantiser = {"transformers": "4.52.4", "torch": "2.11.0",
                             "torchvision": "0.26.0", "torchaudio": "2.11.0",
                             "accelerate": "1.0", "pillow": "12.3.0",
                             "numpy": "2.5.2", "librosa": "1.0.0",
                             "soundfile": "0.13.1"}
        with mock.patch.object(environments, "is_created", return_value=True), \
                mock.patch.object(environments, "_has_cuda",
                                  return_value=True), \
                mock.patch.object(environments, "_distributions",
                                  return_value=dict(without_quantiser)):
            self.assertEqual(environments.missing_packages("minicpm_o"),
                             ["bitsandbytes"])

    def test_an_environment_that_cannot_be_listed_reports_nothing(self):
        # A directory that cannot be read is not evidence that every package is
        # absent: the panel would then claim a complete environment needs repair.
        with mock.patch.object(environments, "is_created", return_value=True), \
                mock.patch.object(environments, "_distributions",
                                  return_value={}):
            self.assertEqual(environments.missing_packages("minicpm_o"), [])
        with mock.patch.object(environments, "is_created", return_value=False):
            self.assertEqual(environments.missing_packages("minicpm_o"), [])

    def test_the_status_line_names_a_package_that_is_missing(self):
        # The describe dialog and the settings table both read this line, so the
        # sentence the user sees for a half-built environment is tested here.
        with mock.patch.object(environments, "is_installed", return_value=True), \
                mock.patch.object(environments, "missing_packages",
                                  return_value=["torchaudio"]):
            note = engine.availability_note("minicpm_o")
        self.assertIn("torchaudio", note)
        self.assertIn("Repair", note)
        with mock.patch.object(environments, "is_installed", return_value=True), \
                mock.patch.object(environments, "missing_packages",
                                  return_value=[]), \
                mock.patch.object(environments, "installed_version",
                                  return_value="4.52.4"):
            self.assertIn("Ready", engine.availability_note("minicpm_o"))

    def test_every_model_installs_a_backend_it_can_be_probed_by(self):
        # ``environments.is_installed`` proves an environment by importing, so
        # each plan has to carry the package behind every backend module.
        for model_id in catalog.model_ids():
            with mock.patch.object(environments, "_has_cuda", return_value=False):
                plan = environments.install_plan(model_id)
            installed = [package for packages, _index in plan
                         for package in packages]
            for module in catalog.backend_modules(model_id):
                self.assertIn(module, " ".join(installed), model_id)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
class ChunkingTest(unittest.TestCase):
    def test_a_video_shorter_than_one_chunk_is_one_chunk(self):
        chunks = pipeline.plan_chunks(100, 120)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 100.0)
        self.assertEqual(chunks[0]["duration"], 100.0)

    def test_chunks_tile_the_video_exactly(self):
        chunks = pipeline.plan_chunks(250, 120)
        self.assertEqual([c["index"] for c in chunks], [0, 1, 2])
        self.assertEqual(chunks[0]["start"], 0.0)
        # Half-open windows: one chunk's end is the next one's start, so no
        # second of video is described twice and none is skipped.
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous["end"], following["start"])
        self.assertEqual(chunks[-1]["end"], 250.0)
        self.assertEqual(chunks[-1]["duration"], 10.0)
        self.assertEqual(sum(c["duration"] for c in chunks), 250.0)

    def test_the_remainder_becomes_a_shorter_last_chunk(self):
        chunks = pipeline.plan_chunks(3600, 120)
        self.assertEqual(len(chunks), 30)
        self.assertEqual(chunks[-1]["duration"], 120.0)
        chunks = pipeline.plan_chunks(3601, 120)
        self.assertEqual(len(chunks), 31)
        self.assertEqual(chunks[-1]["duration"], 1.0)

    def test_an_exact_multiple_does_not_add_an_empty_chunk(self):
        chunks = pipeline.plan_chunks(240, 120)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1]["end"], 240.0)

    def test_no_duration_means_no_chunks(self):
        for value in (0, None, -5, float("nan")):
            self.assertEqual(pipeline.plan_chunks(value, 120), [])
            self.assertEqual(pipeline.chunk_count(value, 120), 0)

    def test_chunk_count_matches_the_plan(self):
        for duration in (1, 59, 120, 121, 600, 3600, 3601):
            self.assertEqual(
                pipeline.chunk_count(duration, 120),
                len(pipeline.plan_chunks(duration, 120)),
                duration,
            )

    def test_a_silly_chunk_length_is_clamped(self):
        # A zero or negative chunk length would otherwise loop forever.
        self.assertEqual(len(pipeline.plan_chunks(10, 0)), 1)
        self.assertEqual(len(pipeline.plan_chunks(10, -5)), 1)

    def test_chunk_at_finds_the_window(self):
        chunks = pipeline.plan_chunks(250, 120)
        self.assertEqual(pipeline.chunk_at(chunks, 0)["index"], 0)
        self.assertEqual(pipeline.chunk_at(chunks, 119.9)["index"], 0)
        self.assertEqual(pipeline.chunk_at(chunks, 120)["index"], 1)
        self.assertEqual(pipeline.chunk_at(chunks, 249.9)["index"], 2)
        self.assertIsNone(pipeline.chunk_at([], 5))

    def test_sample_times_stay_inside_the_window(self):
        times = pipeline.sample_times(10, 18, 0.5)
        self.assertEqual(times, [10.0, 12.0, 14.0, 16.0])
        for moment in times:
            self.assertGreaterEqual(moment, 10.0)
            self.assertLess(moment, 18.0)

    def test_sample_times_are_capped(self):
        times = pipeline.sample_times(0, 3600, 1.0, cap=50)
        self.assertEqual(len(times), 50)

    def test_one_frame_when_no_rate_is_given(self):
        self.assertEqual(pipeline.sample_times(5, 10, 0), [5.0])
        self.assertEqual(pipeline.sample_times(5, 10, None), [5.0])

    def test_an_empty_window_samples_nothing(self):
        self.assertEqual(pipeline.sample_times(10, 10, 1.0), [])
        self.assertEqual(pipeline.sample_times(10, 5, 1.0), [])

    def test_total_frames_follows_the_frame_rate(self):
        fast = pipeline.total_frames(600, 1.0)
        slow = pipeline.total_frames(600, 0.25)
        self.assertGreater(fast, slow)
        self.assertGreater(slow, 0)


# ---------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------
class EstimateTest(unittest.TestCase):
    def _estimate(self, duration, **kwargs):
        options = {"model_id": "qwen2_5_vl", "variant": "7b", "fps": 0.5,
                       "device": "cuda", "chunk_seconds": 120}
        options.update(kwargs)
        return pipeline.estimate(duration, **options)

    def test_nothing_to_do_costs_nothing(self):
        info = self._estimate(0)
        self.assertEqual(info["seconds"], 0.0)
        self.assertEqual(info["chunks"], 0)
        self.assertEqual(info["frames"], 0)

    def test_a_longer_video_is_estimated_to_take_longer(self):
        short = self._estimate(60)["seconds"]
        long = self._estimate(3600)["seconds"]
        self.assertGreater(long, short)

    def test_the_frames_per_second_scale_the_cost(self):
        half = self._estimate(600, fps=0.25)["seconds"]
        full = self._estimate(600, fps=0.5)["seconds"]
        self.assertLess(half, full)
        self.assertAlmostEqual(full - half, self._estimate(
            600)["per_minute"] * 0.5 * 10, delta=1.0)

    def test_the_cpu_is_estimated_to_be_slower_than_the_gpu(self):
        self.assertGreater(self._estimate(600, device="cpu")["seconds"],
                           self._estimate(600, device="cuda")["seconds"])

    def test_the_chunk_plan_and_the_estimate_agree(self):
        info = self._estimate(3600)
        self.assertEqual(info["chunks"], 30)
        self.assertEqual(info["chunks"], len(pipeline.plan_chunks(3600, 120)))

    def test_speech_timings_add_to_the_estimate(self):
        plain = self._estimate(600)["seconds"]
        timed = self._estimate(600, speech_seconds=90)["seconds"]
        self.assertAlmostEqual(timed - plain, 90.0, delta=0.01)

    def test_format_duration_reads_like_a_person_wrote_it(self):
        self.assertEqual(pipeline.format_duration(45), "45 s")
        self.assertEqual(pipeline.format_duration(60), "1 min")
        self.assertEqual(pipeline.format_duration(3599), "59 min")
        self.assertEqual(pipeline.format_duration(3600), "1 h")
        self.assertEqual(pipeline.format_duration(6000), "1 h 40 min")
        self.assertEqual(pipeline.format_duration(None), "0 s")

    def test_the_warning_threshold(self):
        self.assertFalse(pipeline.needs_warning(299))
        self.assertTrue(pipeline.needs_warning(301))
        self.assertFalse(pipeline.needs_warning(0))

    def test_the_warning_says_how_long_and_how_far(self):
        info = self._estimate(3600)
        message = pipeline.warning_message(3600, info, chunked=True)
        self.assertIn("1 h", message)
        self.assertIn("chunk", message)
        untimed = pipeline.warning_message(3600, info, chunked=False)
        self.assertIn("chunks", untimed)


# ---------------------------------------------------------------------------
# Gap fitting
# ---------------------------------------------------------------------------
class GapFittingTest(unittest.TestCase):
    SPEECH = [{"start": 5, "end": 10}, {"start": 30, "end": 40}]

    def test_free_gaps_are_the_silences(self):
        self.assertEqual(pipeline.free_gaps(self.SPEECH, 60),
                         [(0.0, 5.0), (10.0, 30.0), (40.0, 60.0)])

    def test_no_speech_is_one_gap(self):
        self.assertEqual(pipeline.free_gaps([], 60), [(0.0, 60.0)])

    def test_overlapping_phrases_are_merged_first(self):
        speech = [{"start": 0, "end": 10}, {"start": 8, "end": 20}]
        self.assertEqual(pipeline.free_gaps(speech, 30), [(20.0, 30.0)])

    def test_short_gaps_are_ignored(self):
        speech = [{"start": 0, "end": 10}, {"start": 11, "end": 20}]
        # The 1 second hole is below the 1.5 second minimum.
        self.assertEqual(pipeline.free_gaps(speech, 20), [])

    def test_a_phrase_without_timing_does_not_carve_a_hole(self):
        speech = [{"start": None, "end": 10}, {"start": 30, "end": 40}]
        gaps = pipeline.free_gaps(speech, 60)
        # The untimed phrase carved nothing out, so the timeline starts empty
        # and only the timed phrase makes a hole.
        self.assertEqual(gaps, [(0.0, 30.0), (40.0, 60.0)])

    def test_gaps_are_clamped_to_the_video(self):
        speech = [{"start": 50, "end": 900}]
        self.assertEqual(pipeline.free_gaps(speech, 60), [(0.0, 50.0)])

    def test_no_duration_means_no_gaps(self):
        self.assertEqual(pipeline.free_gaps(self.SPEECH, 0), [])

    def test_a_description_lands_in_the_pause_after_it(self):
        result = pipeline.fit_into_gaps(
            [{"start": 3, "end": 4, "text": "A man walks in."}],
            self.SPEECH, duration=60,
        )
        self.assertEqual(result["dropped"], 0)
        self.assertEqual(len(result["placed"]), 1)
        placed = result["placed"][0]
        self.assertEqual(placed["text"], "A man walks in.")
        # At the description's own start inside the first pause, so the words
        # sit as close to the action as the silence allows - and nowhere near
        # the speech that begins at 5 seconds.
        self.assertEqual(placed["start"], 3.0)
        self.assertLessEqual(placed["end"], 5.0)

    def test_a_pause_that_is_already_over_is_skipped(self):
        # A description that starts after the first pause ends must move on to
        # the next one rather than be spoken over the dialogue.
        result = pipeline.fit_into_gaps(
            [{"start": 12, "end": 13, "text": "Later action."}],
            self.SPEECH, duration=60,
        )
        placed = result["placed"][0]
        self.assertEqual(placed["start"], 12.0)
        self.assertLessEqual(placed["end"], 30.0)

    def test_a_description_skips_a_pause_too_small_to_hold_it(self):
        # A sentence that needs about four seconds must not be chopped into a
        # two-word fragment just because the first pause happens to be two
        # seconds long: it waits for a pause that fits it.
        # The first pause holds 2 seconds (from the description's start to the
        # speech at 3); the sentence needs about 3.5.
        speech = [{"start": 3, "end": 12}, {"start": 30, "end": 40}]
        text = "A woman opens a heavy wooden door and steps inside."
        result = pipeline.fit_into_gaps(
            [{"start": 1, "end": 2, "text": text}],
            speech, duration=60,
        )
        placed = result["placed"][0]
        self.assertEqual(placed["text"], text)
        self.assertEqual(placed["start"], 12.0)
        self.assertLessEqual(placed["end"], 30.0)

    def test_a_description_with_only_tight_pauses_is_shortened(self):
        # When nothing can hold it in full, the earliest pause with any room is
        # used and the words are trimmed there rather than spoken over speech.
        speech = [{"start": 0, "end": 2}, {"start": 6, "end": 40}]
        text = "A woman opens a heavy wooden door and steps inside a kitchen."
        result = pipeline.fit_into_gaps(
            [{"start": 2, "end": 3, "text": text}],
            speech, duration=60, max_seconds=2.0,
        )
        placed = result["placed"][0]
        self.assertLessEqual(placed["end"], 6.0)
        self.assertLess(len(placed["text"].split()), len(text.split()))

    def test_a_long_description_is_shortened_not_spoken_over_dialogue(self):
        words = " ".join(["word"] * 60)
        result = pipeline.fit_into_gaps(
            [{"start": 12, "end": 20, "text": words}],
            self.SPEECH, duration=60, max_seconds=5,
        )
        placed = result["placed"][0]
        self.assertLessEqual(placed["end"] - placed["start"], 5.0 + 1e-6)
        self.assertLess(len(placed["text"].split()), 60)
        self.assertLessEqual(placed["end"], 30.0)

    def test_a_description_with_no_pause_left_is_dropped(self):
        speech = [{"start": 0, "end": 60}]
        result = pipeline.fit_into_gaps(
            [{"start": 0, "end": 10, "text": "Nothing fits."}],
            speech, duration=60,
        )
        self.assertEqual(result["placed"], [])
        self.assertEqual(result["dropped"], 1)

    def test_empty_text_is_not_a_description(self):
        result = pipeline.fit_into_gaps(
            [{"start": 0, "end": 1, "text": "   "}], [], duration=60,
        )
        self.assertEqual(result["placed"], [])
        self.assertEqual(result["dropped"], 0)

    def test_two_descriptions_never_share_one_pause(self):
        result = pipeline.fit_into_gaps(
            [
                {"start": 0, "end": 1, "text": "First."},
                {"start": 0, "end": 1, "text": "Second."},
            ],
            self.SPEECH, duration=60,
        )
        starts = [item["start"] for item in result["placed"]]
        self.assertEqual(len(starts), len(set(starts)))

    def test_placed_descriptions_come_back_in_time_order(self):
        result = pipeline.fit_into_gaps(
            [
                {"start": 45, "end": 50, "text": "Late."},
                {"start": 1, "end": 2, "text": "Early."},
            ],
            [], duration=60,
        )
        starts = [item["start"] for item in result["placed"]]
        self.assertEqual(starts, sorted(starts))

    def test_shorten_keeps_a_short_text_whole(self):
        self.assertEqual(pipeline.shorten_to_seconds("Two words", 10),
                         "Two words")

    def test_shorten_ends_on_a_sentence_when_one_is_inside_the_budget(self):
        text = "One two three. Four five six seven eight nine ten eleven."
        trimmed = pipeline.shorten_to_seconds(text, 1.6)
        self.assertLessEqual(len(trimmed.split()), 4)
        self.assertLess(len(trimmed.split()), len(text.split()))
        self.assertEqual(trimmed, "One two three.")


# ---------------------------------------------------------------------------
# Media probe helpers
# ---------------------------------------------------------------------------
class MediaTest(unittest.TestCase):
    BANNER = (
        "ffmpeg version 7.1\n"
        "  Duration: 01:02:03.45, start: 0.000000, bitrate: 1200 kb/s\n"
    )

    def test_the_duration_is_read_from_the_banner(self):
        self.assertAlmostEqual(media.parse_duration(self.BANNER),
                               3723.45, places=2)

    def test_a_banner_without_a_duration_reports_none(self):
        self.assertIsNone(media.parse_duration("no duration here"))
        self.assertIsNone(media.parse_duration(""))

    def test_missing_files_probe_to_none(self):
        self.assertIsNone(media.probe_duration("/no/such/video.mp4"))
        self.assertEqual(media.total_duration(["/no/such/video.mp4"]), 0.0)

    def test_video_extensions_are_recognised(self):
        self.assertTrue(media.is_video_file("movie.MP4"))
        self.assertTrue(media.is_video_file("a/b/clip.webm"))
        self.assertFalse(media.is_video_file("song.mp3"))
        self.assertFalse(media.is_video_file("book.txt"))

    def test_a_folder_yields_its_videos(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_media_")
        open(os.path.join(root, "one.mp4"), "w").close()
        open(os.path.join(root, "two.mkv"), "w").close()
        open(os.path.join(root, "notes.txt"), "w").close()
        nested = os.path.join(root, "sub")
        os.makedirs(nested)
        open(os.path.join(nested, "three.mov"), "w").close()
        found = media.scan_videos(root)
        # Walk order: the folder's own files first (sorted), then sub-folders.
        self.assertEqual([os.path.basename(p) for p in found],
                         ["one.mp4", "two.mkv", "three.mov"])
        self.assertEqual(media.scan_videos(os.path.join(root, "notes.txt")), [])

    def test_a_media_file_maps_to_itself(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_media_")
        path = os.path.join(root, "clip.mp4")
        open(path, "w").close()
        self.assertEqual(media.scan_videos(path), [path])


# ---------------------------------------------------------------------------
# Output documents
# ---------------------------------------------------------------------------
class OutputTest(unittest.TestCase):
    SEGMENTS = [
        {"start": 0, "end": 120, "text": "A woman opens a door."},
        {"start": 120, "end": 240, "text": "She walks into a kitchen."},
    ]

    def test_the_script_carries_the_timing_and_the_text(self):
        text = output.format_script(self.SEGMENTS, title="My film",
                                    source="film.mp4")
        self.assertIn("My film", text)
        self.assertIn("film.mp4", text)
        self.assertIn("[0:00 - 2:00]", text)
        self.assertIn("A woman opens a door.", text)
        self.assertIn("2 description(s)", text)

    def test_the_html_script_is_a_page(self):
        text = output.format_script(self.SEGMENTS, html=True)
        self.assertTrue(text.startswith("<!DOCTYPE html>"))
        self.assertIn("A woman opens a door.", text)

    def test_the_subtitle_formats_carry_their_timing(self):
        srt = output.format_output(self.SEGMENTS, AD_FORMAT_SRT)
        self.assertIn("00:00:00,000 --> 00:02:00,000", srt)
        vtt = output.format_output(self.SEGMENTS, "vtt")
        self.assertTrue(vtt.startswith("WEBVTT"))

    def test_every_format_has_an_extension(self):
        for fmt in output.OUTPUT_FORMATS:
            self.assertTrue(output.FORMAT_EXTENSIONS[fmt].startswith("."))

    def test_write_document_writes_into_the_project_folder(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_out_")
        path = output.write_document(root, "film", self.SEGMENTS,
                                     AD_FORMAT_SCRIPT, source="film.mp4")
        self.assertEqual(os.path.basename(path), "film.txt")
        with open(path, encoding="utf-8") as fh:
            self.assertIn("A woman opens a door.", fh.read())

    def test_the_stem_is_safe_and_readable(self):
        self.assertEqual(output.output_stem("C:/videos/My Film.mp4"),
                         "My Film")
        self.assertEqual(output.output_stem(""), "description")


# ---------------------------------------------------------------------------
# The job (engine stubbed: this environment has no model)
# ---------------------------------------------------------------------------
class DescribeJobTest(unittest.TestCase):
    def _source(self, root: str) -> str:
        path = os.path.join(root, "clip.mp4")
        open(path, "w").close()
        return path

    def _project(self, root: str, source: str) -> str:
        pdir = os.path.join(root, "project")
        project.create_project(
            pdir, "Test", source, "chunked",
            [{"index": 1, "title": os.path.basename(source), "text": "",
              "source": source}],
            project_type=PROJECT_TYPE_AD,
            describe_settings={"model": "smolvlm2", "variant": "2.2b",
                               "output_format": AD_FORMAT_SRT,
                               "sources": [source]},
        )
        return pdir

    def _fake_describe(self, segments, seen):
        def _call(video, **kwargs):
            seen.append(kwargs)
            on_segment = kwargs.get("on_segment")
            for segment in segments:
                if on_segment is not None:
                    on_segment(segment)
            return list(segments)
        return _call

    def test_the_job_writes_the_chosen_format_and_the_script(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_job_")
        source = self._source(root)
        pdir = self._project(root, source)
        segments = [
            {"start": 0, "end": 120, "text": "A woman opens a door."},
            {"start": 120, "end": 240, "text": "She walks into a kitchen."},
        ]
        seen: list = []
        options = {
            "model": "smolvlm2", "variant": "2.2b", "compute": "cpu",
            "fps": AD_DEFAULT_FPS, "chunk_seconds": AD_CHUNK_SECONDS,
            "output_format": AD_FORMAT_SRT, "gap_fitting": False,
        }
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(job.engine, "describe_video",
                                  side_effect=self._fake_describe(segments, seen)):
            describe_job = job.DescribeJob(pdir, [source], options)
            describe_job.run()

        # The chosen format is written...
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip.srt")))
        # ...and so is the reviewable script, because a description without a
        # script a human can edit is only half useful.
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip.txt")))
        with open(os.path.join(pdir, "clip.txt"), encoding="utf-8") as fh:
            self.assertIn("A woman opens a door.", fh.read())

        # The run is recorded in project.json, and its segment is done.
        with open(project.project_file(pdir), encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(len(data["describe_outputs"]), 1)
        self.assertEqual(data["describe_outputs"][0]["described"], 2)
        self.assertEqual(data["describe_outputs"][0]["chunks"], 2)
        self.assertEqual(data["segments"][0]["status"], "done")

    def test_the_job_streams_a_partial_script_before_it_finishes(self):
        # Closing a long run must not lose the chunks already described: the
        # script on disk is refreshed the moment each description exists.
        root = tempfile.mkdtemp(prefix="aivs_ad_job_")
        source = self._source(root)
        pdir = self._project(root, source)
        snapshots: list = []

        def _call(video, **kwargs):
            on_segment = kwargs["on_segment"]
            on_segment({"start": 0, "end": 120, "text": "First chunk."})
            script = os.path.join(pdir, "clip.txt")
            with open(script, encoding="utf-8") as fh:
                snapshots.append(fh.read())
            on_segment({"start": 120, "end": 240, "text": "Second chunk."})
            raise engine.DescribeEngineError("stopped by the user")

        options = {"model": "smolvlm2", "variant": "2.2b",
                   "output_format": AD_FORMAT_SCRIPT}
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(job.engine, "describe_video", side_effect=_call):
            describe_job = job.DescribeJob(pdir, [source], options)
            describe_job.run()
        self.assertTrue(snapshots)
        self.assertIn("First chunk.", snapshots[0])

    def test_the_job_passes_the_gap_fitting_choices_through(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_job_")
        source = self._source(root)
        pdir = self._project(root, source)
        seen: list = []
        options = {"model": "smolvlm2", "variant": "2.2b",
                   "output_format": AD_FORMAT_SCRIPT, "gap_fitting": True,
                   "stt_model": "", "speech_model": "",
                   "min_gap": 2.0, "max_description": 8.0, "language": "en"}
        plan = {"video": source, "duration": 60.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(60.0, AD_CHUNK_SECONDS)}
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(job.engine, "describe_video",
                                  side_effect=self._fake_describe([], seen)):
            job.DescribeJob(pdir, [source], options).run()
        self.assertTrue(seen)
        self.assertTrue(seen[0]["gap_fitting"])
        self.assertEqual(seen[0]["min_gap"], 2.0)
        self.assertEqual(seen[0]["max_description_seconds"], 8.0)

    def test_no_sources_is_an_error(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_job_")
        errors: list = []
        describe_job = job.DescribeJob(
            root, [], {"model": "smolvlm2"},
            on_error=errors.append,
        )
        describe_job.run()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], engine.DescribeEngineError)


class ExtendedPlanTest(unittest.TestCase):
    """Extended audio description: stop the film, speak, go on."""

    DESCRIPTIONS = [
        {"start": 0.0, "end": 120.0, "text": "A woman opens a door.", "chunk": 0},
        {"start": 120.0, "end": 240.0, "text": "She walks into a kitchen.",
         "chunk": 1},
    ]

    def test_each_description_stops_the_film_where_it_was_timed(self):
        plan = extended.plan_extended(self.DESCRIPTIONS, duration=240.0)
        self.assertEqual(plan["count"], 2)
        self.assertEqual([stop["video_time"] for stop in plan["stops"]],
                         [0.0, 120.0])
        # The second one is heard after the first stop has been held.
        first, second = plan["stops"]
        self.assertEqual(second["extended_start"],
                         round(120.0 + first["pause"], 3))

    def test_a_pause_is_the_spoken_length_plus_a_breath(self):
        measured = extended.pause_seconds("A woman opens a door.", 3.5)
        self.assertEqual(measured, 3.5 + extended.AD_PAUSE_PADDING)
        # With no measurement the text is estimated instead.
        estimated = extended.pause_seconds("A woman opens a door.")
        self.assertAlmostEqual(
            estimated, pipeline.speaking_seconds("A woman opens a door.")
            + extended.AD_PAUSE_PADDING, places=3)

    def test_a_short_description_still_holds_the_picture_briefly(self):
        self.assertGreaterEqual(extended.pause_seconds("Hi.", 0.0),
                                extended.AD_MIN_PAUSE_SECONDS)

    def test_measured_spoken_lengths_decide_the_pauses(self):
        plan = extended.plan_extended(self.DESCRIPTIONS, 240.0,
                                      clip_seconds=[2.0, 4.0])
        pauses = [stop["pause"] for stop in plan["stops"]]
        self.assertEqual(pauses, [2.0 + extended.AD_PAUSE_PADDING,
                                  4.0 + extended.AD_PAUSE_PADDING])
        self.assertEqual(plan["added_seconds"], round(sum(pauses), 3))

    def test_the_described_film_is_longer_by_every_hold(self):
        plan = extended.plan_extended(self.DESCRIPTIONS, 240.0,
                                      clip_seconds=[2.0, 3.0])
        added = 2.0 + 3.0 + 2 * extended.AD_PAUSE_PADDING
        self.assertEqual(plan["original_duration"], 240.0)
        self.assertEqual(plan["added_seconds"], round(added, 3))
        self.assertEqual(plan["extended_duration"], round(240.0 + added, 3))
        self.assertGreater(plan["added_ratio"], 0.0)

    def test_a_description_is_never_inserted_before_the_previous_one(self):
        plan = extended.plan_extended(
            [{"start": 30.0, "text": "Later."},
             {"start": 10.0, "text": "Earlier, but second."}], 60.0)
        self.assertEqual([stop["video_time"] for stop in plan["stops"]],
                         [30.0, 30.0])

    def test_a_stop_past_the_end_of_the_film_holds_the_last_frame(self):
        plan = extended.plan_extended(
            [{"start": 999.0, "text": "After the end."}], 60.0)
        self.assertEqual(plan["stops"][0]["video_time"], 60.0)
        self.assertEqual(plan["extended_duration"],
                         round(60.0 + plan["stops"][0]["pause"], 3))

    def test_a_description_with_no_text_does_not_stop_the_film(self):
        plan = extended.plan_extended(
            [{"start": 0.0, "text": "  "}, {"start": 10.0, "text": "Real."}],
            60.0)
        self.assertEqual(plan["count"], 1)
        self.assertEqual(plan["stops"][0]["index"], 0)
        self.assertEqual(plan["stops"][0]["text"], "Real.")

    def test_an_unknown_length_is_still_planned(self):
        # The probe failed: the holds are known, the described length is not.
        plan = extended.plan_extended(self.DESCRIPTIONS, 0.0,
                                      clip_seconds=[2.0, 2.0])
        self.assertEqual(plan["original_duration"], 0.0)
        self.assertEqual(plan["extended_duration"], plan["added_seconds"])
        self.assertEqual(plan["added_ratio"], 0.0)

    def test_the_extended_segments_sit_on_the_described_clock(self):
        plan = extended.plan_extended(self.DESCRIPTIONS, 240.0,
                                      clip_seconds=[2.0, 2.0])
        segments = extended.extended_segments(plan)
        self.assertEqual(len(segments), 2)
        first, second = segments
        self.assertEqual(first["start"], 0.0)
        self.assertEqual(first["end"], round(first["pause"], 3))
        # The second description starts where the described film has reached.
        self.assertEqual(second["start"], plan["stops"][1]["extended_start"])
        self.assertEqual(second["video_start"], 120.0)
        self.assertEqual(second["pause"], plan["stops"][1]["pause"])

    def test_the_placement_is_read_back_the_way_the_dialog_writes_it(self):
        self.assertTrue(extended.is_extended("extended"))
        self.assertTrue(extended.is_extended(" Extended "))
        self.assertFalse(extended.is_extended("standard"))
        self.assertFalse(extended.is_extended(None))
        self.assertEqual(extended.normalise_placement("nonsense"), "standard")
        self.assertEqual(extended.normalise_placement("extended"), "extended")

    def test_the_summary_says_how_much_longer_the_film_became(self):
        plan = extended.plan_extended(self.DESCRIPTIONS, 600.0,
                                      clip_seconds=[4.0, 4.0])
        line = extended.placement_summary(plan, ["clip - described.mp4"])
        self.assertIn("stopped 2 time(s)", line)
        self.assertIn("clip - described.mp4", line)
        self.assertIn("longer", line)
        self.assertEqual(extended.placement_summary({"count": 0},
                                                    []),
                         "No description needed the film to be stopped.")


class ExtendedRenderTest(unittest.TestCase):
    """The described film: where the picture holds and what FFmpeg is asked to do."""

    def _plan(self, stops, duration=240.0):
        return extended.plan_extended(
            [{"start": time, "text": text} for time, text in stops],
            duration, clip_seconds=[2.0] * len(stops))

    def test_the_film_is_played_in_stretches_with_the_holds_between(self):
        plan = self._plan([(0.0, "One."), (120.0, "Two.")])
        parts = extended.render_parts(plan, 240.0)
        # [0,120) held at the first stop, [120,240) with nothing held after it...
        self.assertEqual([(p["start"], p["end"]) for p in parts],
                         [(0.0, 120.0), (120.0, 240.0)])
        hold = 2.0 + extended.AD_PAUSE_PADDING
        self.assertEqual(parts[0]["hold_before"], round(hold, 3))
        self.assertEqual(parts[1]["hold_before"], round(hold, 3))
        self.assertEqual([p["hold_after"] for p in parts], [0.0, 0.0])

    def test_two_descriptions_at_one_moment_hold_the_picture_once(self):
        plan = self._plan([(10.0, "One."), (10.0, "Two.")])
        parts = extended.render_parts(plan, 60.0)
        hold = 2.0 + extended.AD_PAUSE_PADDING
        # Both stops hold the same frame, so the film waits for both of them at
        # that one boundary instead of producing an empty stretch of film.
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["end"], 10.0)
        self.assertEqual(parts[0]["hold_before"], 0.0)
        self.assertEqual(parts[1]["hold_before"], round(2 * hold, 3))

    def test_a_stop_at_the_very_end_holds_the_final_frame(self):
        plan = self._plan([(30.0, "One."), (60.0, "Two.")])
        parts = extended.render_parts(plan, 60.0)
        self.assertEqual([(p["start"], p["end"]) for p in parts],
                         [(0.0, 30.0), (30.0, 60.0)])
        hold = 2.0 + extended.AD_PAUSE_PADDING
        self.assertEqual(parts[-1]["hold_after"], round(hold, 3))

    def test_the_graph_holds_the_picture_and_silences_the_soundtrack(self):
        plan = self._plan([(0.0, "One."), (120.0, "Two.")])
        graph = extended.render_graph(plan, 240.0)
        hold = 2.0 + extended.AD_PAUSE_PADDING
        self.assertEqual(graph.count("tpad=start_mode=clone"), 2)
        self.assertNotIn("stop_mode", graph)
        self.assertIn(f"start_duration={hold:.3f}", graph)
        self.assertIn("[0:v]trim=start=0.000:end=120.000", graph)
        self.assertIn("[0:v]trim=start=120.000:end=240.000", graph)
        self.assertIn("concat=n=2:v=1:a=1[filmv][filma]", graph)
        # The film's own audio waits out the hold instead of playing under it.
        self.assertIn(f"adelay={round(hold * 1000)}", graph)
        # ...and the spoken track is mixed on top without being attenuated.
        self.assertIn("[1:a]", graph)
        self.assertIn("amix=inputs=2:duration=first:normalize=0[outa]", graph)

    def test_every_part_of_the_graph_is_wired_to_the_concat(self):
        plan = self._plan([(0.0, "One."), (60.0, "Two."), (120.0, "Three.")])
        graph = extended.render_graph(plan, 180.0)
        for label in ("[v0]", "[a0]", "[v1]", "[a1]", "[v2]", "[a2]"):
            self.assertIn(label, graph)
        self.assertIn("concat=n=3:v=1:a=1", graph)
        # Every filter chain that opens a bracket stream also feeds a later one.
        self.assertEqual(graph.count("[filmv]"), 1)

    def test_the_command_names_the_source_the_narration_and_the_output(self):
        plan = self._plan([(0.0, "One.")])
        args = extended.render_command(
            "ffmpeg", "in.mp4", "clip - narration.wav", "out.mp4", plan, 60.0)
        self.assertEqual(args[0], "ffmpeg")
        self.assertIn("in.mp4", args)
        self.assertIn("clip - narration.wav", args)
        self.assertEqual(args[-1], "out.mp4")
        self.assertIn("-filter_complex", args)
        self.assertIn("[filmv]", args)
        self.assertIn("libx264", args)

    def test_a_run_with_nothing_to_speak_is_refused(self):
        plan = extended.plan_extended([], 60.0)
        with self.assertRaises(extended.ExtendedRenderError) as caught:
            extended.render_graph(plan, 60.0)
        self.assertIn("nothing to stop the film for", str(caught.exception))

    def test_an_unknown_length_cannot_be_cut(self):
        plan = extended.plan_extended([{"start": 5.0, "text": "One."}], 0.0,
                                      clip_seconds=[2.0])
        with self.assertRaises(extended.ExtendedRenderError):
            extended.render_parts(plan, 0.0)

    def test_the_described_film_is_named_after_the_source(self):
        self.assertEqual(extended.render_stem("C:/videos/My Film.mp4"),
                         "My Film - described")
        self.assertEqual(
            extended.described_path("C:/videos/My Film.mp4", "C:/projects/p"),
            os.path.join("C:/projects/p", "My Film - described.mp4"))

    def test_rendering_without_ffmpeg_says_so(self):
        plan = self._plan([(0.0, "One.")])
        with self.assertRaises(extended.ExtendedRenderError) as caught:
            extended.render(None, "in.mp4", "n.wav", "out.mp4", plan, 60.0)
        self.assertIn("FFmpeg", str(caught.exception))

    def test_a_missing_narration_track_is_reported_before_ffmpeg_runs(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_render_")
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        plan = self._plan([(0.0, "One.")])
        with self.assertRaises(extended.ExtendedRenderError) as caught:
            extended.render("ffmpeg", source, os.path.join(root, "gone.wav"),
                            os.path.join(root, "out.mp4"), plan, 60.0)
        self.assertIn("narration track is missing", str(caught.exception))

    def test_a_failed_ffmpeg_run_reports_its_own_complaint(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_render_")
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        narration_path = os.path.join(root, "clip - narration.wav")
        open(narration_path, "wb").close()
        plan = self._plan([(0.0, "One.")])
        failure = mock.MagicMock(returncode=1, stderr="Unknown filter: tpad\n")
        with mock.patch.object(extended.subprocess, "run",
                               return_value=failure) as runner:
            with self.assertRaises(extended.ExtendedRenderError) as caught:
                extended.render("ffmpeg", source, narration_path,
                                os.path.join(root, "out.mp4"), plan, 60.0)
        self.assertTrue(runner.called)
        self.assertIn("Unknown filter: tpad", str(caught.exception))


def _ffmpeg_available() -> bool:
    from ai_voice_studio.audio import ffmpeg as ffmpeg_module

    return bool(ffmpeg_module.find_ffmpeg())


@unittest.skipUnless(
    _ffmpeg_available(), "FFmpeg not available to render a described film")
class ExtendedRenderFilmTest(unittest.TestCase):
    """The real thing: FFmpeg renders a described film, and it is measured.

    Nothing here is stubbed.  A ten-second film with a 440 Hz tone is stopped
    twice (for three seconds and for two), the described film is written with
    the application's own filter graph, and the result is inspected: how long it
    is, whether the picture really stands still for each hold, whether the frame
    it stands on is the frame the film had reached, and which tone is heard
    while the description speaks.
    """

    FILM_SECONDS = 10.0
    STOPS = [{"start": 2.0, "text": "First description."},
             {"start": 6.0, "text": "Second description."}]
    CLIP_SECONDS = [2.4, 1.4]  # + the 0.6 s breath = 3 s and 2 s holds

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="aivs_ad_film_")
        cls.ffmpeg = _ffmpeg_available() and find_ffmpeg()
        cls.film = os.path.join(cls.root, "film.mp4")
        subprocess.run(
            [cls.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:"
             f"duration={cls.FILM_SECONDS:g}",
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:"
             f"duration={cls.FILM_SECONDS:g}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-shortest", cls.film],
            check=True, capture_output=True, timeout=300,
        )
        cls.plan = extended.plan_extended(
            cls.STOPS, cls.FILM_SECONDS, clip_seconds=cls.CLIP_SECONDS)
        # A tone exactly where each pause is: the spoken track a run would write.
        rate = 44100
        total = round((cls.plan["extended_duration"] + 1.0) * rate)
        times = np.arange(total) / rate
        samples = np.zeros(total, dtype=np.int16)
        for stop in cls.plan["stops"]:
            in_pause = ((times >= stop["extended_start"])
                        & (times < stop["extended_start"] + stop["pause"]))
            samples[in_pause] = (0.6 * 32767 * np.sin(
                2 * np.pi * 1000 * times[in_pause])).astype(np.int16)
        cls.narration = os.path.join(cls.root, "film - extended narration.wav")
        with wave.open(cls.narration, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        cls.out = extended.described_path(cls.film, cls.root)
        extended.render(cls.ffmpeg, cls.film, cls.narration, cls.out, cls.plan,
                        cls.FILM_SECONDS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _frame(self, path: str, seconds: float) -> np.ndarray:
        raw = subprocess.run(
            [self.ffmpeg, "-v", "error", "-ss", str(seconds), "-i", path,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True, timeout=120,
        ).stdout
        return np.frombuffer(raw, dtype=np.uint8).astype(float)

    @staticmethod
    def _difference(a, b) -> float:
        """Mean absolute pixel difference (0 = the same picture).

        Not equality: H.264 is lossy, so a duplicated frame decodes to
        *almost* the same pixels, and a picture that moved differs by a lot.
        """
        return float(np.abs(a - b).mean())

    def _tone(self, seconds: float, window: float = 1.5) -> float:
        pcm = subprocess.run(
            [self.ffmpeg, "-v", "error", "-i", self.out, "-ac", "1",
             "-ar", "44100", "-ss", str(seconds), "-t", str(window),
             "-f", "s16le", "-"],
            check=True, capture_output=True, timeout=120,
        ).stdout
        data = np.frombuffer(pcm, dtype=np.int16).astype(float)
        spectrum = np.abs(np.fft.rfft(data * np.hanning(len(data))))
        freqs = np.fft.rfftfreq(len(data), 1 / 44100)
        return float(freqs[int(np.argmax(spectrum))])

    def test_the_described_film_is_longer_by_exactly_the_holds(self):
        duration = media.probe_duration(self.out)
        self.assertAlmostEqual(duration, 15.0, delta=0.2)
        self.assertEqual(self.plan["extended_duration"], 15.0)

    def test_the_picture_stands_still_for_each_hold(self):
        # Both samples are inside the three-second hold at 0:02.
        self.assertLess(self._difference(self._frame(self.out, 2.1),
                                        self._frame(self.out, 3.5)), 2.0)
        # ...and the film is moving again once the hold is over.
        self.assertGreater(self._difference(self._frame(self.out, 4.8),
                                           self._frame(self.out, 6.5)), 10.0)

    def test_the_held_frame_is_the_frame_the_film_had_reached(self):
        held = self._frame(self.out, 2.1)
        at_stop = self._frame(self.film, 2.0)
        next_frame = self._frame(self.film, 2.1)
        self.assertLess(self._difference(held, at_stop), 5.0)
        # The frame after it differs by about as much as two film frames do.
        self.assertGreater(self._difference(held, next_frame), 5.0)

    def test_the_film_is_quiet_while_the_description_speaks(self):
        self.assertAlmostEqual(self._tone(3.0), 1000.0, delta=20.0)
        self.assertAlmostEqual(self._tone(6.5), 440.0, delta=20.0)
        self.assertAlmostEqual(self._tone(9.3), 1000.0, delta=20.0)
        self.assertAlmostEqual(self._tone(12.0), 440.0, delta=20.0)


class DescribeJobNarrationTest(unittest.TestCase):
    """The job speaks the descriptions as well, when it was asked to."""

    SEGMENTS = [
        {"start": 0.0, "end": 120.0, "text": "A woman opens a door."},
        {"start": 120.0, "end": 240.0, "text": "She walks into a kitchen."},
    ]

    def _source(self, root: str) -> str:
        path = os.path.join(root, "clip.mp4")
        open(path, "w").close()
        return path

    def _project(self, root: str, source: str) -> str:
        pdir = os.path.join(root, "project")
        project.create_project(
            pdir, "Test", source, "chunked",
            [{"index": 1, "title": os.path.basename(source), "text": "",
              "source": source}],
            project_type=PROJECT_TYPE_AD,
            describe_settings={"model": "smolvlm2", "variant": "2.2b",
                               "output_format": AD_FORMAT_SCRIPT,
                               "sources": [source]},
        )
        return pdir

    def _options(self, **overrides) -> dict:
        options = {
            "model": "smolvlm2", "variant": "2.2b", "compute": "cpu",
            "fps": AD_DEFAULT_FPS, "chunk_seconds": AD_CHUNK_SECONDS,
            "output_format": AD_FORMAT_SCRIPT, "gap_fitting": False,
            "narration": True, "voice": dict(VOICE),
            "narration_format": "wav",
            "tts": {"rate": 1.0, "pitch": 1.0, "volume": 1.0,
                    "punctuation": "default", "compute": "cpu"},
        }
        options.update(overrides)
        return options

    def _run(self, root: str, options: dict, progress: list | None = None):
        source = self._source(root)
        pdir = self._project(root, source)
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        errors: list = []
        fake_voice = _FakeVoice(seconds=0.5)
        # FFmpeg is faked out: the described video is one of the three outputs a
        # standard run writes, and where it lands is what these tests are about,
        # not what libx264 does with a source clip that is zero bytes long.
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: list(self.SEGMENTS)), \
                mock.patch.object(job.extended, "mix",
                                  side_effect=self._fake_mix), \
                mock.patch.object(job.narration, "get_engine",
                                  return_value=fake_voice):
            describe_job = job.DescribeJob(
                pdir, [source], options, on_error=errors.append,
                on_progress=(lambda message, fraction: progress.append(
                    (message, fraction))) if progress is not None else None,
            )
            describe_job.run()
        return pdir, describe_job, errors

    @staticmethod
    def _fake_mix(ffmpeg, source, narration_path, out_path, **kwargs):
        """A stand-in for FFmpeg that records what it was asked to mix."""
        assert os.path.isfile(source) and os.path.isfile(narration_path)
        open(out_path, "wb").close()
        return out_path

    def _render_mix_with(self, track_priority):
        """Mix one standard film, recording what the mixer was handed."""
        root = tempfile.mkdtemp(prefix="aivs_ad_prio_")
        self.addCleanup(shutil.rmtree, root, True)
        source = self._source(root)
        pdir = self._project(root, source)
        options = self._options()
        if track_priority is not None:
            options["track_priority"] = track_priority
        seen: dict = {}

        def fake_mix(ffmpeg, source, narration_path, out_path, **kwargs):
            seen.update(kwargs)
            open(out_path, "wb").close()
            return out_path

        run = job.DescribeJob(pdir, [source], options)
        record = {"files": [], "narration": {}}
        with mock.patch.object(job.extended, "mix", side_effect=fake_mix):
            run._render_mix(
                record, source,
                # What the narration track reports about itself: the file, and
                # the moments the voice is actually heard in it.
                {"file": "clip - narration.wav",
                 "turns": [[1.0, 2.5], [9.0, 11.0]]},
            )
        return record, seen

    def test_the_run_tells_the_mixer_which_track_gives_way(self):
        record, seen = self._render_mix_with("higher50")
        # Half the film's own soundtrack comes down; the description does not.
        self.assertEqual(seen.get("film_volume"), 0.5)
        # ...and nothing dips: the film is brought down for its whole length.
        self.assertEqual(seen.get("duck_level"), 1.0)
        self.assertEqual(seen.get("duck_turns"), ())
        self.assertEqual(record["track_priority"], "higher50")
        self.assertNotIn("duck_level", record)
        self.assertEqual(record["film"], "clip - described.mp4")
        self.assertTrue(record["narration"]["mixed"])

    def test_the_ducking_run_hands_over_the_moments_the_voice_speaks(self):
        """*Duck, don't lower*: no gain on the film, a dip under the voice."""
        record, seen = self._render_mix_with("duck")
        self.assertEqual(seen.get("film_volume"), 1.0)
        self.assertEqual(seen.get("duck_level"), 0.25)
        self.assertEqual(seen.get("duck_turns"), [[1.0, 2.5], [9.0, 11.0]])
        self.assertEqual(record["track_priority"], "duck")
        self.assertEqual(record["duck_level"], 0.25)
        self.assertTrue(record["narration"]["mixed"])

    def test_a_run_that_chose_nothing_mixes_both_tracks_as_they_are(self):
        for priority in (None, "same"):
            with self.subTest(priority=priority):
                record, seen = self._render_mix_with(priority)
                self.assertEqual(seen.get("film_volume"), 1.0)
                self.assertEqual(record["track_priority"], "same")

    def test_the_job_speaks_the_descriptions_next_to_the_script(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        pdir, describe_job, errors = self._run(root, self._options())
        self.assertEqual(errors, [])
        name = "clip - narration.wav"
        self.assertTrue(os.path.isfile(os.path.join(pdir, name)))
        # The script and the spoken track sit together, both written.
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip.txt")))
        record = describe_job.outputs[0]
        self.assertIn(name, record["files"])
        self.assertEqual(record["narration"]["spoken"], 2)
        self.assertEqual(record["narration"]["voice_name"], "Amy")

    def test_a_standard_run_writes_the_text_the_audio_and_the_video(self):
        """The three outputs, in one run: script, spoken track, described film.

        Regression: a standard run wrote the script and the spoken track and
        stopped there, so the project had no video at all - the description
        could not be watched, only listened to beside the original film.
        """
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        pdir, describe_job, errors = self._run(root, self._options())
        self.assertEqual(errors, [])
        record = describe_job.outputs[0]
        self.assertEqual(sorted(record["files"]),
                         ["clip - described.mp4", "clip - narration.wav",
                          "clip.txt"])
        # The described film is the third output, and it is the standard one:
        # nothing was stopped, so it carries no extended placement.
        self.assertEqual(record["film"], "clip - described.mp4")
        self.assertTrue(os.path.isfile(os.path.join(pdir, record["film"])))
        self.assertNotIn("extended", record)
        self.assertTrue(record["narration"]["mixed"])

        data = json.loads(open(project.project_file(pdir), encoding="utf-8").read())
        self.assertEqual(len(data[output.NARRATION_KEY]), 1)
        self.assertEqual(data[output.NARRATION_KEY][0]["file"],
                         "clip - narration.wav")
        self.assertEqual(data[output.NARRATION_KEY][0]["source"], record["source"])

    def test_the_progress_line_says_what_is_being_spoken(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        progress: list = []
        self._run(root, self._options(), progress=progress)
        messages = [message for message, _fraction in progress]
        self.assertTrue(any("Narrating description" in m for m in messages))
        self.assertTrue(any("narration.wav" in m for m in messages))

    def test_a_restart_removes_the_spoken_track_too(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        pdir, _describe_job, _errors = self._run(root, self._options())
        data = project.load_project(pdir)
        removed = output.remove_outputs(pdir, data)
        # All three outputs go: the script, the narration and the described film.
        self.assertEqual(removed, 3)
        self.assertFalse(os.path.exists(os.path.join(pdir, "clip - narration.wav")))
        self.assertFalse(os.path.exists(os.path.join(pdir, "clip - described.mp4")))
        after = project.load_project(pdir)
        self.assertEqual(after[output.NARRATION_KEY], [])
        self.assertEqual(after["describe_outputs"], [])

    def test_narration_without_a_voice_stops_before_any_describing(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        source = self._source(root)
        pdir = self._project(root, source)
        errors: list = []
        plan = mock.MagicMock()
        with mock.patch.object(job.engine, "plan_for_video", plan), \
                mock.patch.object(job.engine, "describe_video") as describe:
            describe_job = job.DescribeJob(
                pdir, [source], self._options(voice={}), on_error=errors.append)
            describe_job.run()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], engine.DescribeEngineError)
        self.assertIn("no voice", str(errors[0]).lower())
        self.assertFalse(plan.called)
        self.assertFalse(describe.called)
        # Only the project file exists: nothing was described and nothing spoken.
        self.assertEqual([name for name in os.listdir(pdir)
                          if name != "project.json"], [])

    def test_a_voice_that_fails_is_a_warning_not_a_lost_description(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        source = self._source(root)
        pdir = self._project(root, source)
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        errors: list = []
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: list(self.SEGMENTS)), \
                mock.patch.object(job.narration, "get_engine",
                                  side_effect=RuntimeError("no model on disk")):
            describe_job = job.DescribeJob(
                pdir, [source], self._options(), on_error=errors.append)
            describe_job.run()
        self.assertEqual(errors, [])
        # The text is the product of the run and survives the broken voice...
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip.txt")))
        self.assertFalse(os.path.exists(os.path.join(pdir, "clip - narration.wav")))
        # ...and the user is told exactly what did not happen.
        self.assertEqual(len(describe_job.warnings), 1)
        self.assertIn("no model on disk", describe_job.warnings[0])
        self.assertIsNone(describe_job.outputs[0].get("narration"))

    def test_a_described_video_that_cannot_be_written_is_a_warning(self):
        """The two written outputs survive a failed render; the user is told.

        Rendering the film is the last thing a standard run does and the only
        part that needs FFmpeg, so it must never cost the script and the spoken
        track that are already on disk.
        """
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        source = self._source(root)
        pdir = self._project(root, source)
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        errors: list = []
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: list(self.SEGMENTS)), \
                mock.patch.object(job.narration, "get_engine",
                                  return_value=_FakeVoice(seconds=0.5)), \
                mock.patch.object(
                    job.extended, "mix",
                    side_effect=job.extended.ExtendedRenderError(
                        "FFmpeg is required to write a described video")):
            describe_job = job.DescribeJob(
                pdir, [source], self._options(), on_error=errors.append)
            describe_job.run()
        self.assertEqual(errors, [])
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip.txt")))
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip - narration.wav")))
        self.assertEqual(describe_job.outputs[0].get("film"), None)
        self.assertEqual(len(describe_job.warnings), 1)
        self.assertIn("described video", describe_job.warnings[0])
        self.assertIn("FFmpeg", describe_job.warnings[0])

    # -- extended audio description ----------------------------------------
    def _extended_options(self, **overrides) -> dict:
        options = self._options(narration=False, placement="extended")
        options.update(overrides)
        return options

    def test_extended_mode_stops_the_film_for_every_description(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        pdir, describe_job, errors = self._run(root, self._extended_options())
        self.assertEqual(errors, [])
        record = describe_job.outputs[0]
        # The pause is the measured length of the spoken description.
        hold = 0.5 + extended.AD_PAUSE_PADDING
        self.assertEqual(record["extended"]["count"], 2)
        self.assertEqual([stop["pause"] for stop in record["extended"]["stops"]],
                         [round(hold, 3), round(hold, 3)])
        self.assertEqual(record["extended"]["stops"][1]["extended_start"],
                         round(120.0 + hold, 3))
        # The spoken track is on the described film's clock, and named apart
        # from a standard narration track.
        self.assertEqual(record["narration"]["placement"], "extended")
        self.assertEqual(record["narration"]["file"],
                         "clip - extended narration.wav")
        self.assertTrue(os.path.isfile(os.path.join(
            pdir, "clip - extended narration.wav")))

    def test_the_documents_of_a_described_film_use_its_own_clock(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        pdir, _job, _errors = self._run(root, self._extended_options())
        with open(os.path.join(pdir, "clip.txt"), encoding="utf-8") as fh:
            script = fh.read()
        self.assertIn("Extended audio description", script)
        self.assertIn("picture held", script)
        self.assertIn("A woman opens a door.", script)
        data = json.loads(open(project.project_file(pdir), encoding="utf-8").read())
        self.assertEqual(len(data[output.NARRATION_KEY]), 1)
        self.assertEqual(data[output.NARRATION_KEY][0]["placement"], "extended")

    def test_the_described_film_is_recorded_when_it_is_written(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        source = self._source(root)
        pdir = self._project(root, source)
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}

        def _fake_render(ffmpeg, src, narration_path, out_path, render_plan,
                         duration=0.0, **kwargs):
            self.assertGreater(len(render_plan["stops"]), 0)
            self.assertTrue(os.path.isfile(narration_path))
            open(out_path, "wb").close()
            return out_path

        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: list(self.SEGMENTS)), \
                mock.patch.object(job.narration, "get_engine",
                                  return_value=_FakeVoice(seconds=0.5)), \
                mock.patch.object(job.extended, "render", side_effect=_fake_render):
            describe_job = job.DescribeJob(pdir, [source], self._extended_options())
            describe_job.run()
        record = describe_job.outputs[0]
        self.assertEqual(record["film"], "clip - described.mp4")
        self.assertIn("clip - described.mp4", record["files"])
        self.assertTrue(os.path.isfile(os.path.join(pdir, "clip - described.mp4")))
        self.assertEqual(describe_job.warnings, [])

    def test_a_missing_ffmpeg_leaves_the_pause_points_behind(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        pdir, describe_job, _errors = self._run(root, self._extended_options())
        # No FFmpeg here: the described video cannot be written, and that is
        # said out loud rather than silently skipped.
        self.assertEqual(len(describe_job.warnings), 1)
        self.assertIn("described video", describe_job.warnings[0])
        self.assertIn("FFmpeg", describe_job.warnings[0])
        # Everything else the user asked for is on disk.
        for name in ("clip.txt", "clip - extended narration.wav"):
            self.assertTrue(os.path.isfile(os.path.join(pdir, name)))
        self.assertIsNone(describe_job.outputs[0].get("film"))

    def test_extended_mode_stops_the_film_instead_of_fitting_the_pauses(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        source = self._source(root)
        pdir = self._project(root, source)
        seen: list = []
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        options = self._extended_options(gap_fitting=True, speech_model="whisper")
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: (
                        seen.append(kwargs) or list(self.SEGMENTS))), \
                mock.patch.object(job.narration, "get_engine",
                                  return_value=_FakeVoice(seconds=0.5)):
            job.DescribeJob(pdir, [source], options).run()
        self.assertTrue(seen)
        self.assertFalse(seen[0]["gap_fitting"])
        self.assertEqual(seen[0]["speech_segments"], [])

    def test_a_restart_removes_the_described_film_and_its_track(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        source = self._source(root)
        pdir = self._project(root, source)
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}

        def _fake_render(ffmpeg, src, narration_path, out_path, render_plan,
                         duration=0.0, **kwargs):
            open(out_path, "wb").close()
            return out_path

        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: list(self.SEGMENTS)), \
                mock.patch.object(job.narration, "get_engine",
                                  return_value=_FakeVoice(seconds=0.5)), \
                mock.patch.object(job.extended, "render", side_effect=_fake_render):
            job.DescribeJob(pdir, [source], self._extended_options()).run()
        before = sorted(os.listdir(pdir))
        self.assertIn("clip - described.mp4", before)
        self.assertIn("clip - extended narration.wav", before)
        # Restart All Descriptions leaves the folder as it was: the described
        # film is a description output like any other.
        output.remove_outputs(pdir, project.load_project(pdir))
        self.assertEqual(os.listdir(pdir), ["project.json"])

    def test_extended_mode_needs_a_voice_before_it_describes_anything(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        source = self._source(root)
        pdir = self._project(root, source)
        errors: list = []
        plan = mock.MagicMock()
        with mock.patch.object(job.engine, "plan_for_video", plan), \
                mock.patch.object(job.engine, "describe_video") as describe:
            describe_job = job.DescribeJob(
                pdir, [source], self._extended_options(voice={}),
                on_error=errors.append)
            describe_job.run()
        self.assertEqual(len(errors), 1)
        self.assertIn("voice", str(errors[0]).lower())
        self.assertIn("extended", str(errors[0]).lower())
        self.assertFalse(plan.called)
        self.assertFalse(describe.called)

    def test_a_voice_that_fails_leaves_the_descriptions_on_the_original_clock(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobext_")
        source = self._source(root)
        pdir = self._project(root, source)
        plan = {"video": source, "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        errors: list = []
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(
                    job.engine, "describe_video",
                    side_effect=lambda video, **kwargs: list(self.SEGMENTS)), \
                mock.patch.object(job.narration, "get_engine",
                                  side_effect=RuntimeError("no voice on disk")):
            describe_job = job.DescribeJob(
                pdir, [source], self._extended_options(), on_error=errors.append)
            describe_job.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(describe_job.warnings), 1)
        self.assertIn("no voice on disk", describe_job.warnings[0])
        # There is no pause without a voice, so the descriptions are written for
        # the film as it is rather than for a described film that does not exist.
        script = open(os.path.join(pdir, "clip.txt"), encoding="utf-8").read()
        self.assertNotIn("picture held", script)
        self.assertIsNone(describe_job.outputs[0].get("extended"))

    def test_no_narration_means_no_spoken_file(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_jobnarr_")
        pdir, describe_job, _errors = self._run(
            root, self._options(narration=False))
        self.assertEqual(sorted(os.listdir(pdir)), ["clip.txt", "project.json"])
        self.assertEqual(describe_job.warnings, [])
        data = project.load_project(pdir)
        self.assertEqual(data[output.NARRATION_KEY], [])


# ---------------------------------------------------------------------------
# Narration (the voice is stubbed: this environment has no downloaded one)
# ---------------------------------------------------------------------------
VOICE = {
    "tts": "piper", "tts_name": "Piper", "language": "en_US",
    "variant": "medium", "voice": "amy", "voice_name": "Amy", "sid": 3,
    "engine": "vits", "dir": "C:/models/amy",
}
OTHER_VOICE = {
    "tts": "kokoro", "tts_name": "Kokoro", "language": "en_US",
    "variant": "82m", "voice": "af_heart", "voice_name": "Heart", "sid": 0,
    "engine": "kokoro", "dir": "C:/models/kokoro",
}


class _FakeVoice:
    """A TTS engine that returns a constant tone instead of speaking."""

    def __init__(self, sample_rate=1000, seconds=0.25, value=1234):
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.value = value
        self.calls: list = []

    def synthesize(self, text, sid=0, speed=1.0, pitch=1.0, volume=1.0):
        self.calls.append((text, sid, speed))
        return np.full(
            round(self.sample_rate * self.seconds), self.value,
            dtype=np.int16,
        )


class NarrationTimelineTest(unittest.TestCase):
    """Where a description is heard: the whole point of the narration track."""

    RATE = 1000

    def _clip(self, seconds: float, value: int = 1000) -> np.ndarray:
        return np.full(round(seconds * self.RATE), value, dtype=np.int16)

    def test_a_description_is_spoken_at_its_own_time(self):
        samples, placed = narration.assemble([self._clip(1.0)], self.RATE, [2.0])
        self.assertEqual(placed, [2.0])
        self.assertEqual(int(samples[:2000].max()), 0)
        self.assertEqual(int(samples[2000:3000].min()), 1000)
        # Half a second of silence follows, so the last word is never cut off.
        self.assertEqual(samples.shape[0], 3500)

    def test_one_description_never_starts_before_the_previous_finished(self):
        _samples, placed = narration.assemble(
            [self._clip(1.0), self._clip(1.0)], self.RATE, [0.0, 0.1])
        self.assertEqual(placed, [0.0, 1.0])

    def test_a_description_without_a_time_follows_the_previous_one(self):
        samples, placed = narration.assemble(
            [self._clip(0.5), self._clip(0.5)], self.RATE, [None, None])
        self.assertEqual(placed, [0.0, 0.5])
        self.assertEqual(samples.shape[0], 1500)

    def test_a_long_pause_is_kept_as_silence(self):
        samples, placed = narration.assemble(
            [self._clip(0.25), self._clip(0.25)], self.RATE, [0.0, 5.0])
        self.assertEqual(placed, [0.0, 5.0])
        self.assertEqual(int(samples[500:5000].max()), 0)
        self.assertEqual(samples.shape[0], 5750)

    def test_the_tail_can_be_turned_off(self):
        samples, _placed = narration.assemble(
            [self._clip(1.0)], self.RATE, [0.0], tail_seconds=0.0)
        self.assertEqual(samples.shape[0], 1000)

    def test_an_empty_run_is_just_silence(self):
        samples, placed = narration.assemble([], self.RATE)
        self.assertEqual(placed, [])
        self.assertEqual(samples.shape[0], int(narration.TAIL_SECONDS * self.RATE))


class NarrationTest(unittest.TestCase):
    """Speaking the descriptions into a file, with the voice stubbed out."""

    SEGMENTS = [
        {"start": 0.0, "end": 1.0, "text": "A woman opens a door."},
        {"start": 3.0, "end": 4.0, "text": "She walks into a kitchen."},
    ]

    def _narrator(self, voice=None, **kwargs):
        fake = _FakeVoice()
        narrator = narration.Narrator(
            VOICE if voice is None else voice,
            {"rate": 1.25, "punctuation": "default"},
            engine_factory=lambda: fake,
            **kwargs,
        )
        return narrator, fake

    def test_a_markup_voice_keeps_its_tags_through_the_narration(self):
        """A description's director's notes reach the engine as typed.

        The narration is a second path into the same punctuation modes as a
        recording (``tts.markup``): an engine that reads expression markup is
        given it untouched, however the mode would have expanded it.
        """
        voice = dict(VOICE, tts="kitten_tts_2", engine="kitten_tts_2",
                     voice_name="Bella", sid=0)
        fake = _FakeVoice()
        narrator = narration.Narrator(
            voice,
            {"rate": 1.0, "punctuation": "all"},
            engine_factory=lambda: fake,
        )
        segments = [dict(self.SEGMENTS[0],
                         text="[sad] A woman <sigh> (((slowly))) opens it!")]
        narrator.narrate(segments, tempfile.mkdtemp(prefix="aivs_ad_markup_"))
        spoken = fake.calls[0][0]
        for span in ("[sad]", "<sigh>", "(((slowly)))"):
            self.assertIn(span, spoken)
        # The mode still expanded everything that is not a director's note.
        self.assertIn("exclamation mark", spoken)

    def test_the_narration_is_named_after_the_source_like_the_script(self):
        self.assertEqual(narration.narration_stem("C:/videos/My Film.mp4"),
                         "My Film - narration")
        self.assertEqual(narration.narration_stem(""), "description - narration")
        self.assertEqual(narration.narration_stem("film.mp4"),
                         output.output_stem("film.mp4") + " - narration")

    def test_only_descriptions_with_text_are_spoken(self):
        segments = [*self.SEGMENTS, {"start": 5, "end": 6, "text": "  "}]
        self.assertEqual(len(narration.spoken_segments(segments)), 2)

    def test_narrating_writes_one_file_and_reports_what_it_spoke(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")
        narrator, fake = self._narrator()
        info = narrator.narrate(self.SEGMENTS, root,
                                source="C:/videos/My Film.mp4")
        self.assertEqual(info["file"], "My Film - narration.wav")
        self.assertEqual(info["format"], "wav")
        self.assertEqual(info["spoken"], 2)
        self.assertEqual(info["descriptions"], 2)
        self.assertEqual(info["note"], "")
        self.assertEqual(info["voice_name"], "Amy")
        # The rate and the voice's own speaker id reach the engine.
        self.assertEqual([call[1] for call in fake.calls], [3, 3])
        self.assertEqual(fake.calls[0][2], 1.25)
        self.assertIn("door", fake.calls[0][0])

    def test_the_file_really_holds_the_descriptions_at_their_times(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")
        narrator, _fake = self._narrator()
        info = narrator.narrate(self.SEGMENTS, root,
                                source="C:/videos/My Film.mp4")
        with wave.open(os.path.join(root, info["file"]), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getframerate(), 1000)
            samples = np.frombuffer(handle.readframes(handle.getnframes()),
                                    dtype=np.int16)
        # 0.25 s of voice, silence until the second description at 3 s, then the
        # tail: the timings of the description script, in audio.
        self.assertEqual(int(samples[:250].min()), 1234)
        self.assertEqual(int(samples[250:3000].max()), 0)
        self.assertEqual(int(samples[3000:3250].min()), 1234)
        self.assertEqual(samples.shape[0], 3750)

    def test_narrating_reports_each_description_as_it_is_spoken(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")
        narrator, _fake = self._narrator()
        seen: list = []
        narrator.narrate(self.SEGMENTS, root, source="clip.mp4",
                         on_progress=lambda message, fraction: seen.append(
                             (message, fraction)))
        self.assertEqual(len(seen), 2)
        self.assertIn("1 of 2", seen[0][0])
        self.assertEqual(seen[-1][1], 1.0)

    def test_the_moments_the_voice_is_heard_are_recorded_with_the_track(self):
        """When the voice speaks, so the film can be dipped under it and only it.

        The mixer must not have to guess: the track was written here, from
        measured clips, so where each description is heard is recorded where it
        is known (see ``extended.ducking_expression``).
        """
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")
        narrator, _fake = self._narrator()
        info = narrator.narrate(self.SEGMENTS, root, source="clip.mp4")
        # 0.25 s of voice per description, at 0 s and at 3 s.
        self.assertEqual(info["turns"], [[0.0, 0.25], [3.0, 3.25]])
        # A description whose moment is already taken lands after the one before
        # it, and the record says where it really landed, not where it asked to.
        info = narrator.narrate(
            [{"start": 0.0, "end": 1.0, "text": "One."},
             {"start": 0.1, "end": 1.0, "text": "Two."}],
            root, source="clip.mp4")
        self.assertEqual(info["turns"], [[0.0, 0.25], [0.25, 0.5]])

    def test_a_description_with_no_text_is_not_a_narration(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")
        narrator, fake = self._narrator()
        with self.assertRaises(narration.NarrationError) as caught:
            narrator.narrate([{"start": 0, "end": 1, "text": "  "}], root)
        self.assertIn("no text", str(caught.exception))
        self.assertEqual(fake.calls, [])
        self.assertEqual(os.listdir(root), [])

    def test_a_voice_that_will_not_load_is_reported(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")

        def _boom():
            raise RuntimeError("the model is corrupt")

        narrator = narration.Narrator(VOICE, engine_factory=_boom)
        with self.assertRaises(narration.NarrationError) as caught:
            narrator.narrate(self.SEGMENTS, root, source="clip.mp4")
        self.assertIn("corrupt", str(caught.exception))
        # Nothing was left behind for a file that does not exist.
        self.assertEqual(os.listdir(root), [])

    def test_no_voice_is_an_error_before_anything_is_written(self):
        with self.assertRaises(narration.NarrationError):
            narration.Narrator({})

    def test_a_missing_ffmpeg_falls_back_to_wav_and_says_so(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_narr_")
        narrator, _fake = self._narrator(output_format="mp3")
        with mock.patch("ai_voice_studio.audio.ffmpeg.find_ffmpeg",
                        return_value=None):
            info = narrator.narrate(self.SEGMENTS, root, source="clip.mp4")
        # The narration the user asked for still exists - as a WAV, with a note.
        self.assertEqual(info["format"], "wav")
        self.assertTrue(os.path.isfile(os.path.join(root, "clip - narration.wav")))
        self.assertIn("FFmpeg", info["note"])

    def test_the_voice_is_described_the_way_the_box_shows_it(self):
        narrator, _fake = self._narrator()
        self.assertEqual(narrator.voice_label, "Piper - Amy")

    def test_the_remembered_voice_is_preselected_when_it_is_still_installed(self):
        voices = [VOICE, OTHER_VOICE]
        chosen = narration.resolve_voice(
            voices, {"tts": "kokoro", "language": "en_US",
                     "variant": "82m", "voice": "af_heart"})
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["voice"], "af_heart")

    def test_a_removed_variant_still_finds_its_voice(self):
        chosen = narration.resolve_voice(
            [VOICE, OTHER_VOICE],
            {"tts": "kokoro", "language": "en_US", "variant": "gone",
             "voice": "af_heart"})
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["tts"], "kokoro")

    def test_a_voice_that_is_gone_entirely_picks_nothing(self):
        self.assertIsNone(narration.resolve_voice(
            [VOICE], {"tts": "f5-tts", "language": "en",
                      "variant": "x", "voice": "y"}))
        self.assertIsNone(narration.resolve_voice([VOICE], {}))
        self.assertIsNone(narration.resolve_voice([], {"tts": "piper"}))


# ---------------------------------------------------------------------------
# Engine plumbing
# ---------------------------------------------------------------------------
#: A real (219 byte, 8x8) JPEG, so the worker's frame loading is exercised on a
#: file the imaging library can actually read.
_ONE_PIXEL_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAPTGF2YzYzLjEuMTAyAP/bAEMACAQEBAQEBQUFBQUF"
    "BgYGBgYGBgYGBgYGBgcHBwgICAcHBwYGBwcICAgICQkJCAgICAkJCgoKDAwLCw4ODhERFP/E"
    "AEoAAQAAAAAAAAAAAAAAAAAAAAABAQAAAAAAAAAAAAAAAAAAAAAQAQAAAAAAAAAAAAAAAAAA"
    "AAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAAIAAgDASIAAhEAAxEA/9oADAMBAAIRAxEAPwAA"
    "D//Z"
)


class _FakeProcessor:
    """A transformers processor, in whichever chat-template form it needs.

    ``embedded=False`` stands for a family whose template refuses frames inside
    the message and wants them handed over beside the prompt (Gemma 3).
    """

    def __init__(self, embedded: bool = True):
        self.embedded = embedded
        self.embedded_attempts: list = []
        self.embedded_calls: list = []
        self.placeholder_calls: list = []
        self.text_calls: list = []

    def apply_chat_template(self, messages, **kwargs):
        user = messages[-1]["content"]
        carries_frames = any("image" in item for item in user)
        if carries_frames:
            self.embedded_attempts.append(("embedded", messages[-1]))
            if not self.embedded:
                raise ValueError(
                    "frames must be passed to the processor, not embedded here"
                )
        if kwargs.get("tokenize"):
            self.embedded_calls.append(("embedded", messages[-1]))
            return {"input_ids": np.zeros((1, 3), dtype=np.int64)}
        self.placeholder_calls.append(("placeholder", messages[-1]))
        return "placeholder prompt"

    def __call__(self, **kwargs):
        self.text_calls.append({
            "prompt": (kwargs.get("text") or [""])[0],
            "images": kwargs.get("images", []),
        })
        return {"input_ids": np.zeros((1, 3), dtype=np.int64)}

    def batch_decode(self, _tokens, skip_special_tokens=True):
        return ["A woman opens a door."]


class _FakeModel:
    class _Param:
        device = "cpu"

    def parameters(self):
        return iter([self._Param()])

    def generate(self, **kwargs):
        return np.zeros((1, 10), dtype=np.int64)


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


try:  # Pillow is what the worker uses to measure a frame; optional for tests
    from PIL import Image as _Image
except Exception:
    _Image = None


class EngineTest(unittest.TestCase):
    def test_planning_a_missing_video_is_an_error(self):
        with self.assertRaises(engine.DescribeEngineError):
            engine.plan_for_video("/no/such/video.mp4")

    def test_progress_lines_become_callbacks(self):
        seen: list = []
        engine._forward_progress("AIVS-PROGRESS|42%|Describing chunk 1 of 3...",
                                 lambda message, fraction: seen.append(
                                     (message, fraction)))
        self.assertEqual(seen, [("Describing chunk 1 of 3...", 0.42)])

    def test_a_broken_progress_line_is_ignored(self):
        seen: list = []
        engine._forward_progress("AIVS-PROGRESS|nonsense", lambda *a: seen.append(a))
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0][1])

    def test_worker_noise_is_not_reported_as_the_cause(self):
        detail = engine._worker_error_detail(
            "10%|# | 1/10 [00:01<00:09]\n"
            "Warning: deprecated thing\n"
            "RuntimeError: The model could not be loaded"
        )
        self.assertIn("RuntimeError", detail)
        self.assertNotIn("Warning", detail)

    def test_a_window_permission_error_is_not_buried_under_its_advice(self):
        """Regression: a real run reported only the advice, not the cause.

        Hugging Face raises ``OSError`` whose *message* is several lines telling
        the reader to turn on Developer Mode (which a filter for advice-looking
        lines removes), and the traceback's last lines are that advice plus a
        URL.  The dialog showed those, so a failed download looked like a
        settings recommendation and the ``OSError`` above it was never seen.
        """
        detail = engine._worker_error_detail(
            "Loading weights:  30%|###      | 200/713\n"
            "Traceback (most recent call last):\n"
            "  File \"worker.py\", line 1, in <module>\n"
            "OSError: [WinError 1314] A required privilege is not held"
            " by the client: 'C:\\\\cache\\\\blobs'\n"
            "In order to support symlinks on Windows you either need to"
            " activate Developer Mode or run Python as an administrator\n"
            "https://docs.python.org/3/library/os.html#os.symlink\n"
        )
        # The cause comes first now, with the advice that explains it after it.
        self.assertTrue(detail.startswith("OSError"), detail)
        self.assertIn("WinError 1314", detail)
        self.assertNotIn("Loading weights", detail)

    def test_a_crashed_worker_says_what_its_exit_code_means(self):
        """Regression: a dead process was blamed on a warning three lines up.

        A real run here ended with 0xC0000005 after the model had offloaded part
        of itself to the disk, and the window reported the Windows-symlink
        warning printed above the crash as if it were the cause - no mention of
        the crash at all, and nothing to act on.
        """
        note = engine._crash_note(3221225477)
        self.assertIn("access violation", note)
        self.assertIn("3221225477", note)
        self.assertIn("video memory", note)
        self.assertIn("smaller variant", note)
        # A signal on POSIX, and an ordinary failure, are named as themselves.
        self.assertIn("signal 9", engine._crash_note(-9))
        self.assertIn("exited with code 7", engine._crash_note(7))
        # A process that returned says nothing about crashing.
        self.assertEqual(engine._crash_note(0), "")

    def test_a_silent_worker_still_reports_something(self):
        self.assertEqual(engine._worker_error_detail("noise\n"), "noise")
        self.assertIn("no result", engine._worker_error_detail(""))
        # With no exception line anywhere, the tail is still the best guess.
        tail = engine._worker_error_detail(
            "one\ntwo\nthree\nfour\n"
        )
        self.assertIn("four", tail)
        self.assertNotIn("one", tail)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_the_worker_asks_the_model_for_one_description_of_the_frames(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_worker_")
        frames = [os.path.join(root, "frame_0001.jpg")]
        for path in frames:
            with open(path, "wb") as handle:
                handle.write(_ONE_PIXEL_JPEG)
        processor = _FakeProcessor(embedded=True)
        text = worker._describe_chunk(
            processor, _FakeModel(), _FakeTorch(),
            {"guideline_prompt": "Describe only what is visible.",
             "chunk_instruction": "One sentence."},
            frames,
        )
        self.assertEqual(text, "A woman opens a door.")
        self.assertEqual(len(processor.embedded_attempts), 1)
        self.assertEqual(len(processor.embedded_calls), 1)
        self.assertEqual(processor.placeholder_calls, [])
        message = processor.embedded_calls[0][1]["content"]
        self.assertEqual(message[0]["type"], "image")
        self.assertIn("image", message[0])
        self.assertEqual(message[-1]["text"], "One sentence.")

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_the_worker_hands_the_frames_beside_the_prompt_when_it_must(self):
        """Gemma 3's template takes the frames beside the prompt, not inside it."""
        root = tempfile.mkdtemp(prefix="aivs_ad_worker_")
        frames = [os.path.join(root, "frame_0001.jpg")]
        for path in frames:
            with open(path, "wb") as handle:
                handle.write(_ONE_PIXEL_JPEG)
        processor = _FakeProcessor(embedded=False)
        text = worker._describe_chunk(
            processor, _FakeModel(), _FakeTorch(),
            {"guideline_prompt": "Be objective.", "chunk_instruction": "One sentence."},
            frames,
        )
        self.assertEqual(text, "A woman opens a door.")
        # The embedded form was tried first and refused; the fallback worked.
        self.assertEqual(len(processor.embedded_attempts), 1)
        self.assertEqual(processor.embedded_calls, [])
        self.assertEqual(len(processor.placeholder_calls), 1)
        content = processor.placeholder_calls[0][1]["content"]
        # The prompt carries a placeholder per frame, not the frame itself...
        self.assertEqual(content[0], {"type": "image"})
        # ...and the frames go to the processor with the rendered prompt.
        self.assertEqual(len(processor.text_calls[0]["images"]), 1)
        self.assertIn("placeholder", processor.text_calls[0]["prompt"])

    def test_the_worker_script_is_bundled(self):
        # The worker runs in the model's own environment, so it must exist as a
        # real file an interpreter can be pointed at.
        path = engine.environments.bundled_worker_path()
        self.assertTrue(path and os.path.isfile(path))
        self.assertTrue(callable(worker.run))
        self.assertEqual(engine.environments.worker_script(), path)

    def test_the_environment_folder_is_its_own_family(self):
        # A describing model must never share packages with a voice or a
        # transcription model.
        folder = engine.env_dir("smolvlm2")
        self.assertIn("describe_envs", folder)
        self.assertNotIn("stt_envs", folder)
        self.assertNotIn("tts_envs", folder)

    def test_device_resolution(self):
        with mock.patch.object(engine, "cuda_available", return_value=False):
            self.assertEqual(engine.resolve_device("auto"), "cpu")
            self.assertEqual(engine.resolve_device("cuda"), "cuda")
            self.assertEqual(engine.resolve_device("cpu"), "cpu")
        with mock.patch.object(engine, "cuda_available", return_value=True):
            self.assertEqual(engine.resolve_device("auto"), "cuda")


class _FitProcessor:
    """A processor that records how many frames each prompt was handed."""

    def __init__(self):
        self.image_counts: list = []

    def apply_chat_template(self, messages, **_kwargs):
        content = messages[-1]["content"]
        self.image_counts.append(
            sum(1 for item in content if item.get("type") == "image")
        )
        return {"input_ids": np.zeros((1, 3), dtype=np.int64)}

    def batch_decode(self, _tokens, skip_special_tokens=True):
        return ["A lecturer points at a slide."]


class _FitModel:
    """A model that only holds ``limit`` frames in one prompt."""

    class _Param:
        device = "cpu"

    def __init__(self, processor, limit, failure=None):
        self.processor = processor
        self.limit = limit
        self.failure = failure

    def parameters(self):
        return iter([self._Param()])

    def generate(self, **_kwargs):
        if self.failure is not None:
            raise self.failure("something else went wrong")
        count = self.processor.image_counts[-1]
        if count > self.limit:
            raise _FitTorch.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 1.01 GiB."
            )
        return np.zeros((1, 10), dtype=np.int64)


class _FitTorch:
    """Enough of ``torch`` for the describer: the OOM type and inference mode."""

    class OutOfMemoryError(RuntimeError):
        pass

    class _Cuda:
        @staticmethod
        def empty_cache():
            return None

    cuda = _Cuda

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class WorkerFrameBudgetTest(unittest.TestCase):
    """A chunk must never be sent to the model in a prompt it cannot hold.

    Regression: every frame a chunk sampled went into one prompt with no memory
    limit at all, so a 4B model on a 12 GB card was pushed 3.6 GB past the card -
    where the driver spills to system memory and an answer that takes five
    seconds took three minutes, measured on the frames of a single chunk.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aivs_ad_budget_")
        self.addCleanup(shutil.rmtree, self.root, True)

    def _frames(self, count):
        paths = []
        for index in range(count):
            path = os.path.join(self.root, f"frame_{index:04d}.jpg")
            with open(path, "wb") as handle:
                handle.write(_ONE_PIXEL_JPEG)
            paths.append(path)
        return paths

    def test_frames_are_thinned_to_what_one_prompt_can_hold(self):
        frames = [str(index) for index in range(60)]
        kept = worker.thin_frames(frames, 2304, worker.MAX_PROMPT_PATCHES)
        # Three 1024-pixel frames is what the budget buys, and they are spread
        # over the window instead of bunched at its start.  Four frames plus a
        # window's longer prompt was measured to spill the card and take seventy
        # seconds a window instead of ten, so the budget is the smaller one.
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0], frames[0])
        self.assertEqual(kept[-1], frames[-1])

    def test_a_prompt_that_fits_is_left_alone(self):
        frames = self._frames(3)
        self.assertEqual(
            worker.thin_frames(frames, 2304, worker.MAX_PROMPT_PATCHES), frames)
        self.assertEqual(worker.thin_frames(frames, 2304, 9216), frames)

    def test_thinning_keeps_at_least_one_frame(self):
        frames = self._frames(12)
        for budget in (0, 1, 100):
            kept = worker.thin_frames(frames, 2304, budget)
            self.assertEqual(len(kept), 1, budget)
            self.assertIn(kept[0], frames)

    def test_the_spread_is_even_and_includes_the_ends(self):
        frames = self._frames(10)
        kept = worker.thin_frames(frames, 1, 4)
        self.assertEqual(len(kept), 4)
        self.assertEqual(kept[0], frames[0])
        self.assertEqual(kept[-1], frames[-1])

    def test_a_frame_that_cannot_be_measured_still_costs_one_patch(self):
        # A zero cost would allow an unbounded prompt, so a missing size is
        # treated as no thinning rather than as free frames.
        self.assertEqual(worker.patches_per_frame("/no/such/frame.jpg"), 0)
        if _Image is not None:
            self.assertGreaterEqual(worker.patches_per_frame(self._frames(1)[0]), 1)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_patches_match_the_frame_area(self):
        path = os.path.join(self.root, "big.jpg")
        _Image.new("RGB", (1024, 576)).save(path)
        # A 1024 x 576 frame in 16-pixel patches: what a 1080p video sends.
        self.assertEqual(worker.patches_per_frame(path), 2304)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_prompt_too_big_for_the_gpu_is_retried_with_fewer_frames(self):
        processor = _FitProcessor()
        notes: list = []
        text = worker._describe_chunk(
            processor, _FitModel(processor, limit=2), _FitTorch(),
            {"guideline_prompt": "Be objective.",
             "chunk_instruction": "One sentence."},
            self._frames(8),
            on_note=notes.append,
        )
        self.assertEqual(text, "A lecturer points at a slide.")
        # 8 frames did not fit, 4 did not either, 2 did - and the user is told.
        self.assertEqual(processor.image_counts, [8, 4, 2])
        self.assertEqual(len(notes), 2)
        self.assertIn("2 frame(s)", notes[-1])

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_failure_that_is_not_memory_is_not_retried(self):
        processor = _FitProcessor()
        with self.assertRaises(ValueError):
            worker._describe_chunk(
                processor,
                _FitModel(processor, limit=99, failure=ValueError),
                _FitTorch(),
                {"guideline_prompt": "", "chunk_instruction": "One sentence."},
                self._frames(4),
            )
        self.assertEqual(len(processor.image_counts), 1)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_chunk_that_cannot_fit_at_all_gives_up_instead_of_looping(self):
        processor = _FitProcessor()
        with self.assertRaises(_FitTorch.OutOfMemoryError):
            worker._describe_chunk(
                processor, _FitModel(processor, limit=0), _FitTorch(),
                {"guideline_prompt": "", "chunk_instruction": "One sentence."},
                self._frames(2),
            )
        # Down to the last frame, and then it raises rather than looping.
        self.assertEqual(processor.image_counts, [2, 1])


class PictureChangeTest(unittest.TestCase):
    """A chunk whose picture has barely changed must cost nothing to describe.

    Regression: every chunk was sent to the model whether or not anything in it
    had moved, so the same scene was described again and again - and in extended
    audio description each of those repeats also stopped the film and spoke the
    repeat out loud, lengthening the described film for nothing.  The numbers in
    these tests are the ones measured on a real lecture: a held picture 0.00 to
    0.07 grey levels, a speaker shifting with no slide change 1.15 to 1.47, a
    slide advanced 3.12 to 7.53, a cut to a new still 7.84.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aivs_ad_change_")
        self.addCleanup(shutil.rmtree, self.root, True)

    @staticmethod
    def _signature(level: int) -> bytes:
        width, height = worker.SIGNATURE_SIZE
        return bytes([level]) * (width * height)

    def _touched(self, level: int, cells: int, moved: int) -> bytes:
        """A picture like ``level`` with ``cells`` of its cells moved."""
        signature = bytearray(self._signature(level))
        for index in range(cells):
            signature[index] = min(255, level + moved)
        return bytes(signature)

    def test_a_picture_held_still_is_not_a_change(self):
        still = self._signature(128)
        self.assertEqual(worker.picture_change(still, still), 0.0)
        self.assertFalse(worker.picture_changed([still, still], still))

    def test_encoder_noise_on_a_still_picture_is_not_a_change(self):
        # A held frame replays with a grey level of noise either way in every
        # cell once it has been through JPEG and H.264; that is not movement.
        noisy = bytearray(self._signature(128))
        for index in range(len(noisy)):
            noisy[index] += 1 if index % 2 else -1
        noise = bytes(noisy)
        self.assertEqual(worker.picture_change(noise, self._signature(128)), 1.0)
        self.assertFalse(worker.picture_changed([noise, noise],
                                                self._signature(128)))

    def test_a_cut_to_a_new_still_picture_is_a_change(self):
        # Nothing moves inside the chunk, but the chunk is not the picture the
        # listener was last told about, so it has to be described.
        was, now = self._signature(40), self._signature(200)
        self.assertEqual(worker.picture_change(was, now), 160.0)
        self.assertTrue(worker.picture_changed([now, now], was))
        self.assertTrue(worker.picture_changed([was, now], None))

    def test_a_speaker_shifting_is_still_the_same_scene(self):
        # 30 of the 2304 cells moving hard is 1.30 grey levels over the frame:
        # the measured band of a lecture chunk with no slide change.
        shifting = self._touched(128, 30, moved=100)
        self.assertFalse(worker.picture_changed([shifting, shifting],
                                                self._signature(128)))

    def test_a_slide_advancing_is_a_change(self):
        # 75 cells moving hard is 3.26 grey levels: the measured band of a
        # lecture chunk where a slide advanced (3.12 to 7.53).
        advanced = self._touched(128, 75, moved=100)
        self.assertTrue(worker.picture_changed([advanced, advanced],
                                               self._signature(128)))

    def test_a_handful_of_changed_cells_is_too_little(self):
        nearly = self._touched(128, 5, moved=100)
        self.assertFalse(worker.picture_changed([nearly], self._signature(128)))

    def test_a_chunk_with_nothing_to_compare_it_with_is_described(self):
        # A single frame cannot be compared with itself, and no frames at all is
        # no evidence: both are described rather than guessed away.
        still = self._signature(128)
        self.assertTrue(worker.picture_changed([], None))
        self.assertTrue(worker.picture_changed([still], None))

    def test_a_frame_that_cannot_be_read_is_described(self):
        still = self._signature(128)
        self.assertEqual(worker.picture_change(None, still), 255.0)
        self.assertEqual(worker.picture_change(b"short", still), 255.0)
        self.assertTrue(worker.picture_changed([None], still))

    def test_the_threshold_is_how_much_change_counts(self):
        still = self._signature(128)
        shifting = self._touched(128, 30, moved=100)
        self.assertTrue(worker.picture_changed([shifting, shifting], still,
                                               threshold=1.0))
        self.assertFalse(worker.picture_changed([shifting, shifting], still,
                                                threshold=5.0))
        # Zero is the knob's lower end: nothing counts as a repeat, so every
        # chunk is described, which is how a caller turns the idea off.
        self.assertTrue(worker.picture_changed([still, still], still,
                                              threshold=0.0))

    def test_the_request_can_set_the_threshold(self):
        default = worker.MIN_PICTURE_CHANGE
        self.assertEqual(worker._min_picture_change({}), default)
        self.assertEqual(worker._min_picture_change({"min_picture_change": 0}),
                         0.0)
        self.assertEqual(
            worker._min_picture_change({"min_picture_change": 7.5}), 7.5)
        self.assertEqual(
            worker._min_picture_change({"min_picture_change": "no"}), default)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_signature_is_read_from_the_frame_itself(self):
        width, height = worker.SIGNATURE_SIZE
        grey = os.path.join(self.root, "grey.jpg")
        _Image.new("L", (1024, 576), 90).save(grey, quality=90)
        darker = os.path.join(self.root, "darker.jpg")
        _Image.new("L", (1024, 576), 20).save(darker, quality=90)
        signature = worker.frame_signature(grey)
        self.assertEqual(len(signature), width * height)
        self.assertGreater(
            worker.picture_change(signature, worker.frame_signature(darker)),
            50.0)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_grey_level_apart_is_the_same_picture(self):
        grey = os.path.join(self.root, "a.jpg")
        _Image.new("L", (1024, 576), 90).save(grey, quality=90)
        close = os.path.join(self.root, "b.jpg")
        _Image.new("L", (1024, 576), 91).save(close, quality=90)
        first = worker.frame_signature(grey)
        second = worker.frame_signature(close)
        self.assertFalse(worker.picture_changed([second], first))

    def test_a_frame_that_is_not_there_has_no_signature(self):
        self.assertIsNone(
            worker.frame_signature(os.path.join(self.root, "gone.jpg")))


@unittest.skipIf(_Image is None, "Pillow is not installed")
class WorkerRepeatTest(unittest.TestCase):
    """The worker must not spend a description - or a pause - on a repeat."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aivs_ad_repeat_")
        self.addCleanup(shutil.rmtree, self.root, True)

    def _frame(self, level: int, name: str) -> str:
        path = os.path.join(self.root, name)
        _Image.new("L", (640, 360), level).save(path, quality=90)
        return path

    def _describe(self, chunks: int, per_chunk: dict, **extra):
        """Run the worker over ``chunks`` chunks with the model faked out.

        ``per_chunk`` maps a chunk's start time to the frames it samples, so a
        test says exactly what the picture does without any video at all, and
        ``extra`` adds to the request (``driver``, for instance).
        """
        plan = [{"index": index, "start": index * 20.0, "end": (index + 1) * 20.0}
                for index in range(chunks)]
        asked: list = []

        def describe(_processor, _model, _torch, _request, images, **_kwargs):
            asked.append(list(images))
            return "A lecturer points at a slide."

        def extract(_ffmpeg, _video, start, _end, *_rest, **_kwargs):
            return list(per_chunk[float(start)])

        response = os.path.join(self.root, "response.json")
        request = {
            "video": self._frame(0, "video.jpg"),  # run() only checks it is there
            "chunks": plan, "fps": 0.5, "out": response,
        }
        request.update(extra)
        with mock.patch.object(worker, "_load_model",
                               return_value=(None, None, None)), \
                mock.patch.object(worker, "_extract_frames", side_effect=extract), \
                mock.patch.object(worker, "_describe_chunk", side_effect=describe):
            result = worker.run(request, response)
        with open(response, encoding="utf-8") as handle:
            saved = json.load(handle)
        return result, saved, asked

    def test_a_chunk_that_repeats_the_last_description_is_not_described(self):
        black, white = self._frame(0, "black.jpg"), self._frame(255, "white.jpg")
        result, saved, asked = self._describe(3, {
            0.0: [black, black],   # the opening: always described
            20.0: [black, black],  # the same picture again
            40.0: [black, white],  # a new picture inside the chunk
        })
        self.assertEqual([segment["chunk"] for segment in result["segments"]],
                         [0, 2])
        # The model was asked about the two chunks that changed and no others.
        self.assertEqual(len(asked), 2)
        self.assertEqual(saved["segments"], result["segments"])

    def test_a_one_picture_model_is_handed_the_newest_frame_of_the_window(self):
        """Moondream 2 holds one image per question, so the frame it is given is
        the window's *last*: the picture the listener has not been told about
        yet, and the frame the run has just compared against the last
        description.  The patch budget - how many frames one prompt can hold on
        the card - does not apply to it, and the spread the run keeps for a
        multi-image prompt would otherwise hand it the window's first frame.
        """
        first = self._frame(0, "first.jpg")
        middle = self._frame(128, "middle.jpg")
        last = self._frame(255, "last.jpg")
        window = [first, middle, last]
        _result, _saved, asked = self._describe(1, {0.0: window},
                                                driver="query")
        self.assertEqual(asked, [[last]])
        # A model asked through a prompt still gets the frames that fit it.
        _result, _saved, prompted = self._describe(1, {0.0: window})
        self.assertEqual(prompted, [window])

    def test_a_film_of_one_held_picture_is_described_once(self):
        black = self._frame(0, "black.jpg")
        result, _saved, asked = self._describe(3, {
            0.0: [black, black], 20.0: [black, black], 40.0: [black, black],
        })
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(len(asked), 1)

    def test_the_film_is_not_stopped_for_a_chunk_that_repeated_one(self):
        black, white = self._frame(0, "black.jpg"), self._frame(255, "white.jpg")
        result, _saved, _asked = self._describe(3, {
            0.0: [black, black], 20.0: [black, black], 40.0: [black, white],
        })
        kept = extended.plan_extended(result["segments"], 60.0)
        every = extended.plan_extended(
            [dict(segment, start=index * 20.0)
             for index, segment in enumerate(result["segments"] + [result["segments"][0]])],
            60.0,
        )
        self.assertEqual(kept["count"], 2)
        self.assertEqual(every["count"], 3)
        self.assertLess(kept["added_seconds"], every["added_seconds"])


class WorkerPipeTest(unittest.TestCase):
    """The worker's two pipes must both be read while it runs."""

    class _Runtime:
        python_exe = sys.executable

        def usable_python_exe(self):
            """The model's interpreter, as the real runtime hands it over.

            This stand-in has no ``pyvenv.cfg``, so there is nothing that
            could have to be re-pointed; the real one repairs an environment
            built by an earlier installation before returning its path.
            """
            return self.python_exe

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aivs_ad_pipe_")
        self.addCleanup(shutil.rmtree, self.root, True)

    def _run(self, script_source: str) -> dict:
        script = os.path.join(self.root, "loud_worker.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(script_source)
        result: dict = {}

        def body() -> None:
            try:
                with mock.patch.object(engine.environments, "worker_script",
                                       return_value=script), \
                        mock.patch.object(engine.environments, "runtime",
                                          return_value=self._Runtime()):
                    progress: list = []
                    streamed: list = []
                    result["segments"] = engine._run_worker(
                        "smolvlm2", {"hf_model": "x/y", "chunk_seconds": 1.0},
                        lambda message, fraction: progress.append(message),
                        streamed.append,
                    )
                    result["progress"] = progress
            except BaseException as exc:
                result["error"] = exc

        runner = threading.Thread(target=body, daemon=True)
        runner.start()
        runner.join(timeout=60)
        self.assertFalse(
            runner.is_alive(),
            "the worker run blocked: a pipe nobody read filled up",
        )
        return result

    def test_a_worker_that_floods_stderr_does_not_block(self):
        """Regression: stderr was read only after stdout had ended.

        Loading a real model writes several hundred progress bars to stderr.  On
        a 12 GB card that load filled the stderr pipe, the child blocked writing
        to it and the parent blocked reading stdout - a model load that never
        finished, with no error anywhere to explain it.
        """
        result = self._run(
            "import json, sys\n"
            "request = json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "sys.stderr.write('noise\\n' * 200000)\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write('AIVS-PROGRESS|50%|Halfway\\n')\n"
            "sys.stdout.flush()\n"
            "json.dump({'ok': True, 'segments': [{'start': 0.0, 'end': 2.0,\n"
            "            'text': 'A door opens.'}]},\n"
            "          open(request['out'], 'w', encoding='utf-8'))\n"
        )
        self.assertNotIn("error", result, result.get("error"))
        self.assertEqual(result["segments"][0]["text"], "A door opens.")
        self.assertIn("Halfway", result["progress"])

    def test_a_worker_whose_progress_bars_have_no_newlines_does_not_block(self):
        """Regression: a progress bar writes carriage returns, not lines.

        ``transformers`` reports a checkpoint load as ``\rLoading weights: 30%|...``
        - one long burst with no newline anywhere in it.  Reading stderr a *line*
        at a time holds every one of those bytes in the reader's buffer until a
        newline finally arrives, so the pipe fills: measured on this machine, one
        Qwen3-VL-4B load writes 7,639 bytes before its first newline, against an
        8 KB pipe.  The run then parked at "Loading the describer..." with the
        model already on the card and no error anywhere to explain it.
        """
        result = self._run(
            "import json, sys\n"
            "request = json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "for step in range(400):\n"
            "    sys.stderr.write('\\rLoading weights: %3d%%|%s| %d/713' % "
            "(step, '#' * 10, step))\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write('AIVS-PROGRESS|50%|Halfway\\n')\n"
            "sys.stdout.flush()\n"
            "json.dump({'ok': True, 'segments': [{'start': 0.0, 'end': 2.0,\n"
            "            'text': 'A door opens.'}]},\n"
            "          open(request['out'], 'w', encoding='utf-8'))\n"
        )
        self.assertNotIn("error", result, result.get("error"))
        self.assertEqual(result["segments"][0]["text"], "A door opens.")
        self.assertIn("Halfway", result["progress"])

    def test_a_worker_that_fails_still_reports_its_cause(self):
        # The stderr the worker wrote is still what explains a failure.
        result = self._run(
            "import json, sys\n"
            "request = json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "sys.stderr.write('RuntimeError: the model could not be loaded\\n')\n"
            "json.dump({'ok': False, 'error': 'the model could not be loaded'},\n"
            "          open(request['out'], 'w', encoding='utf-8'))\n"
        )
        self.assertIsInstance(result.get("error"), engine.DescribeEngineError)
        self.assertIn("could not be loaded", str(result["error"]))


class WeightsCacheSymlinkTest(unittest.TestCase):
    """Hugging Face's symlink cache on a Windows machine without Dev Mode.

    The cache stores one copy of each file in ``blobs`` and links every snapshot
    to it.  Creating those links needs Developer Mode, and without it a *fetch*
    of a model that is not on disk yet dies part-way through - measured here: a
    describing model that had never been downloaded produced no description, and
    the error was the library's advice rather than a cause.
    """

    def setUp(self):
        self.saved = {name: os.environ.get(name) for name in
                      ("HF_HUB_DISABLE_SYMLINKS", "HF_HUB_DISABLE_SYMLINKS_WARNING",
                       "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
        for name in self.saved:
            os.environ.pop(name, None)
        self.addCleanup(self._restore)

    def _restore(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_no_symlinks_means_hugging_face_copies_the_files(self):
        with mock.patch.object(worker, "_symlinks_available", return_value=False):
            self.assertTrue(worker.prepare_weights_cache())
        self.assertEqual(os.environ["HF_HUB_DISABLE_SYMLINKS"], "1")
        # The warning describes what this call just handled; the run would
        # otherwise report it instead of its progress.
        self.assertEqual(os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"], "1")

    def test_a_machine_that_can_link_keeps_its_links(self):
        """Copies cost a second copy of every checkpoint; 37 GB is not small."""
        with mock.patch.object(worker, "_symlinks_available", return_value=True):
            self.assertFalse(worker.prepare_weights_cache())
        self.assertNotIn("HF_HUB_DISABLE_SYMLINKS", os.environ)

    def test_an_explicit_choice_is_not_overridden(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        with mock.patch.object(worker, "_symlinks_available") as probe:
            self.assertFalse(worker.prepare_weights_cache())
        probe.assert_not_called()

    def test_the_probe_really_tries_a_symlink(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_symlink_")
        self.addCleanup(shutil.rmtree, root, True)
        # It answers, whatever this machine can do, and leaves nothing behind.
        self.assertIn(worker._symlinks_available(root), (True, False))
        self.assertEqual(os.listdir(root), [])

    def test_the_cache_directory_follows_the_environment(self):
        os.environ["HF_HUB_CACHE"] = os.path.join("C:\\", "somewhere", "hub")
        self.assertTrue(worker._hugging_face_cache_dir().endswith("hub"))
        self.assertTrue(os.path.isabs(worker._hugging_face_cache_dir()))

    def test_the_worker_prepares_the_cache_before_it_touches_a_model(self):
        """The settings are read when ``huggingface_hub`` is first imported."""
        root = tempfile.mkdtemp(prefix="aivs_ad_prepare_")
        self.addCleanup(shutil.rmtree, root, True)
        out = os.path.join(root, "out.json")
        missing = os.path.join(root, "missing.mp4")
        with mock.patch.object(worker, "prepare_weights_cache",
                               return_value=True) as prepare:
            result = worker.run({"video": missing, "chunk_seconds": 1.0}, out)
        prepare.assert_called_once()
        # It ran even though the run stopped at its first check, which is the
        # point: nothing may import huggingface_hub before this call happens.
        self.assertIn("not found", result.get("error", ""))


class WorkerGpuLoadTest(unittest.TestCase):
    """A model larger than the card must never be spilled to the *disk*.

    ``device_map="auto"`` put part of an oversized checkpoint on the disk on
    this machine; a model on the disk is a set of *meta* tensors, and the
    forward pass over them ended the process with an access violation
    (``0xC0000005``).  The loader now places everything on the one card, and
    asks for 4-bit when the checkpoint is bigger than the card.
    """

    class _Loader:
        """A stand-in ``from_pretrained`` that records what it was asked."""

        def __init__(self, failures=()):
            self.calls: list = []
            self.failures = list(failures)

        def __call__(self, model_cls, repo, dtype, trust, device_map,
                     quantization_config=None, load_kwargs=None):
            self.calls.append({
                "repo": repo, "dtype": dtype, "device_map": device_map,
                "quantization_config": quantization_config,
                "load_kwargs": dict(load_kwargs or {}),
            })
            if self.failures:
                raise self.failures.pop(0)
            return "the-model"

    def _load(self, size_gb, card_gb=12.0, quantization="sentinel",
              failures=(), request=None):
        loader = self._Loader(failures)
        request = dict(request or {"model_size_gb": size_gb})
        request.setdefault("model_size_gb", size_gb)
        with mock.patch.object(worker, "_from_pretrained",
                               side_effect=loader), \
                mock.patch.object(worker, "_gpu_total_gib",
                                  return_value=card_gb), \
                mock.patch.object(worker, "_quantized_config",
                                  return_value=(quantization, "not installed")), \
                mock.patch.object(worker, "_free_cuda"), \
                mock.patch.object(worker, "_emit"):
            model = worker._load_on_gpu(
                object(), "Org/Repo", False,
                types.SimpleNamespace(bfloat16="bf16"),
                request,
            )
        return model, loader

    def test_a_model_bigger_than_the_card_is_loaded_in_4_bit(self):
        model, loader = self._load(size_gb=24.0)
        self.assertEqual(model, "the-model")
        self.assertEqual(len(loader.calls), 1)
        self.assertEqual(loader.calls[0]["quantization_config"], "sentinel")
        # Never the automatic placement: that is what reached for the disk.
        self.assertEqual(loader.calls[0]["device_map"], {"": 0})

    def test_a_model_that_fits_is_loaded_at_full_precision(self):
        model, loader = self._load(size_gb=4.4)
        self.assertEqual(model, "the-model")
        self.assertEqual(len(loader.calls), 1)
        self.assertIsNone(loader.calls[0]["quantization_config"])
        self.assertEqual(loader.calls[0]["device_map"], {"": 0})

    def test_a_full_precision_load_that_fails_falls_back_to_4_bit(self):
        model, loader = self._load(size_gb=4.4,
                                   failures=[RuntimeError("out of memory")])
        self.assertEqual(model, "the-model")
        self.assertEqual(len(loader.calls), 2)
        self.assertIsNone(loader.calls[0]["quantization_config"])
        self.assertEqual(loader.calls[1]["quantization_config"], "sentinel")
        self.assertEqual(loader.calls[1]["device_map"], {"": 0})

    def test_a_model_that_cannot_fit_is_refused_with_its_reason(self):
        with self.assertRaises(RuntimeError) as caught:
            self._load(size_gb=24.0, quantization=None,
                       failures=[RuntimeError("no bitsandbytes")])
        message = str(caught.exception)
        self.assertIn("bitsandbytes", message)
        self.assertIn("Repair model environment", message)
        # The size and the card are both named, so the answer is actionable.
        self.assertIn("24.0 GB", message)
        self.assertIn("12.0 GB", message)

    def test_a_model_s_own_load_keywords_reach_from_pretrained(self):
        # MiniCPM-o is loaded vision-only through them; without this the load
        # asks for the whisper encoder and the TTS module it will never use.
        model, loader = self._load(
            size_gb=4.4,
            request={"model_size_gb": 4.4,
                     "load_kwargs": {"init_audio": False, "init_tts": False}},
        )
        self.assertEqual(model, "the-model")
        self.assertEqual(loader.calls[0]["load_kwargs"],
                         {"init_audio": False, "init_tts": False})

    def test_a_model_with_no_load_keywords_sends_none(self):
        _model, loader = self._load(size_gb=4.4)
        self.assertEqual(loader.calls[0]["load_kwargs"], {})

    def test_the_quantiser_is_installed_only_where_it_is_needed(self):
        for model_id in catalog.model_ids():
            self.assertIn("bitsandbytes", catalog.cuda_packages(model_id))
        environments = engine.environments
        with mock.patch.object(environments, "_has_cuda", return_value=True):
            steps = environments.install_plan("gemma3")
        packages = [name for step, _index in steps for name in step]
        self.assertIn("bitsandbytes", packages)
        with mock.patch.object(environments, "_has_cuda", return_value=False):
            steps = environments.install_plan("gemma3")
        packages = [name for step, _index in steps for name in step]
        self.assertNotIn("bitsandbytes", packages)


def _fake_auto_class(factory):
    """One stand-in ``transformers`` Auto class, built around ``factory``."""

    class _Auto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return factory(*args, **kwargs)

    return _Auto


def _fake_transformers_module(classes: dict):
    """A stand-in ``transformers`` with exactly the named Auto classes.

    ``classes`` maps an auto-class name to the callable that stands in for its
    ``from_pretrained``; a name that is absent does not exist in this version,
    which is how the worker's fallback list is exercised.
    """
    module = types.ModuleType("transformers")

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return "the-processor"

    module.AutoProcessor = _AutoProcessor
    for name, factory in classes.items():
        setattr(module, name, _fake_auto_class(factory))
    return module


def _fake_torch_module():
    """A stand-in ``torch`` for the loader's CPU path."""
    module = types.ModuleType("torch")
    module.float32 = "float32"
    module.bfloat16 = "bfloat16"
    module.cuda = types.SimpleNamespace(is_available=lambda: False,
                                        empty_cache=lambda: None)
    return module


class _LoadedModel:
    """A model object with only what the loader touches."""

    def __init__(self):
        self.moved_to: list = []
        self.evaluated = False

    def to(self, device):
        self.moved_to.append(device)
        return self

    def eval(self):
        self.evaluated = True
        return self


class WorkerModelClassTest(unittest.TestCase):
    """Which class loads a checkpoint, and how it is asked for a description.

    A repository that ships its own modelling code registers the class it
    provides in its ``config.json``, and asking for a different one is refused
    before a weight is read.  MiniCPM-o 2.6 was reported exactly that way from a
    *correct* environment - "Unrecognized configuration class <class '...
    MiniCPMOConfig'> for this kind of AutoModel:
    AutoModelForImageTextToText" - so these cover both halves: the class the
    catalogue names is asked for first, and the checkpoint's own ``chat`` is how
    the MiniCPM family is driven.
    """

    #: The real refusal, as the checkpoint produced it, for the retry to match.
    _REFUSAL = (
        "Unrecognized configuration class <class 'transformers_modules.openbmb."
        "MiniCPM-o-2_6.06849bfd36da94da1f5a88fa17e8ba63f08a4.MiniCPMOConfig'> "
        "for this kind of AutoModel: AutoModelForImageTextToText.  Model type "
        "should be one of AriaConfig, AyaVisionConfig, BlipConfig."
    )

    def _load(self, classes, request):
        with mock.patch.dict(sys.modules, {"torch": _fake_torch_module(),
                                           "transformers":
                                           _fake_transformers_module(classes)}):
            return worker._load_model(request)

    def test_the_checkpoint_s_own_class_is_asked_for_first(self):
        asked: list = []

        def _own(*_args, **_kwargs):
            asked.append("AutoModel")
            return _LoadedModel()

        def _task(*_args, **_kwargs):  # pragma: no cover - must not be reached
            asked.append("AutoModelForImageTextToText")
            return _LoadedModel()

        processor, model, _torch = self._load(
            {"AutoModel": _own, "AutoModelForImageTextToText": _task},
            {"hf_model": "openbmb/MiniCPM-o-2_6", "auto_class": "AutoModel",
             "trust_remote_code": True, "device": "cpu"},
        )
        self.assertEqual(asked, ["AutoModel"])
        self.assertEqual(processor, "the-processor")
        self.assertTrue(model.evaluated)

    def test_a_refused_class_falls_through_to_the_next_one(self):
        asked: list = []

        def _refuse(*_args, **_kwargs):
            asked.append("AutoModelForImageTextToText")
            raise ValueError(self._REFUSAL)

        def _works(*_args, **_kwargs):
            asked.append("AutoModelForVision2Seq")
            return _LoadedModel()

        _processor, model, _torch = self._load(
            {"AutoModelForImageTextToText": _refuse,
             "AutoModelForVision2Seq": _works},
            {"hf_model": "Org/Repo", "device": "cpu"},
        )
        self.assertTrue(model.evaluated)
        self.assertEqual(asked, ["AutoModelForImageTextToText",
                                 "AutoModelForVision2Seq"])

    def test_a_failure_that_is_not_a_class_mismatch_is_not_retried(self):
        """Only a refusal of the class is retried; anything else is the answer.

        An out-of-memory load, a corrupt download or a missing package would
        fail identically three times, and hiding the first failure behind the
        third would make a real cause harder to read, not easier.
        """
        asked: list = []

        def _oom(*_args, **_kwargs):
            asked.append("AutoModelForImageTextToText")
            raise RuntimeError("CUDA out of memory.  Tried to allocate 2.00 GiB")

        def _never(*_args, **_kwargs):  # pragma: no cover - must not be reached
            asked.append("AutoModelForVision2Seq")
            return _LoadedModel()

        with self.assertRaises(RuntimeError) as caught:
            self._load({"AutoModelForImageTextToText": _oom,
                        "AutoModelForVision2Seq": _never},
                       {"hf_model": "Org/Repo", "device": "cpu"})
        self.assertIn("out of memory", str(caught.exception))
        self.assertEqual(asked, ["AutoModelForImageTextToText"])

    def test_the_models_own_load_keywords_are_passed_to_from_pretrained(self):
        seen: list = []

        def _model(*_args, **kwargs):
            seen.append(kwargs)
            return _LoadedModel()

        self._load({"AutoModel": _model},
                   {"hf_model": "openbmb/MiniCPM-o-2_6", "auto_class": "AutoModel",
                    "trust_remote_code": True, "device": "cpu",
                    "load_kwargs": {"init_audio": False, "init_tts": False}})
        self.assertEqual(seen[0].get("init_audio"), False)
        self.assertEqual(seen[0].get("init_tts"), False)
        self.assertIs(seen[0].get("trust_remote_code"), True)

    def test_a_model_asked_one_picture_at_a_time_is_loaded_without_a_processor(
            self):
        """Moondream 2 has no processor, and its driver never touches one.

        ``AutoProcessor`` would answer with its tokenizer fallback, so asking for
        one would work today - but the run would then depend on that fallback
        holding for a component the driver does not use, and a checkpoint whose
        own files carry no processor should not be loaded as if it had one.  A
        model asked through a prompt still builds one, so a failure there is
        still a failure.
        """
        asked: list = []

        def _model(*_args, **_kwargs):
            return _LoadedModel()

        class _Processor:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                asked.append("processor")
                return "the-processor"

        with mock.patch.dict(sys.modules, {"torch": _fake_torch_module(),
                                           "transformers":
                                           _fake_transformers_module(
                                               {"AutoModelForCausalLM": _model})}):
            sys.modules["transformers"].AutoProcessor = _Processor
            processor, model, _torch = worker._load_model(
                {"hf_model": "vikhyatk/moondream2",
                 "auto_class": "AutoModelForCausalLM", "driver": "query",
                 "trust_remote_code": True, "device": "cpu",
                 "dtype": "auto"})
        self.assertEqual(asked, [], "a processor was asked of a model with none")
        self.assertIsNone(processor)
        self.assertTrue(model.evaluated)

    def test_the_cpu_load_is_float32_unless_the_checkpoint_says_otherwise(self):
        """The run's dtype is the default; the checkpoint's own wins when named.

        Moondream 2 is the case: its own vision code casts every image crop to
        bfloat16 and its layers are built in the model's own bfloat16, so a
        float32 CPU load dies in the first linear layer.  ``auto`` is
        ``transformers``' spelling for "whatever this checkpoint is".
        """
        seen: list = []

        def _model(*_args, **kwargs):
            seen.append(kwargs)
            return _LoadedModel()

        self._load({"AutoModelForCausalLM": _model},
                   {"hf_model": "vikhyatk/moondream2",
                    "auto_class": "AutoModelForCausalLM",
                    "trust_remote_code": True, "device": "cpu",
                    "dtype": "auto"})
        self.assertEqual(seen[0].get("dtype"), "auto")
        self.assertEqual(seen[0].get("trust_remote_code"), True)

        seen.clear()
        self._load({"AutoModelForCausalLM": _model},
                   {"hf_model": "Org/Repo", "device": "cpu"})
        self.assertEqual(seen[0].get("dtype"), "float32")

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_chat_model_is_asked_through_its_own_chat(self):
        calls: list = []

        class _ChatModel:
            def chat(self, **kwargs):
                calls.append(kwargs)
                return "A woman opens a door."

        class _Processor:
            tokenizer = "the-tokenizer"

        root = tempfile.mkdtemp(prefix="aivs_ad_chat_")
        self.addCleanup(shutil.rmtree, root, True)
        frames = [os.path.join(root, "frame_0001.jpg"),
                  os.path.join(root, "frame_0002.jpg")]
        for path in frames:
            with open(path, "wb") as handle:
                handle.write(_ONE_PIXEL_JPEG)

        text = worker._describe_chunk(
            _Processor(), _ChatModel(), _FakeTorch(),
            {"driver": "chat",
             "guideline_prompt": "Describe only what is visible.",
             "chunk_instruction": "One sentence.",
             "max_new_tokens": 96},
            frames,
        )
        self.assertEqual(text, "A woman opens a door.")
        call = calls[0]
        self.assertIs(call["tokenizer"], _Processor.tokenizer)
        self.assertEqual(call["max_new_tokens"], 96)
        # Greedy, not sampled and not three beams: the same video must describe
        # itself the same way, and a run must not pay triple for one sentence.
        self.assertIs(call["sampling"], False)
        # 4.5's chat names the same choice ``do_sample``; each version ignores
        # the other's spelling, so both are sent.
        self.assertIs(call["do_sample"], False)
        self.assertEqual(call["num_beams"], 1)
        message = call["msgs"][0]
        self.assertEqual(message["role"], "user")
        # The frames go *inside* the user turn - the shape MiniCPM's own chat
        # documents - with the guideline in the same turn, because this family
        # has no system-message channel its ``chat`` accepts.
        self.assertEqual(len(message["content"]), 3)
        for picture in message["content"][:2]:
            self.assertTrue(hasattr(picture, "size"), picture)
        self.assertIn("Describe only what is visible.", message["content"][-1])
        self.assertIn("One sentence.", message["content"][-1])

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_the_chat_driver_asks_only_what_a_signature_names(self):
        """The family's versions do not take the same keywords, so the driver
        reads the signature it was handed instead of assuming one.

        The three checkpoints' own ``chat`` methods were read for this:
        MiniCPM-o 4.5 declares ``msgs, tokenizer, processor, max_new_tokens,
        do_sample, **kwargs``, while MiniCPM-o 2.6 and MiniCPM-V 4 declare
        ``msgs, tokenizer, processor, max_new_tokens, sampling, **kwargs``.
        Each recognises only its own spelling of "do not sample" and
        intersects what it recognised with its own generation settings, so the
        other spelling is discarded inside ``chat`` - 4.5's *sampling* never
        reaches its ``generate`` - and both versions end up greedy.  A version
        that declares no ``**kwargs`` must therefore never be sent a name it
        does not name: the second stand-in here would raise ``TypeError`` if the
        driver guessed.
        """
        seen: list = []

        class _FourFive:
            """The declared names of openbmb/MiniCPM-o-4_5's ``chat``."""

            def chat(self, msgs=None, tokenizer=None, processor=None,
                     max_new_tokens=4096, do_sample=True, **kwargs):
                seen.append({"which": "4.5", "do_sample": do_sample,
                             "kwargs": kwargs, "msgs": msgs})
                return "A man waits for a bus."

        class _DeclaredNamesOnly:
            """A version that takes exactly what it names and nothing else."""

            def chat(self, msgs=None, tokenizer=None, max_new_tokens=4096,
                     do_sample=True):
                seen.append({"which": "strict", "do_sample": do_sample,
                             "msgs": msgs})
                return "A man waits for a bus."

        class _Processor:
            tokenizer = "the-tokenizer"

        root = tempfile.mkdtemp(prefix="aivs_ad_signature_")
        self.addCleanup(shutil.rmtree, root, True)
        frames = [os.path.join(root, "frame_0001.jpg"),
                  os.path.join(root, "frame_0002.jpg")]
        for path in frames:
            with open(path, "wb") as handle:
                handle.write(_ONE_PIXEL_JPEG)

        request = {"driver": "chat", "chunk_instruction": "One sentence."}
        for model in (_FourFive(), _DeclaredNamesOnly()):
            with self.subTest(model=type(model).__name__):
                text = worker._describe_chunk(_Processor(), model, _FakeTorch(),
                                              request, frames)
                self.assertEqual(text, "A man waits for a bus.")
                # Greedy under the spelling this version actually reads.
                self.assertIs(seen[-1]["do_sample"], False)
        # 4.5 absorbs the other spelling, which its own code drops again before
        # it builds the generation settings - so both are sent, and neither
        # reaches a version that would refuse it.
        self.assertIs(seen[0]["kwargs"]["sampling"], False)
        self.assertEqual(seen[0]["kwargs"]["num_beams"], 1)
        self.assertNotIn("kwargs", seen[1])

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_query_model_is_asked_one_question_about_one_picture(self):
        """Moondream 2's own API, as its code and model card document it.

        Three things about the call are properties of the checkpoint rather than
        preferences: the question is the run's guideline plus its window
        instruction, the picture is the window's *newest* frame (``query`` holds
        one image at a time - and that frame is the one the run treats as "the
        picture the listener has already been told about"), and sampling is off,
        because its own defaults (``temperature=0.5, top_p=0.3``) would answer
        differently on two runs of the same video.  No processor is needed
        either: the driver is handed ``None`` to prove it.
        """
        calls: list = []

        class _Moondream:
            def query(self, image, question, settings=None):
                calls.append({"image": image, "question": question,
                              "settings": settings})
                return {"answer": "  A woman opens a door.  "}

        root = tempfile.mkdtemp(prefix="aivs_ad_query_")
        self.addCleanup(shutil.rmtree, root, True)
        frames = []
        for index, size in enumerate(((6, 4), (9, 5)), start=1):
            path = os.path.join(root, f"frame_{index:04d}.jpg")
            _Image.new("RGB", size, (index * 60, 0, 0)).save(path)
            frames.append(path)

        text = worker._describe_chunk(
            None, _Moondream(), _FakeTorch(),
            {"driver": "query",
             "guideline_prompt": "Describe only what is visible.",
             "chunk_instruction": "One sentence.",
             "max_new_tokens": 96},
            frames,
        )
        self.assertEqual(text, "A woman opens a door.")
        self.assertEqual(len(calls), 1,
                         "one question per window, not one per frame")
        call = calls[0]
        self.assertEqual(call["image"].size, (9, 5),
                         "the newest frame of the window is the one asked about")
        self.assertIn("Describe only what is visible.", call["question"])
        self.assertIn("One sentence.", call["question"])
        # Greedy - its own code takes the argmax path at temperature 0 - and
        # capped by the run's budget rather than its own 768-token default.
        self.assertEqual(call["settings"]["temperature"], 0)
        self.assertEqual(call["settings"]["max_tokens"], 96)

    @unittest.skipIf(_Image is None, "Pillow is not installed")
    def test_a_moondream_revision_without_query_is_still_asked(self):
        """The model card tells production users to pin a revision, and the
        revisions it names answer through ``answer_question`` and ``caption``
        instead of ``query``.  A revision with none of the three is refused with
        a reason rather than an ``AttributeError`` from inside the driver.
        """
        asked: list = []

        class _PinnedRevision:
            """The older shape: a question skill that takes no settings."""

            def query(self, image, question):
                asked.append(("query", question))
                return "A bus arrives."

        class _AnswerQuestionOnly:
            def answer_question(self, image, question, tokenizer=None):
                asked.append(("answer_question", question))
                return "A bus arrives."

        class _CaptionOnly:
            def caption(self, image, length="normal"):
                asked.append(("caption", length))
                return {"caption": "  A street at night.  "}

        class _NothingAtAll:
            pass

        root = tempfile.mkdtemp(prefix="aivs_ad_query_old_")
        self.addCleanup(shutil.rmtree, root, True)
        frames = [os.path.join(root, "frame_0001.jpg")]
        with open(frames[0], "wb") as handle:
            handle.write(_ONE_PIXEL_JPEG)

        request = {"driver": "query",
                   "guideline_prompt": "Describe only what is visible.",
                   "chunk_instruction": "One sentence."}
        for model in (_PinnedRevision(), _AnswerQuestionOnly()):
            with self.subTest(model=type(model).__name__):
                text = worker._describe_chunk(None, model, _FakeTorch(), request,
                                              frames)
                self.assertEqual(text, "A bus arrives.")
                self.assertIn("Describe only what is visible.", asked[-1][1])
                self.assertIn("One sentence.", asked[-1][1])
        self.assertEqual([what for what, _ in asked],
                         ["query", "answer_question"])
        self.assertEqual(asked[1][1], asked[0][1],
                         "one spelling of the skill, one question")
        # A caption has no question to carry the guideline, so it is the last
        # resort: reached only by a revision with no question skill at all.
        text = worker._describe_chunk(None, _CaptionOnly(), _FakeTorch(), request,
                                      frames)
        self.assertEqual(text, "A street at night.")
        self.assertEqual(asked[2], ("caption", "long"))
        with self.assertRaises(RuntimeError) as caught:
            worker._describe_chunk(None, _NothingAtAll(), _FakeTorch(), request,
                                   frames)
        self.assertIn("query()", str(caught.exception))

    def test_the_generate_path_is_untouched_by_the_chat_driver(self):
        # A model without the driver still takes the processor-plus-generate
        # path, which is what every other checkpoint in the catalogue uses.
        self.assertNotEqual(catalog.driver("qwen3_vl"), worker.CHAT_DRIVER)
        self.assertNotEqual(catalog.driver("qwen3_vl"), worker.QUERY_DRIVER)

    def test_the_request_carries_how_a_model_is_loaded_and_asked(self):
        seen: dict = {}

        def _fake_worker(model_id, request, on_progress=None, on_segment=None):
            seen.update(request)
            return []

        plan = {"video": "clip.mp4", "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS),
                "window_seconds": 20.0}
        wanted = {
            "minicpm_o": {"auto_class": "AutoModel", "driver": "chat",
                          "dtype": "",
                          "load_kwargs": {"init_audio": False, "init_tts": False}},
            "moondream2": {"auto_class": "AutoModelForCausalLM",
                           "driver": "query", "dtype": "auto",
                           "load_kwargs": {}},
        }
        for model_id, expected in wanted.items():
            seen.clear()
            with self.subTest(model=model_id), \
                    mock.patch.object(engine, "plan_for_video",
                                      return_value=plan), \
                    mock.patch.object(engine.environments, "is_created",
                                      return_value=True), \
                    mock.patch.object(engine.environments, "is_installed",
                                      return_value=True), \
                    mock.patch.object(engine, "_run_worker",
                                      side_effect=_fake_worker):
                engine.describe_video("clip.mp4", model_id=model_id)
            for key, value in expected.items():
                self.assertEqual(seen[key], value, key)
            self.assertIs(seen["trust_remote_code"], True)


class VariantWeightsTest(unittest.TestCase):
    """The checkpoint layer: is this variant here, fetch it, delete it.

    A model's environment is shared by every variant of it, but the weights of a
    variant are its own files in the Hugging Face cache - which is why a user who
    installed one variant could not tell whether another was on disk, and had no
    way to fetch it except by starting a description.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aivs_ad_weights_")
        self.addCleanup(shutil.rmtree, self.root, True)
        cache = os.path.join(self.root, "hub")
        os.makedirs(cache)
        patch = mock.patch.dict(os.environ,
                                {"HUGGINGFACE_HUB_CACHE": cache})
        patch.start()
        self.addCleanup(patch.stop)
        self.cache = cache

    def _download(self, repo: str, payload: bytes = b"weights") -> str:
        """Fabricate the layout ``snapshot_download`` writes."""
        folder = weights.path_for(repo)
        blobs = os.path.join(folder, "blobs")
        os.makedirs(blobs, exist_ok=True)
        with open(os.path.join(blobs, "abc"), "wb") as fh:
            fh.write(payload)
        snapshot = os.path.join(folder, "snapshots", "main")
        os.makedirs(snapshot, exist_ok=True)
        with open(os.path.join(snapshot, "config.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        return folder

    def test_the_cache_is_the_one_hugging_face_uses(self):
        # HF_HOME is what the model environments resolve when it is set, and the
        # default is the user's own cache; both are read the same way here.
        self.assertEqual(weights.cache_dir(), self.cache)
        # HF_HOME is what the environment falls back to when the cache variable
        # is not set at all.
        with mock.patch.dict(os.environ, {"HF_HOME": self.root}, clear=True):
            self.assertEqual(weights.cache_dir(),
                             os.path.join(self.root, "hub"))
        self.assertEqual(weights.repo_folder("Qwen/Qwen3-VL-4B-Instruct"),
                         "models--Qwen--Qwen3-VL-4B-Instruct")

    def test_a_variant_is_downloaded_only_when_a_snapshot_is_populated(self):
        repo = "Qwen/Qwen3-VL-4B-Instruct"
        self.assertFalse(weights.is_downloaded(repo))
        self.assertEqual(weights.size_bytes(repo), 0)
        # A folder without a snapshot is a half-finished download, not a model.
        os.makedirs(weights.path_for(repo), exist_ok=True)
        self.assertFalse(weights.is_downloaded(repo))
        os.makedirs(os.path.join(weights.path_for(repo), "snapshots", "main"))
        self.assertFalse(weights.is_downloaded(repo))
        self._download(repo)
        self.assertTrue(weights.is_downloaded(repo))
        self.assertEqual(weights.size_bytes(repo), len(b"weights"))

    def test_a_download_that_stopped_part_way_is_not_a_downloaded_variant(self):
        """Regression: a cancelled fetch counted as "Downloaded".

        Hugging Face populates the snapshot folder as the files arrive and
        streams each one through ``blobs/<sha>.incomplete``.  Reading only the
        snapshot made a download cancelled at 20% look finished, which disabled
        the button that would have resumed it - the state a user hit with a
        second variant of a model whose first variant had arrived.
        """
        repo = "Qwen/Qwen3-VL-4B-Instruct"
        folder = self._download(repo)
        self.assertTrue(weights.is_downloaded(repo))
        # A file still arriving is the evidence of a fetch that stopped.
        with open(os.path.join(folder, "blobs", "def.incomplete"), "wb") as fh:
            fh.write(b"half")
        self.assertEqual(weights.incomplete_files(repo), 1)
        self.assertTrue(weights.is_partly_downloaded(repo))
        self.assertFalse(weights.is_downloaded(repo))
        # Finishing it clears the marker and the checkpoint is usable again.
        os.remove(os.path.join(folder, "blobs", "def.incomplete"))
        self.assertFalse(weights.is_partly_downloaded(repo))
        self.assertTrue(weights.is_downloaded(repo))
        # A repository nobody touched is neither downloaded nor part-way.
        self.assertFalse(weights.is_partly_downloaded("Qwen/never-here"))
        self.assertEqual(weights.incomplete_files("Qwen/never-here"), 0)
        self.assertFalse(weights.is_partly_downloaded(""))

    def test_a_partly_downloaded_variant_says_so_and_stays_downloadable(self):
        variant = catalog.default_variant("qwen3_vl")
        repo = engine.variant_repo("qwen3_vl", variant)
        folder = self._download(repo)
        with open(os.path.join(folder, "blobs", "def.incomplete"), "wb") as fh:
            fh.write(b"half")
        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True):
            self.assertFalse(engine.variant_downloaded("qwen3_vl", variant))
            self.assertTrue(engine.variant_partial("qwen3_vl", variant))
            label = engine.variant_state_label("qwen3_vl", variant)
        self.assertIn("Partly downloaded", label)
        self.assertIn("continue where it stopped", label)
        # And a fetch that stopped is not a checkpoint to be removed either - it
        # is one to be finished, which is what the panel's download offers.
        self.assertNotIn("Downloaded", label)

    def test_removing_a_variant_deletes_only_its_own_folder(self):
        first, second = "Qwen/Qwen3-VL-4B", "Qwen/Qwen3-VL-8B"
        self._download(first)
        self._download(second)
        self.assertIsNone(weights.remove(first))
        self.assertFalse(weights.is_downloaded(first))
        self.assertTrue(weights.is_downloaded(second))
        # Nothing there is not a failure: it is already the wanted state.
        self.assertIsNone(weights.remove("Qwen/never-downloaded"))

    def test_the_engine_states_both_layers(self):
        variant = catalog.default_variant("qwen3_vl")
        repo = engine.variant_repo("qwen3_vl", variant)
        with mock.patch.object(engine.environments, "is_installed",
                               return_value=False):
            self.assertEqual(engine.variant_state_label("qwen3_vl", variant),
                             "Install the model first")
        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True):
            self.assertIn("Not downloaded",
                          engine.variant_state_label("qwen3_vl", variant))
            self._download(repo)
            self.assertTrue(engine.variant_downloaded("qwen3_vl", variant))
            self.assertEqual(engine.variant_state_label("qwen3_vl", variant),
                             "Downloaded")
            self.assertIsNone(engine.remove_variant("qwen3_vl", variant))
            self.assertFalse(engine.variant_downloaded("qwen3_vl", variant))

    def test_a_variant_without_a_repository_says_so(self):
        variant = catalog.default_variant("qwen3_vl")
        with mock.patch.object(catalog, "hf_model", return_value=""):
            state = engine.variant_state_label("qwen3_vl", variant)
            result = engine.download_variant("qwen3_vl", variant)
        self.assertEqual(state, "No download for this choice")
        self.assertFalse(result["ok"])
        self.assertIn("no repository", result["error"])

    def test_downloading_needs_the_environment_first(self):
        variant = catalog.default_variant("qwen3_vl")
        with mock.patch.object(engine.environments, "is_installed",
                               return_value=False):
            result = engine.download_variant("qwen3_vl", variant)
        self.assertFalse(result["ok"])
        self.assertIn("Install", result["error"])

    def test_downloading_a_variant_runs_the_worker_in_download_mode(self):
        variant = catalog.default_variant("qwen3_vl")
        seen: dict = {}

        def _fake(model_id, request, on_progress=None, on_segment=None,
                  cancel_event=None):
            seen.update(request)
            return {"ok": True, "path": self.cache, "note": "ready"}, []

        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(engine, "_worker_response", side_effect=_fake):
            result = engine.download_variant("qwen3_vl", variant)
        self.assertTrue(result["ok"])
        self.assertEqual(seen["mode"], "download")
        self.assertEqual(seen["hf_model"], engine.variant_repo("qwen3_vl", variant))
        self.assertEqual(seen["variant"], variant)

    def test_the_worker_fetches_the_repository_it_is_asked_for(self):
        calls: list = []

        class _Hub:
            @staticmethod
            def snapshot_download(**kwargs):
                calls.append(kwargs)
                return "C:/cache/models--Org--Repo"

        fake = types.ModuleType("huggingface_hub")
        fake.snapshot_download = _Hub.snapshot_download
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            result = worker.run({"mode": "download",
                                 "hf_model": "Org/Repo"}, "")
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(calls and calls[0].get("repo_id"), "Org/Repo")
        self.assertEqual(result["path"], "C:/cache/models--Org--Repo")
        self.assertIn("Org/Repo", result["note"])

    def test_a_download_without_a_repository_is_refused(self):
        result = worker.run({"mode": "download"}, "")
        self.assertFalse(result["ok"])
        self.assertIn("no repository", result["error"])

    def test_a_refused_or_failed_download_reports_the_cause(self):
        fake = types.ModuleType("huggingface_hub")

        def _boom(**_kwargs):
            raise RuntimeError("401 Client Error: gated repo")

        fake.snapshot_download = _boom
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            result = worker.run({"mode": "download", "hf_model": "Org/Gated"}, "")
        self.assertFalse(result["ok"])
        self.assertIn("gated repo", result["error"])
        # The advice names what actually unblocks a gated model.
        self.assertIn("licence", result["error"])

    def test_download_mode_does_not_need_a_video(self):
        # The mode is checked before the describing path, so a fetch is not
        # refused for want of a video file.
        result = worker.run({"mode": "download", "hf_model": ""}, "")
        self.assertFalse(result["ok"])
        self.assertNotIn("Video file not found", str(result.get("error")))


class ChunkReportTest(unittest.TestCase):
    """Every finished chunk is reported while the run is still going."""

    def _job(self, root, plan, segments, seen):
        source = os.path.join(root, "clip.mp4")
        open(source, "w").close()
        pdir = os.path.join(root, "project")
        project.create_project(
            pdir, "Test", source, "chunked",
            [{"index": 1, "title": "clip.mp4", "text": "", "source": source}],
            project_type=PROJECT_TYPE_AD,
            describe_settings={"model": "smolvlm2", "variant": "2.2b",
                               "sources": [source]},
        )

        def _call(video, **kwargs):
            for segment in segments:
                kwargs["on_segment"](segment)
            return list(segments)

        options = {"model": "smolvlm2", "variant": "2.2b",
                   "output_format": AD_FORMAT_SCRIPT}
        with mock.patch.object(job.engine, "plan_for_video", return_value=plan), \
                mock.patch.object(job.engine, "describe_video", side_effect=_call):
            run = job.DescribeJob(pdir, [source], options,
                                  on_chunk=lambda *args: seen.append(args))
            run.run()
        return run

    def test_each_described_chunk_is_reported_with_its_counts(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_report_")
        self.addCleanup(shutil.rmtree, root, True)
        segments = [
            {"start": 0, "end": 120, "text": "A woman opens a door.", "chunk": 0},
            {"start": 120, "end": 240, "text": "She walks in.", "chunk": 1},
        ]
        plan = {"video": "x", "duration": 240.0, "ffmpeg": "ffmpeg",
                "chunks": pipeline.plan_chunks(240.0, AD_CHUNK_SECONDS)}
        seen: list = []
        self._job(root, plan, segments, seen)
        self.assertEqual([(done, total) for _seg, done, total in seen],
                         [(1, 2), (2, 2)])
        self.assertIn("A woman opens a door.", seen[0][0]["text"])
        # The count is of the chunks described, and the total is the source's
        # own chunk count - so a skipped chunk shows up as fewer descriptions
        # than chunks, which is exactly what the user needs to see.
        self.assertEqual(len(seen), 2)


class WindowPlanTest(unittest.TestCase):
    """One chunk is described window by window, not in one summary.

    Regression: a two-minute chunk was described by one call, so the model was
    shown four frames spread over two minutes and answered with a summary - on a
    real lecture, "a man in a black shirt sits in a chair beside a whiteboard" -
    and the line of code that had appeared on that whiteboard was never
    mentioned.  A window is what one description covers, and this is where the
    number is decided.
    """

    def test_a_chunk_is_tiled_by_windows_with_no_gap_and_no_overlap(self):
        chunk = {"index": 0, "start": 0.0, "end": 120.0, "duration": 120.0}
        windows = pipeline.plan_windows(chunk, 20.0)
        self.assertEqual(len(windows), 6)
        self.assertEqual([w["start"] for w in windows],
                         [0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        self.assertEqual([w["end"] for w in windows],
                         [20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
        self.assertEqual([w["index"] for w in windows], [0, 1, 2, 3, 4, 5])
        self.assertTrue(all(w["chunk"] == 0 for w in windows))

    def test_the_last_window_is_short_rather_than_long(self):
        chunk = {"index": 3, "start": 60.0, "end": 110.0}
        windows = pipeline.plan_windows(chunk, 20.0)
        self.assertEqual([w["duration"] for w in windows], [20.0, 20.0, 10.0])
        self.assertEqual(windows[-1]["end"], 110.0)

    def test_a_chunk_shorter_than_a_window_is_one_window(self):
        chunk = {"index": 0, "start": 0.0, "end": 9.0}
        windows = pipeline.plan_windows(chunk, 20.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["duration"], 9.0)

    def test_an_empty_chunk_has_no_windows(self):
        self.assertEqual(pipeline.plan_windows(
            {"index": 0, "start": 10.0, "end": 10.0}, 20.0), [])
        self.assertEqual(pipeline.plan_windows(
            {"index": 0, "start": 10.0, "end": 5.0}, 20.0), [])

    def test_a_wildly_large_window_is_still_capped(self):
        # The setting decides detail; it may not silently collapse back to one
        # summary per chunk.
        chunk = {"index": 0, "start": 0.0, "end": 180.0}
        windows = pipeline.plan_windows(chunk, 600.0)
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(w["duration"] <= 60.0 for w in windows))

    def test_every_window_of_a_whole_video_is_planned_in_order(self):
        chunks = pipeline.plan_chunks(240.0, 120.0)
        windows = pipeline.plan_windows_for(chunks, 20.0)
        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0]["start"], 0.0)
        self.assertEqual(windows[-1]["end"], 240.0)
        self.assertEqual([w["chunk"] for w in windows],
                         [0] * 6 + [1] * 6)
        # The starts are strictly increasing: the descriptions are a timeline.
        self.assertEqual([w["start"] for w in windows],
                         sorted(w["start"] for w in windows))

    def test_the_count_is_of_descriptions_and_prefers_a_planned_list(self):
        chunks = pipeline.plan_chunks(240.0, 120.0)
        self.assertEqual(pipeline.window_count(chunks, 20.0), 12)
        # A chunk that already carries its own windows is counted from them, so
        # nothing is planned twice and the worker's own plan is what is counted.
        for chunk in chunks:
            chunk["windows"] = pipeline.plan_windows(chunk, 30.0)
        self.assertEqual(pipeline.window_count(chunks, 20.0), 8)

    def test_the_estimate_reports_how_many_descriptions_are_coming(self):
        info = pipeline.estimate(600.0, model_id="smolvlm2", variant="2.2b",
                                 chunk_seconds=120.0, window_seconds=20.0)
        self.assertEqual(info["chunks"], 5)
        self.assertEqual(info["windows"], 30)
        self.assertEqual(info["descriptions"], 30)
        # A window is what one model call costs, and the estimate is per minute
        # of video, so the figure does not change with the window - what changes
        # is how many answers that minute produces.
        small = pipeline.estimate(600.0, model_id="smolvlm2", variant="2.2b",
                                  chunk_seconds=120.0, window_seconds=60.0)
        self.assertEqual(small["seconds"], info["seconds"])
        self.assertEqual(small["descriptions"], 10)

    def test_a_video_of_no_length_is_an_estimate_of_nothing(self):
        info = pipeline.estimate(0.0, model_id="smolvlm2")
        self.assertEqual(info["seconds"], 0.0)
        self.assertEqual(info["windows"], 0)
        self.assertEqual(info["descriptions"], 0)


class WindowedWorkerTest(unittest.TestCase):
    """The worker describes each window, and tells the model what came before.

    The frames are given signatures directly rather than read from real JPEGs:
    what is under test is which pictures are sent, in what order, and what the
    model is told - not what Pillow makes of a grey square.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aivs_ad_window_")
        self.addCleanup(shutil.rmtree, self.root, True)
        #: ``path -> signature``, patched in for ``worker.frame_signature``.
        self.signatures: dict = {}

    def _frame(self, level: int, name: str) -> str:
        path = os.path.join(self.root, name)
        with open(path, "wb") as handle:
            handle.write(b"x")
        width, height = worker.SIGNATURE_SIZE
        self.signatures[path] = bytes([level]) * (width * height)
        return path

    def _run(self, request: dict, per_window: dict):
        """Run the worker with the model, FFmpeg and the frames faked out."""
        asked: list = []

        def describe(_processor, _model, _torch, _request, images,
                     instruction="", **_kwargs):
            asked.append((list(images), instruction))
            return f"Description {len(asked)}."

        def extract(_ffmpeg, _video, start, _end, *_rest, **_kwargs):
            return list(per_window[round(float(start), 3)])

        response = os.path.join(self.root, "response.json")
        request = dict(request, out=response)
        with mock.patch.object(worker, "_load_model",
                               return_value=(None, None, None)), \
                mock.patch.object(worker, "_extract_frames", side_effect=extract), \
                mock.patch.object(worker, "frame_signature",
                                  side_effect=self.signatures.get), \
                mock.patch.object(worker, "_describe_chunk", side_effect=describe):
            result = worker.run(request, response)
        return result, asked

    def test_a_two_minute_chunk_is_described_one_window_at_a_time(self):
        source = self._frame(0, "video.jpg")
        chunk = {"index": 0, "start": 0.0, "end": 120.0, "duration": 120.0}
        chunk["windows"] = pipeline.plan_windows(chunk, 20.0)
        per_window = {}
        for index, window in enumerate(chunk["windows"]):
            # Each window is a different picture, so none is a repeat.
            per_window[round(window["start"], 3)] = [
                self._frame(level, f"f{index}.jpg")
                for level in (200, 20 + index * 30)
            ]
        result, asked = self._run(
            {"video": source, "chunks": [chunk], "fps": 0.5}, per_window)
        # Six model calls for six windows, not one for the chunk.
        self.assertEqual(len(asked), 6)
        segments = result["segments"]
        self.assertEqual([segment["start"] for segment in segments],
                         [window["start"] for window in chunk["windows"]])
        self.assertEqual([segment["end"] for segment in segments],
                         [window["end"] for window in chunk["windows"]])
        self.assertEqual([segment["chunk"] for segment in segments], [0] * 6)
        # The opening call has nothing before it; every later one is told what
        # was just described, so it reports the change instead of the scene.
        self.assertNotIn("previous", asked[0][1].lower())
        self.assertIn("Description 1.", asked[1][1])
        self.assertIn("what has changed", asked[1][1])
        self.assertIn("Description 5.", asked[5][1])

    def test_a_window_that_repeats_the_last_picture_is_not_described(self):
        source = self._frame(0, "video.jpg")
        black = self._frame(0, "black.jpg")
        chunk = {"index": 0, "start": 0.0, "end": 60.0, "duration": 60.0}
        chunk["windows"] = pipeline.plan_windows(chunk, 20.0)
        per_window = {
            0.0: [black, black],   # the opening: always described
            20.0: [black, black],  # the same picture again: skipped
            40.0: [black, black],
        }
        result, asked = self._run(
            {"video": source, "chunks": [chunk], "fps": 0.5}, per_window)
        self.assertEqual(len(asked), 1)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["start"], 0.0)

    def test_a_chunk_without_windows_is_still_described_as_one_piece(self):
        # A recipe written before windows existed describes the chunk itself, so
        # a request from an older install still works.
        source = self._frame(0, "video.jpg")
        black, white = self._frame(0, "black.jpg"), self._frame(255, "white.jpg")
        chunk = {"index": 0, "start": 0.0, "end": 120.0}
        result, asked = self._run(
            {"video": source, "chunks": [chunk], "fps": 0.5},
            {0.0: [black, white]})
        self.assertEqual(len(asked), 1)
        self.assertEqual(result["segments"][0]["end"], 120.0)

    def test_the_prompt_asks_for_the_change_and_carries_the_last_description(self):
        request = {"chunk_instruction": "Describe this piece."}
        # The opening window has no previous description to be told about.
        self.assertEqual(worker._window_instruction(request), "Describe this piece.")
        told = worker._window_instruction(request, "A woman opens a door.")
        self.assertIn("Describe this piece.", told)
        self.assertIn("A woman opens a door.", told)
        self.assertIn("what has changed", told)
        # Nothing said yet, and a blank previous answer, are the same thing.
        self.assertEqual(worker._window_instruction(request, "  "),
                         "Describe this piece.")


class MixedFilmTest(unittest.TestCase):
    """The standard described film: the picture copied, the sound mixed."""

    def test_the_graph_mixes_the_soundtrack_with_the_description(self):
        graph = extended.mix_graph()
        self.assertIn("[film][voice]amix=inputs=2", graph)
        # The film's own length wins: a description only ever shortens the mix.
        self.assertIn("duration=first", graph)
        # ...and neither the film nor the voice is quietly attenuated.
        self.assertIn("normalize=0", graph)
        self.assertIn("[0:a]", graph)
        self.assertIn("[1:a]", graph)

    def test_track_priority_same_leaves_the_film_track_alone(self):
        """Both tracks as they are, which is the graph runs always used."""
        graph = extended.mix_graph()
        self.assertNotIn("volume=", graph)
        self.assertIn("[film][voice]amix=inputs=2", graph)

    def test_a_higher_track_priority_brings_the_film_down_not_the_voice(self):
        graph = extended.mix_graph(0.75)
        # The film's own soundtrack is what moves...
        self.assertIn("volume=0.75", graph)
        self.assertEqual(graph.count("volume="), 1)
        # ...and the description keeps the level the narration settings gave it,
        # so "25 percent higher" is not also a quieter description.
        self.assertIn(
            "[1:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:"
            "channel_layouts=stereo[voice]", graph)
        self.assertIn("[film][voice]amix=inputs=2:duration=first:normalize=0[outa]",
                      graph)
        self.assertIn("volume=0.5", extended.mix_graph(0.5))

    def test_every_priority_is_its_own_gain(self):
        from ai_voice_studio.constants import (
            AD_TRACK_PRIORITY_25,
            AD_TRACK_PRIORITY_50,
            AD_TRACK_PRIORITY_DUCK,
            AD_TRACK_PRIORITY_SAME,
        )

        self.assertEqual(extended.track_priority_gain(AD_TRACK_PRIORITY_SAME), 1.0)
        self.assertEqual(extended.track_priority_gain(AD_TRACK_PRIORITY_25), 0.75)
        self.assertEqual(extended.track_priority_gain(AD_TRACK_PRIORITY_50), 0.5)
        # A project created before the choice existed carries no value at all,
        # and "same" is exactly what those runs did, so the fallback is honest.
        self.assertEqual(extended.track_priority_gain(None), 1.0)
        self.assertEqual(extended.track_priority_gain(""), 1.0)
        self.assertEqual(extended.track_priority_gain("nonsense"), 1.0)
        # A gain already worked out is accepted as it is, and nothing can be
        # turned up past the film's own level.
        self.assertEqual(extended.track_priority_gain(0.6), 0.6)
        self.assertEqual(extended.track_priority_gain(3.0), 1.0)
        # *Duck, don't lower* does not lower the film at all - there is no gain
        # to apply for the whole film - and every other choice never dips.
        self.assertEqual(extended.track_priority_gain(AD_TRACK_PRIORITY_DUCK), 1.0)
        self.assertEqual(extended.track_priority_duck(AD_TRACK_PRIORITY_DUCK), 0.25)
        for choice in (AD_TRACK_PRIORITY_SAME, AD_TRACK_PRIORITY_25,
                       AD_TRACK_PRIORITY_50, None, "", "nonsense"):
            with self.subTest(choice=choice):
                self.assertEqual(extended.track_priority_duck(choice), 1.0)
        self.assertEqual(extended.track_priority_duck(0.4), 0.4)
        self.assertEqual(extended.track_priority_duck(4.0), 1.0)

    def test_ducking_dips_the_film_only_where_the_voice_is_heard(self):
        """*Duck, don't lower*: the film keeps its level and is dipped under the
        description, which is the one gain that changes while the film plays."""
        graph = extended.mix_graph(duck_level=0.25, duck_turns=[(4.0, 6.5)])
        # One gain stage, on the film's own input, and it is an expression
        # evaluated as the film plays rather than a fixed gain.
        self.assertEqual(graph.count("volume="), 1)
        self.assertIn("[rawfilm]volume='", graph)
        self.assertIn(":eval=frame[film]", graph)
        self.assertIn("clip((t-4.000)/0.06,0,1)-clip((t-6.500)/0.06,0,1)", graph)
        self.assertIn("1-0.75*min(1,", graph)
        # The description's own input is untouched, and the two are still mixed
        # at their own levels.
        self.assertIn(
            "[1:a]aresample=44100,aformat=sample_fmts=fltp:sample_rates=44100:"
            "channel_layouts=stereo[voice]", graph)
        self.assertIn("[film][voice]amix=inputs=2:duration=first:normalize=0[outa]",
                      graph)
        # The choices that do not duck have no such stage.
        for other in (extended.mix_graph(), extended.mix_graph(0.75)):
            self.assertNotIn("eval=frame", other)

    def test_a_film_with_nothing_spoken_over_it_is_mixed_as_it_always_was(self):
        """No descriptions, or nothing to dip to: the graph of every earlier run."""
        plain = extended.mix_graph()
        for graph in (
            extended.mix_graph(duck_level=0.25),
            extended.mix_graph(duck_level=0.25, duck_turns=[]),
            extended.mix_graph(duck_level=1.0, duck_turns=[(1.0, 2.0)]),
            extended.mix_graph(duck_turns=[(3.0, 3.0)]),
        ):
            with self.subTest(graph=graph):
                self.assertEqual(graph, plain)
        # Only *Duck, don't lower* ever dips: two descriptions handed over with
        # any other choice produce that choice's own graph, dip and all absent.
        self.assertEqual(extended.mix_graph(0.75, "same", [(1.0, 2.0)]),
                         extended.mix_graph(0.75))
        self.assertEqual(extended.mix_graph(1.0, "higher25", [(1.0, 2.0)]), plain)

    def test_back_to_back_descriptions_are_one_dip(self):
        """The film must not flap back up between two descriptions."""
        turns = [[20.0, 21.0], [10.0, 12.0], [12.2, 14.5], [30.0, 31.0],
                 [5.0, 4.0], "nonsense", None, [float("nan"), 2.0], [1.0, 2.0]]
        self.assertEqual(extended.ducking_spans(turns),
                         [(1.0, 2.0), (10.0, 14.5), (20.0, 21.0), (30.0, 31.0)])
        # Only a gap wider than AD_DUCK_MERGE_SECONDS keeps two dips apart.
        from ai_voice_studio.constants import AD_DUCK_MERGE_SECONDS

        self.assertEqual(extended.ducking_spans([[0.0, 1.0], [1.4, 2.0]]),
                         [(0.0, 2.0)])
        self.assertGreater(0.5, AD_DUCK_MERGE_SECONDS)
        self.assertEqual(extended.ducking_spans([[0.0, 1.0], [1.5, 2.0]]),
                         [(0.0, 1.0), (1.5, 2.0)])
        # A description before the film starts is clamped, never dropped.
        self.assertEqual(extended.ducking_spans([[-5.0, 2.0]]), [(0.0, 2.0)])

    def test_the_ducking_composes_with_a_lower_film(self):
        """A caller that asked for both gets both: a quieter film and a dip."""
        graph = extended.mix_graph(0.5, 0.25, [(1.0, 2.0)])
        self.assertIn("volume='0.5*(1-0.75*min(1,", graph)

    def test_a_dip_that_would_not_fit_on_a_command_line_is_not_applied(self):
        """Hundreds of descriptions must cost a dip, never the film itself."""
        from ai_voice_studio.constants import AD_DUCK_MAX_EXPRESSION

        turns = [[index * 20.0, index * 20.0 + 11.0] for index in range(500)]
        spans = extended.ducking_spans(turns)
        self.assertEqual(len(spans), 500)
        # The ceiling, not the arithmetic, is what refuses it: with a longer
        # command line allowed the same stretches build a bigger expression.
        self.assertGreater(
            len(extended.ducking_expression(spans, 0.25, limit=10 ** 9)),
            AD_DUCK_MAX_EXPRESSION)
        self.assertEqual(extended.ducking_expression(spans, 0.25), "")
        self.assertNotIn("volume=",
                         extended.mix_graph(duck_level=0.25, duck_turns=turns))
        # ...and what the run then writes is the plain mix, so a film too long to
        # duck is still written with both of its tracks.
        graphs = [str(attempt[attempt.index("-filter_complex") + 1])
                  for attempt in extended.mix_attempts(
                      "in.mp4", "voice.wav", "out.mp4", 1.0, 0.25, turns)
                  if "[outa]" in attempt]
        self.assertEqual(graphs, [extended.mix_graph(), extended.mix_graph()])
        # An ordinary film's worth of descriptions - a couple of dozen - fits
        # with room to spare.
        self.assertNotEqual(
            extended.ducking_expression(extended.ducking_spans(turns[:30]), 0.25),
            "")

    def test_a_duck_that_cannot_be_applied_falls_back_to_both_tracks(self):
        """The dip is a refinement, never a reason to lose the film's own sound."""
        attempts = extended.mix_attempts("in.mp4", "voice.wav", "out.mp4",
                                         1.0, 0.25, [(1.0, 2.0)])
        graphs = [str(attempt[attempt.index("-filter_complex") + 1])
                  for attempt in attempts if "[outa]" in attempt]
        self.assertEqual(len(graphs), 4)
        self.assertIn("eval=frame", graphs[0])
        self.assertIn("eval=frame", graphs[1])
        # ...and what follows is the plain mix, not the description on its own:
        # the film keeps its own sound even when the dip is not applied.
        self.assertEqual(graphs[2], extended.mix_graph())
        self.assertEqual(graphs[3], extended.mix_graph())
        alone = [attempt for attempt in attempts if "[outa]" not in attempt]
        self.assertTrue(alone)
        self.assertIn("1:a:0", alone[0])
        # A run that did not choose the ducking has one mix, as it always had.
        plain = extended.mix_attempts("in.mp4", "voice.wav", "out.mp4")
        self.assertEqual(len([a for a in plain if "[outa]" in a]), 2)

    def test_the_priority_reaches_the_mix_every_attempt_makes(self):
        attempts = extended.mix_attempts("in.mp4", "voice.wav", "out.mp4", 0.5)
        mixed = [attempt for attempt in attempts if "[outa]" in attempt]
        self.assertTrue(mixed)
        for attempt in mixed:
            self.assertTrue(any("volume=0.5" in str(part) for part in attempt))
        # The attempts that map the narration alone have no film track to bring
        # down, so a video with no soundtrack is unaffected by the choice.
        alone = [attempt for attempt in attempts if "[outa]" not in attempt]
        for attempt in alone:
            self.assertFalse(any("volume=" in str(part) for part in attempt))

    def test_the_picture_is_copied_first_and_re_encoded_only_if_it_must_be(self):
        attempts = extended.mix_attempts("in.mp4", "voice.wav", "out.mp4")
        self.assertGreaterEqual(len(attempts), 2)
        self.assertEqual(attempts[0][-1], "out.mp4")
        self.assertIn("copy", attempts[0])
        self.assertIn("libx264", attempts[1])
        # Every attempt writes the film's own picture and its own soundtrack.
        for attempt in attempts:
            self.assertIn("0:v:0", attempt)

    def test_a_film_with_no_soundtrack_falls_back_to_the_description_alone(self):
        attempts = extended.mix_attempts("in.mp4", "voice.wav", "out.mp4")
        # One attempt mixes the two, another uses only the description, so a
        # screen recording with no audio at all still gets its described video.
        mixed = [a for a in attempts if "[outa]" in a]
        alone = [a for a in attempts if "[outa]" not in a]
        self.assertTrue(mixed and alone)
        self.assertIn("1:a:0", alone[0])

    def test_no_ffmpeg_is_an_error_not_a_half_written_file(self):
        with self.assertRaises(extended.ExtendedRenderError):
            extended.mix(None, "in.mp4", "voice.wav", "out.mp4")

    def test_a_missing_narration_track_is_an_error(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_mix_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "in.mp4")
        open(source, "wb").close()
        with self.assertRaises(extended.ExtendedRenderError):
            extended.mix("ffmpeg", source, os.path.join(root, "gone.wav"),
                         os.path.join(root, "out.mp4"))

    def test_the_first_attempt_that_works_is_the_one_kept(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_mix_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "in.mp4")
        voice = os.path.join(root, "voice.wav")
        open(source, "wb").close()
        open(voice, "wb").close()
        out = os.path.join(root, "out.mp4")
        runs: list = []

        def run(argv, **_kwargs):
            runs.append(argv)
            # The first way (copy + mix) fails; the second writes the file.
            if len(runs) == 1:
                return types.SimpleNamespace(returncode=1, stderr="no audio")
            open(out, "wb").close()
            return types.SimpleNamespace(returncode=0, stderr="")

        with mock.patch.object(extended.subprocess, "run", side_effect=run):
            path = extended.mix("ffmpeg", source, voice, out)
        self.assertEqual(path, out)
        self.assertEqual(len(runs), 2)
        self.assertIn("libx264", runs[1])

    def test_a_render_that_never_works_reports_ffmpegs_own_complaint(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_mix_")
        self.addCleanup(shutil.rmtree, root, True)
        source = os.path.join(root, "in.mp4")
        voice = os.path.join(root, "voice.wav")
        open(source, "wb").close()
        open(voice, "wb").close()
        out = os.path.join(root, "out.mp4")
        with mock.patch.object(
                extended.subprocess, "run",
                return_value=types.SimpleNamespace(
                    returncode=1, stderr="Invalid data found")):
            with self.assertRaises(extended.ExtendedRenderError) as caught:
                extended.mix("ffmpeg", source, voice, out)
        self.assertIn("Invalid data found", str(caught.exception))
        self.assertFalse(os.path.exists(out))


class EngineWindowPlanTest(unittest.TestCase):
    """The request the worker is handed carries the windows to describe."""

    def test_planning_a_video_attaches_the_windows_of_every_chunk(self):
        root = tempfile.mkdtemp(prefix="aivs_ad_plan_")
        self.addCleanup(shutil.rmtree, root, True)
        video = os.path.join(root, "clip.mp4")
        open(video, "wb").close()
        with mock.patch.object(engine.media, "probe_duration",
                               return_value=240.0):
            plan = engine.plan_for_video(video, 120.0, 20.0)
        self.assertEqual(plan["duration"], 240.0)
        self.assertEqual(len(plan["chunks"]), 2)
        self.assertEqual(len(plan["chunks"][0]["windows"]), 6)
        self.assertEqual(plan["windows"], 12)
        self.assertEqual(plan["window_seconds"], 20.0)
        # The windows of the second chunk continue where the first left off.
        self.assertEqual(plan["chunks"][1]["windows"][0]["start"], 120.0)
        self.assertEqual(plan["chunks"][1]["windows"][-1]["end"], 240.0)

    def test_the_default_window_is_the_catalogue_cadence(self):
        from ai_voice_studio.constants import AD_WINDOW_SECONDS

        root = tempfile.mkdtemp(prefix="aivs_ad_plan_")
        self.addCleanup(shutil.rmtree, root, True)
        video = os.path.join(root, "clip.mp4")
        open(video, "wb").close()
        with mock.patch.object(engine.media, "probe_duration",
                               return_value=60.0):
            plan = engine.plan_for_video(video)
        self.assertEqual(plan["window_seconds"], AD_WINDOW_SECONDS)
        self.assertEqual(plan["windows"], 3)


if __name__ == "__main__":
    unittest.main()
