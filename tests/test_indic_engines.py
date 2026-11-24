"""Indic Parler-TTS (AI4Bharat), the description-conditioned Voice Lab engine.

**Indic Parler-TTS** (``ai4bharat/indic-parler-tts``) is a description-
conditioned TTS: there is no voice-id argument, so a built-in voice travels as
the speaker's name *plus* the description that asks for them, and the tuning
dialog's "Voice description" replaces that description.  It is also the one
Voice Lab engine that is not a cloning engine.

What these tests hold down, in the order they would break:

* the registry the application reads (packages, import modules, the environment
  it installs into) and the worker's own copy of the module list - the worker
  runs in its own interpreter and cannot import the application, so the two are
  duplicated and must not drift;
* the request a voice builds, checked against a fake engine rather than a real
  one (no checkpoint is downloaded to run a test);
* the description that reaches the engine, and the wording a refused (gated)
  download is reported with.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import voicelab
from ai_voice_studio.voicelab import engines as voice_lab
from ai_voice_studio.voicelab import options as tuning

WORKER = os.path.join(
    os.path.dirname(os.path.abspath(voicelab.__file__)), "worker.py"
)

NEW_ENGINES = ("indic_parler",)


def _load_worker():
    """The worker module, loaded standalone (it must not import the app)."""
    spec = importlib.util.spec_from_file_location("aivs_indic_worker", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


class RegistryTest(unittest.TestCase):
    """What the application reads about the engine."""

    def test_every_new_engine_is_registered_and_reachable(self):
        from ai_voice_studio.tts import catalog

        listed = {tts["id"] for tts in catalog.get_tts_list()}
        for engine_id in NEW_ENGINES:
            with self.subTest(engine=engine_id):
                self.assertTrue(voice_lab.is_engine(engine_id))
                self.assertIn(engine_id, listed)
                info = voice_lab.engine(engine_id)
                self.assertTrue(info["note"], "a note explains the engine")
                self.assertTrue(info["license"])
                self.assertTrue(info["homepage"].startswith("https://"))
                self.assertTrue(info["size_hint"])
                self.assertTrue(voice_lab.packages(engine_id))
                self.assertTrue(tuning.specs(engine_id), "it needs a tuning dialog")

    def test_each_one_gets_its_own_python_environment(self):
        """The point of the per-engine environments: nothing is shared."""
        from ai_voice_studio import python_runtime as pr

        folders = {}
        for engine_id in NEW_ENGINES:
            self.assertTrue(voice_lab.own_environment(engine_id), engine_id)
            folders[engine_id] = pr.environment_id(engine_id)
        self.assertEqual(len(set(folders.values())), len(NEW_ENGINES))
        # The OmniVoice engines already in the application are *not* touched:
        # the pair that shares one environment still shares it.
        self.assertEqual(pr.environment_id("omnivoice_server"),
                         pr.environment_id("omnivoice"))

    def test_the_gated_engine_is_named_in_the_hugging_face_registry(self):
        """Its weights are gated, so the token panel has to list the repo."""
        from ai_voice_studio import hf_access

        listed = {row["repo"] for row in hf_access.gated_repos()}
        self.assertEqual(voice_lab.gated_repo("indic_parler"),
                         "ai4bharat/indic-parler-tts")
        self.assertIn(voice_lab.gated_repo("indic_parler"), listed)

    def test_the_parler_voices_are_the_model_cards_speakers(self):
        voices = voice_lab.builtin_voices("indic_parler")
        by_language = {}
        for voice in voices:
            by_language.setdefault(voice["language"], []).append(voice["voice"])
        self.assertEqual(by_language["hi"], ["Rohit", "Divya", "Aman", "Rani"])
        self.assertEqual(by_language["ta"], ["Kavitha", "Jaya"])
        # Every voice carries the description that is what actually selects it.
        for voice in voices:
            self.assertIn(voice["voice"], voice["description"])
        # The two languages the model card calls unofficial are not offered.
        self.assertNotIn("Punjabi", list(by_language))
        self.assertEqual(voice_lab.count_builtin_voices("indic_parler"),
                         sum(len(names) for _code, names in
                             voice_lab._INDIC_PARLER_VOICES))



class RequestTest(unittest.TestCase):
    """What the application sends for a voice of each new engine."""

    def test_a_parler_voice_carries_its_description(self):
        voice = voice_lab.builtin_voices("indic_parler")[0]
        request = voicelab.build_synthesize_request(
            engine_id="indic_parler", device="cuda", text="नमस्ते",
            voice_entry=voice,
        )
        self.assertEqual(request["voice"], voice["voice"])
        self.assertEqual(request["description"], voice["description"])
        self.assertEqual(request["language"], voice["language"])

    def test_a_clone_request_carries_the_reference_and_its_transcript(self):
        for engine_id in ("gptsovits",):
            with self.subTest(engine=engine_id):
                entry = {
                    "voice": voice_lab.CLONE_VOICE_ID,
                    "language": "hi",
                    "variant": voice_lab.clone_variant(engine_id),
                    "engine": engine_id,
                    "ref_audio": r"C:\voices\clone\reference.wav",
                    "ref_text": "नमस्ते, यह एक परीक्षण है।",
                }
                request = voicelab.build_synthesize_request(
                    engine_id=engine_id, device="cpu", text="Hello",
                    voice_entry=entry,
                )
                self.assertEqual(request["ref_audio"], entry["ref_audio"])
                self.assertEqual(request["ref_text"], entry["ref_text"])


class _FakeIds:
    """Stands in for the ``BatchEncoding`` a tokenizer returns."""

    def __init__(self):
        self.input_ids = "prompt-ids"
        self.attention_mask = "prompt-mask"

    def to(self, _device):
        return self


class _FakeTokenizer:
    def __init__(self, seen, name):
        self.seen = seen
        self.name = name

    def __call__(self, text, return_tensors=None):
        self.seen.append((self.name, text, return_tensors))
        return _FakeIds()


class _FakeParlerModel:
    """Captures what ``generate`` was asked for."""

    sampling_rate = 44100

    def __init__(self, seen):
        self.seen = seen

    def generate(self, **kwargs):
        self.seen.append(("generate", kwargs))
        return np.zeros(2400, dtype=np.float32)


def _parler_backend(worker, request=None):
    """An IndicParlerBackend with no checkpoint loaded (no ``parler_tts``)."""
    backend = worker.IndicParlerBackend.__new__(worker.IndicParlerBackend)
    backend.device = "cuda"
    backend.request = request or {}
    backend.report = worker.OptionReport()
    backend.notes = []
    backend.sample_rate = 44100
    return backend


class ParlerBackendTest(unittest.TestCase):
    """The description *is* the voice - so it has to travel."""

    def test_the_voice_description_is_sent_to_the_description_tokenizer(self):
        worker = _load_worker()
        backend = _parler_backend(worker)
        seen = []
        backend.tokenizer = _FakeTokenizer(seen, "prompt")
        backend.description_tokenizer = _FakeTokenizer(seen, "description")
        backend.model = _FakeParlerModel(seen)
        voice = voice_lab.builtin_voices("indic_parler")[0]
        _audio, rate, speed_handled = backend.synthesize(
            "नमस्ते", {"voice": voice["voice"], "description": voice["description"]}
        )
        # The description of the *selected voice* is what was tokenized, with
        # the description tokenizer rather than the prompt one.
        self.assertEqual(seen[0][0], "description")
        self.assertEqual(seen[0][1], voice["description"])
        self.assertEqual(seen[1][0], "prompt")
        self.assertEqual(seen[1][1], "नमस्ते")
        # The two tokenizations and the sampling rate all reach the engine.
        self.assertEqual(seen[2][0], "generate")
        self.assertEqual(seen[2][1]["input_ids"], "prompt-ids")
        self.assertEqual(seen[2][1]["prompt_input_ids"], "prompt-ids")
        self.assertEqual(rate, 44100)
        self.assertFalse(speed_handled)

    def test_the_tuning_dialog_rewrites_the_description(self):
        """Typing a description is the model's own voice-design mode."""
        worker = _load_worker()
        backend = _parler_backend(worker)
        described = backend._description({
            "description": "A named voice.",
            "options": {"description": "A calm male speaker reads slowly."},
        })
        self.assertEqual(described, "A calm male speaker reads slowly.")

    def test_a_voice_without_a_description_still_names_its_speaker(self):
        """A clone, or an entry from a release that carried no description."""
        worker = _load_worker()
        backend = _parler_backend(worker)
        described = backend._description({"voice": "Rohit"})
        self.assertIn("Rohit", described)
        # A voice that arrives as a *path* (a clone) must not be read as a name.
        from_path = backend._description({"voice": r"C:\voices\clip.wav"})
        self.assertNotIn("clip.wav", from_path)
        self.assertTrue(from_path)

    def test_the_gated_refusal_names_the_page_and_the_panel(self):
        worker = _load_worker()
        failure = worker._gated_failure(
            RuntimeError("401 Client Error: Cannot access gated repo"),
            "ai4bharat/indic-parler-tts", "Indic Parler-TTS",
        )
        self.assertIsNotNone(failure)
        message = str(failure)
        self.assertIn("https://huggingface.co/ai4bharat/indic-parler-tts", message)
        self.assertIn("Settings > Hugging Face account", message)
        self.assertIn("licence", message)

    def test_a_real_bug_is_not_dressed_up_as_a_licence(self):
        worker = _load_worker()
        self.assertIsNone(worker._gated_failure(
            RuntimeError("OSError: [Errno 28] No space left on device"),
            "ai4bharat/indic-parler-tts", "Indic Parler-TTS",
        ))

