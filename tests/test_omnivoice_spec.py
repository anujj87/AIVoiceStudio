"""Tests for the OmniVoice feature layer (spec, request builders, HTTP client).

None of these tests touch the GPU, torch or the managed venv: they cover the
pure logic added to enable OmniVoice's full feature set in both the direct
(worker) and server engines.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.request
import wave
from unittest import mock

import numpy as np

from ai_voice_studio.omnivoice import build_synthesize_request, spec
from ai_voice_studio.omnivoice_server import OmniVoiceServerManager


class CleanLanguageTest(unittest.TestCase):
    def test_auto_means_none(self):
        for value in (None, "", "auto", "AUTO", "Automatic", "  auto  ", "any"):
            self.assertIsNone(spec.clean_language(value), value)

    def test_codes_lowercased(self):
        self.assertEqual(spec.clean_language("EN"), "en")
        self.assertEqual(spec.clean_language(" zh "), "zh")
        self.assertEqual(spec.clean_language("en-US"), "en-us")


class InstructResolutionTest(unittest.TestCase):
    def test_direct_catalog_voices_map_to_design_attributes(self):
        self.assertEqual(spec.resolve_instruct("omnivoice", "female"), "female")
        self.assertEqual(spec.resolve_instruct("omnivoice", "male"), "male")
        self.assertEqual(spec.resolve_instruct("omnivoice", "auto"), "")

    def test_server_presets_map_to_official_prompts(self):
        self.assertEqual(
            spec.resolve_instruct("omnivoice_server", "alloy"),
            "female, young adult, moderate pitch, american accent",
        )
        self.assertEqual(
            spec.resolve_instruct("omnivoice_server", "echo"),
            "male, middle-aged, moderate pitch, canadian accent",
        )
        self.assertEqual(
            spec.resolve_instruct("omnivoice_server", "shimmer"),
            "female, young adult, very high pitch, american accent",
        )

    def test_server_preset_table_is_the_full_official_set(self):
        # 13 presets are shipped by omnivoice-server 0.2.5.
        self.assertEqual(len(spec.OPENAI_VOICE_PRESETS), 13)
        for name in (
            "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
            "marin", "nova", "onyx", "sage", "shimmer", "verse",
        ):
            self.assertIn(name, spec.OPENAI_VOICE_PRESETS)

    def test_explicit_instruction_wins(self):
        self.assertEqual(
            spec.resolve_instruct("omnivoice", "female", "male, whisper"),
            "male, whisper",
        )
        self.assertEqual(
            spec.resolve_instruct("omnivoice_server", "alloy", "  "),  # blank
            "female, young adult, moderate pitch, american accent",
        )

    def test_unknown_voice_is_empty(self):
        self.assertEqual(spec.resolve_instruct("omnivoice", "not_a_voice"), "")
        self.assertEqual(spec.resolve_instruct(None, None), "")


class FilterKwargsTest(unittest.TestCase):
    @staticmethod
    def _method(text: str, num_step: int = 32):
        return text

    def test_skips_unsupported_kwargs(self):
        supported, skipped = spec.filter_kwargs(
            self._method, {"text": "x", "num_step": 48, "speed": 2.0}
        )
        self.assertEqual(supported, {"text": "x", "num_step": 48})
        self.assertEqual(skipped, ["speed"])

    def test_var_kwargs_accept_everything(self):
        def flexible(text, **kwargs):
            return text

        supported, skipped = spec.filter_kwargs(
            flexible, {"text": "x", "speed": 2.0, "anything": 1}
        )
        self.assertEqual(skipped, [])
        self.assertIn("anything", supported)

    def test_uninspectable_callable_is_forwarded(self):
        # A callable we cannot introspect should assume compatibility.
        with mock.patch(
            "ai_voice_studio.omnivoice.spec.inspect.signature",
            side_effect=ValueError("no signature"),
        ):
            supported, skipped = spec.filter_kwargs(
                len, {"__length_hint__": None, "speed": 2.0}
            )
        self.assertEqual(skipped, [])
        self.assertIn("speed", supported)


class OmniDictTest(unittest.TestCase):
    def test_defaults(self):
        omni = spec.build_omni()
        self.assertEqual(omni["mode"], "auto")
        self.assertEqual(omni["instruct"], "")
        self.assertIsNone(omni["language"])
        self.assertIsNone(omni["num_step"])

    def test_mode_validated(self):
        self.assertEqual(spec.build_omni(mode="bogus")["mode"], "auto")
        self.assertEqual(spec.build_omni(mode="clone")["mode"], "clone")

    def test_apply_omni_clone(self):
        entry = {
            "engine": "omnivoice",
            "voice": "auto",
            "language": "auto",
            "instruct": "",
        }
        out = spec.apply_omni_to_voice(
            entry,
            {"mode": "clone", "ref_audio": "C:/ref.wav", "ref_text": "Hello"},
        )
        self.assertEqual(out["ref_audio"], "C:/ref.wav")
        self.assertEqual(out["ref_text"], "Hello")
        self.assertEqual(out["instruct"], "")
        self.assertIsNone(out["language"])  # auto -> None
        # The original entry must not be mutated.
        self.assertEqual(entry.get("ref_audio"), None)

    def test_apply_omni_design(self):
        out = spec.apply_omni_to_voice(
            {"engine": "omnivoice", "voice": "auto"},
            {"mode": "design", "instruct": "female, whisper, british accent",
             "language": "en", "num_step": 48},
        )
        self.assertEqual(out["instruct"], "female, whisper, british accent")
        self.assertEqual(out["language"], "en")
        self.assertEqual(out["omni"]["num_step"], 48)

    def test_apply_omni_auto_keeps_catalog_design_voice(self):
        out = spec.apply_omni_to_voice(
            {"engine": "omnivoice", "voice": "female", "language": "auto"}, {}
        )
        self.assertEqual(out["instruct"], "female")

    def test_apply_omni_server_preset(self):
        out = spec.apply_omni_to_voice(
            {"engine": "omnivoice_server", "voice": "nova", "language": "auto"}, {}
        )
        self.assertEqual(out["instruct"], "female, young adult, high pitch, american accent")


class WorkerRequestBuilderTest(unittest.TestCase):
    def test_basic_request(self):
        req = build_synthesize_request(text="Hello")
        self.assertEqual(req["cmd"], "synthesize")
        self.assertEqual(req["mode"], "triton")
        self.assertIsNone(req["language"])  # auto-detect
        self.assertNotIn("num_step", req)

    def test_full_knobs(self):
        req = build_synthesize_request(
            text="Hi", num_step=64, guidance_scale=4.0,
            class_temperature=0.2, duration=5.0, seed=42,
            language="hi", instruct="male, whisper",
        )
        self.assertEqual(req["num_step"], 64)
        self.assertEqual(req["guidance_scale"], 4.0)
        self.assertEqual(req["class_temperature"], 0.2)
        self.assertEqual(req["duration"], 5.0)
        self.assertEqual(req["seed"], 42)
        self.assertEqual(req["language"], "hi")
        self.assertEqual(req["instruct"], "male, whisper")

    def test_clone_request(self):
        req = build_synthesize_request(
            text="Hello", ref_audio="r.wav", ref_text="transcript"
        )
        self.assertEqual(req["ref_audio"], "r.wav")
        self.assertEqual(req["ref_text"], "transcript")


class _FakeResponse:
    status = 200

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(np.zeros(200, dtype=np.int16).tobytes())
    return buf.getvalue()


class ServerSpeechPayloadTest(unittest.TestCase):
    """The manager must build exactly the payload the real omnivoice-server
    HTTP API documents, without needing a live server."""

    def setUp(self):
        self.captured = {}
        self._patch = mock.patch.object(urllib.request, "urlopen", self._fake_urlopen)
        self._patch.start()
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8880)

    def tearDown(self):
        self._patch.stop()

    def _fake_urlopen(self, req, timeout=None):
        self.captured["url"] = req.full_url
        self.captured["method"] = req.get_method()
        self.captured["body"] = req.data
        return _FakeResponse(_wav_bytes())

    def _payload(self):
        return json.loads(self.captured["body"].decode("utf-8"))

    def test_default_payload_matches_previous_behaviour(self):
        self.mgr.synthesize("hello world")
        payload = self._payload()
        self.assertIn("guidance_scale", payload)
        self.assertEqual(payload["guidance_scale"], 3.0)
        self.assertEqual(payload["denoise"], True)
        self.assertNotIn("instructions", payload)

    def test_all_generation_fields_sent_when_set(self):
        self.mgr.synthesize(
            "hello",
            instructions="female, whisper",
            num_step=48,
            guidance_scale=None,  # explicit None -> server default (omitted)
            denoise=False,
            t_shift=0.2,
            position_temperature=2.0,
            class_temperature=0.5,
            duration=4.0,
            language="en",
            seed=7,
        )
        payload = self._payload()
        self.assertNotIn("guidance_scale", payload)
        self.assertEqual(payload["num_step"], 48)
        self.assertEqual(payload["denoise"], False)
        self.assertEqual(payload["t_shift"], 0.2)
        self.assertEqual(payload["position_temperature"], 2.0)
        self.assertEqual(payload["class_temperature"], 0.5)
        self.assertEqual(payload["duration"], 4.0)
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["instructions"], "female, whisper")
        self.assertTrue(self.captured["url"].endswith("/v1/audio/speech"))

    def test_clone_form_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as fh:
                fh.write(_wav_bytes())
            self.mgr.synthesize_clone(
                "clone me",
                ref_audio_path=ref,
                ref_text="Reference",
                num_step=16,
                seed=3,
            )
        body = self.captured["body"].decode("utf-8", errors="replace")
        self.assertTrue(self.captured["url"].endswith("/v1/audio/speech/clone"))
        self.assertIn('name="text"', body)
        self.assertIn("clone me", body)
        self.assertIn('name="num_step"', body)
        self.assertIn("16", body)
        self.assertIn('name="seed"', body)
        self.assertIn('name="ref_audio"', body)

    def test_profile_and_model_endpoints(self):
        def fake_json_urlopen(req, timeout=None):
            self.captured["url"] = req.full_url
            if req.full_url.endswith("/v1/models"):
                data = {"data": [{"id": "omnivoice"}]}
            else:
                data = {
                    "voices": [
                        {"id": "alloy", "type": "preset"},
                        {"id": "clone:my_voice", "type": "clone",
                         "profile_id": "my_voice"},
                    ],
                    "design_attributes": {"gender": ["male", "female"]},
                }
            return _FakeResponse(json.dumps(data).encode("utf-8"))

        with mock.patch.object(urllib.request, "urlopen", fake_json_urlopen):
            self.assertEqual(self.mgr.get_models(), [{"id": "omnivoice"}])
            profiles = self.mgr.list_profiles()
            self.assertEqual(profiles, [{"profile_id": "my_voice", "description": ""}])
            info = self.mgr.get_voice_info()
            self.assertEqual(
                info["design_attributes"], {"gender": ["male", "female"]}
            )


class EmbeddedWorkerParityTest(unittest.TestCase):
    """The frozen-app worker copy must stay in sync with worker.py."""

    def test_embedded_source_has_modern_knobs(self):
        import ai_voice_studio.omnivoice as ov

        embedded = ov._WORKER_SOURCE
        here = os.path.dirname(ov.__file__)
        with open(os.path.join(here, "worker.py"), encoding="utf-8") as fh:
            on_disk = fh.read()
        for sentinel in (
            "guidance_scale", "class_temperature", "duration",
            "_resample_speed", "mode_used", "skipped",
        ):
            self.assertIn(sentinel, embedded, sentinel)
            self.assertIn(sentinel, on_disk, sentinel)


if __name__ == "__main__":
    unittest.main()
