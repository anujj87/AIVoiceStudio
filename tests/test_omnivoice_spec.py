"""Tests for the OmniVoice feature layer (spec, request builders, HTTP client).

None of these tests touch the GPU, torch or the managed venv: they cover the
pure logic added to enable OmniVoice's full feature set in both the direct
(worker) and server engines.
"""

from __future__ import annotations

import email.message
import io
import json
import os
import tempfile
import unittest
import urllib.request
import urllib.error
import wave
from unittest import mock

import numpy as np

from ai_voice_studio.omnivoice import build_synthesize_request, spec
from ai_voice_studio.omnivoice_quality import (
    RETRY_SEED_STRIDE,
    attempt_seed,
    retry_seed,
)
from ai_voice_studio.omnivoice_server import (
    REQUEST_ATTEMPTS,
    OmniVoiceServerManager,
    OmniVoiceServerError,
    split_text_for_server,
)


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


class ServerRobustnessTest(unittest.TestCase):
    """Regression tests: the server client must not declare a healthy server
    "not responding" (cloning stalls /health briefly) and must not kill long
    recordings with a fixed 120s timeout."""

    def setUp(self):
        self.captured = {}
        self.calls = []
        self.health_failures = 2  # first N /health probes fail
        self._patch = mock.patch.object(urllib.request, "urlopen", self._fake_urlopen)
        self._patch.start()
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8881)

    def tearDown(self):
        self._patch.stop()

    def _fake_urlopen(self, req, timeout=None):
        self.calls.append((req.full_url, timeout))
        self.captured["url"] = req.full_url
        self.captured["timeout"] = timeout
        if req.full_url.endswith("/health"):
            if len(self.calls) <= self.health_failures:
                raise urllib.error.URLError("timed out")
            return _FakeResponse(b"OK")
        return _FakeResponse(_wav_bytes())

    def test_health_check_retries_transient_failures(self):
        self.assertTrue(self.mgr.health_check(retries=3, delay=0.0))
        self.assertEqual(len(self.calls), 3)

    def test_health_check_gives_up_after_retries(self):
        self.health_failures = 99  # every probe fails
        self.assertFalse(self.mgr.health_check(retries=3, delay=0.0))
        self.assertEqual(len(self.calls), 3)

    def test_health_check_default_is_retried(self):
        """The zero-arg call used by the engine must also retry."""
        self.assertTrue(self.mgr.health_check())

    def test_synthesize_timeout_scales_with_text(self):
        """A long text must get a longer socket timeout than the old fixed
        120s, and request_timeout_s still wins when given."""
        short = "hello"
        # Keep this text under the server's 10k-char request cap so it is
        # still ONE request; long-text splitting has its own test below.
        long_text = "word " * 1200  # 6000 chars
        self.mgr.synthesize(short)
        short_timeout = self.captured["timeout"]
        self.mgr.synthesize(long_text)
        long_timeout = self.captured["timeout"]
        self.assertGreaterEqual(short_timeout, 120.0)
        self.assertGreater(long_timeout, short_timeout)
        self.assertLessEqual(long_timeout, 1800.0)

        self.mgr.synthesize(short, request_timeout_s=25)
        self.assertEqual(self.captured["timeout"], 25.0)

    def test_clone_timeout_scales_with_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as fh:
                fh.write(_wav_bytes())
            self.mgr.synthesize_clone(
                "word " * 1200, ref_audio_path=ref)  # 6000 chars, one request
        self.assertGreater(self.captured["timeout"], 120.0)


class ServerLongTextChunkingTest(unittest.TestCase):
    """The omnivoice-server API caps ``input``/``text`` at 10,000 characters
    and answers anything longer with HTTP 422 "Request validation failed" —
    the exact error long book chapters hit on their very first segment.
    The client must split long texts into server-sized chunks at sentence
    boundaries, send one request per chunk and join the audio in order."""

    def setUp(self):
        self.requests = []
        self._patch = mock.patch.object(urllib.request, "urlopen", self._fake_urlopen)
        self._patch.start()
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8880)

    def tearDown(self):
        self._patch.stop()

    def _fake_urlopen(self, req, timeout=None):
        self.requests.append(req)
        return _FakeResponse(_wav_bytes())

    def test_split_respects_limit(self):
        text = (". ".join(f"Sentence number {i} here" for i in range(400)))
        chunks = split_text_for_server(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 9500)
        self.assertEqual("".join(chunks).replace(" ", ""),
                         text.replace(" ", ""))

    def test_split_short_text_is_one_chunk(self):
        self.assertEqual(split_text_for_server("hello world"), ["hello world"])

    def test_split_handles_devanagari_danda(self):
        sentence = "यह एक वाक्य है। "
        text = sentence * 600  # ~10,800 chars of danda-terminated Hindi
        chunks = split_text_for_server(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 9500)

    def test_long_text_sent_as_multiple_requests_in_order(self):
        text = ("This is a test sentence for chunking. " * 400)  # ~16k chars
        samples = self.mgr.synthesize(text, voice="alloy", instructions="male")
        self.assertGreaterEqual(len(self.requests), 2)
        sent = [json.loads(r.data.decode("utf-8"))["input"] for r in self.requests]
        self.assertEqual("".join(s.replace(" ", "") for s in sent),
                         text.replace(" ", ""))
        # Every parameter rides along on every chunk.
        for r in self.requests:
            payload = json.loads(r.data.decode("utf-8"))
            self.assertEqual(payload["voice"], "alloy")
            self.assertEqual(payload["instructions"], "male")
        self.assertEqual(len(samples), 200 * len(self.requests))  # one _wav_bytes() per chunk

    def test_long_clone_text_sent_as_multiple_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as fh:
                fh.write(_wav_bytes())
            text = ("यह किताब का एक लंबा अध्याय है। " * 400)  # ~12.4k chars
            samples = self.mgr.synthesize_clone(
                text, ref_audio_path=ref, ref_text="sample transcript")
        self.assertGreater(len(self.requests), 1)
        for r in self.requests:
            body = r.data.decode("utf-8", errors="replace")
            self.assertIn('name="text"', body)
            self.assertIn('name="ref_audio"', body)
            self.assertIn('name="ref_text"', body)
        self.assertEqual(len(samples), 200 * len(self.requests))

    def test_no_chunk_exceeds_server_limit(self):
        """Even a punctuation-less wall of text must be hard-cut below 10k."""
        text = "अ" * 25000
        chunks = split_text_for_server(text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 9500)
        with mock.patch.object(urllib.request, "urlopen", self._fake_urlopen):
            self.mgr.synthesize(text)
        for r in self.requests:
            payload = json.loads(r.data.decode("utf-8"))
            self.assertLessEqual(len(payload["input"]), 10000)


class ServerHttpErrorDetailTest(unittest.TestCase):
    """HTTP 500/4xx bodies carry the server's real error; the client must
    surface it ("server returned HTTP 500: ...") instead of the bare
    "HTTP Error 500", and the GUI must not offer a restart for these
    request-level failures (the server is demonstrably up)."""

    def _http_error(self, code: int, body: bytes) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            url="http://127.0.0.1:8880/v1/audio/speech/clone",
            code=code, msg="Internal Server Error",
            hdrs=email.message.Message(), fp=io.BytesIO(body),
        )

    def test_detail_extracted_from_json_error_body(self):
        body = json.dumps(
            {"error": {"code": "inference_failed",
                        "message": "Synthesis failed: [Errno 22] Invalid argument"}}
        ).encode()
        detail = OmniVoiceServerManager._http_error_detail(self._http_error(500, body))
        self.assertIn("server returned HTTP 500", detail)
        self.assertIn("[Errno 22] Invalid argument", detail)

    def test_detail_extracts_fastapi_detail_field(self):
        body = json.dumps({"detail": "Upload too large"}).encode()
        detail = OmniVoiceServerManager._http_error_detail(self._http_error(413, body))
        self.assertIn("Upload too large", detail)

    def test_detail_falls_back_to_plain_body(self):
        detail = OmniVoiceServerManager._http_error_detail(
            self._http_error(502, b"bad gateway"))
        self.assertIn("bad gateway", detail)

    def test_non_http_errors_pass_through(self):
        self.assertEqual(
            OmniVoiceServerManager._http_error_detail(TimeoutError("timed out")),
            "timed out",
        )

    def test_synthesize_error_includes_server_detail(self):
        def fake_urlopen(req, timeout=None):
            raise self._http_error(
                500,
                json.dumps({"error": {"message": "boom reason"}}).encode())
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            mgr = OmniVoiceServerManager(host="127.0.0.1", port=8881)
            with self.assertRaises(OmniVoiceServerError) as ctx:
                mgr.synthesize("hello")
        self.assertIn("server returned HTTP 500", str(ctx.exception))
        self.assertIn("boom reason", str(ctx.exception))

    def test_clone_error_includes_server_detail(self):
        def fake_urlopen(req, timeout=None):
            raise self._http_error(
                500,
                json.dumps({"error": {"message": "clone boom"}}).encode())
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            with tempfile.TemporaryDirectory() as tmp:
                ref = os.path.join(tmp, "ref.wav")
                with open(ref, "wb") as fh:
                    fh.write(_wav_bytes())
                mgr = OmniVoiceServerManager(host="127.0.0.1", port=8881)
                with self.assertRaises(OmniVoiceServerError) as ctx:
                    mgr.synthesize_clone("hello", ref_audio_path=ref)
        self.assertIn("clone boom", str(ctx.exception))


class ServerErrorDialogClassificationTest(unittest.TestCase):
    """Request-level failures must NOT trigger the "restart the server?"
    dialog: the server answered, so it is alive."""

    def test_http_500_is_not_restartable(self):
        from ai_voice_studio.gui.dialogs import looks_like_server_error
        msg = ("Clone synthesis failed: server returned HTTP 500: "
               "Synthesis failed: [Errno 22] Invalid argument")
        self.assertFalse(looks_like_server_error(msg))

    def test_dead_server_still_offers_restart(self):
        from ai_voice_studio.gui.dialogs import looks_like_server_error
        self.assertTrue(looks_like_server_error(
            "Synthesis failed: server is not responding"))
        self.assertTrue(looks_like_server_error(
            "Server did not become ready within 120s."))


class ServerLogRedirectionTest(unittest.TestCase):
    """The server subprocess must log to a real file, not an undrained
    PIPE (a full pipe made every server log write fail with [Errno 22],
    which surfaced as HTTP 500 on every request)."""

    def test_manager_has_log_path_helpers(self):
        mgr = OmniVoiceServerManager(host="127.0.0.1", port=8881)
        self.assertIsNone(mgr._log_path)
        self.assertIsNone(mgr._log_fh)
        tail = mgr._log_tail()
        self.assertIsInstance(tail, str)



def _multipart_field(body: bytes, name: str) -> str:
    """The value of one form field in a multipart request body."""
    raw = body.decode("utf-8", errors="replace")
    marker = f'name="{name}"\r\n\r\n'
    start = raw.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = raw.find("\r\n", start)
    return raw[start:end if end >= 0 else None]


class RetrySeedTest(unittest.TestCase):
    """A request re-sent inside one attempt must not reuse a drone seed."""

    def test_first_send_keeps_the_callers_seed(self):
        self.assertEqual(retry_seed(7, 0), 7)
        self.assertIsNone(retry_seed(None, 0))

    def test_an_unseeded_request_stays_unseeded(self):
        # Nothing to rotate: an unseeded draw already samples freely, so
        # pinning a retry to a seed would make it *less* likely to differ.
        self.assertIsNone(retry_seed(None, 1))
        self.assertIsNone(retry_seed(None, 2))

    def test_a_seeded_request_moves_off_its_seed(self):
        self.assertEqual(retry_seed(7, 1), 7 + RETRY_SEED_STRIDE)
        # Its own stride: a request retry never meets a drone retry's seed.
        self.assertNotEqual(retry_seed(7, 1), attempt_seed(7, 1))
        self.assertNotEqual(retry_seed(7, 1), 7)


class RequestRetryTest(unittest.TestCase):
    """A failed request is a bad draw, not a lost segment.

    The server answers HTTP 500 with a generic "Internal Server Error" body
    when one generation comes back empty — its own log names the text,
    "Generation returned no audio for text: '629'" — while every request around
    it succeeds (measured: 3 such answers among 152 clone requests while
    recording one Hindi chapter).  One of them used to end a segment with
    "Segment 19 failed: Clone synthesis failed: server returned HTTP 500",
    discarding the rest of the chapter, so the same text is sent again with a
    different draw.
    """

    def setUp(self):
        self.requests: list = []
        self.mgr = OmniVoiceServerManager(host="127.0.0.1", port=8881)

    def _fake_urlopen(self, plan):
        def urlopen(req, timeout=None):
            self.requests.append(req)
            return plan(len(self.requests))
        return urlopen

    def _http_error(self, code: int, message: str) -> urllib.error.HTTPError:
        body = json.dumps({"error": {"message": message}}).encode()
        return urllib.error.HTTPError(
            url="http://127.0.0.1:8881/v1/audio/speech/clone",
            code=code, msg=message,
            hdrs=email.message.Message(), fp=io.BytesIO(body),
        )

    @staticmethod
    def _flagged(data: bytes):
        """The server's own "no speech" verdict on an otherwise fine take."""
        resp = _FakeResponse(data)
        resp.headers = {"X-No-Speech-Detected": "true"}
        return resp

    def _clone(self, seed=7):
        """One clone synthesis against a real (temporary) reference file."""
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as fh:
                fh.write(_wav_bytes())
            return self.mgr.synthesize_clone(
                "629", ref_audio_path=ref, ref_text="sample", seed=seed,
            )

    def _run(self, plan):
        with mock.patch.object(urllib.request, "urlopen",
                               self._fake_urlopen(plan)):
            return self._clone()

    def test_a_failed_request_is_sent_again(self):
        def plan(i):
            if i == 1:
                raise self._http_error(500, "Internal Server Error")
            return _FakeResponse(_wav_bytes())

        samples = self._run(plan)

        self.assertEqual(len(self.requests), 2)
        self.assertEqual(samples.size, 200)          # the take that came back
        # Same text both times: the request is repeated, not changed.
        texts = [_multipart_field(r.data, "text") for r in self.requests]
        self.assertEqual(texts, ["629", "629"])
        # ... and the retry is a different draw, off its own stride.
        self.assertEqual(
            [_multipart_field(r.data, "seed") for r in self.requests],
            ["7", str(7 + RETRY_SEED_STRIDE)],
        )

    def test_a_clean_request_is_sent_once(self):
        samples = self._run(lambda i: _FakeResponse(_wav_bytes()))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(samples.size, 200)

    def test_every_attempt_failing_still_raises(self):
        with mock.patch.object(
            urllib.request, "urlopen",
            self._fake_urlopen(
                lambda i: (_ for _ in ()).throw(
                    self._http_error(500, "Internal Server Error"))
            ),
        ):
            with self.assertRaises(OmniVoiceServerError) as ctx:
                self._clone()

        self.assertEqual(len(self.requests), REQUEST_ATTEMPTS)
        message = str(ctx.exception)
        self.assertIn("Clone synthesis failed", message)
        self.assertIn("server returned HTTP 500", message)
        # A request-level 500 is still an answered request: no restart dialog.
        from ai_voice_studio.gui.dialogs import looks_like_server_error
        self.assertFalse(looks_like_server_error(message))

    def test_a_rejected_request_is_not_sent_again(self):
        """A 4xx is the server refusing the request, so a repeat cannot help."""
        with mock.patch.object(
            urllib.request, "urlopen",
            self._fake_urlopen(
                lambda i: (_ for _ in ()).throw(
                    self._http_error(422, "Request validation failed"))
            ),
        ):
            with self.assertRaises(OmniVoiceServerError) as ctx:
                self._clone()

        self.assertEqual(len(self.requests), 1)
        self.assertIn("server returned HTTP 422", str(ctx.exception))

    def test_a_dead_server_is_still_reported_as_one(self):
        refused = ConnectionRefusedError(
            "[WinError 10061] No connection could be made because the target "
            "machine actively refused it"
        )
        with mock.patch.object(
            urllib.request, "urlopen",
            self._fake_urlopen(lambda i: (_ for _ in ()).throw(refused)),
        ):
            with self.assertRaises(OmniVoiceServerError) as ctx:
                self._clone()

        self.assertEqual(len(self.requests), REQUEST_ATTEMPTS)
        from ai_voice_studio.gui.dialogs import looks_like_server_error
        self.assertTrue(looks_like_server_error(str(ctx.exception)))

    def test_a_failed_request_does_not_use_up_a_drone_attempt(self):
        """The two retries are independent: the drone policy still gets its
        own seeds (the caller's first, then ``attempt_seed``), so a failed
        request costs a re-send and nothing else."""
        def plan(i):
            if i == 1:
                raise self._http_error(500, "Internal Server Error")
            if i == 2:
                return self._flagged(_wav_bytes())   # drone attempt 0
            return _FakeResponse(_wav_bytes())       # drone attempt 1: clean

        samples = self._run(plan)

        self.assertEqual(len(self.requests), 3)
        self.assertEqual(
            [_multipart_field(r.data, "seed") for r in self.requests],
            ["7", str(7 + RETRY_SEED_STRIDE), str(attempt_seed(7, 1))],
        )
        self.assertEqual(samples.size, 200)
        self.assertEqual(self.mgr.last_repairs, [])

    def test_the_error_quotes_the_servers_own_reason(self):
        """"Internal Server Error" alone says nothing; the server log does."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "omnivoice_server.log")
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(
                    'INFO:     127.0.0.1:3155 - "POST /v1/audio/speech/clone '
                    'HTTP/1.1" 200 OK\n'
                    "2026-09-21T21:34:44Z [INFO ] "
                    "[omnivoice_server.services.inference] [TRACE] CLONE mode "
                    "kwargs prepared: ref_audio=C:\\tmp\\ref_audio.wav\n"
                    "2026-09-21T21:34:47Z [WARNING] "
                    "[omnivoice_server.services.inference] Generation "
                    "returned no audio for text: '629'\n"
                    'INFO:     127.0.0.1:3157 - "POST /v1/audio/speech/clone '
                    'HTTP/1.1" 500 Internal Server Error\n'
                    "ERROR:    Exception in ASGI application\n"
                    "Traceback (most recent call last):\n"
                    '  File "speech.py", line 744, in create_speech_clone\n'
                    "ValueError: tensors_to_wav_bytes[0]: tensor is empty "
                    "(size=0)\n"
                )
            self.mgr._log_path = log_path
            with mock.patch.object(
                urllib.request, "urlopen",
                self._fake_urlopen(
                    lambda i: (_ for _ in ()).throw(
                        self._http_error(500, "Internal Server Error"))
                ),
            ):
                with self.assertRaises(OmniVoiceServerError) as ctx:
                    self._clone()

        message = str(ctx.exception)
        self.assertIn(
            "server log: Generation returned no audio for text: '629'", message
        )
        # The traceback is not quoted: it is long and it is not the reason.
        self.assertNotIn("Traceback", message)

    def test_the_reason_is_found_when_another_process_started_the_server(self):
        """The manager answering requests may not be the one that started the
        server, so the server's standard log location is read as a fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "omnivoice_server.log"), "w",
                      encoding="utf-8") as fh:
                fh.write(
                    "2026-09-21T21:34:47Z [WARNING] "
                    "[omnivoice_server.services.inference] Generation "
                    "returned no audio for text: '633'\n"
                )
            self.mgr._log_path = None
            with mock.patch("ai_voice_studio.paths.logs_dir", return_value=tmp):
                self.assertEqual(
                    self.mgr._last_server_problem(),
                    "Generation returned no audio for text: '633'",
                )

    def test_a_5xx_without_a_server_log_still_reports_the_body(self):
        with mock.patch.object(
            urllib.request, "urlopen",
            self._fake_urlopen(
                lambda i: (_ for _ in ()).throw(
                    self._http_error(500, "Synthesis failed: out of memory"))
            ),
        ):
            with self.assertRaises(OmniVoiceServerError) as ctx:
                self._clone()
        self.assertIn("out of memory", str(ctx.exception))


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
