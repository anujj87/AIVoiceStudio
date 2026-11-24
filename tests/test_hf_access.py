"""Hugging Face access: the token a gated model needs, and where it is used.

Google's Gemma 3 is the one entry of the describing catalog whose repository is
*gated*: an anonymous download answers a bare ``401`` ("Cannot access gated
repo ... Please log in"), which names neither the page that grants access nor a
way to give the application a token.  That is the failure this module exists to
prevent, so the tests here are about the three things that fix it:

* where a token comes from (Settings, ``HF_TOKEN``, or ``huggingface-cli``);
* that it reaches the *worker* processes, which are the ones that download;
* that a gated run says so before it starts, and that a bare 401 becomes the
  two steps that unblock it rather than being passed through.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import hf_access
from ai_voice_studio.describe import catalog as describe_catalog
from ai_voice_studio.describe import engine
from ai_voice_studio.describe import worker as describe_worker
from ai_voice_studio.settings import Settings
from ai_voice_studio.voicelab import engines as voicelab_engines

#: The verbatim body of the failure the user reported, request id and all.
GATED_401 = (
    "401 Client Error. (Request ID: "
    "Root=1-6aca6617-4f1df26a3a3dc71723b2738f;deb4b85a-f513-49a7-ba29-1901e292f95f)\n"
    "Cannot access gated repo for url "
    "https://huggingface.co/google/gemma-3-12b-it/resolve/96b6f1eccf38110c56df3a15bffe176da0\n"
    "Access to model google/gemma-3-12b-it is restricted. You must have access "
    "to it and be authenticated to access it. Please log in."
)

#: Kyutai's own text when Pocket TTS's voice-cloning weights could not be
#: fetched - quoted from ``pocket_tts/models/tts_model.py`` (its
#: ``VOICE_CLONING_UNSUPPORTED`` names the pre-made voices where the ellipsis
#: is).  It is the second gated repository this application ships, and its
#: advice (log in locally with ``hf auth login``) is as unusable as the Gemma 3
#: 401 was for a person who installed the setup file.
POCKET_CLONING_UNSUPPORTED = (
    "We could not download the weights for the model with voice cloning, but "
    "you're trying to use voice cloning. Without voice cloning, you can use "
    "our catalog of voices [...]. If you want access to the model with voice "
    "cloning, go to https://huggingface.co/kyutai/pocket-tts and accept the "
    "terms, then make sure you're logged in locally with `uvx hf auth login`."
)


class _SettingsCase(unittest.TestCase):
    """Every test gets its own settings file - never the user's own."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="aivs_hf_settings_")
        self.settings = Settings(path=os.path.join(self.folder, "settings.json"))
        self.addCleanup(lambda: hf_access.use_settings(None))
        hf_access.use_settings(self.settings)
        # A token in the machine's own environment or CLI cache must not decide
        # what a test sees.
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in hf_access.ENV_VARS:
            os.environ.pop(name, None)
        self.cli_token = os.path.join(self.folder, "cli-token")
        os.environ["HF_TOKEN_PATH"] = self.cli_token


class TokenLookupTest(_SettingsCase):
    """One question - which token - with three answers, in a fixed order."""

    def test_no_token_anywhere_is_reported_as_none(self):
        self.assertEqual(hf_access.token(), "")
        self.assertFalse(hf_access.has_token())
        self.assertEqual(hf_access.token_source(), "")

    def test_the_settings_panel_comes_first(self):
        hf_access.store_token(self.settings, "hf_from_settings")
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
            self.assertEqual(hf_access.token(), "hf_from_settings")
        self.assertEqual(hf_access.token_source(),
                         "Settings > Hugging Face account")

    def test_the_environment_variable_is_used_when_settings_are_empty(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
            self.assertEqual(hf_access.token(), "hf_from_env")
            self.assertIn("HF_TOKEN", hf_access.token_source())

    def test_the_older_environment_spelling_is_honoured(self):
        with mock.patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "hf_old"}):
            self.assertEqual(hf_access.token(), "hf_old")

    def test_a_previous_cli_login_is_used_last(self):
        with open(self.cli_token, "w", encoding="utf-8") as handle:
            handle.write("hf_from_cli\n")
        self.assertEqual(hf_access.token(), "hf_from_cli")
        self.assertIn("huggingface-cli", hf_access.token_source())

    def test_a_token_saved_a_moment_ago_is_visible_without_a_restart(self):
        # The panel writes through the live store, so the very next run sees it.
        hf_access.store_token(self.settings, "hf_just_pasted")
        self.assertEqual(hf_access.worker_environment()["HF_TOKEN"],
                         "hf_just_pasted")

    def test_the_token_is_read_from_the_settings_file_when_no_store_is_given(
            self):
        # The engines run in threads that were handed no store; the file is the
        # fallback, so a token saved by the panel is still found.
        hf_access.use_settings(None)
        self.settings.set(hf_access.TOKEN_SETTING, "hf_from_file")
        self.settings.save()
        with mock.patch.object(hf_access, "settings_store",
                               return_value=self.settings):
            self.assertEqual(hf_access.token(), "hf_from_file")


class WorkerEnvironmentTest(_SettingsCase):
    """The token has to reach the process that does the download."""

    def test_every_worker_spelling_is_set(self):
        hf_access.store_token(self.settings, "hf_worker")
        env = hf_access.worker_environment()
        for name in hf_access.ENV_VARS:
            self.assertEqual(env[name], "hf_worker")

    def test_a_worker_without_a_token_gets_no_variable(self):
        env = hf_access.worker_environment()
        for name in hf_access.ENV_VARS:
            self.assertNotIn(name, env)

    def test_the_rest_of_the_environment_is_carried_through(self):
        os.environ["AIVS_TEST_MARKER"] = "1"
        self.addCleanup(os.environ.pop, "AIVS_TEST_MARKER", None)
        self.assertEqual(hf_access.worker_environment()["AIVS_TEST_MARKER"], "1")

    def test_the_describe_worker_is_started_with_the_token(self):
        """The downloader is the worker, so the token must be in its env.

        The launch is captured rather than the whole run: the run is *meant* to
        fail here (no response file is written), and what is being checked is
        the ``env=`` the process was started with.
        """
        hf_access.store_token(self.settings, "hf_describe")
        seen = self._capture_describe_launch()
        self.assertEqual(seen.get("env", {}).get("HF_TOKEN"), "hf_describe")

    def _capture_describe_launch(self) -> dict:
        seen: dict = {}

        class _Stream:
            def __iter__(self):
                return iter(())

            def close(self):
                pass

        class _Proc:
            returncode = 0
            stdout = _Stream()

            def wait(self, timeout=None):
                return 0

        def _popen(cmd, **kwargs):
            seen.update(kwargs)
            return _Proc()

        folder = tempfile.mkdtemp(prefix="aivs_hf_launch_")
        script = os.path.join(folder, "worker.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("# never executed: the process is captured")
        with mock.patch.object(engine.environments, "worker_script",
                               return_value=script), \
                mock.patch.object(engine.environments, "runtime") as runtime, \
                mock.patch.object(engine.subprocess, "Popen", side_effect=_popen):
            runtime.return_value.usable_python_exe.return_value = sys.executable
            with self.assertRaises(engine.DescribeEngineError):
                engine._run_worker("smolvlm2", {"hf_model": "Org/Repo"}, None)
        return seen

    def test_the_stt_worker_is_started_with_the_token(self):
        from ai_voice_studio.stt import engine as stt_engine

        hf_access.store_token(self.settings, "hf_stt")
        seen: dict = {}

        def _run(cmd, **kwargs):
            seen.update(kwargs)
            return mock.Mock(returncode=1, stdout="", stderr="")

        folder = tempfile.mkdtemp(prefix="aivs_hf_stt_")
        script = os.path.join(folder, "worker.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write("# never executed")
        with mock.patch.object(stt_engine.environments, "worker_script",
                               return_value=script), \
                mock.patch.object(stt_engine.environments, "runtime") as runtime, \
                mock.patch.object(stt_engine.subprocess, "run", side_effect=_run):
            runtime.return_value.usable_python_exe.return_value = sys.executable
            with self.assertRaises(stt_engine.STTEngineError):
                stt_engine._run_worker("whisper_large_v3", {}, None)
        self.assertEqual(seen.get("env", {}).get("HF_TOKEN"), "hf_stt")

    def test_the_other_worker_launches_pass_the_same_environment(self):
        """Voice Lab, OmniVoice and its server name the helper in their launch.

        Checked by reading the call site: standing each of those three up for
        real needs a per-model virtualenv, and what can regress here is exactly
        one thing - a new ``subprocess.Popen`` that forgets the environment and
        would download anonymously, which is the failure being fixed.
        """
        root = os.path.join(os.path.dirname(__file__), "..", "ai_voice_studio")
        for relative in ("describe/engine.py", "stt/engine.py",
                         "voicelab/__init__.py", "omnivoice/__init__.py",
                         "omnivoice_server/__init__.py"):
            with self.subTest(module=relative):
                with open(os.path.join(root, relative), encoding="utf-8") as handle:
                    text = handle.read()
                self.assertIn("hf_access.worker_environment()", text)


class AccessErrorTest(unittest.TestCase):
    """A gated failure must be recognised, and a real crash must not be."""

    def test_the_reported_failure_is_recognised(self):
        self.assertTrue(hf_access.is_access_error(GATED_401))

    def test_a_missing_file_is_not_an_access_failure(self):
        self.assertFalse(hf_access.is_access_error(
            "RuntimeError: Video file not found: C:/clips/a.mp4"))

    def test_a_number_that_only_looks_like_a_status_is_not_one(self):
        # The failure text carries a request id, and a crash can carry an
        # address; neither is an HTTP status.
        self.assertFalse(hf_access.is_access_error(
            "MemoryError at 0x00000000014013ff in module torch_cpu.dll"))
        self.assertFalse(hf_access.is_access_error(
            "OSError: [WinError 112] not enough space (14013 bytes wanted)"))

    def test_the_help_names_the_page_the_panel_and_the_licence(self):
        text = hf_access.gated_help("google/gemma-3-12b-it", "Gemma 3")
        self.assertIn("https://huggingface.co/google/gemma-3-12b-it", text)
        self.assertIn(hf_access.TOKEN_PAGE, text)
        self.assertIn("Settings > Hugging Face account", text)
        self.assertIn("licence", text)
        self.assertIn("Gemma 3", text)

    def test_a_non_access_failure_is_left_alone(self):
        self.assertEqual(
            hf_access.access_message("RuntimeError: out of memory", "Org/Repo"),
            "")


class VerifyTokenTest(unittest.TestCase):
    """Check token: the difference between a working token and a mistyped one."""

    def _response(self, status, payload=None):
        response = mock.Mock()
        response.status_code = status
        response.json.return_value = payload if payload is not None else {}
        return response

    def test_a_good_token_reports_the_account(self):
        with mock.patch.object(hf_access.requests, "get",
                               return_value=self._response(200, {"name": "anujj87"})) \
                as get:
            ok, detail = hf_access.verify("hf_good")
        self.assertTrue(ok, detail)
        self.assertEqual(detail, "anujj87")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"],
                         "Bearer hf_good")

    def test_a_refused_token_says_so(self):
        with mock.patch.object(hf_access.requests, "get",
                               return_value=self._response(401)):
            ok, detail = hf_access.verify("hf_wrong")
        self.assertFalse(ok)
        self.assertIn("did not accept", detail)

    def test_an_empty_field_is_refused_before_any_request(self):
        with mock.patch.object(hf_access.requests, "get") as get:
            ok, detail = hf_access.verify("   ")
        self.assertFalse(ok)
        self.assertIn("Enter a token", detail)
        get.assert_not_called()

    def test_no_network_is_reported_rather_than_raised(self):
        with mock.patch.object(hf_access.requests, "get",
                               side_effect=OSError("no route to host")):
            ok, detail = hf_access.verify("hf_any")
        self.assertFalse(ok)
        self.assertIn("Could not reach", detail)

    def test_storing_a_token_and_its_account_is_one_save(self):
        folder = tempfile.mkdtemp(prefix="aivs_hf_store_")
        settings = Settings(path=os.path.join(folder, "settings.json"))
        hf_access.store_token(settings, "hf_x", "anujj87")
        reopened = Settings(path=os.path.join(folder, "settings.json"))
        self.assertEqual(reopened.get(hf_access.TOKEN_SETTING), "hf_x")
        self.assertEqual(reopened.get(hf_access.ACCOUNT_SETTING), "anujj87")
        hf_access.store_token(settings, "", None)
        self.assertEqual(hf_access.account(settings), "")
        self.assertEqual(hf_access.token(settings), "")


class GatedCatalogTest(unittest.TestCase):
    """Which entries need a token - checked against the live API for each one."""

    def test_gemma3_is_marked_gated_for_every_variant(self):
        for value, _label in describe_catalog.variants("gemma3"):
            with self.subTest(variant=value):
                self.assertTrue(describe_catalog.gated("gemma3", value))

    def test_the_open_models_are_not(self):
        # Gemma 4 is Apache-2.0 and downloadable anonymously, which is why the
        # flag is per model: a family can hold both kinds.
        self.assertFalse(describe_catalog.gated("gemma4", "e4b"))
        for model_id in ("smolvlm2", "qwen3_vl", "qwen2_5_omni", "moondream2"):
            with self.subTest(model=model_id):
                self.assertFalse(describe_catalog.gated(model_id))

    def test_the_note_names_the_extra_step_and_only_for_gated_entries(self):
        note = describe_catalog.gated_note("gemma3", "12b")
        self.assertIn("Gated repository", note)
        self.assertIn("Hugging Face account", note)
        self.assertEqual(describe_catalog.gated_note("smolvlm2", "2.2b"), "")

    def test_the_settings_state_column_says_it_before_the_download(self):
        label = engine.variant_state_label("gemma3", "12b")
        self.assertIn("Gated repository", label)
        self.assertNotIn("Gated repository",
                         engine.variant_state_label("smolvlm2", "2.2b"))

    def test_the_model_page_is_the_repositorys_own(self):
        self.assertEqual(hf_access.model_page("google/gemma-3-12b-it"),
                         "https://huggingface.co/google/gemma-3-12b-it")
        self.assertEqual(hf_access.model_page(""), "")


class GatedRunTest(_SettingsCase):
    """A gated run stops with the two steps, before it downloads anything."""

    def setUp(self):
        super().setUp()
        os.environ.pop("HF_TOKEN_PATH", None)

    def test_downloading_a_gated_variant_without_a_token_explains_itself(self):
        """No worker is started, so no gigabyte is fetched, before access is in
        place - the whole point of checking ahead of the download."""

        def _never(*args, **kwargs):
            raise AssertionError("the worker must not start without access")

        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(engine.subprocess, "Popen",
                                  side_effect=_never):
            result = engine.download_variant("gemma3", "12b")
        self.assertFalse(result["ok"])
        self.assertIn("https://huggingface.co/google/gemma-3-12b-it",
                      result["error"])
        self.assertIn("Settings > Hugging Face account", result["error"])

    def test_a_gated_run_says_so_even_before_the_environment_is_ready(self):
        """The licence step is a user action, so it is the first thing said.

        A machine that has never installed Gemma 3 would otherwise be told to
        install its environment - true, but not what is blocking a download that
        cannot start anonymously anyway.
        """
        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(engine.environments, "runtime",
                                  side_effect=AssertionError("touched")):
            result = engine.download_variant("gemma3", "12b")
        self.assertFalse(result["ok"])
        self.assertIn("Settings > Hugging Face account", result["error"])

    def test_a_configured_token_lets_the_run_go_ahead(self):
        hf_access.store_token(self.settings, "hf_configured")
        seen: dict = {}

        def _fake(model_id, request, on_progress=None, on_segment=None,
                  cancel_event=None):
            seen.update(request)
            return {"ok": True, "path": "C:/cache", "note": "ready"}, []

        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(engine, "_worker_response",
                                  side_effect=_fake):
            result = engine.download_variant("gemma3", "12b")
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(seen["hf_model"], "google/gemma-3-12b-it")

    def test_an_open_model_needs_no_token_at_all(self):
        seen: dict = {}

        def _fake(model_id, request, on_progress=None, on_segment=None,
                  cancel_event=None):
            seen.update(request)
            return {"ok": True, "path": "C:/cache", "note": "ready"}, []

        self.assertFalse(hf_access.has_token())
        with mock.patch.object(engine.environments, "is_installed",
                               return_value=True), \
                mock.patch.object(engine, "_worker_response",
                                  side_effect=_fake):
            result = engine.download_variant("smolvlm2", "2.2b")
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(seen["hf_model"], "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    def test_a_401_that_gets_through_the_worker_is_translated(self):
        # The licence may be unaccepted even with a token, and a run can fail
        # inside the worker; the bare library text must not be the answer.
        detail = engine._explain_access(GATED_401, "google/gemma-3-12b-it",
                                        "gemma3")
        self.assertIn("Settings > Hugging Face account", detail)
        self.assertIn("google/gemma-3-12b-it", detail)
        self.assertNotIn("Request ID", detail)

    def test_a_worker_crash_keeps_its_own_cause(self):
        detail = engine._explain_access("RuntimeError: the model could not be "
                                        "loaded", "Org/Repo", "gemma3")
        self.assertEqual(detail, "RuntimeError: the model could not be loaded")


class DescribeWorkerGatedTest(unittest.TestCase):
    """The worker itself is standalone, so its wording is checked directly."""

    def _load_failure(self, message):
        fake = types.ModuleType("huggingface_hub")
        fake.snapshot_download = mock.Mock(side_effect=RuntimeError(message))
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
            return describe_worker.run(
                {"mode": "download", "hf_model": "google/gemma-3-12b-it"}, "")

    def test_the_workers_own_message_names_the_page_and_the_panel(self):
        result = self._load_failure(
            "401 Client Error: Cannot access gated repo for "
            "https://huggingface.co/google/gemma-3-12b-it/resolve/abc")
        self.assertFalse(result["ok"])
        self.assertIn("https://huggingface.co/google/gemma-3-12b-it",
                      result["error"])
        self.assertIn("Settings > Hugging Face account", result["error"])
        self.assertIn("licence", result["error"])

    def test_a_disk_full_failure_is_not_dressed_up_as_a_licence(self):
        result = self._load_failure("OSError: [Errno 28] No space left on device")
        self.assertFalse(result["ok"])
        self.assertIn("No space left on device", result["error"])
        self.assertNotIn("gated", result["error"])


class GatedHintTest(unittest.TestCase):
    """The hint helper is pure text, so it is exercised on its own."""

    def test_a_recognised_failure_carries_the_hint(self):
        hint = describe_worker._gated_hint(
            RuntimeError("403 Forbidden"), "Org/Gated")
        self.assertIn("https://huggingface.co/Org/Gated", hint)

    def test_an_unrelated_failure_carries_nothing(self):
        self.assertEqual(
            describe_worker._gated_hint(RuntimeError("bad zip file"), "Org/Repo"),
            "")


class _FakeResponse:
    """Just enough of a ``requests.Response`` for the access check."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.content = b""

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class _FakeHuggingFace:
    """A huggingface.co answering the three calls an access check makes.

    ``/api/models/<repo>`` says what the repository is and lists its files,
    ``/api/whoami-v2`` says whose token it is, and ``/<repo>/resolve/...`` is
    the download a gated repository refuses until the licence is accepted.
    """

    def __init__(self, *, gated=True, files=(('config.json', 35),),
                 resolve_status=200, whoami=(200, {"name": "anujj87"})):
        self.gated = gated
        self.files = files
        self.resolve_status = resolve_status
        self.whoami = whoami
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if "/api/models/" in url:
            return _FakeResponse(200, {
                "gated": "manual" if self.gated else False,
                "siblings": [{"rfilename": name, "size": size}
                             for name, size in self.files],
            })
        if "/api/whoami" in url:
            status, payload = self.whoami
            return _FakeResponse(status, payload)
        return _FakeResponse(self.resolve_status,
                             {"error": "Cannot access gated repo"})


class AccessCheckTest(_SettingsCase):
    """*This* token, *this* repository, right now.

    The failure itself ("Cannot access gated repo ... Please log in") is the
    same for a mistyped token and for a licence that was never accepted, and
    the two are fixed in different places - so the check has to name which one
    it is.
    """

    def _patch(self, hf):
        return mock.patch.object(hf_access.requests, "get", hf.get)

    def test_the_registry_holds_every_gated_model_this_application_ships(self):
        listed = {row["repo"] for row in hf_access.gated_repos()}
        for value, _label in describe_catalog.variants("gemma3"):
            with self.subTest(variant=value):
                self.assertIn(describe_catalog.hf_model("gemma3", value), listed)
        # Every Voice Lab engine, with the *one* gated repository each of them
        # downloads from - and only that one: an engine that named a repository
        # nobody can be told about is exactly what this test is for.
        gated_voices = {
            "pocket_tts": "kyutai/pocket-tts",
            "indic_parler": "ai4bharat/indic-parler-tts",
        }
        for engine_id in voicelab_engines.engine_ids():
            expected = gated_voices.get(engine_id, "")
            with self.subTest(engine=engine_id):
                self.assertEqual(voicelab_engines.gated_repo(engine_id), expected)
                if expected:
                    self.assertIn(expected, listed)

    def test_a_gated_model_with_no_token_names_the_page_and_the_panel(self):
        hf = _FakeHuggingFace()
        with self._patch(hf):
            state, detail = hf_access.check_access("google/gemma-3-12b-it")
        self.assertEqual(state, hf_access.ACCESS_NO_TOKEN)
        self.assertIn("https://huggingface.co/google/gemma-3-12b-it", detail)
        self.assertIn(hf_access.TOKEN_PAGE, detail)
        self.assertIn("Settings > Hugging Face account", detail)
        # Nothing is fetched from the repository before access is in place.
        self.assertFalse([call for call in hf.calls
                          if "/resolve/" in call["url"]])

    def test_an_accepted_licence_is_reported_as_access_in_place(self):
        hf = _FakeHuggingFace(resolve_status=200)
        hf_access.store_token(self.settings, "hf_good")
        with self._patch(hf):
            state, detail = hf_access.check_access("google/gemma-3-12b-it")
        self.assertEqual(state, hf_access.ACCESS_OK)
        self.assertIn("Access is in place", detail)

    def test_the_smallest_file_is_the_one_used_as_the_probe(self):
        # A successful check must not download a checkpoint to find out that it
        # may: the file is chosen by size, so "yes" costs a few hundred bytes.
        hf = _FakeHuggingFace(files=(("model.safetensors", 24_000_000_000),
                                     ("tiny.json", 35)))
        hf_access.store_token(self.settings, "hf_good")
        with self._patch(hf):
            hf_access.check_access("google/gemma-3-12b-it")
        probes = [call for call in hf.calls if "/resolve/" in call["url"]]
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0]["url"].endswith("/resolve/main/tiny.json"))
        self.assertEqual(probes[0]["headers"]["Authorization"], "Bearer hf_good")

    def test_a_refusal_with_a_good_token_blames_the_missing_licence(self):
        hf = _FakeHuggingFace(resolve_status=401,
                              whoami=(200, {"name": "anujj87"}))
        hf_access.store_token(self.settings, "hf_good")
        with self._patch(hf):
            state, detail = hf_access.check_access("google/gemma-3-12b-it")
        self.assertEqual(state, hf_access.ACCESS_REFUSED)
        self.assertIn("anujj87", detail)
        self.assertIn("https://huggingface.co/google/gemma-3-12b-it", detail)
        self.assertIn("licence", detail)

    def test_a_refusal_with_a_bad_token_blames_the_token_first(self):
        hf = _FakeHuggingFace(resolve_status=401, whoami=(401, None))
        hf_access.store_token(self.settings, "hf_wrong")
        with self._patch(hf):
            state, detail = hf_access.check_access("google/gemma-3-12b-it")
        self.assertEqual(state, hf_access.ACCESS_REFUSED)
        self.assertIn("Check token", detail)
        self.assertNotIn("has not been granted", detail)

    def test_a_server_error_is_not_reported_as_a_missing_licence(self):
        hf = _FakeHuggingFace(resolve_status=500)
        hf_access.store_token(self.settings, "hf_good")
        with self._patch(hf):
            state, detail = hf_access.check_access("google/gemma-3-12b-it")
        self.assertEqual(state, hf_access.ACCESS_UNKNOWN)
        self.assertIn("500", detail)

    def test_an_open_model_says_no_account_is_needed(self):
        hf = _FakeHuggingFace(gated=False)
        with self._patch(hf):
            state, detail = hf_access.check_access(
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        self.assertEqual(state, hf_access.ACCESS_OPEN)
        self.assertIn("not gated", detail)

    def test_hugging_face_being_unreachable_is_reported_not_raised(self):
        def _boom(*_args, **_kwargs):
            raise OSError("no network")

        with mock.patch.object(hf_access.requests, "get", _boom):
            state, detail = hf_access.check_access("google/gemma-3-12b-it")
        self.assertEqual(state, hf_access.ACCESS_UNKNOWN)
        self.assertIn("Could not reach huggingface.co", detail)

    def test_the_gated_help_says_one_token_serves_every_model(self):
        # The user's question - "how do I make a token *for Gemma 3*?" - has no
        # answer in the token itself; it is answered by saying so.
        help_text = hf_access.gated_help("google/gemma-3-12b-it", "Gemma 3 12B")
        self.assertIn("One token serves every gated model", help_text)
        self.assertIn("Check access", help_text)

    def test_kyutais_own_voice_cloning_wording_is_recognised(self):
        # Pocket TTS's library says "accept the terms, then make sure you're
        # logged in locally with `uvx hf auth login`" - recognised here so the
        # Voice Lab can answer with the page and the application's own panel.
        self.assertTrue(hf_access.is_access_error(POCKET_CLONING_UNSUPPORTED))
        detail = hf_access.access_message(
            POCKET_CLONING_UNSUPPORTED, "kyutai/pocket-tts",
            "Pocket TTS voice cloning")
        self.assertIn("https://huggingface.co/kyutai/pocket-tts", detail)
        self.assertIn("Settings > Hugging Face account", detail)
        self.assertNotIn("hf auth login", detail)

    def test_a_disk_full_failure_is_not_dressed_up_as_a_licence(self):
        self.assertFalse(hf_access.is_access_error(
            "OSError: [Errno 28] No space left on device"))
        self.assertEqual(hf_access.access_message(
            "OSError: [Errno 28] No space left on device", "Org/Repo"), "")


class SettingsDefaultsTest(unittest.TestCase):
    """The keys exist in a fresh settings file, so nothing has to guess."""

    def test_a_new_settings_store_has_the_hugging_face_keys(self):
        folder = tempfile.mkdtemp(prefix="aivs_hf_defaults_")
        path = os.path.join(folder, "settings.json")
        settings = Settings(path=path)
        self.assertEqual(settings.get(hf_access.TOKEN_SETTING), "")
        self.assertEqual(settings.get(hf_access.ACCOUNT_SETTING), "")
        settings.save()
        with open(path, encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertIn("hf", written)


if __name__ == "__main__":
    unittest.main()
