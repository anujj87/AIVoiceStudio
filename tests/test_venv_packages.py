"""Tests for the cached managed-venv package probe (venv_packages).

The managed virtualenv is replaced by a fake runtime, so no Python interpreter
is ever started: these tests cover the cache semantics the GUI relies on
(single background probe, instant answers, explicit invalidation).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import python_runtime as pr  # noqa: E402
from ai_voice_studio import venv_packages as vp  # noqa: E402
from ai_voice_studio.tts import catalog  # noqa: E402


class _Result:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


class _FakeRuntime:
    """Stands in for PythonRuntime: records every probe it is asked to run."""

    def __init__(self, installed: dict, created: bool = True):
        self.installed = installed
        self._created = created
        self.calls: list = []
        self.lock = threading.Lock()

    @property
    def is_created(self) -> bool:
        return self._created

    def run_in_env(self, script: str) -> _Result:
        with self.lock:
            self.calls.append(script)
        for name, version in self.installed.items():
            if repr(name) in script:
                return _Result(version + "\n")
        return _Result("")


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class _CacheMixin(unittest.TestCase):
    def setUp(self):
        self._clear()
        self.addCleanup(self._clear)

    def _clear(self):
        with vp._lock:
            vp._cache.clear()
            vp._disk_cache.clear()
            vp._listeners.clear()
            vp._loading.clear()


class VersionTest(_CacheMixin):
    def test_unknown_package_returns_none_then_the_version(self):
        runtime = _FakeRuntime({"omnivoice-triton": "0.1.0"})
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            self.assertIsNone(vp.version("omnivoice-triton"))
            self.assertTrue(_wait_for(lambda: vp.version("omnivoice-triton")))
            self.assertEqual(vp.version("omnivoice-triton"), "0.1.0")
        self.assertEqual(len(runtime.calls), 1, "the probe ran more than once")

    def test_not_installed_package_caches_none_and_does_not_reprobe(self):
        runtime = _FakeRuntime({})
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            vp.version("missing-package")
            self.assertTrue(_wait_for(lambda: vp.is_known("missing-package")))
            self.assertIsNone(vp.version("missing-package"))
            self.assertFalse(vp.installed("missing-package"))
        self.assertEqual(len(runtime.calls), 1)

    def test_missing_virtualenv_is_not_probed(self):
        runtime = _FakeRuntime({}, created=False)
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            self.assertIsNone(vp.version("omnivoice-triton"))
            self.assertTrue(vp.is_known("omnivoice-triton"))
        self.assertEqual(runtime.calls, [])

    def test_request_notifies_listeners_once(self):
        runtime = _FakeRuntime({"omnivoice-server": "0.2.5"})
        seen: list = []
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            vp.request("omnivoice-server", lambda value: seen.append(value))
            self.assertTrue(_wait_for(lambda: seen))
        self.assertEqual(seen, ["0.2.5"])

    def test_request_on_a_cached_answer_notifies_immediately(self):
        runtime = _FakeRuntime({"omnivoice-server": "0.2.5"})
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            vp.request("omnivoice-server")
            self.assertTrue(_wait_for(lambda: vp.is_known("omnivoice-server")))
            seen: list = []
            vp.request("omnivoice-server", lambda value: seen.append(value))
        self.assertEqual(seen, ["0.2.5"])

    def test_invalidate_forgets_the_answer(self):
        runtime = _FakeRuntime({"omnivoice-triton": "0.1.0"})
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            vp.version("omnivoice-triton")
            self.assertTrue(_wait_for(lambda: vp.is_known("omnivoice-triton")))
            vp.invalidate("omnivoice-triton")
            self.assertFalse(vp.is_known("omnivoice-triton"))
        # After invalidation exactly one more probe runs.
        self.assertEqual(len(runtime.calls), 1)

    def test_concurrent_callers_share_one_probe(self):
        started = threading.Event()
        release = threading.Event()

        class _SlowRuntime(_FakeRuntime):
            def run_in_env(self, script):
                started.set()
                release.wait(10)
                return super().run_in_env(script)

        runtime = _SlowRuntime({"omnivoice-triton": "0.1.0"})
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            for _ in range(10):
                vp.version("omnivoice-triton")
            self.assertTrue(started.wait(10))
            for _ in range(10):
                vp.version("omnivoice-triton")
            release.set()
            self.assertTrue(_wait_for(lambda: vp.version("omnivoice-triton")))
        self.assertEqual(len(runtime.calls), 1)

class DiskProbeTest(_CacheMixin):
    """The first answer comes from the environment on disk.

    Regression: ``version()`` used to answer ``None`` until the background
    probe had finished, so "is this engine installed?" was answered "no" on a
    cold cache - and the worker was then started in the wrong environment.
    """

    def _runtime(self, env_dir, packages, created=True):
        if created:
            site = os.path.join(env_dir, "Lib", "site-packages")
            os.makedirs(site)
        for package in packages:
            stem = package.replace("-", "_")
            os.makedirs(os.path.join(site, stem))
            os.makedirs(os.path.join(site, f"{stem}-1.0.dist-info"))
        os.makedirs(os.path.join(env_dir, "Scripts"), exist_ok=True)
        with open(os.path.join(env_dir, "Scripts", "python.exe"), "w") as fh:
            fh.write("")
        return pr.PythonRuntime(env_dir)

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime = self._runtime(
            os.path.join(self.tmp.name, "engine_env"),
            ["torch", "transformers", "f5-tts"],
        )
        patcher = mock.patch("ai_voice_studio.python_runtime.get_runtime",
                             return_value=self.runtime)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_cold_cache_still_answers_from_the_disk(self):
        self.assertFalse(vp.is_known("torch"))
        self.assertEqual(vp.version("torch"), "1.0")
        self.assertTrue(vp.installed("torch"))

    def test_the_name_is_normalised_like_pip_does(self):
        # The distribution is ``f5_tts-1.0.dist-info`` but the package the app
        # asks about is ``f5-tts``.
        self.assertTrue(vp.installed("f5-tts"))
        self.assertTrue(vp.package_present("f5_tts"))
        self.assertTrue(vp.package_present("F5.TTS"))

    def test_package_present_is_synchronous_and_needs_no_probe(self):
        self.assertTrue(vp.package_present("transformers"))
        self.assertFalse(vp.package_present("numpy"))
        self.assertFalse(vp.package_present(""))

    def test_a_missing_environment_reports_nothing(self):
        missing = pr.PythonRuntime(os.path.join(self.tmp.name, "nope"))
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=missing):
            self.assertFalse(vp.package_present("torch"))
            self.assertIsNone(vp.version("torch"))

    def test_the_disk_answer_does_not_replace_a_probe_answer(self):
        # The disk listing is only the *first* answer: once the probe has run,
        # its answer (which also covers hand-patched environments) wins.
        with vp._lock:
            vp._cache[vp._key("torch", None)] = (time.time(), "9.9.9")
        self.assertEqual(vp.version("torch"), "9.9.9")

    def test_the_disk_answer_does_not_stop_the_probe(self):
        self.assertTrue(vp.installed("torch"))
        self.assertFalse(vp.is_known("torch"),
                         "the panel must still refresh from the probe")

    def test_invalidate_forgets_the_disk_listing_too(self):
        self.assertTrue(vp.package_present("torch"))
        vp._disk_cache["sentinel"] = (0.0, {})
        vp.invalidate("torch")
        self.assertEqual(vp._disk_cache, {})


class WarmTest(_CacheMixin):
    def test_warm_probes_the_catalog_packages(self):
        runtime = _FakeRuntime({"omnivoice-triton": "0.1.0",
                                "omnivoice-server": "0.2.5",
                                "pocket-tts": "1.0.0",
                                "transformers": "4.44.0",
                                "f5-tts": "1.1.0"})
        with mock.patch("ai_voice_studio.python_runtime.get_runtime",
                        return_value=runtime):
            vp.warm()
            # Every engine whose package the catalog lists is probed once:
            # the two OmniVoice engines plus the Voice Lab engines.
            packages = {
                tts["requires_package"]
                for tts in catalog.get_tts_list()
                if tts.get("requires_package")
            }
            self.assertTrue(_wait_for(
                lambda: all(vp.installed(pkg) for pkg in packages)
            ), packages)
        # One probe per package, and nothing else was run.
        self.assertEqual(len(runtime.calls), len(packages))
        probed = " ".join(runtime.calls)
        for package in packages:
            self.assertIn(repr(package), probed, package)


if __name__ == "__main__":
    unittest.main()
