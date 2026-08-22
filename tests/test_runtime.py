"""Tests for the optional GPU (CUDA) runtime download/activation."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import runtime  # noqa: E402


def _make_tarball(path: str) -> None:
    files = {
        "sherpa-onnx-cuda-win-x64/lib/onnxruntime.dll": b"ORT",
        "sherpa-onnx-cuda-win-x64/lib/onnxruntime_providers_cuda.dll": b"EP",
        "sherpa-onnx-cuda-win-x64/lib/onnxruntime_providers_shared.dll": b"SHARED",
        "sherpa-onnx-cuda-win-x64/lib/skip-me.dll": b"SKIP",
    }
    with tarfile.open(path, "w:bz2") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _make_wheel(path: str, prefix: str, names) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(f"{prefix}/{name}", b"DATA")


class RuntimeInstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime_dir = os.path.join(self.tmp.name, "runtime", "cuda")

        tarball = os.path.join(self.tmp.name, "cuda.tar.bz2")
        _make_tarball(tarball)
        self.artifacts = {"TARBALL": tarball}

        wheel_specs = [
            ("nvidia-cudnn-cu12", "nvidia/cudnn/bin",
             ["cudnn64_9.dll", "cudnn_ops64_9.dll"]),
            ("nvidia-cublas-cu12", "nvidia/cublas/bin",
             ["cublas64_12.dll", "cublasLt64_12.dll"]),
            ("nvidia-cufft-cu12", "nvidia/cufft/bin", ["cufft64_11.dll"]),
            ("nvidia-cuda-runtime-cu12", "nvidia/cuda_runtime/bin",
             ["cudart64_12.dll"]),
        ]
        for pkg, folder, names in wheel_specs:
            path = os.path.join(self.tmp.name, f"{pkg}.whl")
            _make_wheel(path, folder, names)
            self.artifacts[pkg] = path

        # Point the installer at the local artifacts.
        self.patches = [
            mock.patch.object(runtime, "runtime_dir", return_value=self.runtime_dir),
            mock.patch.object(runtime, "_CUDA_SHERPA_URL", "TARBALL"),
            mock.patch.object(runtime, "_wheel_url", side_effect=lambda pkg, _v: pkg),
            mock.patch.object(
                runtime, "download_file", side_effect=self._fake_download
            ),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def _fake_download(self, url, dest_dir, filename=None, progress=None,
                       cancel_event=None):
        src = self.artifacts[url]
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename or os.path.basename(src))
        shutil.copyfile(src, dest)
        return dest

    def test_install_extracts_expected_files(self):
        runtime.install()
        self.assertTrue(runtime.is_installed())
        for name in (
            "onnxruntime.dll",
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_shared.dll",
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cufft64_11.dll",
            "cudart64_12.dll",
            "installed.json",
        ):
            self.assertTrue(
                os.path.isfile(os.path.join(self.runtime_dir, name)),
                f"missing {name}",
            )
        # The unrequested member must not be copied.
        self.assertFalse(os.path.isfile(os.path.join(self.runtime_dir, "skip-me.dll")))
        self.assertIsNotNone(runtime.installed_version())

    def test_install_is_idempotent(self):
        runtime.install()
        runtime.install()  # second call returns early without re-downloading
        self.assertTrue(runtime.is_installed())

    def test_install_can_be_cancelled(self):
        import threading

        cancel = threading.Event()

        def cancelling(url, dest_dir, filename=None, progress=None,
                       cancel_event=None):
            cancel.set()
            from ai_voice_studio.tts.downloader import DownloadCancelled
            raise DownloadCancelled()

        with mock.patch.object(runtime, "download_file", side_effect=cancelling):
            from ai_voice_studio.tts.downloader import DownloadCancelled
            with self.assertRaises(DownloadCancelled):
                runtime.install(cancel_event=cancel)
        self.assertFalse(runtime.is_installed())

    def test_remove_deletes_the_folder(self):
        runtime.install()
        runtime.remove()
        self.assertFalse(runtime.is_installed())
        self.assertFalse(os.path.isdir(self.runtime_dir))

    def test_activate_returns_false_when_not_installed(self):
        self.assertFalse(runtime.activate())

    def test_activate_fails_gracefully_on_broken_runtime(self):
        # A fake onnxruntime.dll cannot be loaded; activate() must return
        # False instead of raising.
        runtime.install()
        with open(os.path.join(self.runtime_dir, "onnxruntime.dll"), "wb") as fh:
            fh.write(b"not a dll")
        self.assertFalse(runtime.activate())


if __name__ == "__main__":
    unittest.main()
