"""Where a downloaded voice's files have to end up on disk.

The engine reads a voice from ``<variant>/<voice_id>``: piper keeps one folder
per voice, matcha and the Chinese VITS voice use a single ``default`` folder
(their variants ship one artifact each), and a variant-level artifact (kokoro,
kitten) belongs at the variant root because all of its speakers share one model.

Every build up to 2027.1.0 extracted *each* archive of a variant straight into
the variant folder, so the last voice's ``en_US-lessac-high.onnx`` and
``tokens.txt`` overwrote the ones before it: the per-voice folders stayed empty
and every voice of a multi-voice variant - piper medium/high, matcha, the
Chinese VITS voice - failed with "Model files are incomplete (missing model,
tokens)" while its files sat one folder above.  These tests pin down the layout,
the one shared ``espeak-ng-data`` that goes with it, and the repair of an
install that the old build left in that state (the download keeps its archives,
so the voice can be rebuilt from its own archive without re-downloading).

They also pin the corollary: since a repair is only ever made from an archive
that is really there, a voice whose download never finished is left alone and
still reports itself as incomplete.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.tts import downloader as downloader_mod
from ai_voice_studio.tts import engine as engine_mod
from ai_voice_studio.tts.downloader import (
    ModelDownloader,
    install_archive,
    repair_voice_layout,
    voice_for_artifact,
)
from ai_voice_studio.tts.models import ModelStore, resolve_voice_files


def make_archive(path: str, top: str, entries: dict) -> str:
    """Write a ``.tar.bz2`` shaped like the sherpa-onnx releases.

    Those archives hold a single top-level folder named after the archive
    (``vits-piper-en_US-lessac-medium/``); ``safe_extract_tarball`` strips it.
    """
    with tarfile.open(path, "w:bz2") as tar:
        for name, data in entries.items():
            payload = data.encode("utf-8")
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def piper_archive(folder: str, voice: str = "lessac", quality: str = "medium") -> str:
    """The archive of one piper voice, as the TTS-models release ships it."""
    top = f"vits-piper-en_US-{voice}-{quality}"
    return make_archive(
        os.path.join(folder, f"{top}.tar.bz2"),
        top,
        {
            f"en_US-{voice}-{quality}.onnx": "onnx-" + voice,
            f"en_US-{voice}-{quality}.onnx.json": json.dumps({"voice": voice}),
            "tokens.txt": "tokens",
            "MODEL_CARD": "card",
            "espeak-ng-data/en_dict": "dict",
        },
    )


#: A piper-shaped catalog entry whose artifacts are built by the tests, so the
#: downloader is exercised end to end without touching the network.
FAKE_PIPER = {
    "id": "piper",
    "name": "Piper",
    "engine": "vits",
    "artifact_base": "https://example.invalid/tts-models/",
    "languages": [
        {
            "code": "en_US",
            "variants": [
                {
                    "id": "medium",
                    "voices": [
                        {
                            "id": "lessac",
                            "name": "Lessac",
                            "artifact": "vits-piper-en_US-lessac-medium.tar.bz2",
                        },
                        {
                            "id": "amy",
                            "name": "Amy",
                            "artifact": "vits-piper-en_US-amy-medium.tar.bz2",
                        },
                    ],
                }
            ],
        }
    ],
}


class _ModelsDirCase(unittest.TestCase):
    """Every test gets its own models folder - never the user's own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aivs_layout_")
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch(
            "ai_voice_studio.paths.user_data_dir", return_value=self.tmp.name
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.models = os.path.join(self.tmp.name, "models")
        os.makedirs(self.models, exist_ok=True)

    def variant_dir(self, *parts: str, make: bool = True) -> str:
        d = os.path.join(self.models, *parts)
        if make:
            os.makedirs(d, exist_ok=True)
        return d


def entries_of(directory: str) -> set:
    return set(os.listdir(directory)) if os.path.isdir(directory) else set()


def staging_folders(directory: str) -> list:
    """The extraction folders a run left behind (there must be none)."""
    return [n for n in entries_of(directory) if n.startswith(".staging")]


class InstallArchiveTest(_ModelsDirCase):
    """One archive in, files in the folder the engine reads for that voice."""

    def test_a_voice_archive_lands_in_the_voice_folder(self):
        variants = self.variant_dir("piper", "en_US", "medium")
        archive = piper_archive(self.tmp.name, "lessac")

        target = install_archive(archive, variants, {"id": "lessac"})

        self.assertEqual(target, os.path.join(variants, "lessac"))
        self.assertTrue(
            os.path.isfile(os.path.join(variants, "lessac", "en_US-lessac-medium.onnx"))
        )
        self.assertTrue(os.path.isfile(os.path.join(variants, "lessac", "tokens.txt")))
        # The staging folder is never left behind for the next download to find.
        self.assertEqual(staging_folders(variants), [])

    def test_the_shared_espeak_data_is_installed_once(self):
        """Two voices, one espeak-ng-data - one per voice would double ~13 MB."""
        variants = self.variant_dir("piper", "en_US", "medium")

        install_archive(piper_archive(self.tmp.name, "lessac"), variants, {"id": "lessac"})
        install_archive(piper_archive(self.tmp.name, "amy"), variants, {"id": "amy"})

        shared = os.path.join(variants, "espeak-ng-data")
        self.assertTrue(os.path.isfile(os.path.join(shared, "en_dict")))
        # The second extraction must not bury the first copy one level deeper.
        self.assertFalse(os.path.isdir(os.path.join(shared, "espeak-ng-data")))
        self.assertFalse(os.path.isdir(os.path.join(variants, "lessac", "espeak-ng-data")))
        self.assertTrue(
            os.path.isfile(os.path.join(variants, "amy", "en_US-amy-medium.onnx"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(variants, "lessac", "en_US-lessac-medium.onnx"))
        )

    def test_extracting_the_same_archive_twice_replaces_it(self):
        variants = self.variant_dir("piper", "en_US", "medium")
        archive = piper_archive(self.tmp.name, "lessac")

        install_archive(archive, variants, {"id": "lessac"})
        install_archive(archive, variants, {"id": "lessac"})

        self.assertTrue(
            os.path.isfile(os.path.join(variants, "lessac", "en_US-lessac-medium.onnx"))
        )
        self.assertFalse(os.path.isdir(os.path.join(variants, "lessac", "lessac")))
        self.assertFalse(
            os.path.isdir(os.path.join(variants, "espeak-ng-data", "espeak-ng-data"))
        )

    def test_a_variant_artifact_stays_at_the_variant_root(self):
        """Kokoro and Kitten ship one model for all of their speakers."""
        variants = self.variant_dir("kokoro", "multi", "v1_0")
        archive = make_archive(
            os.path.join(self.tmp.name, "kokoro-multi-lang-v1_0.tar.bz2"),
            "kokoro-multi-lang-v1_0",
            {"model.onnx": "onnx", "tokens.txt": "tokens", "voices.bin": "voices"},
        )

        install_archive(archive, variants, None)

        self.assertTrue(os.path.isfile(os.path.join(variants, "model.onnx")))
        self.assertTrue(os.path.isfile(os.path.join(variants, "voices.bin")))

    def test_the_voice_an_artifact_belongs_to_is_found(self):
        variant = FAKE_PIPER["languages"][0]["variants"][0]
        self.assertEqual(
            voice_for_artifact(variant, "vits-piper-en_US-amy-medium.tar.bz2")["id"],
            "amy",
        )
        # An artifact on the variant itself belongs to no single voice.
        self.assertIsNone(voice_for_artifact({"artifact": "one.tar.bz2"}, "one.tar.bz2"))


class DownloadLayoutTest(_ModelsDirCase):
    """`download_variant` installs every voice where its engine looks for it."""

    def setUp(self):
        super().setUp()
        self.archives = os.path.join(self.tmp.name, "server")
        os.makedirs(self.archives, exist_ok=True)
        self.store = ModelStore(state_file=os.path.join(self.tmp.name, "models.json"))
        self.downloaded: list[str] = []

        real = piper_archive
        server = self.archives
        downloaded = self.downloaded

        def fake_download(url, dest_dir, filename=None, progress=None, cancel_event=None):
            """Serve the pre-built fixture archive of ``filename``."""
            downloaded.append(filename)
            # vits-piper-en_US-<voice>-medium.tar.bz2
            voice = os.path.basename(filename).split("-")[3]
            shutil.copyfile(real(server, voice), os.path.join(dest_dir, filename))
            return os.path.join(dest_dir, filename)

        patcher = mock.patch.object(downloader_mod, "download_file", fake_download)
        patcher.start()
        self.addCleanup(patcher.stop)
        catalog_patcher = mock.patch.object(
            downloader_mod.catalog, "find_tts", lambda tts_id: FAKE_PIPER
        )
        catalog_patcher.start()
        self.addCleanup(catalog_patcher.stop)

    def test_every_voice_of_the_variant_is_installed_in_its_own_folder(self):
        ModelDownloader(self.store).download_variant("piper", "en_US", "medium")

        variants = os.path.join(self.models, "piper", "en_US", "medium")
        for voice in ("lessac", "amy"):
            with self.subTest(voice=voice):
                folder = os.path.join(variants, voice)
                files = resolve_voice_files(
                    {
                        "dir": folder,
                        "engine": "vits",
                        "sid": 0,
                        "voice": voice,
                        "frontend": "piper",
                    }
                )
                self.assertTrue(files["model"].endswith(f"en_US-{voice}-medium.onnx"))
                self.assertTrue(files["tokens"].endswith("tokens.txt"))
                self.assertTrue(files["data_dir"].endswith("espeak-ng-data"))
        # The variant is complete, its archives are gone and the shared data
        # folder holds the only copy of espeak-ng-data.
        self.assertTrue(os.path.isfile(os.path.join(variants, ".aivs_complete")))
        self.assertEqual(
            [f for f in os.listdir(variants) if f.endswith(".tar.bz2")], []
        )
        self.assertEqual(staging_folders(variants), [])

    def test_an_archive_left_by_an_interrupted_run_is_not_fetched_again(self):
        variants = os.path.join(self.models, "piper", "en_US", "medium")
        os.makedirs(variants, exist_ok=True)
        shutil.copyfile(
            piper_archive(self.archives, "lessac"),
            os.path.join(variants, "vits-piper-en_US-lessac-medium.tar.bz2"),
        )

        ModelDownloader(self.store).download_variant("piper", "en_US", "medium")

        self.assertEqual(self.downloaded, ["vits-piper-en_US-amy-medium.tar.bz2"])
        self.assertTrue(
            os.path.isfile(os.path.join(variants, "lessac", "en_US-lessac-medium.onnx"))
        )


class RepairVoiceLayoutTest(_ModelsDirCase):
    """An install the old build left flat is repaired, not re-downloaded."""

    def legacy_variant(self, voice: str = "lessac", quality: str = "medium") -> str:
        """The state the old build left behind: loose files, archive still there."""
        variants = self.variant_dir("piper", "en_US", quality)
        for name, data in (
            (f"en_US-{voice}-{quality}.onnx", "onnx"),
            (f"en_US-{voice}-{quality}.onnx.json", "{}"),
            ("tokens.txt", "tokens"),
        ):
            with open(os.path.join(variants, name), "w", encoding="utf-8") as fh:
                fh.write(data)
        os.makedirs(os.path.join(variants, "espeak-ng-data"), exist_ok=True)
        with open(os.path.join(variants, "espeak-ng-data", "en_dict"), "w") as fh:
            fh.write("dict")
        # The old build kept every archive it extracted... except the download
        # of this one, which is the case this test is about.
        return variants

    def test_a_voice_left_loose_by_an_older_build_is_repaired(self):
        variants = self.legacy_variant()
        archive = piper_archive(self.tmp.name, "lessac")
        shutil.move(archive, os.path.join(variants, os.path.basename(archive)))
        entry = {
            "tts": "piper",
            "language": "en_US",
            "variant": "medium",
            "voice": "lessac",
            "engine": "vits",
            "sid": 0,
            "dir": os.path.join(variants, "lessac"),
            "artifact": "vits-piper-en_US-lessac-medium.tar.bz2",
        }
        # What the engine saw before the repair: no model, no tokens.
        broken = resolve_voice_files(entry)
        self.assertIsNone(broken["model"])
        self.assertIsNone(broken["tokens"])

        self.assertTrue(repair_voice_layout(entry))

        fixed = resolve_voice_files(entry)
        self.assertTrue(fixed["model"].endswith("en_US-lessac-medium.onnx"))
        self.assertTrue(fixed["tokens"].endswith("tokens.txt"))
        self.assertTrue(fixed["data_dir"].endswith("espeak-ng-data"))
        # The archive was consumed: the files it held are in place.
        self.assertFalse(
            os.path.exists(os.path.join(variants, "vits-piper-en_US-lessac-medium.tar.bz2"))
        )

    def test_a_voice_whose_archive_is_missing_is_left_alone(self):
        """A download that never finished is reported, never guessed at."""
        variants = self.legacy_variant()
        entry = {
            "tts": "piper",
            "language": "en_US",
            "variant": "medium",
            "voice": "lessac",
            "engine": "vits",
            "sid": 0,
            "dir": os.path.join(variants, "lessac"),
            "artifact": "vits-piper-en_US-lessac-medium.tar.bz2",
        }

        self.assertFalse(repair_voice_layout(entry))

        self.assertFalse(os.path.exists(os.path.join(variants, "lessac")))
        self.assertTrue(os.path.isfile(os.path.join(variants, "en_US-lessac-medium.onnx")))

    def test_a_custom_voice_folder_is_never_touched(self):
        custom = self.variant_dir("custom", "my piper voice")
        with open(os.path.join(custom, "tokens.txt"), "w", encoding="utf-8") as fh:
            fh.write("tokens")
        entry = {"dir": custom, "engine": "vits", "sid": 0}

        self.assertFalse(repair_voice_layout(entry))

        self.assertEqual(entries_of(custom), {"tokens.txt"})

    def test_an_installed_voice_is_left_where_it_is(self):
        voice_dir = self.variant_dir("piper", "en_US", "medium", "lessac")
        with open(os.path.join(voice_dir, "en_US-lessac-medium.onnx"), "w") as fh:
            fh.write("onnx")
        entry = {
            "tts": "piper",
            "language": "en_US",
            "variant": "medium",
            "voice": "lessac",
            "engine": "vits",
            "sid": 0,
            "dir": voice_dir,
            "artifact": "vits-piper-en_US-lessac-medium.tar.bz2",
        }

        self.assertTrue(repair_voice_layout(entry))

        self.assertEqual(entries_of(voice_dir), {"en_US-lessac-medium.onnx"})

    def test_a_voice_the_catalog_does_not_know_is_left_alone(self):
        voice_dir = self.variant_dir("piper", "en_US", "medium", "nosuchvoice")
        entry = {
            "tts": "piper",
            "language": "en_US",
            "variant": "medium",
            "voice": "nosuchvoice",
            "engine": "vits",
            "sid": 0,
            "dir": voice_dir,
            "artifact": "vits-piper-en_US-nosuchvoice-medium.tar.bz2",
        }

        self.assertFalse(repair_voice_layout(entry))


class BorrowedEspeakDataTest(_ModelsDirCase):
    """Where a voice without its own espeak-ng-data finds one."""

    def test_the_variant_folder_copy_is_borrowed(self):
        variants = self.variant_dir("piper", "en_US", "medium")
        folder = os.path.join(variants, "lessac")
        os.makedirs(folder, exist_ok=True)
        for name in ("en_US-lessac-medium.onnx", "tokens.txt"):
            open(os.path.join(folder, name), "w").close()
        os.makedirs(os.path.join(variants, "espeak-ng-data"), exist_ok=True)

        files = resolve_voice_files({"dir": folder, "engine": "vits", "sid": 0})

        self.assertEqual(files["data_dir"], os.path.join(variants, "espeak-ng-data"))

    def test_a_voice_with_its_own_copy_keeps_it(self):
        variants = self.variant_dir("piper", "en_US", "medium")
        folder = os.path.join(variants, "lessac")
        os.makedirs(os.path.join(folder, "espeak-ng-data"), exist_ok=True)
        for name in ("en_US-lessac-medium.onnx", "tokens.txt"):
            open(os.path.join(folder, name), "w").close()
        os.makedirs(os.path.join(variants, "espeak-ng-data"), exist_ok=True)

        files = resolve_voice_files({"dir": folder, "engine": "vits", "sid": 0})

        self.assertEqual(files["data_dir"], os.path.join(folder, "espeak-ng-data"))

    def test_a_lexicon_model_still_gets_none(self):
        """sherpa-onnx would switch to the piper frontend and crash on it."""
        variants = self.variant_dir("vits_zh", "zh", "aishell3")
        folder = os.path.join(variants, "aishell3")
        os.makedirs(folder, exist_ok=True)
        for name in ("model.onnx", "tokens.txt", "lexicon.txt"):
            open(os.path.join(folder, name), "w").close()
        os.makedirs(os.path.join(variants, "espeak-ng-data"), exist_ok=True)

        files = resolve_voice_files({"dir": folder, "engine": "vits", "sid": 0})

        self.assertEqual(files["data_dir"], "")

    def test_the_application_wide_copy_is_the_last_resort(self):
        folder = self.variant_dir("custom", "my piper voice")
        for name in ("model.onnx", "tokens.txt"):
            open(os.path.join(folder, name), "w").close()
        shared = self.variant_dir("shared", "espeak-ng-data")

        files = resolve_voice_files({"dir": folder, "engine": "vits", "sid": 0})

        self.assertEqual(files["data_dir"], shared)


class _FakeSherpa:
    """Enough of sherpa-onnx to build a config and hand back an object."""

    sample_rate = 22050

    class _Config:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def __getattr__(self, name):
        return self._Config


class EngineRepairTest(_ModelsDirCase):
    """The voice starts working instead of asking for a re-download."""

    def voice_entry(self, folder: str) -> dict:
        return {
            "tts": "piper",
            "language": "en_US",
            "variant": "medium",
            "voice": "lessac",
            "engine": "vits",
            "sid": 0,
            "dir": folder,
            "artifact": "vits-piper-en_US-lessac-medium.tar.bz2",
        }

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(engine_mod, "_import_sherpa", _FakeSherpa)
        patcher.start()
        self.addCleanup(patcher.stop)

    def flat_install(self) -> dict:
        variants = self.variant_dir("piper", "en_US", "medium")
        for name in ("en_US-lessac-medium.onnx", "tokens.txt"):
            open(os.path.join(variants, name), "w").close()
        os.makedirs(os.path.join(variants, "espeak-ng-data"), exist_ok=True)
        shutil.move(
            piper_archive(self.tmp.name, "lessac"),
            os.path.join(variants, "vits-piper-en_US-lessac-medium.tar.bz2"),
        )
        return self.voice_entry(os.path.join(variants, "lessac"))

    def test_the_engine_repairs_a_flat_install_and_speaks(self):
        entry = self.flat_install()

        engine = engine_mod.TtsEngine(entry)

        self.assertTrue(engine.files["model"].endswith("en_US-lessac-medium.onnx"))
        self.assertTrue(os.path.isfile(engine.files["model"]))
        self.assertEqual(engine.sample_rate, 22050)

    def test_without_an_archive_the_error_is_still_raised(self):
        variants = self.variant_dir("piper", "en_US", "medium")
        for name in ("en_US-lessac-medium.onnx", "tokens.txt"):
            open(os.path.join(variants, name), "w").close()
        entry = self.voice_entry(os.path.join(variants, "lessac"))

        with self.assertRaises(engine_mod.EngineUnavailableError) as caught:
            engine_mod.TtsEngine(entry)

        self.assertIn("Model files are incomplete (missing model, tokens)", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
