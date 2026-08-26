"""Tests for the model catalog and installed-model file resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from ai_voice_studio.tts import catalog
from ai_voice_studio.tts.models import ModelStore, resolve_voice_files


class CatalogStructureTest(unittest.TestCase):
    def test_tts_ids_unique(self):
        ids = [t["id"] for t in catalog.get_tts_list()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_tts_present(self):
        ids = {t["id"] for t in catalog.get_tts_list()}
        for expected in ("piper", "kokoro", "matcha", "vits_zh"):
            self.assertIn(expected, ids, f"missing catalog entry {expected}")
        # Non-commercial TTS engines must be removed
        for removed in ("vits_mms", "pocket-tts", "coqui"):
            self.assertNotIn(removed, ids, f"non-commercial entry {removed} should be removed")

    def test_every_voice_has_id_and_name(self):
        for tts in catalog.get_tts_list():
            for lang in tts.get("languages", []):
                for variant in lang.get("variants", []):
                    for voice in variant.get("voices", []):
                        self.assertTrue(voice["id"], (tts["id"], lang["code"], variant["id"]))
                        self.assertTrue(voice.get("name"), (tts["id"], lang["code"], variant["id"]))

    def test_every_downloadable_variant_has_an_artifact(self):
        for tts in catalog.get_tts_list():
            for lang in tts.get("languages", []):
                for variant in lang.get("variants", []):
                    if variant.get("files") or variant.get("hf_files") or variant.get("hf_repo"):
                        continue  # raw-file variant or HuggingFace variant
                    # pip-installed variants (e.g. omnivoice-triton) have no artifact.
                    if tts.get("requires_package"):
                        continue
                    artifacts = []
                    if variant.get("artifact"):
                        artifacts.append(variant["artifact"])
                    artifacts += [v["artifact"] for v in variant.get("voices", []) if v.get("artifact")]
                    self.assertTrue(artifacts, (tts["id"], lang["code"], variant["id"]))
                    # Piper/kokoro variants may put the artifact on the variant;
                    # per-voice artifacts must match the voice it belongs to.
                    for voice in variant.get("voices", []):
                        if voice.get("artifact"):
                            self.assertTrue(
                                voice["artifact"].endswith(".tar.bz2")
                                or voice["artifact"].endswith(".onnx"),
                                (tts["id"], voice["id"], voice["artifact"]),
                            )

    def test_shared_artifacts_shape(self):
        matcha = catalog.find_tts("matcha")
        self.assertEqual(
            matcha["shared_artifacts"][0]["artifact"], "vocos-22khz-univ.onnx"
        )
        self.assertEqual(matcha["shared_artifacts"][0]["kind"], "file")
        self.assertEqual(matcha["shared_artifacts"][0]["dest"], "shared/vocoders")

    def test_matcha_voices_carry_artifact(self):
        matcha = catalog.find_tts("matcha")
        for lang in matcha["languages"]:
            for variant in lang["variants"]:
                for voice in variant["voices"]:
                    self.assertTrue(voice["artifact"].startswith("matcha-"), voice["artifact"])

    def test_kokoro_multilang_variants(self):
        kokoro = catalog.find_tts("kokoro")
        expected = {
            "v1_0": ("kokoro-multi-lang-v1_0.tar.bz2", 53),
            "int8_v1_0": ("kokoro-int8-multi-lang-v1_0.tar.bz2", 53),
            "v1_1": ("kokoro-multi-lang-v1_1.tar.bz2", 103),
            "int8_v1_1": ("kokoro-int8-multi-lang-v1_1.tar.bz2", 103),
        }
        for variant_id, (artifact, count) in expected.items():
            variant = catalog.find_variant(kokoro, "multi", variant_id)
            self.assertIsNotNone(variant, variant_id)
            self.assertEqual(variant["artifact"], artifact, variant_id)
            sids = [v["sid"] for v in variant["voices"]]
            self.assertEqual(sids, list(range(count)), f"{variant_id} has sids 0..{count - 1}")

    def test_kokoro_v0_19_has_only_11_english_speakers(self):
        """Regression: kokoro-en-v0_19 contains 11 speakers (sids 0-10).

        The catalog used to list 38 voices (Hindi, French, Chinese, ...) for
        this English-only artifact; selecting any of them made sherpa-onnx
        silently fall back to sid=0 (the first US English voice).
        """
        kokoro = catalog.find_tts("kokoro")
        variant = catalog.find_variant(kokoro, "en", "v0_19")
        self.assertIsNotNone(variant)
        voices = variant["voices"]
        self.assertEqual(len(voices), 11)
        sids = [v["sid"] for v in voices]
        self.assertEqual(sids, list(range(11)))
        ids = {v["id"] for v in voices}
        self.assertEqual(
            ids,
            {"af", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam",
             "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"},
        )

    def test_kokoro_multilang_contains_hindi_voices(self):
        """The multilingual v1_0 model must expose the language voices
        (Hindi, French, Chinese, Japanese, Italian, Portuguese) that the
        English-only v0_19 artifact cannot provide."""
        kokoro = catalog.find_tts("kokoro")
        variant = catalog.find_variant(kokoro, "multi", "v1_0")
        self.assertIsNotNone(variant)
        ids = {v["id"] for v in variant["voices"]}
        for expected in ("hf_alpha", "hf_beta", "hm_omega", "hm_psi",
                         "ff_siwis", "zf_xiaobei", "jf_alpha", "pf_dora",
                         "if_sara"):
            self.assertIn(expected, ids, expected)

    def test_kokoro_v1_1_speaker_map_matches_official_docs(self):
        kokoro = catalog.find_tts("kokoro")
        variant = catalog.find_variant(kokoro, "multi", "v1_1")
        by_sid = {v["sid"]: v["id"] for v in variant["voices"]}
        spot = {
            0: "af_maple", 1: "af_sol", 2: "bf_vale", 3: "zf_001",
            11: "zf_017", 57: "zf_099", 58: "zm_009", 66: "zm_020",
            93: "zm_080", 102: "zm_100",
        }
        for sid, name in spot.items():
            self.assertEqual(by_sid[sid], name, f"sid {sid}")

    def test_kitten_entry(self):
        kitten = catalog.find_tts("kitten")
        self.assertIsNotNone(kitten)
        self.assertEqual(kitten["engine"], "kitten")
        self.assertEqual(kitten["license"], "Apache-2.0")
        expected = {
            "v0_8_nano_int8": "kitten-nano-en-v0_8-int8.tar.bz2",
            "v0_8_nano_fp32": "kitten-nano-en-v0_8-fp32.tar.bz2",
            "v0_8_micro": "kitten-micro-en-v0_8.tar.bz2",
            "v0_8_mini": "kitten-mini-en-v0_8.tar.bz2",
        }
        variants = kitten["languages"][0]["variants"]
        self.assertEqual({v["id"]: v["artifact"] for v in variants}, expected)
        # The official KittenTTS voice order (Bella, Jasper, Luna, Bruno,
        # Rosie, Hugo, Kiki, Leo) maps to sids 0..7.
        first = variants[0]
        self.assertEqual(
            [v["id"] for v in first["voices"]],
            ["bella", "jasper", "luna", "bruno", "rosie", "hugo", "kiki", "leo"],
        )
        self.assertEqual([v["sid"] for v in first["voices"]], list(range(8)))

    def test_piper_voices_have_artifacts(self):
        piper = catalog.find_tts("piper")
        self.assertIsNotNone(piper)
        self.assertTrue(piper["voice_cloning"])
        # Spot-check English voices
        en = catalog.find_language(piper, "en_US")
        self.assertIsNotNone(en)
        medium = catalog.find_variant(piper, "en_US", "medium")
        self.assertTrue(medium["voices"])

    def test_aishell3_entry(self):
        tts = catalog.find_tts("vits_zh")
        voice = catalog.find_voice(tts, "zh", "aishell3", "aishell3")
        self.assertIsNotNone(voice)
        self.assertEqual(voice["artifact"], "vits-icefall-zh-aishell3.tar.bz2")


class ResolveVoiceFilesTest(unittest.TestCase):
    """resolve_voice_files must find the right files per engine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_root = os.path.join(self.tmp.name, "models")
        # models.py calls paths.models_dir() via the paths module.
        patcher = mock.patch(
            "ai_voice_studio.paths.user_data_dir", return_value=self.tmp.name
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _voice_dir(self, name: str) -> str:
        d = os.path.join(self.models_root, name)
        os.makedirs(d, exist_ok=True)
        return d

    def test_matcha_finds_acoustic_model_and_shared_vocoder(self):
        vdir = self._voice_dir("matcha_en/default")
        for f in ("model-steps-3.onnx", "tokens.txt", "espeak-ng-data"):
            p = os.path.join(vdir, f)
            if f.endswith(".onnx") or f.endswith(".txt"):
                open(p, "w").close()
            else:
                os.makedirs(p)
        voc = self._voice_dir("shared/vocoders")
        open(os.path.join(voc, "vocos-22khz-univ.onnx"), "w").close()

        entry = {"dir": vdir, "engine": "matcha", "sid": 0}
        files = resolve_voice_files(entry)
        self.assertEqual(
            os.path.basename(files["model"]), "model-steps-3.onnx"
        )
        self.assertTrue(files["vocoder"].endswith("vocos-22khz-univ.onnx"))
        self.assertTrue(files["data_dir"].endswith("espeak-ng-data"))

    def test_matcha_lexicon_variant(self):
        vdir = self._voice_dir("matcha_zh/default")
        for f in ("model-steps-3.onnx", "tokens.txt", "lexicon.txt"):
            open(os.path.join(vdir, f), "w").close()
        # shared espeak-ng-data exists: a lexicon-based model must still get
        # data_dir="" (sherpa-onnx would otherwise use the piper frontend
        # and crash on its tokens.txt).
        shared = self._voice_dir("shared/espeak-ng-data")
        open(os.path.join(shared, "en_dict"), "w").close()
        entry = {"dir": vdir, "engine": "matcha", "sid": 0}
        files = resolve_voice_files(entry)
        self.assertTrue(files["lexicon"].endswith("lexicon.txt"))
        self.assertEqual(files["data_dir"], "")

    def test_matcha_missing_vocoder(self):
        vdir = self._voice_dir("matcha_no_voc/default")
        open(os.path.join(vdir, "model-steps-3.onnx"), "w").close()
        open(os.path.join(vdir, "tokens.txt"), "w").close()
        files = resolve_voice_files({"dir": vdir, "engine": "matcha", "sid": 0})
        self.assertEqual(files["vocoder"], "")

    def test_kokoro_multilang_joins_lexicons(self):
        vdir = self._voice_dir("kokoro_ml")
        for f in ("model.onnx", "tokens.txt", "voices.bin",
                  "lexicon-us-en.txt", "lexicon-gb-en.txt", "lexicon-zh.txt"):
            open(os.path.join(vdir, f), "w").close()
        os.makedirs(os.path.join(vdir, "espeak-ng-data"))
        files = resolve_voice_files({"dir": vdir, "engine": "kokoro", "sid": 12})
        self.assertTrue(files["voices_file"].endswith("voices.bin"))
        self.assertTrue(files["data_dir"].endswith("espeak-ng-data"))
        parts = files["lexicon"].split(",")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.endswith(".txt") for p in parts))

    def test_piper_uses_shared_espeak_data(self):
        vdir = self._voice_dir("piper/en_US/medium")
        open(os.path.join(vdir, "model.onnx"), "w").close()
        open(os.path.join(vdir, "tokens.txt"), "w").close()
        shared = self._voice_dir("shared/espeak-ng-data")
        open(os.path.join(shared, "en_dict"), "w").close()
        files = resolve_voice_files({"dir": vdir, "engine": "vits", "sid": 0})
        self.assertTrue(files["data_dir"].endswith("espeak-ng-data"))

    def test_aishell3_uses_lexicon_not_shared_data(self):
        vdir = self._voice_dir("vits_zh/aishell3")
        open(os.path.join(vdir, "model.onnx"), "w").close()
        open(os.path.join(vdir, "tokens.txt"), "w").close()
        open(os.path.join(vdir, "lexicon.txt"), "w").close()
        # Even when the shared espeak-ng-data exists (installed by MMS), the
        # lexicon-based AISHELL3 model must not receive it -- passing data_dir
        # hard-crashes sherpa-onnx on this model.
        shared = self._voice_dir("shared/espeak-ng-data")
        open(os.path.join(shared, "en_dict"), "w").close()
        files = resolve_voice_files({"dir": vdir, "engine": "vits", "sid": 0})
        self.assertTrue(files["lexicon"].endswith("lexicon.txt"))
        self.assertEqual(files["data_dir"], "")

    def test_installed_voices_sid_passthrough(self):
        store = ModelStore(state_file=os.path.join(self.tmp.name, "models.json"))
        tts = catalog.find_tts("kokoro")
        self.assertIsNotNone(catalog.find_variant(tts, "multi", "v1_0"))
        store.mark_installed(
            "kokoro", "multi", "v1_0",
            os.path.join(self.models_root, "kokoro", "multi", "v1_0"),
        )
        # The variant dir does not exist yet; create it so is_variant_installed
        # passes.
        os.makedirs(store.variant_dir("kokoro", "multi", "v1_0"), exist_ok=True)
        voices = store.installed_voices()
        self.assertTrue(voices)
        sids = sorted(v["sid"] for v in voices)
        self.assertEqual(sids[0], 0)
        self.assertEqual(sids[-1], 52)

    def test_kitten_installed_voices_and_files(self):
        store = ModelStore(state_file=os.path.join(self.tmp.name, "models2.json"))
        vdir = store.variant_dir("kitten", "en", "v0_8_nano_int8")
        for name in ("model.int8.onnx", "tokens.txt", "voices.bin"):
            open(os.path.join(vdir, name), "w").close()
        os.makedirs(os.path.join(vdir, "espeak-ng-data"), exist_ok=True)
        store.mark_installed("kitten", "en", "v0_8_nano_int8", vdir)

        voices = [v for v in store.installed_voices() if v["engine"] == "kitten"]
        self.assertEqual(len(voices), 8)
        self.assertEqual([v["voice"] for v in voices][0], "bella")

        files = resolve_voice_files(voices[0])
        self.assertEqual(files["engine"], "kitten")
        self.assertTrue(files["model"].endswith("model.int8.onnx"))
        self.assertTrue(files["voices_file"].endswith("voices.bin"))
        self.assertTrue(files["data_dir"].endswith("espeak-ng-data"))

    def test_piper_char_frontend_ignores_shared_espeak(self):
        """Character-frontend voices must not receive the
        shared espeak-ng-data."""
        vdir = self._voice_dir("some_vits/default")
        open(os.path.join(vdir, "model.onnx"), "w").close()
        open(os.path.join(vdir, "tokens.txt"), "w").close()
        shared = self._voice_dir("shared/espeak-ng-data")
        open(os.path.join(shared, "en_dict"), "w").close()
        files = resolve_voice_files(
            {"dir": vdir, "engine": "vits", "sid": 0, "frontend": "char"}
        )
        self.assertEqual(files["data_dir"], "")
        self.assertEqual(files["lexicon"], "")


if __name__ == "__main__":
    unittest.main()
