"""Speech-to-Text unit tests: catalog, output documents and engine helpers.

These are pure-Python (no wx): the GUI side is covered by the smoke tests,
which build the settings category and the wizard pages.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio.stt import catalog, engine, environments, output, worker


class CatalogTest(unittest.TestCase):
    def test_the_models_from_the_request_are_present(self):
        names = [m["name"] for m in catalog.models()]
        for expected in (
            "Whisper Large V3",
            "Whisper Small/Base (CPU)",
            "Qwen3-ASR",
            "NVIDIA Canary-Qwen",
            "NVIDIA Parakeet TDT",
            "Moonshine (CPU)",
            "SpeechBrain (CPU)",
            "Voxtral",
            "IBM Granite Speech (CPU)",
        ):
            self.assertIn(expected, names)

    def test_no_model_needs_a_login_or_a_token(self):
        # A repository that is gated (Cohere Transcribe was one: its
        # config.json answered 401 without accepting a licence and signing in)
        # cannot be downloaded by a user who has no Hugging Face account, so
        # the catalog only holds repositories anyone can fetch.
        for model_id in catalog.model_ids():
            self.assertFalse(catalog.needs_remote_code(model_id), model_id)
            for package in catalog.pip_packages(model_id):
                self.assertNotIn("cohere", package.lower(), model_id)
        self.assertNotIn("cohere_transcribe", catalog.model_ids())

    def test_every_model_declares_the_device_it_prefers(self):
        for entry in catalog.models():
            self.assertIn(entry["fit"], ("cpu", "gpu"), entry["id"])
            for variant, _label in entry["variants"]:
                self.assertIn(catalog.fit(entry["id"], variant), ("cpu", "gpu"),
                              f"{entry['id']}/{variant}")

    def _all_choices(self):
        for model_id in catalog.model_ids():
            for variant, _label in catalog.variants(model_id):
                yield model_id, variant

    def test_a_model_that_can_run_on_a_cpu_offers_both_devices(self):
        # Many users have no NVIDIA card at all, so every model that can run
        # on a CPU must still be offered the GPU when one is present - and be
        # offered the CPU when it is not.
        for model_id, variant in self._all_choices():
            if catalog.gpu_only(model_id, variant):
                continue
            with_cuda = [v for v, _ in catalog.device_options(
                model_id, variant, True)]
            self.assertEqual(with_cuda, ["cpu", "cuda"] if
                             catalog.fit(model_id, variant) == "cpu"
                             else ["cuda", "cpu"],
                             f"{model_id}/{variant}")
            without_cuda = [v for v, _ in catalog.device_options(
                model_id, variant, False)]
            self.assertEqual(without_cuda, ["cpu"], f"{model_id}/{variant}")

    def test_only_a_model_that_needs_a_gpu_is_offered_the_gpu_alone(self):
        for model_id, variant in self._all_choices():
            values = [v for v, _ in catalog.device_options(
                model_id, variant, True)]
            if catalog.gpu_only(model_id, variant):
                self.assertEqual(values, ["cuda"], f"{model_id}/{variant}")
            else:
                self.assertIn("cpu", values, f"{model_id}/{variant}")

    def test_a_gpu_only_model_says_a_gpu_is_needed_without_a_card(self):
        for model_id, variant in self._all_choices():
            if not catalog.gpu_only(model_id, variant):
                continue
            options = catalog.device_options(model_id, variant, False)
            self.assertEqual([v for v, _ in options], ["cuda"],
                             f"{model_id}/{variant}")
            self.assertIn("no NVIDIA GPU", options[0][1])

    def test_the_preferred_device_is_offered_first(self):
        for model_id, variant in self._all_choices():
            if catalog.gpu_only(model_id, variant):
                continue
            options = catalog.device_options(model_id, variant, True)
            expected = ("cuda" if catalog.fit(model_id, variant) == "gpu"
                        else "cpu")
            self.assertEqual(options[0][0], expected, f"{model_id}/{variant}")

    def test_no_model_is_offered_an_automatic_device(self):
        for model_id, variant in self._all_choices():
            for has_cuda in (False, True):
                values = [v for v, _ in catalog.device_options(
                    model_id, variant, has_cuda)]
                self.assertNotIn("auto", values, f"{model_id}/{variant}")

    def test_the_gpu_only_choices_are_the_speech_language_models(self):
        # Canary-Qwen 2.5B (a SALM) and Granite Speech 8B (~17 GB of weights)
        # are the two checkpoints that genuinely cannot run on a CPU.
        only = [
            (model_id, variant)
            for model_id, variant in self._all_choices()
            if catalog.gpu_only(model_id, variant)
        ]
        self.assertEqual(sorted(only), [("canary_qwen", "2.5b"),
                                        ("granite_speech", "8b")])

    def test_canary_qwen_1b_is_the_cpu_friendly_variant(self):
        self.assertTrue(catalog.gpu_only("canary_qwen", "2.5b"))
        self.assertFalse(catalog.gpu_only("canary_qwen", "1b"))
        self.assertEqual(catalog.fit("canary_qwen", "1b"), "cpu")
        self.assertEqual(catalog.fit("canary_qwen", "2.5b"), "gpu")
        # The 1B is offered both devices, the 2.5B the GPU alone.
        self.assertEqual(
            [v for v, _ in catalog.device_options("canary_qwen", "1b", True)],
            ["cpu", "cuda"],
        )
        self.assertEqual(
            [v for v, _ in catalog.device_options("canary_qwen", "2.5b", True)],
            ["cuda"],
        )

    def test_the_gpu_preferred_models_are_the_gpu_ones(self):
        self.assertEqual(
            sorted(catalog.gpu_fit_model_ids()),
            sorted({"whisper_large_v3", "qwen3_asr", "canary_qwen",
                    "parakeet_tdt", "voxtral"}),
        )

    def test_voxtral_only_offers_a_model_that_fits_an_ordinary_card(self):
        # The 24B checkpoint is ~48 GB of bf16 weights: it cannot load on a
        # 12 GB card at all, so the catalog must not offer it.
        values = [v for v, _ in catalog.variants("voxtral")]
        self.assertEqual(values, ["mini-3b"])

    def test_model_ids_are_unique(self):
        ids = catalog.model_ids()
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_model_has_the_table_columns(self):
        for entry in catalog.models():
            for key in ("name", "license", "languages_label", "fit_label", "notes"):
                self.assertTrue(entry.get(key), f"{entry['id']} missing {key}")
            self.assertTrue(entry["variants"], entry["id"])
            self.assertTrue(entry["languages"], entry["id"])

    def test_default_variant_is_one_of_the_variants(self):
        for model_id in catalog.model_ids():
            values = [v for v, _ in catalog.variants(model_id)]
            self.assertIn(catalog.default_variant(model_id), values, model_id)

    def test_summary_rows_match_the_table(self):
        rows = catalog.summary_rows()
        self.assertEqual(len(rows), len(catalog.models()))
        self.assertTrue(all(len(row) == 5 for row in rows))

    def test_translation_targets_exist_only_when_supported(self):
        for model_id in catalog.model_ids():
            targets = catalog.translation_targets(model_id)
            if catalog.supports_translation(model_id):
                self.assertTrue(targets, model_id)
            else:
                self.assertEqual(targets, [], model_id)


class OutputTest(unittest.TestCase):
    SEGMENTS = [
        {"start": 0.0, "end": 2.5, "text": "Hello world."},
        {"start": 2.5, "end": 5.0, "text": "Second line."},
    ]

    def test_timestamp_formats(self):
        self.assertEqual(output.format_timestamp(0), "00:00:00,000")
        self.assertEqual(output.format_timestamp(3661.5), "01:01:01,500")
        self.assertEqual(
            output.format_timestamp(1.25, millisecond_sep="."), "00:00:01.250"
        )

    def test_txt_plain_and_timestamped(self):
        plain = output.format_document(self.SEGMENTS, "txt")
        self.assertIn("Hello world.", plain)
        self.assertNotIn("[", plain)
        stamped = output.format_document(self.SEGMENTS, "txt", include_timestamp=True)
        self.assertIn("[0:00] Hello world.", stamped)

    def test_srt_and_vtt_always_carry_timing(self):
        srt = output.format_document(self.SEGMENTS, "srt")
        self.assertIn("1\n00:00:00,000 --> 00:00:02,500\nHello world.", srt)
        vtt = output.format_document(self.SEGMENTS, "vtt")
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:02.500", vtt)

    def test_html_escapes_text(self):
        html = output.format_document(
            [{"start": 0, "end": 1, "text": "a < b & c"}], "html",
        )
        self.assertIn("a &lt; b &amp; c", html)

    def test_json_and_csv_parse(self):
        payload = json.loads(output.format_document(self.SEGMENTS, "json"))
        self.assertEqual(len(payload["segments"]), 2)
        csv_text = output.format_document(self.SEGMENTS, "csv",
                                          include_timestamp=True)
        self.assertIn("start,end,text", csv_text)

    def test_split_by_time_groups_without_cutting_a_segment(self):
        segments = [
            {"start": 0.0, "end": 300.0, "text": "a"},
            {"start": 300.0, "end": 600.0, "text": "b"},
            {"start": 600.0, "end": 601.0, "text": "c"},
        ]
        chunks = output.split_by_time(segments, 10)
        # 10 minutes = 600 s: 'a' and 'b' fill the first chunk, 'c' starts one.
        self.assertEqual(len(chunks), 2)
        self.assertEqual([s["text"] for s in chunks[0]], ["a", "b"])
        self.assertEqual([s["text"] for s in chunks[1]], ["c"])

    def test_plan_documents_names_parts_like_lecture_one(self):
        segments = [
            {"start": 0.0, "end": 600.0, "text": "a"},
            {"start": 600.0, "end": 1200.0, "text": "b"},
        ]
        source = os.path.join("audio", "lecture 1.wav")
        plans = output.plan_documents(
            [source], "by_time", 10,
            segments_by_source={source: segments},
        )
        self.assertEqual([p["stem"] for p in plans],
                         ["lecture 1 part 1", "lecture 1 part 2"])

    def test_scan_source_finds_media_files(self):
        folder = tempfile.mkdtemp(prefix="aivs_stt_")
        self.addCleanup(shutil.rmtree, folder, True)
        with open(os.path.join(folder, "a.wav"), "wb") as fh:
            fh.write(b"RIFF")
        with open(os.path.join(folder, "b.txt"), "w", encoding="utf-8") as fh:
            fh.write("not media")
        sub = os.path.join(folder, "sub")
        os.makedirs(sub)
        with open(os.path.join(sub, "c.mp3"), "wb") as fh:
            fh.write(b"ID3")
        found = output.scan_source(folder)
        self.assertEqual([os.path.basename(p) for p in found], ["a.wav", "c.mp3"])


class EngineTest(unittest.TestCase):
    def test_backend_module_map_covers_every_model(self):
        for model_id in catalog.model_ids():
            backend = catalog.backend(model_id)
            self.assertIn(backend, engine.BACKEND_MODULES, model_id)

    def test_installed_backends_is_a_boolean_map(self):
        result = engine.installed_backends()
        self.assertEqual(set(result), set(engine.BACKEND_MODULES))
        self.assertTrue(all(isinstance(v, bool) for v in result.values()))

    def test_beam_sizes_increase_with_accuracy(self):
        self.assertLess(engine._beam_size("fast"), engine._beam_size("balanced"))
        self.assertLess(engine._beam_size("balanced"), engine._beam_size("best"))

    def test_availability_note_is_always_a_string(self):
        for model_id in catalog.model_ids():
            self.assertTrue(engine.availability_note(model_id))

    def test_the_dialog_offers_both_devices_wherever_both_work(self):
        with mock.patch.object(engine, "cuda_available", return_value=True):
            # A CPU-preferred model still offers the faster GPU option.
            self.assertEqual(
                [o["value"] for o in engine.compute_options("whisper_small_base")],
                ["cpu", "cuda"],
            )
            # A GPU-preferred model still offers the CPU for a machine that
            # has no card.
            self.assertEqual(
                [o["value"] for o in engine.compute_options("whisper_large_v3")],
                ["cuda", "cpu"],
            )
        with mock.patch.object(engine, "cuda_available", return_value=False):
            self.assertEqual(
                [o["value"] for o in engine.compute_options("whisper_large_v3")],
                ["cpu"],
            )

    def test_the_variant_decides_whether_the_cpu_is_offered(self):
        with mock.patch.object(engine, "cuda_available", return_value=True):
            self.assertEqual(
                [o["value"] for o in engine.compute_options(
                    "canary_qwen", "2.5b")],
                ["cuda"],
            )
            self.assertEqual(
                [o["value"] for o in engine.compute_options(
                    "canary_qwen", "1b")],
                ["cpu", "cuda"],
            )

    def test_a_missing_cuda_runtime_is_reported_for_the_whisper_models(self):
        # Whisper Large V3's environment was installed before the cuBLAS wheel
        # joined the catalog: the GPU run then fell back to the CPU.  The
        # catalog names the wheel, the engine reports it as missing.
        self.assertIn("nvidia-cublas-cu12",
                      catalog.cuda_packages("whisper_large_v3"))
        with mock.patch.object(engine.environments, "_has_cuda",
                               return_value=True), \
                mock.patch.object(engine.environments, "_distributions",
                                  return_value={"faster-whisper": "1.2.1"}):
            self.assertEqual(engine.gpu_runtime_missing("whisper_large_v3"),
                             ["nvidia-cublas-cu12"])

    def test_a_gpu_choice_does_not_need_the_onnx_gpu_runtime(self):
        # The speech models are pip-installed PyTorch, which brings its own
        # CUDA runtime: ``compute.resolve_compute`` waits for the optional
        # sherpa-onnx runtime, and requiring that here would quietly send a
        # GPU transcription to the CPU on a machine that downloaded nothing.
        from ai_voice_studio import compute as _compute

        with mock.patch.object(_compute, "has_nvidia_gpu", return_value=True), \
                mock.patch.object(_compute.runtime, "is_installed",
                                  return_value=False):
            self.assertEqual(engine._resolve_device("cuda"), "cuda")
            self.assertEqual(engine._resolve_device("auto"), "cuda")
        with mock.patch.object(_compute, "has_nvidia_gpu", return_value=False):
            self.assertEqual(engine._resolve_device("auto"), "cpu")
            self.assertEqual(engine._resolve_device("cpu"), "cpu")

    def test_missing_packages_names_the_install_package(self):
        # Moonshine used to install ``useful-moonshine``, whose ``torch==2.4.1``
        # pin has no wheel for the Python the application builds environments
        # with, so the install always ended in "pip install failed (exit 1)".
        self.assertIn("transformers", engine.missing_packages("moonshine"))

    def test_repository_backed_models_name_their_hugging_face_repo(self):
        for model_id in catalog.model_ids():
            variant = catalog.default_variant(model_id)
            backend = catalog.backend(model_id)
            if backend not in engine.REPOSITORY_BACKENDS:
                continue
            self.assertTrue(
                engine.HF_MODELS.get(model_id, {}).get(variant),
                f"{model_id} has no repository for its default variant",
            )

    def test_every_repository_variant_has_a_model(self):
        for model_id, variants in engine.HF_MODELS.items():
            for value, _label in catalog.variants(model_id):
                self.assertIn(value, variants, f"{model_id}/{value}")

    def test_the_worker_detail_keeps_the_cause_and_drops_the_rust_note(self):
        # A Rust panic prints its cause and *then* the RUST_BACKTRACE hint;
        # reporting only the last line showed the hint and hid the cause.
        detail = engine._worker_error_detail(
            "thread 'main' panicked at src/lib.rs:12\n"
            "called `Result::unwrap()` on an `Err` value\n"
            "note: run with `RUST_BACKTRACE=1` environment variable to display "
            "a backtrace\n"
        )
        self.assertIn("panicked", detail)
        self.assertNotIn("RUST_BACKTRACE", detail)

    def test_the_worker_detail_survives_empty_output(self):
        self.assertTrue(engine._worker_error_detail(""))

    def test_the_worker_detail_drops_progress_bars_and_warnings(self):
        # A model download's tqdm bar and transformers' deprecation warning were
        # the last lines a crashed worker wrote, so they became the "error".
        detail = engine._worker_error_detail(
            "Fetching 6 files: 100%|##########| 6/6 [03:02<00:00, 32.05s/it]\n"
            "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!\n"
            "OSError: The paging file is too small for this operation to complete.\n"
        )
        self.assertIn("paging file", detail)
        self.assertNotIn("Fetching", detail)
        self.assertNotIn("deprecated", detail)

    def test_nemo_log_lines_are_not_the_error(self):
        detail = engine._worker_error_detail(
            "[NeMo I 2026-10-02 common:1210] Downloading model from HuggingFace\n"
            "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!\n"
            "RuntimeError: the real cause\n"
        )
        self.assertEqual(detail, "RuntimeError: the real cause")

    def test_the_worker_detail_falls_back_when_only_noise_was_written(self):
        detail = engine._worker_error_detail(
            "Fetching 6 files: 100%|##########| 6/6 [03:02<00:00, 32.05s/it]\n"
        )
        self.assertIn("no result", detail)

    def test_a_crashed_worker_reports_its_exit_code(self):
        class Result:
            returncode = 3221225477  # 0xC0000005, an access violation
            stdout = ""
            stderr = (
                "Fetching 6 files: 100%|##########| 6/6 [03:02<00:00, 32.05s/it]\n"
                "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!\n"
            )

        runtime = mock.MagicMock()
        runtime.python_exe = "python.exe"
        runtime.usable_python_exe.return_value = "python.exe"

        def fake_isfile(path):
            return str(path).endswith("python.exe")

        with mock.patch.object(engine.environments, "worker_script",
                               return_value="worker.py"), \
                mock.patch.object(engine.environments, "runtime",
                                  return_value=runtime), \
                mock.patch.object(engine.os.path, "isfile", fake_isfile), \
                mock.patch.object(engine.subprocess, "run", return_value=Result()), \
                mock.patch.object(engine.tempfile, "mkdtemp",
                                  return_value=tempfile.mkdtemp()):
            with self.assertRaises(engine.STTEngineError) as caught:
                engine._run_worker("canary_qwen", {"audio": "clip.wav"}, None)
        message = str(caught.exception)
        self.assertIn("3221225477", message)
        self.assertNotIn("Fetching", message)
        self.assertNotIn("deprecated", message)

    def test_the_worker_error_message_is_cleaned(self):
        cleaned = engine._clean_error_message(
            "OSError: The paging file is too small\n"
            "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!\n"
        )
        self.assertIn("paging file", cleaned)
        self.assertNotIn("deprecated", cleaned)

    def test_the_salm_variant_travels_with_the_request(self):
        seen = {}

        def fake_worker(model_id, request, on_progress):
            seen.update(request)
            return []

        with mock.patch.object(engine.os.path, "isfile", return_value=True), \
                mock.patch.object(engine.environments, "is_created",
                                  return_value=True), \
                mock.patch.object(engine, "_installed_backend",
                                  return_value="nemo"), \
                mock.patch.object(engine, "_run_worker", side_effect=fake_worker):
            engine.transcribe_audio("clip.wav", model_id="canary_qwen")
        self.assertEqual(seen["runner"], catalog.RUNNER_SALM)
        self.assertEqual(seen["hf_model"], "nvidia/canary-qwen-2.5b")

        with mock.patch.object(engine.os.path, "isfile", return_value=True), \
                mock.patch.object(engine.environments, "is_created",
                                  return_value=True), \
                mock.patch.object(engine, "_installed_backend",
                                  return_value="nemo"), \
                mock.patch.object(engine, "_run_worker", side_effect=fake_worker):
            engine.transcribe_audio("clip.wav", model_id="canary_qwen",
                                    variant="1b")
        self.assertEqual(seen["runner"], catalog.RUNNER_ASR)
        self.assertEqual(seen["hf_model"], "nvidia/canary-1b-v2")


class TimestampCapabilityTest(unittest.TestCase):
    """transformers refuses to return timestamps for anything but Whisper and
    CTC models, so a model without timings must never ask for them."""

    def test_the_audio_language_models_report_no_timings(self):
        for model_id in ("qwen3_asr", "voxtral", "granite_speech",
                         "moonshine"):
            self.assertFalse(catalog.supports_timestamps(model_id), model_id)

    def test_the_back_ends_with_their_own_timings_report_them(self):
        for model_id in ("whisper_large_v3", "whisper_small_base",
                         "canary_qwen", "parakeet_tdt"):
            self.assertTrue(catalog.supports_timestamps(model_id), model_id)

    def test_transcribe_audio_tells_the_worker_what_the_model_can_do(self):
        seen = {}

        def fake_worker(model_id, request, on_progress):
            seen.update(request)
            return []

        with mock.patch.object(engine.os.path, "isfile", return_value=True), \
                mock.patch.object(engine.environments, "is_created", return_value=True), \
                mock.patch.object(engine, "_installed_backend",
                                  return_value="transformers"), \
                mock.patch.object(engine, "_run_worker", side_effect=fake_worker):
            engine.transcribe_audio("clip.wav", model_id="qwen3_asr")
        self.assertIs(seen["timestamps"], False)
        self.assertEqual(seen["hf_model"], "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertEqual(seen["runner"], catalog.RUNNER_TRANSCRIPTION_REQUEST)

    def test_trust_remote_code_travels_with_the_request(self):
        seen = {}

        def fake_worker(model_id, request, on_progress):
            seen.update(request)
            return []

        entry = dict(catalog.model("qwen3_asr"))
        entry["trust_remote_code"] = True
        with mock.patch.object(engine.os.path, "isfile", return_value=True), \
                mock.patch.object(engine.environments, "is_created", return_value=True), \
                mock.patch.object(engine, "_installed_backend",
                                  return_value="transformers"), \
                mock.patch.object(engine.catalog, "model", return_value=entry), \
                mock.patch.object(engine, "_run_worker", side_effect=fake_worker):
            engine.transcribe_audio("clip.wav", model_id="qwen3_asr")
        self.assertIs(seen["trust_remote_code"], True)
        self.assertFalse(seen["timestamps"])

    def test_the_catalog_models_do_not_trust_remote_code(self):
        for model_id in ("whisper_large_v3", "qwen3_asr"):
            self.assertFalse(catalog.needs_remote_code(model_id), model_id)


class RunnerChoiceTest(unittest.TestCase):
    """The generic ASR pipeline cannot run every architecture, so each
    transformers-backed model names the strategy that can."""

    KNOWN = {catalog.RUNNER_PIPELINE, catalog.RUNNER_TRANSCRIPTION_REQUEST,
             catalog.RUNNER_CHAT}

    def test_every_transformers_model_names_a_known_runner(self):
        for model_id in catalog.model_ids():
            if catalog.backend(model_id) != "transformers":
                continue
            self.assertIn(catalog.runner(model_id), self.KNOWN, model_id)

    def test_the_models_reported_broken_declare_their_own_strategy(self):
        self.assertEqual(catalog.runner("qwen3_asr"),
                         catalog.RUNNER_TRANSCRIPTION_REQUEST)
        self.assertEqual(catalog.runner("voxtral"),
                         catalog.RUNNER_TRANSCRIPTION_REQUEST)
        self.assertEqual(catalog.runner("granite_speech"), catalog.RUNNER_CHAT)

    def test_a_model_without_a_runner_uses_the_pipeline(self):
        self.assertEqual(catalog.runner("moonshine"), catalog.RUNNER_PIPELINE)
        self.assertEqual(catalog.runner("speechbrain"), catalog.RUNNER_PIPELINE)

    def test_transcribe_audio_rejects_a_missing_file(self):
        with self.assertRaises(engine.STTEngineError):
            engine.transcribe_audio(
                os.path.join("does", "not", "exist.wav"),
                model_id="whisper_large_v3",
            )


class EnvironmentTest(unittest.TestCase):
    """Every model gets its own environment, so dependencies never clash."""

    def test_each_model_has_a_distinct_environment_folder(self):
        dirs = {m: environments.env_dir(m) for m in catalog.model_ids()}
        self.assertEqual(len(set(dirs.values())), len(dirs))
        for model_id, path in dirs.items():
            self.assertIn("stt_envs", path)
            self.assertTrue(os.path.basename(path), model_id)

    def test_environment_folder_is_named_after_the_model(self):
        for model_id in catalog.model_ids():
            self.assertEqual(
                os.path.basename(environments.env_dir(model_id)), model_id
            )

    def test_cpu_install_plan_is_a_single_pypi_step(self):
        with mock.patch.object(environments, "_has_cuda", return_value=False):
            steps = environments.install_plan("whisper_large_v3")
        self.assertEqual(len(steps), 1)
        self.assertIn("faster-whisper", steps[0][0])
        self.assertIsNone(steps[0][1])

    def test_gpu_install_plan_puts_torch_on_the_cuda_index_first(self):
        with mock.patch.object(environments, "_has_cuda", return_value=True):
            steps = environments.install_plan("qwen3_asr")
        self.assertEqual(steps[0][0], ["torch"])
        self.assertIn("pytorch.org", steps[0][1])
        self.assertIn("transformers", steps[1][0])
        self.assertIsNone(steps[1][1])

    def test_gpu_install_plan_adds_the_cublas_runtime_for_whisper(self):
        # CTranslate2 is a CUDA build on Windows and needs cuBLAS: without it
        # every Whisper run failed with "Library cublas64_12.dll is not found
        # or cannot be loaded".
        with mock.patch.object(environments, "_has_cuda", return_value=True):
            packages = environments.install_plan("whisper_large_v3")[-1][0]
        self.assertIn("nvidia-cublas-cu12", packages)
        self.assertIn("faster-whisper", packages)

    def test_cpu_install_plan_never_downloads_the_cuda_runtime(self):
        with mock.patch.object(environments, "_has_cuda", return_value=False):
            packages = environments.install_plan("whisper_large_v3")[-1][0]
        self.assertNotIn("nvidia-cublas-cu12", packages)

    def test_a_cpu_only_machine_is_never_asked_for_the_cuda_runtime(self):
        with mock.patch.object(environments, "_has_cuda", return_value=False):
            self.assertEqual(environments.missing_cuda_runtime("whisper_large_v3"),
                             [])

    def test_a_gpu_machine_reports_the_cuda_runtime_a_stale_env_lacks(self):
        # Whisper Large V3 was installed before ``nvidia-cublas-cu12`` joined
        # the catalog; its GPU run silently fell back to the CPU because
        # CTranslate2 could not load cuBLAS.
        with mock.patch.object(environments, "_has_cuda", return_value=True), \
                mock.patch.object(environments, "_distributions",
                                  return_value={"faster-whisper": "1.2.1"}), \
                mock.patch.object(environments, "module_present",
                                  return_value=True):
            self.assertEqual(environments.missing_cuda_runtime("whisper_large_v3"),
                             ["nvidia-cublas-cu12"])

    def test_a_complete_environment_reports_nothing_missing(self):
        with mock.patch.object(environments, "_has_cuda", return_value=True), \
                mock.patch.object(
                    environments, "_distributions",
                    return_value={"faster-whisper": "1.2.1",
                                  "nvidia-cublas-cu12": "12.4.5"},
                ):
            self.assertEqual(environments.missing_cuda_runtime("whisper_large_v3"),
                             [])

    def test_packages_for_covers_the_cuda_runtime_on_uninstall(self):
        self.assertIn("nvidia-cublas-cu12",
                      environments.packages_for("whisper_small_base"))
        self.assertIn("nvidia-cublas-cu12",
                      environments.packages_for("whisper_large_v3"))
        self.assertEqual(
            list(environments.packages_for("whisper_large_v3")).count(
                "faster-whisper"),
            1,
        )

    def test_packages_for_strips_extras_for_uninstall(self):
        self.assertIn("nemo-toolkit", environments.packages_for("canary_qwen"))
        self.assertIn("faster-whisper", environments.packages_for("whisper_large_v3"))

    def test_nemo_models_pin_transformers_below_five(self):
        # NeMo 3.0 predates transformers 5; with 5.18 the Canary-Qwen SALM load
        # segfaulted.  The install must keep transformers on the 4.x series.
        for model_id in ("canary_qwen", "parakeet_tdt"):
            self.assertIn("transformers<5", catalog.pip_packages(model_id),
                          model_id)
        self.assertIn("transformers", environments.packages_for("canary_qwen"))
        self.assertNotIn("transformers<5",
                         environments.packages_for("canary_qwen"))

    def test_a_model_is_not_installed_without_its_backend(self):
        with mock.patch.object(environments, "is_created", return_value=True), \
                mock.patch.object(environments, "module_present", return_value=False):
            self.assertFalse(environments.is_installed("whisper_large_v3"))

    def test_availability_note_names_the_models_own_environment(self):
        note = engine.availability_note("whisper_small_base")
        self.assertIn("stt_envs", note)


class PyAvCompatTest(unittest.TestCase):
    """faster-whisper still calls ``av.open(metadata_errors=...)`` although PyAV
    19 removed that argument; the worker wraps ``av.open`` so both old and new
    PyAV releases keep decoding audio for the Whisper models."""

    @staticmethod
    def _fake_av(accept_metadata_errors: bool):
        module = types.ModuleType("av")
        module.calls = []

        def fake_open(*args, **kwargs):
            module.calls.append(dict(kwargs))
            if not accept_metadata_errors and "metadata_errors" in kwargs:
                raise TypeError(
                    "open() got an unexpected keyword argument 'metadata_errors'"
                )
            return "opened"

        module.open = fake_open
        return module

    def test_shim_drops_metadata_errors_on_new_pyav(self):
        module = self._fake_av(accept_metadata_errors=False)
        with mock.patch.dict(sys.modules, {"av": module}):
            worker._patch_pyav_open()
            result = module.open("clip.wav", mode="r", metadata_errors="ignore")
        self.assertEqual(result, "opened")
        self.assertNotIn("metadata_errors", module.calls[-1])

    def test_shim_keeps_metadata_errors_on_old_pyav(self):
        module = self._fake_av(accept_metadata_errors=True)
        with mock.patch.dict(sys.modules, {"av": module}):
            worker._patch_pyav_open()
            result = module.open("clip.wav", mode="r", metadata_errors="ignore")
        self.assertEqual(result, "opened")
        self.assertIn("metadata_errors", module.calls[-1])

    def test_shim_is_idempotent(self):
        module = self._fake_av(accept_metadata_errors=False)
        with mock.patch.dict(sys.modules, {"av": module}):
            worker._patch_pyav_open()
            first = module.open
            worker._patch_pyav_open()
            self.assertIs(module.open, first)


class _FakeTensor:
    """Just enough of a tensor for the worker's audio-language-model path."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        return (1, len(self.data))

    def __getitem__(self, _item):
        return self.data


class _FakeInputs(dict):
    def to(self, *_args, **_kwargs):
        return self


def _fake_transformers(monkeypatch_modules=None, pipeline_factory=None,
                       processor_factory=None, multimodal_factory=None):
    """A stand-in ``transformers`` module for the worker's imports."""
    module = types.ModuleType("transformers")
    module.pipeline = pipeline_factory or (lambda *a, **k: None)
    if processor_factory is not None:
        class _AutoProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return processor_factory(*args, **kwargs)
        module.AutoProcessor = _AutoProcessor
    if multimodal_factory is not None:
        class _AutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return multimodal_factory(*args, **kwargs)
        module.AutoModelForMultimodalLM = _AutoModel
    return module


class WhisperChunkingTest(unittest.TestCase):
    """Whisper decodes in fixed 30-second windows and emits at most 448 tokens
    for each one, so a script its tokenizer splits finely does not fit: the
    model stops at the window edge and faster-whisper advances a whole window,
    losing the words in between.  A 54-second Hindi recording came back as 100
    words where the audio holds 196 - with the gap exactly on the window
    boundary - and decoding it in pieces gave the complete 199.

    One fixed piece size has to be the size that is safe for the costliest
    language, which slices an English lecture far more than its own tokens
    need, so the size is worked out from the language instead.
    """

    @staticmethod
    def _model(calls, detected=None):
        class FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append((len(audio), dict(kwargs)))
                # One segment per piece, timed from the start of the piece, the
                # way a real model times each window it is handed.  The info is
                # where a real model reports the language it heard.
                info = (None if detected is None
                        else types.SimpleNamespace(language=detected))
                return ([{"start": 0.5, "end": 2.5, "text": "piece"}], info)

        return FakeModel()

    def _run(self, seconds, detected=None, **request):
        calls = []
        samples = [0.0] * int(seconds * 16000)
        with mock.patch.object(worker, "_decode_audio",
                               return_value=(samples, 16000)):
            segments = worker._transcribe_with_whisper(
                self._model(calls, detected), {"audio": "clip.wav", **request},
            )
        return segments, calls

    def _pieces(self, seconds, detected=None, **request):
        """The recording time each model call was handed."""
        _segments, calls = self._run(seconds, detected, **request)
        return [frames / 16000.0 for frames, _kwargs in calls]

    def test_a_dense_script_keeps_the_pieces_that_were_measured_to_work(self):
        # 54.41 s of Hindi lost a quarter of its words to the 30-second window,
        # and short pieces recovered the whole transcript: 13 seconds on the
        # 29-minute chapter, where 19.6 still ran out of window.  Devanagari is
        # the most expensive language there is, so its pieces must not grow
        # past that.
        pieces = self._pieces(54.41, language="hi")
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(piece, 14.0 + 0.001)

    def test_a_language_left_to_the_model_is_sized_like_the_costliest_one(self):
        # Nothing is known about a recording whose language is not given, and
        # guessing "cheap" where the script is dense loses the transcript.
        self.assertAlmostEqual(worker._whisper_piece_seconds(None),
                               worker._whisper_piece_seconds("hi"))
        for piece in self._pieces(54.41):
            self.assertLessEqual(piece, 14.0 + 0.001)

    def test_an_english_lecture_is_sliced_far_less_than_a_hindi_one(self):
        english = self._pieces(600.0, language="en")
        hindi = self._pieces(600.0, language="hi")
        self.assertLess(len(english), len(hindi))
        self.assertGreater(max(english), max(hindi))
        self.assertLessEqual(max(english),
                             worker.WHISPER_MAX_PIECE_SECONDS + 0.001)

    def test_a_recording_short_enough_for_one_piece_is_not_cut(self):
        # 25 seconds of English used to become a 20-second piece and a 5-second
        # one; a piece may now hold the whole recording.
        self.assertEqual(len(self._pieces(25.0, language="en")), 1)

    def test_the_whole_recording_is_still_covered(self):
        for request in ({"language": "en"}, {"language": "hi"}, {}):
            pieces = self._pieces(54.41, **request)
            self.assertAlmostEqual(sum(pieces), 54.41, places=2)
            self.assertGreater(len(pieces), 1)

    def test_a_short_recording_is_one_piece(self):
        self.assertEqual(len(self._pieces(3.0)), 1)

    def test_the_last_piece_is_never_a_stub(self):
        # 57 seconds in 28-second pieces would be 28, 28 and then a whole model
        # call for one second; the tail is split evenly instead.
        pieces = self._pieces(57.0, language="en")
        self.assertGreaterEqual(min(pieces), max(pieces) / 2)
        self.assertAlmostEqual(sum(pieces), 57.0, places=2)

    def test_segments_are_timed_against_the_whole_recording(self):
        segments, calls = self._run(54.41, language="hi")
        # Each piece reports 0.5-2.5 s of itself; shifted onto the recording the
        # starts must climb by one piece at a time, never restart at 0.5.  The
        # first two pieces are a full piece apart; the tail is split evenly, so
        # only those two can be compared against the piece length.
        starts = [s["start"] for s in segments]
        self.assertEqual(len(starts), len(calls))
        self.assertGreater(starts[1], starts[0])
        self.assertAlmostEqual(starts[1] - starts[0],
                               worker._whisper_piece_seconds("hi"), places=2)
        self.assertLessEqual(segments[-1]["end"], 54.41)

    def test_the_detected_language_resizes_the_rest_of_the_recording(self):
        # Only the language that was left to the model can be learned this way,
        # and it is learned from the first piece: cautious to start with, long
        # pieces once the model has said it heard English.
        pieces = self._pieces(200.0, detected="en")
        self.assertLessEqual(pieces[0], 14.0 + 0.001)
        self.assertAlmostEqual(pieces[1], worker.WHISPER_MAX_PIECE_SECONDS,
                               places=1)
        self.assertAlmostEqual(sum(pieces), 200.0, places=2)

    def test_a_detected_dense_language_keeps_the_small_pieces(self):
        pieces = self._pieces(200.0, detected="hi")
        for piece in pieces:
            self.assertLessEqual(piece, 14.0 + 0.001)

    def test_a_given_language_is_not_second_guessed_by_the_model(self):
        # A hint is what the recording is priced with, so the pieces do not
        # change shape partway through a transcription the user asked for.
        pieces = self._pieces(200.0, detected="hi", language="en")
        self.assertAlmostEqual(pieces[0], worker.WHISPER_MAX_PIECE_SECONDS,
                               places=1)
        self.assertAlmostEqual(pieces[1], worker.WHISPER_MAX_PIECE_SECONDS,
                               places=1)

    def test_word_timings_are_asked_for_because_they_complete_the_decode(self):
        _segments, calls = self._run(54.41)
        for _frames, options in calls:
            self.assertTrue(options["word_timestamps"])
            self.assertIs(options["without_timestamps"], False)
            self.assertTrue(options["vad_filter"])

    def test_translation_keeps_the_plain_decode(self):
        # Word timings are aligned against the language that was spoken, so
        # they are not asked for when the text comes back in another language.
        _segments, calls = self._run(20.0, translate_to="en")
        for _frames, options in calls:
            self.assertEqual(options["task"], "translate")
            self.assertNotIn("word_timestamps", options)

    def test_the_language_hint_and_accuracy_still_reach_the_model(self):
        _segments, calls = self._run(20.0, language="hi", accuracy="best")
        _frames, options = calls[0]
        self.assertEqual(options["language"], "hi")
        self.assertEqual(options["beam_size"], 8)

    def test_an_automatic_language_is_left_to_the_model(self):
        _segments, calls = self._run(20.0, language="auto")
        self.assertIsNone(calls[0][1]["language"])


class WhisperCutTest(unittest.TestCase):
    """A piece that ends in the middle of a word leaves a fragment behind and
    the next piece carries on after the rest of it, which is how a transcript
    comes back with parts of words missing.  The piece budget says how much
    speech the decoder can finish, so the cut is moved back onto the quietest
    moment before that budget runs out instead.
    """

    RATE = 16000

    def _speech(self, seconds, quiet=None):
        """Loud-ish samples, with one stretch of near-silence in them."""
        samples = [0.4] * int(seconds * self.RATE)
        if quiet is not None:
            low, high = (int(edge * self.RATE) for edge in quiet)
            for index in range(low, high):
                samples[index] = 0.0005
        return samples

    def test_a_cut_lands_on_the_quietest_moment_before_the_budget(self):
        # The pause has to be inside the last few seconds of the piece - that
        # is the stretch the search looks at - and the cut goes into it rather
        # than through the speech that follows.
        samples = self._speech(13.0, quiet=(10.5, 11.5))
        cut = worker._whisper_cut_frame(samples, self.RATE, 0, len(samples))
        self.assertGreaterEqual(cut / self.RATE, 10.5)
        self.assertLessEqual(cut / self.RATE, 11.5)

    def test_a_cut_is_never_moved_later_than_the_budget(self):
        # Moving a piece later would hand the decoder more tokens than it can
        # finish, which is what loses the tail of the piece in the first place.
        end = int(13.0 * self.RATE)
        samples = self._speech(20.0, quiet=(12.5, 13.0))
        self.assertLessEqual(worker._whisper_cut_frame(samples, self.RATE, 0, end),
                             end)

    def test_a_boundary_that_is_already_in_silence_is_left_alone(self):
        samples = [0.0] * int(13.0 * self.RATE)
        end = len(samples)
        self.assertEqual(worker._whisper_cut_frame(samples, self.RATE, 0, end),
                         end)

    def test_a_quiet_tail_does_not_shorten_the_piece(self):
        # A piece whose last frame is its quietest is a piece that already ends
        # in a pause; there is nothing to gain by moving it seconds earlier.
        samples = self._speech(13.0, quiet=(12.9, 13.0))
        end = len(samples)
        self.assertEqual(worker._whisper_cut_frame(samples, self.RATE, 0, end),
                         end)

    def test_a_piece_too_short_to_search_is_not_cut(self):
        samples = self._speech(1.0, quiet=(0.9, 1.0))
        end = len(samples)
        self.assertEqual(worker._whisper_cut_frame(samples, self.RATE, 0, end),
                         end)

    def test_the_quiet_search_needs_no_numpy(self):
        samples = self._speech(13.0, quiet=(9.0, 10.0))
        end = len(samples)
        with mock.patch.dict(sys.modules, {"numpy": None}):
            self.assertEqual(
                worker._whisper_cut_frame(samples, self.RATE, 0, end), end)

    def test_the_pieces_still_cover_the_recording_exactly(self):
        samples = self._speech(54.41, quiet=(12.0, 12.5))
        calls = []

        class FakeModel:
            def transcribe(self, audio, **kwargs):
                calls.append(len(audio))
                return ([{"start": 0.0, "end": 1.0, "text": "piece"}], None)

        with mock.patch.object(worker, "_decode_audio",
                               return_value=(samples, self.RATE)):
            worker._transcribe_with_whisper(FakeModel(),
                                            {"audio": "clip.wav", "language": "hi"})
        self.assertEqual(sum(calls), len(samples))
        self.assertTrue(all(frames > 0 for frames in calls))

    def test_a_dense_piece_is_short_enough_for_its_tokens(self):
        # Measured on a 29-minute Hindi chapter: 19.6-second pieces still ran
        # out of window, 13-second ones did not, and shorter gained nothing.
        seconds = worker._whisper_piece_seconds("hi")
        self.assertGreaterEqual(seconds, 12.0)
        self.assertLessEqual(seconds, 14.0)


class WhisperCudaFallbackTest(unittest.TestCase):
    """CTranslate2 cannot load on a machine without the CUDA runtime, so a GPU
    run has to fall back to the CPU rather than fail the transcription."""

    @staticmethod
    def _faster_whisper(created, fail_on_cuda):
        module = types.ModuleType("faster_whisper")

        class FakeModel:
            def __init__(self, variant, device, compute_type):
                created.append((device, compute_type))
                if fail_on_cuda and device == "cuda":
                    raise RuntimeError(
                        "Library cublas64_12.dll is not found or cannot be loaded"
                    )

            def transcribe(self, _audio, **_kwargs):
                return ([{"start": 0.0, "end": 1.5, "text": "hello"}], None)

        module.WhisperModel = FakeModel
        return module

    def _run(self, created, fail_on_cuda, compute, cuda_present):
        module = self._faster_whisper(created, fail_on_cuda)
        with mock.patch.dict(sys.modules, {"faster_whisper": module}), \
                mock.patch.object(worker, "_patch_pyav_open"), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), \
                mock.patch.object(worker, "_cuda_libraries_present",
                                  return_value=cuda_present), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0] * 16000, 16000)):
            return worker.run({
                "backend": "faster_whisper", "audio": "clip.wav",
                "variant": "small", "compute": compute,
            })

    def test_a_cuda_run_that_cannot_load_falls_back_to_the_cpu(self):
        created = []
        result = self._run(created, True, "cuda", True)
        self.assertEqual(created, [("cuda", "float16"), ("cpu", "int8")])
        self.assertEqual(result["device"], "cpu")
        self.assertIn("CPU", result["note"])
        self.assertEqual(result["segments"][0]["text"], "hello")

    def test_a_missing_cuda_runtime_goes_straight_to_the_cpu(self):
        created = []
        result = self._run(created, False, "cuda", False)
        self.assertEqual(created, [("cpu", "int8")])
        self.assertEqual(result["device"], "cpu")
        self.assertIn("CUDA runtime", result["note"])

    def test_an_explicit_cpu_request_uses_the_cpu(self):
        created = []
        result = self._run(created, False, "cpu", True)
        self.assertEqual(created, [("cpu", "int8")])
        self.assertEqual(result["note"], "")

    def test_a_cpu_failure_is_reported(self):
        module = types.ModuleType("faster_whisper")

        class Boom:
            def __init__(self, *_a, **_k):
                raise RuntimeError("no model")

        module.WhisperModel = Boom
        with mock.patch.dict(sys.modules, {"faster_whisper": module}), \
                mock.patch.object(worker, "_patch_pyav_open"), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0] * 16000, 16000)), self.assertRaises(RuntimeError):
            worker.run({"backend": "faster_whisper", "audio": "clip.wav",
                        "variant": "small", "compute": "cpu"})


class TransformersRunnerTest(unittest.TestCase):
    """The ASR pipeline must never be asked for timestamps an architecture
    cannot produce, and an audio language model must use its own API."""

    def _pipeline_run(self, req, kind="seq2seq"):
        calls = []

        class FakePipeline:
            type = kind

            def __call__(self, _inputs, **kwargs):
                calls.append(dict(kwargs))
                if kwargs.get("return_timestamps"):
                    raise ValueError(
                        "We cannot return_timestamps yet on non-CTC models "
                        "apart from Whisper!"
                    )
                return {"text": "hello there"}

        module = _fake_transformers(pipeline_factory=lambda *a, **k: FakePipeline())
        with mock.patch.dict(sys.modules, {"transformers": module}), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0], 16000)):
            result = worker.run(req)
        return result, calls

    def test_a_model_without_timestamps_does_not_ask_for_them(self):
        result, calls = self._pipeline_run({
            "backend": "transformers", "audio": "clip.wav",
            "hf_model": "moonshine-ai/moonshine-base", "timestamps": False,
        })
        self.assertNotIn("return_timestamps", calls[-1])
        self.assertEqual(result["segments"][0]["text"], "hello there")

    def test_a_timestamp_capable_model_still_avoid_a_seq2seq_mismatch(self):
        # Asking a plain seq2seq pipeline for timings is exactly what raised
        # "We cannot return_timestamps yet ...", so the request is filtered.
        _result, calls = self._pipeline_run({
            "backend": "transformers", "audio": "clip.wav",
            "hf_model": "ibm-granite/granite-4.0-1b-speech", "timestamps": True,
        }, kind="seq2seq")
        self.assertNotIn("return_timestamps", calls[-1])

    def test_whisper_still_gets_its_timestamps(self):
        _result, calls = self._pipeline_run({
            "backend": "transformers", "audio": "clip.wav",
            "hf_model": "openai/whisper-tiny", "timestamps": True,
        }, kind="seq2seq_whisper")
        # The fake pipeline rejects timings, so the first call asked for them
        # and the retry dropped them; Whisper is the architecture that may ask.
        self.assertIs(calls[0]["return_timestamps"], True)

    def test_a_rejected_timestamp_request_is_retried_without_it(self):
        calls = []

        class FakePipeline:
            type = "seq2seq_whisper"

            def __call__(self, _inputs, **kwargs):
                calls.append(dict(kwargs))
                if kwargs.get("return_timestamps"):
                    raise ValueError(
                        "We cannot return_timestamps yet on non-CTC models "
                        "apart from Whisper!"
                    )
                return {"text": "retried"}

        module = _fake_transformers(pipeline_factory=lambda *a, **k: FakePipeline())
        with mock.patch.dict(sys.modules, {"transformers": module}), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0], 16000)), \
                mock.patch.object(worker, "_max_new_tokens", return_value=256):
            result = worker.run({
                "backend": "transformers", "audio": "clip.wav",
                "hf_model": "openai/whisper-tiny", "timestamps": True,
            })
        self.assertEqual(len(calls), 2)
        self.assertNotIn("return_timestamps", calls[-1])
        self.assertEqual(result["segments"][0]["text"], "retried")

    def test_a_language_hint_only_reaches_whisper(self):
        _result, calls = self._pipeline_run({
            "backend": "transformers", "audio": "clip.wav",
            "hf_model": "ibm-granite/granite-4.0-1b-speech",
            "language": "de", "timestamps": False,
        })
        self.assertNotIn("generate_kwargs", calls[-1])

    def test_a_missing_repository_is_reported_clearly(self):
        with self.assertRaises(RuntimeError) as caught:
            worker.run({"backend": "transformers", "audio": "clip.wav",
                        "hf_model": ""})
        self.assertIn("repository", str(caught.exception).lower())

    def test_an_audio_language_model_uses_its_own_transcription_api(self):
        used = {}

        class FakeProcessor:
            def apply_transcription_request(self, audio=None, language=None):
                used["audio"] = audio
                used["language"] = language
                return _FakeInputs(input_ids=_FakeTensor([1, 2, 3]))

            def decode(self, tokens, return_format=None):
                used["return_format"] = return_format
                return ["a transcript"]

        class FakeModel:
            dtype = "float32"
            device = "cpu"

            def generate(self, **_kwargs):
                return _FakeTensor([1, 2, 3, 4, 5])

        torch_module = types.ModuleType("torch")
        torch_module.no_grad = lambda: mock.MagicMock(
            __enter__=lambda self: None, __exit__=lambda *a: False
        )
        module = _fake_transformers(
            processor_factory=lambda *a, **k: FakeProcessor(),
            multimodal_factory=lambda *a, **k: FakeModel(),
        )
        with mock.patch.dict(sys.modules, {"transformers": module,
                                           "torch": torch_module}), \
                mock.patch.object(worker, "_max_new_tokens", return_value=256):
            result = worker.run({
                "backend": "transformers", "audio": "clip 1.wav",
                "runner": "transcription_request",
                "hf_model": "Qwen/Qwen3-ASR-1.7B-hf", "language": "en",
            })
        self.assertEqual(used["audio"], "clip 1.wav")
        self.assertEqual(used["language"], "en")
        self.assertEqual(used["return_format"], "transcription_only")
        self.assertEqual(result["segments"][0]["text"], "a transcript")
        self.assertFalse(result["timestamps"])


class ChatAudioRunnerTest(unittest.TestCase):
    """Granite Speech's feature extractor rejects the ASR pipeline's
    ``sampling_rate`` argument, so it is driven by a chat prompt instead.
    """

    def test_the_chat_prompt_and_the_audio_reach_the_processor(self):
        captured = {}

        class FakeTokenizer:
            def apply_chat_template(self, chat, tokenize=False,
                                    add_generation_prompt=False):
                captured["chat"] = chat
                return "<prompt>"

            def decode(self, tokens, skip_special_tokens=False):
                return "granite text"

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def __call__(self, prompt, audio, **kwargs):
                captured["prompt"] = prompt
                captured["samples"] = audio
                captured["kwargs"] = kwargs
                return _FakeInputs(input_ids=_FakeTensor([1, 2, 3]))

        class FakeModel:
            device = "cpu"
            dtype = "float32"

            def generate(self, **_kwargs):
                return _FakeTensor([1, 2, 3, 9, 9])

        torch_module = types.ModuleType("torch")
        torch_module.no_grad = lambda: mock.MagicMock(
            __enter__=lambda self: None, __exit__=lambda *a: False
        )
        module = _fake_transformers(
            processor_factory=lambda *a, **k: FakeProcessor(),
            multimodal_factory=lambda *a, **k: FakeModel(),
        )
        module.AutoModelForSpeechSeq2Seq = module.AutoModelForMultimodalLM
        with mock.patch.dict(sys.modules, {"transformers": module,
                                           "torch": torch_module}), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.5], 16000)), \
                mock.patch.object(worker, "_max_new_tokens", return_value=256):
            result = worker.run({
                "backend": "transformers", "audio": "clip.wav",
                "runner": "chat",
                "hf_model": "ibm-granite/granite-4.0-1b-speech",
            })
        self.assertEqual(captured["prompt"], "<prompt>")
        self.assertIn("audio", worker.TRANSCRIBE_PROMPT)
        self.assertEqual(captured["samples"], [0.5])
        self.assertEqual(result["segments"][0]["text"], "granite text")

    def test_a_processor_without_a_chat_template_falls_back_to_the_pipeline(self):
        calls = []

        class FakeProcessor:
            tokenizer = object()

        class FakePipeline:
            type = "seq2seq"

            def __call__(self, _inputs, **_kwargs):
                calls.append(True)
                return {"text": "from the pipeline"}

        module = _fake_transformers(
            pipeline_factory=lambda *a, **k: FakePipeline(),
            processor_factory=lambda *a, **k: FakeProcessor(),
        )
        with mock.patch.dict(sys.modules, {"transformers": module}), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0], 16000)):
            result = worker.run({
                "backend": "transformers", "audio": "clip.wav",
                "runner": "chat", "hf_model": "ibm-granite/granite-4.0-1b-speech",
            })
        self.assertEqual(calls, [True])
        self.assertEqual(result["segments"][0]["text"], "from the pipeline")


class NemoRunnerTest(unittest.TestCase):
    """Canary-Qwen and Parakeet TDT used to end in "The 'nemo' back-end is
    not supported by the worker yet."
    """

    @staticmethod
    def _nemo_modules(model, calls):
        asr_models = types.ModuleType("nemo.collections.asr.models")
        asr_models.ASRModel = model
        collections = types.ModuleType("nemo.collections")
        collections.asr = types.ModuleType("nemo.collections.asr")
        collections.asr.models = asr_models
        nemo = types.ModuleType("nemo")
        nemo.collections = collections
        return {"nemo": nemo, "nemo.collections": collections,
                "nemo.collections.asr": collections.asr,
                "nemo.collections.asr.models": asr_models}

    def test_segment_timestamps_become_timed_segments(self):
        class Hypothesis:
            text = "hello world"
            timestamp = {"segment": [
                {"start": 0.0, "end": 1.0, "segment": "hello "},
                {"start": 1.0, "end": 2.0, "segment": "world"},
            ]}

        captured = {}

        class Model:
            @staticmethod
            def from_pretrained(name):
                captured["name"] = name
                return Model()

            def to(self, _device):
                return self

            def transcribe(self, audio, **kwargs):
                captured["kwargs"] = kwargs
                return [Hypothesis()]

        modules = self._nemo_modules(Model, captured)
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), \
                mock.patch.object(worker, "_audio_duration", return_value=2.0):
            result = worker.run({
                "backend": "nemo", "audio": "clip.wav", "compute": "cpu",
                "hf_model": "nvidia/parakeet-tdt-0.6b-v2",
            })
        self.assertEqual(captured["name"], "nvidia/parakeet-tdt-0.6b-v2")
        self.assertIs(captured["kwargs"]["timestamps"], True)
        self.assertEqual([s["text"] for s in result["segments"]],
                         ["hello", "world"])
        self.assertEqual(result["segments"][1]["start"], 1.0)

    def test_a_model_without_timestamps_is_retried_plainly(self):
        calls = []

        class Model:
            @staticmethod
            def from_pretrained(_name):
                return Model()

            def to(self, _device):
                return self

            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                if kwargs.get("timestamps"):
                    raise TypeError("timestamps not supported")
                return ["plain text result"]

        modules = self._nemo_modules(Model, {})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), \
                mock.patch.object(worker, "_audio_duration", return_value=3.0):
            result = worker.run({"backend": "nemo", "audio": "clip.wav",
                                 "compute": "cpu", "hf_model": "m"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["segments"][0]["text"], "plain text result")

    def test_translation_is_refused_rather_than_silently_ignored(self):
        modules = self._nemo_modules(object(), {})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), self.assertRaises(RuntimeError) as caught:
            worker.run({"backend": "nemo", "audio": "clip.wav",
                        "hf_model": "m", "translate_to": "en"})
        self.assertIn("translation", str(caught.exception).lower())

    def test_an_asr_variant_reports_the_asr_error(self):
        # ``nvidia/canary-1b-v2`` is an ordinary Canary checkpoint, not a SALM:
        # its config has no audio locator tag, so the SALM fallback's message
        # ("Missing key audio_locator_tag") hid the real ASR failure.
        class Model:
            @staticmethod
            def from_pretrained(_name):
                raise OSError("no space left on device")

        modules = self._nemo_modules(Model, {})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), self.assertRaises(OSError) as caught:
            worker.run({"backend": "nemo", "audio": "clip.wav",
                        "compute": "cpu", "runner": "asr",
                        "hf_model": "nvidia/canary-1b-v2"})
        self.assertIn("no space left", str(caught.exception))

    def test_a_failed_asr_load_surfaces_when_the_repo_is_not_a_salm(self):
        class Model:
            @staticmethod
            def from_pretrained(_name):
                raise OSError("no space left on device")

        modules = self._nemo_modules(Model, {})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), self.assertRaises(RuntimeError) as caught:
            worker.run({"backend": "nemo", "audio": "clip.wav",
                        "compute": "cpu", "hf_model": "m"})
        # The SALM import fails too, so the ASR error is what is reported.
        self.assertIn("no space left", str(caught.exception))
        self.assertNotIn("audio_locator_tag", str(caught.exception))

    def test_a_salm_variant_uses_the_speech_language_model_loader(self):
        captured = {}

        class Tokenizer:
            @staticmethod
            def ids_to_text(_tokens):
                return "hello from salm"

        class SALM:
            audio_locator_tag = "<|audio|>"
            tokenizer = Tokenizer()

            @staticmethod
            def from_pretrained(name):
                captured["name"] = name
                return SALM()

            def to(self, _device):
                return self

            def eval(self):
                return self

            def generate(self, prompts=None, max_new_tokens=None):
                captured["prompt"] = prompts[0][0]

                class _Row:
                    @staticmethod
                    def tolist():
                        return [1, 2, 3]

                class _Answer:
                    def __getitem__(self, _index):
                        return _Row()

                return _Answer()

        models = types.ModuleType("nemo.collections.speechlm2.models")
        models.SALM = SALM
        speechlm2 = types.ModuleType("nemo.collections.speechlm2")
        speechlm2.models = models
        collections = types.ModuleType("nemo.collections")
        collections.speechlm2 = speechlm2
        nemo = types.ModuleType("nemo")
        nemo.collections = collections
        modules = {"nemo": nemo, "nemo.collections": collections,
                   "nemo.collections.speechlm2": speechlm2,
                   "nemo.collections.speechlm2.models": models}

        torch_module = types.ModuleType("torch")
        torch_module.no_grad = lambda: mock.MagicMock(
            __enter__=lambda self: None, __exit__=lambda *a: False
        )
        with mock.patch.dict(sys.modules, {**modules, "torch": torch_module}), \
                mock.patch.object(worker, "_prepare_native_libraries",
                                  return_value=[]), \
                mock.patch.object(worker, "_max_new_tokens", return_value=256):
            result = worker.run({"backend": "nemo", "audio": "clip.wav",
                                 "compute": "cpu", "runner": "salm",
                                 "hf_model": "nvidia/canary-qwen-2.5b"})
        self.assertEqual(captured["name"], "nvidia/canary-qwen-2.5b")
        self.assertIn("<|audio|>", captured["prompt"]["content"])
        self.assertEqual(result["segments"][0]["text"], "hello from salm")


class SpeechBrainRunnerTest(unittest.TestCase):
    """The SpeechBrain model used to end in "The 'speechbrain' back-end is
    not supported by the worker yet.
    """

    @staticmethod
    def _modules(recipe, fetching=None):
        """A stand-in ``speechbrain`` package for the worker's imports."""
        asr = types.ModuleType("speechbrain.inference.ASR")
        asr.EncoderDecoderASR = recipe
        inference = types.ModuleType("speechbrain.inference")
        inference.ASR = asr
        speechbrain = types.ModuleType("speechbrain")
        speechbrain.inference = inference
        modules = {"speechbrain": speechbrain,
                   "speechbrain.inference": inference,
                   "speechbrain.inference.ASR": asr}
        if fetching is not None:
            utils = types.ModuleType("speechbrain.utils")
            utils.fetching = fetching
            speechbrain.utils = utils
            modules["speechbrain.utils"] = utils
            modules["speechbrain.utils.fetching"] = fetching
        return modules

    @staticmethod
    def _torch():
        module = types.ModuleType("torch")

        class FakeTensor:
            def __init__(self, data):
                self.data = data

            def unsqueeze(self, _dim):
                return self

            def to(self, _device):
                return self

        module.from_numpy = FakeTensor
        module.tensor = FakeTensor
        return module

    def test_the_recipe_transcribes_and_copies_instead_of_symlinking(self):
        captured = {}

        class EncoderDecoderASR:
            device = "cpu"

            @staticmethod
            def from_hparams(source=None, savedir=None, run_opts=None,
                             **kwargs):
                captured["source"] = source
                captured["run_opts"] = run_opts
                captured["extra"] = kwargs
                return EncoderDecoderASR()

            def audio_normalizer(self, signal, _rate):
                return signal

            def transcribe_batch(self, _batch, _lengths):
                return (["speechbrain text"], None)

        fetching = types.ModuleType("speechbrain.utils.fetching")
        fetching.LocalStrategy = types.SimpleNamespace(COPY="copy")
        with mock.patch.dict(sys.modules, {
            **self._modules(EncoderDecoderASR, fetching),
            "torch": self._torch(),
        }), mock.patch.object(worker.os, "makedirs"), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0], 16000)):
            result = worker.run({
                "backend": "speechbrain", "audio": "clip.wav",
                "compute": "cpu",
                "hf_model": "speechbrain/asr-crdnn-rnnlm-librispeech",
            })
        self.assertEqual(captured["source"],
                         "speechbrain/asr-crdnn-rnnlm-librispeech")
        self.assertEqual(captured["run_opts"], {"device": "cpu"})
        # A symlink into the destination needs Developer Mode on Windows; the
        # first download failed with "[WinError 1314] A required privilege is
        # not held by the client".
        self.assertEqual(captured["extra"].get("local_strategy"), "copy")
        self.assertEqual(result["segments"][0]["text"], "speechbrain text")

    def test_an_old_speechbrain_falls_back_to_transcribe_file(self):
        captured = {}

        class EncoderDecoderASR:
            @staticmethod
            def from_hparams(**_kwargs):
                return EncoderDecoderASR()

            def transcribe_file(self, path, **_kwargs):
                captured["path"] = path
                return "old api text"

        with mock.patch.dict(sys.modules, self._modules(EncoderDecoderASR)), \
                mock.patch.object(worker.os, "makedirs"):
            result = worker.run({
                "backend": "speechbrain", "audio": "clip.wav",
                "compute": "cpu", "hf_model": "speechbrain/recipe",
            })
        self.assertEqual(captured["path"], "clip.wav")
        self.assertEqual(result["segments"][0]["text"], "old api text")

    def test_a_missing_recipe_is_reported_clearly(self):
        with mock.patch.dict(sys.modules, self._modules(object())):
            with self.assertRaises(RuntimeError) as caught:
                worker.run({"backend": "speechbrain", "audio": "clip.wav"})
        self.assertIn("SpeechBrain", str(caught.exception))


class ErrorReportingTest(unittest.TestCase):
    """A gated repository reports a bare 401, which tells the user nothing they
    can act on; the worker explains it instead."""

    def test_a_gated_repository_is_explained(self):
        text = worker._explain(Exception("401 Client Error: Unauthorized"))
        self.assertIn("gated", text.lower())
        self.assertIn("HF_TOKEN", text)

    def test_an_ordinary_failure_is_left_alone(self):
        text = worker._explain(RuntimeError("the model exploded"))
        self.assertEqual(text, "RuntimeError: the model exploded")

    def test_progress_bars_and_warnings_are_dropped_from_the_message(self):
        text = worker._explain(RuntimeError(
            "Fetching 6 files: 100%|##########| 6/6 [03:02<00:00, 32.05s/it]\n"
            "[transformers] `torch_dtype` is deprecated! Use `dtype` instead!"
        ))
        self.assertNotIn("Fetching", text)
        self.assertNotIn("deprecated", text)
        self.assertTrue(text.strip())

    def test_every_backend_is_reachable(self):
        for backend in worker._RUNNERS:
            self.assertTrue(worker._RUNNERS[backend])

    def test_an_unknown_backend_is_still_reported(self):
        with self.assertRaises(RuntimeError) as caught:
            worker.run({"backend": "telepathy"})
        self.assertIn("telepathy", str(caught.exception))


class AudioDecodingTest(unittest.TestCase):
    """The transformers pipeline refuses to resample without torchaudio, so the
    worker decodes to the 16 kHz every ASR model expects itself."""

    def test_resampling_changes_the_number_of_samples(self):
        samples = list(range(8000))
        resampled = worker._resample(samples, 8000, 16000)
        self.assertEqual(len(resampled), 16000)

    def test_resampling_to_the_same_rate_is_a_no_op(self):
        samples = list(range(10))
        self.assertEqual(list(worker._resample(samples, 16000, 16000)),
                         samples)

    def test_decoding_asks_soundfile_and_then_resamples(self):
        import numpy as np

        def fake_read(path, always_2d=False):
            return np.arange(44100, dtype="float32"), 44100

        soundfile = types.ModuleType("soundfile")
        soundfile.read = fake_read
        with mock.patch.dict(sys.modules, {"soundfile": soundfile}):
            samples, rate = worker._decode_audio("clip.wav")
        self.assertEqual(rate, 16000)
        self.assertEqual(len(samples), 16000)

    def test_a_stereo_recording_is_mixed_down(self):
        import numpy as np

        def fake_read(path, always_2d=False):
            return np.ones((16000, 2), dtype="float32"), 16000

        soundfile = types.ModuleType("soundfile")
        soundfile.read = fake_read
        with mock.patch.dict(sys.modules, {"soundfile": soundfile}):
            samples, _rate = worker._decode_audio("clip.wav")
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(len(samples), 16000)


class TokenBudgetTest(unittest.TestCase):
    """A decoder-only speech model writes its transcript one token at a time
    and stops at its own end-of-sequence token, so the budget it is given is a
    runaway guard - never the thing that ends a complete transcript.

    A 54.41-second Hindi recording really was cut off mid-sentence: it stopped
    on token 432 of the 435 the old eight-tokens-per-second rate allowed, while
    105 words of ordinary Hindi need 432 tokens.  Devanagari costs a tokenizer
    roughly one token per character, so an English-sized budget truncates it.
    """

    def _budget(self, seconds):
        with mock.patch.object(worker, "_audio_duration",
                               return_value=seconds):
            return worker._max_new_tokens("clip.wav")

    def test_the_recording_that_was_cut_off_is_no_longer_capped_short(self):
        self.assertGreater(self._budget(54.41), 435)
        self.assertGreaterEqual(self._budget(54.41), 1000)

    def test_the_rate_is_several_times_what_the_worst_script_needs(self):
        # Hindi measured at about eight tokens per second of speech before the
        # cap ended it, so a budget that only just covers it is not enough.
        self.assertGreaterEqual(worker.TOKENS_PER_SECOND, 32)

    def test_a_short_clip_still_gets_the_floor(self):
        for seconds in (0.5, 2.0, 10.0):
            self.assertGreaterEqual(self._budget(seconds),
                                    worker.MIN_NEW_TOKENS)

    def test_a_long_recording_is_still_bounded(self):
        self.assertEqual(self._budget(6 * 3600), worker.MAX_NEW_TOKENS)

    def test_an_unmeasurable_recording_is_given_the_ceiling(self):
        # A recording whose length cannot be read (a video container, say) is
        # exactly the one a floor-sized budget would cut short.
        self.assertEqual(self._budget(0), worker.MAX_NEW_TOKENS)

    def test_the_budget_grows_with_the_recording(self):
        self.assertLess(self._budget(60), self._budget(600))

    def test_a_video_container_is_measured_instead_of_read_as_zero(self):
        # SoundFile cannot open an .mp4; before the container fallback that
        # read as "no length at all" and the recording was given only the
        # floor.  PyAV knows a container's duration without decoding a frame.
        seconds = 300.0

        class _Stream:
            type = "audio"
            duration = None
            time_base = None

        class _Container:
            duration = int(seconds * 1_000_000)
            streams = [_Stream()]

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        soundfile = types.ModuleType("soundfile")

        def boom(*_a, **_k):
            raise RuntimeError("SoundFile cannot read this container")

        soundfile.info = boom
        soundfile.read = boom
        av = types.ModuleType("av")
        av.open = lambda *_a, **_k: _Container()
        with mock.patch.dict(sys.modules, {"soundfile": soundfile, "av": av}):
            self.assertAlmostEqual(worker._audio_duration("clip.mp4"), seconds)
            self.assertEqual(worker._max_new_tokens("clip.mp4"),
                             int(seconds * worker.TOKENS_PER_SECOND))


class TranscriptTruncationTest(unittest.TestCase):
    """A transcript that stopped on the cap has to be reported: written to the
    project on its own it is indistinguishable from a finished one, which is
    how a half-transcribed recording reached the user unnoticed."""

    def test_generation_that_filled_the_budget_is_flagged(self):
        note = worker._truncation_note(_FakeTensor([0] * 435), 435)
        self.assertIn("435", note)
        self.assertIn("incomplete", note.lower())

    def test_generation_that_stopped_on_its_own_is_not_flagged(self):
        self.assertEqual(worker._truncation_note(_FakeTensor([0] * 100), 435),
                         "")

    def test_a_plain_list_of_token_ids_is_understood(self):
        self.assertEqual(worker._truncation_note([0] * 100, 435), "")
        self.assertIn("incomplete",
                      worker._truncation_note([0] * 435, 435).lower())

    def test_something_with_no_length_is_left_alone(self):
        self.assertEqual(worker._truncation_note(None, 435), "")


class AudioLevelTest(unittest.TestCase):
    """A recording is scaled to a healthy level before a model sees it.

    Both detectors that can silently drop words judge a quiet recording as
    silence: Silero VAD, which faster-whisper's ``vad_filter`` runs, and
    Whisper's own ``no_speech_threshold``, which skips a whole 30-second
    window below the log-probability floor.
    """

    def test_a_quiet_recording_is_lifted_to_the_target_peak(self):
        scaled = worker._normalise_level([0.02, -0.05, 0.1])
        loudest = float(max(abs(float(s)) for s in scaled))
        self.assertAlmostEqual(loudest, worker.TARGET_PEAK, places=5)

    def test_only_the_level_changes_not_the_shape(self):
        original = [0.1, -0.2, 0.4]
        scaled = worker._normalise_level(original)
        factors = [float(s) / o for o, s in zip(original, scaled)]
        self.assertAlmostEqual(factors[0], factors[1], places=6)
        self.assertAlmostEqual(factors[1], factors[2], places=6)

    def test_a_loud_recording_is_not_pushed_past_full_scale(self):
        scaled = worker._normalise_level([-1.0, 1.0])
        loudest = float(max(abs(float(s)) for s in scaled))
        self.assertLessEqual(loudest, 1.0)
        self.assertGreater(loudest, 0.9)

    def test_silence_stays_silent(self):
        self.assertEqual([float(s) for s in worker._normalise_level([0.0, 0.0])],
                         [0.0, 0.0])

    def test_decoding_returns_a_scaled_recording(self):
        import numpy as np

        def fake_read(path, always_2d=False):
            return np.arange(16000, dtype="float32") / 16000.0, 16000

        soundfile = types.ModuleType("soundfile")
        soundfile.read = fake_read
        with mock.patch.dict(sys.modules, {"soundfile": soundfile}):
            samples, _rate = worker._decode_audio("clip.wav")
        self.assertAlmostEqual(float(abs(samples).max()), worker.TARGET_PEAK,
                               places=5)


class LongRecordingTest(unittest.TestCase):
    """Qwen3-ASR was handed a 29-minute chapter in a single request: 22,969
    tokens of self-attention asked for a 31.5 GiB tensor on a 12 GB card, and
    the run died with "Tried to allocate 31.45 GiB" before a word was written.

    A recording too long for one call is cut into pieces that fit, and the
    pieces are handed to the processor as paths - exactly what a processor has
    always been given here.
    """

    RATE = 1000

    @staticmethod
    def _torch():
        module = types.ModuleType("torch")
        module.no_grad = lambda: mock.MagicMock(
            __enter__=lambda self: None, __exit__=lambda *a: False
        )
        return module

    def test_a_recording_that_fits_is_handed_over_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(worker, "_audio_duration", return_value=60.0):
                pieces = worker._recording_pieces("clip.wav", folder)
        self.assertEqual(pieces, ["clip.wav"])

    def test_a_recording_whose_length_cannot_be_read_is_not_cut_blind(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(worker, "_audio_duration", return_value=0.0):
                pieces = worker._recording_pieces("clip.mp4", folder)
        self.assertEqual(pieces, ["clip.mp4"])

    def test_a_long_recording_is_cut_into_pieces_that_fit(self):
        import wave

        samples = [0.25] * (400 * self.RATE)  # 400 s: longer than one piece
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(worker, "_audio_duration",
                                   return_value=400.0), \
                    mock.patch.object(worker, "_decode_audio",
                                      return_value=(samples, self.RATE)):
                pieces = worker._recording_pieces("chapter.mp3", folder)
            self.assertEqual(len(pieces), 3)  # ceil(400 / 180)
            self.assertEqual(len(set(pieces)), 3)
            total = 0
            for path in pieces:
                self.assertTrue(path.startswith(folder))
                with wave.open(path, "rb") as handle:
                    self.assertEqual(handle.getnchannels(), 1)
                    self.assertEqual(handle.getframerate(), self.RATE)
                    self.assertLessEqual(
                        handle.getnframes(),
                        int(worker.AUDIO_CHUNK_SECONDS * self.RATE),
                    )
                    total += handle.getnframes()
            self.assertEqual(total, len(samples))

    def test_the_pieces_share_the_recording_instead_of_leaving_a_stub(self):
        # 400 seconds in 180-second pieces would be 180, 180 and a 40-second
        # tail - a whole model call for a tail - so the split is equal.
        rate = 16000
        samples = [0.0] * (400 * rate)
        pieces = worker._split_audio(samples, rate)
        self.assertEqual(len(pieces), 3)
        for piece in pieces:
            self.assertLessEqual(len(piece), 180 * rate)
            self.assertGreater(len(piece), 100 * rate)
        self.assertEqual(sum(len(p) for p in pieces), len(samples))

    def test_a_recording_that_fits_is_one_untouched_piece(self):
        samples = [0.0] * 16000
        pieces = worker._split_audio(samples, 16000)
        self.assertEqual(len(pieces), 1)
        self.assertIs(pieces[0], samples)

    def test_a_piece_is_written_as_a_readable_wav(self):
        import wave

        rate = 16000
        samples = [min(1.0, i / rate) for i in range(rate)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "piece.wav")
            worker._write_wav(path, samples, rate)
            with wave.open(path, "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), rate)
                self.assertEqual(handle.getnframes(), rate)

    def test_every_piece_of_a_long_recording_reaches_the_model(self):
        """The pieces' text comes back joined, in order, as one transcript."""
        seen = []

        class FakeProcessor:
            def apply_transcription_request(self, audio=None, language=None):
                seen.append(audio)
                return _FakeInputs(input_ids=_FakeTensor([1, 2, 3]))

            def decode(self, _tokens, return_format=None):
                return [f"piece{len(seen)}"]

        class FakeModel:
            dtype = "float32"
            device = "cpu"

            def generate(self, **_kwargs):
                return _FakeTensor([1, 2, 3, 4, 5])

        module = _fake_transformers(
            processor_factory=lambda *a, **k: FakeProcessor(),
            multimodal_factory=lambda *a, **k: FakeModel(),
        )
        with mock.patch.dict(sys.modules, {"transformers": module,
                                           "torch": self._torch()}), \
                mock.patch.object(worker, "_audio_duration",
                                  return_value=400.0), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0] * (400 * 1600), 1600)), \
                mock.patch.object(worker, "_max_new_tokens", return_value=256):
            result = worker.run({
                "backend": "transformers", "audio": "chapter.mp3",
                "runner": "transcription_request", "language": "hi",
                "hf_model": "Qwen/Qwen3-ASR-1.7B-hf",
            })
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3)  # three different piece files
        self.assertEqual(result["segments"][0]["text"],
                         "piece1 piece2 piece3")
        self.assertFalse(result["timestamps"])

    def test_the_chat_runner_cuts_its_recording_into_pieces_too(self):
        seen = []

        class FakeTokenizer:
            def apply_chat_template(self, *_a, **_k):
                return "<prompt>"

            def decode(self, _tokens, skip_special_tokens=False):
                return f"piece{len(seen)}"

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def __call__(self, _prompt, audio, **_kwargs):
                seen.append(audio)
                return _FakeInputs(input_ids=_FakeTensor([1, 2, 3]))

        class FakeModel:
            device = "cpu"
            dtype = "float32"

            def generate(self, **_kwargs):
                return _FakeTensor([1, 2, 3, 9, 9])

        module = _fake_transformers(
            processor_factory=lambda *a, **k: FakeProcessor(),
            multimodal_factory=lambda *a, **k: FakeModel(),
        )
        module.AutoModelForSpeechSeq2Seq = module.AutoModelForMultimodalLM
        with mock.patch.dict(sys.modules, {"transformers": module,
                                           "torch": self._torch()}), \
                mock.patch.object(worker, "_decode_audio",
                                  return_value=([0.0] * (400 * 1600), 1600)):
            result = worker.run({
                "backend": "transformers", "audio": "chapter.mp3",
                "runner": "chat",
                "hf_model": "ibm-granite/granite-4.0-1b-speech",
            })
        self.assertEqual(len(seen), 3)
        self.assertEqual(result["segments"][0]["text"],
                         "piece1 piece2 piece3")


class OutOfMemoryHalvingTest(unittest.TestCase):
    """A piece that still does not fit is cut in half and each half
    transcribed, so a memory failure costs time rather than transcript - and
    when even the shortest piece will not fit, the user is told what to do
    about it instead of being shown PyTorch's allocator dump.
    """

    @staticmethod
    def _memory_error():
        return RuntimeError(
            "OutOfMemoryError: CUDA out of memory. Tried to allocate "
            "31.45 GiB. GPU 0 has a total capacity of 12.00 GiB."
        )

    def test_a_piece_that_fits_is_transcribed_directly(self):
        def must_not_be_halved(_piece):
            self.fail("a piece that fits must not be cut")

        self.assertEqual(
            worker._transcribe_in_halves("clip.wav",
                                         lambda p: ("the text", ""),
                                         must_not_be_halved, "cpu"),
            ("the text", ""),
        )

    def test_a_memory_failure_cuts_the_piece_and_keeps_every_word(self):
        def transcribe(piece):
            if len(piece) > 1:
                raise self._memory_error()
            return (f"<{piece}>", "")

        def halve(piece):
            middle = len(piece) // 2
            return (piece[:middle], piece[middle:]) if middle else None

        text, note = worker._transcribe_in_halves("abcd", transcribe, halve,
                                                  "cpu")
        self.assertEqual(text, "<a> <b> <c> <d>")
        self.assertEqual(note, "")

    def test_any_other_failure_is_reported_rather_than_hidden(self):
        def transcribe(_piece):
            raise ValueError("the checkpoint is corrupt")

        with self.assertRaises(ValueError):
            worker._transcribe_in_halves("clip.wav", transcribe,
                                         lambda p: None, "cpu")

    def test_a_piece_that_cannot_be_cut_says_what_to_do(self):
        def transcribe(_piece):
            raise self._memory_error()

        with self.assertRaises(RuntimeError) as caught:
            worker._transcribe_in_halves("clip.wav", transcribe,
                                         lambda p: None, "cuda")
        message = str(caught.exception)
        self.assertIn("out of memory", message.lower())
        self.assertIn("Compute to CPU", message)
        self.assertIn("30-second pieces", message)
        # The original message is kept for the log, not for the dialog.
        self.assertNotIn("31.45", message)
        self.assertIn("31.45", str(caught.exception.__cause__))


class ModelDtypeTest(unittest.TestCase):
    """Loading a 1.7B checkpoint as float32 put 7.3 GiB on a 12 GB card -
    more than twice the weights as they are published - and made every tensor
    allocated afterwards twice as large as well.  The checkpoint's own
    bfloat16 is what it was trained with, so that is what it is loaded as.
    """

    @staticmethod
    def _module(version):
        module = types.ModuleType("transformers")
        module.__version__ = version
        return module

    def test_transformers_five_is_told_the_new_name(self):
        with mock.patch.dict(sys.modules,
                             {"transformers": self._module("5.18.0")}):
            self.assertEqual(worker._dtype_kwargs(), {"dtype": "auto"})

    def test_transformers_four_is_told_the_name_it_knows(self):
        with mock.patch.dict(sys.modules,
                             {"transformers": self._module("4.57.1")}):
            self.assertEqual(worker._dtype_kwargs(), {"torch_dtype": "auto"})

    def test_a_version_that_cannot_be_read_keeps_the_default(self):
        with mock.patch.dict(sys.modules,
                             {"transformers": self._module("dev")}):
            self.assertEqual(worker._dtype_kwargs(), {})
        with mock.patch.dict(sys.modules, {"transformers": object()}):
            self.assertEqual(worker._dtype_kwargs(), {})

    def test_the_dtype_reaches_from_pretrained(self):
        seen = {}

        class FakeAuto:
            @staticmethod
            def from_pretrained(hf_id, **kwargs):
                seen["hf_id"] = hf_id
                seen.update(kwargs)
                return "model"

        with mock.patch.dict(sys.modules,
                             {"transformers": self._module("5.18.0")}):
            model = worker._load_repository_model(
                FakeAuto, "Qwen/Qwen3-ASR-1.7B-hf",
                {"trust_remote_code": False}, "cuda",
            )
        self.assertEqual(model, "model")
        self.assertEqual(seen["dtype"], "auto")
        self.assertEqual(seen["device_map"], "cuda:0")
        self.assertIs(seen["trust_remote_code"], False)


if __name__ == "__main__":
    unittest.main()
