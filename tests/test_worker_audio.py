"""Reading the reference audio without a system FFmpeg.

TorchAudio 2.9 and later decode through TorchCodec (its ``backend`` argument
is accepted and ignored), and TorchCodec on Windows needs FFmpeg's *shared*
libraries.  F5-TTS loads its reference clip with ``torchaudio.load``, so on a
computer without FFmpeg its preview died with "Could not load
libtorchcodec".  The Voice Lab worker now backs ``torchaudio.load``/``save``
with ``soundfile`` - which ships libsndfile inside its own wheel - but only
after measuring that TorchCodec is really broken, so a machine with FFmpeg
keeps torchaudio's own decoder.

These tests run the worker module with fake torchaudio/soundfile/torch
objects, so neither dependency has to be installed here.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from ai_voice_studio import voicelab  # noqa: E402

WORKER = os.path.join(
    os.path.dirname(os.path.abspath(voicelab.__file__)), "worker.py"
)


class _FakeTensor:
    """Just enough of a torch tensor for the shim's callers."""

    def __init__(self, data):
        import numpy as np

        self.data = np.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype is not None else self.data

    def __eq__(self, other):  # pragma: no cover - convenience for assertions
        return self.data == other


class _FakeTorch:
    """``torch.from_numpy`` only, which is all the shim needs."""

    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)


class _FakeSoundfile:
    """A minimal stand-in for ``soundfile`` backed by the stdlib ``wave``.

    ``read``/``write`` mirror the real signatures the shim relies on
    (``start``/``frames``/``dtype``/``always_2d``, and a ``subtype`` for
    16-bit output), so the shim's argument handling is what is under test.
    """

    def __init__(self):
        self.written = []

    def read(self, path, start=0, frames=-1, dtype="float32", always_2d=True):
        target = path if hasattr(path, "read") else str(path)
        with wave.open(target, "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            handle.setpos(int(start))
            raw = handle.readframes(handle.getnframes() if frames in (-1, None)
                                    else int(frames))
        if width != 2:
            raise RuntimeError("the fake reader handles 16-bit PCM only")
        count = len(raw) // 2
        samples = np.array(struct.unpack(f"<{count}h", raw), dtype=np.float32)
        samples = (samples / 32768.0).astype(np.float32)
        if always_2d:
            samples = samples.reshape(-1, channels)
        elif channels == 1:
            samples = samples.reshape(-1)
        return samples, rate

    def write(self, path, data, samplerate, subtype=None):
        self.written.append((str(path), np.asarray(data), int(samplerate),
                             subtype))
        array = np.asarray(data)
        if array.ndim == 1:
            array = array[:, None]
        channels = array.shape[1]
        clipped = np.clip(array, -1.0, 1.0) if array.dtype.kind == "f" \
            else array.astype(np.int16)
        pcm = (clipped * 32767.0).astype("<i2") if array.dtype.kind == "f" \
            else clipped.astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(int(channels))
            handle.setsampwidth(2)
            handle.setframerate(int(samplerate))
            handle.writeframes(pcm.tobytes())


class _FakeTorchaudio:
    """TorchAudio whose (TorchCodec) decoder raises, like a machine without FFmpeg."""

    def __init__(self, works=False):
        self.works = works
        self.calls = []

    def load(self, uri, *args, **kwargs):
        self.calls.append(("load", str(uri)))
        if not self.works:
            raise RuntimeError(
                "Could not load libtorchcodec: "
                "libtorchcodec_core9.dll was not found"
            )
        return "torchaudio-decoded", 16000

    def save(self, uri, src, sample_rate, *args, **kwargs):
        self.calls.append(("save", str(uri)))
        if not self.works:
            raise RuntimeError("Could not load libtorchcodec")


class AudioShimTest(unittest.TestCase):
    def setUp(self):
        import importlib.util as ilu

        spec = ilu.spec_from_file_location("aivs_voicelab_worker_audio", WORKER)
        self.worker = ilu.module_from_spec(spec)
        spec.loader.exec_module(self.worker)
        self.sf = _FakeSoundfile()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    # -- helpers ------------------------------------------------------------
    def _write_wav(self, name="ref.wav", channels=1, frames=100, rate=16000,
                   value=0.5):
        """A real 16-bit WAV file on disk (what the engines load)."""
        path = os.path.join(self.tmp.name, name)
        signal = np.full((frames, channels), value, dtype=np.float32)
        self.sf.write(path, signal, rate)
        return path

    # -- the probe ----------------------------------------------------------
    def test_the_probe_recognises_a_working_decoder(self):
        self.assertTrue(self.worker._torchcodec_works(_FakeTorchaudio(works=True)))

    def test_the_probe_recognises_a_broken_decoder(self):
        torchaudio = _FakeTorchaudio(works=False)
        self.assertFalse(self.worker._torchcodec_works(torchaudio))
        # The probe must not leave the temporary wav behind.
        probe = os.path.join(tempfile.gettempdir(), "aivs_voicelab_probe.wav")
        self.assertFalse(os.path.isfile(probe))

    # -- the swap -----------------------------------------------------------
    def test_a_broken_decoder_is_replaced_by_soundfile(self):
        torchaudio = _FakeTorchaudio(works=False)
        note = self.worker._install_audio_io_shim(
            torchaudio, soundfile=self.sf, torch=_FakeTorch
        )
        self.assertTrue(note)
        self.assertIn("soundfile", note)
        # The replacement must work on a real file.
        tensor, rate = torchaudio.load(self._write_wav())
        self.assertEqual(rate, 16000)
        self.assertEqual(tensor.shape, (1, 100))

    def test_a_working_decoder_is_left_alone(self):
        torchaudio = _FakeTorchaudio(works=True)
        note = self.worker._install_audio_io_shim(
            torchaudio, soundfile=self.sf, torch=_FakeTorch
        )
        self.assertIsNone(note)
        self.assertEqual(torchaudio.load("anything"), ("torchaudio-decoded", 16000))

    def test_the_swap_happens_once(self):
        torchaudio = _FakeTorchaudio(works=False)
        self.worker._install_audio_io_shim(
            torchaudio, soundfile=self.sf, torch=_FakeTorch
        )
        replacement = torchaudio.load
        self.assertIsNone(
            self.worker._install_audio_io_shim(
                torchaudio, soundfile=self.sf, torch=_FakeTorch
            )
        )
        self.assertIs(torchaudio.load, replacement)

    def test_the_swap_can_be_switched_off(self):
        torchaudio = _FakeTorchaudio(works=False)
        os.environ[self.worker._AUDIO_SHIM_ENV] = "0"
        self.addCleanup(os.environ.pop, self.worker._AUDIO_SHIM_ENV, None)
        self.assertIsNone(
            self.worker._install_audio_io_shim(
                torchaudio, soundfile=self.sf, torch=_FakeTorch
            )
        )

    def test_without_soundfile_the_note_explains_the_gap(self):
        with mock.patch.object(self.worker, "_import_soundfile",
                               side_effect=ImportError("no soundfile")):
            note = self.worker._install_audio_io_shim(
                _FakeTorchaudio(works=False), torch=_FakeTorch
            )
        self.assertTrue(note)
        self.assertIn("soundfile", note)

    # -- load semantics -----------------------------------------------------
    def test_load_returns_channels_first_float32(self):
        tensor, rate = self.worker._soundfile_load(
            self.sf, _FakeTorch, self._write_wav(channels=2)
        )
        self.assertEqual(rate, 16000)
        self.assertEqual(tensor.shape, (2, 100))
        self.assertEqual(tensor.data.dtype.kind, "f")

    def test_load_can_return_time_first(self):
        tensor, _rate = self.worker._soundfile_load(
            self.sf, _FakeTorch, self._write_wav(channels=2),
            channels_first=False,
        )
        self.assertEqual(tensor.shape, (100, 2))

    def test_load_honours_frame_offset_and_num_frames(self):
        tensor, _rate = self.worker._soundfile_load(
            self.sf, _FakeTorch, self._write_wav(frames=100),
            frame_offset=10, num_frames=25,
        )
        self.assertEqual(tensor.shape, (1, 25))

    def test_load_reads_a_file_object(self):
        path = self._write_wav(frames=40)
        with open(path, "rb") as handle:
            tensor, rate = self.worker._soundfile_load(self.sf, _FakeTorch, handle)
        self.assertEqual((tensor.shape, rate), ((1, 40), 16000))

    # -- save semantics -----------------------------------------------------
    def test_save_writes_channels_first_data(self):
        destination = os.path.join(self.tmp.name, "out.wav")
        self.worker._soundfile_save(
            self.sf, _FakeTorch, destination,
            np.zeros((2, 50), dtype=np.float32), 24000,
        )
        _path, data, rate, _subtype = self.sf.written[-1]
        self.assertEqual(rate, 24000)
        self.assertEqual(data.shape, (50, 2))

    def test_save_accepts_a_tensor(self):
        destination = os.path.join(self.tmp.name, "out_tensor.wav")
        self.worker._soundfile_save(
            self.sf, _FakeTorch, destination, _FakeTensor(np.zeros((1, 30))), 16000
        )
        self.assertEqual(self.sf.written[-1][1].shape, (30, 1))


class ReferenceConversionTest(unittest.TestCase):
    """A non-WAV reference clip is converted, because pydub needs FFmpeg.

    F5-TTS cleans its reference with pydub, which shells out to FFmpeg for
    anything that is not a WAV.  soundfile reads MP3/FLAC/OGG by itself, so
    the clip is converted once and the engine always sees a WAV.
    """

    def setUp(self):
        import importlib.util as ilu

        spec = ilu.spec_from_file_location("aivs_voicelab_worker_refs", WORKER)
        self.worker = ilu.module_from_spec(spec)
        spec.loader.exec_module(self.worker)
        self.sf = _FakeSoundfile()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.converted = []
        self.addCleanup(self._cleanup_converted)

    def _cleanup_converted(self):
        for path in self.converted:
            try:
                os.remove(path)
            except OSError:
                pass

    def _source(self, name, frames=50):
        """A file whose *bytes* are a WAV but whose name is not `.wav`."""
        path = os.path.join(self.tmp.name, name)
        self.sf.write(path, np.full((frames, 1), 0.25, dtype=np.float32), 16000)
        return path

    def _convert(self, path):
        result = self.worker._convert_to_wav(path, soundfile=self.sf)
        if result != path:
            self.converted.append(result)
        return result

    def test_a_non_wav_clip_is_written_as_a_wav(self):
        source = self._source("reference.mp3")
        converted = self._convert(source)
        self.assertNotEqual(converted, source)
        self.assertTrue(converted.lower().endswith(".wav"))
        self.assertTrue(os.path.isfile(converted))
        self.assertGreater(os.path.getsize(converted), 0)
        # ... and it really holds audio the engine can read back.
        tensor, rate = self.worker._soundfile_load(self.sf, _FakeTorch, converted)
        self.assertEqual((tensor.shape, rate), ((1, 50), 16000))

    def test_the_conversion_is_reused(self):
        source = self._source("reference.flac")
        first = self._convert(source)
        writes = len(self.sf.written)
        second = self._convert(source)
        self.assertEqual(first, second)
        self.assertEqual(len(self.sf.written), writes)

    def test_an_unreadable_clip_keeps_its_original_path(self):
        broken = os.path.join(self.tmp.name, "reference.ogg")
        with open(broken, "wb") as handle:
            handle.write(b"not really audio")
        self.assertEqual(self._convert(broken), broken)

    def test_a_missing_clip_is_left_alone(self):
        missing = os.path.join(self.tmp.name, "nope.mp3")
        self.assertEqual(self._convert(missing), missing)

    # -- the backend's use of it -------------------------------------------
    def _backend(self):
        with mock.patch.object(self.worker, "_install_audio_io_shim",
                               return_value=None):
            return self.worker.F5TtsBackend("cpu", {})

    def test_a_wav_reference_is_passed_through_untouched(self):
        backend = self._backend()
        source = self._source("reference.wav")
        writes = len(self.sf.written)
        with mock.patch.object(self.worker, "_import_soundfile",
                               return_value=self.sf):
            self.assertEqual(backend._readable_reference(source), source)
        self.assertEqual(len(self.sf.written), writes)

    def test_the_backend_converts_and_remembers_a_non_wav_reference(self):
        backend = self._backend()
        source = self._source("reference.mp3")
        with mock.patch.object(self.worker, "_import_soundfile",
                               return_value=self.sf):
            first = backend._readable_reference(source)
            second = backend._readable_reference(source)
        self.converted.append(first)
        self.assertTrue(first.lower().endswith(".wav"))
        self.assertEqual(first, second)
        self.assertEqual(backend.notes, [])

    def test_the_backend_explains_a_clip_it_cannot_convert(self):
        backend = self._backend()
        with mock.patch.object(self.worker, "_import_soundfile",
                               side_effect=ImportError("no soundfile")):
            path = backend._readable_reference(self._source("reference.mp3"))
        self.assertTrue(path.endswith(".mp3"))
        self.assertTrue(backend.notes)
        self.assertIn("FFmpeg", backend.notes[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
