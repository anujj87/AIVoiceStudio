"""Resumable downloads: a stale or resumable ``.part`` must never fail the install.

The reported bug: installing FFmpeg from the audio-description wizard failed
with ``416 Client Error: Requested Range Not Satisfiable`` once gyan.dev
replaced ``ffmpeg-9.0-essentials_build.zip`` with ``ffmpeg-9.1-...`` behind the
same ``ffmpeg-release-essentials.zip`` URL.  The ``.part`` file left by the
older, longer build asked for a byte the new file no longer has, so the
download could never finish.  These tests drive ``download_file`` with a fake
session instead of the network.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import requests

from ai_voice_studio.tts import downloader

URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
NAME = "ffmpeg-release-essentials.zip"


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"", length: int | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        self.closed = False

    def iter_content(self, chunk_size: int = 1):
        step = max(1, chunk_size)
        for start in range(0, len(self._body), step):
            yield self._body[start:start + step]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: bad status", response=self)

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "headers": dict(headers or {})})
        return self.responses.pop(0)

    def close(self) -> None:
        pass


class DownloadResumeTest(unittest.TestCase):
    def _run(self, responses: list[FakeResponse], part_bytes: bytes | None) -> tuple[str, FakeSession, str]:
        root = tempfile.mkdtemp(prefix="aivs_dl_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        part = os.path.join(root, NAME + ".part")
        if part_bytes is not None:
            with open(part, "wb") as fh:
                fh.write(part_bytes)
        session = FakeSession(responses)
        with mock.patch.object(downloader.requests, "Session", return_value=session):
            dest = downloader.download_file(URL, root, filename=NAME)
        return dest, session, part

    def test_a_stale_part_from_an_older_build_is_restarted(self):
        """416 with a longer .part on disk: drop it and fetch from byte zero."""
        body = b"ffmpeg-9.1-essentials"
        dest, session, part = self._run(
            [FakeResponse(416), FakeResponse(200, body)],
            part_bytes=b"x" * (len(body) + 4096),
        )
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["headers"], {"Range": f"bytes={len(body) + 4096}-"})
        self.assertEqual(session.calls[1]["headers"], {})
        with open(dest, "rb") as fh:
            self.assertEqual(fh.read(), body)
        self.assertFalse(os.path.exists(part))

    def test_a_server_that_ignores_the_range_does_not_corrupt_the_file(self):
        """A 200 answer to a Range request replaces the old bytes."""
        body = b"whole new build"
        dest, session, part = self._run(
            [FakeResponse(200, body), FakeResponse(200, body)],
            part_bytes=b"old bytes",
        )
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1]["headers"], {})
        with open(dest, "rb") as fh:
            self.assertEqual(fh.read(), body)
        self.assertFalse(os.path.exists(part))

    def test_a_real_resume_appends_to_the_part(self):
        """206 keeps the bytes already on disk and appends the rest."""
        dest, session, part = self._run(
            [FakeResponse(206, b"def", length=3)],
            part_bytes=b"abc",
        )
        self.assertEqual(session.calls[0]["headers"], {"Range": "bytes=3-"})
        with open(dest, "rb") as fh:
            self.assertEqual(fh.read(), b"abcdef")
        self.assertFalse(os.path.exists(part))


if __name__ == "__main__":
    unittest.main()
