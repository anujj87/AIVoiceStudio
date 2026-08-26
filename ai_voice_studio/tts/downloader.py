"""Model downloads.

Downloads official model artifacts (tar.bz2) with:
* resume support (Range requests + .part files),
* progress callbacks (called from the worker thread),
* cancellation via a threading.Event,
* safe extraction with path-traversal protection,
* installation into the per-user models folder and state tracking.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from typing import Callable, Dict, List, Optional

import requests

from .. import paths, constants
from ..constants import DOWNLOAD_CHUNK, DOWNLOAD_TIMEOUT
from ..tts import catalog
from ..tts.models import ModelStore

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


class DownloadCancelled(Exception):
    """Raised inside the worker when the user cancels a download."""


class DownloadError(Exception):
    """Raised on download failure."""


def download_file(
    url: str,
    dest_dir: str,
    filename: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Download ``url`` to ``dest_dir`` (resumable). Returns final file path."""
    os.makedirs(dest_dir, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    dest = os.path.join(dest_dir, filename)
    part = dest + ".part"

    headers = {}
    downloaded = 0
    if os.path.exists(part):
        downloaded = os.path.getsize(part)
        headers["Range"] = f"bytes={downloaded}-"

    session = requests.Session()
    try:
        resp = session.get(
            url,
            headers=headers,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0)) + downloaded
        mode = "ab" if downloaded > 0 else "wb"

        with open(part, mode) as fh:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if cancel_event and cancel_event.is_set():
                    raise DownloadCancelled()
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(filename, downloaded, total)
    except requests.RequestException as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc
    finally:
        session.close()

    if os.path.exists(dest):
        os.remove(dest)
    os.replace(part, dest)
    return dest


def hf_file_list(
    repo: str,
    prefixes: Optional[List[str]] = None,
    exact: Optional[List[str]] = None,
) -> List[str]:
    """List files of a HuggingFace model repo matching ``prefixes``/``exact``.

    Used for raw-file model variants that are served
    as loose files on the Hub instead of a tar.bz2 archive. Returns relative
    paths (e.g. ``fp16/voice_clone/talker_decode.onnx``).
    """
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        resp = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "ai-voice-studio"},
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise DownloadError(f"Could not list files of {repo}: {exc}") from exc

    siblings = data.get("siblings", [])
    result: List[str] = []
    for s in siblings:
        name = s.get("rfilename", "")
        if not name:
            continue
        name = name.replace("\\", "/")
        if exact:
            if name in exact:
                result.append(name)
        elif prefixes:
            if any(name.startswith(p) for p in prefixes):
                result.append(name)
        else:
            result.append(name)
    return sorted(set(result))


def safe_extract_tarball(
    tarball_path: str,
    dest_dir: str,
    strip_top: bool = True,
) -> None:
    """Extract a tar.bz2 archive; returns the directory containing the payload.

    The archives we ship are self-contained and have a single top-level folder
    (e.g. ``vits-piper-en_US-lessac-medium/``). With ``strip_top`` the contents
    of that folder are moved into ``dest_dir`` so engine paths are stable.
    """
    os.makedirs(dest_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="aivs_")
    try:
        with tarfile.open(tarball_path, "r:bz2") as tar:
            for member in tar.getmembers():
                name = member.name.replace("\\", "/")
                parts = name.split("/")
                rel = "/".join(parts[1:]) if len(parts) > 1 else name
                if rel.startswith("..") or rel.startswith("/"):
                    raise DownloadError(f"Unsafe path in archive: {name}")
                if not rel or rel.endswith("/"):
                    continue
            tar.extractall(tmp, filter="data")

        entries = os.listdir(tmp)
        src = tmp
        if strip_top and len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
            src = os.path.join(tmp, entries[0])

        for entry in os.listdir(src):
            s = os.path.join(src, entry)
            d = os.path.join(dest_dir, entry)
            if os.path.isdir(s):
                shutil.move(s, d)
            elif os.path.isfile(s):
                shutil.move(s, d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class ModelDownloader:
    """Downloads and removes catalog variants using a ModelStore."""

    def __init__(self, store: ModelStore):
        self.store = store

    def _artifact_url(self, tts: Dict, artifact: str) -> str:
        base = tts.get("artifact_base", "")
        return base.rstrip("/") + "/" + artifact

    def download_variant(
        self,
        tts_id: str,
        lang_code: str,
        variant_id: str,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        tts = catalog.find_tts(tts_id)
        if not tts:
            raise DownloadError(f"Unknown TTS: {tts_id}")
        variant = catalog.find_variant(tts, lang_code, variant_id)
        if not variant:
            raise DownloadError(f"Unknown variant: {tts_id}/{lang_code}/{variant_id}")

        dest = self.store.variant_dir(tts_id, lang_code, variant_id)
        os.makedirs(dest, exist_ok=True)

        artifacts = self._variant_artifacts(variant)
        for artifact in artifacts:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled()
            url = self._artifact_url(tts, artifact)
            tarball = download_file(
                url,
                dest,
                filename=os.path.basename(artifact),
                progress=progress,
                cancel_event=cancel_event,
            )

            # Extract .tar.bz2 archives so model.onnx / tokens.txt are
            # available immediately (not just the raw archive).
            if tarball.endswith(".tar.bz2") or tarball.endswith(".tar.gz"):
                if progress:
                    progress(f"Extracting {os.path.basename(tarball)}", 0, 0)
                safe_extract_tarball(tarball, dest)
                # Keep the archive so future calls can skip re-download
                # (the _variant_artifacts check already handles this).

        # Download shared artifacts (espeak-ng-data, vocoders, etc.)
        self._download_extras(tts, tts.get("shared_artifacts", []), progress, cancel_event)

        # Organize per-voice files
        self._organize_voice(dest, variant, artifacts)

        # Download HF files for raw-file variants
        hf_files = variant.get("hf_files")
        if hf_files:
            hf_prefixes = variant.get("hf_prefixes", [])
            all_files = hf_file_list(variant.get("hf_repo", ""), prefixes=hf_prefixes)
            # Also download exact files
            if hf_files:
                for f in hf_files:
                    if f not in all_files:
                        all_files.append(f)
            for fname in all_files:
                if cancel_event and cancel_event.is_set():
                    raise DownloadCancelled()
                hf_base = f"https://huggingface.co/{variant.get('hf_repo', '')}/resolve/main/"
                relative_dir = os.path.dirname(fname)
                target_dir = os.path.join(dest, relative_dir) if relative_dir else dest
                os.makedirs(target_dir, exist_ok=True)
                download_file(
                    hf_base + fname,
                    target_dir,
                    filename=os.path.basename(fname),
                    progress=progress,
                    cancel_event=cancel_event,
                )

        self.store.mark_installed(tts_id, lang_code, variant_id, dest)

    def _variant_artifacts(self, variant: Dict) -> List[str]:
        artifacts: List[str] = []
        if variant.get("artifact"):
            artifacts.append(variant["artifact"])
        for voice in variant.get("voices", []):
            if voice.get("artifact") and voice["artifact"] not in artifacts:
                artifacts.append(voice["artifact"])
        return artifacts

    def _download_extras(
        self,
        tts: Dict,
        extras: List[Dict],
        progress: Optional[ProgressCallback],
        cancel_event: Optional[threading.Event],
    ) -> None:
        for extra in extras:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled()
            base = extra.get("base") or tts.get("artifact_base", "")
            url = base.rstrip("/") + "/" + extra["artifact"]
            filename = extra.get("filename") or extra["artifact"]
            dest_dir = os.path.join(paths.models_dir(), *extra.get("dest", "shared").split("/"))
            os.makedirs(dest_dir, exist_ok=True)

            if extra.get("kind") == "file":
                download_file(
                    url,
                    dest_dir,
                    filename=filename,
                    progress=progress,
                    cancel_event=cancel_event,
                )
            else:
                # tarball: skip re-extraction when already populated.
                if os.path.isdir(dest_dir) and os.listdir(dest_dir):
                    if progress:
                        progress(filename, 1, 1)
                    continue
                cache = os.path.join(paths.models_dir(), "shared", ".cache")
                os.makedirs(cache, exist_ok=True)
                tarball = download_file(
                    url,
                    cache,
                    filename=filename,
                    progress=progress,
                    cancel_event=cancel_event,
                )
                safe_extract_tarball(tarball, dest_dir)
                os.remove(tarball)
            if progress:
                progress(filename, 1, 1)

    def _organize_voice(self, dest: str, variant: Dict, artifact: str) -> None:
        """Move extracted files for one artifact into ``dest/<voice_id>/``."""
        voice = next(
            (v for v in variant.get("voices", []) if v.get("artifact") == artifact), None
        )
        target = dest if not voice else os.path.join(dest, voice["id"])
        os.makedirs(target, exist_ok=True)
        tmp = os.path.join(dest, "tmp_extract")
        if not os.path.isdir(tmp):
            return
        for entry in os.listdir(tmp):
            src = os.path.join(tmp, entry)
            dst = os.path.join(target, entry)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
            elif os.path.isfile(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
        shutil.rmtree(tmp, ignore_errors=True)

    def remove_variant(self, tts_id: str, lang_code: str, variant_id: str) -> bool:
        return self.store.remove_variant(tts_id, lang_code, variant_id)

    # -- shared espeak-ng-data (needed by raw custom piper voices) ----------
    def ensure_shared_espeak_data(
        self,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        shared = os.path.join(paths.models_dir(), "shared", "espeak-ng-data")
        if os.path.isdir(shared) and os.listdir(shared):
            return shared
        tts = {"artifact_base": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"}
        artifact = "espeak-ng-data.tar.bz2"
        os.makedirs(os.path.dirname(shared), exist_ok=True)
        tarball = download_file(
            self._artifact_url(tts, artifact),
            os.path.dirname(shared),
            filename=artifact,
            progress=progress,
            cancel_event=cancel_event,
        )
        safe_extract_tarball(tarball, shared)
        os.remove(tarball)
        return shared
