"""Model downloads.

Downloads official model artifacts (tar.bz2) with:
* resume support (Range requests + .part files),
* progress callbacks (called from the worker thread),
* cancellation via a threading.Event,
* safe extraction with path-traversal protection,
* installation into the per-user models folder and state tracking.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import threading
from collections.abc import Callable
from typing import Any

import requests

from .. import paths
from ..constants import DOWNLOAD_CHUNK, DOWNLOAD_TIMEOUT
from ..tts import catalog
from ..tts.models import ModelStore

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


class DownloadCancelled(Exception):
    """Raised inside the worker when the user cancels a download."""


class DownloadError(Exception):
    """Raised on download failure."""


def _remove_file(path: str) -> None:
    """Delete a partial download, ignoring a lock or an already-gone file."""
    try:
        os.remove(path)
    except OSError:
        log.debug("Could not remove %s", path, exc_info=True)


def download_file(
    url: str,
    dest_dir: str,
    filename: str | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Download ``url`` to ``dest_dir`` (resumable). Returns final file path."""
    os.makedirs(dest_dir, exist_ok=True)
    if filename is None:
        filename = url.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0]
    dest = os.path.join(dest_dir, filename)
    part = dest + ".part"

    downloaded = os.path.getsize(part) if os.path.exists(part) else 0

    session = requests.Session()
    try:
        # At most two attempts: the second one starts from byte 0 when the
        # bytes already on disk cannot be resumed.
        for _attempt in (0, 1):
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            resp = session.get(
                url,
                headers=headers,
                timeout=DOWNLOAD_TIMEOUT,
                stream=True,
            )
            try:
                if downloaded and resp.status_code == 416:
                    # Requested Range Not Satisfiable: the file the server has
                    # now is shorter than the bytes already downloaded.  A
                    # gyan.dev release URL does exactly this when the build
                    # behind it is replaced (ffmpeg-9.0 -> ffmpeg-9.1), and
                    # its stale ``.part`` used to fail the whole install with
                    # a 416.  Drop the partial file and fetch from the start.
                    log.info("Restarting %s: the server no longer serves byte %d",
                             filename, downloaded)
                    _remove_file(part)
                    downloaded = 0
                    continue
                resp.raise_for_status()
                if downloaded and resp.status_code != 206:
                    # The server ignored the Range request and sent the whole
                    # file: appending it to the old bytes would corrupt it.
                    log.info("Restarting %s: the server ignored the Range request",
                             filename)
                    _remove_file(part)
                    downloaded = 0
                    continue

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
                break
            finally:
                resp.close()
        else:
            raise DownloadError(
                f"Failed to download {url}: the server refused every request")
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
    prefixes: list[str] | None = None,
    exact: list[str] | None = None,
) -> list[str]:
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
    result: list[str] = []
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
                if rel.startswith(("..", "/")):
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
            # shutil.move() moves a directory *into* an existing directory of
            # the same name, so extracting a second archive over the first
            # buried espeak-ng-data/espeak-ng-data one level deeper every time.
            # The older copy is replaced instead.
            _replace(d)
            shutil.move(s, d)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _replace(path: str) -> None:
    """Clear ``path`` so the next ``shutil.move`` does not land inside it."""
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path) or os.path.islink(path):
            os.remove(path)
    except OSError as exc:  # a locked file must not abort an install
        log.warning("Could not replace %s: %s", path, exc)


#: Prefix of the folder an extraction is staged in, inside the variant folder.
#: Each call makes its own (a download and a repair can run at the same time)
#: and removes it as soon as the files are in place.
_STAGING_PREFIX = ".staging-"

#: The espeak-ng-data folder every voice of a variant shares.
_ESPEAK_DIR = "espeak-ng-data"


def voice_for_artifact(variant: dict, artifact: str) -> dict[str, Any] | None:
    """The catalog voice an artifact belongs to.

    ``None`` means the artifact belongs to the variant itself (kokoro and
    kitten ship one model for all of their speakers), in which case its files
    belong at the variant root.
    """
    if variant.get("artifact") == artifact:
        return None
    for voice in variant.get("voices", []):
        if voice.get("artifact") == artifact:
            return voice
    return None


def install_archive(archive: str, dest: str, voice: dict | None = None) -> str:
    """Move one archive's files into the folder the engine reads.

    ``dest`` is the variant folder.  An artifact that belongs to a catalog
    voice lands in ``dest/<voice_id>/``, which is where the engine looks for a
    per-voice model (piper: one folder per voice; matcha and vits-zh: a single
    ``default`` folder); an artifact that belongs to the variant itself stays
    in ``dest`` (kokoro, kitten).

    The ``espeak-ng-data`` a piper archive carries is the same for every voice
    of a variant, so one copy is kept in the variant folder rather than one per
    voice (~13 MB each).  Returns the folder the files were moved into.
    """
    target = dest if not voice else os.path.join(dest, voice["id"])
    os.makedirs(target, exist_ok=True)
    staging = tempfile.mkdtemp(dir=dest, prefix=_STAGING_PREFIX)
    try:
        safe_extract_tarball(archive, staging)
        for entry in sorted(os.listdir(staging)):
            src = os.path.join(staging, entry)
            if entry == _ESPEAK_DIR:
                shared = os.path.join(dest, _ESPEAK_DIR)
                if os.path.isdir(shared) and os.listdir(shared):
                    continue
                _replace(shared)
                shutil.move(src, shared)
                continue
            dst = os.path.join(target, entry)
            _replace(dst)
            shutil.move(src, dst)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def drop_staging_folders(dest: str) -> None:
    """Remove the staging folders an interrupted extraction left behind."""
    try:
        names = os.listdir(dest)
    except OSError:
        return
    for name in names:
        if name.startswith(_STAGING_PREFIX):
            shutil.rmtree(os.path.join(dest, name), ignore_errors=True)


def _has_onnx(directory: str) -> bool:
    try:
        return any(name.lower().endswith(".onnx") for name in os.listdir(directory))
    except OSError:
        return False


def repair_voice_layout(voice_entry: dict[str, Any]) -> bool:
    """Give a voice its own folder when an older build left its files loose.

    Every build up to 2027.1.0 extracted each archive of a variant into the
    variant folder itself, where the last voice's ``model.onnx``/``tokens.txt``
    overwrote the ones before it: the per-voice folders the engine reads
    (``<variant>/<voice>``) stayed empty, so every voice of a multi-voice
    variant - piper medium/high, matcha, the Chinese VITS voice - failed with
    "Model files are incomplete (missing model, tokens)" while the files it
    wanted were sitting one folder above it.

    The archives of that download are still on disk, so the voice is put back
    together from its own archive here - no download, no re-install.  Only a
    voice that the catalog knows, whose folder is the one its variant owns,
    that still has its archive beside it and that holds no model is touched.
    Returns ``True`` once the voice folder holds a model.
    """
    voice_dir = str(voice_entry.get("dir") or "")
    if not voice_dir:
        return False
    # A voice installed by the old build has no folder of its own at all; one
    # that has been repaired already holds a model and is left alone.
    if os.path.isdir(voice_dir) and _has_onnx(voice_dir):
        return True
    variant_dir = os.path.dirname(os.path.normpath(voice_dir))
    if not os.path.isdir(variant_dir):
        return False

    language = str(voice_entry.get("language") or "")
    variant_id = str(voice_entry.get("variant") or "")
    # The folder has to be the one this voice's variant owns, so an archive
    # that happens to sit in an unrelated folder is never unpacked into it.
    if os.path.basename(variant_dir) != variant_id:
        return False
    if os.path.basename(os.path.dirname(variant_dir)) != language:
        return False

    tts = catalog.find_tts(str(voice_entry.get("tts") or ""))
    if not tts:
        return False
    variant = catalog.find_variant(tts, language, variant_id)
    if not variant:
        return False
    voice = next(
        (v for v in variant.get("voices", []) if v.get("id") == voice_entry.get("voice")),
        None,
    )
    if not voice:
        return False

    artifact = os.path.basename(str(voice.get("artifact") or ""))
    archive = os.path.join(variant_dir, artifact) if artifact else ""
    if not archive or not os.path.isfile(archive):
        return False
    try:
        install_archive(archive, variant_dir, voice)
    except (tarfile.TarError, OSError) as exc:
        log.warning("Could not repair %s from %s: %s", voice_dir, archive, exc)
        return False
    if not _has_onnx(voice_dir):
        return False
    try:
        os.remove(archive)
    except OSError:
        log.debug("Ignoring a failure in repair_voice_layout", exc_info=True)
    log.info("Repaired %s from %s", voice_dir, os.path.basename(archive))
    return True


class ModelDownloader:
    """Downloads and removes catalog variants using a ModelStore."""

    def __init__(self, store: ModelStore):
        self.store = store

    def _artifact_url(self, tts: dict, artifact: str) -> str:
        base = tts.get("artifact_base", "")
        return base.rstrip("/") + "/" + artifact

    def download_variant(
        self,
        tts_id: str,
        lang_code: str,
        variant_id: str,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
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
        # Archives are kept until the whole variant is in place: a download
        # that fails on the fourth voice then re-uses the first three instead
        # of fetching them again.  They are deleted once it has succeeded.
        archives: list[str] = []
        for artifact in artifacts:
            if cancel_event and cancel_event.is_set():
                raise DownloadCancelled()
            filename = os.path.basename(artifact)
            archive = os.path.join(dest, filename)
            voice = voice_for_artifact(variant, artifact)
            target = dest if not voice else os.path.join(dest, voice["id"])
            if not os.path.isfile(archive) and _has_onnx(target):
                # This voice is already installed (its folder holds a model), so
                # a variant that stopped part-way - or one downloaded by an
                # older build that only installed the first voice - fetches
                # what is missing and leaves the rest alone.
                if progress:
                    progress(f"{filename} (already installed)", 1, 1)
                continue
            if os.path.isfile(archive):
                # Already downloaded by an interrupted run (download_file only
                # renames a finished file into place, so an archive that exists
                # is complete).
                if progress:
                    progress(filename, 1, 1)
            else:
                url = self._artifact_url(tts, artifact)
                archive = download_file(
                    url,
                    dest,
                    filename=filename,
                    progress=progress,
                    cancel_event=cancel_event,
                )
            archives.append(archive)

            # Extract the archive into the folder the engine reads, so a
            # per-voice artifact ends up in <variant>/<voice>/ and the
            # variant's own artifact in <variant>/ (see install_archive).
            if archive.endswith((".tar.bz2", ".tar.gz")):
                if progress:
                    progress(f"Extracting {filename}", 0, 0)
                try:
                    install_archive(archive, dest, voice)
                except (tarfile.TarError, OSError) as exc:
                    # Never trust a corrupt archive on the next run.
                    try:
                        os.remove(archive)
                    except OSError:
                        log.debug("Ignoring a failure in ModelDownloader.download_variant", exc_info=True)
                    raise DownloadError(f"Could not extract {filename}: {exc}") from exc

        # Download shared artifacts (espeak-ng-data, vocoders, etc.)
        self._download_extras(tts, tts.get("shared_artifacts", []), progress, cancel_event)

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

        # Only now is the variant complete: drop the archives (the extracted
        # files are what the engine reads) and any staging folder left behind.
        drop_staging_folders(dest)
        for archive in archives:
            try:
                os.remove(archive)
            except OSError as exc:
                log.warning("Could not remove %s: %s", archive, exc)

    def _variant_artifacts(self, variant: dict) -> list[str]:
        artifacts: list[str] = []
        if variant.get("artifact"):
            artifacts.append(variant["artifact"])
        for voice in variant.get("voices", []):
            if voice.get("artifact") and voice["artifact"] not in artifacts:
                artifacts.append(voice["artifact"])
        return artifacts

    def _download_extras(
        self,
        tts: dict,
        extras: list[dict],
        progress: ProgressCallback | None,
        cancel_event: threading.Event | None,
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

    def remove_variant(self, tts_id: str, lang_code: str, variant_id: str) -> bool:
        removed = self.store.remove_variant(tts_id, lang_code, variant_id)
        self._remove_unused_shared_model(tts_id)
        return removed

    def _remove_unused_shared_model(self, tts_id: str) -> None:
        """Delete an engine's shared model once its last variant is gone.

        An engine whose archive serves every one of its languages (Supertonic
        3: one int8 model for all 31) keeps the model in ``models/shared``
        rather than in a language folder.  Removing a language must therefore
        free the real download as soon as nothing else uses it - otherwise the
        Remove button would report success and leave ~130 MB on the disk with
        no engine left to read it.
        """
        tts = catalog.find_tts(tts_id)
        if not tts or not tts.get("shared_model"):
            return
        if any(entry.get("tts") == tts_id for entry in self.store.installed_variants()):
            # Another language of this engine still reads the same model.
            return
        for extra in tts.get("shared_artifacts", []):
            folder = os.path.join(
                paths.models_dir(), *str(extra.get("dest", "shared")).split("/")
            )
            if not os.path.isdir(folder):
                continue
            shutil.rmtree(folder, ignore_errors=True)
            log.info("Removed the shared model of %s from %s", tts_id, folder)

    # -- shared espeak-ng-data (needed by raw custom piper voices) ----------
    def ensure_shared_espeak_data(
        self,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
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
