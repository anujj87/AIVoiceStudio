"""Where a describing model's checkpoint files live.

Two different things are called "the model" in Settings > Manage the video
describer, and conflating them is what makes the category confusing:

* the **environment** - the packages (``transformers``, PyTorch, a multimodal
  processor) one model runs on.  It is per *model id* and is what *Install
  selected model* builds; and
* the **checkpoint** - the weights of one variant, downloaded from Hugging Face
  the first time that variant is described with.

Because the weights arrive during a run, a user who installed one variant of a
model could not tell whether another variant was on disk, nor fetch it without
starting a description.  This module is the second half: it answers "is this
variant's checkpoint here yet?" and "delete it", without importing
``huggingface_hub`` and without starting a subprocess - it only reads the cache
directory Hugging Face itself writes to.

The layout it reads is the one ``snapshot_download`` produces::

    <cache>/models--Qwen--Qwen3-VL-4B-Instruct/blobs/...
    <cache>/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/<revision>/...

so a variant counts as downloaded when a snapshot revision exists and is not
empty.  A half-finished download leaves the folder without a populated
snapshot, which is exactly right: the next run resumes it.
"""

from __future__ import annotations

import logging
import os
import shutil

log = logging.getLogger(__name__)

#: Environment variables ``huggingface_hub`` itself honours, in its own order.
_CACHE_ENV_VARS = ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE")


def cache_dir() -> str:
    """The Hugging Face cache folder the model environments download into.

    The application never sets ``HF_HOME``, so the worker subprocess resolves
    this the same way - which is what makes a status read from the cache agree
    with what a run will actually do.
    """
    for name in _CACHE_ENV_VARS:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return os.path.abspath(os.path.expanduser(value))
    home = str(os.environ.get("HF_HOME") or "").strip()
    if home:
        return os.path.join(os.path.abspath(os.path.expanduser(home)), "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                        "hub")


def repo_folder(repo: str) -> str:
    """The cache folder name Hugging Face gives a repository."""
    return "models--" + str(repo or "").strip().replace("/", "--")


def path_for(repo: str) -> str:
    """Absolute path of a repository's cache folder."""
    return os.path.join(cache_dir(), repo_folder(repo))


def incomplete_files(repo: str) -> int:
    """How many files of this repository are still arriving.

    ``huggingface_hub`` streams a file into ``blobs/<sha>.incomplete`` and
    renames it when it is whole, so the marker is exactly the evidence of a
    fetch that stopped part-way.  It matters because the *snapshot* folder is
    populated as soon as the first file lands: without this a download cancelled
    at 20% counted as "Downloaded", and a panel that believes that disables the
    one button that would have resumed it.  (A marker left by a cancelled
    *other* revision keeps a variant listed as not downloaded, which only means
    the download stays available - the harmless direction.)
    """
    blobs = os.path.join(path_for(repo), "blobs")
    try:
        names = os.listdir(blobs)
    except OSError:
        return 0
    return sum(1 for name in names if name.endswith(".incomplete"))


def is_partly_downloaded(repo: str) -> bool:
    """True when a fetch of this repository stopped before it had finished."""
    if not str(repo or "").strip():
        return False
    return bool(incomplete_files(repo))


def is_downloaded(repo: str) -> bool:
    """True when this repository has a populated, *complete* snapshot.

    A snapshot with unfinished files in it is not a checkpoint that can be
    loaded, so it does not answer yes; see :func:`incomplete_files`.
    """
    if not str(repo or "").strip():
        return False
    if is_partly_downloaded(repo):
        return False
    snapshots = os.path.join(path_for(repo), "snapshots")
    try:
        revisions = os.listdir(snapshots)
    except OSError:
        return False
    for revision in revisions:
        folder = os.path.join(snapshots, revision)
        try:
            if os.path.isdir(folder) and os.listdir(folder):
                return True
        except OSError:
            continue
    return False


def size_bytes(repo: str) -> int:
    """Bytes of a repository already on disk (0 when nothing is there)."""
    folder = os.path.join(path_for(repo), "blobs")
    total = 0
    for root, _dirs, names in os.walk(folder):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def remove(repo: str) -> str | None:
    """Delete a repository's cached files; a note when it was not clean.

    ``None`` means "nothing left behind", which is also the answer for a
    variant that was never downloaded - there is nothing to report.
    """
    if not str(repo or "").strip():
        return None
    folder = path_for(repo)
    if not os.path.isdir(folder):
        return None
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        log.debug("could not remove the cached files of %s", repo, exc_info=True)
        return (f"Some files of {repo} could not be deleted ({exc}); they can "
                "be removed by hand from " + folder)
    return None
