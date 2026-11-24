"""Update checks against the project's GitHub releases.

The application is published on GitHub, so the single source of truth for
"is there a newer version?" is the *latest release* of the repository:

    https://api.github.com/repos/anujj87/AIVoiceStudio/releases/latest

Nothing here talks to the network by accident.  The module is split so the
interesting part is a pure function (``evaluate_release`` turns a release
payload into a decision) and only ``fetch_latest_release`` performs I/O.
That keeps the version comparison and the asset picking testable without a
network connection, which is how the test suite uses it.

The installer is downloaded through ``tts.downloader.download_file`` so an
update gets exactly the same resumable, cancellable transfer the model
downloader already uses (Range requests + ``.part`` files).

**The download is verified before it is offered.**  The installer is the one
thing this application fetches and then *runs*, and the installer itself is not
code-signed yet, so the SHA-256 GitHub publishes for the asset (``digest`` in
the release payload) is the only integrity check there is: a file whose hash
does not match is deleted and reported as an error rather than handed to
`launch_installer`.  A release published before GitHub added digests has no
hash to check, which is logged and allowed (refusing would strand everyone on
an older version).
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from .constants import APP_VERSION, DOWNLOAD_TIMEOUT

log = logging.getLogger(__name__)

GITHUB_OWNER = "anujj87"
GITHUB_REPO = "AIVoiceStudio"
GITHUB_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
RELEASES_PAGE = f"{GITHUB_URL}/releases"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"

# GitHub rejects API requests without a User-Agent.
USER_AGENT = f"AI-Voice-Studio/{APP_VERSION}"

# The asset names the release pipeline produces (packaging/build.ps1):
#   AI-Voice-Studio-v-2026-3-7-Setup-x64.exe
#   AI-Voice-Studio-v-2026-3-7-Setup-x86.exe
ASSET_MARKERS = {"x64": "setup-x64", "x86": "setup-x86"}

#: How much of a file is read per chunk while hashing it.
_HASH_CHUNK = 1024 * 1024

ProgressCallback = Callable[[str, int, int], None]


class UpdateError(Exception):
    """Raised when the release information cannot be retrieved."""


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
def parse_version(text: str) -> tuple:
    """Turn ``"v2026.3.7"`` / ``"2026.3.7-beta"`` into ``(2026, 3, 7)``.

    Only the leading numeric run(s) are used, so a release tag may carry a
    ``v`` prefix or a suffix without breaking the comparison.  A tag with no
    digits at all yields ``()`` and simply never compares as newer.
    """
    numbers: list[int] = []
    current = ""
    for char in str(text or ""):
        if char.isdigit():
            current += char
        elif char == "." and current:
            numbers.append(int(current))
            current = ""
        elif current:
            break
    if current:
        numbers.append(int(current))
    return tuple(numbers)


def _pad(version: tuple, size: int = 4) -> tuple:
    return version + (0,) * max(0, size - len(version))


def is_newer(remote: str, local: str = APP_VERSION) -> bool:
    """True when ``remote`` names a strictly newer version than ``local``."""
    remote_parts = parse_version(remote)
    if not remote_parts:
        return False
    return _pad(remote_parts) > _pad(parse_version(local))


# ---------------------------------------------------------------------------
# Release model
# ---------------------------------------------------------------------------
@dataclass
class ReleaseAsset:
    name: str
    url: str
    size: int = 0
    #: ``"sha256:<hex>"`` as GitHub publishes it, or ``""`` on an old release.
    digest: str = ""


@dataclass
class ReleaseInfo:
    version: str
    tag: str
    name: str = ""
    notes: str = ""
    page_url: str = RELEASES_PAGE
    published_at: str = ""
    prerelease: bool = False
    assets: list = field(default_factory=list)

    def asset_for_arch(self, arch: str) -> ReleaseAsset | None:
        marker = ASSET_MARKERS.get(arch, ASSET_MARKERS["x64"])
        for asset in self.assets:
            if marker in asset.name.lower():
                return asset
        return None


@dataclass
class UpdateCheck:
    """The outcome of one check, success or failure."""

    current: str
    latest: ReleaseInfo | None = None
    available: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def current_arch() -> str:
    """``"x64"`` or ``"x86"`` for the *running* interpreter."""
    return "x64" if struct.calcsize("P") * 8 == 64 else "x86"


def evaluate_release(payload: dict, current: str = APP_VERSION) -> UpdateCheck:
    """Decide from a GitHub ``releases/latest`` payload whether to offer an update.

    A draft release is ignored; a prerelease is offered like any other
    release because this project publishes its installers exactly that way.
    """
    if not isinstance(payload, dict):
        return UpdateCheck(current=current, error="Unexpected release data from GitHub.")
    if payload.get("draft"):
        return UpdateCheck(current=current, latest=None, available=False)

    tag = str(payload.get("tag_name") or payload.get("name") or "")
    version = ".".join(str(part) for part in parse_version(tag))
    assets = [
        ReleaseAsset(
            name=str(item.get("name") or ""),
            url=str(item.get("browser_download_url") or ""),
            size=int(item.get("size") or 0),
            digest=str(item.get("digest") or ""),
        )
        for item in (payload.get("assets") or [])
        if isinstance(item, dict)
    ]
    release = ReleaseInfo(
        version=version or tag,
        tag=tag,
        name=str(payload.get("name") or ""),
        notes=str(payload.get("body") or ""),
        page_url=str(payload.get("html_url") or RELEASES_PAGE),
        published_at=str(payload.get("published_at") or ""),
        prerelease=bool(payload.get("prerelease")),
        assets=assets,
    )
    if not version:
        return UpdateCheck(
            current=current, latest=release, available=False,
            error="The latest release has no usable version number.",
        )
    return UpdateCheck(
        current=current, latest=release, available=is_newer(version, current)
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def fetch_latest_release(timeout: int = DOWNLOAD_TIMEOUT) -> dict:
    """GET the latest release payload. Raises :class:`UpdateError` on failure."""
    try:
        resp = requests.get(
            LATEST_RELEASE_API,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise UpdateError(f"Could not reach GitHub: {exc}") from exc
    except ValueError as exc:
        raise UpdateError(f"GitHub returned an unreadable answer: {exc}") from exc


def check_for_update(current: str = APP_VERSION, timeout: int = DOWNLOAD_TIMEOUT) -> UpdateCheck:
    """Look for a newer release. Never raises: a failure is reported in the result."""
    try:
        payload = fetch_latest_release(timeout=timeout)
    except UpdateError as exc:
        return UpdateCheck(current=current, error=str(exc))
    except Exception as exc:
        log.debug("Update check failed", exc_info=True)
        return UpdateCheck(current=current, error=f"Could not check for updates: {exc}")
    return evaluate_release(payload, current)


# ---------------------------------------------------------------------------
# Downloading the installer
# ---------------------------------------------------------------------------
def download_installer(
    release: ReleaseInfo,
    arch: str | None = None,
    dest_dir: str | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Download this release's installer for ``arch``; returns the file path.

    Raises :class:`UpdateError` when the release has no installer for this
    architecture (an older release may only carry one).
    """
    from .tts.downloader import DownloadCancelled, DownloadError, download_file

    asset = release.asset_for_arch(arch or current_arch())
    if asset is None:
        raise UpdateError(
            "This release does not carry an installer for this Windows "
            "version. Open the release page instead."
        )
    if dest_dir is None:
        dest_dir = os.path.join(tempfile.gettempdir(), "AIVoiceStudio-update")
    try:
        path = download_file(
            asset.url,
            dest_dir,
            filename=asset.name,
            progress=progress,
            cancel_event=cancel_event,
        )
    except DownloadCancelled:
        raise
    except DownloadError as exc:
        raise UpdateError(str(exc)) from exc
    verify_download(path, asset)
    return path


def file_digest(path: str) -> str:
    """The SHA-256 of ``path`` as ``"sha256:<hex>"`` (the form GitHub uses)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def verify_download(path: str, asset: ReleaseAsset) -> None:
    """Check a downloaded asset against the hash the release published.

    Raises :class:`UpdateError` - and deletes the file - when the hash does not
    match: a truncated download, a proxy that rewrote the body, or a release
    swapped under the same URL all look like this, and none of them should end
    up being executed.  A release without a published digest cannot be checked;
    that is logged rather than refused, so an older release can still be
    installed deliberately.
    """
    expected = (asset.digest or "").strip().lower()
    if not expected:
        log.info("Release asset %s publishes no digest; the download is unverified",
                 asset.name)
        return
    try:
        actual = file_digest(path)
    except OSError as exc:
        raise UpdateError(f"Could not read the downloaded installer: {exc}") from exc
    if actual.lower() != expected:
        # Never leave a bad installer where launch_installer could find it.
        try:
            os.remove(path)
        except OSError as exc:
            log.debug("Could not delete the unverified installer %s: %s", path, exc)
        raise UpdateError(
            "The downloaded installer did not match the SHA-256 hash GitHub "
            f"published for it (expected {expected}, got {actual}). It has been "
            "deleted - open the release page and download the installer by "
            "hand if this keeps happening."
        )
    log.info("Verified %s against its published SHA-256", asset.name)


def launch_installer(path: str, silent: bool = False) -> None:
    """Start the downloaded installer (and let go of it).

    ``silent`` runs Inno Setup with ``/SILENT`` (a progress window, no
    questions); it is the same switch winget uses.  Failures are raised so
    the caller can tell the user instead of leaving a dead button.
    """
    if not path or not os.path.isfile(path):
        raise UpdateError("The downloaded installer is no longer on disk.")
    args = [path]
    if silent:
        args += ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
    try:
        if os.name == "nt" and not silent:
            # A normal double-click: let Windows treat it like any other
            # installer, so the user sees the familiar wizard.
            os.startfile(path)  # type: ignore[attr-defined]
        else:  # pragma: no cover - the app ships for Windows
            import subprocess

            subprocess.Popen(args)
    except OSError as exc:
        raise UpdateError(f"Could not start the installer: {exc}") from exc
