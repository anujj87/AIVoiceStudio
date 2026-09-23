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
"""

from __future__ import annotations

import logging
import os
import struct
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

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

    def asset_for_arch(self, arch: str) -> Optional[ReleaseAsset]:
        marker = ASSET_MARKERS.get(arch, ASSET_MARKERS["x64"])
        for asset in self.assets:
            if marker in asset.name.lower():
                return asset
        return None


@dataclass
class UpdateCheck:
    """The outcome of one check, success or failure."""

    current: str
    latest: Optional[ReleaseInfo] = None
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
    except Exception as exc:  # noqa: BLE001 - a check must never break the app
        log.debug("Update check failed", exc_info=True)
        return UpdateCheck(current=current, error=f"Could not check for updates: {exc}")
    return evaluate_release(payload, current)


# ---------------------------------------------------------------------------
# Downloading the installer
# ---------------------------------------------------------------------------
def download_installer(
    release: ReleaseInfo,
    arch: Optional[str] = None,
    dest_dir: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Download this release's installer for ``arch``; returns the file path.

    Raises :class:`UpdateError` when the release has no installer for this
    architecture (an older release may only carry one).
    """
    from .tts.downloader import DownloadCancelled, DownloadError, download_file  # noqa: PLC0415

    asset = release.asset_for_arch(arch or current_arch())
    if asset is None:
        raise UpdateError(
            "This release does not carry an installer for this Windows "
            "version. Open the release page instead."
        )
    if dest_dir is None:
        dest_dir = os.path.join(tempfile.gettempdir(), "AIVoiceStudio-update")
    try:
        return download_file(
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
            os.startfile(path)  # type: ignore[attr-defined]  # noqa: SIM115 - Windows
        else:  # pragma: no cover - the app ships for Windows
            import subprocess  # noqa: PLC0415

            subprocess.Popen(args)  # noqa: S603
    except OSError as exc:
        raise UpdateError(f"Could not start the installer: {exc}") from exc
