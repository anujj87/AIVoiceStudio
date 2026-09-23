"""Check the winget manifests in ``winget/manifests`` for compliance.

winget runs its own validator when a package is submitted to
``microsoft/winget-pkgs``, and a rejected manifest means another review round.
This checker mirrors those rules locally so the manifests are known-good
before they are submitted, and it is what ``tests/test_winget.py`` runs in
the suite:

    .venv/Scripts/python.exe tools/winget_validate.py

It checks the *shipped* files (not freshly generated ones), so a manifest
that was edited by hand or left behind by an older version is caught.  When
an installer of that version is present in ``dist/``, its SHA256 must match
the published one - a manifest that points at bytes which changed is the one
failure winget cannot recover from.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import winget_manifests as wm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _app_version() -> str:
    from ai_voice_studio.constants import APP_VERSION

    return APP_VERSION


def published_hashes(manifests: dict) -> dict:
    """The InstallerSha256 per architecture published by ``manifests``."""
    doc = manifests.get(f"{wm.PACKAGE_IDENTIFIER}.installer.yaml") or {}
    result = {}
    for entry in doc.get("Installers") or []:
        if isinstance(entry, dict) and entry.get("Architecture"):
            result[str(entry["Architecture"])] = str(
                entry.get("InstallerSha256") or ""
            )
    return result


def published_versions(root: str = ROOT) -> list:
    """Every version that currently has a manifest folder in the repository."""
    pattern = os.path.join(root, "winget", "manifests", "*", "AnujSharma",
                           "AIVoiceStudio", "*")
    versions = [os.path.basename(path) for path in glob.glob(pattern)
                if os.path.isdir(path)]
    return sorted(versions)


def check(version: str | None = None, root: str = ROOT) -> list:
    """Return the compliance problems of the shipped manifests (empty is good)."""
    version = str(version or _app_version())
    problems: list = []

    folder = wm.manifest_dir(root, version)
    if not os.path.isdir(folder):
        published = published_versions(root)
        detail = (", ".join(published)) if published else "none"
        return [
            f"no winget manifests for the current version {version} "
            f"(winget/manifests/... has: {detail}). Run tools/make_winget.py "
            "after building the installer."
        ]

    manifests = wm.load_manifests(folder)
    for name in (f"{wm.PACKAGE_IDENTIFIER}.yaml",
                 f"{wm.PACKAGE_IDENTIFIER}.installer.yaml",
                 f"{wm.PACKAGE_IDENTIFIER}.locale.{wm.LOCALE}.yaml"):
        if name not in manifests:
            problems.append(f"{name} is missing from {os.path.relpath(folder, root)}")

    # Compare the published hashes with the installer of the same version in
    # dist/, when it has been built here.
    sha256 = {}
    installer_exists = {}
    for arch, digest in published_hashes(manifests).items():
        path = wm.installer_path(root, version, arch)
        installer_exists[arch] = os.path.isfile(path)
        if installer_exists[arch]:
            sha256[arch] = wm.file_sha256(path)

    problems.extend(wm.validate(manifests, version, sha256 or None,
                                installer_exists or None))
    return problems


def main(argv: list) -> int:
    version = None
    if len(argv) > 1 and not argv[1].startswith("-"):
        version = argv[1]
    version = version or _app_version()

    print(f"winget manifests for {wm.PACKAGE_IDENTIFIER} {version}")
    folder = os.path.relpath(wm.manifest_dir(ROOT, version), ROOT)
    print(f"  folder: {folder}")
    problems = check(version)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print("  ! " + problem)
        return 1
    print("\nNo winget compliance problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
