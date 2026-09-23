"""Tests that keep the winget package compliant and in step with the app.

winget rejects a package whose manifests disagree with the installer it
points at: a wrong version, a silent switch the installer does not answer, a
download URL that is not the published asset, or a SHA256 that no longer
matches the bytes.  Every one of those is produced silently by an otherwise
healthy build, so they are checked here rather than at submission time.

The shipped files under ``winget/manifests`` are read back and compared with
what the generator would write from the same published hashes, which also
guards the small YAML reader/writer the tools rely on.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import winget_manifests as wm  # noqa: E402
import winget_validate  # noqa: E402

from ai_voice_studio import __version__  # noqa: E402
from ai_voice_studio.constants import APP_VERSION  # noqa: E402

INSTALLER_ISS = os.path.join(ROOT, "packaging", "installer_common.iss")
FAKE_HASH = "A" * 64


class ManifestGeneratorTest(unittest.TestCase):
    """The generated manifests are compliant before anything is published."""

    def test_generated_manifests_pass_validation(self):
        manifests = wm.build_all(
            APP_VERSION, {"x64": FAKE_HASH}, release_date="2026-09-23",
            release_notes="Hello.",
        )
        problems = wm.validate(manifests, APP_VERSION, {"x64": FAKE_HASH},
                              {"x64": True})
        self.assertEqual(problems, [])

    def test_an_architecture_without_a_built_installer_is_not_published(self):
        manifests = wm.build_all(APP_VERSION, {}, release_date="2026-09-23")
        installer = manifests[f"{wm.PACKAGE_IDENTIFIER}.installer.yaml"]
        self.assertEqual(installer["Installers"], [])
        # Nothing to publish is a problem: winget needs at least one installer.
        problems = wm.validate(manifests, APP_VERSION)
        self.assertIn("installer manifest: Installers must not be empty", problems)

    def test_a_wrong_silent_switch_is_reported(self):
        manifests = wm.build_all(APP_VERSION, {"x64": FAKE_HASH},
                                 release_date="2026-09-23")
        installer = manifests[f"{wm.PACKAGE_IDENTIFIER}.installer.yaml"]
        installer["InstallerSwitches"]["Silent"] = "/S"
        problems = wm.validate(manifests, APP_VERSION)
        self.assertTrue(any("/VERYSILENT" in problem for problem in problems))

    def test_a_changed_installer_is_reported(self):
        manifests = wm.build_all(APP_VERSION, {"x64": FAKE_HASH},
                                 release_date="2026-09-23")
        problems = wm.validate(manifests, APP_VERSION, {"x64": "B" * 64})
        self.assertTrue(any("does not match the installer" in p for p in problems))

    def test_a_latest_redirect_is_reported(self):
        manifests = wm.build_all(
            APP_VERSION, {"x64": FAKE_HASH}, release_date="2026-09-23",
            urls={"x64": "https://github.com/anujj87/AIVoiceStudio/releases/latest/download/AI-Voice-Studio-Setup-x64.exe"},
        )
        problems = wm.validate(manifests, APP_VERSION)
        self.assertTrue(any("versioned release asset" in problem
                            for problem in problems))

    def test_a_version_that_would_not_sort_is_reported(self):
        manifests = wm.build_all("2026.3.7-beta", {"x64": FAKE_HASH},
                                 release_date="2026-09-23")
        problems = wm.validate(manifests, "2026.3.7-beta")
        self.assertTrue(any("dotted numbers" in problem for problem in problems))

    def test_missing_required_metadata_is_reported(self):
        manifests = wm.build_all(APP_VERSION, {"x64": FAKE_HASH},
                                 release_date="2026-09-23")
        locale = manifests[f"{wm.PACKAGE_IDENTIFIER}.locale.{wm.LOCALE}.yaml"]
        del locale["License"]
        problems = wm.validate(manifests, APP_VERSION)
        self.assertTrue(any("'License'" in problem for problem in problems))


class YamlTest(unittest.TestCase):
    """The manifests are read back by our own reader, so it must be exact."""

    def test_round_trip(self):
        data = wm.build_all(
            APP_VERSION, {"x64": FAKE_HASH, "x86": "B" * 64},
            release_date="2026-09-23", release_notes="Some notes: with a colon.",
        )
        for name, manifest in data.items():
            with self.subTest(name):
                self.assertEqual(wm.load_yaml(wm.dump_yaml(manifest)), manifest)

    def test_comments_and_blank_lines_are_ignored(self):
        text = "# header\nPackageIdentifier: A.B\n\n# trailing\nPackageVersion: 1.2.3\n"
        self.assertEqual(
            wm.load_yaml(text),
            {"PackageIdentifier": "A.B", "PackageVersion": "1.2.3"},
        )

    def test_a_hash_inside_a_quoted_value_is_not_a_comment(self):
        data = {"ProductCode": "{AABB}_is1", "Url": "https://x/y"}
        self.assertEqual(wm.load_yaml(wm.dump_yaml(data)), data)


class ManifestPresentTest(unittest.TestCase):
    """A release must ship the manifests of the version it ships."""

    def test_manifests_exist_for_this_version(self):
        folder = wm.manifest_dir(ROOT, APP_VERSION)
        self.assertTrue(
            os.path.isdir(folder),
            "run tools/make_winget.py after building the installer",
        )
        manifests = wm.load_manifests(folder)
        for name in (f"{wm.PACKAGE_IDENTIFIER}.yaml",
                     f"{wm.PACKAGE_IDENTIFIER}.installer.yaml",
                     f"{wm.PACKAGE_IDENTIFIER}.locale.{wm.LOCALE}.yaml"):
            self.assertIn(name, manifests)


class ShippedManifestTest(unittest.TestCase):
    """What is in ``winget/manifests`` must be the current, correct package."""

    def setUp(self):
        self.folder = wm.manifest_dir(ROOT, APP_VERSION)
        if not os.path.isdir(self.folder):
            self.skipTest("no manifests generated yet "
                          "(run tools/make_winget.py after building)")
        self.manifests = wm.load_manifests(self.folder)

    def test_shipped_manifests_are_compliant(self):
        problems = winget_validate.check(APP_VERSION, root=ROOT)
        self.assertEqual(problems, [])

    def test_version_matches_the_application_and_the_installer(self):
        version_doc = self.manifests[f"{wm.PACKAGE_IDENTIFIER}.yaml"]
        self.assertEqual(version_doc["PackageVersion"], APP_VERSION)
        self.assertEqual(__version__, APP_VERSION)
        with open(INSTALLER_ISS, encoding="utf-8") as handle:
            script = handle.read()
        for line in ("AppVersion=", "VersionInfoVersion=", "VersionInfoProductVersion="):
            match = re.search(rf"^{line}(.+)$", script, re.MULTILINE)
            self.assertIsNotNone(match, line)
            self.assertEqual(match.group(1).strip(), APP_VERSION, line)

    def test_the_manifest_folder_holds_no_other_version(self):
        versions = winget_validate.published_versions(ROOT)
        self.assertEqual(versions, [APP_VERSION],
                         "an old manifest folder was left behind")

    def test_the_shipped_files_are_what_the_generator_would_write(self):
        published = winget_validate.published_hashes(self.manifests)
        installer_doc = self.manifests[
            f"{wm.PACKAGE_IDENTIFIER}.installer.yaml"]
        expected = wm.build_all(
            APP_VERSION,
            published,
            release_date=str(installer_doc.get("ReleaseDate") or ""),
            release_notes=str(
                self.manifests[
                    f"{wm.PACKAGE_IDENTIFIER}.locale.{wm.LOCALE}.yaml"
                ].get("ReleaseNotes") or ""
            ),
        )
        for name in expected:
            with self.subTest(name):
                self.assertEqual(self.manifests[name], expected[name])

    def test_the_published_hash_matches_the_built_installer(self):
        published = winget_validate.published_hashes(self.manifests)
        checked = 0
        for arch, digest in published.items():
            path = wm.installer_path(ROOT, APP_VERSION, arch)
            if not os.path.isfile(path):
                continue
            checked += 1
            self.assertEqual(wm.file_sha256(path), digest,
                             f"the {arch} installer changed after the manifest "
                             "was generated - run tools/make_winget.py again")
        if not checked:
            self.skipTest("no built installer in dist/ to compare against")

    def test_the_inno_app_id_is_the_published_product_code(self):
        with open(INSTALLER_ISS, encoding="utf-8") as handle:
            script = handle.read()
        match = re.search(r"^AppId=(.+)$", script, re.MULTILINE)
        self.assertIsNotNone(match)
        # Inno escapes a literal brace by doubling it in the script.
        app_id = match.group(1).strip().replace("{{", "{")
        self.assertEqual(app_id, wm.INNO_APP_ID)
        self.assertEqual(wm.PRODUCT_CODE, app_id + "_is1")

    def test_the_installer_is_silent_installable(self):
        with open(INSTALLER_ISS, encoding="utf-8") as handle:
            script = handle.read()
        # Every [Run] entry is skipped in a silent install, so winget's
        # /VERYSILENT never launches the application it just installed.
        self.assertIn("AllowNoIcons=yes", script)
        run_block = re.search(r"^\[Run\](.*?)(?=^\[|\Z)", script,
                              re.MULTILINE | re.DOTALL)
        self.assertIsNotNone(run_block)
        self.assertIn("skipifsilent", run_block.group(1))


if __name__ == "__main__":
    unittest.main()
