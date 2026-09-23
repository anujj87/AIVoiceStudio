"""Tests for the GitHub update check.

The interesting parts are pure: comparing versions, deciding what a release
payload means, and picking the installer for this machine.  Those are driven
directly.  The two places that touch the network (``requests.get`` and the
model downloader) are replaced, so the suite never depends on GitHub being
reachable and never writes a downloaded installer into ``dist/``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wx_test_app import get_app  # noqa: E402

from ai_voice_studio import updates  # noqa: E402
from ai_voice_studio.constants import APP_VERSION  # noqa: E402
from ai_voice_studio.gui import update_dialog  # noqa: E402
from ai_voice_studio.settings import Settings  # noqa: E402

UTC = timezone.utc


def _release_payload(version="2030.1.1", *, draft=False, prerelease=False,
                     assets=None, body="Notes here."):
    dashed = version.replace(".", "-")
    default_assets = [
        {
            "name": f"AI-Voice-Studio-v-{dashed}-Setup-x64.exe",
            "browser_download_url": (
                "https://github.com/anujj87/AIVoiceStudio/releases/download/"
                f"v{version}/AI-Voice-Studio-v-{dashed}-Setup-x64.exe"
            ),
            "size": 41_000_000,
        },
        {
            "name": f"AI-Voice-Studio-v-{dashed}-Setup-x86.exe",
            "browser_download_url": (
                "https://github.com/anujj87/AIVoiceStudio/releases/download/"
                f"v{version}/AI-Voice-Studio-v-{dashed}-Setup-x86.exe"
            ),
            "size": 38_000_000,
        },
    ]
    return {
        "tag_name": f"v{version}",
        "name": f"AI Voice Studio {version}",
        "body": body,
        "html_url": f"https://github.com/anujj87/AIVoiceStudio/releases/tag/v{version}",
        "published_at": "2030-01-01T00:00:00Z",
        "draft": draft,
        "prerelease": prerelease,
        "assets": default_assets if assets is None else assets,
    }


class _Response:
    """The two attributes ``fetch_latest_release`` uses."""

    def __init__(self, payload: str):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self._payload)


class VersionComparisonTest(unittest.TestCase):
    def test_parse_version_accepts_v_prefix_and_suffix(self):
        self.assertEqual(updates.parse_version("v2026.3.7"), (2026, 3, 7))
        self.assertEqual(updates.parse_version("2026.3.7"), (2026, 3, 7))
        self.assertEqual(updates.parse_version("2026.3.7-beta.1"), (2026, 3, 7))
        self.assertEqual(updates.parse_version("no digits here"), ())

    def test_is_newer_compares_numerically_not_as_text(self):
        self.assertTrue(updates.is_newer("2026.10.0", "2026.3.7"))
        self.assertTrue(updates.is_newer("2026.3.10", "2026.3.9"))
        self.assertTrue(updates.is_newer("v2027.1.0", "2026.3.7"))
        self.assertFalse(updates.is_newer("2026.3.7", "2026.3.7"))
        self.assertFalse(updates.is_newer("2026.3.6", "2026.3.7"))
        self.assertFalse(updates.is_newer("nightly", "2026.3.7"))


class EvaluateReleaseTest(unittest.TestCase):
    def test_newer_release_is_offered(self):
        check = updates.evaluate_release(_release_payload(), APP_VERSION)
        self.assertTrue(check.ok)
        self.assertTrue(check.available)
        self.assertEqual(check.latest.version, "2030.1.1")
        self.assertEqual(check.latest.notes, "Notes here.")

    def test_same_version_is_not_offered(self):
        check = updates.evaluate_release(_release_payload(APP_VERSION), APP_VERSION)
        self.assertFalse(check.available)
        self.assertTrue(check.ok)

    def test_older_release_is_not_offered(self):
        check = updates.evaluate_release(_release_payload("2000.1.1"), APP_VERSION)
        self.assertFalse(check.available)

    def test_draft_release_is_ignored(self):
        check = updates.evaluate_release(
            _release_payload(draft=True), APP_VERSION
        )
        self.assertFalse(check.available)
        self.assertIsNone(check.latest)

    def test_release_without_a_usable_version_reports_why(self):
        payload = _release_payload()
        payload["tag_name"] = "nightly"
        payload["name"] = "nightly"
        check = updates.evaluate_release(payload, APP_VERSION)
        self.assertFalse(check.available)
        self.assertIn("version number", check.error)

    def test_garbage_payload_does_not_raise(self):
        check = updates.evaluate_release("not a dict", APP_VERSION)
        self.assertFalse(check.ok)
        self.assertFalse(check.available)

    def test_asset_for_arch_picks_the_right_installer(self):
        release = updates.evaluate_release(_release_payload(), APP_VERSION).latest
        self.assertIn("Setup-x64.exe", release.asset_for_arch("x64").name)
        self.assertIn("Setup-x86.exe", release.asset_for_arch("x86").name)

    def test_asset_for_arch_is_none_when_that_installer_is_missing(self):
        payload = _release_payload(assets=[])
        release = updates.evaluate_release(payload, APP_VERSION).latest
        self.assertIsNone(release.asset_for_arch("x64"))


class NetworkTest(unittest.TestCase):
    def test_check_for_update_uses_the_releases_api(self):
        with mock.patch.object(updates.requests, "get",
                               return_value=_Response(json.dumps(_release_payload()))) as get:
            check = updates.check_for_update(APP_VERSION)
        self.assertTrue(check.available)
        self.assertEqual(get.call_args[0][0], updates.LATEST_RELEASE_API)
        self.assertIn("AI-Voice-Studio/", get.call_args[1]["headers"]["User-Agent"])

    def test_a_network_failure_is_reported_not_raised(self):
        import requests

        with mock.patch.object(updates.requests, "get",
                               side_effect=requests.ConnectionError("no internet")):
            check = updates.check_for_update(APP_VERSION)
        self.assertFalse(check.ok)
        self.assertIn("Could not reach GitHub", check.error)
        self.assertFalse(check.available)

    def test_unreadable_answer_is_reported(self):
        with mock.patch.object(updates.requests, "get",
                               return_value=_Response("{not json")):
            check = updates.check_for_update(APP_VERSION)
        self.assertFalse(check.ok)
        self.assertIn("unreadable", check.error)

    def test_download_without_an_installer_for_this_arch_is_an_error(self):
        release = updates.evaluate_release(
            _release_payload(assets=[]), APP_VERSION
        ).latest
        with self.assertRaises(updates.UpdateError):
            updates.download_installer(release, dest_dir=tempfile.gettempdir())

    def test_download_uses_the_model_downloader(self):
        release = updates.evaluate_release(_release_payload(), APP_VERSION).latest
        with mock.patch("ai_voice_studio.tts.downloader.download_file",
                        return_value=r"C:\temp\setup.exe") as download:
            path = updates.download_installer(
                release, arch="x64", dest_dir=r"C:\temp"
            )
        self.assertEqual(path, r"C:\temp\setup.exe")
        url, dest = download.call_args[0][0], download.call_args[0][1]
        self.assertIn("Setup-x64.exe", url)
        self.assertEqual(dest, r"C:\temp")

    def test_launching_a_missing_installer_is_an_error(self):
        with self.assertRaises(updates.UpdateError):
            updates.launch_installer(os.path.join(tempfile.gettempdir(),
                                                  "definitely-not-here.exe"))


class _SettingsCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.settings = Settings(path=os.path.join(self._tmp.name, "settings.json"))


class AutoCheckTest(_SettingsCase):
    def test_due_when_never_checked(self):
        self.assertTrue(update_dialog.auto_check_due(self.settings))

    def test_not_due_within_a_day(self):
        update_dialog.mark_checked(self.settings)
        self.assertFalse(update_dialog.auto_check_due(self.settings))

    def test_due_again_after_a_day(self):
        old = datetime.now(UTC) - timedelta(days=2)
        update_dialog.mark_checked(self.settings, now=old)
        self.assertTrue(update_dialog.auto_check_due(self.settings))

    def test_disabled_by_the_setting(self):
        self.settings.set("updates.auto_check", False)
        self.assertFalse(update_dialog.auto_check_due(self.settings))

    def test_unreadable_timestamp_still_checks(self):
        self.settings.set("updates.last_check", "yesterday-ish")
        self.assertTrue(update_dialog.auto_check_due(self.settings))

    def test_the_check_is_remembered_on_disk(self):
        update_dialog.mark_checked(self.settings)
        reopened = Settings(path=os.path.join(self._tmp.name, "settings.json"))
        self.assertTrue(reopened.get("updates.last_check"))
        self.assertFalse(update_dialog.auto_check_due(reopened))


class MuteVersionTest(_SettingsCase):
    def test_a_muted_version_is_not_offered(self):
        check = updates.evaluate_release(_release_payload(), APP_VERSION)
        self.assertTrue(update_dialog.should_offer(self.settings, check))
        update_dialog.mute_version(self.settings, check.latest.version)
        self.assertFalse(update_dialog.should_offer(self.settings, check))

    def test_another_version_is_still_offered(self):
        update_dialog.mute_version(self.settings, "2030.1.1")
        check = updates.evaluate_release(_release_payload("2031.2.0"), APP_VERSION)
        self.assertTrue(update_dialog.should_offer(self.settings, check))

    def test_a_failed_check_is_never_offered(self):
        check = updates.UpdateCheck(current=APP_VERSION, error="offline")
        self.assertFalse(update_dialog.should_offer(self.settings, check))

    def test_the_mute_is_remembered_on_disk(self):
        update_dialog.mute_version(self.settings, "2030.1.1")
        reopened = Settings(path=os.path.join(self._tmp.name, "settings.json"))
        self.assertEqual(update_dialog.muted_version(reopened), "2030.1.1")


class UpdateDialogTest(unittest.TestCase):
    """The dialog itself, built for real so its controls are exercised."""

    @classmethod
    def setUpClass(cls):
        import wx

        cls.wx = wx
        cls.app = get_app()

    def test_dialog_shows_both_versions_and_can_mute(self):
        check = updates.evaluate_release(_release_payload(), APP_VERSION)
        dialog = update_dialog.UpdateAvailableDialog(None, check)
        try:
            labels = []

            def collect(window):
                for child in window.GetChildren():
                    if isinstance(child, self.wx.StaticText):
                        labels.append(child.GetLabel())
                    collect(child)

            collect(dialog)
            text = "\n".join(labels)
            self.assertIn(APP_VERSION, text)
            self.assertIn("2030.1.1", text)
            self.assertFalse(dialog.mute_requested())
            dialog.mute_box.SetValue(True)
            self.assertTrue(dialog.mute_requested())
        finally:
            dialog.Destroy()

    def test_dialog_has_three_actions_with_access_keys(self):
        check = updates.evaluate_release(_release_payload(), APP_VERSION)
        dialog = update_dialog.UpdateAvailableDialog(None, check)
        try:
            buttons = []

            def collect(window):
                for child in window.GetChildren():
                    if isinstance(child, self.wx.Button):
                        buttons.append(child.GetLabel())
                    collect(child)

            collect(dialog)
            labels = [b.replace("&", "") for b in buttons]
            self.assertIn("Download and install", labels)
            self.assertIn("Open release page", labels)
            self.assertIn("Remind me later", labels)
            # Every button keeps a live access key: the dialog is keyboard-only.
            for label in buttons:
                self.assertIn("&", label)
        finally:
            dialog.Destroy()


class PresentOutcomeTest(_SettingsCase):
    """A check must always answer the user, and never raise into the UI."""

    @classmethod
    def setUpClass(cls):
        import wx

        cls.wx = wx
        cls.app = get_app()

    def test_a_failed_check_says_why_when_the_user_asked(self):
        check = updates.UpdateCheck(current=APP_VERSION, error="no internet")
        with mock.patch.object(self.wx, "MessageBox") as box:
            update_dialog._present(None, self.settings, check, manual=True)
        self.assertIn("no internet", box.call_args[0][0])

    def test_a_failed_automatic_check_stays_silent(self):
        check = updates.UpdateCheck(current=APP_VERSION, error="no internet")
        with mock.patch.object(self.wx, "MessageBox") as box:
            update_dialog._present(None, self.settings, check, manual=False)
        box.assert_not_called()

    def test_an_up_to_date_answer_is_only_shown_on_request(self):
        check = updates.evaluate_release(_release_payload(APP_VERSION), APP_VERSION)
        with mock.patch.object(self.wx, "MessageBox") as box:
            update_dialog._present(None, self.settings, check, manual=True)
        self.assertIn("newest version", box.call_args[0][0])
        with mock.patch.object(self.wx, "MessageBox") as quiet:
            update_dialog._present(None, self.settings, check, manual=False)
        quiet.assert_not_called()

    def test_an_available_update_is_offered(self):
        check = updates.evaluate_release(_release_payload(), APP_VERSION)
        with mock.patch.object(update_dialog, "UpdateAvailableDialog") as dialog, \
                mock.patch.object(self.wx, "CallLater"):
            dialog.return_value.ShowModal.return_value = self.wx.ID_CANCEL
            dialog.return_value.mute_requested.return_value = False
            update_dialog._present(None, self.settings, check, manual=True)
            dialog.assert_called_once()

    def test_a_muted_version_is_not_offered_again(self):
        check = updates.evaluate_release(_release_payload(), APP_VERSION)
        update_dialog.mute_version(self.settings, check.latest.version)
        with mock.patch.object(update_dialog, "UpdateAvailableDialog") as dialog, \
                mock.patch.object(self.wx, "MessageBox") as box:
            update_dialog._present(None, self.settings, check, manual=True)
            dialog.assert_not_called()
        # Asking on purpose must say the truth: a newer version exists, it is
        # only muted - not "you already have the newest version".
        self.assertIn("asked not to be reminded", box.call_args[0][0])
        self.assertNotIn("newest version", box.call_args[0][0])


class SchedulingTest(_SettingsCase):
    """The start-up hook: quiet, once a day, and never fatal."""

    def test_a_due_check_starts_a_background_worker_that_checks(self):
        with mock.patch.object(update_dialog.threading, "Thread") as thread:
            update_dialog.schedule_auto_check(None, self.settings)
        thread.assert_called_once()
        kwargs = thread.call_args[1]
        self.assertTrue(kwargs.get("daemon"), "must not keep the app alive")

        # Run the worker's body directly: it must check, remember the check
        # and hand the outcome back to the UI thread.
        with mock.patch.object(update_dialog, "updates") as fake, \
                mock.patch.object(update_dialog.wx, "CallAfter") as call_after:
            fake.check_for_update.return_value = updates.UpdateCheck(
                current=APP_VERSION)
            kwargs["target"]()
            fake.check_for_update.assert_called_once()
        call_after.assert_called_once()
        self.assertTrue(self.settings.get("updates.last_check"))

    def test_a_check_that_is_not_due_starts_no_worker(self):
        update_dialog.mark_checked(self.settings)
        with mock.patch.object(update_dialog.threading, "Thread") as thread:
            update_dialog.schedule_auto_check(None, self.settings)
        thread.assert_not_called()

    def test_startup_survives_a_broken_check(self):
        """``main._schedule_update_check`` must never stop the application."""
        import main as entry

        with mock.patch("ai_voice_studio.gui.update_dialog.schedule_auto_check",
                        side_effect=RuntimeError("boom")):
            entry._schedule_update_check(None, self.settings)  # must not raise


if __name__ == "__main__":
    unittest.main()
