"""Update dialogs: check for a newer release, and install it.

Two entry points, both used by the main window and the entry point:

* :func:`schedule_auto_check` - a quiet background check at startup, at most
  once a day (``settings["updates"]["auto_check"]``).  It only ever speaks
  when a newer version is found and the user has not muted that version.
* :func:`check_for_updates_interactive` - the Help menu's *Check for
  updates*: always answers, including "you already have the newest
  version" and a plain reason when GitHub cannot be reached.

The download reuses the model downloader's transfer (resumable, cancellable)
and Inno Setup's own silent switches, the same ones winget uses.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading

import wx

from .. import updates
from ..constants import APP_NAME, APP_VERSION
from .a11y import finalize_accessibility, set_accessible_name
from .dialogs import run_action_dialog
from .progress import TaskProgressDialog

log = logging.getLogger(__name__)

# How long the automatic check stays quiet after a successful one.
AUTO_CHECK_INTERVAL_SECONDS = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Settings helpers (pure, so the tests can drive them without a GUI)
# ---------------------------------------------------------------------------
def _updates_settings(settings) -> dict:
    block = settings.get("updates", None) if settings is not None else None
    return block if isinstance(block, dict) else {}


def auto_check_enabled(settings) -> bool:
    return bool(_updates_settings(settings).get("auto_check", True))


def auto_check_due(settings, now: _dt.datetime | None = None) -> bool:
    """True when an automatic check is enabled and not done in the last day."""
    if not auto_check_enabled(settings):
        return False
    last = str(_updates_settings(settings).get("last_check") or "")
    if not last:
        return True
    try:
        previous = _dt.datetime.fromisoformat(last)
    except ValueError:
        return True  # an unreadable timestamp must not disable checking
    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=_dt.timezone.utc)
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return (now - previous).total_seconds() >= AUTO_CHECK_INTERVAL_SECONDS


def mark_checked(settings, now: _dt.datetime | None = None) -> None:
    """Remember that a check just succeeded (throttles the automatic one)."""
    if settings is None:
        return
    block = dict(_updates_settings(settings))
    now = now or _dt.datetime.now(_dt.timezone.utc)
    block["last_check"] = now.isoformat(timespec="seconds")
    settings.set("updates", block)
    settings.save()


def muted_version(settings) -> str:
    return str(_updates_settings(settings).get("skip_version") or "")


def should_offer(settings, check: updates.UpdateCheck) -> bool:
    """True when this check warrants interrupting the user."""
    if not check.available or check.latest is None:
        return False
    return updates.parse_version(check.latest.version) != updates.parse_version(
        muted_version(settings)
    )


def mute_version(settings, version: str) -> None:
    """Never remind about ``version`` again (the user asked to be left alone)."""
    if settings is None:
        return
    block = dict(_updates_settings(settings))
    block["skip_version"] = str(version or "")
    settings.set("updates", block)
    settings.save()


# ---------------------------------------------------------------------------
# The "a newer version is available" dialog
# ---------------------------------------------------------------------------
class UpdateAvailableDialog(wx.Dialog):
    """Offers the new release, with an opt-out for that one version."""

    def __init__(self, parent: wx.Window, check: updates.UpdateCheck):
        release = check.latest
        super().__init__(parent, title=f"Update available - {APP_NAME}")
        self.muted = False

        sizer = wx.BoxSizer(wx.VERTICAL)

        headline = wx.StaticText(
            self,
            label=(
                f"A newer version of {APP_NAME} is available.\n\n"
                f"Installed version: {check.current}\n"
                f"New version: {release.version}"
            ),
        )
        headline.SetName("Update available")
        headline.Wrap(520)
        sizer.Add(headline, 0, wx.EXPAND | wx.ALL, 12)

        notes_label = wx.StaticText(self, label="What is new:")
        sizer.Add(notes_label, 0, wx.LEFT | wx.RIGHT, 12)
        notes = wx.TextCtrl(
            self,
            value=(release.notes or "No release notes were published.").strip(),
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(520, 160),
        )
        set_accessible_name(notes, "What is new in this version")
        sizer.Add(notes, 1, wx.EXPAND | wx.ALL, 12)

        self.mute_box = wx.CheckBox(
            self, label="Do not remind me about this version again"
        )
        set_accessible_name(
            self.mute_box, "Do not remind me about this version again"
        )
        sizer.Add(self.mute_box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        row = wx.BoxSizer(wx.HORIZONTAL)
        row.AddStretchSpacer(1)
        install = wx.Button(self, label="&Download and install")
        install.SetToolTip(
            "Download the installer for this Windows version and start it"
        )
        install.Bind(wx.EVT_BUTTON, lambda _: self.EndModal(wx.ID_OK))
        page = wx.Button(self, label="Open release &page")
        page.SetToolTip("Open the release page in your web browser")
        page.Bind(wx.EVT_BUTTON, lambda _: self.EndModal(wx.ID_YES))
        later = wx.Button(self, label="Re&mind me later")
        later.SetToolTip("Close this window without updating")
        later.Bind(wx.EVT_BUTTON, lambda _: self.EndModal(wx.ID_CANCEL))
        for button in (install, page, later):
            button.SetName(button.GetLabel().replace("&", ""))
            row.Add(button, 0, wx.ALL, 6)
        sizer.Add(row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.SetSizer(sizer)
        self.Fit()
        self.SetMinSize(self.GetSize())
        finalize_accessibility(self)
        if parent is not None and parent.IsShown():
            self.CentreOnParent()
        else:
            self.Centre()
        wx.CallAfter(install.SetFocus)

    def mute_requested(self) -> bool:
        """True when the user ticked “do not remind me about this version”."""
        return bool(self.mute_box.GetValue())

    #: Button result asking for the browser instead of the download.
    RESULT_OPEN_PAGE = wx.ID_YES


# ---------------------------------------------------------------------------
# Downloading and running the installer
# ---------------------------------------------------------------------------
def _open_release_page(release: updates.ReleaseInfo) -> None:
    import webbrowser  # noqa: PLC0415

    try:
        webbrowser.open(release.page_url or updates.RELEASES_PAGE)
    except Exception:  # noqa: BLE001 - opening a page must never raise
        log.debug("Could not open %s", release.page_url, exc_info=True)


def _offer_install(parent: wx.Window, release: updates.ReleaseInfo, path: str) -> None:
    """Ask whether to run the freshly downloaded installer now."""
    choice = run_action_dialog(
        parent,
        "Install the update",
        f"{APP_NAME} {release.version} has been downloaded.\n\n"
        "Install it now? The application will close so the installer can "
        "replace its files. Choose \u201cInstall later\u201d to keep the copy "
        "and start it yourself.",
        [("install", "&Install now"), ("later", "Install &later")],
        default_key="install",
    )
    if choice != "install":
        wx.MessageBox(
            "The installer was kept here:\n\n" + path,
            "Update downloaded", style=wx.OK | wx.ICON_INFORMATION, parent=parent,
        )
        return
    try:
        updates.launch_installer(path)
    except updates.UpdateError as exc:
        wx.MessageBox(str(exc), "Could not start the installer",
                      style=wx.OK | wx.ICON_ERROR, parent=parent)
        return
    top = parent.GetTopLevelParent() if parent is not None else None
    if top is not None:
        # Give the installer a moment to appear before our window goes away.
        wx.CallLater(600, top.Close)


def download_and_install(parent: wx.Window, release: updates.ReleaseInfo) -> None:
    """Download this release's installer, then offer to run it."""
    cancel = threading.Event()
    progress = TaskProgressDialog(
        parent,
        title="Downloading update",
        message=f"Downloading {APP_NAME} {release.version}...",
        cancel_label="Cancel",
        on_cancel=cancel.set,
    )
    progress.Show()
    progress.start_pulse()

    def on_progress(_name: str, done: int, total: int) -> None:
        if not total:
            return
        percent = int(done * 100 / total)
        wx.CallAfter(
            progress.update, percent,
            f"Downloading {APP_NAME} {release.version}: {percent}% "
            f"({done // 1024} of {max(1, total // 1024)} KB)",
        )

    def finished(path: str | None, error: str | None) -> None:
        progress.finish()
        if error:
            wx.MessageBox(error, "Update failed", style=wx.OK | wx.ICON_ERROR,
                          parent=parent)
            return
        _offer_install(parent, release, path)

    def worker() -> None:
        from ..tts.downloader import DownloadCancelled  # noqa: PLC0415

        try:
            path = updates.download_installer(release, progress=on_progress,
                                              cancel_event=cancel)
        except DownloadCancelled:
            log.info("Update download cancelled by the user")
            wx.CallAfter(progress.finish)
            return
        except Exception as exc:  # noqa: BLE001 - reported in the dialog
            log.warning("Update download failed: %s", exc)
            wx.CallAfter(finished, None, str(exc))
            return
        wx.CallAfter(finished, path, None)

    threading.Thread(target=worker, daemon=True, name="aivs-update-download").start()


# ---------------------------------------------------------------------------
# Presenting a finished check
# ---------------------------------------------------------------------------
def _present(parent: wx.Window, settings, check: updates.UpdateCheck,
             manual: bool) -> None:
    """Show the outcome of a check (a quiet one stays quiet unless there is news)."""
    if not check.ok:
        if manual:
            wx.MessageBox(
                "Could not check for updates.\n\n" + check.error,
                "Check for updates", style=wx.OK | wx.ICON_WARNING, parent=parent,
            )
        else:
            log.info("Automatic update check failed: %s", check.error)
        return

    if not should_offer(settings, check):
        if not manual:
            return
        if check.available and check.latest is not None:
            # A newer version exists, but the user muted exactly this one.
            # Saying "you have the newest version" would be untrue.
            wx.MessageBox(
                f"{APP_NAME} {check.latest.version} is available, but you "
                "asked not to be reminded about that version.\n\n"
                f"Installed version: {check.current}\n"
                f"Release page: {check.latest.page_url}",
                "Check for updates", style=wx.OK | wx.ICON_INFORMATION,
                parent=parent,
            )
            return
        wx.MessageBox(
            f"You are using the newest version of {APP_NAME}.\n\n"
            f"Installed version: {check.current}",
            "Check for updates", style=wx.OK | wx.ICON_INFORMATION,
            parent=parent,
        )
        return

    release = check.latest
    log.info("Update available: %s (installed %s)", release.version, check.current)
    dialog = UpdateAvailableDialog(parent, check)
    try:
        result = dialog.ShowModal()
        muted = dialog.mute_requested()
    finally:
        dialog.Destroy()

    if muted:
        mute_version(settings, release.version)
        log.info("User asked not to be reminded about %s", release.version)
    if result == UpdateAvailableDialog.RESULT_OPEN_PAGE:
        _open_release_page(release)
    elif result == wx.ID_OK:
        download_and_install(parent, release)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def check_for_updates_interactive(parent: wx.Window, settings) -> None:
    """Help menu action: check, and always tell the user the outcome."""
    progress = TaskProgressDialog(
        parent,
        title="Check for updates",
        message=f"Looking for a newer version of {APP_NAME}...",
    )
    progress.Show()
    progress.start_pulse()
    outcome: dict = {}

    def worker() -> None:
        outcome["check"] = updates.check_for_update()
        wx.CallAfter(done)

    def done() -> None:
        progress.finish()
        check = outcome.get("check") or updates.UpdateCheck(
            current=APP_VERSION, error="The check did not finish."
        )
        if check.ok:
            mark_checked(settings)
        _present(parent, settings, check, manual=True)

    threading.Thread(target=worker, daemon=True, name="aivs-update-check").start()


def schedule_auto_check(parent: wx.Window, settings) -> None:
    """Start the once-a-day background check, if it is due and enabled."""
    if not auto_check_due(settings):
        return

    def worker() -> None:
        check = updates.check_for_update()
        if check.ok:
            mark_checked(settings)
        wx.CallAfter(landed, check)

    def landed(check: updates.UpdateCheck) -> None:
        try:
            _present(parent, settings, check, manual=False)
        except Exception:  # noqa: BLE001 - a background check must never crash
            log.debug("Automatic update check presentation failed", exc_info=True)

    threading.Thread(target=worker, daemon=True, name="aivs-update-auto").start()
