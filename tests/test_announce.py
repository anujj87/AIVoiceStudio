"""Live announcements: what gets pushed to the screen reader, and when.

Two layers are tested separately:

* the **policy** in ``announce.Announcer`` - politeness, coalescing, pacing and
  the assertive override - with a fake clock and a recording ``notify``
  function, so the timing rules are asserted exactly instead of being waited
  for; and
* the **delivery** - the live-region control, its MSAA name and state, and the
  events raised at its window handle - through the suite's ``wx`` harness.

The behaviour these tests lock is the reason the module exists: a progress bar
that moves, a status line that changes and an error that appears in a label are
otherwise silent, and a screen reader hears them only when the user navigates
to the control it happens on.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import wx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wx_test_app import get_app

from ai_voice_studio.gui import a11y, announce
from ai_voice_studio.gui.progress import ANNOUNCE_STEP, TaskProgressDialog
from ai_voice_studio.settings import Settings
from ai_voice_studio.tts.models import ModelStore


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Recorder:
    """Records the MSAA events instead of sending them to the desktop."""

    def __init__(self):
        self.events: list[tuple[int, int]] = []

    def __call__(self, hwnd: int, event_id: int) -> bool:
        self.events.append((hwnd, event_id))
        return True

    @property
    def ids(self) -> list[int]:
        return [event_id for _hwnd, event_id in self.events]


def _announcer(**kwargs) -> tuple[announce.Announcer, _Clock, _Recorder]:
    clock = kwargs.pop("clock", None) or _Clock()
    recorder = kwargs.pop("notify", None) or _Recorder()
    announcer = announce.Announcer(
        clock=clock,
        notify=recorder,
        is_gui_thread=lambda: True,
        auto_pump=False,
        **kwargs,
    )
    return announcer, clock, recorder


class AnnouncementPolicyTest(unittest.TestCase):
    """Pacing and priority, with no window involved."""

    def test_the_first_polite_message_is_delivered_at_once(self):
        announcer, _clock, _recorder = _announcer()

        accepted = announcer.announce("Loading the model")

        self.assertTrue(accepted)
        self.assertEqual(announcer.history, ["Loading the model"])
        self.assertEqual(announcer.pending, [])

    def test_an_empty_message_is_not_an_announcement(self):
        announcer, _clock, _recorder = _announcer()

        self.assertFalse(announcer.announce(""))
        self.assertFalse(announcer.announce("   \n\t "))
        self.assertEqual(announcer.history, [])

    def test_whitespace_is_collapsed_before_it_reaches_the_reader(self):
        announcer, _clock, _recorder = _announcer()

        announcer.announce("Saved  segment\n  3  of 12")

        self.assertEqual(announcer.history, ["Saved segment 3 of 12"])

    def test_repeating_what_was_just_read_is_dropped(self):
        announcer, clock, _recorder = _announcer()
        announcer.announce("Recording complete.")
        clock.advance(announce.DEFAULT_MIN_INTERVAL * 2)

        self.assertFalse(announcer.announce("Recording complete."))
        self.assertEqual(announcer.history, ["Recording complete."])

    def test_a_second_message_inside_the_interval_waits_its_turn(self):
        announcer, clock, _recorder = _announcer()
        announcer.announce("20 percent")

        announcer.announce("30 percent", key="progress")

        self.assertEqual(announcer.history, ["20 percent"])
        self.assertEqual(announcer.pending, ["30 percent"])

        clock.advance(announce.DEFAULT_MIN_INTERVAL)
        self.assertTrue(announcer.pump())
        self.assertEqual(announcer.history, ["20 percent", "30 percent"])
        self.assertEqual(announcer.pending, [])

    def test_pump_does_nothing_while_the_interval_is_still_running(self):
        announcer, _clock, _recorder = _announcer()
        announcer.announce("20 percent")
        announcer.announce("30 percent", key="progress")

        self.assertFalse(announcer.pump())
        self.assertEqual(announcer.history, ["20 percent"])

    def test_progress_updates_collapse_to_the_latest_one_per_key(self):
        # A download reporting every megabyte must not leave a backlog of
        # percentages for the reader to wade through.
        announcer, clock, _recorder = _announcer()
        announcer.announce("Downloading", key="progress-message")
        for percent in (10, 20, 30, 40):
            announcer.announce(f"Downloading: {percent} MB", key="progress")

        self.assertEqual(announcer.pending, ["Downloading: 40 MB"])

        clock.advance(announce.DEFAULT_MIN_INTERVAL)
        announcer.pump()
        self.assertEqual(announcer.history, ["Downloading", "Downloading: 40 MB"])

    def test_a_status_update_is_not_starved_by_a_progress_stream(self):
        announcer, clock, _recorder = _announcer()
        announcer.announce("10 percent", key="progress")
        announcer.announce("20 percent", key="progress")
        announcer.announce("Saved segment 1 of 9", key="status")

        self.assertEqual(announcer.pending,
                         ["20 percent", "Saved segment 1 of 9"])

        clock.advance(announce.DEFAULT_MIN_INTERVAL)
        announcer.pump()
        self.assertEqual(announcer.history[-1], "20 percent")

        clock.advance(announce.DEFAULT_MIN_INTERVAL)
        announcer.pump()
        self.assertEqual(announcer.history[-1], "Saved segment 1 of 9")

    def test_the_queue_is_bounded(self):
        announcer, _clock, _recorder = _announcer()
        announcer.announce("first")

        for index in range(announce.MAX_PENDING + 3):
            announcer.announce(f"status {index}", key=f"k{index}")

        self.assertEqual(len(announcer.pending), announce.MAX_PENDING)
        # The newest survive; the ones that fell off are the oldest.
        self.assertEqual(announcer.pending[-1], f"status {announce.MAX_PENDING + 2}")

    def test_an_error_interrupts_and_drops_the_stale_percentages(self):
        announcer, clock, _recorder = _announcer()
        announcer.announce("10 percent", key="progress")
        announcer.announce("20 percent", key="progress")

        announcer.error("Choose a document to read first.")

        self.assertEqual(announcer.pending, [])
        self.assertEqual(announcer.history,
                         ["10 percent", "Choose a document to read first."])
        # An error is not paced: the next one is delivered immediately too.
        clock.advance(0)
        self.assertTrue(announcer.error("There is no audio to export."))
        self.assertEqual(announcer.history[-1], "There is no audio to export.")

    def test_an_error_is_delivered_even_as_a_repeat(self):
        # "Choose a document" is exactly what the user may have done wrong
        # twice; the second press must answer again.
        announcer, _clock, _recorder = _announcer()
        announcer.error("Choose a document to read first.")

        self.assertTrue(announcer.error("Choose a document to read first."))
        self.assertEqual(announcer.history.count("Choose a document to read first."), 2)

    def test_flush_delivers_what_is_waiting_and_reports_it(self):
        announcer, _clock, _recorder = _announcer()
        announcer.announce("10 percent")
        announcer.announce("20 percent", key="progress")
        announcer.announce("Finished", key="status")

        sent = announcer.flush()

        self.assertEqual(sent, ["20 percent", "Finished"])
        self.assertEqual(announcer.pending, [])
        self.assertIn("Finished", announcer.history)

    def test_a_disabled_announcer_is_silent(self):
        announcer, _clock, _recorder = _announcer()
        announcer.enabled = False

        self.assertFalse(announcer.announce("Loading"))
        self.assertFalse(announcer.error("Failed"))
        self.assertEqual(announcer.history, [])

    def test_a_background_thread_posts_the_message_instead_of_touching_wx(self):
        # A download worker cannot touch wx: announce() must hand the message
        # to the GUI thread and return, not call SetName from the worker.
        on_gui_thread = [False]
        posted: list = []
        announcer = announce.Announcer(
            clock=_Clock(),
            notify=_Recorder(),
            is_gui_thread=lambda: on_gui_thread[0],
            post=posted.append,
            auto_pump=False,
        )

        self.assertTrue(announcer.announce("Downloaded 4 of 9 files"))
        self.assertEqual(len(posted), 1)
        self.assertEqual(announcer.history, [])

        on_gui_thread[0] = True
        posted[0]()

        self.assertEqual(announcer.history, ["Downloaded 4 of 9 files"])

    def test_history_is_bounded(self):
        announcer, clock, _recorder = _announcer(min_interval=0.0)

        for index in range(announce.HISTORY_LENGTH + 5):
            clock.advance(1)
            announcer.announce(f"message {index}")

        self.assertEqual(len(announcer.history), announce.HISTORY_LENGTH)
        self.assertEqual(announcer.history[-1],
                         f"message {announce.HISTORY_LENGTH + 4}")


class _WindowTest(unittest.TestCase):
    """Base for the tests that need a real window and its MSAA tree."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def setUp(self):
        self.frame = wx.Frame(None, title="Announcement test")
        self.frame.SetClientSize((320, 200))
        # MSAA only answers for a window that is on screen: a hidden window's
        # accessible object reports "not implemented" for everything.
        self.frame.Show()
        self.previous = announce.get_announcer()
        self.clock = _Clock()
        self.recorder = _Recorder()
        self.announcer = announce.Announcer(
            self.frame,
            clock=self.clock,
            notify=self.recorder,
            is_gui_thread=lambda: True,
            auto_pump=False,
        )
        announce.set_announcer(self.announcer)

    def tearDown(self):
        announce.set_announcer(self.previous)
        # A test may have destroyed the frame itself; that is what it tested.
        with contextlib.suppress(RuntimeError):
            self.frame.Destroy()


class LiveRegionDeliveryTest(_WindowTest):
    """What the screen reader is actually handed."""

    def test_the_region_is_an_accessible_static_text_in_the_window(self):
        self.announcer.announce("Loading the model")

        region = self.announcer.live_region
        self.assertIsNotNone(region)
        self.assertIs(region.GetParent(), self.frame)
        self.assertIsInstance(region.GetAccessible(), a11y.LiveRegionAccessible)

    def test_the_region_carries_the_current_message_as_its_msaa_name(self):
        self.announcer.announce("Rendering page 3 of 12")
        region = self.announcer.live_region

        code, name = region.GetAccessible().GetName(0)

        self.assertEqual(code, wx.ACC_OK)
        self.assertEqual(name, "Rendering page 3 of 12")

    def test_the_region_is_marked_as_a_live_region_for_the_screen_reader(self):
        self.announcer.announce("Saved")
        accessible = self.announcer.live_region.GetAccessible()

        _code, role = accessible.GetRole(0)
        _code, state = accessible.GetState(0)

        self.assertEqual(role, wx.ROLE_SYSTEM_STATICTEXT)
        self.assertTrue(state & a11y.STATE_SYSTEM_LIVE_REGION)

    def test_the_region_can_be_told_apart_from_the_interface(self):
        """A 1x1 label that is empty most of the time is not a broken control."""
        self.announcer.announce("Saved")
        region = self.announcer.live_region

        self.assertTrue(announce.is_live_region(region))
        self.assertFalse(announce.is_live_region(self.frame))
        self.assertFalse(announce.is_live_region(wx.StaticText(self.frame)))

    def test_both_events_are_raised_at_the_region(self):
        self.announcer.announce("Saved")

        self.assertEqual(self.recorder.ids,
                         [announce.EVENT_OBJECT_LIVEREGIONCHANGED,
                          announce.EVENT_OBJECT_NAMECHANGE])
        hwnd = self.announcer.live_region.GetHandle()
        self.assertTrue(hwnd)
        self.assertEqual({hwnd for hwnd, _event in self.recorder.events}, {hwnd})

    def test_nothing_is_raised_for_a_repeat(self):
        self.announcer.announce("Saved")
        self.recorder.events.clear()

        self.assertFalse(self.announcer.announce("Saved"))

        self.assertEqual(self.recorder.events, [])

    def test_the_region_follows_the_announcer_to_another_window(self):
        self.announcer.announce("First window")
        other = wx.Frame(None, title="Second")
        try:
            self.announcer.attach(other)
            self.clock.advance(announce.DEFAULT_MIN_INTERVAL * 2)
            self.announcer.announce("Second window")

            self.assertIs(self.announcer.live_region.GetParent(), other)
            _code, name = self.announcer.live_region.GetAccessible().GetName(0)
            self.assertEqual(name, "Second window")
        finally:
            other.Destroy()

    def test_the_region_follows_the_user_into_the_next_window(self):
        """A modal dialog has the focus; the live region has to move with it."""
        announcer = announce.Announcer(
            self.frame, clock=self.clock, notify=self.recorder,
            is_gui_thread=lambda: True, auto_pump=False, follow_active=True,
        )
        announce.set_announcer(announcer)
        dialog = wx.Frame(None, title="Dialog")
        self.addCleanup(dialog.Destroy)

        with mock.patch.object(wx, "GetActiveWindow", return_value=self.frame):
            announce.announce("On the main window")
        with mock.patch.object(wx, "GetActiveWindow", return_value=dialog):
            self.clock.advance(announce.DEFAULT_MIN_INTERVAL * 2)
            announce.announce("In the dialog")

        self.assertIs(announcer.live_region.GetParent(), dialog)
        _code, name = announcer.live_region.GetAccessible().GetName(0)
        self.assertEqual(name, "In the dialog")

    def test_a_background_thread_does_not_look_up_the_active_window(self):
        # wx may not be touched off the GUI thread, so a worker's message must
        # be posted without asking which window is in front.
        announcer = announce.Announcer(
            self.frame, clock=self.clock, notify=self.recorder,
            is_gui_thread=lambda: False, post=lambda call: None,
            auto_pump=False, follow_active=True,
        )
        with mock.patch.object(wx, "GetActiveWindow") as active:
            self.assertTrue(announcer.follow_the_active_window() is self.frame)
        self.assertFalse(active.called)

    def test_a_destroyed_window_does_not_break_announcing(self):
        self.announcer.announce("Before the window goes")
        self.clock.advance(announce.DEFAULT_MIN_INTERVAL * 2)
        self.frame.Destroy()

        # No live region, no exception: the message is still recorded, and a
        # window that is gone simply cannot be read from.
        self.assertTrue(self.announcer.announce("After the window goes"))
        self.assertEqual(self.announcer.history[-1], "After the window goes")



class StatusLineTest(_WindowTest):
    """The helpers that wire a visible label to the announcement."""

    def setUp(self):
        super().setUp()
        self.label = wx.StaticText(self.frame, label="")

    def test_status_text_sets_the_label_and_announces_the_same_string(self):
        announce.status_text(self.label, "Opened project 'Alice'")

        self.assertEqual(self.label.GetLabel(), "Opened project 'Alice'")
        self.assertEqual(self.announcer.history, ["Opened project 'Alice'"])

    def test_status_error_is_assertive(self):
        self.announcer.announce("50 percent", key="progress")
        self.announcer.announce("60 percent", key="progress")

        announce.status_error(self.label, "Could not start recording: no device")

        self.assertEqual(self.label.GetLabel(),
                         "Could not start recording: no device")
        self.assertEqual(self.announcer.pending, [])
        self.assertEqual(self.announcer.history[-1],
                         "Could not start recording: no device")


class ProgressDialogAnnouncementTest(_WindowTest):
    """A moving progress bar is heard, not just seen."""

    def _dialog(self) -> TaskProgressDialog:
        return TaskProgressDialog(self.frame, title="Working", message="Preparing...")

    def test_the_first_message_is_announced_when_the_dialog_appears(self):
        dialog = self._dialog()
        try:
            dialog.set_message("Loading the model")

            self.assertEqual(self.label_text(dialog), "Loading the model")
            self.assertIn("Loading the model", self.announcer.history)
        finally:
            dialog.finish()

    def label_text(self, dialog: TaskProgressDialog) -> str:
        return dialog._message.GetLabel()

    def test_progress_is_announced_in_steps_not_for_every_value(self):
        dialog = self._dialog()
        try:
            self.announcer.reset()
            dialog.show_progress(0)
            for percent in range(1, ANNOUNCE_STEP):
                self.clock.advance(announce.DEFAULT_MIN_INTERVAL)
                dialog.show_progress(percent)

            self.assertEqual(self.announcer.history, ["0 percent"])

            self.clock.advance(announce.DEFAULT_MIN_INTERVAL)
            dialog.show_progress(ANNOUNCE_STEP)
            self.assertEqual(self.announcer.history, ["0 percent", "10 percent"])
        finally:
            dialog.finish()

    def test_finishing_flushes_the_message_still_waiting(self):
        dialog = self._dialog()
        try:
            self.announcer.reset()
            dialog.show_progress(0)
            dialog.set_message("Saving the DAISY book")   # lands in the queue
            self.assertEqual(self.announcer.pending, ["Saving the DAISY book"])

            dialog.finish()

            self.assertEqual(self.announcer.pending, [])
            self.assertIn("Saving the DAISY book", self.announcer.history)
        except Exception:
            dialog.finish()
            raise

    def test_the_label_and_the_announcement_never_disagree(self):
        dialog = self._dialog()
        try:
            for message in ("Parsing Alice.epub", "Splitting into 12 parts",
                            "Finished"):
                self.clock.advance(announce.DEFAULT_MIN_INTERVAL * 2)
                dialog.set_message(message)
                self.assertEqual(dialog._message.GetLabel(), message)
                self.assertIn(message, self.announcer.history)
        finally:
            dialog.finish()


class MainFrameStatusTest(unittest.TestCase):
    """The main window's status bar reports through the announcement layer."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def setUp(self):
        import shutil
        import tempfile

        from ai_voice_studio.gui.main_frame import MainFrame
        from ai_voice_studio.settings import Settings
        from ai_voice_studio.tts.models import ModelStore

        self.previous = announce.get_announcer()
        self.clock = _Clock()
        self.recorder = _Recorder()
        self.announcer = announce.Announcer(
            clock=self.clock,
            notify=self.recorder,
            is_gui_thread=lambda: True,
            auto_pump=False,
        )
        announce.set_announcer(self.announcer)

        folder = tempfile.mkdtemp(prefix="aivs_announce_")
        self.addCleanup(shutil.rmtree, folder, True)
        state = os.path.join(folder, "models.json")
        self.frame = MainFrame(
            settings=Settings(path=os.path.join(folder, "settings.json")),
            store=ModelStore(state_file=state),
        )
        self.addCleanup(self.frame.Destroy)

    def tearDown(self):
        announce.set_announcer(self.previous)

    def test_a_status_line_update_is_written_and_announced(self):
        self.frame._set_status("Opened C:\\books\\Alice")

        self.assertEqual(self.frame.GetStatusBar().GetStatusText(),
                         "Opened C:\\books\\Alice")
        self.assertEqual(self.announcer.history, ["Opened C:\\books\\Alice"])

    def test_the_status_bar_and_the_reader_hear_the_same_string(self):
        for message in ("Ready.", "Removed project 'Alice'.",
                        "Deleted 12 file(s); the project starts from the beginning."):
            self.clock.advance(announce.DEFAULT_MIN_INTERVAL * 2)
            self.frame._set_status(message)
            self.assertEqual(self.frame.GetStatusBar().GetStatusText(), message)
            self.assertEqual(self.announcer.history[-1], message)


class RealWindowStatusTest(_WindowTest):
    """Every window that shows a status line pushes it, not just once.

    These build the real widgets rather than a stand-in label: the point is
    that a user who cannot see the label hears the same string from the same
    code path, so the wiring is what is under test here.
    """

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="aivs_announce_status_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.settings = Settings(path=os.path.join(self.folder, "settings.json"))
        self.store = ModelStore(state_file=os.path.join(self.folder, "models.json"))

    # -- speech to text -----------------------------------------------------
    def _stt_dialog(self):
        """A transcribe dialog on a real (if empty) STT project."""
        from ai_voice_studio import project
        from ai_voice_studio.constants import MODE_PAGE_WITH_H1, PROJECT_TYPE_STT
        from ai_voice_studio.gui.stt_dialog import STTTranscribeDialog

        folder = os.path.join(self.folder, "project")
        os.makedirs(folder, exist_ok=True)
        project.create_project(
            folder, "STT", "doc.txt", MODE_PAGE_WITH_H1,
            [{"index": 1, "title": "One", "text": "Hello."}],
            project_type=PROJECT_TYPE_STT,
            stt_settings={"model": "whisper_large_v3", "sources": []},
        )
        dialog = STTTranscribeDialog(self.frame, folder, self.settings,
                                     self.store)
        self.addCleanup(dialog.Destroy)
        return dialog

    def test_stt_progress_and_failure_are_heard(self):
        dialog = self._stt_dialog()
        self.announcer.reset()

        dialog._update_progress("Transcribing file 2 of 5", 0.4)

        self.assertEqual(dialog.status.GetLabel(), "Transcribing file 2 of 5")
        self.assertEqual(self.announcer.history, ["Transcribing file 2 of 5"])

        # A failure interrupts: it must not wait behind the progress stream.
        self.clock.advance(0)
        with mock.patch.object(wx, "MessageBox", return_value=wx.OK):
            dialog._on_error(RuntimeError("model files are missing"))

        self.assertEqual(dialog.status.GetLabel(), "Failed: model files are missing")
        self.assertEqual(self.announcer.history[-1],
                         "Failed: model files are missing")
        self.assertEqual(self.announcer.pending, [])

    def test_stt_state_changes_are_announced_once_each(self):
        dialog = self._stt_dialog()
        self.announcer.reset()

        dialog._sync_start_state()
        self.assertIn(dialog.status.GetLabel(), self.announcer.history)

        heard = len(self.announcer.history)
        dialog._sync_start_state()      # the same state again: silence
        self.assertEqual(len(self.announcer.history), heard)

    # -- theme studio -------------------------------------------------------
    def test_saving_a_theme_says_so(self):
        from ai_voice_studio.gui.theme_studio import ThemeStudioDialog

        dialog = ThemeStudioDialog(self.frame, self.settings)
        self.addCleanup(dialog.Destroy)
        dialog.name_ctrl.SetValue("Ocean Night")
        self.announcer.reset()

        dialog._save()

        self.assertEqual(dialog.status.GetLabel(), "Saved 'Ocean Night'.")
        self.assertEqual(self.announcer.history, ["Saved 'Ocean Night'."])

    # -- the Available TTS list --------------------------------------------
    def test_a_preview_failure_in_the_voice_list_is_assertive(self):
        from ai_voice_studio.gui import dialogs as gui_dialogs
        from ai_voice_studio.gui.model_panels import AvailablePanel

        panel = AvailablePanel(self.frame, self.store)
        self.addCleanup(panel.Destroy)
        self.announcer.announce("20 percent", key="progress")
        self.announcer.announce("30 percent", key="progress")

        with mock.patch.object(gui_dialogs, "notify_engine_error"):
            panel._preview_done(None, "Preview failed: no voice selected")

        self.assertEqual(panel.detail.GetLabel(), "Preview failed: no voice selected")
        self.assertEqual(self.announcer.history[-1],
                         "Preview failed: no voice selected")
        self.assertEqual(self.announcer.pending, [])

    # -- the new-project wizard --------------------------------------------
    def test_adding_sources_to_the_wizard_is_heard(self):
        from ai_voice_studio.gui.new_project_wizard import NewProjectWizard

        wizard = NewProjectWizard(self.frame, self.settings, self.store)
        self.addCleanup(wizard.Destroy)
        wizard.stt_sources.extend([
            os.path.join(self.folder, "lecture 1.wav"),
            os.path.join(self.folder, "lecture 2.wav"),
        ])
        self.announcer.reset()

        wizard.page_details._refresh_sources()

        self.assertEqual(wizard.page_details.sources_count.GetLabel(),
                         "2 source(s) added.")
        self.assertEqual(self.announcer.history, ["2 source(s) added."])

    def test_adding_videos_to_the_wizard_is_heard(self):
        from ai_voice_studio.gui.new_project_wizard import NewProjectWizard

        wizard = NewProjectWizard(self.frame, self.settings, self.store)
        self.addCleanup(wizard.Destroy)
        wizard.describe_sources.append(os.path.join(self.folder, "film.mp4"))
        self.announcer.reset()

        wizard.page_details._refresh_ad_sources()

        self.assertEqual(wizard.page_details.ad_sources_count.GetLabel(),
                         "1 video(s) added.")
        self.assertEqual(self.announcer.history, ["1 video(s) added."])

    # -- the voice clone panel ---------------------------------------------
    def test_install_progress_in_the_clone_panel_is_heard(self):
        from ai_voice_studio.gui.clone_engines_panel import VoiceClonePanel

        panel = VoiceClonePanel(self.frame, self.settings, self.store)
        self.addCleanup(panel.Destroy)
        label = wx.StaticText(self.frame, label="Starting...")

        class _InstallDialog:
            _aivs_label = label

        panel._install_dialog = _InstallDialog()
        self.announcer.reset()

        panel._update_progress("Downloading dependencies...")
        panel._update_progress("Installing packages...")
        panel._update_progress("Verifying the engine...")

        # The label always shows the latest step; the reader hears one message
        # per interval, and the steps it missed collapse to the newest.
        self.assertEqual(label.GetLabel(), "Verifying the engine...")
        self.assertEqual(self.announcer.history,
                         ["Downloading dependencies..."])
        self.assertEqual(self.announcer.pending, ["Verifying the engine..."])

        self.clock.advance(announce.DEFAULT_MIN_INTERVAL)
        self.announcer.pump()
        self.assertEqual(self.announcer.history[-1], "Verifying the engine...")


class StatusLineAuditTest(unittest.TestCase):
    """No status line may be painted without being announced.

    The window tests above prove the wiring that exists; this one is the guard
    against new code that quietly writes a status label with ``SetLabel`` and
    leaves a screen-reader user with silence.  The attribute names are the ones
    the application uses for status feedback (a label that reports what just
    happened), not for content that happens to sit in a ``StaticText``.
    """

    STATUS_LABELS = (
        "access_status", "detail", "dependency_status", "engine_status",
        "gpu_status", "lib_status", "omnivoice_status", "pip_status",
        "preview_status", "profiles_status", "server_status", "status",
        "status_label", "sources_count", "ad_sources_count", "sample_hint",
    )

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_no_status_label_is_set_without_announcing_it(self):
        import re

        target = (r"(?:self\.(?:" + "|".join(self.STATUS_LABELS) + r")|"
                  r"eng\[\"status_label\"\]|dlg\._aivs_label)\.SetLabel\(")
        # Clearing a status line is not an update: there is nothing to read.
        pattern = re.compile(target + r"\s*[\"']{2}")
        silent = re.compile(target)
        offenders = []
        gui = os.path.join(ROOT, "ai_voice_studio", "gui")
        for name in sorted(os.listdir(gui)):
            if not name.endswith(".py") or name == "announce.py":
                continue
            with open(os.path.join(gui, name), encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if silent.search(line) and not pattern.search(line):
                        offenders.append(f"{name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [],
                         "these status labels are silent to a screen reader")


class ModuleApiTest(unittest.TestCase):
    """The module-level functions use (and can replace) the app announcer."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_set_announcer_installs_the_one_the_helpers_use(self):
        previous = announce.get_announcer()
        replacement = announce.Announcer(
            clock=_Clock(),
            notify=_Recorder(),
            is_gui_thread=lambda: True,
            auto_pump=False,
        )
        try:
            announce.set_announcer(replacement)
            announce.announce("Hello")

            self.assertIs(announce.get_announcer(), replacement)
            self.assertEqual(replacement.history, ["Hello"])
        finally:
            announce.set_announcer(previous)

    def test_notify_win_event_is_a_no_op_before_anything_is_delivered(self):
        # The real notifier must never raise for a window handle of 0, which is
        # what a window that is being destroyed reports.
        self.assertFalse(announce.notify_win_event(0, announce.EVENT_OBJECT_NAMECHANGE))


if __name__ == "__main__":
    unittest.main()
