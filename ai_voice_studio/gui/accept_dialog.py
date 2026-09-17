"""First-launch terms acceptance dialog.

Shown by ``main.py`` before the main window. The application only starts
when every checkbox is ticked and the user presses "I Agree". Disagreeing,
unchecking a box, pressing Escape or closing the dialog (even from the
title bar) ends the application instead.

Acceptance is stored in settings together with ``TERMS_VERSION`` (bumped
in ``constants.py`` for each release), so the dialog shows again after an
update or a reinstall into a fresh user profile - but never again during
the same version.
"""

from __future__ import annotations

import textwrap

import wx

from ..constants import TERMS_VERSION

# One entry per required checkbox, in display order.
_TERMS = [
    "I will not use any voice for voice cloning which I don't have the "
    "right to use, or prior permission to use, for voice cloning.",
    "I will not use this application for any illegal work. If I do, I "
    "will be fully responsible for that use.",
    "If I use this application for any commercial purpose, I will read "
    "every TTS license before use and I will use only those TTS engines "
    "which allow commercial use, or for which I have prior permission.",
    "I will use only those SAPI5 and Windows core voices which I have "
    "the right to use.",
    "The creator of this application is not responsible for anything, "
    "including legal or illegal matters, and this project and "
    "application are provided without any guarantee or warranty.",
    "Voice cloning is a regular feature of modern neural TTS, just as a "
    "camera is a regular feature of a phone, so I will use it carefully "
    "and legally.",
]

_AGREE_ALL_LABEL = "I agree with all above terms"

_AGREE_LABEL = "I Agree"
_DISAGREE_LABEL = "Disagree (Close Program)"


class AcceptanceDialog(wx.Dialog):
    """Modal first-run dialog. ``ShowModal()`` returns ``wx.ID_OK`` only
    when the user accepted; every other exit path yields ``wx.ID_NO``."""

    def __init__(self, parent: wx.Window | None = None):
        super().__init__(
            parent,
            title="Welcome to AI Voice Studio - Terms of Use",
            style=wx.DEFAULT_DIALOG_STYLE & ~wx.CLOSE_BOX,
        )
        self._accepted = False

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        intro = wx.StaticText(
            panel,
            label=(
                "Welcome to AI Voice Studio.\n\n"
                "Please read each point below carefully. You must accept "
                "all of them to use this application."
            ),
        )
        intro.Wrap(560)
        sizer.Add(intro, 0, wx.ALL, 10)

        self._checkboxes: list[wx.CheckBox] = []
        for index, text in enumerate(_TERMS, start=1):
            wrapped = "\n".join(textwrap.wrap(text, width=90))
            box = wx.CheckBox(panel, label=f"{index}.  {wrapped}")
            box.Bind(wx.EVT_CHECKBOX, self._on_toggle)
            self._checkboxes.append(box)
            sizer.Add(box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # The last checkbox stands for "all above": ticking it automatically
        # ticks (and un-ticking it un-ticks) every term checkbox.
        self._agree_all = wx.CheckBox(panel, label=_AGREE_ALL_LABEL)
        self._agree_all.SetFont(self._agree_all.GetFont().Bold())
        self._agree_all.Bind(wx.EVT_CHECKBOX, self._on_agree_all)
        self._checkboxes.append(self._agree_all)
        sizer.Add(self._agree_all, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        # Both buttons stay disabled until every checkbox is ticked.
        self._agree_btn = wx.Button(panel, wx.ID_OK, _AGREE_LABEL)
        self._disagree_btn = wx.Button(panel, wx.ID_CANCEL, _DISAGREE_LABEL)
        self._agree_btn.Bind(wx.EVT_BUTTON, self._on_agree)
        self._disagree_btn.Bind(wx.EVT_BUTTON, self._on_disagree)
        btn_row.Add(self._disagree_btn, 0, wx.RIGHT, 8)
        btn_row.Add(self._agree_btn, 0, wx.LEFT, 8)
        sizer.Add(btn_row, 0, wx.ALIGN_CENTRE | wx.ALL, 12)

        panel.SetSizer(sizer)
        sizer.Fit(self)
        self.CentreOnScreen()

        # Nothing may dismiss this dialog except the Agree button:
        # not Escape, not the (hidden) close box, not Alt+F4.
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._on_toggle(None)

    # -- state ---------------------------------------------------------------
    @property
    def accepted(self) -> bool:
        return self._accepted

    def _all_checked(self) -> bool:
        return all(box.GetValue() for box in self._checkboxes)

    @property
    def _term_checkboxes(self) -> list[wx.CheckBox]:
        """Every checkbox except the final "agree with all" one."""
        return self._checkboxes[:-1]

    def _on_agree_all(self, event) -> None:
        value = self._agree_all.GetValue()
        for box in self._term_checkboxes:
            box.SetValue(value)
        self._on_toggle(None)

    def _on_toggle(self, event) -> None:
        ready = self._all_checked()
        self._agree_btn.Enable(ready)
        self._disagree_btn.Enable(ready)

    # -- exits ----------------------------------------------------------------
    def _on_agree(self, event) -> None:
        if not self._all_checked():
            return
        self._accepted = True
        self._finish(wx.ID_OK)

    def _on_disagree(self, event) -> None:
        self._accepted = False
        self._finish(wx.ID_NO)

    def _finish(self, code: int) -> None:
        """End the modal loop, or just close when shown non-modally (tests)."""
        if self.IsModal():
            self.EndModal(code)
        else:
            self.Close()

    def _on_char_hook(self, event) -> None:
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            # Swallow Escape: refusing the terms is an explicit action.
            return
        event.Skip()

    def _on_close(self, event) -> None:
        if not self._accepted:
            # Force-close counts as disagreement: the app must not start.
            self._accepted = False
            if self.IsModal():
                self.EndModal(wx.ID_NO)
            else:
                event.Skip()
            return
        event.Skip()


def terms_current(settings) -> bool:
    """True when the user already accepted the current terms version."""
    return settings.get("terms_version", "") == TERMS_VERSION


def record_acceptance(settings) -> None:
    settings.set("terms_version", TERMS_VERSION)
    settings.save()
