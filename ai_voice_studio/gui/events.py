"""Custom wx events used to communicate between worker threads and the GUI.

Events are posted with ``wx.PostEvent`` from worker threads and handled on the
UI thread. Every event carries a short, screen-reader-friendly ``message``.

wxPython convention: a raw event type (``wx.NewEventType()``) is passed to the
event class constructor, while a ``wx.PyEventBinder`` (``EVT_*``) is used in
``Bind(...)`` calls.
"""

from __future__ import annotations

import wx

_SYNTH_STATUS = wx.NewEventType()
_SYNTH_SEGMENT_DONE = wx.NewEventType()
_SYNTH_FINISHED = wx.NewEventType()
_SYNTH_ERROR = wx.NewEventType()

_DOWNLOAD_PROGRESS = wx.NewEventType()
_DOWNLOAD_FINISHED = wx.NewEventType()
_DOWNLOAD_ERROR = wx.NewEventType()

EVT_SYNTH_STATUS = wx.PyEventBinder(_SYNTH_STATUS, 1)
EVT_SYNTH_SEGMENT_DONE = wx.PyEventBinder(_SYNTH_SEGMENT_DONE, 1)
EVT_SYNTH_FINISHED = wx.PyEventBinder(_SYNTH_FINISHED, 1)
EVT_SYNTH_ERROR = wx.PyEventBinder(_SYNTH_ERROR, 1)

EVT_DOWNLOAD_PROGRESS = wx.PyEventBinder(_DOWNLOAD_PROGRESS, 1)
EVT_DOWNLOAD_FINISHED = wx.PyEventBinder(_DOWNLOAD_FINISHED, 1)
EVT_DOWNLOAD_ERROR = wx.PyEventBinder(_DOWNLOAD_ERROR, 1)


class SynthStatusEvent(wx.PyCommandEvent):
    def __init__(self, message: str):
        super().__init__(_SYNTH_STATUS)
        self.message = message


class SynthSegmentDoneEvent(wx.PyCommandEvent):
    def __init__(self, index: int, title: str, path: str):
        super().__init__(_SYNTH_SEGMENT_DONE)
        self.index = index
        self.title = title
        self.path = path


class SynthFinishedEvent(wx.PyCommandEvent):
    def __init__(self):
        super().__init__(_SYNTH_FINISHED)


class SynthErrorEvent(wx.PyCommandEvent):
    def __init__(self, message: str):
        super().__init__(_SYNTH_ERROR)
        self.message = message


class DownloadProgressEvent(wx.PyCommandEvent):
    def __init__(self, name: str, done: int, total: int):
        super().__init__(_DOWNLOAD_PROGRESS)
        self.name = name
        self.done = done
        self.total = total


class DownloadFinishedEvent(wx.PyCommandEvent):
    def __init__(self, success: bool, message: str):
        super().__init__(_DOWNLOAD_FINISHED)
        self.success = success
        self.message = message
