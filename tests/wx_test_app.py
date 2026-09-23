"""One ``wx.App`` for the whole test process.

wxPython allows a single ``wx.App`` per process: constructing a second one
shuts the first down (``wxEntryCleanup``), while the first one's windows,
sizers and event handlers are still alive as Python objects.  Those orphans
are then deallocated during the interpreter's final garbage collection, when
wx's own static state has already gone, and the process dies with an access
violation *after* every test has passed::

    Garbage-collecting
    <no Python frame>

That is exactly what the suite did, so ``pytest`` reported a failure (exit
code 139) for a run in which nothing failed.  Measured before the fix: any
two of the GUI modules, or four of the ``_AppMixin`` classes inside
``test_omni_language_boxes.py``, were enough to crash; each on its own was
clean, because only a second ``wx.App`` starts the shutdown of the first.

Every GUI test module therefore asks for the app here instead of building its
own.  The instance is held at module level so it outlives every test class and
is torn down once, at interpreter exit.

Usage::

    from wx_test_app import get_app

    class SomeGuiTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.app = get_app()
"""

from __future__ import annotations

import wx

_APP: wx.App | None = None


def get_app() -> wx.App:
    """The process-wide test application (created on first use)."""
    global _APP
    if _APP is None:
        _APP = wx.App(False)
    return _APP
