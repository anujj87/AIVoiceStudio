"""The suite may build exactly one ``wx.App``.

A second ``wx.App`` shuts the first one down while its windows, sizers and
event handlers are still alive as Python objects; those orphans are then freed
during the interpreter's final garbage collection, after wx's own state is
gone, and the process dies with an access violation *after* every test has
passed.  That made a green run report exit code 139.

This is the guard for the fix: every GUI test module must ask
``tests/wx_test_app.py`` for the shared application instead of constructing
its own, and no other module in the suite may construct one at all.  Without
it, the next module that writes ``cls.app = wx.App(False)`` would bring back a
failure that looks like a random crash instead of a test failure.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import unittest

import wx

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(TESTS, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, TESTS)

from wx_test_app import get_app  # noqa: E402

# The GUI modules that must share the application.
SHARING_MODULES = (
    "test_a11y_access_keys.py",
    "test_gestures.py",
    "test_gui_smoke.py",
    "test_omni_language_boxes.py",
    "test_updates.py",
)

# Constructing an application: ``wx.App(False)`` and friends, but not the
# class methods such as ``wx.App.Get()``.
_BUILDS_AN_APP = re.compile(r"wx\.App\s*\(")


def _text(name: str) -> str:
    with open(os.path.join(TESTS, name), encoding="utf-8") as handle:
        return handle.read()


def _app_constructions(name: str) -> list:
    """Line numbers where ``name`` really *calls* ``wx.App(...)``.

    Parsed rather than searched: a docstring or a comment that mentions
    ``wx.App(False)`` (as this file does, several times) is not a
    construction, and a text search cannot tell them apart.
    """
    lines = []
    for node in ast.walk(ast.parse(_text(name))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "App"
                and isinstance(func.value, ast.Name)
                and func.value.id == "wx"):
            lines.append(node.lineno)
    return lines


class SharedAppTest(unittest.TestCase):
    def test_get_app_returns_the_same_application(self):
        first = get_app()
        self.assertIsInstance(first, wx.App)
        self.assertIs(get_app(), first,
                      "a second wx.App would shut the first one down")
        self.assertIs(wx.App.Get(), first)

    def test_no_test_module_builds_its_own_application(self):
        offenders = []
        for name in sorted(os.listdir(TESTS)):
            if not name.startswith("test_") or not name.endswith(".py"):
                continue
            for line in _app_constructions(name):
                offenders.append(f"{name}:{line}")
        self.assertEqual(
            offenders, [],
            "use tests/wx_test_app.get_app() instead: "
            + ", ".join(offenders),
        )

    def test_the_gui_modules_share_it(self):
        for name in SHARING_MODULES:
            with self.subTest(module=name):
                self.assertIn("get_app()", _text(name))

    def test_the_helper_is_the_only_place_that_constructs_one(self):
        text = _text("wx_test_app.py")
        self.assertEqual(len(_BUILDS_AN_APP.findall(text)), 1)
        # ... and it holds the instance, so it outlives every test class.
        self.assertIn("_APP = wx.App(False)", text)


if __name__ == "__main__":
    unittest.main()
