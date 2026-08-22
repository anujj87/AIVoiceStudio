"""Tests for custom storage locations (Settings -> paths)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import paths  # noqa: E402


class _FakeSettings:
    """Stand-in for Settings that answers only the values we configure."""

    def __init__(self, values: dict):
        self._values = values

    def get(self, key: str, default=None):
        return self._values.get(key, default)


class TestPaths(unittest.TestCase):
    def test_models_dir_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeSettings({})
            with mock.patch("ai_voice_studio.settings.Settings", return_value=fake), \
                    mock.patch("ai_voice_studio.paths.user_data_dir", return_value=tmp):
                self.assertEqual(paths.models_dir(), os.path.join(tmp, "models"))

    def test_models_dir_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = os.path.join(tmp, "my_models")
            fake = _FakeSettings({"paths.models_dir": custom})
            with mock.patch("ai_voice_studio.settings.Settings", return_value=fake):
                self.assertEqual(paths.models_dir(), custom)
                self.assertTrue(os.path.isdir(custom))

    def test_recordings_dir_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeSettings({})
            with mock.patch("ai_voice_studio.settings.Settings", return_value=fake), \
                    mock.patch("ai_voice_studio.paths.user_data_dir", return_value=tmp):
                self.assertEqual(paths.recordings_dir(), os.path.join(tmp, "projects"))
                self.assertEqual(paths.projects_dir(), paths.recordings_dir())

    def test_recordings_dir_custom(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = os.path.join(tmp, "my_recordings")
            fake = _FakeSettings({"paths.recordings_dir": custom})
            with mock.patch("ai_voice_studio.settings.Settings", return_value=fake):
                self.assertEqual(paths.recordings_dir(), custom)
                self.assertTrue(os.path.isdir(custom))

    def test_project_dir_uses_custom_recordings(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = os.path.join(tmp, "recordings")
            fake = _FakeSettings({"paths.recordings_dir": custom})
            with mock.patch("ai_voice_studio.settings.Settings", return_value=fake):
                d = paths.project_dir("My Book")
                self.assertTrue(d.startswith(custom + os.sep))
                self.assertTrue(os.path.isdir(d))

    def test_non_absolute_custom_value_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeSettings({"paths.models_dir": "relative/path"})
            with mock.patch("ai_voice_studio.settings.Settings", return_value=fake), \
                    mock.patch("ai_voice_studio.paths.user_data_dir", return_value=tmp):
                self.assertEqual(paths.models_dir(), os.path.join(tmp, "models"))


if __name__ == "__main__":
    unittest.main()
