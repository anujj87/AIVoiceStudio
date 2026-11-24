"""The configurable keyboard-gesture model (after NVDA's Input Gestures).

The GUI side (the settings panel and the capture dialog) is covered by the
smoke tests; these are the pure-model guarantees: normalization, the user map
replacing the defaults, conflict reporting, and the round trip through the
settings file.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_voice_studio import gestures
from ai_voice_studio.settings import Settings


class NormalizeTest(unittest.TestCase):
    def test_modifier_order_and_case_do_not_matter(self):
        for spelling in ("kb:shift+control+n", "KB:Control+Shift+N",
                         "control+shift+n"):
            self.assertEqual(gestures.normalize(spelling),
                             "kb:control+shift+n", spelling)

    def test_aliases_are_understood(self):
        self.assertEqual(gestures.normalize("kb:ctrl+alt+space"),
                         "kb:control+alt+space")
        self.assertEqual(gestures.normalize("kb:win+escape"),
                         "kb:windows+escape")

    def test_a_gesture_needs_exactly_one_key(self):
        self.assertEqual(gestures.normalize("kb:control"), "")
        self.assertEqual(gestures.normalize("kb:"), "")
        self.assertEqual(gestures.normalize("kb:control+a+b"), "")

    def test_display_name_is_readable(self):
        self.assertEqual(gestures.display_name("kb:control+shift+n"),
                         "Ctrl+Shift+N")
        self.assertEqual(gestures.display_name("kb:control+comma"),
                         "Ctrl+Comma")

    def test_is_valid_requires_the_source_prefix(self):
        self.assertTrue(gestures.is_valid("kb:control+shift+n"))
        self.assertFalse(gestures.is_valid("control+shift+n"))
        self.assertFalse(gestures.is_valid("kb:"))


class CommandCatalogTest(unittest.TestCase):
    def test_every_command_has_a_known_category_and_label(self):
        for command in gestures.COMMANDS:
            self.assertIn(command.category, gestures.CATEGORIES, command.id)
            self.assertTrue(command.label, command.id)

    def test_command_ids_are_unique(self):
        ids = gestures.command_ids()
        self.assertEqual(len(ids), len(set(ids)))

    def test_defaults_are_already_normalized(self):
        for command in gestures.COMMANDS:
            for gesture in command.defaults:
                self.assertEqual(gestures.normalize(gesture), gesture,
                                 f"{command.id}: {gesture}")

    def test_the_shipped_defaults_do_not_collide(self):
        # Two commands must never be given the same key out of the box; the
        # dialog would report it as a conflict the user never asked for.
        self.assertEqual(gestures.GestureMap().conflicts(), {})

    def test_every_category_is_reachable(self):
        for category in gestures.categories_present():
            self.assertTrue(gestures.commands_in(category), category)


class BindingMapTest(unittest.TestCase):
    def test_defaults_apply_when_the_user_has_not_changed_anything(self):
        gesture_map = gestures.GestureMap()
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+shift+n"])
        self.assertFalse(gesture_map.is_customized("project.new"))

    def test_a_user_binding_replaces_the_defaults(self):
        gesture_map = gestures.GestureMap({"project.new": ["kb:control+n"]})
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+n"])
        self.assertTrue(gesture_map.is_customized("project.new"))
        self.assertIsNone(gesture_map.owner("kb:control+shift+n"))

    def test_an_empty_list_unbinds_a_command(self):
        gesture_map = gestures.GestureMap({"project.new": []})
        self.assertEqual(gesture_map.bindings_for("project.new"), [])
        self.assertIsNone(gesture_map.owner("kb:control+shift+n"))

    def test_unknown_commands_and_malformed_gestures_are_ignored(self):
        gesture_map = gestures.GestureMap({
            "not.a.command": ["kb:control+x"],
            "project.new": ["kb:control", "kb:control+m"],
        })
        self.assertNotIn("not.a.command", gesture_map.export())
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+m"])

    def test_bindings_are_normalized_and_de_duplicated(self):
        gesture_map = gestures.GestureMap({
            "project.new": ["KB:Shift+Control+N", "kb:control+shift+n"],
        })
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+shift+n"])

    def test_a_scalar_binding_is_accepted(self):
        gesture_map = gestures.GestureMap({"project.new": "kb:control+n"})
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+n"])

    def test_add_and_remove(self):
        gesture_map = gestures.GestureMap()
        gesture_map.add("project.open", "kb:control+shift+o")
        self.assertIn("kb:control+shift+o",
                      gesture_map.bindings_for("project.open"))
        gesture_map.remove("project.open", "kb:control+shift+o")
        self.assertNotIn("kb:control+shift+o",
                         gesture_map.bindings_for("project.open"))

    def test_add_with_replace_owner_takes_the_key_from_its_old_command(self):
        gesture_map = gestures.GestureMap()
        gesture_map.add("project.open", "kb:control+shift+n",
                        replace_owner=True)
        self.assertNotIn("kb:control+shift+n",
                         gesture_map.bindings_for("project.new"))
        self.assertIn("kb:control+shift+n",
                      gesture_map.bindings_for("project.open"))

    def test_add_without_replace_owner_leaves_a_reported_conflict(self):
        gesture_map = gestures.GestureMap()
        gesture_map.add("project.open", "kb:control+shift+n")
        conflicts = gesture_map.conflicts()
        self.assertIn("kb:control+shift+n", conflicts)
        self.assertEqual(sorted(conflicts["kb:control+shift+n"]),
                         ["project.new", "project.open"])

    def test_reset_restores_the_defaults(self):
        gesture_map = gestures.GestureMap({"project.new": ["kb:control+n"]})
        gesture_map.reset("project.new")
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+shift+n"])
        self.assertFalse(gesture_map.is_customized("project.new"))

    def test_reset_all_clears_every_override(self):
        gesture_map = gestures.GestureMap({"project.new": ["kb:control+n"],
                                           "project.open": []})
        gesture_map.reset_all()
        self.assertEqual(gesture_map.export(), {})
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+shift+n"])

    def test_export_round_trips_through_load(self):
        original = gestures.GestureMap({"project.new": ["kb:control+n"],
                                        "project.open": []})
        restored = gestures.GestureMap(original.export())
        self.assertEqual(restored.export(), original.export())

    def test_owner_finds_the_command_for_a_gesture(self):
        gesture_map = gestures.GestureMap()
        self.assertEqual(gesture_map.owner("kb:control+shift+n"), "project.new")
        self.assertIsNone(gesture_map.owner("kb:control+shift+z"))

    def test_set_bindings_rejects_an_unknown_command(self):
        with self.assertRaises(KeyError):
            gestures.GestureMap().set_bindings("nope", ["kb:control+n"])


class SettingsRoundTripTest(unittest.TestCase):
    def test_the_default_settings_carry_an_empty_gesture_map(self):
        # A fresh settings file, not the user's: this asks what the shipped
        # default is, and the shortcuts someone has given themselves in the
        # running application have nothing to do with that answer.
        import shutil
        import tempfile

        folder = tempfile.mkdtemp(prefix="aivs_gestures_")
        self.addCleanup(shutil.rmtree, folder, True)
        settings = Settings(path=os.path.join(folder, "settings.json"))
        self.assertEqual(settings.get("gestures.bindings"), {})

    def test_bindings_survive_a_save_and_reload(self):
        import tempfile

        folder = tempfile.mkdtemp(prefix="aivs_gestures_")
        path = os.path.join(folder, "settings.json")
        settings = Settings(path=path)
        settings.set("gestures.bindings", {"project.new": ["kb:control+n"]})
        settings.save()
        reloaded = Settings(path=path)
        gesture_map = gestures.GestureMap(
            reloaded.get("gestures.bindings", {}) or {}
        )
        self.assertEqual(gesture_map.bindings_for("project.new"),
                         ["kb:control+n"])


if __name__ == "__main__":
    unittest.main()
