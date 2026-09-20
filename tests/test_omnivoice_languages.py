"""Tests for the OmniVoice language table (the manual language picker).

The table is a generated copy of upstream ``k2-fsa/OmniVoice``
``docs/languages.md``.  These tests pin its shape (so a bad regeneration is
caught instead of reaching the engine), the lookup rules the dialogs rely on
(name, ISO 639-3 code, picker label), and the way a stored hint travels from
``project.json`` to the request builders.
"""

from __future__ import annotations

import unittest

from ai_voice_studio.omnivoice import languages, spec


class LanguageTableTest(unittest.TestCase):
    def test_the_table_holds_every_language_upstream_lists(self):
        self.assertEqual(languages.COUNT, 646)
        self.assertEqual(len(languages.LANGUAGES), 646)
        self.assertEqual(len(languages.language_ids()), 646)

    def test_every_row_is_complete_and_lower_case(self):
        for name, omni_id, iso in languages.LANGUAGES:
            with self.subTest(language=name):
                self.assertTrue(name.strip())
                self.assertEqual(omni_id, omni_id.lower())
                self.assertEqual(iso, iso.lower())
                self.assertEqual(len(iso), 3)

    def test_ids_and_names_are_unique(self):
        ids = languages.language_ids()
        self.assertEqual(len(set(ids)), len(ids))
        names = languages.names()
        self.assertEqual(len({name.lower() for name in names}), len(names))

    def test_the_well_known_languages_are_present(self):
        for omni_id, name in (
            ("en", "English"), ("zh", "Chinese"), ("hi", "Hindi"),
            ("arb", "Standard Arabic"), ("es", "Spanish"), ("fr", "French"),
            ("de", "German"), ("ja", "Japanese"), ("ko", "Korean"),
            ("ru", "Russian"), ("pt", "Portuguese"), ("vi", "Vietnamese"),
            ("yue", "Cantonese"), ("nb", "Norwegian Bokmål"),
        ):
            with self.subTest(language=name):
                self.assertIn(omni_id, languages.language_ids())
                self.assertEqual(languages.display_name(omni_id), name)

    def test_display_name_keeps_an_unknown_value(self):
        self.assertEqual(languages.display_name(""), "")
        self.assertEqual(languages.display_name(None), "")
        self.assertEqual(languages.display_name("  xx  "), "xx")


class LanguageLookupTest(unittest.TestCase):
    def test_auto_means_no_hint(self):
        for value in (None, "", "   ", "auto", "AUTO", "Automatic", "any",
                      languages.AUTO_LABEL):
            with self.subTest(value=value):
                self.assertIsNone(languages.normalise(value))
                self.assertEqual(languages.selection_index(value), 0)

    def test_ids_iso_codes_names_and_labels_all_resolve(self):
        self.assertEqual(languages.normalise("en"), "en")
        self.assertEqual(languages.normalise("EN"), "en")
        self.assertEqual(languages.normalise(" eng "), "en")
        self.assertEqual(languages.normalise("cmn"), "zh")      # ISO 639-3 -> id
        self.assertEqual(languages.normalise("Chinese"), "zh")
        self.assertEqual(languages.normalise("English (en)"), "en")
        self.assertEqual(languages.normalise("en (English)"), "en")

    def test_every_row_resolves_from_its_name_and_its_iso_code(self):
        for name, omni_id, iso in languages.LANGUAGES:
            with self.subTest(language=name):
                self.assertEqual(languages.normalise(name), omni_id)
                self.assertEqual(languages.normalise(iso), omni_id)
                self.assertEqual(languages.normalise(languages.label(omni_id)),
                                 omni_id)

    def test_an_unknown_hint_is_passed_through_lower_cased(self):
        # A newer engine may know languages this table does not: refusing the
        # hint here would break a project that used to work.
        self.assertEqual(languages.normalise("xx-YY"), "xx-yy")
        self.assertEqual(languages.normalise(" Elvish "), "elvish")

    def test_the_picker_starts_with_auto(self):
        choices = languages.choices()
        self.assertEqual(len(choices), languages.COUNT + 1)
        self.assertEqual(choices[0], (languages.AUTO_LABEL, languages.AUTO))
        self.assertEqual([value for _label, value in choices[1:]],
                         list(languages.language_ids()))

    def test_selection_index_finds_a_row(self):
        choices = languages.choices()
        self.assertEqual(choices[languages.selection_index("en")][1], "en")
        self.assertEqual(
            choices[languages.selection_index("eng")][1], "en")  # ISO code
        self.assertEqual(choices[languages.selection_index("de")][1], "de")
        # Auto is index 0, and so is anything the table cannot place.
        self.assertEqual(languages.selection_index(""), 0)
        self.assertEqual(languages.selection_index("no-such-language"), 0)


class SpecIntegrationTest(unittest.TestCase):
    """``clean_language`` is what the engines actually receive."""

    def test_clean_language_uses_the_table(self):
        self.assertEqual(spec.clean_language("English"), "en")
        self.assertEqual(spec.clean_language("English (en)"), "en")
        self.assertEqual(spec.clean_language("cmn"), "zh")
        self.assertEqual(spec.clean_language(" german "), "de")

    def test_clean_language_still_means_auto_for_auto(self):
        for value in (None, "", "auto", "Automatic", languages.AUTO_LABEL):
            with self.subTest(value=value):
                self.assertIsNone(spec.clean_language(value))

    def test_the_examples_are_real_language_ids(self):
        """The short list must not drift from the table it illustrates."""
        for example in spec.LANGUAGE_EXAMPLES:
            omni_id, _bracket, name = example.partition(" (")
            omni_id = omni_id.strip()
            with self.subTest(example=example):
                self.assertEqual(languages.normalise(omni_id), omni_id)
                self.assertIn(omni_id, languages.language_ids())
                self.assertEqual(languages.display_name(omni_id),
                                 name.rstrip(")").strip())

    def test_a_chosen_language_reaches_the_request(self):
        request = spec.build_omni(mode="auto", language="Hindi")
        self.assertEqual(request["language"], "hi")
        voice = spec.apply_omni_to_voice({"engine": "omnivoice"}, request)
        self.assertEqual(voice["language"], "hi")

    def test_the_recording_summary_names_the_pinned_language(self):
        """The recording window has to say which language was forced."""
        from ai_voice_studio.gui.recording_dialog import RecordingDialog

        summary = RecordingDialog._omni_summary_text(
            {"mode": "auto", "language": "hi", "num_step": 32})
        self.assertIn("language Hindi", summary)
        self.assertIn("32 steps", summary)
        # Auto is no hint: it must not claim a language it does not send.
        self.assertNotIn("language",
                         RecordingDialog._omni_summary_text({"mode": "auto"}))


if __name__ == "__main__":
    unittest.main()
