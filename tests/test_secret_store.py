"""The two stored credentials are encrypted at rest (DPAPI).

``settings.json`` holds the Hugging Face read token and the OmniVoice server's
API key.  They used to sit there as plain text, which any process running as the
user could read - a backup agent, a sync client, a support bundle.  Windows
already offers the right primitive for a per-user desktop application (DPAPI),
so the values are wrapped with it and the file stays readable everywhere else.

The tests assert the properties that matter:

* the token is not in the file, in any spelling;
* it comes back unchanged through a save/load cycle;
* an older settings file with a plain-text token still loads, and is rewritten
  encrypted on the spot, so nobody's token is lost by the upgrade;
* a value that cannot be decrypted is *empty*, not a crash;
* the rest of the file - key names, other settings, the JSON structure -
  is untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_voice_studio import secret_store
from ai_voice_studio.settings import Settings

TOKEN = "hf_thisIsTheUsersReadToken0000"
API_KEY = "omni-server-key-1234"


class SecretStoreTest(unittest.TestCase):
    def setUp(self):
        if not secret_store.available():
            self.skipTest("DPAPI is not available on this platform")

    def _temp_settings(self) -> tuple[Settings, str]:
        folder = tempfile.mkdtemp(prefix="aivs_secrets_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "settings.json")
        return Settings(path=path), path

    def _raw(self, path: str) -> str:
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_a_round_trip_returns_the_same_token(self):
        self.assertEqual(
            secret_store.unprotect(secret_store.protect(TOKEN)), TOKEN
        )

    def test_protecting_twice_does_not_double_wrap(self):
        once = secret_store.protect(TOKEN)
        self.assertEqual(secret_store.protect(once), once)

    def test_plain_text_passes_through_unprotect(self):
        self.assertEqual(secret_store.unprotect("just-text"), "just-text")

    def test_a_corrupt_value_is_empty_rather_than_an_exception(self):
        self.assertEqual(secret_store.unprotect(secret_store.PREFIX + "!!!"), "")

    def test_a_saved_token_is_not_in_the_file(self):
        settings, path = self._temp_settings()
        settings.set("hf.token", TOKEN)

        settings.save()

        raw = self._raw(path)
        self.assertNotIn(TOKEN, raw, "the read token is on disk in plain text")
        self.assertIn(secret_store.PREFIX, raw)

    def test_a_saved_api_key_is_not_in_the_file(self):
        settings, path = self._temp_settings()
        settings.set("omnivoice_server.api_key", API_KEY)

        settings.save()

        self.assertNotIn(API_KEY, self._raw(path))

    def test_the_rest_of_the_file_stays_readable(self):
        settings, path = self._temp_settings()
        settings.set("hf.token", TOKEN)
        settings.set("theme", "dark")
        settings.set("omnivoice_server.host", "127.0.0.1")

        settings.save()
        stored = json.loads(self._raw(path))

        self.assertEqual(stored["theme"], "dark")
        self.assertEqual(stored["omnivoice_server"]["host"], "127.0.0.1")
        self.assertIn("hf", stored)
        self.assertIn("token", stored["hf"])

    def test_a_saved_token_reads_back_unchanged(self):
        settings, path = self._temp_settings()
        settings.set("hf.token", TOKEN)
        settings.set("omnivoice_server.api_key", API_KEY)
        settings.save()

        reopened = Settings(path=path)

        self.assertEqual(reopened.get("hf.token"), TOKEN)
        self.assertEqual(reopened.get("omnivoice_server.api_key"), API_KEY)

    def test_an_old_plain_text_file_loads_and_is_migrated(self):
        folder = tempfile.mkdtemp(prefix="aivs_secrets_legacy_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "settings.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"hf": {"token": TOKEN}, "theme": "dark"}, handle)

        settings = Settings(path=path)

        self.assertEqual(settings.get("hf.token"), TOKEN,
                         "an upgrade must not lose the user's token")
        raw = self._raw(path)
        self.assertNotIn(TOKEN, raw,
                         "the plain-text token was left on disk after loading")
        self.assertEqual(json.loads(raw)["theme"], "dark")

    def test_an_empty_token_is_not_encrypted(self):
        settings, path = self._temp_settings()
        settings.set("hf.token", "")
        settings.save()

        self.assertEqual(json.loads(self._raw(path))["hf"]["token"], "")

    def test_has_plaintext_secrets_finds_both_keys(self):
        plain = {"hf": {"token": TOKEN}, "omnivoice_server": {"api_key": API_KEY}}
        self.assertTrue(secret_store.has_plaintext_secrets(plain))

        wrapped = secret_store.protect_settings(plain)
        self.assertFalse(secret_store.has_plaintext_secrets(wrapped))
        self.assertEqual(secret_store.unprotect_settings(wrapped), plain)

    def test_a_non_secret_looking_key_is_not_touched(self):
        data = {"tts": {"token_budget": "120"}}
        self.assertEqual(secret_store.protect_settings(data), data)


class SecretStoreUnavailableTest(unittest.TestCase):
    """Without DPAPI the behaviour is exactly the old plain-text one."""

    def test_protect_and_unprotect_are_the_identity(self):
        original = secret_store._api_loaded, secret_store._api
        secret_store._api_loaded, secret_store._api = True, None
        try:
            self.assertFalse(secret_store.available())
            self.assertEqual(secret_store.protect(TOKEN), TOKEN)
            self.assertEqual(secret_store.unprotect(TOKEN), TOKEN)
        finally:
            secret_store._api_loaded, secret_store._api = original


if __name__ == "__main__":
    unittest.main()
