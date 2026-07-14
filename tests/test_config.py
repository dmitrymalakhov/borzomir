import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from borzomir_bot.config import ConfigError, load_settings, parse_user_ids
from borzomir_bot.http_json import redact_sensitive_url


class ConfigTest(unittest.TestCase):
    def test_admin_user_id_defaults_to_requested_id(self):
        settings = load_settings({"TELEGRAM_BOT_TOKEN": "token"})

        self.assertEqual(settings.admin_user_id, 92174505)

    def test_admin_is_allowed_even_with_allowlist(self):
        settings = load_settings(
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "ADMIN_USER_ID": "92174505",
                "ALLOWED_USER_IDS": "100,200",
            }
        )

        self.assertTrue(settings.is_user_allowed(92174505))
        self.assertTrue(settings.is_user_allowed(100))
        self.assertFalse(settings.is_user_allowed(300))

    def test_voice_settings_default_to_local_ruslan_piper_voice(self):
        settings = load_settings({"TELEGRAM_BOT_TOKEN": "token"})

        self.assertTrue(settings.voice_input_enabled)
        self.assertEqual(settings.voice_reply_mode, "voice-input")
        self.assertEqual(settings.whisper_language, "ru")
        self.assertIn("ru_RU-ruslan-medium", settings.piper_model_path)

    def test_voice_reply_mode_is_validated(self):
        with self.assertRaises(ConfigError):
            load_settings({"TELEGRAM_BOT_TOKEN": "token", "VOICE_REPLY_MODE": "sometimes"})

    def test_parse_user_ids_accepts_commas_and_spaces(self):
        self.assertEqual(parse_user_ids("1, 2,3"), frozenset({1, 2, 3}))

    def test_token_is_required(self):
        with self.assertRaises(ConfigError):
            load_settings({})

    def test_redact_sensitive_telegram_url(self):
        url = "https://api.telegram.org/bot123456:secret/getUpdates"

        self.assertEqual(redact_sensitive_url(url), "https://api.telegram.org/bot<redacted>/getUpdates")


if __name__ == "__main__":
    unittest.main()
