import unittest

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from borzomir_bot.bot import (
    BotApp,
    DEFAULT_IMAGE_PROMPT,
    ConversationStore,
    decode_text_document,
    format_summary_prompt,
    is_image_file,
    image_media_type_for_file_path,
    is_text_file,
    media_type_for_file_path,
    normalize_image_prompt,
    select_largest_photo,
    user_requested_voice_reply,
)
from borzomir_bot.config import Settings
from borzomir_bot.speech import normalize_tts_text, normalize_whisper_transcript
from borzomir_bot.state import StateStore
from borzomir_bot.telegram_api import MAX_TELEGRAM_MESSAGE_LENGTH, encode_multipart_form_data, split_telegram_message


class ConversationStoreTest(unittest.TestCase):
    def test_history_is_limited(self):
        store = ConversationStore(max_messages=3)

        for index in range(5):
            store.append(chat_id=1, role="user", content=f"message {index}")

        messages = store.build_messages(chat_id=1, system_prompt="system", user_text="current")

        self.assertEqual([message["content"] for message in messages], ["system", "message 2", "message 3", "message 4", "current"])

    def test_zero_history_keeps_no_messages(self):
        store = ConversationStore(max_messages=0)
        store.append(chat_id=1, role="user", content="hello")

        self.assertEqual(store.stats(), {"chats": 0, "messages": 0})


class TelegramMessageSplitTest(unittest.TestCase):
    def test_long_message_is_split(self):
        parts = split_telegram_message("x" * (MAX_TELEGRAM_MESSAGE_LENGTH + 10))

        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(part) <= MAX_TELEGRAM_MESSAGE_LENGTH for part in parts))

    def test_multipart_form_data_contains_file(self):
        body, content_type = encode_multipart_form_data(
            fields={"chat_id": "100"},
            files={"voice": ("answer.ogg", b"voice-data", "audio/ogg")},
        )

        self.assertIn("multipart/form-data", content_type)
        self.assertIn(b'name="chat_id"', body)
        self.assertIn(b'filename="answer.ogg"', body)
        self.assertIn(b"voice-data", body)


class SpeechHelpersTest(unittest.TestCase):
    def test_normalize_whisper_transcript_removes_timestamps(self):
        transcript = normalize_whisper_transcript(
            "[00:00:00.000 --> 00:00:01.000] Привет\n"
            "[00:00:01.000 --> 00:00:02.000] как дела?"
        )

        self.assertEqual(transcript, "Привет как дела?")

    def test_normalize_tts_text_softens_code_and_links(self):
        text = normalize_tts_text("Посмотри `foo()` и https://example.com\n```python\nprint(1)\n```")

        self.assertIn("foo()", text)
        self.assertIn("ссылка", text)
        self.assertIn("фрагмент кода", text)


class ImageHelpersTest(unittest.TestCase):
    def test_select_largest_photo_prefers_file_size(self):
        photo = select_largest_photo(
            [
                {"file_id": "small", "file_size": 10, "width": 1000, "height": 1000},
                {"file_id": "large", "file_size": 20, "width": 10, "height": 10},
            ]
        )

        self.assertEqual(photo["file_id"], "large")

    def test_select_largest_photo_falls_back_to_dimensions(self):
        photo = select_largest_photo(
            [
                {"file_id": "small", "width": 20, "height": 20},
                {"file_id": "large", "width": 100, "height": 100},
            ]
        )

        self.assertEqual(photo["file_id"], "large")

    def test_normalize_image_prompt_uses_default_for_empty_caption(self):
        self.assertEqual(normalize_image_prompt(None), DEFAULT_IMAGE_PROMPT)

    def test_normalize_image_prompt_strips_ask_command(self):
        self.assertEqual(normalize_image_prompt("/ask что на фото?"), "что на фото?")

    def test_media_type_for_file_path_defaults_to_octet_stream(self):
        self.assertEqual(media_type_for_file_path("photos/file"), "application/octet-stream")
        self.assertEqual(media_type_for_file_path("photos/file.png"), "image/png")

    def test_image_media_type_for_file_path_defaults_to_jpeg(self):
        self.assertEqual(image_media_type_for_file_path("photos/file"), "image/jpeg")
        self.assertEqual(image_media_type_for_file_path("photos/file.png"), "image/png")
        self.assertEqual(image_media_type_for_file_path("photos/file", hint="image/webp"), "image/webp")

    def test_file_type_helpers(self):
        self.assertTrue(is_image_file(file_name="image.jpg", media_type="application/octet-stream"))
        self.assertTrue(is_text_file(file_name="notes.md", media_type="application/octet-stream"))
        self.assertFalse(is_text_file(file_name="archive.zip", media_type="application/zip"))

    def test_decode_text_document_uses_cp1251_fallback(self):
        text, encoding = decode_text_document("привет".encode("cp1251"))

        self.assertEqual(text, "привет")
        self.assertEqual(encoding, "cp1251")


class StateStoreTest(unittest.TestCase):
    def test_state_store_persists_history_notes_settings_and_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = str(Path(directory) / "state.sqlite3")
            store = StateStore(database_path, max_history_messages=2)
            store.append_history(chat_id=10, role="user", content="one")
            store.append_history(chat_id=10, role="assistant", content="two")
            store.append_history(chat_id=10, role="user", content="three")
            store.add_note(chat_id=10, content="likes concise answers")
            store.set_temperature(chat_id=10, temperature=0.2)
            store.set_model(chat_id=10, model="second-model")
            store.add_allowed_user(123)
            store.close()

            reopened = StateStore(database_path, max_history_messages=2)

            self.assertEqual(
                reopened.list_history(10, limit=2),
                [{"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}],
            )
            self.assertEqual(len(reopened.list_history(10)), 3)
            self.assertEqual(reopened.list_notes(10)[0]["content"], "likes concise answers")
            self.assertEqual(reopened.get_chat_settings(10)["temperature"], 0.2)
            self.assertEqual(reopened.get_chat_settings(10)["model"], "second-model")
            self.assertEqual(reopened.allowed_user_ids(), frozenset({123}))
            reopened.close()

    def test_summary_and_history_prune(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = str(Path(directory) / "state.sqlite3")
            store = StateStore(database_path, max_history_messages=2)
            store.append_history(chat_id=10, role="user", content="one")
            store.append_history(chat_id=10, role="assistant", content="two")
            store.append_history(chat_id=10, role="user", content="three")

            overflow = store.list_overflow_history(chat_id=10, keep_messages=2)
            self.assertEqual([message["content"] for message in overflow], ["one"])

            store.set_summary(chat_id=10, content="Пользователь начал с one.")
            store.delete_history_ids([message["id"] for message in overflow])

            self.assertEqual(store.get_summary(10), "Пользователь начал с one.")
            self.assertEqual(
                store.list_history(10),
                [{"role": "assistant", "content": "two"}, {"role": "user", "content": "three"}],
            )
            store.close()

    def test_format_summary_prompt_includes_existing_summary_and_new_messages(self):
        prompt = format_summary_prompt(
            existing_summary="Старый факт.",
            messages=[{"role": "user", "content": "Новый факт."}],
        )

        self.assertIn("Старый факт.", prompt)
        self.assertIn("Новый факт.", prompt)


class AccessControlTest(unittest.TestCase):
    def test_unauthorized_user_is_silently_ignored_even_for_start(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(str(Path(directory) / "state.sqlite3"), max_history_messages=16)
            telegram = FakeTelegram()
            app = BotApp(
                settings=make_settings(database_path=state.database_path),
                telegram=telegram,
                lm_studio=FakeLMStudio(),
                state=state,
            )

            app.handle_update(
                {
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 100},
                        "from": {"id": 999},
                        "text": "/start",
                    }
                }
            )

            self.assertEqual(telegram.sent_messages, [])
            self.assertEqual(state.history_count(100), 0)
            app.close()


class ModelSelectionTest(unittest.TestCase):
    def test_admin_switches_current_chat_model_by_number(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(str(Path(directory) / "state.sqlite3"), max_history_messages=16)
            telegram = RecordingTelegram()
            lm_studio = RecordingLMStudio(models=["first-model", "second-model"])
            app = BotApp(
                settings=make_settings(database_path=state.database_path),
                telegram=telegram,
                lm_studio=lm_studio,
                state=state,
            )

            app.handle_update(
                {
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 100},
                        "from": {"id": 92174505},
                        "text": "/model 2",
                    }
                }
            )
            app.handle_update(
                {
                    "message": {
                        "message_id": 2,
                        "chat": {"id": 100},
                        "from": {"id": 92174505},
                        "text": "hello",
                    }
                }
            )

            self.assertEqual(state.get_chat_settings(100)["model"], "second-model")
            self.assertEqual(lm_studio.chat_calls[-1]["model"], "second-model")
            self.assertIn("second-model", telegram.sent_messages[0]["text"])
            app.close()


class VoiceReplyRequestTest(unittest.TestCase):
    def test_explicit_text_request_gets_voice_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(str(Path(directory) / "state.sqlite3"), max_history_messages=16)
            telegram = VoiceTelegram(voice_data=b"unused")
            speech = RecordingSpeech(transcript="unused", voice_answer=b"assistant-ogg")
            app = BotApp(
                settings=make_settings(database_path=state.database_path),
                telegram=telegram,
                lm_studio=RecordingLMStudio(models=[]),
                state=state,
                speech=speech,
            )

            app.handle_update(
                {
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 100},
                        "from": {"id": 92174505},
                        "text": "Ответь голосом: какая сегодня погода?",
                    }
                }
            )

            self.assertEqual(speech.synthesized, ["ok"])
            self.assertEqual(telegram.sent_voices[0]["voice_data"], b"assistant-ogg")
            self.assertEqual(telegram.sent_messages, [])
            app.close()

    def test_voice_word_without_request_keeps_text_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(str(Path(directory) / "state.sqlite3"), max_history_messages=16)
            telegram = VoiceTelegram(voice_data=b"unused")
            speech = RecordingSpeech(transcript="unused", voice_answer=b"assistant-ogg")
            app = BotApp(
                settings=make_settings(database_path=state.database_path),
                telegram=telegram,
                lm_studio=RecordingLMStudio(models=[]),
                state=state,
                speech=speech,
            )

            app.handle_update(
                {
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 100},
                        "from": {"id": 92174505},
                        "text": "Что такое голосовой интерфейс?",
                    }
                }
            )

            self.assertEqual(speech.synthesized, [])
            self.assertEqual(telegram.sent_voices, [])
            self.assertEqual(telegram.sent_messages[0]["text"], "ok")
            app.close()

    def test_voice_request_helper_requires_voice_and_action_words(self):
        self.assertTrue(user_requested_voice_reply("пожалуйста, ответь голосом"))
        self.assertTrue(user_requested_voice_reply("можешь voice ответить?"))
        self.assertFalse(user_requested_voice_reply("что такое голосовой интерфейс?"))


class VoiceMessageTest(unittest.TestCase):
    def test_voice_message_is_transcribed_and_answered_with_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(str(Path(directory) / "state.sqlite3"), max_history_messages=16)
            telegram = VoiceTelegram(voice_data=b"telegram-ogg")
            lm_studio = RecordingLMStudio(models=[])
            speech = RecordingSpeech(transcript="привет голосом", voice_answer=b"assistant-ogg")
            app = BotApp(
                settings=make_settings(database_path=state.database_path),
                telegram=telegram,
                lm_studio=lm_studio,
                state=state,
                speech=speech,
            )

            app.handle_update(
                {
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 100},
                        "from": {"id": 92174505},
                        "voice": {"file_id": "voice-file", "duration": 3, "file_size": 12},
                    }
                }
            )

            self.assertEqual(speech.transcribed, [b"telegram-ogg"])
            self.assertEqual(speech.synthesized, ["ok"])
            self.assertEqual(lm_studio.chat_calls[-1]["messages"][-1]["content"], "привет голосом")
            self.assertEqual(telegram.sent_voices[0]["voice_data"], b"assistant-ogg")
            self.assertEqual(telegram.sent_messages, [])
            app.close()

    def test_long_voice_message_is_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(str(Path(directory) / "state.sqlite3"), max_history_messages=16)
            telegram = VoiceTelegram(voice_data=b"telegram-ogg")
            app = BotApp(
                settings=make_settings(database_path=state.database_path),
                telegram=telegram,
                lm_studio=RecordingLMStudio(models=[]),
                state=state,
                speech=RecordingSpeech(transcript="ignored", voice_answer=b"ignored"),
            )

            app.handle_update(
                {
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 100},
                        "from": {"id": 92174505},
                        "voice": {"file_id": "voice-file", "duration": 999, "file_size": 12},
                    }
                }
            )

            self.assertEqual(telegram.downloaded_paths, [])
            self.assertIn("слишком длинное", telegram.sent_messages[0]["text"])
            app.close()


class FakeTelegram:
    def __init__(self):
        self.sent_messages = []

    def send_message(self, *, chat_id, text, reply_to_message_id=None):
        self.sent_messages.append(
            {"chat_id": chat_id, "text": text, "reply_to_message_id": reply_to_message_id}
        )

    def send_chat_action(self, *, chat_id, action="typing"):
        raise AssertionError("Unauthorized update should not send chat actions")


class FakeLMStudio:
    def chat(self, messages, *, temperature=None, max_tokens=None):
        raise AssertionError("Unauthorized update should not reach LM Studio")

    def list_models(self):
        return []


class RecordingTelegram(FakeTelegram):
    def send_chat_action(self, *, chat_id, action="typing"):
        return None


class VoiceTelegram(RecordingTelegram):
    def __init__(self, *, voice_data):
        super().__init__()
        self.voice_data = voice_data
        self.downloaded_paths = []
        self.sent_voices = []

    def get_file_path(self, file_id):
        return f"voice/{file_id}.ogg"

    def download_file(self, file_path):
        self.downloaded_paths.append(file_path)
        return self.voice_data

    def send_voice(self, *, chat_id, voice_data, filename="answer.ogg", reply_to_message_id=None):
        self.sent_voices.append(
            {
                "chat_id": chat_id,
                "voice_data": voice_data,
                "filename": filename,
                "reply_to_message_id": reply_to_message_id,
            }
        )


class RecordingLMStudio:
    def __init__(self, *, models):
        self.models = models
        self.chat_calls = []

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None):
        self.chat_calls.append(
            {"messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens}
        )
        return "ok"

    def list_models(self):
        return self.models


class RecordingSpeech:
    def __init__(self, *, transcript, voice_answer):
        self.transcript = transcript
        self.voice_answer = voice_answer
        self.transcribed = []
        self.synthesized = []

    def transcribe_telegram_voice(self, voice_data):
        self.transcribed.append(voice_data)
        return self.transcript

    def synthesize_telegram_voice(self, text):
        self.synthesized.append(text)
        return self.voice_answer


def make_settings(*, database_path: str) -> Settings:
    return Settings(
        telegram_bot_token="token",
        admin_user_id=92174505,
        allowed_user_ids=frozenset({92174505}),
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_model="test-model",
        database_path=database_path,
        system_prompt="test prompt",
        max_history_messages=16,
        summary_enabled=True,
        summary_max_tokens=4096,
        max_image_bytes=20 * 1024 * 1024,
        max_text_document_bytes=1024 * 1024,
        lm_studio_temperature=0.7,
        lm_studio_max_tokens=4096,
        request_timeout_seconds=180.0,
        telegram_poll_timeout_seconds=50,
        voice_input_enabled=True,
        voice_reply_mode="voice-input",
        max_voice_bytes=20 * 1024 * 1024,
        max_voice_seconds=120,
        max_tts_chars=2000,
        speech_timeout_seconds=180.0,
        whisper_bin="/usr/local/bin/whisper-cli",
        whisper_model_path="/models/whisper/ggml-small.bin",
        whisper_language="ru",
        piper_bin="/usr/local/bin/piper",
        piper_model_path="/models/piper/ru_RU-dmitri-medium.onnx",
        piper_config_path="/models/piper/ru_RU-dmitri-medium.onnx.json",
        log_level="INFO",
    )


if __name__ == "__main__":
    unittest.main()
