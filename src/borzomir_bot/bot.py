from __future__ import annotations

import base64
import logging
import mimetypes
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import ConfigError, Settings, load_settings
from .lmstudio import LMStudioClient, LMStudioError
from .speech import LocalSpeechService, SpeechError
from .state import StateStore
from .telegram_api import TelegramClient, TelegramError


LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE_PROMPT = "Опиши изображение и ответь на вопрос пользователя, если он есть."
DEFAULT_DOCUMENT_PROMPT = "Проанализируй файл и ответь на вопрос пользователя, если он есть."
SUMMARY_SYSTEM_PROMPT = (
    "Ты модуль памяти Telegram-бота. Обновляй краткую, но полезную выжимку диалога. "
    "Сохраняй факты, предпочтения, решения, открытые задачи, важный технический контекст, "
    "упомянутые файлы и изображения. Убирай болтовню, повторы и устаревшие детали. "
    "Не добавляй фактов, которых нет в переписке. Ответь только новой выжимкой."
)
TEXT_DOCUMENT_CHAR_LIMIT = 80_000
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_TEXT_EXTENSIONS = {
    ".c",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".log",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
VOICE_REPLY_REQUEST_RE = re.compile(
    r"\b(ответь|отвечай|пришли|отправь|скажи|запиши|озвучь|произнеси|можешь)\b"
    r".{0,80}\b(голос\w*|войс\w*|voice)\b"
    r"|\b(голос\w*|войс\w*|voice)\b"
    r".{0,80}\b(ответь|отвечай|пришли|отправь|скажи|запиши|озвучь|произнеси)\b"
)
VOICE_REDELIVERY_REQUEST_RE = re.compile(
    r"^(озвучь|озвучь\s+это|скажи\s+голосом|голосом|войсом|voice)$"
)
VOICE_REPLY_COMMAND_CLEANUP_RE = re.compile(
    r"^\s*(пожалуйста[, ]*)?"
    r"(ответь|отвечай|пришли|отправь|скажи|запиши|озвучь|произнеси|можешь)"
    r"\s+(мне\s+)?(голос\w*|войс\w*|voice)(\s+сообщени\w*)?"
    r"[:,-]?\s*",
    re.IGNORECASE,
)
VOICE_REPLY_SUFFIX_CLEANUP_RE = re.compile(
    r"\s*(ответь|отвечай|пришли|отправь|скажи|запиши|озвучь|произнеси)?"
    r"\s*(голосом|войсом|voice)\s*$",
    re.IGNORECASE,
)


@dataclass
class ConversationStore:
    max_messages: int
    storage: StateStore | None = None
    _history: dict[int, list[dict[str, str]]] = field(default_factory=dict)

    def build_messages(self, *, chat_id: int, system_prompt: str, user_text: str) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history_for(chat_id))
        messages.append({"role": "user", "content": user_text})
        return messages

    def build_image_messages(
        self,
        *,
        chat_id: int,
        system_prompt: str,
        user_text: str,
        image_data: bytes,
        media_type: str,
    ) -> list[dict[str, Any]]:
        image_url = f"data:{media_type};base64,{base64.b64encode(image_data).decode('ascii')}"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history_for(chat_id))
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
        return messages

    def append(self, *, chat_id: int, role: str, content: str) -> None:
        if self.max_messages <= 0:
            return
        if self.storage is not None:
            self.storage.append_history(chat_id=chat_id, role=role, content=content)
            return
        history = self._history.setdefault(chat_id, [])
        history.append({"role": role, "content": content})
        del history[:-self.max_messages]

    def reset(self, chat_id: int) -> None:
        if self.storage is not None:
            self.storage.reset_history(chat_id)
            return
        self._history.pop(chat_id, None)

    def stats(self) -> dict[str, int]:
        if self.storage is not None:
            return self.storage.history_stats()
        return {
            "chats": len(self._history),
            "messages": sum(len(messages) for messages in self._history.values()),
        }

    def _history_for(self, chat_id: int) -> list[dict[str, str]]:
        if self.storage is not None:
            return self.storage.list_history(chat_id, limit=self.max_messages)
        return self._history.get(chat_id, [])


class BotApp:
    def __init__(
        self,
        *,
        settings: Settings,
        telegram: TelegramClient,
        lm_studio: LMStudioClient,
        state: StateStore,
        speech: Any | None = None,
    ) -> None:
        self.settings = settings
        self.telegram = telegram
        self.lm_studio = lm_studio
        self.state = state
        self.speech = speech
        self.conversations = ConversationStore(settings.max_history_messages, storage=state)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.state.close()

    def run(self) -> None:
        offset: int | None = None
        LOGGER.info("Bot started. Admin user id: %s", self.settings.admin_user_id)
        LOGGER.info("LM Studio endpoint: %s", self.settings.lm_studio_base_url)
        LOGGER.info("LM Studio model: %s", self.settings.lm_studio_model)
        LOGGER.info("State database: %s", self.settings.database_path)

        while not self._stop.is_set():
            try:
                updates = self.telegram.get_updates(
                    offset=offset,
                    timeout=self.settings.telegram_poll_timeout_seconds,
                )
            except TelegramError as exc:
                LOGGER.warning("Failed to fetch Telegram updates: %s", exc)
                time.sleep(5)
                continue

            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1
                try:
                    self.handle_update(update)
                except Exception:
                    LOGGER.exception("Failed to handle update: %s", update)

    def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat")
        sender = message.get("from")
        if not isinstance(chat, dict) or not isinstance(sender, dict):
            return

        chat_id = chat.get("id")
        user_id = sender.get("id")
        message_id = message.get("message_id")
        if not isinstance(chat_id, int) or not isinstance(user_id, int):
            return
        if not isinstance(message_id, int):
            message_id = None

        if not self.is_user_allowed(user_id):
            LOGGER.debug("Ignoring unauthorized update from user id %s", user_id)
            return

        text = _read_text(message.get("text"))
        caption = _read_text(message.get("caption"))
        photo = select_largest_photo(message.get("photo"))
        document = message.get("document") if isinstance(message.get("document"), dict) else None
        voice = message.get("voice") if isinstance(message.get("voice"), dict) else None

        if text and text.startswith("/"):
            self.handle_command(chat_id=chat_id, user_id=user_id, message_id=message_id, text=text)
            return

        if voice is not None:
            self.ask_model_about_voice(chat_id=chat_id, user_id=user_id, message_id=message_id, voice=voice)
            return

        if photo is not None:
            self.ask_model_about_photo(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                photo=photo,
                prompt=normalize_attachment_prompt(caption, DEFAULT_IMAGE_PROMPT),
            )
            return

        if document is not None:
            self.ask_model_about_document(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                document=document,
                prompt=normalize_attachment_prompt(caption, DEFAULT_DOCUMENT_PROMPT),
            )
            return

        if text:
            if user_requested_last_answer_voice(text):
                self.reply_last_answer_as_voice(chat_id=chat_id, message_id=message_id)
                return
            reply_as_voice = self.should_reply_with_voice(
                source_is_voice=False,
                user_requested_voice=user_requested_voice_reply(text),
            )
            self.ask_model(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                prompt=strip_voice_reply_request(text) if reply_as_voice else text,
                reply_as_voice=reply_as_voice,
            )

    def handle_command(self, *, chat_id: int, user_id: int, message_id: int | None, text: str) -> None:
        command, _, argument = text.partition(" ")
        command = command.split("@", 1)[0].lower()
        argument = argument.strip()

        if command == "/start":
            self.reply(chat_id, WELCOME_TEXT, message_id)
        elif command == "/help":
            self.reply(chat_id, HELP_TEXT, message_id)
        elif command == "/admin":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.reply(reply_chat_id, ADMIN_HELP_TEXT, reply_message_id),
            )
        elif command == "/ask":
            if not argument:
                self.reply(chat_id, "Напишите вопрос после команды: /ask ваш вопрос", message_id)
                return
            reply_as_voice = self.should_reply_with_voice(
                source_is_voice=False,
                user_requested_voice=user_requested_voice_reply(argument),
            )
            self.ask_model(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                prompt=strip_voice_reply_request(argument) if reply_as_voice else argument,
                reply_as_voice=reply_as_voice,
            )
        elif command == "/reset":
            self.conversations.reset(chat_id)
            self.reply(chat_id, "Короткая история и выжимка этого чата очищены. Постоянные заметки остались.", message_id)
        elif command == "/remember":
            self.remember(chat_id=chat_id, user_id=user_id, message_id=message_id, argument=argument)
        elif command == "/memory":
            self.show_memory(chat_id=chat_id, user_id=user_id, message_id=message_id)
        elif command == "/summary":
            self.show_summary(chat_id=chat_id, user_id=user_id, message_id=message_id)
        elif command == "/resetsummary":
            self.reset_summary(chat_id=chat_id, user_id=user_id, message_id=message_id)
        elif command == "/forget":
            self.forget(chat_id=chat_id, user_id=user_id, message_id=message_id)
        elif command == "/models":
            self.show_models(chat_id=chat_id, message_id=message_id)
        elif command == "/model":
            if not argument:
                self.show_model(chat_id=chat_id, message_id=message_id)
                return
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.set_model(reply_chat_id, reply_message_id, argument),
            )
        elif command == "/params":
            self.show_params(chat_id, message_id)
        elif command == "/health":
            self.require_admin(chat_id=chat_id, user_id=user_id, message_id=message_id, action=self.health)
        elif command == "/stats":
            self.require_admin(chat_id=chat_id, user_id=user_id, message_id=message_id, action=self.stats)
        elif command == "/prompt":
            self.require_admin(chat_id=chat_id, user_id=user_id, message_id=message_id, action=self.show_prompt)
        elif command == "/setprompt":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.set_prompt(reply_chat_id, reply_message_id, argument),
            )
        elif command == "/resetprompt":
            self.require_admin(chat_id=chat_id, user_id=user_id, message_id=message_id, action=self.reset_prompt)
        elif command == "/temperature":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.set_temperature(
                    reply_chat_id, reply_message_id, argument
                ),
            )
        elif command == "/maxtokens":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.set_max_tokens(
                    reply_chat_id, reply_message_id, argument
                ),
            )
        elif command == "/resetmodel":
            self.require_admin(chat_id=chat_id, user_id=user_id, message_id=message_id, action=self.reset_model)
        elif command == "/allowed":
            self.require_admin(chat_id=chat_id, user_id=user_id, message_id=message_id, action=self.allowed)
        elif command == "/allow":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.allow_user(reply_chat_id, reply_message_id, argument),
            )
        elif command == "/deny":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.deny_user(reply_chat_id, reply_message_id, argument),
            )
        elif command == "/access":
            self.require_admin(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                action=lambda reply_chat_id, reply_message_id: self.access(reply_chat_id, reply_message_id, argument),
            )
        else:
            self.reply(chat_id, "Не знаю такую команду. Попробуйте /help.", message_id)

    def ask_model(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        prompt: str,
        reply_as_voice: bool = False,
    ) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return

        self.send_typing(chat_id)
        messages = self.conversations.build_messages(
            chat_id=chat_id,
            system_prompt=self.system_prompt_for_chat(chat_id),
            user_text=prompt,
        )
        answer = self.chat_or_reply_error(chat_id=chat_id, message_id=message_id, messages=messages)
        if answer is None:
            return

        self.record_exchange(chat_id=chat_id, user_content=prompt, assistant_content=answer)
        if reply_as_voice:
            self.reply_voice_or_text(chat_id=chat_id, text=answer, message_id=message_id)
        else:
            self.reply(chat_id, answer, message_id)
        self.maybe_summarize_chat(chat_id)

    def ask_model_about_voice(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        voice: dict[str, Any],
    ) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return
        if not self.settings.voice_input_enabled:
            self.reply(chat_id, "Голосовые сообщения сейчас выключены в настройках бота.", message_id)
            return
        if self.speech is None:
            self.reply(chat_id, "Локальный speech-сервис не настроен. Проверьте Whisper/Piper в .env.", message_id)
            return

        file_id = voice.get("file_id")
        if not isinstance(file_id, str):
            self.reply(chat_id, "Не смог прочитать file_id голосового сообщения.", message_id)
            return

        duration = voice.get("duration")
        if isinstance(duration, int) and duration > self.settings.max_voice_seconds:
            self.reply(
                chat_id,
                f"Голосовое сообщение слишком длинное: {duration} сек. "
                f"Лимит сейчас {self.settings.max_voice_seconds} сек.",
                message_id,
            )
            return

        file_size = voice.get("file_size")
        if isinstance(file_size, int) and file_size > self.settings.max_voice_bytes:
            self.reply(
                chat_id,
                f"Голосовое сообщение слишком большое: {file_size} байт. "
                f"Лимит сейчас {self.settings.max_voice_bytes} байт.",
                message_id,
            )
            return

        self.send_typing(chat_id)
        try:
            file_path = self.telegram.get_file_path(file_id)
            voice_data = self.telegram.download_file(file_path)
        except TelegramError as exc:
            LOGGER.warning("Failed to download Telegram voice: %s", exc)
            self.reply(chat_id, f"Не смог скачать голосовое сообщение из Telegram.\n\nДетали: {exc}", message_id)
            return

        if len(voice_data) > self.settings.max_voice_bytes:
            self.reply(chat_id, "Голосовое сообщение оказалось больше лимита после скачивания.", message_id)
            return

        try:
            transcript = self.speech.transcribe_telegram_voice(voice_data)
        except SpeechError as exc:
            LOGGER.warning("Local speech recognition failed: %s", exc)
            self.reply(chat_id, f"Не смог распознать голосовое сообщение локально.\n\nДетали: {exc}", message_id)
            return

        self.ask_model(
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            prompt=transcript,
            reply_as_voice=self.should_reply_with_voice(
                source_is_voice=True,
                user_requested_voice=user_requested_voice_reply(transcript),
            ),
        )

    def ask_model_about_photo(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        photo: dict[str, Any],
        prompt: str,
    ) -> None:
        file_id = photo.get("file_id")
        if not isinstance(file_id, str):
            self.reply(chat_id, "Не смог прочитать file_id изображения.", message_id)
            return
        self.ask_model_about_file_image(
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            file_id=file_id,
            prompt=prompt,
            source_name="photo.jpg",
        )

    def ask_model_about_document(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        document: dict[str, Any],
        prompt: str,
    ) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return

        file_id = document.get("file_id")
        if not isinstance(file_id, str):
            self.reply(chat_id, "Не смог прочитать file_id документа.", message_id)
            return

        file_name = str(document.get("file_name") or "document")
        media_type = str(document.get("mime_type") or media_type_for_file_path(file_name))
        file_size = document.get("file_size")

        if is_image_file(file_name=file_name, media_type=media_type):
            self.ask_model_about_file_image(
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                file_id=file_id,
                prompt=prompt,
                source_name=file_name,
                media_type_hint=media_type,
            )
            return

        if not is_text_file(file_name=file_name, media_type=media_type):
            self.reply(
                chat_id,
                "Пока умею читать фото и текстовые документы: txt, md, csv, json, yaml, код, логи.",
                message_id,
            )
            return

        if isinstance(file_size, int) and file_size > self.settings.max_text_document_bytes:
            self.reply(
                chat_id,
                f"Текстовый файл слишком большой: {file_size} байт. "
                f"Лимит сейчас {self.settings.max_text_document_bytes} байт.",
                message_id,
            )
            return

        self.send_typing(chat_id)
        try:
            file_path = self.telegram.get_file_path(file_id)
            document_data = self.telegram.download_file(file_path)
        except TelegramError as exc:
            LOGGER.warning("Failed to download Telegram document: %s", exc)
            self.reply(chat_id, f"Не смог скачать документ из Telegram.\n\nДетали: {exc}", message_id)
            return

        if len(document_data) > self.settings.max_text_document_bytes:
            self.reply(chat_id, "Документ оказался больше лимита после скачивания.", message_id)
            return

        text, encoding = decode_text_document(document_data)
        truncated = len(text) > TEXT_DOCUMENT_CHAR_LIMIT
        if truncated:
            text = text[:TEXT_DOCUMENT_CHAR_LIMIT]
        model_prompt = format_document_prompt(
            prompt=prompt,
            file_name=file_name,
            media_type=media_type,
            encoding=encoding,
            content=text,
            truncated=truncated,
        )
        messages = self.conversations.build_messages(
            chat_id=chat_id,
            system_prompt=self.system_prompt_for_chat(chat_id),
            user_text=model_prompt,
        )
        answer = self.chat_or_reply_error(chat_id=chat_id, message_id=message_id, messages=messages)
        if answer is None:
            return

        self.record_exchange(chat_id=chat_id, user_content=f"[document:{file_name}] {prompt}", assistant_content=answer)
        self.reply(chat_id, answer, message_id)
        self.maybe_summarize_chat(chat_id)

    def ask_model_about_file_image(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        file_id: str,
        prompt: str,
        source_name: str,
        media_type_hint: str | None = None,
    ) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return

        self.send_typing(chat_id)
        try:
            file_path = self.telegram.get_file_path(file_id)
            image_data = self.telegram.download_file(file_path)
        except TelegramError as exc:
            LOGGER.warning("Failed to download Telegram image: %s", exc)
            self.reply(chat_id, f"Не смог скачать изображение из Telegram.\n\nДетали: {exc}", message_id)
            return

        if len(image_data) > self.settings.max_image_bytes:
            self.reply(chat_id, "Изображение слишком большое для обработки.", message_id)
            return

        media_type = image_media_type_for_file_path(file_path, hint=media_type_hint)
        messages = self.conversations.build_image_messages(
            chat_id=chat_id,
            system_prompt=self.system_prompt_for_chat(chat_id),
            user_text=prompt,
            image_data=image_data,
            media_type=media_type,
        )
        answer = self.chat_or_reply_error(
            chat_id=chat_id,
            message_id=message_id,
            messages=messages,
            prefix="LM Studio не ответил на запрос с изображением. Проверьте, что загруженная модель поддерживает Vision.",
        )
        if answer is None:
            return

        self.record_exchange(chat_id=chat_id, user_content=f"[image:{source_name}] {prompt}", assistant_content=answer)
        self.reply(chat_id, answer, message_id)
        self.maybe_summarize_chat(chat_id)

    def chat_or_reply_error(
        self,
        *,
        chat_id: int,
        message_id: int | None,
        messages: list[dict[str, Any]],
        prefix: str = "LM Studio не ответил. Проверьте, что Local Server запущен и модель загружена.",
    ) -> str | None:
        temperature, max_tokens = self.params_for_chat(chat_id)
        model = self.model_for_chat(chat_id)
        try:
            return self.lm_studio.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        except LMStudioError as exc:
            LOGGER.warning("LM Studio request failed: %s", exc)
            self.reply(chat_id, f"{prefix}\n\nДетали: {exc}", message_id)
            return None

    def remember(self, *, chat_id: int, user_id: int, message_id: int | None, argument: str) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return
        if not argument:
            self.reply(chat_id, "Напишите, что запомнить: /remember факт или предпочтение", message_id)
            return
        self.state.add_note(chat_id=chat_id, content=argument)
        self.reply(chat_id, "Запомнил для этого чата.", message_id)

    def show_memory(self, *, chat_id: int, user_id: int, message_id: int | None) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return
        notes = self.state.list_notes(chat_id)
        history = self.conversations.stats()
        if not notes:
            note_text = "Постоянных заметок пока нет."
        else:
            note_text = "\n".join(f"{index}. {note['content']}" for index, note in enumerate(notes, start=1))
        summary = self.state.get_summary(chat_id)
        summary_text = summary if summary else "Выжимки прошлой переписки пока нет."
        self.reply(
            chat_id,
            f"Память этого чата:\n{note_text}\n\n"
            f"Выжимка прошлой переписки:\n{summary_text}\n\n"
            f"Короткая история всего: {history['messages']} сообщений.\n"
            f"Короткая история этого чата: {self.state.history_count(chat_id)} сообщений.",
            message_id,
        )

    def show_summary(self, *, chat_id: int, user_id: int, message_id: int | None) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return
        summary = self.state.get_summary(chat_id)
        self.reply(chat_id, summary or "Выжимки прошлой переписки пока нет.", message_id)

    def reset_summary(self, *, chat_id: int, user_id: int, message_id: int | None) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return
        self.state.set_summary(chat_id=chat_id, content=None)
        self.reply(chat_id, "Выжимка прошлой переписки очищена.", message_id)

    def forget(self, *, chat_id: int, user_id: int, message_id: int | None) -> None:
        if not self.is_user_allowed(user_id):
            self.deny_access(chat_id=chat_id, user_id=user_id, message_id=message_id)
            return
        removed = self.state.clear_notes(chat_id)
        self.reply(chat_id, f"Удалил постоянные заметки этого чата: {removed}.", message_id)

    def show_model(self, *, chat_id: int, message_id: int | None) -> None:
        chat_settings = self.state.get_chat_settings(chat_id)
        model = self.model_for_chat(chat_id)
        source = "выбрана для этого чата" if chat_settings.get("model") else "из .env"
        self.reply(
            chat_id,
            f"Модель: {model}\nИсточник: {source}\nLM Studio: {self.settings.lm_studio_base_url}",
            message_id,
        )

    def show_models(self, *, chat_id: int, message_id: int | None) -> None:
        try:
            models = self.lm_studio.list_models()
        except LMStudioError as exc:
            self.reply(chat_id, f"Не смог получить список моделей из LM Studio:\n{exc}", message_id)
            return

        active_model = self.model_for_chat(chat_id)
        self.reply(
            chat_id,
            "Доступные модели LM Studio:\n"
            f"{format_model_list(models, active_model=active_model)}\n\n"
            "Админ может переключить этот чат: /model <номер или id>",
            message_id,
        )

    def set_model(self, chat_id: int, message_id: int | None, argument: str) -> None:
        try:
            models = self.lm_studio.list_models()
        except LMStudioError as exc:
            self.reply(chat_id, f"Не смог получить список моделей из LM Studio:\n{exc}", message_id)
            return

        model = resolve_model_argument(argument, models)
        if model is None:
            self.reply(
                chat_id,
                "Не нашёл такую модель. Используйте номер или точный id из списка:\n"
                f"{format_model_list(models, active_model=self.model_for_chat(chat_id))}",
                message_id,
            )
            return

        self.state.set_model(chat_id=chat_id, model=model)
        self.reply(chat_id, f"Модель этого чата переключена:\n{model}", message_id)

    def reset_model(self, chat_id: int, message_id: int | None) -> None:
        self.state.set_model(chat_id=chat_id, model=None)
        self.reply(chat_id, f"Модель этого чата сброшена к .env:\n{self.settings.lm_studio_model}", message_id)

    def require_admin(
        self,
        *,
        chat_id: int,
        user_id: int,
        message_id: int | None,
        action: Callable[[int, int | None], None],
    ) -> None:
        if not self.settings.is_admin(user_id):
            self.reply(chat_id, "Эта команда доступна только администратору.", message_id)
            return
        action(chat_id, message_id)

    def health(self, chat_id: int, message_id: int | None) -> None:
        try:
            models = self.lm_studio.list_models()
        except LMStudioError as exc:
            self.reply(chat_id, f"LM Studio недоступен:\n{exc}", message_id)
            return

        active_model = self.model_for_chat(chat_id)
        loaded = format_model_list(models, active_model=active_model)
        marker = "да" if active_model in models else "нет"
        self.reply(
            chat_id,
            f"LM Studio отвечает.\nАктивная модель: {active_model}\nАктивная модель найдена: {marker}\n\nМодели:\n{loaded}",
            message_id,
        )

    def stats(self, chat_id: int, message_id: int | None) -> None:
        stats = self.state.stats()
        env_allowed = len(self.settings.allowed_user_ids)
        runtime_allowed = stats["allowed_users"]
        mode = "открыт" if env_allowed == 0 and runtime_allowed == 0 else "ограничен"
        self.reply(
            chat_id,
            "Статистика:\n"
            f"- чатов в короткой памяти: {stats['chats']}\n"
            f"- сообщений в короткой памяти: {stats['messages']}\n"
            f"- постоянных заметок: {stats['notes']}\n"
            f"- чатов с выжимкой: {stats['summaries']}\n"
            f"- символов в выжимках: {stats['summary_chars']}\n"
            f"- доступ: {mode}\n"
            f"- env allowlist: {env_allowed}\n"
            f"- runtime allowlist: {runtime_allowed}\n"
            f"- админ: {self.settings.admin_user_id}",
            message_id,
        )

    def show_prompt(self, chat_id: int, message_id: int | None) -> None:
        self.reply(chat_id, f"Активный system prompt:\n\n{self.system_prompt_for_chat(chat_id)}", message_id)

    def set_prompt(self, chat_id: int, message_id: int | None, argument: str) -> None:
        if not argument:
            self.reply(chat_id, "Напишите новый prompt: /setprompt текст", message_id)
            return
        self.state.set_system_prompt(chat_id=chat_id, system_prompt=argument)
        self.reply(chat_id, "System prompt для этого чата обновлён.", message_id)

    def reset_prompt(self, chat_id: int, message_id: int | None) -> None:
        self.state.set_system_prompt(chat_id=chat_id, system_prompt=None)
        self.reply(chat_id, "System prompt этого чата сброшен к значению из .env.", message_id)

    def show_params(self, chat_id: int, message_id: int | None) -> None:
        temperature, max_tokens = self.params_for_chat(chat_id)
        self.reply(
            chat_id,
            "Параметры этого чата:\n"
            f"- temperature: {temperature}\n"
            f"- max_tokens: {max_tokens}\n"
            f"- history messages: {self.settings.max_history_messages}\n"
            f"- voice input: {format_bool(self.settings.voice_input_enabled)}\n"
            f"- voice reply mode: {self.settings.voice_reply_mode}",
            message_id,
        )

    def set_temperature(self, chat_id: int, message_id: int | None, argument: str) -> None:
        try:
            value = float(argument)
        except ValueError:
            self.reply(chat_id, "Temperature должен быть числом, например: /temperature 0.8", message_id)
            return
        if not 0 <= value <= 2:
            self.reply(chat_id, "Temperature должен быть в диапазоне 0..2.", message_id)
            return
        self.state.set_temperature(chat_id=chat_id, temperature=value)
        self.reply(chat_id, f"Temperature для этого чата: {value}.", message_id)

    def set_max_tokens(self, chat_id: int, message_id: int | None, argument: str) -> None:
        value = parse_positive_int(argument)
        if value is None:
            self.reply(chat_id, "Max tokens должен быть числом, например: /maxtokens 4096", message_id)
            return
        if value > 32768:
            self.reply(chat_id, "Слишком много. Поставьте значение до 32768.", message_id)
            return
        self.state.set_max_tokens(chat_id=chat_id, max_tokens=value)
        self.reply(chat_id, f"Max tokens для этого чата: {value}.", message_id)

    def allowed(self, chat_id: int, message_id: int | None) -> None:
        env_ids = sorted(self.settings.allowed_user_ids)
        runtime_ids = sorted(self.state.allowed_user_ids())
        mode = "открыт для всех" if not env_ids and not runtime_ids else "ограничен allowlist"
        self.reply(
            chat_id,
            f"Режим доступа: {mode}\n"
            f"Env allowlist: {format_int_list(env_ids)}\n"
            f"Runtime allowlist: {format_int_list(runtime_ids)}\n"
            f"Админ всегда разрешён: {self.settings.admin_user_id}",
            message_id,
        )

    def allow_user(self, chat_id: int, message_id: int | None, argument: str) -> None:
        user_id = parse_positive_int(argument)
        if user_id is None:
            self.reply(chat_id, "Напишите user id: /allow 123456", message_id)
            return
        self.state.add_allowed_user(user_id)
        self.reply(chat_id, f"Добавил user id {user_id} в runtime allowlist.", message_id)

    def deny_user(self, chat_id: int, message_id: int | None, argument: str) -> None:
        user_id = parse_positive_int(argument)
        if user_id is None:
            self.reply(chat_id, "Напишите user id: /deny 123456", message_id)
            return
        removed = self.state.remove_allowed_user(user_id)
        status = "удалён" if removed else "не был в runtime allowlist"
        self.reply(chat_id, f"User id {user_id}: {status}.", message_id)

    def access(self, chat_id: int, message_id: int | None, argument: str) -> None:
        if argument == "open":
            removed = self.state.clear_allowed_users()
            self.reply(
                chat_id,
                f"Runtime allowlist очищен: {removed}. "
                "Если ALLOWED_USER_IDS в .env пустой, бот снова открыт для всех.",
                message_id,
            )
            return
        self.allowed(chat_id, message_id)

    def system_prompt_for_chat(self, chat_id: int) -> str:
        chat_settings = self.state.get_chat_settings(chat_id)
        prompt = chat_settings.get("system_prompt") or self.settings.system_prompt
        parts = [prompt]
        summary = self.state.get_summary(chat_id)
        if summary:
            parts.append(
                "Сжатая история предыдущей части диалога. Используй её как контекст, "
                f"но не цитируй без необходимости:\n{summary}"
            )
        notes = self.state.list_notes(chat_id)
        if notes:
            note_lines = "\n".join(f"- {note['content']}" for note in notes[-50:])
            parts.append(f"Постоянная память этого чата:\n{note_lines}")
        return "\n\n".join(parts)

    def params_for_chat(self, chat_id: int) -> tuple[float, int]:
        chat_settings = self.state.get_chat_settings(chat_id)
        temperature = chat_settings.get("temperature")
        max_tokens = chat_settings.get("max_tokens")
        return (
            float(temperature) if temperature is not None else self.settings.lm_studio_temperature,
            int(max_tokens) if max_tokens is not None else self.settings.lm_studio_max_tokens,
        )

    def model_for_chat(self, chat_id: int) -> str:
        chat_settings = self.state.get_chat_settings(chat_id)
        model = chat_settings.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        return self.settings.lm_studio_model

    def is_user_allowed(self, user_id: int) -> bool:
        if self.settings.is_admin(user_id):
            return True
        allowed_ids = self.settings.allowed_user_ids | self.state.allowed_user_ids()
        return not allowed_ids or user_id in allowed_ids

    def should_reply_with_voice(self, *, source_is_voice: bool, user_requested_voice: bool = False) -> bool:
        if self.speech is None:
            return False
        if self.settings.voice_reply_mode == "off":
            return False
        if user_requested_voice:
            return True
        if self.settings.voice_reply_mode == "always":
            return True
        return self.settings.voice_reply_mode == "voice-input" and source_is_voice

    def deny_access(self, *, chat_id: int, user_id: int, message_id: int | None) -> None:
        LOGGER.debug("Ignoring unauthorized action from user id %s in chat %s", user_id, chat_id)

    def record_exchange(self, *, chat_id: int, user_content: str, assistant_content: str) -> None:
        self.conversations.append(chat_id=chat_id, role="user", content=user_content)
        self.conversations.append(chat_id=chat_id, role="assistant", content=assistant_content)

    def last_assistant_message(self, chat_id: int) -> str | None:
        for message in reversed(self.conversations._history_for(chat_id)):
            if message.get("role") == "assistant":
                content = message.get("content", "").strip()
                if content:
                    return content
        return None

    def maybe_summarize_chat(self, chat_id: int) -> None:
        if not self.settings.summary_enabled or self.settings.max_history_messages <= 0:
            return
        overflow = self.state.list_overflow_history(chat_id=chat_id, keep_messages=self.settings.max_history_messages)
        if not overflow:
            return
        existing_summary = self.state.get_summary(chat_id)
        prompt = format_summary_prompt(existing_summary=existing_summary, messages=overflow)
        try:
            updated_summary = self.lm_studio.chat(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model_for_chat(chat_id),
                temperature=0.2,
                max_tokens=self.settings.summary_max_tokens,
            )
        except LMStudioError as exc:
            LOGGER.warning("Failed to summarize chat %s: %s", chat_id, exc)
            return
        self.state.set_summary(chat_id=chat_id, content=updated_summary)
        removed = self.state.delete_history_ids([int(message["id"]) for message in overflow])
        LOGGER.info("Summarized chat %s and removed %s old history messages", chat_id, removed)

    def send_typing(self, chat_id: int) -> None:
        self.send_action(chat_id, "typing")

    def send_action(self, chat_id: int, action: str) -> None:
        try:
            self.telegram.send_chat_action(chat_id=chat_id, action=action)
        except TelegramError:
            LOGGER.warning("Failed to send Telegram chat action %s", action, exc_info=True)

    def reply(self, chat_id: int, text: str, message_id: int | None = None) -> None:
        try:
            self.telegram.send_message(chat_id=chat_id, text=text, reply_to_message_id=message_id)
        except TelegramError:
            LOGGER.exception("Failed to send Telegram message")

    def reply_last_answer_as_voice(self, *, chat_id: int, message_id: int | None = None) -> None:
        if self.settings.voice_reply_mode == "off":
            self.reply(chat_id, "Голосовые ответы сейчас выключены в настройках бота.", message_id)
            return
        answer = self.last_assistant_message(chat_id)
        if answer is None:
            self.reply(chat_id, "Пока нечего озвучивать: я ещё не отвечал в этом чате.", message_id)
            return
        self.reply_voice_or_text(chat_id=chat_id, text=answer, message_id=message_id)

    def reply_voice_or_text(self, *, chat_id: int, text: str, message_id: int | None = None) -> None:
        if self.speech is None or self.settings.voice_reply_mode == "off":
            self.reply(chat_id, text, message_id)
            return
        if len(text) > self.settings.max_tts_chars:
            self.reply(chat_id, text, message_id)
            return

        self.send_action(chat_id, "upload_voice")
        try:
            voice_data = self.speech.synthesize_telegram_voice(text)
            self.telegram.send_voice(chat_id=chat_id, voice_data=voice_data, reply_to_message_id=message_id)
        except SpeechError as exc:
            LOGGER.warning("Local speech synthesis failed: %s", exc)
            self.reply(chat_id, text, message_id)
        except TelegramError as exc:
            if is_voice_messages_forbidden(exc):
                LOGGER.warning("Telegram forbids voice messages, falling back to MP3 audio upload")
                try:
                    audio_data = self.speech.synthesize_telegram_audio(text)
                    self.send_action(chat_id, "upload_document")
                    self.telegram.send_audio(
                        chat_id=chat_id,
                        audio_data=audio_data,
                        reply_to_message_id=message_id,
                        title="Ответ голосом",
                    )
                    return
                except SpeechError as audio_exc:
                    LOGGER.warning("Local MP3 synthesis failed: %s", audio_exc)
                except TelegramError:
                    LOGGER.exception("Failed to send Telegram audio fallback")
                    try:
                        self.send_action(chat_id, "upload_document")
                        self.telegram.send_document(
                            chat_id=chat_id,
                            document_data=audio_data,
                            filename="answer.mp3",
                            media_type="audio/mpeg",
                            reply_to_message_id=message_id,
                            caption="Ответ голосом",
                        )
                        return
                    except TelegramError:
                        LOGGER.exception("Failed to send Telegram document fallback")
            else:
                LOGGER.exception("Failed to send Telegram voice")
            self.reply(chat_id, text, message_id)


WELCOME_TEXT = (
    "Привет. Я Telegram-бот, который общается с локальной моделью через LM Studio.\n\n"
    "Пиши текст, отправляй голосовые сообщения, фото, картинку-файл или текстовый документ "
    "с подписью-вопросом. Команды: /help"
)

HELP_TEXT = (
    "Команды:\n"
    "/ask <текст> - отправить вопрос модели\n"
    "/remember <факт> - сохранить постоянную заметку для этого чата\n"
    "/memory - показать постоянную память\n"
    "/summary - показать выжимку прошлой переписки\n"
    "/resetsummary - очистить выжимку прошлой переписки\n"
    "/forget - очистить постоянные заметки этого чата\n"
    "/reset - очистить короткую историю и выжимку\n"
    "/params - показать параметры генерации\n"
    "/model - показать текущую модель\n"
    "/models - показать доступные модели LM Studio\n"
    "\n"
    "Файлы:\n"
    "Голосовое сообщение - распознать локально и спросить модель; ответ вернётся голосом, если включён voice-режим\n"
    "Текст с просьбой 'ответь голосом' - получить голосовой ответ даже на текстовый вопрос\n"
    "Фото или картинка-документ с подписью - спросить модель об изображении\n"
    "Текстовый документ с подписью - спросить модель о файле\n"
    "\n"
    "Админ: /admin"
)

ADMIN_HELP_TEXT = (
    "Админ-команды:\n"
    "/health - проверить LM Studio\n"
    "/stats - статистика\n"
    "/prompt - показать активный system prompt\n"
    "/setprompt <текст> - задать prompt для текущего чата\n"
    "/resetprompt - сбросить prompt текущего чата\n"
    "/temperature <0..2> - задать temperature текущего чата\n"
    "/maxtokens <число> - задать max_tokens текущего чата\n"
    "/model <номер или id> - выбрать модель для текущего чата\n"
    "/resetmodel - сбросить модель текущего чата к .env\n"
    "/allowed - показать доступ\n"
    "/allow <user_id> - добавить пользователя в runtime allowlist\n"
    "/deny <user_id> - убрать пользователя из runtime allowlist\n"
    "/access open - очистить runtime allowlist"
)


def _read_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def select_largest_photo(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    photos = [item for item in value if isinstance(item, dict) and isinstance(item.get("file_id"), str)]
    if not photos:
        return None
    return max(photos, key=_photo_score)


def normalize_attachment_prompt(caption: str | None, default: str) -> str:
    if not caption:
        return default
    command, _, argument = caption.partition(" ")
    if command.split("@", 1)[0].lower() == "/ask" and argument.strip():
        return argument.strip()
    return caption


def normalize_image_prompt(caption: str | None) -> str:
    return normalize_attachment_prompt(caption, DEFAULT_IMAGE_PROMPT)


def media_type_for_file_path(file_path: str) -> str:
    media_type, _encoding = mimetypes.guess_type(file_path)
    if media_type:
        return media_type
    return "application/octet-stream"


def image_media_type_for_file_path(file_path: str, *, hint: str | None = None) -> str:
    if is_image_media_type(hint):
        return str(hint)
    media_type = media_type_for_file_path(file_path)
    if is_image_media_type(media_type):
        return media_type
    return "image/jpeg"


def is_image_media_type(media_type: str | None) -> bool:
    return isinstance(media_type, str) and media_type.startswith("image/")


def is_image_file(*, file_name: str, media_type: str) -> bool:
    return is_image_media_type(media_type) or file_extension(file_name) in SUPPORTED_IMAGE_EXTENSIONS


def is_text_file(*, file_name: str, media_type: str) -> bool:
    return (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/xml", "application/x-yaml"}
        or file_extension(file_name) in SUPPORTED_TEXT_EXTENSIONS
    )


def file_extension(file_name: str) -> str:
    dot = file_name.rfind(".")
    if dot == -1:
        return ""
    return file_name[dot:].lower()


def decode_text_document(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def format_document_prompt(
    *,
    prompt: str,
    file_name: str,
    media_type: str,
    encoding: str,
    content: str,
    truncated: bool,
) -> str:
    truncation_note = "\n\nФайл был усечён перед отправкой модели." if truncated else ""
    return (
        f"{prompt}\n\n"
        f"Файл: {file_name}\n"
        f"Тип: {media_type}\n"
        f"Кодировка: {encoding}{truncation_note}\n\n"
        "Содержимое файла:\n"
        "```text\n"
        f"{content}\n"
        "```"
    )


def format_summary_prompt(*, existing_summary: str | None, messages: list[dict[str, Any]]) -> str:
    current = existing_summary or "Выжимки пока нет."
    transcript = format_transcript(messages)
    return (
        "Обнови выжимку переписки для продолжения диалога.\n\n"
        f"Текущая выжимка:\n{current}\n\n"
        "Новая часть переписки, которую нужно встроить в выжимку:\n"
        f"{transcript}\n\n"
        "Верни цельную обновлённую выжимку на русском языке. "
        "Держи её компактной, но не теряй важные факты и незавершённые задачи."
    )


def format_transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = "Пользователь" if message.get("role") == "user" else "Ассистент"
        content = str(message.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def parse_positive_int(value: str) -> int | None:
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def resolve_model_argument(argument: str, models: list[str]) -> str | None:
    value = argument.strip()
    if not value:
        return None

    index = parse_positive_int(value)
    if index is not None and 1 <= index <= len(models):
        return models[index - 1]

    for model in models:
        if model == value:
            return model

    lowered = value.lower()
    case_matches = [model for model in models if model.lower() == lowered]
    if len(case_matches) == 1:
        return case_matches[0]

    return None


def format_model_list(models: list[str], *, active_model: str) -> str:
    if not models:
        return "нет моделей в ответе /models"
    lines = []
    for index, model in enumerate(models, start=1):
        marker = " [текущая]" if model == active_model else ""
        lines.append(f"{index}. {model}{marker}")
    return "\n".join(lines)


def format_int_list(values: list[int]) -> str:
    return ", ".join(str(value) for value in values) if values else "пусто"


def format_bool(value: bool) -> str:
    return "да" if value else "нет"


def user_requested_voice_reply(text: str) -> bool:
    normalized = normalize_voice_request_text(text)
    return VOICE_REPLY_REQUEST_RE.search(normalized) is not None or user_requested_last_answer_voice(normalized)


def user_requested_last_answer_voice(text: str) -> bool:
    return VOICE_REDELIVERY_REQUEST_RE.search(normalize_voice_request_text(text)) is not None


def normalize_voice_request_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е").strip(" .!?,:;"))


def strip_voice_reply_request(text: str) -> str:
    cleaned = VOICE_REPLY_COMMAND_CLEANUP_RE.sub("", text, count=1)
    cleaned = VOICE_REPLY_SUFFIX_CLEANUP_RE.sub("", cleaned, count=1)
    cleaned = cleaned.strip(" \t\n\r:,-")
    return cleaned or text


def is_voice_messages_forbidden(error: TelegramError) -> bool:
    return "VOICE_MESSAGES_FORBIDDEN" in str(error)


def _photo_score(photo: dict[str, Any]) -> int:
    file_size = photo.get("file_size")
    if isinstance(file_size, int) and file_size > 0:
        return file_size
    width = photo.get("width")
    height = photo.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return width * height
    return 0


def build_app(settings: Settings) -> BotApp:
    telegram = TelegramClient(settings.telegram_bot_token, timeout=30.0)
    lm_studio = LMStudioClient(
        base_url=settings.lm_studio_base_url,
        model=settings.lm_studio_model,
        temperature=settings.lm_studio_temperature,
        max_tokens=settings.lm_studio_max_tokens,
        timeout=settings.request_timeout_seconds,
    )
    state = StateStore(settings.database_path, max_history_messages=settings.max_history_messages)
    speech = LocalSpeechService(settings) if settings.voice_input_enabled or settings.voice_reply_mode != "off" else None
    return BotApp(settings=settings, telegram=telegram, lm_studio=lm_studio, state=state, speech=speech)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.error("Configuration error: %s", exc)
        return 2

    configure_logging(settings.log_level)
    app = build_app(settings)

    def handle_signal(signum: int, _frame: Any) -> None:
        LOGGER.info("Received signal %s, stopping", signum)
        app.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        app.run()
    finally:
        app.close()
    return 0
