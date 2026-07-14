from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


DEFAULT_ADMIN_USER_ID = 92174505
DEFAULT_LM_STUDIO_BASE_URL = "http://host.docker.internal:1234/v1"
DEFAULT_LM_STUDIO_MODEL = "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive"
DEFAULT_DATABASE_PATH = "/data/borzomir.sqlite3"
DEFAULT_WHISPER_BIN = "/usr/local/bin/whisper-cli"
DEFAULT_WHISPER_MODEL_PATH = "/models/whisper/ggml-small.bin"
DEFAULT_PIPER_BIN = "/usr/local/bin/piper"
DEFAULT_PIPER_MODEL_PATH = "/models/piper/ru_RU-ruslan-medium.onnx"
DEFAULT_PIPER_CONFIG_PATH = "/models/piper/ru_RU-ruslan-medium.onnx.json"
VOICE_REPLY_MODES = frozenset({"off", "voice-input", "always"})
DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. Отвечай по-русски, кратко и по делу, "
    "если пользователь не попросил иначе."
)


class ConfigError(ValueError):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    admin_user_id: int
    allowed_user_ids: frozenset[int]
    lm_studio_base_url: str
    lm_studio_model: str
    database_path: str
    system_prompt: str
    max_history_messages: int
    summary_enabled: bool
    summary_max_tokens: int
    max_image_bytes: int
    max_text_document_bytes: int
    lm_studio_temperature: float
    lm_studio_max_tokens: int
    request_timeout_seconds: float
    telegram_poll_timeout_seconds: int
    voice_input_enabled: bool
    voice_reply_mode: str
    max_voice_bytes: int
    max_voice_seconds: int
    max_tts_chars: int
    speech_timeout_seconds: float
    whisper_bin: str
    whisper_model_path: str
    whisper_language: str
    piper_bin: str
    piper_model_path: str
    piper_config_path: str
    log_level: str

    def is_admin(self, user_id: int) -> bool:
        return user_id == self.admin_user_id

    def is_user_allowed(self, user_id: int) -> bool:
        return self.is_admin(user_id) or not self.allowed_user_ids or user_id in self.allowed_user_ids


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if env is None else env
    token = source.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token or token == "replace-me":
        raise ConfigError("TELEGRAM_BOT_TOKEN is required")

    return Settings(
        telegram_bot_token=token,
        admin_user_id=_read_int(source, "ADMIN_USER_ID", DEFAULT_ADMIN_USER_ID, minimum=1),
        allowed_user_ids=parse_user_ids(source.get("ALLOWED_USER_IDS", "")),
        lm_studio_base_url=_read_str(source, "LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL).rstrip("/"),
        lm_studio_model=_read_str(source, "LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL),
        database_path=_read_str(source, "DATABASE_PATH", DEFAULT_DATABASE_PATH),
        system_prompt=_read_str(source, "SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        max_history_messages=_read_int(source, "MAX_HISTORY_MESSAGES", 16, minimum=0),
        summary_enabled=_read_bool(source, "SUMMARY_ENABLED", True),
        summary_max_tokens=_read_int(source, "SUMMARY_MAX_TOKENS", 4096, minimum=128),
        max_image_bytes=_read_int(source, "MAX_IMAGE_BYTES", 20 * 1024 * 1024, minimum=1),
        max_text_document_bytes=_read_int(source, "MAX_TEXT_DOCUMENT_BYTES", 1 * 1024 * 1024, minimum=1),
        lm_studio_temperature=_read_float(source, "LM_STUDIO_TEMPERATURE", 0.7, minimum=0.0),
        lm_studio_max_tokens=_read_int(source, "LM_STUDIO_MAX_TOKENS", 4096, minimum=1),
        request_timeout_seconds=_read_float(source, "REQUEST_TIMEOUT_SECONDS", 180.0, minimum=1.0),
        telegram_poll_timeout_seconds=_read_int(source, "TELEGRAM_POLL_TIMEOUT_SECONDS", 50, minimum=1),
        voice_input_enabled=_read_bool(source, "VOICE_INPUT_ENABLED", True),
        voice_reply_mode=_read_voice_reply_mode(source.get("VOICE_REPLY_MODE", "voice-input")),
        max_voice_bytes=_read_int(source, "MAX_VOICE_BYTES", 20 * 1024 * 1024, minimum=1),
        max_voice_seconds=_read_int(source, "MAX_VOICE_SECONDS", 120, minimum=1),
        max_tts_chars=_read_int(source, "MAX_TTS_CHARS", 2000, minimum=1),
        speech_timeout_seconds=_read_float(source, "SPEECH_TIMEOUT_SECONDS", 180.0, minimum=1.0),
        whisper_bin=_read_str(source, "WHISPER_BIN", DEFAULT_WHISPER_BIN),
        whisper_model_path=_read_str(source, "WHISPER_MODEL_PATH", DEFAULT_WHISPER_MODEL_PATH),
        whisper_language=_read_str(source, "WHISPER_LANGUAGE", "ru"),
        piper_bin=_read_str(source, "PIPER_BIN", DEFAULT_PIPER_BIN),
        piper_model_path=_read_str(source, "PIPER_MODEL_PATH", DEFAULT_PIPER_MODEL_PATH),
        piper_config_path=_read_str(source, "PIPER_CONFIG_PATH", DEFAULT_PIPER_CONFIG_PATH),
        log_level=_read_str(source, "LOG_LEVEL", "INFO").upper(),
    )


def parse_user_ids(raw: str) -> frozenset[int]:
    ids: set[int] = set()
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ConfigError(f"Invalid user id in ALLOWED_USER_IDS: {value!r}") from exc
        if parsed <= 0:
            raise ConfigError(f"User ids must be positive: {parsed}")
        ids.add(parsed)
    return frozenset(ids)


def _read_str(source: Mapping[str, str], key: str, default: str) -> str:
    value = source.get(key, default).strip()
    return value or default


def _read_int(source: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
    raw = source.get(key, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}")
    return value


def _read_float(source: Mapping[str, str], key: str, default: float, *, minimum: float) -> float:
    raw = source.get(key, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}")
    return value


def _read_bool(source: Mapping[str, str], key: str, default: bool) -> bool:
    raw = source.get(key, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be a boolean")


def _read_voice_reply_mode(raw: str) -> str:
    value = raw.strip().lower()
    if value in VOICE_REPLY_MODES:
        return value
    options = ", ".join(sorted(VOICE_REPLY_MODES))
    raise ConfigError(f"VOICE_REPLY_MODE must be one of: {options}")
