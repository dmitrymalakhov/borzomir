from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .http_json import JsonHttpError, request_json


MAX_TELEGRAM_MESSAGE_LENGTH = 4096


class TelegramError(RuntimeError):
    """Raised when Telegram Bot API call fails."""


class TelegramClient:
    def __init__(self, token: str, *, timeout: float = 30.0) -> None:
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.file_base_url = f"https://api.telegram.org/file/bot{token}"
        self.timeout = timeout

    def get_updates(self, *, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        data = self._call("getUpdates", payload, timeout=timeout + 10)
        result = data.get("result", [])
        if not isinstance(result, list):
            raise TelegramError("Telegram getUpdates result is not a list")
        return [item for item in result if isinstance(item, dict)]

    def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> None:
        for chunk in split_telegram_message(text):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": chunk,
            }
            if reply_to_message_id is not None:
                payload["reply_parameters"] = {"message_id": reply_to_message_id}
            self._call("sendMessage", payload)

    def send_voice(
        self,
        *,
        chat_id: int,
        voice_data: bytes,
        filename: str = "answer.ogg",
        reply_to_message_id: int | None = None,
    ) -> None:
        fields = {"chat_id": str(chat_id)}
        if reply_to_message_id is not None:
            fields["reply_parameters"] = json.dumps({"message_id": reply_to_message_id})
        self._call_multipart(
            "sendVoice",
            fields=fields,
            files={"voice": (filename, voice_data, "audio/ogg")},
        )

    def send_chat_action(self, *, chat_id: int, action: str = "typing") -> None:
        self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    def get_file_path(self, file_id: str) -> str:
        data = self._call("getFile", {"file_id": file_id})
        result = data.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("file_path"), str):
            raise TelegramError("Telegram getFile response has no file_path")
        return result["file_path"]

    def download_file(self, file_path: str) -> bytes:
        try:
            with urlopen(f"{self.file_base_url}/{file_path}", timeout=self.timeout) as response:
                return response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise TelegramError(f"HTTP {exc.code} while downloading Telegram file: {details}") from exc
        except URLError as exc:
            raise TelegramError(f"Cannot download Telegram file: {exc.reason}") from exc

    def _call(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            data = request_json(
                f"{self.base_url}/{method}",
                method="POST",
                payload=payload,
                timeout=self.timeout if timeout is None else timeout,
            )
        except JsonHttpError as exc:
            raise TelegramError(str(exc)) from exc

        if data.get("ok") is not True:
            raise TelegramError(f"Telegram {method} failed: {data}")
        return data

    def _call_multipart(
        self,
        method: str,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> dict[str, Any]:
        body, content_type = encode_multipart_form_data(fields=fields, files=files)
        request = Request(
            f"{self.base_url}/{method}",
            data=body,
            headers={"Accept": "application/json", "Content-Type": content_type},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise TelegramError(f"HTTP {exc.code} from Telegram {method}: {details}") from exc
        except URLError as exc:
            raise TelegramError(f"Cannot reach Telegram {method}: {exc.reason}") from exc

        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise TelegramError(f"Invalid JSON from Telegram {method}: {raw[:300]}") from exc
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise TelegramError(f"Telegram {method} failed: {data}")
        return data


def split_telegram_message(text: str) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > MAX_TELEGRAM_MESSAGE_LENGTH:
        split_at = remaining.rfind("\n", 0, MAX_TELEGRAM_MESSAGE_LENGTH)
        if split_at < MAX_TELEGRAM_MESSAGE_LENGTH // 2:
            split_at = remaining.rfind(" ", 0, MAX_TELEGRAM_MESSAGE_LENGTH)
        if split_at < MAX_TELEGRAM_MESSAGE_LENGTH // 2:
            split_at = MAX_TELEGRAM_MESSAGE_LENGTH
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def encode_multipart_form_data(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----borzomir-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")
    for name, (filename, data, media_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {media_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"
