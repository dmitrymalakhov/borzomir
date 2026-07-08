from __future__ import annotations

from typing import Any

from .http_json import JsonHttpError, request_json


class LMStudioError(RuntimeError):
    """Raised when LM Studio returns an invalid or failed response."""


class LMStudioClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model if model is None else model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        try:
            data = request_json(
                f"{self.base_url}/chat/completions",
                method="POST",
                payload=payload,
                timeout=self.timeout,
            )
        except JsonHttpError as exc:
            raise LMStudioError(str(exc)) from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LMStudioError("LM Studio response has no choices")

        first = choices[0]
        if not isinstance(first, dict):
            raise LMStudioError("LM Studio choice is not an object")

        finish_reason = first.get("finish_reason")
        message = first.get("message")
        content = None
        reasoning_content = None
        if isinstance(message, dict):
            content = _extract_text(message.get("content"))
            reasoning_content = _extract_text(message.get("reasoning_content"))
        if content is None:
            content = _extract_text(first.get("text"))
        if not isinstance(content, str) or not content.strip():
            detail = "LM Studio returned an empty final answer"
            if finish_reason:
                detail += f" (finish_reason={finish_reason})"
            if reasoning_content:
                detail += (
                    ". The model produced reasoning tokens but no visible answer. "
                    "Increase LM_STUDIO_MAX_TOKENS or use a less reasoning-heavy model."
                )
            raise LMStudioError(detail)
        return content.strip()

    def list_models(self) -> list[str]:
        try:
            data = request_json(f"{self.base_url}/models", timeout=self.timeout)
        except JsonHttpError as exc:
            raise LMStudioError(str(exc)) from exc

        models = data.get("data")
        if not isinstance(models, list):
            raise LMStudioError("LM Studio /models response has no data list")

        result: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                result.append(item["id"])
        return result


def _extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks) if chunks else None
    return None
