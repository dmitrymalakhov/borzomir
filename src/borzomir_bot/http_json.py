from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class JsonHttpError(RuntimeError):
    """Raised when an HTTP JSON request fails."""


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise JsonHttpError(f"HTTP {exc.code} from {redact_sensitive_url(url)}: {details}") from exc
    except URLError as exc:
        raise JsonHttpError(f"Cannot reach {redact_sensitive_url(url)}: {exc.reason}") from exc

    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonHttpError(f"Invalid JSON from {url}: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise JsonHttpError(f"Expected JSON object from {redact_sensitive_url(url)}")
    return data


def redact_sensitive_url(url: str) -> str:
    marker = "/bot"
    index = url.find(marker)
    if index == -1:
        return url
    token_start = index + len(marker)
    token_end = url.find("/", token_start)
    if token_end == -1:
        return url[:token_start] + "<redacted>"
    return url[:token_start] + "<redacted>" + url[token_end:]
