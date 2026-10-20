"""清理持久 Trace 中的秘密配置与隐藏思维内容。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeAlias

SanitizedJson: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["SanitizedJson"]
    | dict[str, "SanitizedJson"]
)

_OMIT = object()
_SECRET_FIELD = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|"
    r"private[_-]?key|client[_-]?secret|password|secret|credential|cookie)"
)
_THOUGHT_FIELD = re.compile(r"(?i)(thinking|thought|chain[_-]?of[_-]?thought)")
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_NAMED_SECRET = re.compile(
    r"(?i)\b(?P<name>(?:[A-Za-z0-9]+[_-])*(?:api[_ -]?key|access[_ -]?token|"
    r"refresh[_ -]?token|private[_ -]?key|client[_ -]?secret|token|secret|"
    r"password|credential))\b[\"']?\s*[:=]\s*[\"']?[^\"'\s,;}&]+"
)
_OPENAI_STYLE_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_URL_SECRET_QUERY = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret)=)[^&#\s]+"
)


def sanitize_trace_payload(
    value: object,
    *,
    secret_values: tuple[str, ...] = (),
) -> dict[str, SanitizedJson]:
    """递归脱敏 JSON 负载；思维字段和 Thinking 内容块直接省略。"""

    sanitized = _sanitize(value, secret_values=secret_values)
    if not isinstance(sanitized, dict):
        return {"value": sanitized if sanitized is not _OMIT else "[OMITTED]"}
    return sanitized


def sanitize_trace_text(text: str, *, secret_values: tuple[str, ...] = ()) -> str:
    """清理文本中的显式秘密值、Bearer Token 和常见 API Key 格式。"""

    sanitized = text
    for secret in sorted(set(secret_values), key=len, reverse=True):
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = _BEARER_SECRET.sub("Bearer [REDACTED]", sanitized)
    sanitized = _NAMED_SECRET.sub(
        lambda match: f"{match.group('name')}=[REDACTED]",
        sanitized,
    )
    sanitized = _OPENAI_STYLE_SECRET.sub("[REDACTED]", sanitized)
    return _URL_SECRET_QUERY.sub(r"\1[REDACTED]", sanitized)


def _sanitize(value: object, *, secret_values: tuple[str, ...]) -> object:
    if isinstance(value, Mapping):
        normalized = {str(key): item for key, item in value.items()}
        if str(normalized.get("type", "")).casefold() == "thinking":
            return _OMIT
        sanitized: dict[str, SanitizedJson] = {}
        for key, item in normalized.items():
            if _THOUGHT_FIELD.search(key):
                continue
            if _SECRET_FIELD.search(key):
                sanitized[key] = "[REDACTED]"
                continue
            child = _sanitize(item, secret_values=secret_values)
            if child is not _OMIT:
                sanitized[key] = child  # type: ignore[assignment]
        return sanitized
    if isinstance(value, (list, tuple)):
        return [
            item
            for original in value
            if (item := _sanitize(original, secret_values=secret_values)) is not _OMIT
        ]
    if isinstance(value, str):
        return sanitize_trace_text(value, secret_values=secret_values)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_trace_text(str(value), secret_values=secret_values)


__all__ = ["sanitize_trace_payload", "sanitize_trace_text"]
