"""验证持久 Trace 的秘密和隐藏思维清理规则。"""

from __future__ import annotations

from sparkle_navigator.security.redaction import sanitize_trace_payload


def test_trace_sanitizer_redacts_named_keys_and_omits_thinking_blocks() -> None:
    payload = sanitize_trace_payload(
        {
            "configuration": {"LLM_API_KEY": "plain-secret-value"},
            "content": [
                {
                    "type": "thinking",
                    "thinking": "private reasoning must not persist",
                },
                {
                    "type": "text",
                    "text": (
                        'LLM_API_KEY="named-secret-value" '
                        "Authorization: Bearer bearer-secret-value"
                    ),
                },
            ],
        }
    )

    serialized = str(payload)
    assert "plain-secret-value" not in serialized
    assert "named-secret-value" not in serialized
    assert "bearer-secret-value" not in serialized
    assert "private reasoning" not in serialized
    assert "thinking" not in serialized.casefold()
