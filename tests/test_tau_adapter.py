"""第 1 天 Tau 运行时边界的契约测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from sparkle_navigator.runtime.tau_adapter import TauRuntimeAdapter, TauSessionRequest
from tau_agent.messages import AssistantMessage, TextContent
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent, TextDeltaEvent
from tau_agent.session import InMemorySessionStorage
from tau_ai.fake import FakeProvider


def test_adapter_runs_a_coding_session_prompt() -> None:
    asyncio.run(_run_prompt())


async def _run_prompt() -> None:
    assistant = AssistantMessage(
        content=[TextContent(text="第 1 天模拟回复")],
        api="test",
        provider="fake",
        model="day1-smoke-model",
        stop_reason="stop",
    )
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=assistant.model_copy(deep=True)),
                TextDeltaEvent(
                    content_index=0,
                    delta="第 1 天模拟回复",
                    partial=assistant.model_copy(deep=True),
                ),
                AssistantDoneEvent(reason="stop", message=assistant),
            ]
        ]
    )

    with TemporaryDirectory() as temp_dir:
        session = await TauRuntimeAdapter().open_session(
            TauSessionRequest(
                cwd=Path(temp_dir),
                model="day1-smoke-model",
                provider=provider,
                storage=InMemorySessionStorage(),
                tools=(),
            )
        )
        try:
            event_types = []
            async for event in session.prompt("第 1 天测试请求"):
                event_types.append(event.type)

            messages = session.messages
            entries = await session.session_entries()

            assert "agent_start" in event_types
            assert "message_update" in event_types
            assert "agent_end" in event_types
            assert any(
                message.role == "user" and message.text == "第 1 天测试请求"
                for message in messages
            )
            assert any(
                message.role == "assistant" and message.text == "第 1 天模拟回复"
                for message in messages
            )
            assert len(provider.calls) == 1
            assert len(entries) >= 3
        finally:
            await session.aclose()
