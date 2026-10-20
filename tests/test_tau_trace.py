"""第 2 天 Tau Event 标准化与 Trace 测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from tau_agent.events import AgentEndEvent, AgentStartEvent, ToolExecutionStartEvent
from tau_agent.messages import AssistantMessage, TextContent
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent, TextDeltaEvent
from tau_agent.session import InMemorySessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig, TraceRecorder


def test_trace_recorder_preserves_agent_and_tool_events() -> None:
    asyncio.run(_run_synthetic_trace())


async def _run_synthetic_trace() -> None:
    async def events():
        yield AgentStartEvent()
        yield ToolExecutionStartEvent(
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "README.md"},
        )
        yield AgentEndEvent(messages=[])

    recorder = TraceRecorder(session_id="synthetic-session")
    forwarded = [event async for event in recorder.capture(events())]
    traces = recorder.events

    assert [event.type for event in forwarded] == [
        "agent_start",
        "tool_execution_start",
        "agent_end",
    ]
    assert [trace.sequence for trace in traces] == [1, 2, 3]
    assert all(trace.session_id == "synthetic-session" for trace in traces)
    assert [trace.event_type for trace in traces] == [event.type for event in forwarded]
    assert all(trace.event is event for trace, event in zip(traces, forwarded, strict=True))


def test_trace_recorder_captures_coding_session_events() -> None:
    asyncio.run(_run_coding_session_trace())


async def _run_coding_session_trace() -> None:
    assistant = AssistantMessage(
        content=[TextContent(text="第 2 天模拟回复")],
        api="test",
        provider="fake",
        model="day2-trace-model",
        stop_reason="stop",
    )
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=assistant.model_copy(deep=True)),
                TextDeltaEvent(
                    content_index=0,
                    delta="第 2 天模拟回复",
                    partial=assistant.model_copy(deep=True),
                ),
                AssistantDoneEvent(reason="stop", message=assistant),
            ]
        ]
    )

    with TemporaryDirectory() as temp_dir:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="day2-trace-model",
                storage=InMemorySessionStorage(),
                cwd=Path(temp_dir),
                system="Day 2 trace test",
                tools=[],
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )
        try:
            recorder = TraceRecorder(session_id="coding-session")
            forwarded = [
                event async for event in recorder.capture(session.prompt("第 2 天测试请求"))
            ]
            traces = recorder.events

            assert traces

            assert [trace.sequence for trace in traces] == list(range(1, len(traces) + 1))
            assert "agent_start" in [trace.event_type for trace in traces]
            assert "message_update" in [trace.event_type for trace in traces]
            assert "agent_end" in [trace.event_type for trace in traces]
            assert traces[-1].event_type == "agent_settled"
            assert all(trace.session_id == "coding-session" for trace in traces)
            assert all(trace.event is event for trace, event in zip(traces, forwarded, strict=True))
            assert len(provider.calls) == 1
        finally:
            await session.aclose()
