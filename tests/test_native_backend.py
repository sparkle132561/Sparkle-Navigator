"""NativeBackend 与 Tau CodingSession 的公开集成契约测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from tau_agent.messages import (
    AssistantMessage,
    ThinkingContent,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
)
from tau_agent.provider_events import (
    AssistantErrorEvent,
    AssistantDoneEvent,
    AssistantStartEvent,
    TextDeltaEvent,
)
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentTool, AgentToolResult
from tau_ai.events import AssistantMessageEvent
from tau_ai.fake import FakeProvider
from tau_coding import CodingSessionConfig, CodingToolRegistration

from sparkle_navigator.domain.evaluation import EvaluationCase, EvaluationRequest
from sparkle_navigator.evaluation import EvaluationLab, NativeBackend, RunExplorer
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime


def test_native_backend_persists_tau_trace_usage_timing_and_jsonl_session(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _native_backend_persists_tau_trace_usage_timing_and_jsonl_session(tmp_path)
    )


async def _native_backend_persists_tau_trace_usage_timing_and_jsonl_session(
    tmp_path: Path,
) -> None:
    secret = "sk-test-day16-sensitive-value"
    session_paths: list[Path] = []

    def make_config(agent_input) -> CodingSessionConfig:
        session_path = tmp_path / f"{agent_input.run_id}.jsonl"
        session_paths.append(session_path)
        return CodingSessionConfig(
            provider=_provider_with_tool_call(secret),
            model="native-backend-test-model",
            storage=JsonlSessionStorage(session_path),
            cwd=tmp_path,
            system="Use the registered lookup tool.",
            additional_tool_registrations=(_lookup_registration(secret),),
            tool_access_policy=ToolProfileRuntime(),
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

    backend = NativeBackend(
        session_config_factory=make_config,
        redacted_values=(secret,),
        max_trace_event_chars=256,
    )
    lab = EvaluationLab(
        database_path=tmp_path / "evaluation.sqlite3",
        runner=backend,
        evaluator=_NoopEvaluator(),
    )
    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="native-smoke",
            snapshot_id=None,
            cases=(
                EvaluationCase(
                    case_id="native-run",
                    user_input="Look up the test value.",
                    ground_truth={},
                ),
            ),
        )
    )
    [event async for event in handle.events()]
    run = (await handle.snapshot()).runs[0]
    detail = await RunExplorer(database_path=tmp_path / "evaluation.sqlite3").get_run(
        run.experiment_id,
        run.run_id,
    )

    assert detail.run.output is not None
    assert "fake lookup result" in detail.run.output
    assert secret not in detail.run.output
    assert detail.run.failure_reason is None
    assert detail.run.tau_session_id == run.run_id
    assert detail.run.usage is not None
    assert detail.run.usage.input_tokens == 18
    assert detail.run.usage.output_tokens == 6
    assert detail.run.usage.total_tokens == 24
    assert detail.run.usage.cost_total == 0.12
    assert detail.run.timing is not None
    assert detail.run.timing.provider_duration_ms >= 0
    assert detail.run.timing.time_to_first_output_ms is not None
    assert (
        detail.run.timing.time_to_first_output_ms
        <= detail.run.timing.provider_duration_ms
    )
    assert [event.sequence for event in detail.trace_events] == list(
        range(1, len(detail.trace_events) + 1)
    )
    assert "tool_execution_start" in [event.event_type for event in detail.trace_events]
    assert "tool_execution_end" in [event.event_type for event in detail.trace_events]
    assert detail.trace_events[-1].event_type == "agent_settled"
    assert any(event.truncated for event in detail.trace_events)
    assert secret not in repr(detail.trace_events)
    assert all("thinking" not in repr(event.payload).casefold() for event in detail.trace_events)
    assert len(session_paths) == 1
    entries = await JsonlSessionStorage(session_paths[0]).read_all()
    assert len(entries) > 3


def test_native_backend_persists_redacted_provider_failure(tmp_path: Path) -> None:
    asyncio.run(_native_backend_persists_redacted_provider_failure(tmp_path))


async def _native_backend_persists_redacted_provider_failure(tmp_path: Path) -> None:
    secret = "sk-test-provider-failure-secret"
    provider = FakeProvider(
        [
            [
                AssistantErrorEvent(
                    reason="error",
                    error=AssistantMessage(
                        content=[],
                        api="test",
                        provider="fake",
                        model="native-backend-test-model",
                        stop_reason="error",
                        error_message=f"invalid api_key={secret}",
                    ),
                )
            ]
        ]
    )

    def make_config(agent_input) -> CodingSessionConfig:
        return CodingSessionConfig(
            provider=provider,
            model="native-backend-test-model",
            storage=JsonlSessionStorage(tmp_path / f"{agent_input.run_id}.jsonl"),
            cwd=tmp_path,
            system="Run the test request.",
            tools=[],
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

    database_path = tmp_path / "failed-native-run.sqlite3"
    lab = EvaluationLab(
        database_path=database_path,
        runner=NativeBackend(
            session_config_factory=make_config,
            redacted_values=(secret,),
        ),
        evaluator=_NoopEvaluator(),
    )
    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="native-failure-smoke",
            snapshot_id=None,
            cases=(
                EvaluationCase(
                    case_id="provider-error",
                    user_input="Trigger provider error.",
                    ground_truth={},
                ),
            ),
        )
    )
    [event async for event in handle.events()]
    run = (await handle.snapshot()).runs[0]
    detail = await RunExplorer(database_path=database_path).get_run(
        run.experiment_id,
        run.run_id,
    )

    assert detail.run.status.value == "FAILED"
    assert detail.run.failure_reason == "provider_error"
    assert detail.run.failure_message is not None
    assert secret not in detail.run.failure_message
    assert "[REDACTED]" in detail.run.failure_message
    assert detail.trace_events[-1].event_type == "agent_settled"
    assert secret not in repr(detail.trace_events)


def test_native_backend_cancellation_persists_partial_trace(tmp_path: Path) -> None:
    asyncio.run(_native_backend_cancellation_persists_partial_trace(tmp_path))


async def _native_backend_cancellation_persists_partial_trace(tmp_path: Path) -> None:
    provider = _BlockingProvider()

    def make_config(_agent_input) -> CodingSessionConfig:
        return CodingSessionConfig(
            provider=provider,
            model="native-backend-blocking-model",
            storage=JsonlSessionStorage(tmp_path / "cancelled.jsonl"),
            cwd=tmp_path,
            system="Wait for cancellation.",
            tools=[],
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

    database_path = tmp_path / "cancelled-native-run.sqlite3"
    lab = EvaluationLab(
        database_path=database_path,
        runner=NativeBackend(session_config_factory=make_config),
        evaluator=_NoopEvaluator(),
    )
    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="native-cancel-smoke",
            snapshot_id=None,
            cases=(
                EvaluationCase(
                    case_id="cancel-provider-stream",
                    user_input="Wait and cancel.",
                    ground_truth={},
                ),
            ),
        )
    )
    await provider.started.wait()
    await handle.cancel()
    run = (await handle.snapshot()).runs[0]
    detail = await RunExplorer(database_path=database_path).get_run(
        run.experiment_id,
        run.run_id,
    )

    assert detail.run.status.value == "CANCELLED"
    assert detail.run.failure_reason is None
    assert detail.trace_events
    assert "message_start" in [event.event_type for event in detail.trace_events]
    assert [event.sequence for event in detail.trace_events] == list(
        range(1, len(detail.trace_events) + 1)
    )


def _lookup_registration(secret: str) -> CodingToolRegistration:
    async def execute(_tool_call_id, _arguments, _signal=None, _on_update=None):
        return AgentToolResult(
            content=[TextContent(text=f"fake lookup result: api_key={secret}")]
        )

    tool = AgentTool(
        name="lookup_test_value",
        label="lookup test value",
        description="Read one fixed test value.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute_fn=execute,
        execution_mode="sequential",
    )
    return CodingToolRegistration(
        tool=tool,
        identity="test.lookup-test-value:v1",
        source="test",
        capabilities=frozenset({"read_only"}),
    )


def _provider_with_tool_call(secret: str) -> FakeProvider:
    tool_message = AssistantMessage(
        content=[
            ToolCall(
                id="lookup-call",
                name="lookup_test_value",
                arguments={},
            )
        ],
        api="test",
        provider="fake",
        model="native-backend-test-model",
        stop_reason="toolUse",
        usage=Usage(
            input=10,
            output=2,
            total_tokens=12,
            cost=UsageCost(input=0.01, output=0.02, total=0.03),
        ),
    )


    final_message = AssistantMessage(
        content=[
            ThinkingContent(thinking="private reasoning must not be stored"),
            TextContent(
                text=(
                    f"Answer: fake lookup result; api_key={secret}; "
                    + "long detail " * 50
                )
            ),
        ],
        api="test",
        provider="fake",
        model="native-backend-test-model",
        stop_reason="stop",
        usage=Usage(
            input=8,
            output=4,
            total_tokens=12,
            cost=UsageCost(input=0.04, output=0.05, total=0.09),
        ),
    )
    tool_message.timing = _timing(10, 50)
    final_message.timing = _timing(20, 70)
    return FakeProvider(
        [
            [
                AssistantStartEvent(partial=tool_message.model_copy(deep=True)),
                AssistantDoneEvent(reason="toolUse", message=tool_message),
            ],
            [
                AssistantStartEvent(partial=final_message.model_copy(deep=True)),
                TextDeltaEvent(
                    content_index=0,
                    delta=final_message.text,
                    partial=final_message.model_copy(deep=True),
                ),
                AssistantDoneEvent(reason="stop", message=final_message),
            ],
        ]
    )


class _BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages,
        tools,
        signal=None,
        session_id=None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        del model, system, messages, tools, signal, session_id
        partial = AssistantMessage(
            content=[TextContent(text="partial assistant response")],
            api="test",
            provider="blocking",
            model="native-backend-blocking-model",
            stop_reason="stop",
        )

        async def events() -> AsyncIterator[AssistantMessageEvent]:
            yield AssistantStartEvent(partial=partial)
            self.started.set()
            await asyncio.Event().wait()

        return events()


def _timing(time_to_first_output_ms: int, total_duration_ms: int):
    from tau_agent.messages import ResponseTiming

    return ResponseTiming(
        time_to_first_output_ms=time_to_first_output_ms,
        total_duration_ms=total_duration_ms,
    )


class _NoopEvaluator:
    def evaluate(self, _case, _output):
        return ()
