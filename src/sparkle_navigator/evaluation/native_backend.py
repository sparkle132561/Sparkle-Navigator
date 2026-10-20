"""通过 Tau CodingSession 执行一次持久 AgentRun。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from tau_agent.events import MessageEndEvent, ToolExecutionEndEvent
from tau_agent.messages import AssistantMessage, sum_usage
from tau_coding import CodingSession, CodingSessionConfig, TraceRecorder
from tau_coding.events import CodingSessionEvent
from tau_coding.trace import TraceEvent

from sparkle_navigator.domain.evaluation import (
    AgentRunFailure,
    AgentRunInput,
    AgentRunOutput,
    AgentRunTiming,
    AgentRunUsage,
    RunTraceEvent,
)
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.evaluation.core import AgentRunCancelled
from sparkle_navigator.security.redaction import (
    sanitize_trace_payload,
    sanitize_trace_text,
)


class CodingSessionConfigFactory(Protocol):
    """为每个 AgentRun 提供新的 Tau CodingSession 配置。"""

    def __call__(self, agent_input: AgentRunInput) -> CodingSessionConfig: ...


class NativeBackend:
    """把一个公开 Run 输入连接到独立 Tau CodingSession 的 Backend。"""

    def __init__(
        self,
        *,
        session_config_factory: CodingSessionConfigFactory,
        redacted_values: tuple[str, ...] = (),
        max_trace_event_chars: int = 100_000,
    ) -> None:
        if max_trace_event_chars < 256:
            raise ValueError("max_trace_event_chars 至少为 256。")
        self._session_config_factory = session_config_factory
        self._redacted_values = tuple(value for value in redacted_values if value)
        self._max_trace_event_chars = max_trace_event_chars

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        """通过 Tau prompt/Event/Session 路径执行一次任务并收集观测。"""

        config = self._session_config_factory(agent_input)
        if config.session_id not in {None, agent_input.run_id}:
            raise ValueError("NativeBackend 要求每个 AgentRun 使用独立 Tau Session ID。")
        session = await CodingSession.load(
            replace(config, session_id=agent_input.run_id)
        )
        recorder = TraceRecorder(session_id=session.session_id or agent_input.run_id)
        events: list[CodingSessionEvent] = []
        started_at = perf_counter()
        failure: AgentRunFailure | None = None
        cancelled = False
        try:
            async for event in recorder.capture(session.prompt(agent_input.user_input)):
                events.append(event)
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            failure = AgentRunFailure(
                code=type(exc).__name__,
                message=sanitize_trace_text(
                    str(exc) or "Tau CodingSession failed.",
                    secret_values=self._redacted_values,
                ),
            )
        finally:
            try:
                await session.aclose()
            except asyncio.CancelledError:
                cancelled = True
            except Exception as exc:
                if failure is None:
                    failure = AgentRunFailure(
                        code=type(exc).__name__,
                        message=sanitize_trace_text(
                            str(exc) or "Tau CodingSession cleanup failed.",
                            secret_values=self._redacted_values,
                        ),
                    )

        assistant_messages = tuple(
            event.message
            for event in events
            if isinstance(event, MessageEndEvent)
            and isinstance(event.message, AssistantMessage)
        )
        final_message = next(
            (
                message
                for message in reversed(assistant_messages)
                if message.stop_reason != "toolUse"
            ),
            None,
        )
        if not cancelled and failure is None and final_message is None:
            failure = AgentRunFailure(
                code="missing_final_answer",
                message="Tau CodingSession ended without a final AssistantMessage.",
            )
        elif not cancelled and failure is None and final_message is not None:
            if final_message.stop_reason == "error":
                failure = AgentRunFailure(
                    code="provider_error",
                    message=sanitize_trace_text(
                        final_message.error_message or "Provider returned an error.",
                        secret_values=self._redacted_values,
                    ),
                )
            elif final_message.stop_reason == "aborted":
                failure = AgentRunFailure(
                    code="assistant_aborted",
                    message="Tau Agent Loop ended with stop_reason=aborted.",
                )

        usage = _usage_from_messages(assistant_messages)
        timing = _timing_from_messages(
            assistant_messages,
            run_duration_ms=max(0, round((perf_counter() - started_at) * 1000)),
        )
        trace_events = tuple(
            _persistable_trace_event(
                trace,
                secret_values=self._redacted_values,
                max_chars=self._max_trace_event_chars,
            )
            for trace in recorder.events
        )
        try:
            evidence = _code_evidence_from_events(
                events,
                snapshot_id=agent_input.snapshot_id,
            )
        except ValueError as exc:
            evidence = ()
            if failure is None:
                failure = AgentRunFailure(
                    code="invalid_code_evidence",
                    message=sanitize_trace_text(
                        str(exc),
                        secret_values=self._redacted_values,
                    ),
                )
        output = (
            sanitize_trace_text(
                final_message.text,
                secret_values=self._redacted_values,
            )
            if final_message is not None
            else ""
        )
        output = AgentRunOutput(
            output=output,
            evidence_paths=tuple(dict.fromkeys(item.path for item in evidence)),
            evidence=evidence,
            tau_session_id=session.session_id,
            trace_events=trace_events,
            usage=usage,
            timing=timing,
            failure=failure,
        )
        if cancelled:
            raise AgentRunCancelled(output)
        return output


def _code_evidence_from_events(
    events: list[CodingSessionEvent],
    *,
    snapshot_id: str | None,
) -> tuple[CodeEvidence, ...]:
    found: dict[tuple[str, str, int | None, int | None, str], CodeEvidence] = {}
    for event in events:
        if not isinstance(event, ToolExecutionEndEvent):
            continue
        details = event.result.details
        if (
            not isinstance(details, Mapping)
            or details.get("sparkle_code_evidence_version") != 1
        ):
            continue
        raw_evidence = details.get("code_evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("Sparkle code evidence payload must be a list.")
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                raise ValueError("Sparkle code evidence item must be an object.")
            try:
                evidence = CodeEvidence(**dict(item))
            except (TypeError, ValueError) as exc:
                raise ValueError("Sparkle code evidence item is malformed.") from exc
            if snapshot_id is None or evidence.snapshot_id != snapshot_id:
                raise ValueError("Code evidence Snapshot does not match the AgentRun.")
            key = (
                evidence.snapshot_id,
                evidence.path,
                evidence.line_start,
                evidence.line_end,
                evidence.sha256,
            )
            found.setdefault(key, evidence)
    return tuple(found.values())


def _usage_from_messages(
    messages: tuple[AssistantMessage, ...],
) -> AgentRunUsage | None:
    if not messages:
        return None
    usage = sum_usage(message.usage for message in messages)
    return AgentRunUsage(
        input_tokens=usage.input,
        output_tokens=usage.output,
        total_tokens=usage.total_tokens,
        cache_read_tokens=usage.cache_read,
        cache_write_tokens=usage.cache_write,
        reasoning_tokens=usage.reasoning,
        cost_input=usage.cost.input,
        cost_output=usage.cost.output,
        cost_cache_read=usage.cost.cache_read,
        cost_cache_write=usage.cost.cache_write,
        cost_total=usage.cost.total,
    )


def _timing_from_messages(
    messages: tuple[AssistantMessage, ...],
    *,
    run_duration_ms: int,
) -> AgentRunTiming:
    timings = tuple(message.timing for message in messages if message.timing is not None)
    first_output = next(
        (
            timing.time_to_first_output_ms
            for timing in timings
            if timing.time_to_first_output_ms is not None
        ),
        None,
    )
    return AgentRunTiming(
        run_duration_ms=run_duration_ms,
        provider_duration_ms=sum(timing.total_duration_ms for timing in timings),
        time_to_first_output_ms=first_output,
    )


def _persistable_trace_event(
    trace: TraceEvent,
    *,
    secret_values: tuple[str, ...],
    max_chars: int,
) -> RunTraceEvent:
    raw_payload = trace.event.model_dump(mode="json", by_alias=False)
    payload = sanitize_trace_payload(raw_payload, secret_values=secret_values)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    truncated = len(serialized) > max_chars
    if truncated:
        payload = {
            "type": trace.event_type,
            "truncated_preview": serialized[:max_chars],
            "original_size_chars": len(serialized),
        }
    return RunTraceEvent(
        sequence=trace.sequence,
        session_id=trace.session_id,
        event_type=trace.event_type,
        timestamp=trace.timestamp,
        payload=payload,
        truncated=truncated,
    )


__all__ = ["CodingSessionConfigFactory", "NativeBackend"]
