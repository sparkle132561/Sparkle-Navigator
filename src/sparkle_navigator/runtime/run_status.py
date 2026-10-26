"""把一个回合的 Provider 请求聚合为可核对的状态行。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tau_agent.messages import AssistantMessage, sum_usage

TerminalState = Literal["completed", "truncated", "provider_error", "aborted", "no_final"]
_STATE_LABELS: dict[str, str] = {
    "completed": "已完成",
    "truncated": "输出截断（stop_reason=length）",
    "provider_error": "Provider 错误",
    "aborted": "已中止",
    "no_final": "异常终止（无最终回答）",
}


@dataclass(frozen=True, slots=True)
class RunTurnStats:
    """一次回合中 Provider 报告的请求、Token、耗时与终态合计。"""

    request_count: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int | None
    provider_duration_ms: int
    time_to_first_output_ms: int | None
    terminal_state: TerminalState
    cost_total: float | None


def final_answer_message(
    messages: tuple[AssistantMessage, ...],
) -> AssistantMessage | None:
    """回合内最后一条非工具调用请求；确定终态与回答文本的单一来源。"""

    return next(
        (
            message
            for message in reversed(messages)
            if message.stop_reason != "toolUse"
        ),
        None,
    )


def run_turn_stats(
    messages: tuple[AssistantMessage, ...],
) -> RunTurnStats | None:
    """聚合一个回合的 AssistantMessage；无模型请求时返回 None。"""

    if not messages:
        return None
    final = final_answer_message(messages)
    if final is None:
        terminal_state: TerminalState = "no_final"
    elif final.stop_reason == "length":
        terminal_state = "truncated"
    elif final.stop_reason == "error":
        terminal_state = "provider_error"
    elif final.stop_reason == "aborted":
        terminal_state = "aborted"
    else:
        terminal_state = "completed"
    usage = sum_usage(message.usage for message in messages)
    cost_reported = any(
        message.usage.cost.total != 0
        or message.usage.cost.input != 0
        or message.usage.cost.output != 0
        or message.usage.cost.cache_read != 0
        or message.usage.cost.cache_write != 0
        for message in messages
    )
    timings = tuple(
        message.timing for message in messages if message.timing is not None
    )
    return RunTurnStats(
        request_count=len(messages),
        input_tokens=usage.input,
        output_tokens=usage.output,
        cache_read_tokens=usage.cache_read,
        cache_write_tokens=usage.cache_write,
        reasoning_tokens=usage.reasoning,
        provider_duration_ms=sum(timing.total_duration_ms for timing in timings),
        time_to_first_output_ms=next(
            (
                timing.time_to_first_output_ms
                for timing in timings
                if timing.time_to_first_output_ms is not None
            ),
            None,
        ),
        terminal_state=terminal_state,
        cost_total=usage.cost.total if cost_reported else None,
    )


def format_run_status(stats: RunTurnStats) -> str:
    """渲染状态行；费用未报告时显示未知，不以 0 代替。"""

    token_parts = [f"输入 {stats.input_tokens}", f"输出 {stats.output_tokens}"]
    if stats.cache_read_tokens:
        token_parts.append(f"缓存读 {stats.cache_read_tokens}")
    if stats.cache_write_tokens:
        token_parts.append(f"缓存写 {stats.cache_write_tokens}")
    if stats.reasoning_tokens is not None:
        token_parts.append(f"思考 {stats.reasoning_tokens}")
    parts = [
        f"{stats.request_count} 次请求",
        " / ".join(token_parts) + " tokens",
        f"{stats.provider_duration_ms / 1000:.1f}s",
    ]
    if stats.time_to_first_output_ms is not None:
        parts.append(f"首输出 {stats.time_to_first_output_ms / 1000:.1f}s")
    parts.append(_STATE_LABELS[stats.terminal_state])
    parts.append("费用未知" if stats.cost_total is None else f"费用 ${stats.cost_total:.4f}")
    return " · ".join(parts)


__all__ = [
    "RunTurnStats",
    "TerminalState",
    "final_answer_message",
    "format_run_status",
    "run_turn_stats",
]
