"""验证每回合运行统计与终态（含截断）在产品 TUI 与评测记录中的呈现。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widget import Widget

from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunOutput,
    AgentRunStatus,
    AgentRunUsage,
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.evaluation.core import EvaluationLab
from sparkle_navigator.evaluation.native_backend import NativeBackend
from sparkle_navigator.evaluation.smoke_set import EvidencePathEvaluator
from sparkle_navigator.runtime.run_status import format_run_status, run_turn_stats
from tau_agent.messages import AssistantMessage, ResponseTiming, TextContent
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import InMemorySessionStorage, JsonlSessionStorage
from tau_agent.tools import ToolCall
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.paths import TauPaths
from tau_coding.session_manager import SessionManager
from tau_coding.tui import TauTuiApp, TuiProductPresentation


def _assistant(
    text: str,
    *,
    stop_reason: str,
    input_tokens: int = 100,
    output_tokens: int = 20,
    provider_duration_ms: int = 500,
    cost_total: float = 0.0,
) -> AssistantMessage:
    message = AssistantMessage(
        content=[TextContent(text=text)],
        api="test",
        provider="fake",
        model="run-status-test",
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )
    message.usage.input = input_tokens
    message.usage.output = output_tokens
    message.usage.total_tokens = input_tokens + output_tokens
    message.usage.cost.total = cost_total
    message.timing = ResponseTiming(total_duration_ms=provider_duration_ms)
    return message


def test_completed_turn_formats_requests_tokens_duration_and_unknown_cost() -> None:
    messages = (
        _assistant("thinking", stop_reason="toolUse"),
        _assistant("final", stop_reason="stop"),
    )
    stats = run_turn_stats(messages)
    assert stats is not None
    assert stats.request_count == 2
    assert stats.input_tokens == 200
    assert stats.output_tokens == 40
    assert stats.terminal_state == "completed"
    assert stats.cost_total is None
    rendered = format_run_status(stats)
    assert "2 次请求" in rendered
    assert "输入 200 / 输出 40 tokens" in rendered
    assert "1.0s" in rendered
    assert "已完成" in rendered
    assert "费用未知" in rendered
    # 缓存与思考 Token 为零时不显示，避免以 0 冒充信息。
    assert "缓存读" not in rendered
    assert "思考" not in rendered


def test_cache_and_reasoning_tokens_display_when_reported() -> None:
    final = _assistant("final", stop_reason="stop")
    final.usage.cache_read = 512
    final.usage.cache_write = 64
    final.usage.reasoning = 30
    stats = run_turn_stats((final,))
    assert stats is not None
    assert stats.cache_read_tokens == 512
    rendered = format_run_status(stats)
    assert "缓存读 512" in rendered
    assert "缓存写 64" in rendered
    assert "思考 30" in rendered


def test_length_stop_reason_is_marked_as_truncated_not_completed() -> None:
    stats = run_turn_stats((_assistant("partial", stop_reason="length"),))
    assert stats is not None
    assert stats.terminal_state == "truncated"
    assert "输出截断" in format_run_status(stats)


def test_error_and_aborted_stop_reasons_have_distinct_labels() -> None:
    error_stats = run_turn_stats((_assistant("boom", stop_reason="error"),))
    aborted_stats = run_turn_stats((_assistant("stopped", stop_reason="aborted"),))
    assert error_stats is not None and aborted_stats is not None
    assert "Provider 错误" in format_run_status(error_stats)
    assert "已中止" in format_run_status(aborted_stats)


def test_all_tool_use_turns_are_flagged_no_final() -> None:
    stats = run_turn_stats((_assistant("call", stop_reason="toolUse"),))
    assert stats is not None
    assert stats.terminal_state == "no_final"


def test_no_assistant_messages_yield_no_stats() -> None:
    assert run_turn_stats(()) is None


def test_reported_cost_is_shown_instead_of_unknown() -> None:
    stats = run_turn_stats((_assistant("final", stop_reason="stop", cost_total=0.02),))
    assert stats is not None
    assert stats.cost_total == 0.02
    assert "费用 $0.0200" in format_run_status(stats)
    assert "费用未知" not in format_run_status(stats)


def test_native_backend_records_final_stop_reason_without_failure(
    tmp_path: Path,
) -> None:
    asyncio.run(_native_backend_records_final_stop_reason_without_failure(tmp_path))


async def _native_backend_records_final_stop_reason_without_failure(
    tmp_path: Path,
) -> None:
    truncated = _assistant("partial answer", stop_reason="length")
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=truncated.model_copy(deep=True)),
                AssistantDoneEvent(reason=truncated.stop_reason, message=truncated),
            ]
        ]
    )

    def factory(agent_input: AgentRunInput) -> CodingSessionConfig:
        return CodingSessionConfig(
            provider=provider,
            model="run-status-test",
            cwd=tmp_path,
            storage=JsonlSessionStorage(tmp_path / f"{agent_input.run_id}.jsonl"),
            session_id=agent_input.run_id,
            extensions_enabled=False,
            skills_enabled=False,
            trust_default="never",
        )

    backend = NativeBackend(session_config_factory=factory)
    output = await backend.run(
        AgentRunInput(
            run_id="run-truncated",
            case_id="case",
            repetition=1,
            snapshot_id=None,
            user_input="hi",
        )
    )
    assert output.final_stop_reason == "length"
    assert output.failure is None
    assert output.usage is not None and output.usage.input_tokens == 100


def test_evaluation_lab_persists_final_stop_reason(tmp_path: Path) -> None:
    asyncio.run(_evaluation_lab_persists_final_stop_reason(tmp_path))


async def _evaluation_lab_persists_final_stop_reason(tmp_path: Path) -> None:
    class _FakeRunner:
        def __init__(self) -> None:
            self.inputs: list[AgentRunInput] = []

        async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
            self.inputs.append(agent_input)
            return AgentRunOutput(
                output="Truncated answer",
                usage=AgentRunUsage(input_tokens=10, output_tokens=4, total_tokens=14),
                final_stop_reason="length",
            )

    runner = _FakeRunner()
    lab = EvaluationLab(
        database_path=tmp_path / "evaluation.sqlite3",
        runner=runner,
        evaluator=EvidencePathEvaluator(),
    )
    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="run-status",
            snapshot_id=None,
            cases=(
                EvaluationCase(
                    case_id="truncated-case",
                    user_input="hi",
                    ground_truth={"expected_paths": ["src/handler.py"]},
                ),
            ),
        )
    )
    [event async for event in handle.events()]
    snapshot = await handle.snapshot()
    assert snapshot.experiment.status is ExperimentStatus.COMPLETED
    (run,) = snapshot.runs
    assert run.status is AgentRunStatus.COMPLETED
    assert run.final_stop_reason == "length"


def test_tui_appends_run_status_line_after_settled_turn(tmp_path: Path) -> None:
    asyncio.run(_tui_appends_run_status_line_after_settled_turn(tmp_path))


async def _tui_appends_run_status_line_after_settled_turn(tmp_path: Path) -> None:
    tool = AssistantMessage(
        content=[ToolCall(id="status-call", name="read", arguments={"path": "a"})],
        api="test",
        provider="fake",
        model="run-status-test",
        stop_reason="toolUse",
    )
    final = _assistant("Final answer.", stop_reason="stop", input_tokens=120)
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=tool.model_copy(deep=True)),
                AssistantDoneEvent(reason=tool.stop_reason, message=tool),
            ],
            [
                AssistantStartEvent(partial=final.model_copy(deep=True)),
                AssistantDoneEvent(reason=final.stop_reason, message=final),
            ],
        ]
    )
    manager = SessionManager(paths=TauPaths(home=tmp_path / "tau"))
    record = manager.create_session(cwd=tmp_path, model="run-status-test", title="preset")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="run-status-test",
            cwd=tmp_path,
            storage=InMemorySessionStorage(),
            session_id=record.id,
            session_manager=manager,
            extensions_enabled=False,
            skills_enabled=False,
            trust_default="never",
        )
    )
    presentation = TuiProductPresentation(
        title="Sparkle Navigator",
        workspace_panel_factory=lambda context: _PlaceholderPanel(),
        evaluation_panel_factory=lambda context: _PlaceholderPanel(),
        run_status_formatter=lambda messages, session: _format_status_line(messages),
    )
    app = TauTuiApp(session, product_presentation=presentation)
    app.animation_level = "none"
    try:
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.text = "answer me"
            app.set_focus(prompt)
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            status_items = [item for item in app.state.items if item.role == "status"]
            assert any(
                "2 次请求" in item.text
                and "输入 120 / 输出 20 tokens" in item.text
                and "已完成" in item.text
                and "费用未知" in item.text
                for item in status_items
            ), [item.text for item in status_items]
            assert len(status_items) == 1
    finally:
        await app.close_product_sessions()


def _format_status_line(messages) -> str | None:
    stats = run_turn_stats(tuple(messages))
    if stats is None:
        return None
    return format_run_status(stats)


class _PlaceholderPanel(Widget):
    pass
