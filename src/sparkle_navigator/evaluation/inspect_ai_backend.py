"""通过可选 Inspect AI 执行单个 Sparkle AgentRun。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from sparkle_navigator.domain.evaluation import AgentRunInput, AgentRunOutput
from sparkle_navigator.evaluation.core import AgentRunExecutor
from sparkle_navigator.security.redaction import sanitize_trace_text


InspectTaskFactory = Callable[[AgentRunInput], object]


class InspectAIBackend(AgentRunExecutor):
    """把一个公开 Sparkle 输入交给一个 Inspect Task 样本执行。"""

    def __init__(
        self,
        *,
        task_factory: InspectTaskFactory,
        model: object,
        log_dir: Path | None = None,
        redacted_values: tuple[str, ...] = (),
    ) -> None:
        if model is None:
            raise ValueError("InspectAIBackend 必须显式指定 model。")
        self._task_factory = task_factory
        self._model = model
        self._log_dir = log_dir
        self._redacted_values = tuple(value for value in redacted_values if value)

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        """异步执行 Inspect Task，并只返回 Sparkle 的公开领域记录。"""

        try:
            from inspect_ai import eval_async
        except ImportError as exc:
            raise RuntimeError(
                "Inspect AI 后端需要安装可选依赖：pip install '.[inspect-ai]'。"
            ) from exc

        task = self._task_factory(agent_input)
        if self._log_dir is not None:
            log_directory = self._log_dir / agent_input.run_id
            logs = await _evaluate_task(
                eval_async=eval_async,
                task=task,
                model=self._model,
                log_directory=log_directory,
            )
            return _output_from_logs(
                logs,
                redacted_values=self._redacted_values,
            )

        with TemporaryDirectory(prefix="sparkle-inspect-") as temporary_directory:
            logs = await _evaluate_task(
                eval_async=eval_async,
                task=task,
                model=self._model,
                log_directory=Path(temporary_directory),
            )
            return _output_from_logs(
                logs,
                redacted_values=self._redacted_values,
            )


async def _evaluate_task(
    *,
    eval_async: Callable[..., Any],
    task: object,
    model: object,
    log_directory: Path,
) -> Any:
    try:
        logs = await eval_async(
            cast(Any, task),
            model=cast(Any, model),
            log_dir=str(log_directory),
            limit=1,
            score=False,
        )
    except BaseExceptionGroup as exc:
        if _cancellation_requested():
            raise asyncio.CancelledError from exc
        raise
    if _cancellation_requested():
        raise asyncio.CancelledError
    return logs


def _cancellation_requested() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _output_from_logs(
    logs: object,
    *,
    redacted_values: tuple[str, ...],
) -> AgentRunOutput:
    if isinstance(logs, (list, tuple)) and not logs:
        # Inspect eval_async 在外部取消后会返回空日志集合，而不是抛出取消异常。
        raise asyncio.CancelledError
    if not isinstance(logs, (list, tuple)) or len(logs) != 1:
        raise ValueError("Inspect AI 必须恰好返回一个 Task 日志。")
    log = logs[0]
    status = _value(getattr(log, "status", None))
    if status in {"cancelled", "canceled"}:
        raise asyncio.CancelledError
    if status != "success":
        raise RuntimeError("Inspect AI Task 未成功完成。")
    samples = getattr(log, "samples", None)
    if not isinstance(samples, (list, tuple)) or len(samples) != 1:
        raise ValueError("Inspect AI Task 必须恰好执行一个样本。")
    sample = samples[0]
    if getattr(sample, "error", None) is not None:
        raise RuntimeError("Inspect AI 样本执行失败。")
    model_output = getattr(sample, "output", None)
    completion = getattr(model_output, "completion", None)
    if not isinstance(completion, str):
        raise ValueError("Inspect AI 样本没有字符串 completion。")
    return AgentRunOutput(
        output=sanitize_trace_text(
            completion,
            secret_values=redacted_values,
        )
    )


def _value(value: object) -> object:
    if isinstance(value, Mapping):
        return value.get("value")
    return getattr(value, "value", value)


__all__ = ["InspectAIBackend", "InspectTaskFactory"]
