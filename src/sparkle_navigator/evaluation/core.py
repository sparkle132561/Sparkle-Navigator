"""协调评测请求、AgentRun、Evaluator 与持久化事件流。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Protocol

from sparkle_navigator.domain.evaluation import (
    AgentRunFailure,
    AgentRunInput,
    AgentRunOutput,
    AgentRunStatus,
    EvaluationCase,
    EvaluationEvent,
    EvaluationRequest,
    EvaluationResult,
    ExperimentComparison,
    EventCursor,
    ExperimentSnapshot,
    ExperimentStatus,
)
from sparkle_navigator.persistence.evaluation_store import EvaluationStore
from sparkle_navigator.evaluation.regression import RegressionAnalyzer


class AgentRunExecutor(Protocol):
    """执行公开 Case 输入；Ground Truth 不属于该接口。"""

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput: ...


class CaseEvaluator(Protocol):
    """在 AgentRun 完成后读取 Case Ground Truth 并产生分项结果。"""

    def evaluate(
        self,
        case: EvaluationCase,
        output: AgentRunOutput,
    ) -> tuple[EvaluationResult, ...] | Awaitable[tuple[EvaluationResult, ...]]: ...


class AgentRunCancelled(asyncio.CancelledError):
    """取消信号携带可持久化的部分 Run Trace。"""

    def __init__(self, output: AgentRunOutput) -> None:
        super().__init__("AgentRun was cancelled.")
        self.output = output


class EvaluationLab:
    """Evaluation Core 的持久化入口。"""

    def __init__(
        self,
        *,
        database_path: Path,
        runner: AgentRunExecutor,
        evaluator: CaseEvaluator,
    ) -> None:
        self._store = EvaluationStore(database_path)
        self._runner = runner
        self._evaluator = evaluator
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def evaluate(self, request: EvaluationRequest) -> ExperimentHandle:
        """持久化 DRAFT/QUEUED 后启动后台执行并立即返回 Handle。"""

        experiment_id = await asyncio.to_thread(
            self._store.create_experiment,
            request,
        )
        handle = ExperimentHandle(experiment_id=experiment_id, lab=self)
        self._tasks[experiment_id] = asyncio.create_task(
            self._execute(experiment_id, request)
        )
        return handle

    async def open(self, experiment_id: str) -> ExperimentHandle:
        """打开已保存的 Experiment；遗留活动状态从 Case 边界恢复。"""

        snapshot = await asyncio.to_thread(self._store.get_snapshot, experiment_id)
        task = self._tasks.get(experiment_id)
        if not snapshot.experiment.status.is_terminal and (
            task is None or task.done()
        ):
            recovered = await asyncio.to_thread(
                self._store.recover_interrupted_experiment,
                experiment_id,
            )
            if recovered:
                request = await asyncio.to_thread(
                    self._store.get_request,
                    experiment_id,
                )
                self._tasks[experiment_id] = asyncio.create_task(
                    self._execute(experiment_id, request)
                )
        return ExperimentHandle(experiment_id=experiment_id, lab=self)

    async def compare(
        self,
        baseline_experiment_id: str,
        candidate_experiment_id: str,
        *,
        thresholds: Mapping[str, float] | None = None,
    ) -> ExperimentComparison:
        """比较两个已完成 Experiment 的质量指标并持久化回归结果。"""

        baseline = await asyncio.to_thread(
            self._store.get_snapshot,
            baseline_experiment_id,
        )
        candidate = await asyncio.to_thread(
            self._store.get_snapshot,
            candidate_experiment_id,
        )
        comparison = RegressionAnalyzer().compare(
            baseline,
            candidate,
            thresholds=dict(thresholds or {}),
        )
        await asyncio.to_thread(self._store.save_comparison, comparison)
        return comparison

    async def open_comparison(self, comparison_id: str) -> ExperimentComparison:
        """读取已持久化的 Comparison，不重新执行 Agent 或 Evaluator。"""

        return await asyncio.to_thread(self._store.get_comparison, comparison_id)

    async def _execute(
        self,
        experiment_id: str,
        request: EvaluationRequest,
    ) -> None:
        had_failure = False
        try:
            snapshot = await asyncio.to_thread(
                self._store.get_snapshot,
                experiment_id,
            )
            await asyncio.to_thread(
                self._store.transition_experiment,
                experiment_id,
                ExperimentStatus.RUNNING,
            )
            cases = {case.case_id: case for case in request.cases}
            for run in snapshot.runs:
                if run.status is not AgentRunStatus.CREATED:
                    continue
                await asyncio.to_thread(
                    self._store.transition_run,
                    run.run_id,
                    AgentRunStatus.QUEUED,
                )
                await asyncio.to_thread(
                    self._store.transition_run,
                    run.run_id,
                    AgentRunStatus.RUNNING,
                )
                case = cases[run.case_id]
                agent_input = AgentRunInput(
                    run_id=run.run_id,
                    case_id=case.case_id,
                    repetition=run.repetition,
                    snapshot_id=request.snapshot_id,
                    user_input=case.user_input,
                )
                try:
                    output = await self._runner.run(agent_input)
                except AgentRunCancelled as exc:
                    await asyncio.to_thread(
                        self._store.cancel_run,
                        run.run_id,
                        output=exc.output,
                    )
                    await asyncio.to_thread(
                        self._store.cancel_experiment,
                        experiment_id,
                    )
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    had_failure = True
                    failed_output = AgentRunOutput(
                        output="",
                        failure=AgentRunFailure(
                            code=type(exc).__name__,
                            message="Runner failed before producing an output.",
                        ),
                    )
                    await asyncio.to_thread(
                        self._store.fail_run,
                        run.run_id,
                        output=failed_output,
                    )
                    continue

                if output.failure is not None:
                    had_failure = True
                    await asyncio.to_thread(
                        self._store.fail_run,
                        run.run_id,
                        output=output,
                    )
                    continue

                if output.evidence and (
                    request.snapshot_id is None
                    or any(
                        evidence.snapshot_id != request.snapshot_id
                        for evidence in output.evidence
                    )
                ):
                    had_failure = True
                    await asyncio.to_thread(
                        self._store.fail_run,
                        run.run_id,
                        output=replace(
                            output,
                            evidence_paths=(),
                            evidence=(),
                            failure=AgentRunFailure(
                                code="invalid_evidence_snapshot",
                                message=(
                                    "Runner Evidence must match the Experiment Snapshot."
                                ),
                            ),
                        ),
                    )
                    continue

                try:
                    await asyncio.to_thread(
                        self._store.transition_run,
                        run.run_id,
                        AgentRunStatus.EVALUATING,
                    )
                    pending_results = self._evaluator.evaluate(case, output)
                    results = (
                        await pending_results
                        if inspect.isawaitable(pending_results)
                        else pending_results
                    )
                    await asyncio.to_thread(
                        self._store.complete_run,
                        run.run_id,
                        output=output,
                        results=results,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    had_failure = True
                    await asyncio.to_thread(
                        self._store.fail_run,
                        run.run_id,
                        output=replace(
                            output,
                            failure=AgentRunFailure(
                                code=type(exc).__name__,
                                message="Evaluation failed after the AgentRun completed.",
                            ),
                        ),
                    )

            await asyncio.to_thread(
                self._store.transition_experiment,
                experiment_id,
                ExperimentStatus.EVALUATING,
            )
            final_status = (
                ExperimentStatus.FAILED
                if had_failure
                else ExperimentStatus.COMPLETED
            )
            failure_reason = "one_or_more_agent_runs_failed" if had_failure else None
            await asyncio.to_thread(
                self._store.transition_experiment,
                experiment_id,
                final_status,
                failure_reason=failure_reason,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(self._store.cancel_experiment, experiment_id)

    async def _cancel(self, experiment_id: str) -> None:
        task = self._tasks.get(experiment_id)
        if task is None:
            await asyncio.to_thread(self._store.cancel_experiment, experiment_id)
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._tasks.pop(experiment_id, None)


class ExperimentHandle:
    """读取持久快照和事件，或取消当前本地 Experiment。"""

    def __init__(self, *, experiment_id: str, lab: EvaluationLab) -> None:
        self.experiment_id = experiment_id
        self._lab = lab

    async def snapshot(self) -> ExperimentSnapshot:
        """只读取已持久化的 Experiment、AgentRun 和 EvaluationResult。"""

        return await asyncio.to_thread(
            self._lab._store.get_snapshot,
            self.experiment_id,
        )

    async def events(
        self,
        *,
        after: EventCursor | None = None,
    ) -> AsyncIterator[EvaluationEvent]:
        """按游标持续读取已先写入 SQLite 的事件，直到 Experiment 终态。"""

        if after is not None and after.experiment_id != self.experiment_id:
            raise ValueError("EventCursor 必须属于当前 Experiment。")
        sequence = after.sequence if after is not None else 0
        while True:
            new_events = await asyncio.to_thread(
                self._lab._store.get_events,
                self.experiment_id,
                after_sequence=sequence,
            )
            for event in new_events:
                sequence = event.sequence
                yield event
            snapshot = await self.snapshot()
            if snapshot.experiment.status.is_terminal:
                if not new_events:
                    return
                continue
            await asyncio.sleep(0.02)

    async def cancel(self) -> None:
        """请求取消本地任务，并持久化每个未完成 Run 和 Experiment 的终态。"""

        await self._lab._cancel(self.experiment_id)


__all__ = [
    "AgentRunCancelled",
    "AgentRunExecutor",
    "CaseEvaluator",
    "EvaluationLab",
    "EventCursor",
    "ExperimentHandle",
]
