"""读取单个 AgentRun 的持久化输出、Trace 与代码 Evidence。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sparkle_navigator.domain.evaluation import (
    AgentRunDetail,
    ExperimentComparison,
    ExperimentRecord,
    ExperimentSnapshot,
)
from sparkle_navigator.persistence.evaluation_store import EvaluationStore


class RunExplorer:
    """按 Experiment/Run 身份查询已保存的运行详情。"""

    def __init__(self, *, database_path: Path) -> None:
        self._store = EvaluationStore(database_path)

    async def list_experiments(self, *, limit: int = 50) -> tuple[ExperimentRecord, ...]:
        """按最近更新时间列出已持久化的 Experiment。"""

        return await asyncio.to_thread(self._store.list_experiments, limit=limit)

    async def get_experiment(self, experiment_id: str) -> ExperimentSnapshot:
        """读取 Experiment 与其 Run 快照，不恢复或重跑任务。"""

        return await asyncio.to_thread(self._store.get_snapshot, experiment_id)

    async def get_comparison(self, comparison_id: str) -> ExperimentComparison:
        """读取已保存的 Comparison，不重新比较 Experiment。"""

        return await asyncio.to_thread(self._store.get_comparison, comparison_id)

    async def list_comparisons(self, *, limit: int = 50) -> tuple[ExperimentComparison, ...]:
        """按创建时间倒序列出已保存的 Comparison。"""

        return await asyncio.to_thread(self._store.list_comparisons, limit=limit)

    async def get_run(self, experiment_id: str, run_id: str) -> AgentRunDetail:
        """读取一次运行的回答、指标、Usage、Timing、Trace 和 Evidence。"""

        return await asyncio.to_thread(
            self._store.get_run_detail,
            experiment_id,
            run_id,
        )


__all__ = ["RunExplorer"]
