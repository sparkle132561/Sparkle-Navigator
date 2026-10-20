"""组合确定性与异步评测器，并保持分项结果彼此独立。"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from sparkle_navigator.domain.evaluation import (
    AgentRunOutput,
    EvaluationCase,
    EvaluationResult,
)
from sparkle_navigator.evaluation.core import CaseEvaluator


class CompositeEvaluator:
    """按配置顺序运行多个 Evaluator，不汇总成掩盖分项的总分。"""

    def __init__(self, evaluators: Sequence[CaseEvaluator]) -> None:
        self._evaluators = tuple(evaluators)
        if not self._evaluators:
            raise ValueError("CompositeEvaluator 至少需要一个 Evaluator。")

    async def evaluate(
        self,
        case: EvaluationCase,
        output: AgentRunOutput,
    ) -> tuple[EvaluationResult, ...]:
        results: list[EvaluationResult] = []
        for evaluator in self._evaluators:
            pending = evaluator.evaluate(case, output)
            result = await pending if inspect.isawaitable(pending) else pending
            results.extend(result)
        return tuple(results)


__all__ = ["CompositeEvaluator"]
