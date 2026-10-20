"""将单项 Ragas collection Metric 转为 Sparkle EvaluationResult。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from sparkle_navigator.domain.evaluation import (
    AgentRunOutput,
    EvaluationCase,
    EvaluationResult,
    MetricDirection,
)
from sparkle_navigator.evaluation.core import CaseEvaluator
from sparkle_navigator.security.redaction import sanitize_trace_text


class RagasMetricResult(Protocol):
    """Ragas v0.4 collection metric 的最小结果接口。"""

    value: float | None
    reason: str | None


class RagasCollectionMetric(Protocol):
    """Ragas collection Metric 的异步 `ascore(**kwargs)` 接口。"""

    async def ascore(self, **kwargs: object) -> RagasMetricResult: ...


RagasInputBuilder = Callable[
    [EvaluationCase, AgentRunOutput],
    Mapping[str, object],
]


class RagasMetricAdapter(CaseEvaluator):
    """单项 Ragas Metric 适配器；不创建 Ragas Dataset 或 Experiment。"""

    def __init__(
        self,
        *,
        metric: RagasCollectionMetric,
        metric_name: str,
        ragas_version: str,
        input_builder: RagasInputBuilder,
        direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    ) -> None:
        if not metric_name.strip() or not ragas_version.strip():
            raise ValueError("Ragas Metric 必须有 metric_name 和 ragas_version。")
        self._metric = metric
        self._metric_name = metric_name
        self._ragas_version = ragas_version
        self._input_builder = input_builder
        self._direction = direction

    async def evaluate(
        self,
        case: EvaluationCase,
        output: AgentRunOutput,
    ) -> tuple[EvaluationResult, ...]:
        metric_input = dict(self._input_builder(case, output))
        raw_result = await self._metric.ascore(**metric_input)
        value = raw_result.value
        if value is not None and (
            not isinstance(value, (float, int)) or isinstance(value, bool)
        ):
            raise TypeError("Ragas MetricResult.value 必须是数值或 None。")
        reason = raw_result.reason
        details = {
            "reason": sanitize_trace_text(reason) if isinstance(reason, str) else None,
            "input_fields": sorted(metric_input),
        }
        return (
            EvaluationResult(
                metric_name=self._metric_name,
                value=float(value) if value is not None else None,
                details=details,
                evaluator_id=f"ragas.{self._metric_name}",
                evaluator_version=self._ragas_version,
                direction=self._direction,
            ),
        )


def faithfulness_input(
    case: EvaluationCase,
    output: AgentRunOutput,
) -> Mapping[str, object]:
    """构造 Ragas Faithfulness 输入；不将 Ground Truth 交给 Judge。"""

    return {
        "user_input": case.user_input,
        "response": output.output,
        "retrieved_contexts": [item.snippet for item in output.evidence],
    }


def context_relevance_input(
    case: EvaluationCase,
    output: AgentRunOutput,
) -> Mapping[str, object]:
    """构造 Ragas ContextRelevance 输入，不含隐藏真值。"""

    return {
        "user_input": case.user_input,
        "retrieved_contexts": [item.snippet for item in output.evidence],
    }


__all__ = [
    "RagasCollectionMetric",
    "RagasInputBuilder",
    "RagasMetricAdapter",
    "RagasMetricResult",
    "context_relevance_input",
    "faithfulness_input",
]
