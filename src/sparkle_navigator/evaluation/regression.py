"""对同套件重复 Run 的质量分项计算可解释回归信号。"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from statistics import fmean, pvariance
from uuid import uuid4

from sparkle_navigator.domain.evaluation import (
    AgentRunRecord,
    AgentRunStatus,
    ExperimentComparison,
    ExperimentSnapshot,
    ExperimentStatus,
    MetricDirection,
    RegressionResult,
)


class RegressionAnalyzer:
    """比较固定基线和候选 Experiment 的版本化质量指标。"""

    def compare(
        self,
        baseline: ExperimentSnapshot,
        candidate: ExperimentSnapshot,
        *,
        thresholds: dict[str, float] | None = None,
    ) -> ExperimentComparison:
        baseline_experiment = baseline.experiment
        candidate_experiment = candidate.experiment
        if baseline_experiment.status is not ExperimentStatus.COMPLETED:
            raise ValueError("Baseline Experiment 必须先完成。")
        if candidate_experiment.status is not ExperimentStatus.COMPLETED:
            raise ValueError("Candidate Experiment 必须先完成。")
        if baseline_experiment.suite_id != candidate_experiment.suite_id:
            raise ValueError("比较双方必须使用同一 Evaluation Suite。")
        if baseline_experiment.snapshot_id != candidate_experiment.snapshot_id:
            raise ValueError("比较双方必须绑定同一 Repository Snapshot。")
        if baseline_experiment.repetitions != candidate_experiment.repetitions:
            raise ValueError("比较双方必须使用相同的重复次数。")
        if _case_ids(baseline.runs) != _case_ids(candidate.runs):
            raise ValueError("比较双方必须覆盖相同的 Evaluation Case。")

        baseline_values = _metric_values(baseline.runs)
        candidate_values = _metric_values(candidate.runs)
        if not baseline_values or baseline_values.keys() != candidate_values.keys():
            raise ValueError("比较双方必须具有相同的已完成指标和 Evaluator 版本。")

        threshold_values = dict(thresholds or {})
        metric_names = {key[0] for key in baseline_values}
        if set(threshold_values) - metric_names:
            raise ValueError("Regression threshold 指向了当前结果中不存在的指标。")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in threshold_values.values()
        ):
            raise ValueError("Regression threshold 必须是有限的非负数。")

        results: list[RegressionResult] = []
        for key in sorted(baseline_values):
            baseline_scores, baseline_direction = baseline_values[key]
            candidate_scores, candidate_direction = candidate_values[key]
            if baseline_direction is not candidate_direction:
                raise ValueError(f"指标 {key[0]} 的比较方向不一致。")
            baseline_mean = fmean(baseline_scores)
            candidate_mean = fmean(candidate_scores)
            delta = candidate_mean - baseline_mean
            threshold = threshold_values.get(key[0], 0.0)
            is_regression = (
                delta < -threshold
                if baseline_direction is MetricDirection.HIGHER_IS_BETTER
                else delta > threshold
            )
            results.append(
                RegressionResult(
                    metric_name=key[0],
                    evaluator_id=key[1],
                    evaluator_version=key[2],
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    delta=delta,
                    baseline_variance=pvariance(baseline_scores),
                    candidate_variance=pvariance(candidate_scores),
                    baseline_sample_count=len(baseline_scores),
                    candidate_sample_count=len(candidate_scores),
                    direction=baseline_direction,
                    regression_threshold=threshold,
                    is_regression=is_regression,
                )
            )

        return ExperimentComparison(
            comparison_id=str(uuid4()),
            baseline_experiment_id=baseline_experiment.experiment_id,
            candidate_experiment_id=candidate_experiment.experiment_id,
            thresholds=threshold_values,
            results=tuple(results),
            created_at=datetime.now(UTC),
        )


def _case_ids(runs: tuple[AgentRunRecord, ...]) -> frozenset[str]:
    return frozenset(run.case_id for run in runs)


def _metric_values(
    runs: tuple[AgentRunRecord, ...],
) -> dict[tuple[str, str, str], tuple[list[float], MetricDirection]]:
    latest_runs: dict[tuple[str, int], AgentRunRecord] = {}
    for run in sorted(runs, key=lambda item: (item.case_id, item.repetition, item.attempt)):
        latest_runs[(run.case_id, run.repetition)] = run

    values: dict[tuple[str, str, str], tuple[list[float], MetricDirection]] = {}
    for run in latest_runs.values():
        if run.status is not AgentRunStatus.COMPLETED:
            continue
        for result in run.results:
            if result.value is None:
                continue
            key = (
                result.metric_name,
                result.evaluator_id,
                result.evaluator_version,
            )
            scores, direction = values.setdefault(
                key,
                ([], result.direction),
            )
            if direction is not result.direction:
                raise ValueError(f"指标 {result.metric_name} 的方向定义冲突。")
            scores.append(result.value)
    if any(not scores for scores, _direction in values.values()):
        raise ValueError("比较指标必须至少有一个非空分数。")
    return values


__all__ = ["RegressionAnalyzer"]
