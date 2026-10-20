"""验证重复 AgentRun 的基线比较只聚合质量分项。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunOutput,
    AgentRunRecord,
    AgentRunStatus,
    AgentRunUsage,
    EvaluationCase,
    EvaluationRequest,
    EvaluationResult,
    ExperimentRecord,
    ExperimentSnapshot,
    ExperimentStatus,
    MetricDirection,
)
from sparkle_navigator.evaluation import EvaluationLab, EvidencePathEvaluator
from sparkle_navigator.evaluation.regression import RegressionAnalyzer


def test_regression_analyzer_compares_repetition_means_and_thresholds() -> None:
    baseline = _snapshot("baseline", (0.7, 0.8, 0.9))
    improved = _snapshot("improved", (0.85, 0.9, 0.95))
    regressed = _snapshot("regressed", (0.5, 0.6, 0.7))
    analyzer = RegressionAnalyzer()

    improvement = analyzer.compare(
        baseline,
        improved,
        thresholds={"faithfulness": 0.05},
    )
    regression = analyzer.compare(
        baseline,
        regressed,
        thresholds={"faithfulness": 0.05},
    )

    improved_metric = improvement.results[0]
    assert round(improved_metric.baseline_mean, 3) == 0.8
    assert round(improved_metric.candidate_mean, 3) == 0.9
    assert round(improved_metric.delta, 3) == 0.1
    assert improved_metric.baseline_sample_count == 3
    assert improved_metric.candidate_sample_count == 3
    assert improved_metric.baseline_variance > 0
    assert not improved_metric.is_regression
    assert regression.results[0].delta < -0.05
    assert regression.results[0].is_regression


def test_regression_analyzer_does_not_fold_usage_into_quality() -> None:
    baseline = _snapshot("baseline-cost", (0.8, 0.8, 0.8), token_usage=10)
    candidate = _snapshot("candidate-cost", (0.8, 0.8, 0.8), token_usage=100_000)

    comparison = RegressionAnalyzer().compare(baseline, candidate)

    assert len(comparison.results) == 1
    assert comparison.results[0].delta == 0.0
    assert not comparison.results[0].is_regression


def test_regression_analyzer_respects_lower_is_better_direction() -> None:
    baseline = _snapshot("latency-baseline", (120.0, 130.0, 125.0), metric="latency_ms")
    improved = _snapshot("latency-improved", (90.0, 100.0, 95.0), metric="latency_ms")
    regressed = _snapshot("latency-regressed", (150.0, 160.0, 155.0), metric="latency_ms")

    better_result = RegressionAnalyzer().compare(baseline, improved)
    worse_result = RegressionAnalyzer().compare(baseline, regressed)

    assert better_result.results[0].direction is MetricDirection.LOWER_IS_BETTER
    assert not better_result.results[0].is_regression
    assert worse_result.results[0].is_regression


def test_evaluation_lab_persists_baseline_candidate_comparison(
    tmp_path: Path,
) -> None:
    asyncio.run(_evaluation_lab_persists_baseline_candidate_comparison(tmp_path))


async def _evaluation_lab_persists_baseline_candidate_comparison(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "comparison.sqlite3"
    baseline_lab = EvaluationLab(
        database_path=database_path,
        runner=_EvidenceRunner(matches_expected=True, usage_tokens=12),
        evaluator=EvidencePathEvaluator(),
    )
    candidate_lab = EvaluationLab(
        database_path=database_path,
        runner=_EvidenceRunner(matches_expected=False, usage_tokens=900),
        evaluator=EvidencePathEvaluator(),
    )
    request = EvaluationRequest(
        suite_id="retrieval-comparison",
        snapshot_id=None,
        cases=(
            EvaluationCase(
                case_id="find-expected-file",
                user_input="Find the expected file.",
                ground_truth={"expected_paths": ["src/expected.py"]},
            ),
        ),
        repetitions=3,
    )
    baseline_handle = await baseline_lab.evaluate(
        EvaluationRequest(
            suite_id=request.suite_id,
            snapshot_id=request.snapshot_id,
            cases=request.cases,
            repetitions=request.repetitions,
            configuration={"variant": "baseline"},
        )
    )
    candidate_handle = await candidate_lab.evaluate(
        EvaluationRequest(
            suite_id=request.suite_id,
            snapshot_id=request.snapshot_id,
            cases=request.cases,
            repetitions=request.repetitions,
            configuration={"variant": "candidate"},
        )
    )
    [event async for event in baseline_handle.events()]
    [event async for event in candidate_handle.events()]

    comparison = await baseline_lab.compare(
        baseline_handle.experiment_id,
        candidate_handle.experiment_id,
        thresholds={"evidence_precision": 0.05, "evidence_recall": 0.05},
    )
    reopened = await candidate_lab.open_comparison(comparison.comparison_id)

    assert reopened == comparison
    assert {result.metric_name for result in comparison.results} == {
        "evidence_precision",
        "evidence_recall",
    }
    assert all(result.baseline_sample_count == 3 for result in comparison.results)
    assert all(result.candidate_sample_count == 3 for result in comparison.results)
    assert all(result.is_regression for result in comparison.results)
    assert all(result.baseline_mean == 1.0 for result in comparison.results)
    assert all(result.candidate_mean == 0.0 for result in comparison.results)
    assert all(not hasattr(result, "baseline_usage") for result in comparison.results)


def _snapshot(
    experiment_id: str,
    scores: tuple[float, ...],
    *,
    token_usage: int = 20,
    metric: str = "faithfulness",
) -> ExperimentSnapshot:
    now = datetime.now(UTC)
    experiment = ExperimentRecord(
        experiment_id=experiment_id,
        suite_id="retrieval-suite",
        snapshot_id="a" * 64,
        repetitions=len(scores),
        configuration={"variant": experiment_id},
        status=ExperimentStatus.COMPLETED,
        failure_reason=None,
        created_at=now,
        updated_at=now,
    )
    runs = tuple(
        AgentRunRecord(
            run_id=f"{experiment_id}-{repetition}",
            experiment_id=experiment_id,
            case_id="case-1",
            repetition=repetition,
            attempt=1,
            status=AgentRunStatus.COMPLETED,
            output=f"answer-{repetition}",
            tau_session_id=f"session-{repetition}",
            failure_reason=None,
            failure_message=None,
            usage=_usage(token_usage),
            timing=None,
            results=(
                EvaluationResult(
                    metric_name=metric,
                    value=score,
                    evaluator_id=(
                        "sparkle.latency"
                        if metric == "latency_ms"
                        else "ragas.faithfulness"
                    ),
                    evaluator_version="0.4.3",
                    direction=(
                        MetricDirection.LOWER_IS_BETTER
                        if metric == "latency_ms"
                        else MetricDirection.HIGHER_IS_BETTER
                    ),
                ),
            ),
            created_at=now,
            updated_at=now,
        )
        for repetition, score in enumerate(scores, start=1)
    )
    return ExperimentSnapshot(experiment=experiment, runs=runs)


def _usage(token_count: int) -> AgentRunUsage:
    return AgentRunUsage(
        input_tokens=token_count,
        output_tokens=0,
        total_tokens=token_count,
    )


class _EvidenceRunner:
    def __init__(self, *, matches_expected: bool, usage_tokens: int) -> None:
        self._matches_expected = matches_expected
        self._usage_tokens = usage_tokens

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        del agent_input
        path = "src/expected.py" if self._matches_expected else "src/wrong.py"
        return AgentRunOutput(
            output="The selected file is " + path,
            evidence_paths=(path,),
            usage=_usage(self._usage_tokens),
        )
