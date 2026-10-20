"""Evaluation 报告导出以 Sparkle 持久化记录为输入。"""

from __future__ import annotations

import json
import csv
from datetime import UTC, datetime
from pathlib import Path

from sparkle_navigator.domain.evaluation import (
    AgentRunDetail,
    AgentRunRecord,
    AgentRunStatus,
    AgentRunTiming,
    AgentRunUsage,
    EvaluationResult,
    ExperimentComparison,
    ExperimentRecord,
    ExperimentSnapshot,
    ExperimentStatus,
    MetricDirection,
    RegressionResult,
    RunTraceEvent,
)
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.presentation.reports import EvaluationReportExporter


def test_json_experiment_report_contains_run_metrics_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    output_path = _export_example(
        tmp_path,
        format="json",
        output="Found src/example.py; token=private-value",
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["experiment"]["experiment_id"] == "experiment-1"
    assert payload["runs"][0]["run_id"] == "run-1"
    assert payload["runs"][0]["results"][0]["value"] == 1.0
    assert payload["runs"][0]["trace_events"][0]["payload"] == {
        "authorization": "[REDACTED]",
        "status": "completed",
    }
    assert payload["runs"][0]["evidence"][0]["snippet"] == "return '[REDACTED]'"
    assert "ground_truth" not in repr(payload)
    assert "private-value" not in repr(payload)


def _export_example(
    tmp_path: Path,
    *,
    format: str,
    output: str,
) -> Path:
    timestamp = datetime(2026, 10, 10, tzinfo=UTC)
    result = EvaluationResult(
        metric_name="evidence_recall",
        value=1.0,
        details={"api_key": "private-value", "matched_paths": ["src/example.py"]},
        evaluator_id="sparkle.evidence-path",
        evaluator_version="1",
        direction=MetricDirection.HIGHER_IS_BETTER,
    )
    run = AgentRunRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        case_id="case-1",
        repetition=1,
        attempt=1,
        status=AgentRunStatus.COMPLETED,
        output=output,
        tau_session_id=None,
        failure_reason=None,
        failure_message=None,
        usage=AgentRunUsage(input_tokens=10, output_tokens=4, total_tokens=14),
        timing=AgentRunTiming(
            run_duration_ms=40,
            provider_duration_ms=30,
            time_to_first_output_ms=8,
        ),
        results=(result,),
        created_at=timestamp,
        updated_at=timestamp,
    )
    snapshot = ExperimentSnapshot(
        experiment=ExperimentRecord(
            experiment_id="experiment-1",
            suite_id="retrieval-smoke",
            snapshot_id="a" * 64,
            repetitions=1,
            configuration={"model": "mock-model", "api_key": "private-value"},
            status=ExperimentStatus.COMPLETED,
            failure_reason=None,
            created_at=timestamp,
            updated_at=timestamp,
        ),
        runs=(run,),
    )
    detail = AgentRunDetail(
        run=run,
        trace_events=(
            RunTraceEvent(
                sequence=1,
                session_id=None,
                event_type="agent_end",
                timestamp=1.0,
                payload={
                    "authorization": "private-value",
                    "thinking": "not for the report",
                    "status": "completed",
                },
            ),
        ),
        evidence=(
            CodeEvidence(
                snapshot_id="a" * 64,
                path="src/example.py",
                line_start=3,
                line_end=3,
                snippet="return 'private-value'",
                sha256="b" * 64,
                source="text",
            ),
        ),
    )
    output_path = tmp_path / "reports" / f"experiment.{format}"

    exported = EvaluationReportExporter().export_experiment(
        snapshot,
        (detail,),
        output_path,
        format=format,
        redacted_values=("private-value",),
    )
    assert exported == output_path
    return exported


def test_html_experiment_report_escapes_model_output(tmp_path: Path) -> None:
    output_path = _export_example(
        tmp_path,
        format="html",
        output="<script>alert('x')</script> token=private-value",
    )
    html_report = output_path.read_text(encoding="utf-8")

    assert "<!doctype html>" in html_report.lower()
    assert "&lt;script&gt;" in html_report
    assert "<script>" not in html_report
    assert "private-value" not in html_report


def test_csv_experiment_report_neutralizes_spreadsheet_formulas(
    tmp_path: Path,
) -> None:
    output_path = _export_example(
        tmp_path,
        format="csv",
        output='=HYPERLINK("https://example.invalid") token=private-value',
    )
    with output_path.open(encoding="utf-8", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert len(rows) == 1
    assert rows[0]["output"].startswith("'=HYPERLINK(")
    assert "[REDACTED]" in rows[0]["output"]
    assert "private-value" not in repr(rows)


def _comparison_example() -> ExperimentComparison:
    return ExperimentComparison(
        comparison_id="comparison-1",
        baseline_experiment_id="baseline-1",
        candidate_experiment_id="candidate-1",
        thresholds={"evidence_recall": 0.05},
        results=(
            RegressionResult(
                metric_name="evidence_recall",
                evaluator_id="sparkle.evidence-path",
                evaluator_version="1",
                baseline_mean=0.9,
                candidate_mean=0.6,
                delta=-0.3,
                baseline_variance=0.01,
                candidate_variance=0.04,
                baseline_sample_count=3,
                candidate_sample_count=3,
                direction=MetricDirection.HIGHER_IS_BETTER,
                regression_threshold=0.05,
                is_regression=True,
            ),
        ),
        created_at=datetime(2026, 10, 10, tzinfo=UTC),
    )


def test_json_comparison_report_includes_metric_direction_and_regression(
    tmp_path: Path,
) -> None:
    comparison = _comparison_example()
    output_path = tmp_path / "reports" / "comparison.json"

    exported = EvaluationReportExporter().export_comparison(
        comparison,
        output_path,
        format="json",
    )
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["comparison"]["comparison_id"] == "comparison-1"
    assert payload["results"][0]["direction"] == "higher_is_better"
    assert payload["results"][0]["is_regression"] is True


def test_html_comparison_report_escapes_experiment_identifiers(tmp_path: Path) -> None:
    comparison = _comparison_example()
    comparison = ExperimentComparison(
        comparison_id=comparison.comparison_id,
        baseline_experiment_id="<script>baseline</script>",
        candidate_experiment_id=comparison.candidate_experiment_id,
        thresholds=comparison.thresholds,
        results=comparison.results,
        created_at=comparison.created_at,
    )
    output_path = tmp_path / "reports" / "comparison.html"

    EvaluationReportExporter().export_comparison(
        comparison,
        output_path,
        format="html",
    )
    html_report = output_path.read_text(encoding="utf-8")

    assert "&lt;script&gt;baseline&lt;/script&gt;" in html_report
    assert "<script>baseline</script>" not in html_report
    assert "Regression" in html_report


def test_csv_comparison_report_includes_regression_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "comparison.csv"

    EvaluationReportExporter().export_comparison(
        _comparison_example(),
        output_path,
        format="csv",
    )
    with output_path.open(encoding="utf-8", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert rows[0]["comparison_id"] == "comparison-1"
    assert rows[0]["delta"] == "-0.3"
    assert rows[0]["direction"] == "higher_is_better"
    assert rows[0]["is_regression"] == "True"


def test_html_comparison_report_escapes_untrusted_names(tmp_path: Path) -> None:
    comparison = _comparison_example()
    comparison = ExperimentComparison(
        comparison_id=comparison.comparison_id,
        baseline_experiment_id=comparison.baseline_experiment_id,
        candidate_experiment_id="<script>candidate</script>",
        thresholds=comparison.thresholds,
        results=comparison.results,
        created_at=comparison.created_at,
    )
    output_path = tmp_path / "reports" / "comparison.html"

    EvaluationReportExporter().export_comparison(
        comparison,
        output_path,
        format="html",
    )
    report = output_path.read_text(encoding="utf-8")

    assert "&lt;script&gt;candidate&lt;/script&gt;" in report
    assert "<script>candidate</script>" not in report


def test_csv_comparison_report_neutralizes_formula_like_metric_names(
    tmp_path: Path,
) -> None:
    comparison = _comparison_example()
    original = comparison.results[0]
    comparison = ExperimentComparison(
        comparison_id=comparison.comparison_id,
        baseline_experiment_id=comparison.baseline_experiment_id,
        candidate_experiment_id=comparison.candidate_experiment_id,
        thresholds=comparison.thresholds,
        results=(
            RegressionResult(
                metric_name="=HYPERLINK(\"https://example.invalid\")",
                evaluator_id=original.evaluator_id,
                evaluator_version=original.evaluator_version,
                baseline_mean=original.baseline_mean,
                candidate_mean=original.candidate_mean,
                delta=original.delta,
                baseline_variance=original.baseline_variance,
                candidate_variance=original.candidate_variance,
                baseline_sample_count=original.baseline_sample_count,
                candidate_sample_count=original.candidate_sample_count,
                direction=original.direction,
                regression_threshold=original.regression_threshold,
                is_regression=original.is_regression,
            ),
        ),
        created_at=comparison.created_at,
    )
    output_path = tmp_path / "reports" / "comparison.csv"

    EvaluationReportExporter().export_comparison(
        comparison,
        output_path,
        format="csv",
    )
    with output_path.open(encoding="utf-8", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert rows[0]["metric_name"].startswith("'=HYPERLINK(")
