"""把已持久化的 Sparkle Evaluation 结果导出为用户报告。"""

from __future__ import annotations

import csv
import html
import io
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from sparkle_navigator.domain.evaluation import (
    AgentRunDetail,
    ExperimentComparison,
    ExperimentSnapshot,
)
from sparkle_navigator.security.redaction import sanitize_trace_payload

ReportFormat = Literal["html", "json", "csv"]


class EvaluationReportExporter:
    """将已完成的评测记录投影为可下载或留存的报告文件。"""

    def export_experiment(
        self,
        snapshot: ExperimentSnapshot,
        run_details: Sequence[AgentRunDetail],
        output_path: Path,
        *,
        format: ReportFormat,
        redacted_values: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> Path:
        """导出单个 Experiment；Ground Truth 不属于报告输入。"""

        details_by_run = {detail.run.run_id: detail for detail in run_details}
        expected_run_ids = {run.run_id for run in snapshot.runs}
        if len(details_by_run) != len(run_details) or set(details_by_run) != expected_run_ids:
            raise ValueError("Run detail 必须与 ExperimentSnapshot 中的 AgentRun 一一对应。")
        if any(
            detail.run.experiment_id != snapshot.experiment.experiment_id
            for detail in run_details
        ):
            raise ValueError("Run detail 必须属于当前 Experiment。")
        payload = _experiment_payload(snapshot, details_by_run)
        sanitized = sanitize_trace_payload(
            payload,
            secret_values=redacted_values,
        )
        if format == "json":
            content = json.dumps(
                sanitized,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        elif format == "html":
            content = _render_html_report(sanitized)
        elif format == "csv":
            content = _render_csv_report(sanitized)
        else:
            raise ValueError(f"不支持的 Evaluation 报告格式：{format}")
        return _write_report(
            output_path,
            content,
            overwrite=overwrite,
        )

    def export_comparison(
        self,
        comparison: ExperimentComparison,
        output_path: Path,
        *,
        format: ReportFormat,
        redacted_values: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> Path:
        """导出已计算的 Comparison，不重新运行 Agent 或 Evaluator。"""

        sanitized = sanitize_trace_payload(
            _comparison_payload(comparison),
            secret_values=redacted_values,
        )
        if format == "json":
            content = json.dumps(
                sanitized,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        elif format == "html":
            content = _render_comparison_html(sanitized)
        elif format == "csv":
            content = _render_comparison_csv(sanitized)
        else:
            raise ValueError(f"不支持的 Evaluation 报告格式：{format}")
        return _write_report(output_path, content, overwrite=overwrite)


def _experiment_payload(
    snapshot: ExperimentSnapshot,
    details_by_run: dict[str, AgentRunDetail],
) -> dict[str, object]:
    experiment = snapshot.experiment
    return {
        "report_version": 1,
        "report_type": "experiment",
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "suite_id": experiment.suite_id,
            "snapshot_id": experiment.snapshot_id,
            "repetitions": experiment.repetitions,
            "configuration": dict(experiment.configuration),
            "status": experiment.status.value,
            "failure_reason": experiment.failure_reason,
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
        },
        "runs": [
            _run_payload(run_detail)
            for run_detail in (
                details_by_run[run.run_id]
                for run in snapshot.runs
            )
        ],
    }


def _comparison_payload(comparison: ExperimentComparison) -> dict[str, object]:
    return {
        "report_version": 1,
        "report_type": "comparison",
        "comparison": {
            "comparison_id": comparison.comparison_id,
            "baseline_experiment_id": comparison.baseline_experiment_id,
            "candidate_experiment_id": comparison.candidate_experiment_id,
            "thresholds": dict(comparison.thresholds),
            "created_at": comparison.created_at.isoformat(),
        },
        "results": [
            {
                "metric_name": result.metric_name,
                "evaluator_id": result.evaluator_id,
                "evaluator_version": result.evaluator_version,
                "baseline_mean": result.baseline_mean,
                "candidate_mean": result.candidate_mean,
                "delta": result.delta,
                "baseline_variance": result.baseline_variance,
                "candidate_variance": result.candidate_variance,
                "baseline_sample_count": result.baseline_sample_count,
                "candidate_sample_count": result.candidate_sample_count,
                "direction": result.direction.value,
                "regression_threshold": result.regression_threshold,
                "is_regression": result.is_regression,
            }
            for result in comparison.results
        ],
    }


def _run_payload(detail: AgentRunDetail) -> dict[str, object]:
    run = detail.run
    return {
        "run_id": run.run_id,
        "case_id": run.case_id,
        "repetition": run.repetition,
        "attempt": run.attempt,
        "status": run.status.value,
        "output": run.output,
        "tau_session_id": run.tau_session_id,
        "failure_reason": run.failure_reason,
        "failure_message": run.failure_message,
        "usage": asdict(run.usage) if run.usage is not None else None,
        "timing": asdict(run.timing) if run.timing is not None else None,
        "results": [
            {
                "metric_name": result.metric_name,
                "value": result.value,
                "details": dict(result.details),
                "evaluator_id": result.evaluator_id,
                "evaluator_version": result.evaluator_version,
                "direction": result.direction.value,
            }
            for result in run.results
        ],
        "trace_events": [
            {
                "sequence": event.sequence,
                "session_id": event.session_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "payload": dict(event.payload),
                "truncated": event.truncated,
            }
            for event in detail.trace_events
        ],
        "evidence": [
            {
                "snapshot_id": item.snapshot_id,
                "path": item.path,
                "line_start": item.line_start,
                "line_end": item.line_end,
                "snippet": item.snippet,
                "sha256": item.sha256,
                "source": item.source,
                "confidence": item.confidence,
                "symbol_name": item.symbol_name,
                "relationship_type": item.relationship_type,
            }
            for item in detail.evidence
        ],
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def _render_html_report(payload: object) -> str:
    report = _mapping(payload)
    experiment = _mapping(report.get("experiment"))
    rows: list[str] = []
    for run_value in _objects(report.get("runs")):
        run = _mapping(run_value)
        metrics = "".join(
            "<tr>"
            f"<td>{_escape(metric.get('metric_name'))}</td>"
            f"<td>{_escape(metric.get('value'))}</td>"
            f"<td>{_escape(metric.get('direction'))}</td>"
            f"<td>{_escape(metric.get('evaluator_id'))} "
            f"{_escape(metric.get('evaluator_version'))}</td>"
            "</tr>"
            for metric in _objects(run.get("results"))
        )
        evidence = "".join(
            "<li>"
            f"<code>{_escape(item.get('path'))}</code>"
            f"{_line_range(item)}"
            f"<pre>{_escape(item.get('snippet'))}</pre>"
            "</li>"
            for item in _objects(run.get("evidence"))
        )
        trace = html.escape(
            json.dumps(run.get("trace_events", []), ensure_ascii=False, indent=2),
            quote=True,
        )
        rows.append(
            "<section class=\"run\">"
            f"<h2>{_escape(run.get('case_id'))} — {_escape(run.get('status'))}</h2>"
            "<dl>"
            f"<dt>Run ID</dt><dd><code>{_escape(run.get('run_id'))}</code></dd>"
            f"<dt>Repetition / Attempt</dt><dd>{_escape(run.get('repetition'))} / "
            f"{_escape(run.get('attempt'))}</dd>"
            "</dl>"
            f"<pre class=\"output\">{_escape(run.get('output'))}</pre>"
            f"<p class=\"failure\">{_escape(run.get('failure_message'))}</p>"
            "<h3>Metrics</h3><table><thead><tr><th>Metric</th><th>Value</th>"
            "<th>Direction</th><th>Evaluator</th></tr></thead>"
            f"<tbody>{metrics}</tbody></table>"
            f"<h3>Code evidence</h3><ul>{evidence}</ul>"
            f"<details><summary>Trace events</summary><pre>{trace}</pre></details>"
            "</section>"
        )
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>Evaluation — {_escape(experiment.get('suite_id'))}</title>"
        "<style>body{font:15px system-ui;margin:2rem auto;max-width:1000px;padding:0 1rem;"
        "color:#20242a}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;"
        "padding:.4rem;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f4f5f7;padding:.75rem}.run{border-top:2px solid #888;margin-top:2rem}"
        ".failure:empty{display:none}</style></head><body>"
        f"<h1>Evaluation report</h1><p>Experiment: "
        f"<code>{_escape(experiment.get('experiment_id'))}</code></p>"
        f"<p>Suite: {_escape(experiment.get('suite_id'))} | "
        f"Snapshot: {_escape(experiment.get('snapshot_id'))} | "
        f"Status: {_escape(experiment.get('status'))}</p>"
        + "".join(rows)
        + "</body></html>"
    )


def _render_csv_report(payload: object) -> str:
    report = _mapping(payload)
    experiment = _mapping(report.get("experiment"))
    fields = (
        "experiment_id",
        "suite_id",
        "snapshot_id",
        "run_id",
        "case_id",
        "repetition",
        "attempt",
        "status",
        "output",
        "failure_reason",
        "failure_message",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "run_duration_ms",
        "provider_duration_ms",
        "time_to_first_output_ms",
        "metric_name",
        "metric_value",
        "metric_direction",
        "evaluator_id",
        "evaluator_version",
        "metric_details_json",
        "trace_events_json",
        "evidence_json",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for run_value in _objects(report.get("runs")):
        run = _mapping(run_value)
        usage = _mapping(run.get("usage"))
        timing = _mapping(run.get("timing"))
        results = _objects(run.get("results")) or ({},)
        for result_value in results:
            result = _mapping(result_value)
            row = {
                "experiment_id": experiment.get("experiment_id"),
                "suite_id": experiment.get("suite_id"),
                "snapshot_id": experiment.get("snapshot_id"),
                "run_id": run.get("run_id"),
                "case_id": run.get("case_id"),
                "repetition": run.get("repetition"),
                "attempt": run.get("attempt"),
                "status": run.get("status"),
                "output": run.get("output"),
                "failure_reason": run.get("failure_reason"),
                "failure_message": run.get("failure_message"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "run_duration_ms": timing.get("run_duration_ms"),
                "provider_duration_ms": timing.get("provider_duration_ms"),
                "time_to_first_output_ms": timing.get("time_to_first_output_ms"),
                "metric_name": result.get("metric_name"),
                "metric_value": result.get("value"),
                "metric_direction": result.get("direction"),
                "evaluator_id": result.get("evaluator_id"),
                "evaluator_version": result.get("evaluator_version"),
                "metric_details_json": _json_cell(result.get("details", {})),
                "trace_events_json": _json_cell(run.get("trace_events", [])),
                "evidence_json": _json_cell(run.get("evidence", [])),
            }
            writer.writerow({key: _spreadsheet_safe(value) for key, value in row.items()})
    return output.getvalue()


def _render_comparison_html(payload: object) -> str:
    report = _mapping(payload)
    comparison = _mapping(report.get("comparison"))
    rows = "".join(
        "<tr>"
        f"<td>{_escape(result.get('metric_name'))}</td>"
        f"<td>{_escape(result.get('direction'))}</td>"
        f"<td>{_escape(result.get('baseline_mean'))}</td>"
        f"<td>{_escape(result.get('candidate_mean'))}</td>"
        f"<td>{_escape(result.get('delta'))}</td>"
        f"<td>{_escape(result.get('regression_threshold'))}</td>"
        f"<td>{'Regression' if result.get('is_regression') else 'No regression'}</td>"
        "</tr>"
        for result in _objects(report.get("results"))
    )
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>Evaluation comparison</title>"
        "<style>body{font:15px system-ui;margin:2rem auto;max-width:1000px;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.4rem;"
        "text-align:left}</style></head><body><h1>Evaluation comparison</h1>"
        f"<p>Baseline: <code>{_escape(comparison.get('baseline_experiment_id'))}</code></p>"
        f"<p>Candidate: <code>{_escape(comparison.get('candidate_experiment_id'))}</code></p>"
        "<table><thead><tr><th>Metric</th><th>Direction</th><th>Baseline mean</th>"
        "<th>Candidate mean</th><th>Delta</th><th>Threshold</th><th>Regression</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></body></html>"
    )


def _render_comparison_csv(payload: object) -> str:
    report = _mapping(payload)
    comparison = _mapping(report.get("comparison"))
    fields = (
        "comparison_id",
        "baseline_experiment_id",
        "candidate_experiment_id",
        "metric_name",
        "evaluator_id",
        "evaluator_version",
        "direction",
        "baseline_mean",
        "candidate_mean",
        "delta",
        "baseline_variance",
        "candidate_variance",
        "baseline_sample_count",
        "candidate_sample_count",
        "regression_threshold",
        "is_regression",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for result in _objects(report.get("results")):
        row = {
            "comparison_id": comparison.get("comparison_id"),
            "baseline_experiment_id": comparison.get("baseline_experiment_id"),
            "candidate_experiment_id": comparison.get("candidate_experiment_id"),
            **{field: result.get(field) for field in fields[3:]},
        }
        writer.writerow({key: _spreadsheet_safe(value) for key, value in row.items()})
    return output.getvalue()


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_mapping(item) for item in value if isinstance(item, dict))


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _spreadsheet_safe(value: object) -> object:
    if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _line_range(evidence: dict[str, object]) -> str:
    start = evidence.get("line_start")
    end = evidence.get("line_end")
    if start is None:
        return ""
    return f":{_escape(start)}" if end in {None, start} else f":{_escape(start)}-{_escape(end)}"


def _write_report(
    output_path: Path,
    content: str,
    *,
    overwrite: bool,
) -> Path:
    destination = output_path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"报告文件已存在：{destination}")
    destination.write_text(content, encoding="utf-8", newline="\n")
    return destination


__all__ = ["EvaluationReportExporter", "ReportFormat"]
