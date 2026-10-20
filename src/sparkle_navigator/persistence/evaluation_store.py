"""以 SQLite 持久化 Evaluation Experiment、AgentRun、Event 与结果。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sparkle_navigator.domain.evaluation import (
    AgentRunDetail,
    AgentRunOutput,
    AgentRunRecord,
    AgentRunStatus,
    AgentRunTiming,
    AgentRunUsage,
    EvaluationCase,
    EvaluationEvent,
    EvaluationRequest,
    EvaluationResult,
    ExperimentComparison,
    ExperimentRecord,
    ExperimentSnapshot,
    ExperimentStatus,
    RegressionResult,
    MetricDirection,
    RunTraceEvent,
)
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.persistence.database import open_sqlite_database
from sparkle_navigator.persistence.models import (
    EvaluationAgentRunRow,
    EvaluationCaseRow,
    EvaluationComparisonRow,
    EvaluationEventRow,
    EvaluationExperimentRow,
    EvaluationResultRow,
    EvaluationRegressionResultRow,
    EvaluationRunEvidenceRow,
    EvaluationRunTraceEventRow,
)
from sparkle_navigator.security.redaction import (
    sanitize_trace_payload,
    sanitize_trace_text,
)


class EvaluationStoreError(RuntimeError):
    """Evaluation Core 持久化或状态迁移失败。"""


_EXPERIMENT_TRANSITIONS = {
    ExperimentStatus.DRAFT: {ExperimentStatus.QUEUED, ExperimentStatus.CANCELLED},
    ExperimentStatus.QUEUED: {
        ExperimentStatus.RUNNING,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
    },
    ExperimentStatus.RUNNING: {
        ExperimentStatus.EVALUATING,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
    },
    ExperimentStatus.EVALUATING: {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
    },
}
_RUN_TRANSITIONS = {
    AgentRunStatus.CREATED: {
        AgentRunStatus.QUEUED,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.QUEUED: {
        AgentRunStatus.RUNNING,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.RUNNING: {
        AgentRunStatus.EVALUATING,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
    },
    AgentRunStatus.EVALUATING: {
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.CANCELLED,
    },
}


class EvaluationStore:
    """封装 SQLite 事务；每个 Store 实例内的写事务串行提交。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self._engine = open_sqlite_database(self.database_path)
        self._write_lock = Lock()

    def create_experiment(self, request: EvaluationRequest) -> str:
        experiment_id = str(uuid4())
        now = datetime.now(UTC)
        with self._write_lock, Session(self._engine) as session, session.begin():
            experiment = EvaluationExperimentRow(
                experiment_id=experiment_id,
                suite_id=request.suite_id,
                snapshot_id=request.snapshot_id,
                repetitions=request.repetitions,
                configuration_json=_json_dumps(request.configuration),
                status=ExperimentStatus.DRAFT.value,
                created_at=now,
                updated_at=now,
            )
            session.add(experiment)
            session.flush()
            for ordinal, case in enumerate(request.cases):
                session.add(
                    EvaluationCaseRow(
                        experiment_id=experiment_id,
                        case_id=case.case_id,
                        ordinal=ordinal,
                        user_input=case.user_input,
                        ground_truth_json=_json_dumps(case.ground_truth),
                    )
                )
            session.flush()
            _append_event(
                session,
                experiment_id,
                "experiment.created",
                payload={
                    "suite_id": request.suite_id,
                    "repetitions": request.repetitions,
                },
            )
            for case in request.cases:
                for repetition in range(1, request.repetitions + 1):
                    run_id = str(uuid4())
                    session.add(
                        EvaluationAgentRunRow(
                            run_id=run_id,
                            experiment_id=experiment_id,
                            case_id=case.case_id,
                            repetition=repetition,
                            attempt=1,
                            status=AgentRunStatus.CREATED.value,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    session.flush()
                    _append_event(
                        session,
                        experiment_id,
                        "agent_run.created",
                        run_id=run_id,
                        payload={
                            "case_id": case.case_id,
                            "repetition": repetition,
                        },
                    )
            self._transition_experiment(
                session,
                experiment,
                ExperimentStatus.QUEUED,
            )
        return experiment_id

    def list_experiments(self, *, limit: int = 50) -> tuple[ExperimentRecord, ...]:
        """按最近更新时间读取 Experiment 列表，不启动或恢复运行。"""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Experiment 列表 limit 必须是正整数。")
        with Session(self._engine) as session:
            rows = session.execute(
                select(EvaluationExperimentRow)
                .order_by(
                    EvaluationExperimentRow.updated_at.desc(),
                    EvaluationExperimentRow.experiment_id,
                )
                .limit(limit)
            ).scalars().all()
        return tuple(
            ExperimentRecord(
                experiment_id=row.experiment_id,
                suite_id=row.suite_id,
                snapshot_id=row.snapshot_id,
                repetitions=row.repetitions,
                configuration=_json_object(row.configuration_json),
                status=ExperimentStatus(row.status),
                failure_reason=row.failure_reason,
                created_at=_aware_datetime(row.created_at),
                updated_at=_aware_datetime(row.updated_at),
            )
            for row in rows
        )

    def get_request(self, experiment_id: str) -> EvaluationRequest:
        with Session(self._engine) as session:
            experiment = session.get(EvaluationExperimentRow, experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment 不存在：{experiment_id}")
            rows = session.execute(
                select(EvaluationCaseRow)
                .where(EvaluationCaseRow.experiment_id == experiment_id)
                .order_by(EvaluationCaseRow.ordinal)
            ).scalars()
            cases = tuple(
                EvaluationCase(
                    case_id=row.case_id,
                    user_input=row.user_input,
                    ground_truth=_json_object(row.ground_truth_json),
                )
                for row in rows
            )
            return EvaluationRequest(
                suite_id=experiment.suite_id,
                snapshot_id=experiment.snapshot_id,
                cases=cases,
                repetitions=experiment.repetitions,
                configuration=_json_object(experiment.configuration_json),
            )

    def get_snapshot(self, experiment_id: str) -> ExperimentSnapshot:
        with Session(self._engine) as session:
            experiment = session.get(EvaluationExperimentRow, experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment 不存在：{experiment_id}")
            run_rows = session.execute(
                select(EvaluationAgentRunRow)
                .join(
                    EvaluationCaseRow,
                    (EvaluationAgentRunRow.experiment_id == EvaluationCaseRow.experiment_id)
                    & (EvaluationAgentRunRow.case_id == EvaluationCaseRow.case_id),
                )
                .where(EvaluationAgentRunRow.experiment_id == experiment_id)
                .order_by(
                    EvaluationCaseRow.ordinal,
                    EvaluationAgentRunRow.repetition,
                    EvaluationAgentRunRow.attempt,
                )
            ).scalars()
            runs: list[AgentRunRecord] = []
            for row in run_rows:
                result_rows = session.execute(
                    select(EvaluationResultRow)
                    .where(EvaluationResultRow.run_id == row.run_id)
                    .order_by(EvaluationResultRow.metric_name)
                ).scalars()
                results = tuple(
                    EvaluationResult(
                        metric_name=result.metric_name,
                        value=result.value,
                        details=_json_object(result.details_json),
                        evaluator_id=result.evaluator_id,
                        evaluator_version=result.evaluator_version,
                        direction=MetricDirection(result.direction),
                    )
                    for result in result_rows
                )
                runs.append(
                    AgentRunRecord(
                        run_id=row.run_id,
                        experiment_id=row.experiment_id,
                        case_id=row.case_id,
                        repetition=row.repetition,
                        attempt=row.attempt,
                        status=AgentRunStatus(row.status),
                        output=row.output,
                        tau_session_id=row.tau_session_id,
                        failure_reason=row.failure_reason,
                        failure_message=row.failure_message,
                        usage=_run_usage_from_json(row.usage_json),
                        timing=_run_timing_from_json(row.timing_json),
                        results=results,
                        created_at=_aware_datetime(row.created_at),
                        updated_at=_aware_datetime(row.updated_at),
                    )
                )
            record = ExperimentRecord(
                experiment_id=experiment.experiment_id,
                suite_id=experiment.suite_id,
                snapshot_id=experiment.snapshot_id,
                repetitions=experiment.repetitions,
                configuration=_json_object(experiment.configuration_json),
                status=ExperimentStatus(experiment.status),
                failure_reason=experiment.failure_reason,
                created_at=_aware_datetime(experiment.created_at),
                updated_at=_aware_datetime(experiment.updated_at),
            )
        return ExperimentSnapshot(experiment=record, runs=tuple(runs))

    def recover_interrupted_experiment(self, experiment_id: str) -> bool:
        """把未完成运行标记中断，并为未完成 Case 创建新 attempt。"""

        with self._write_lock, Session(self._engine) as session, session.begin():
            experiment = session.get(EvaluationExperimentRow, experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment 不存在：{experiment_id}")
            if ExperimentStatus(experiment.status).is_terminal:
                return False

            runs = session.execute(
                select(EvaluationAgentRunRow)
                .join(
                    EvaluationCaseRow,
                    (EvaluationAgentRunRow.experiment_id == EvaluationCaseRow.experiment_id)
                    & (EvaluationAgentRunRow.case_id == EvaluationCaseRow.case_id),
                )
                .where(EvaluationAgentRunRow.experiment_id == experiment_id)
                .order_by(
                    EvaluationCaseRow.ordinal,
                    EvaluationAgentRunRow.repetition,
                    EvaluationAgentRunRow.attempt,
                )
            ).scalars().all()
            for run in runs:
                if not AgentRunStatus(run.status).is_terminal:
                    self._transition_run(
                        session,
                        run,
                        AgentRunStatus.FAILED,
                        failure_reason="process_interrupted",
                    )

            latest_by_case_repetition: dict[tuple[str, int], EvaluationAgentRunRow] = {}
            for run in runs:
                latest_by_case_repetition[(run.case_id, run.repetition)] = run
            cases = session.execute(
                select(EvaluationCaseRow)
                .where(EvaluationCaseRow.experiment_id == experiment_id)
                .order_by(EvaluationCaseRow.ordinal)
            ).scalars().all()
            for case in cases:
                for repetition in range(1, experiment.repetitions + 1):
                    latest_run = latest_by_case_repetition.get(
                        (case.case_id, repetition)
                    )
                    if (
                        latest_run is not None
                        and AgentRunStatus(latest_run.status)
                        is AgentRunStatus.COMPLETED
                    ):
                        continue
                    attempt = latest_run.attempt + 1 if latest_run is not None else 1
                    run_id = str(uuid4())
                    run = EvaluationAgentRunRow(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        case_id=case.case_id,
                        repetition=repetition,
                        attempt=attempt,
                        status=AgentRunStatus.CREATED.value,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                    session.add(run)
                    session.flush()
                    _append_event(
                        session,
                        experiment_id,
                        "agent_run.created",
                        run_id=run_id,
                        payload={
                            "case_id": case.case_id,
                            "repetition": repetition,
                            "attempt": run.attempt,
                        },
                    )

            experiment.status = ExperimentStatus.QUEUED.value
            experiment.failure_reason = None
            experiment.updated_at = datetime.now(UTC)
            _append_event(
                session,
                experiment_id,
                "experiment.recovered",
                payload={"resumed_at_case_boundary": True},
            )
            return True

    def get_events(self, experiment_id: str, *, after_sequence: int) -> tuple[EvaluationEvent, ...]:
        with Session(self._engine) as session:
            if session.get(EvaluationExperimentRow, experiment_id) is None:
                raise KeyError(f"Experiment 不存在：{experiment_id}")
            rows = session.execute(
                select(EvaluationEventRow)
                .where(
                    EvaluationEventRow.experiment_id == experiment_id,
                    EvaluationEventRow.sequence > after_sequence,
                )
                .order_by(EvaluationEventRow.sequence)
            ).scalars()
            return tuple(
                EvaluationEvent(
                    experiment_id=row.experiment_id,
                    sequence=row.sequence,
                    event_type=row.event_type,
                    run_id=row.run_id,
                    payload=_json_object(row.payload_json),
                    occurred_at=_aware_datetime(row.occurred_at),
                )
                for row in rows
            )

    def transition_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        failure_reason: str | None = None,
    ) -> None:
        with self._write_lock, Session(self._engine) as session, session.begin():
            experiment = session.get(EvaluationExperimentRow, experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment 不存在：{experiment_id}")
            self._transition_experiment(
                session,
                experiment,
                status,
                failure_reason=failure_reason,
            )

    def transition_run(
        self,
        run_id: str,
        status: AgentRunStatus,
        *,
        failure_reason: str | None = None,
    ) -> None:
        with self._write_lock, Session(self._engine) as session, session.begin():
            run = session.get(EvaluationAgentRunRow, run_id)
            if run is None:
                raise KeyError(f"AgentRun 不存在：{run_id}")
            self._transition_run(
                session,
                run,
                status,
                failure_reason=failure_reason,
            )

    def complete_run(
        self,
        run_id: str,
        *,
        output: AgentRunOutput,
        results: tuple[EvaluationResult, ...],
    ) -> None:
        if output.failure is not None:
            raise EvaluationStoreError("失败的 AgentRun 必须使用 fail_run() 保存。")
        metric_versions = tuple(
            (result.metric_name, result.evaluator_id, result.evaluator_version)
            for result in results
        )
        if len(metric_versions) != len(set(metric_versions)):
            raise EvaluationStoreError("同一 AgentRun 的指标与 Evaluator 版本不能重复。")
        with self._write_lock, Session(self._engine) as session, session.begin():
            run = session.get(EvaluationAgentRunRow, run_id)
            if run is None:
                raise KeyError(f"AgentRun 不存在：{run_id}")
            if AgentRunStatus(run.status) is not AgentRunStatus.EVALUATING:
                raise EvaluationStoreError("只有 EVALUATING 状态的 AgentRun 才能保存结果。")
            _persist_run_output(session, run, output)
            session.add_all(
                EvaluationResultRow(
                    result_id=str(uuid4()),
                    run_id=run_id,
                    metric_name=result.metric_name,
                    evaluator_id=result.evaluator_id,
                    evaluator_version=result.evaluator_version,
                    direction=result.direction.value,
                    value=result.value,
                    details_json=_json_dumps(result.details),
                )
                for result in results
            )
            self._transition_run(session, run, AgentRunStatus.COMPLETED)

    def fail_run(self, run_id: str, *, output: AgentRunOutput) -> None:
        if output.failure is None:
            raise EvaluationStoreError("失败的 AgentRun 必须提供 AgentRunFailure。")
        with self._write_lock, Session(self._engine) as session, session.begin():
            run = session.get(EvaluationAgentRunRow, run_id)
            if run is None:
                raise KeyError(f"AgentRun 不存在：{run_id}")
            _persist_run_output(session, run, output)
            self._transition_run(
                session,
                run,
                AgentRunStatus.FAILED,
                failure_reason=output.failure.code,
                failure_message=output.failure.message,
            )

    def cancel_run(self, run_id: str, *, output: AgentRunOutput) -> None:
        """保存部分观测后把当前 Run 标记为 CANCELLED。"""

        if output.failure is not None:
            raise EvaluationStoreError("取消的 AgentRun 不能同时带失败摘要。")
        with self._write_lock, Session(self._engine) as session, session.begin():
            run = session.get(EvaluationAgentRunRow, run_id)
            if run is None:
                raise KeyError(f"AgentRun 不存在：{run_id}")
            if AgentRunStatus(run.status).is_terminal:
                return
            _persist_run_output(session, run, output)
            self._transition_run(session, run, AgentRunStatus.CANCELLED)

    def get_run_detail(self, experiment_id: str, run_id: str) -> AgentRunDetail:
        snapshot = self.get_snapshot(experiment_id)
        run = next((item for item in snapshot.runs if item.run_id == run_id), None)
        if run is None:
            raise KeyError(f"AgentRun 不属于指定 Experiment：{run_id}")
        with Session(self._engine) as session:
            trace_rows = session.execute(
                select(EvaluationRunTraceEventRow)
                .where(EvaluationRunTraceEventRow.run_id == run_id)
                .order_by(EvaluationRunTraceEventRow.sequence)
            ).scalars().all()
            evidence_rows = session.execute(
                select(EvaluationRunEvidenceRow)
                .where(EvaluationRunEvidenceRow.run_id == run_id)
                .order_by(EvaluationRunEvidenceRow.ordinal)
            ).scalars().all()
        trace_events = tuple(
            RunTraceEvent(
                sequence=row.sequence,
                session_id=row.session_id,
                event_type=row.event_type,
                timestamp=row.timestamp,
                payload=_json_object(row.payload_json),
                truncated=row.truncated,
            )
            for row in trace_rows
        )
        evidence = tuple(
            CodeEvidence(
                snapshot_id=row.snapshot_id,
                path=row.path,
                line_start=row.line_start,
                line_end=row.line_end,
                snippet=row.snippet,
                sha256=row.sha256,
                source=row.source,
                confidence=row.confidence,
                symbol_name=row.symbol_name,
                relationship_type=row.relationship_type,
            )
            for row in evidence_rows
        )
        return AgentRunDetail(run=run, trace_events=trace_events, evidence=evidence)

    def save_comparison(self, comparison: ExperimentComparison) -> None:
        metric_keys = tuple(
            (
                result.metric_name,
                result.evaluator_id,
                result.evaluator_version,
            )
            for result in comparison.results
        )
        if len(metric_keys) != len(set(metric_keys)):
            raise EvaluationStoreError("Comparison 中的指标与 Evaluator 版本不能重复。")
        with self._write_lock, Session(self._engine) as session, session.begin():
            if session.get(EvaluationComparisonRow, comparison.comparison_id) is not None:
                raise EvaluationStoreError("Comparison ID 已存在。")
            session.add(
                EvaluationComparisonRow(
                    comparison_id=comparison.comparison_id,
                    baseline_experiment_id=comparison.baseline_experiment_id,
                    candidate_experiment_id=comparison.candidate_experiment_id,
                    thresholds_json=_json_dumps(comparison.thresholds),
                    created_at=comparison.created_at,
                )
            )
            session.add_all(
                EvaluationRegressionResultRow(
                    comparison_id=comparison.comparison_id,
                    metric_name=result.metric_name,
                    evaluator_id=result.evaluator_id,
                    evaluator_version=result.evaluator_version,
                    baseline_mean=result.baseline_mean,
                    candidate_mean=result.candidate_mean,
                    delta=result.delta,
                    baseline_variance=result.baseline_variance,
                    candidate_variance=result.candidate_variance,
                    baseline_sample_count=result.baseline_sample_count,
                    candidate_sample_count=result.candidate_sample_count,
                    direction=result.direction.value,
                    regression_threshold=result.regression_threshold,
                    is_regression=result.is_regression,
                )
                for result in comparison.results
            )

    def get_comparison(self, comparison_id: str) -> ExperimentComparison:
        with Session(self._engine) as session:
            row = session.get(EvaluationComparisonRow, comparison_id)
            if row is None:
                raise KeyError(f"ExperimentComparison 不存在：{comparison_id}")
            regression_rows = session.execute(
                select(EvaluationRegressionResultRow)
                .where(EvaluationRegressionResultRow.comparison_id == comparison_id)
                .order_by(
                    EvaluationRegressionResultRow.metric_name,
                    EvaluationRegressionResultRow.evaluator_id,
                    EvaluationRegressionResultRow.evaluator_version,
                )
            ).scalars().all()
        return ExperimentComparison(
            comparison_id=row.comparison_id,
            baseline_experiment_id=row.baseline_experiment_id,
            candidate_experiment_id=row.candidate_experiment_id,
            thresholds={
                key: float(value)
                for key, value in _json_object(row.thresholds_json).items()
            },
            results=tuple(
                RegressionResult(
                    metric_name=result.metric_name,
                    evaluator_id=result.evaluator_id,
                    evaluator_version=result.evaluator_version,
                    baseline_mean=result.baseline_mean,
                    candidate_mean=result.candidate_mean,
                    delta=result.delta,
                    baseline_variance=result.baseline_variance,
                    candidate_variance=result.candidate_variance,
                    baseline_sample_count=result.baseline_sample_count,
                    candidate_sample_count=result.candidate_sample_count,
                    direction=MetricDirection(result.direction),
                    regression_threshold=result.regression_threshold,
                    is_regression=result.is_regression,
                )
                for result in regression_rows
            ),
            created_at=_aware_datetime(row.created_at),
        )

    def list_comparisons(self, *, limit: int = 50) -> tuple[ExperimentComparison, ...]:
        """按创建时间倒序列出已保存的 ExperimentComparison。"""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Comparison 列表 limit 必须是正整数。")
        with Session(self._engine) as session:
            comparison_ids = session.execute(
                select(EvaluationComparisonRow.comparison_id)
                .order_by(
                    EvaluationComparisonRow.created_at.desc(),
                    EvaluationComparisonRow.comparison_id,
                )
                .limit(limit)
            ).scalars().all()
        return tuple(self.get_comparison(item) for item in comparison_ids)

    def cancel_experiment(self, experiment_id: str) -> None:
        with self._write_lock, Session(self._engine) as session, session.begin():
            experiment = session.get(EvaluationExperimentRow, experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment 不存在：{experiment_id}")
            if ExperimentStatus(experiment.status).is_terminal:
                return
            runs = session.execute(
                select(EvaluationAgentRunRow).where(
                    EvaluationAgentRunRow.experiment_id == experiment_id
                )
            ).scalars()
            for run in runs:
                if not AgentRunStatus(run.status).is_terminal:
                    self._transition_run(session, run, AgentRunStatus.CANCELLED)
            self._transition_experiment(
                session,
                experiment,
                ExperimentStatus.CANCELLED,
            )

    def dispose(self) -> None:
        self._engine.dispose()

    def _transition_experiment(
        self,
        session: Session,
        experiment: EvaluationExperimentRow,
        status: ExperimentStatus,
        *,
        failure_reason: str | None = None,
    ) -> None:
        current = ExperimentStatus(experiment.status)
        if status not in _EXPERIMENT_TRANSITIONS.get(current, set()):
            raise EvaluationStoreError(
                f"Experiment 状态不能从 {current.value} 转为 {status.value}。"
            )
        experiment.status = status.value
        experiment.failure_reason = failure_reason
        experiment.updated_at = datetime.now(UTC)
        _append_event(
            session,
            experiment.experiment_id,
            "experiment.status_changed",
            payload={"status": status.value, "failure_reason": failure_reason},
        )

    def _transition_run(
        self,
        session: Session,
        run: EvaluationAgentRunRow,
        status: AgentRunStatus,
        *,
        failure_reason: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        current = AgentRunStatus(run.status)
        if status not in _RUN_TRANSITIONS.get(current, set()):
            raise EvaluationStoreError(
                f"AgentRun 状态不能从 {current.value} 转为 {status.value}。"
            )
        run.status = status.value
        run.failure_reason = failure_reason
        run.failure_message = failure_message
        run.updated_at = datetime.now(UTC)
        _append_event(
            session,
            run.experiment_id,
            "agent_run.status_changed",
            run_id=run.run_id,
            payload={
                "case_id": run.case_id,
                "status": status.value,
                "failure_reason": failure_reason,
            },
        )


def _append_event(
    session: Session,
    experiment_id: str,
    event_type: str,
    *,
    run_id: str | None = None,
    payload: dict[str, object],
) -> None:
    latest_sequence = session.scalar(
        select(func.max(EvaluationEventRow.sequence)).where(
            EvaluationEventRow.experiment_id == experiment_id
        )
    )
    session.add(
        EvaluationEventRow(
            experiment_id=experiment_id,
            sequence=(latest_sequence or 0) + 1,
            event_type=event_type,
            run_id=run_id,
            payload_json=_json_dumps(payload),
            occurred_at=datetime.now(UTC),
        )
    )


def _persist_run_output(
    session: Session,
    run: EvaluationAgentRunRow,
    output: AgentRunOutput,
) -> None:
    experiment = session.get(EvaluationExperimentRow, run.experiment_id)
    if experiment is None:
        raise EvaluationStoreError("AgentRun 关联的 Experiment 不存在。")
    trace_sequences = tuple(event.sequence for event in output.trace_events)
    if trace_sequences != tuple(range(1, len(trace_sequences) + 1)):
        raise EvaluationStoreError("RunTraceEvent 必须从 1 开始连续递增。")
    if output.tau_session_id is not None and any(
        event.session_id not in {None, output.tau_session_id}
        for event in output.trace_events
    ):
        raise EvaluationStoreError("TraceEvent 的 Tau Session ID 与 AgentRun 不一致。")
    if any(
        experiment.snapshot_id is None
        or evidence.snapshot_id != experiment.snapshot_id
        for evidence in output.evidence
    ):
        raise EvaluationStoreError("AgentRun Evidence 必须绑定当前 Experiment Snapshot。")

    run.output = sanitize_trace_text(output.output)
    run.tau_session_id = output.tau_session_id
    run.usage_json = (
        _json_dumps(asdict(output.usage)) if output.usage is not None else None
    )
    run.timing_json = (
        _json_dumps(asdict(output.timing)) if output.timing is not None else None
    )
    run.failure_reason = output.failure.code if output.failure is not None else None
    run.failure_message = (
        sanitize_trace_text(output.failure.message)
        if output.failure is not None
        else None
    )
    session.add_all(
        EvaluationRunTraceEventRow(
            run_id=run.run_id,
            sequence=event.sequence,
            session_id=event.session_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            payload_json=_json_dumps(sanitize_trace_payload(event.payload)),
            truncated=event.truncated,
        )
        for event in output.trace_events
    )
    session.add_all(
        EvaluationRunEvidenceRow(
            run_id=run.run_id,
            ordinal=ordinal,
            snapshot_id=evidence.snapshot_id,
            path=evidence.path,
            line_start=evidence.line_start,
            line_end=evidence.line_end,
            snippet=sanitize_trace_text(evidence.snippet),
            sha256=evidence.sha256,
            source=evidence.source,
            confidence=evidence.confidence,
            symbol_name=evidence.symbol_name,
            relationship_type=evidence.relationship_type,
        )
        for ordinal, evidence in enumerate(output.evidence)
    )


def _json_dumps(value: object) -> str:
    try:
        if isinstance(value, Mapping):
            value = dict(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise EvaluationStoreError("评测记录只能保存 JSON 可序列化数据。") from exc


def _json_object(value: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvaluationStoreError("SQLite 中的评测 JSON 数据损坏。") from exc
    if not isinstance(decoded, dict):
        raise EvaluationStoreError("SQLite 中的评测 JSON 记录必须是 object。")
    return decoded


def _aware_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _run_usage_from_json(value: str | None) -> AgentRunUsage | None:
    if value is None:
        return None
    try:
        return AgentRunUsage(**_json_object(value))
    except TypeError as exc:
        raise EvaluationStoreError("SQLite 中的 Run Usage 格式无效。") from exc


def _run_timing_from_json(value: str | None) -> AgentRunTiming | None:
    if value is None:
        return None
    try:
        return AgentRunTiming(**_json_object(value))
    except TypeError as exc:
        raise EvaluationStoreError("SQLite 中的 Run Timing 格式无效。") from exc


__all__ = ["EvaluationStore", "EvaluationStoreError"]
