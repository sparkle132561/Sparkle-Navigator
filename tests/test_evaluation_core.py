"""EvaluationLab 公开接口的 SQLite 与状态机契约测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunOutput,
    AgentRunStatus,
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.evaluation.core import EvaluationLab, EventCursor
from sparkle_navigator.evaluation.smoke_set import EvidencePathEvaluator


def test_evaluation_lab_persists_smoke_run_and_resumes_event_cursor(
    tmp_path: Path,
) -> None:
    asyncio.run(_evaluation_lab_persists_smoke_run_and_resumes_event_cursor(tmp_path))


async def _evaluation_lab_persists_smoke_run_and_resumes_event_cursor(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(
        AgentRunOutput(
            output="The handler is in the expected source file.",
            evidence_paths=("src/handler.py", "src/other.py"),
        )
    )
    database_path = tmp_path / "evaluation.sqlite3"
    lab = EvaluationLab(
        database_path=database_path,
        runner=runner,
        evaluator=EvidencePathEvaluator(),
    )
    request = EvaluationRequest(
        suite_id="retrieval-smoke",
        snapshot_id=None,
        configuration={"retriever": "fts5"},
        cases=(
            EvaluationCase(
                case_id="find-handler",
                user_input="Which file contains the handler?",
                ground_truth={"expected_paths": ["src/handler.py"]},
            ),
        ),
    )

    handle = await lab.evaluate(request)
    events = [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.COMPLETED
    assert len(snapshot.runs) == 1
    run = snapshot.runs[0]
    assert run.status is AgentRunStatus.COMPLETED
    assert run.output == "The handler is in the expected source file."
    assert {result.metric_name: result.value for result in run.results} == {
        "evidence_precision": 0.5,
        "evidence_recall": 1.0,
    }
    assert {result.evaluator_version for result in run.results} == {"1"}
    assert "expected_paths" not in repr(runner.inputs[0])
    assert runner.inputs[0].user_input == request.cases[0].user_input
    assert [event.sequence for event in events] == list(
        range(1, len(events) + 1)
    )
    assert events[-1].event_type == "experiment.status_changed"
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value

    cursor = EventCursor(
        experiment_id=snapshot.experiment.experiment_id,
        sequence=events[0].sequence,
    )
    remaining_events = [event async for event in handle.events(after=cursor)]
    assert [event.sequence for event in remaining_events] == [
        event.sequence for event in events[1:]
    ]

    reopened_lab = EvaluationLab(
        database_path=database_path,
        runner=runner,
        evaluator=EvidencePathEvaluator(),
    )
    reopened_handle = await reopened_lab.open(snapshot.experiment.experiment_id)
    reopened_snapshot = await reopened_handle.snapshot()
    assert reopened_snapshot == snapshot
    assert len(runner.inputs) == 1


def test_agent_run_failure_is_persisted_without_raw_exception_text(
    tmp_path: Path,
) -> None:
    asyncio.run(_agent_run_failure_is_persisted_without_raw_exception_text(tmp_path))


async def _agent_run_failure_is_persisted_without_raw_exception_text(
    tmp_path: Path,
) -> None:
    lab = EvaluationLab(
        database_path=tmp_path / "failed-evaluation.sqlite3",
        runner=_FailingRunner(),
        evaluator=EvidencePathEvaluator(),
    )
    handle = await lab.evaluate(_one_case_request())
    events = [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.FAILED
    assert snapshot.experiment.failure_reason == "one_or_more_agent_runs_failed"
    assert snapshot.runs[0].status is AgentRunStatus.FAILED
    assert snapshot.runs[0].failure_reason == "RuntimeError"
    assert all("sensitive exception detail" not in repr(event) for event in events)
    assert events[-1].payload["status"] == ExperimentStatus.FAILED.value


def test_repetitions_create_independent_runs_not_recovery_attempts(
    tmp_path: Path,
) -> None:
    asyncio.run(_repetitions_create_independent_runs_not_recovery_attempts(tmp_path))


async def _repetitions_create_independent_runs_not_recovery_attempts(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(
        AgentRunOutput(
            output="repeatable output",
            evidence_paths=("src/example.py",),
        )
    )
    lab = EvaluationLab(
        database_path=tmp_path / "repeated-evaluation.sqlite3",
        runner=runner,
        evaluator=EvidencePathEvaluator(),
    )
    request = _one_case_request()
    request = EvaluationRequest(
        suite_id=request.suite_id,
        snapshot_id=request.snapshot_id,
        cases=request.cases,
        repetitions=3,
        configuration=request.configuration,
    )

    handle = await lab.evaluate(request)
    [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert snapshot.experiment.repetitions == 3
    assert [run.repetition for run in snapshot.runs] == [1, 2, 3]
    assert [run.attempt for run in snapshot.runs] == [1, 1, 1]
    assert all(run.status is AgentRunStatus.COMPLETED for run in snapshot.runs)
    assert [agent_input.repetition for agent_input in runner.inputs] == [1, 2, 3]


def test_mismatched_runner_evidence_fails_run_without_stalling_experiment(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _mismatched_runner_evidence_fails_run_without_stalling_experiment(tmp_path)
    )


async def _mismatched_runner_evidence_fails_run_without_stalling_experiment(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(
        AgentRunOutput(
            output="A response with unrelated evidence.",
            evidence_paths=("src/example.py",),
            evidence=(
                CodeEvidence(
                    snapshot_id="c" * 64,
                    path="src/example.py",
                    line_start=1,
                    line_end=1,
                    snippet="return 1",
                    sha256="d" * 64,
                    source="text",
                ),
            ),
        )
    )
    lab = EvaluationLab(
        database_path=tmp_path / "invalid-evidence.sqlite3",
        runner=runner,
        evaluator=EvidencePathEvaluator(),
    )
    handle = await lab.evaluate(_one_case_request())
    [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.FAILED
    assert snapshot.runs[0].status is AgentRunStatus.FAILED
    assert snapshot.runs[0].failure_reason == "invalid_evidence_snapshot"


def test_experiment_cancel_marks_active_runs_terminal(tmp_path: Path) -> None:
    asyncio.run(_experiment_cancel_marks_active_runs_terminal(tmp_path))


async def _experiment_cancel_marks_active_runs_terminal(tmp_path: Path) -> None:
    runner = _BlockingRunner()
    lab = EvaluationLab(
        database_path=tmp_path / "cancelled-evaluation.sqlite3",
        runner=runner,
        evaluator=EvidencePathEvaluator(),
    )
    handle = await lab.evaluate(_one_case_request())
    await runner.started.wait()
    await handle.cancel()
    events = [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.CANCELLED
    assert snapshot.runs[0].status is AgentRunStatus.CANCELLED
    assert events[-1].payload["status"] == ExperimentStatus.CANCELLED.value


def test_open_resumes_interrupted_experiment_from_case_boundary(
    tmp_path: Path,
) -> None:
    asyncio.run(_open_resumes_interrupted_experiment_from_case_boundary(tmp_path))


async def _open_resumes_interrupted_experiment_from_case_boundary(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "interrupted-evaluation.sqlite3"
    interrupted_runner = _FirstSuccessThenBlockingRunner()
    original_lab = EvaluationLab(
        database_path=database_path,
        runner=interrupted_runner,
        evaluator=EvidencePathEvaluator(),
    )
    original_handle = await original_lab.evaluate(_two_case_request())
    await interrupted_runner.second_case_started.wait()

    resumed_runner = _FakeRunner(
        AgentRunOutput(
            output="Recovered run output.",
            evidence_paths=("src/example.py",),
        )
    )
    resumed_lab = EvaluationLab(
        database_path=database_path,
        runner=resumed_runner,
        evaluator=EvidencePathEvaluator(),
    )
    resumed_handle = await resumed_lab.open(original_handle.experiment_id)
    events = [event async for event in resumed_handle.events()]
    snapshot = await resumed_handle.snapshot()
    await original_handle.cancel()

    assert any(event.event_type == "experiment.recovered" for event in events)
    assert snapshot.experiment.status is ExperimentStatus.COMPLETED
    assert [
        (run.case_id, run.repetition, run.attempt, run.status)
        for run in snapshot.runs
    ] == [
        ("first-case", 1, 1, AgentRunStatus.COMPLETED),
        ("first-case", 2, 1, AgentRunStatus.FAILED),
        ("first-case", 2, 2, AgentRunStatus.COMPLETED),
        ("second-case", 1, 1, AgentRunStatus.FAILED),
        ("second-case", 1, 2, AgentRunStatus.COMPLETED),
        ("second-case", 2, 1, AgentRunStatus.FAILED),
        ("second-case", 2, 2, AgentRunStatus.COMPLETED),
    ]
    assert snapshot.runs[1].failure_reason == "process_interrupted"
    assert snapshot.runs[2].output == "Recovered run output."
    assert [
        (agent_input.case_id, agent_input.repetition)
        for agent_input in resumed_runner.inputs
    ] == [("first-case", 2), ("second-case", 1), ("second-case", 2)]


class _FakeRunner:
    def __init__(self, output: AgentRunOutput) -> None:
        self._output = output
        self.inputs: list[AgentRunInput] = []

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        self.inputs.append(agent_input)
        return self._output


class _FailingRunner:
    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        del agent_input
        raise RuntimeError("sensitive exception detail")


class _BlockingRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        del agent_input
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("取消后不应继续执行。")


class _FirstSuccessThenBlockingRunner:
    def __init__(self) -> None:
        self.second_case_started = asyncio.Event()

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        if agent_input.case_id == "first-case" and agent_input.repetition == 1:
            return AgentRunOutput(
                output="First case completed.",
                evidence_paths=("src/example.py",),
            )
        self.second_case_started.set()
        await asyncio.Event().wait()
        raise AssertionError("中断的 Case 不应自行完成。")


def _one_case_request() -> EvaluationRequest:
    return EvaluationRequest(
        suite_id="retrieval-smoke",
        snapshot_id=None,
        cases=(
            EvaluationCase(
                case_id="single-case",
                user_input="Find the expected source file.",
                ground_truth={"expected_paths": ["src/example.py"]},
            ),
        ),
    )


def _two_case_request() -> EvaluationRequest:
    return EvaluationRequest(
        suite_id="retrieval-smoke",
        snapshot_id=None,
        repetitions=2,
        cases=(
            EvaluationCase(
                case_id="first-case",
                user_input="Find the first source file.",
                ground_truth={"expected_paths": ["src/example.py"]},
            ),
            EvaluationCase(
                case_id="second-case",
                user_input="Find the second source file.",
                ground_truth={"expected_paths": ["src/example.py"]},
            ),
        ),
    )
