"""Inspect AI Task 通过 Sparkle EvaluationLab 运行。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunStatus,
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.evaluation.core import EvaluationLab
from sparkle_navigator.evaluation.inspect_ai_backend import InspectAIBackend
from sparkle_navigator.evaluation.smoke_set import EvidencePathEvaluator


def test_inspect_task_runs_one_public_case_through_evaluation_lab(
    tmp_path: Path,
) -> None:
    asyncio.run(_inspect_task_runs_one_public_case_through_evaluation_lab(tmp_path))


async def _inspect_task_runs_one_public_case_through_evaluation_lab(
    tmp_path: Path,
) -> None:
    pytest.importorskip("inspect_ai")
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.solver import generate

    task_inputs: list[str] = []

    def task_factory(agent_input: AgentRunInput) -> object:
        task_inputs.append(agent_input.user_input)
        assert not hasattr(agent_input, "ground_truth")
        return Task(
            dataset=[Sample(input=agent_input.user_input, id=agent_input.run_id)],
            solver=generate(),
        )

    lab = EvaluationLab(
        database_path=tmp_path / "inspect-evaluation.sqlite3",
        runner=InspectAIBackend(
            task_factory=task_factory,
            model=get_model(
                "mockllm/model",
                custom_outputs=[
                    ModelOutput.from_content(
                        "mockllm/model",
                        "Inspect AI produced this deterministic answer.",
                    )
                ],
                memoize=False,
            ),
            log_dir=tmp_path / "inspect-logs",
        ),
        evaluator=EvidencePathEvaluator(),
    )
    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="inspect-smoke",
            snapshot_id=None,
            cases=(
                EvaluationCase(
                    case_id="inspect-standard-task",
                    user_input="Run this through a standard Inspect Task.",
                    ground_truth={"expected_paths": ["src/example.py"]},
                ),
            ),
        )
    )
    events = [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert task_inputs == ["Run this through a standard Inspect Task."]
    assert snapshot.experiment.status is ExperimentStatus.COMPLETED
    assert len(snapshot.runs) == 1
    assert snapshot.runs[0].status is AgentRunStatus.COMPLETED
    assert snapshot.runs[0].output == "Inspect AI produced this deterministic answer."
    assert [result.metric_name for result in snapshot.runs[0].results] == [
        "evidence_precision",
        "evidence_recall",
    ]
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value


def test_cancelling_inspect_task_cancels_the_sparkle_run(tmp_path: Path) -> None:
    asyncio.run(_cancelling_inspect_task_cancels_the_sparkle_run(tmp_path))


async def _cancelling_inspect_task_cancels_the_sparkle_run(
    tmp_path: Path,
) -> None:
    pytest.importorskip("inspect_ai")
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import solver

    task_started = asyncio.Event()

    @solver
    def wait_for_cancel():
        async def solve(state: object, generate: object) -> object:
            task_started.set()
            await asyncio.Event().wait()
            return state

        return solve

    def task_factory(agent_input: AgentRunInput) -> object:
        return Task(
            dataset=[Sample(input=agent_input.user_input, id=agent_input.run_id)],
            solver=wait_for_cancel(),
        )

    backend = InspectAIBackend(
        task_factory=task_factory,
        model="mockllm/model",
        log_dir=tmp_path / "inspect-cancel-logs",
    )

    lab = EvaluationLab(
        database_path=tmp_path / "inspect-cancel.sqlite3",
        runner=backend,
        evaluator=EvidencePathEvaluator(),
    )
    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="inspect-cancel-smoke",
            snapshot_id=None,
            cases=(
                EvaluationCase(
                    case_id="inspect-cancel-case",
                    user_input="Wait until the run is cancelled.",
                    ground_truth={"expected_paths": ["src/example.py"]},
                ),
            ),
        )
    )
    await asyncio.wait_for(task_started.wait(), timeout=15)
    await handle.cancel()
    events = [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.CANCELLED
    assert snapshot.runs[0].status is AgentRunStatus.CANCELLED
    assert events[-1].payload["status"] == ExperimentStatus.CANCELLED.value
