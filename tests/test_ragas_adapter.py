"""用离线 Fake Metric 验证 Ragas collection Metric 适配契约。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunOutput,
    EvaluationCase,
    EvaluationRequest,
    MetricDirection,
)
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.evaluation import (
    CompositeEvaluator,
    EvaluationLab,
    EvidencePathEvaluator,
    CompositeEvaluator,
    EvidencePathEvaluator,
    RagasMetricAdapter,
    faithfulness_input,
)


def test_ragas_adapter_preserves_metric_version_and_hides_ground_truth() -> None:
    asyncio.run(_ragas_adapter_preserves_metric_version_and_hides_ground_truth())


async def _ragas_adapter_preserves_metric_version_and_hides_ground_truth() -> None:
    metric = _FakeRagasMetric()
    adapter = RagasMetricAdapter(
        metric=metric,
        metric_name="faithfulness",
        ragas_version="0.4.3",
        input_builder=faithfulness_input,
        direction=MetricDirection.HIGHER_IS_BETTER,
    )
    case = EvaluationCase(
        case_id="faithfulness-case",
        user_input="What does the function return?",
        ground_truth={
            "expected_paths": ["src/example.py"],
            "answer": "hidden reference answer",
        },
    )
    output = AgentRunOutput(
        output="It returns a greeting.",
        evidence_paths=("src/example.py",),
        evidence=(
            CodeEvidence(
                snapshot_id="a" * 64,
                path="src/example.py",
                line_start=1,
                line_end=2,
                snippet="return 'hello'",
                sha256="b" * 64,
                source="text",
            ),
        ),
    )

    results = await CompositeEvaluator((EvidencePathEvaluator(), adapter)).evaluate(
        case,
        output,
    )

    assert [item.metric_name for item in results] == [
        "evidence_precision",
        "evidence_recall",
        "faithfulness",
    ]
    ragas_result = results[-1]
    assert ragas_result.value == 0.75
    assert ragas_result.evaluator_id == "ragas.faithfulness"
    assert ragas_result.evaluator_version == "0.4.3"
    assert ragas_result.direction is MetricDirection.HIGHER_IS_BETTER
    assert "hidden reference answer" not in repr(metric.inputs[0])
    assert metric.inputs[0] == {
        "user_input": case.user_input,
        "response": output.output,
        "retrieved_contexts": ["return 'hello'"],
    }
    assert "LLM_API_KEY" in ragas_result.details["reason"]
    assert "synthetic-secret-value" not in ragas_result.details["reason"]


def test_async_ragas_evaluator_runs_once_for_each_repetition(
    tmp_path: Path,
) -> None:
    asyncio.run(_async_ragas_evaluator_runs_once_for_each_repetition(tmp_path))


async def _async_ragas_evaluator_runs_once_for_each_repetition(
    tmp_path: Path,
) -> None:
    metric = _FakeRagasMetric()
    adapter = RagasMetricAdapter(
        metric=metric,
        metric_name="faithfulness",
        ragas_version="0.4.3",
        input_builder=faithfulness_input,
    )
    runner = _NoEvidenceRunner()
    lab = EvaluationLab(
        database_path=tmp_path / "ragas-evaluation.sqlite3",
        runner=runner,
        evaluator=CompositeEvaluator((EvidencePathEvaluator(), adapter)),
    )

    handle = await lab.evaluate(
        EvaluationRequest(
            suite_id="ragas-smoke",
            snapshot_id=None,
            repetitions=2,
            cases=(
                EvaluationCase(
                    case_id="faithfulness-case",
                    user_input="What is in the evidence?",
                    ground_truth={"expected_paths": []},
                ),
            ),
        )
    )
    [event async for event in handle.events()]
    snapshot = await handle.snapshot()

    assert [run.repetition for run in snapshot.runs] == [1, 2]
    assert all(run.status.value == "COMPLETED" for run in snapshot.runs)
    assert all(
        next(result for result in run.results if result.metric_name == "faithfulness").value
        == 0.75
        for run in snapshot.runs
    )
    assert len(metric.inputs) == 2
    assert len(runner.inputs) == 2


class _FakeRagasMetric:
    def __init__(self) -> None:
        self.inputs: list[dict[str, object]] = []

    async def ascore(self, **kwargs: object) -> SimpleNamespace:
        self.inputs.append(kwargs)
        return SimpleNamespace(
            value=0.75,
            reason="Supported by evidence. LLM_API_KEY=synthetic-secret-value",
        )


class _NoEvidenceRunner:
    def __init__(self) -> None:
        self.inputs: list[AgentRunInput] = []

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        self.inputs.append(agent_input)
        return AgentRunOutput(
            output="The code returns hello.",
        )
