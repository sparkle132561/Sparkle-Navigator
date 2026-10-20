"""外部 HTTP Agent 适配器通过 EvaluationLab 的公开接口运行。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from sparkle_navigator.domain.evaluation import (
    AgentRunInput,
    AgentRunStatus,
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.evaluation.core import EvaluationLab
from sparkle_navigator.evaluation.external_http_agent import ExternalHttpAgentAdapter
from sparkle_navigator.evaluation.smoke_set import EvidencePathEvaluator


def test_external_http_agent_runs_one_case_through_evaluation_lab(
    tmp_path: Path,
) -> None:
    asyncio.run(_external_http_agent_runs_one_case_through_evaluation_lab(tmp_path))


async def _external_http_agent_runs_one_case_through_evaluation_lab(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    async def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "protocol_version": 1,
                "output": "The answer is in src/answer.py. token=test-secret",
                "evidence_paths": ["src/answer.py"],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        runner = ExternalHttpAgentAdapter(
            endpoint="https://agent.example/v1/run",
            client=client,
            redacted_values=("test-secret",),
        )
        lab = EvaluationLab(
            database_path=tmp_path / "external-agent.sqlite3",
            runner=runner,
            evaluator=EvidencePathEvaluator(),
        )
        handle = await lab.evaluate(
            EvaluationRequest(
                suite_id="external-agent-smoke",
                snapshot_id=None,
                cases=(
                    EvaluationCase(
                        case_id="find-answer",
                        user_input="Find the answer implementation.",
                        ground_truth={"expected_paths": ["src/answer.py"]},
                    ),
                ),
            )
        )
        events = [event async for event in handle.events()]
        snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.COMPLETED
    assert len(snapshot.runs) == 1
    run = snapshot.runs[0]
    assert run.status is AgentRunStatus.COMPLETED
    assert run.output == "The answer is in src/answer.py. token=[REDACTED]"
    assert {result.metric_name: result.value for result in run.results} == {
        "evidence_precision": 1.0,
        "evidence_recall": 1.0,
    }
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value

    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert request.method == "POST"
    assert request.url == "https://agent.example/v1/run"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["idempotency-key"] == run.run_id
    assert payload == {
        "protocol_version": 1,
        "run_id": run.run_id,
        "case_id": "find-answer",
        "repetition": 1,
        "snapshot_id": None,
        "input": "Find the answer implementation.",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"protocol_version": 2, "output": "answer"}, "protocol_version"),
        ({"protocol_version": True, "output": "answer"}, "protocol_version"),
        (
            {
                "protocol_version": 1,
                "output": "answer",
                "evidence_paths": ["../outside.py"],
            },
            "相对文件路径",
        ),
        ({"protocol_version": 1, "output": 123}, "字符串"),
    ],
)
def test_external_http_agent_rejects_invalid_response(
    payload: dict[str, object],
    message: str,
) -> None:
    asyncio.run(_external_http_agent_rejects_invalid_response(payload, message))


async def _external_http_agent_rejects_invalid_response(
    payload: dict[str, object],
    message: str,
) -> None:
    async def handle_request(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        runner = ExternalHttpAgentAdapter(
            endpoint="https://agent.example/v1/run",
            client=client,
        )
        with pytest.raises(ValueError, match=message):
            await runner.run(
                AgentRunInput(
                    run_id="run-1",
                    case_id="case-1",
                    repetition=1,
                    snapshot_id=None,
                    user_input="Explain this code.",
                )
            )


def test_external_http_agent_rejects_oversized_response() -> None:
    asyncio.run(_external_http_agent_rejects_oversized_response())


async def _external_http_agent_rejects_oversized_response() -> None:
    async def handle_request(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"protocol_version": 1, "output": "too long"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        runner = ExternalHttpAgentAdapter(
            endpoint="https://agent.example/v1/run",
            client=client,
            max_response_bytes=10,
        )
        with pytest.raises(ValueError, match="超过大小限制"):
            await runner.run(
                AgentRunInput(
                    run_id="run-1",
                    case_id="case-1",
                    repetition=1,
                    snapshot_id=None,
                    user_input="Explain this code.",
                )
            )


def test_external_http_error_marks_agent_run_failed(tmp_path: Path) -> None:
    asyncio.run(_external_http_error_marks_agent_run_failed(tmp_path))


async def _external_http_error_marks_agent_run_failed(tmp_path: Path) -> None:
    async def handle_request(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, text="service detail contains secret")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        lab = EvaluationLab(
            database_path=tmp_path / "external-agent-error.sqlite3",
            runner=ExternalHttpAgentAdapter(
                endpoint="https://agent.example/v1/run",
                client=client,
            ),
            evaluator=EvidencePathEvaluator(),
        )
        handle = await lab.evaluate(
            EvaluationRequest(
                suite_id="external-agent-smoke",
                snapshot_id=None,
                cases=(
                    EvaluationCase(
                        case_id="find-answer",
                        user_input="Find the answer implementation.",
                        ground_truth={"expected_paths": ["src/answer.py"]},
                    ),
                ),
            )
        )
        events = [event async for event in handle.events()]
        snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.FAILED
    assert snapshot.runs[0].status is AgentRunStatus.FAILED
    assert snapshot.runs[0].failure_reason == "HTTPStatusError"
    assert "service detail contains secret" not in repr(snapshot)
    assert events[-1].payload["status"] == ExperimentStatus.FAILED.value


def test_external_http_timeout_marks_agent_run_failed(tmp_path: Path) -> None:
    asyncio.run(_external_http_timeout_marks_agent_run_failed(tmp_path))


async def _external_http_timeout_marks_agent_run_failed(tmp_path: Path) -> None:
    async def handle_request(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ReadTimeout("sensitive timeout detail")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        lab = EvaluationLab(
            database_path=tmp_path / "external-agent-timeout.sqlite3",
            runner=ExternalHttpAgentAdapter(
                endpoint="https://agent.example/v1/run",
                client=client,
            ),
            evaluator=EvidencePathEvaluator(),
        )
        handle = await lab.evaluate(
            EvaluationRequest(
                suite_id="external-agent-smoke",
                snapshot_id=None,
                cases=(
                    EvaluationCase(
                        case_id="find-answer",
                        user_input="Find the answer implementation.",
                        ground_truth={"expected_paths": ["src/answer.py"]},
                    ),
                ),
            )
        )
        events = [event async for event in handle.events()]
        snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.FAILED
    assert snapshot.runs[0].status is AgentRunStatus.FAILED
    assert snapshot.runs[0].failure_reason == "ReadTimeout"
    assert "sensitive timeout detail" not in repr(snapshot)
    assert events[-1].payload["status"] == ExperimentStatus.FAILED.value


def test_external_http_cancellation_is_persisted(tmp_path: Path) -> None:
    asyncio.run(_external_http_cancellation_is_persisted(tmp_path))


async def _external_http_cancellation_is_persisted(tmp_path: Path) -> None:
    request_started = asyncio.Event()

    async def handle_request(request: httpx.Request) -> httpx.Response:
        del request
        request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("取消的 HTTP 请求不应继续执行。")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        lab = EvaluationLab(
            database_path=tmp_path / "external-agent-cancel.sqlite3",
            runner=ExternalHttpAgentAdapter(
                endpoint="https://agent.example/v1/run",
                client=client,
            ),
            evaluator=EvidencePathEvaluator(),
        )
        handle = await lab.evaluate(
            EvaluationRequest(
                suite_id="external-agent-smoke",
                snapshot_id=None,
                cases=(
                    EvaluationCase(
                        case_id="find-answer",
                        user_input="Find the answer implementation.",
                        ground_truth={"expected_paths": ["src/answer.py"]},
                    ),
                ),
            )
        )
        await request_started.wait()
        await handle.cancel()
        events = [event async for event in handle.events()]
        snapshot = await handle.snapshot()

    assert snapshot.experiment.status is ExperimentStatus.CANCELLED
    assert snapshot.runs[0].status is AgentRunStatus.CANCELLED
    assert events[-1].payload["status"] == ExperimentStatus.CANCELLED.value
