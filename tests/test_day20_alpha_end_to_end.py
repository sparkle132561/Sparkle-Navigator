"""固定 Sparkle commit 上的 Tau/Sparkle 评测端到端集成契约。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from tau_agent.messages import AssistantMessage, TextContent, ToolCall
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSessionConfig

from sparkle_navigator.domain.evaluation import (
    EvaluationCase,
    EvaluationRequest,
    ExperimentStatus,
)
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.evaluation.core import EvaluationLab
from sparkle_navigator.evaluation.native_backend import NativeBackend
from sparkle_navigator.evaluation.run_explorer import RunExplorer
from sparkle_navigator.presentation.reports import EvaluationReportExporter
from sparkle_navigator.presentation.tui.services import SparklePresentationServices
from sparkle_navigator.persistence.repository_store import RepositoryStore
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime
from sparkle_navigator.evaluation.smoke_set import EvidencePathEvaluator

ALPHA_CASES_PATH = Path(__file__).parent / "fixtures" / "day20_alpha" / "cases.json"
ALPHA_CASES = json.loads(ALPHA_CASES_PATH.read_text(encoding="utf-8"))
ALPHA_REPOSITORY_PATH = Path(__file__).parents[1].resolve()


def test_alpha_compares_tau_without_snapshot_evidence_to_sparkle_tools(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "day20-alpha.sqlite3"
    imported = _register_pinned_snapshot(tmp_path, database_path)
    asyncio.run(
        _alpha_compares_tau_without_snapshot_evidence_to_sparkle_tools(
            tmp_path,
            database_path,
            imported,
        )
    )


async def _alpha_compares_tau_without_snapshot_evidence_to_sparkle_tools(
    tmp_path: Path,
    database_path: Path,
    imported: RepositoryImportResult,
) -> None:
    services = SparklePresentationServices(
        database_path=database_path,
        snapshots_root=tmp_path / "s",
        working_copies_root=tmp_path / "w",
        reports_root=tmp_path / "reports",
    )
    snapshot = imported.snapshot
    snapshot_paths = {item.path for item in snapshot.files}
    expected_paths = {
        path
        for case in ALPHA_CASES["cases"]
        for path in case["ground_truth"]["expected_paths"]
    }
    assert expected_paths <= snapshot_paths

    await services.build_snapshot_indexes(snapshot.snapshot_id)
    working_copy = await services.open_working_copy(
        imported.repository.repository_id
    )
    sessions_root = tmp_path / "tau-sessions"
    sessions_root.mkdir()

    baseline_providers: list[FakeProvider] = []
    candidate_providers: list[FakeProvider] = []
    baseline_lab = EvaluationLab(
        database_path=database_path,
        runner=NativeBackend(
            session_config_factory=_config_factory(
                sessions_root=sessions_root,
                database_path=database_path,
                snapshot_id=snapshot.snapshot_id,
                working_copy_root=working_copy.root,
                with_snapshot_tools=False,
                providers=baseline_providers,
            )
        ),
        evaluator=EvidencePathEvaluator(),
    )
    candidate_lab = EvaluationLab(
        database_path=database_path,
        runner=NativeBackend(
            session_config_factory=_config_factory(
                sessions_root=sessions_root,
                database_path=database_path,
                snapshot_id=snapshot.snapshot_id,
                working_copy_root=working_copy.root,
                with_snapshot_tools=True,
                providers=candidate_providers,
            )
        ),
        evaluator=EvidencePathEvaluator(),
    )

    baseline = await _run_variant(
        baseline_lab,
        snapshot_id=snapshot.snapshot_id,
        variant="tau-without-snapshot-tools",
    )
    candidate = await _run_variant(
        candidate_lab,
        snapshot_id=snapshot.snapshot_id,
        variant="sparkle-snapshot-explain-tool",
    )
    comparison = await baseline_lab.compare(
        baseline.experiment.experiment_id,
        candidate.experiment.experiment_id,
    )

    assert baseline.experiment.status is ExperimentStatus.COMPLETED
    assert candidate.experiment.status is ExperimentStatus.COMPLETED
    assert len(baseline.runs) == len(candidate.runs) == 15
    expected_session_ids = {
        run.run_id
        for run in (*baseline.runs, *candidate.runs)
    }
    session_files = tuple(sessions_root.glob("*.jsonl"))
    assert {path.stem for path in session_files} == expected_session_ids
    assert all(path.stat().st_size > 0 for path in session_files)
    assert all(
        {result.metric_name: result.value for result in run.results}
        == {"evidence_precision": 0.0, "evidence_recall": 0.0}
        for run in baseline.runs
    )
    assert all(
        {result.metric_name: result.value for result in run.results}
        == {"evidence_precision": 1.0, "evidence_recall": 1.0}
        for run in candidate.runs
    )
    assert {
        result.metric_name: (result.baseline_mean, result.candidate_mean)
        for result in comparison.results
    } == {
        "evidence_precision": (0.0, 1.0),
        "evidence_recall": (0.0, 1.0),
    }
    assert all(result.baseline_variance == result.candidate_variance == 0.0 for result in comparison.results)
    assert all(result.baseline_sample_count == result.candidate_sample_count == 15 for result in comparison.results)
    assert all(not result.is_regression for result in comparison.results)

    explorer = RunExplorer(database_path=database_path)
    candidate_details = tuple(
        [
            await explorer.get_run(candidate.experiment.experiment_id, run.run_id)
            for run in candidate.runs
        ]
    )
    assert all(detail.evidence for detail in candidate_details)
    assert all(
        any(event.event_type == "tool_execution_end" for event in detail.trace_events)
        for detail in candidate_details
    )
    assert all(detail.run.tau_session_id == detail.run.run_id for detail in candidate_details)

    for provider in (*baseline_providers, *candidate_providers):
        first_provider_input = repr(provider.calls[0])
        assert "expected_paths" not in first_provider_input
        assert "ground_truth" not in first_provider_input
        assert all(path not in first_provider_input for path in expected_paths)

    exporter = EvaluationReportExporter()
    report_directory = tmp_path / "reports"
    candidate_report = exporter.export_experiment(
        candidate,
        candidate_details,
        report_directory / "candidate.json",
        format="json",
    )
    comparison_report = exporter.export_comparison(
        comparison,
        report_directory / "comparison.json",
        format="json",
    )
    candidate_payload = json.loads(candidate_report.read_text(encoding="utf-8"))
    comparison_payload = json.loads(comparison_report.read_text(encoding="utf-8"))
    assert candidate_payload["report_type"] == "experiment"
    assert comparison_payload["report_type"] == "comparison"
    assert candidate_payload["runs"][0]["results"][0]["details"]["expected_paths"]


def _register_pinned_snapshot(
    tmp_path: Path,
    database_path: Path,
) -> RepositoryImportResult:
    staging_root = tmp_path / "selected-source-files"
    snapshot_paths = sorted(
        {
            path
            for case in ALPHA_CASES["cases"]
            for path in case["ground_truth"]["expected_paths"]
        }
    )
    files: list[SnapshotFile] = []
    for relative_path in snapshot_paths:
        content = subprocess.run(
            [
                "git",
                "-C",
                str(ALPHA_REPOSITORY_PATH),
                "show",
                f"{ALPHA_CASES['source_commit']}:{relative_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        destination = staging_root.joinpath(*relative_path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        files.append(
            SnapshotFile(
                path=relative_path,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )

    snapshot_without_root = RepositorySnapshot.create(
        source_commit=ALPHA_CASES["source_commit"],
        files=files,
    )
    snapshot_root = tmp_path / "s" / snapshot_without_root.snapshot_id
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.replace(snapshot_root)
    snapshot = RepositorySnapshot.create(
        source_commit=ALPHA_CASES["source_commit"],
        files=files,
        snapshot_root=snapshot_root,
    )
    repository = Repository(
        repository_id="alpha",
        name="sparkle-day20-alpha",
        source_type="git",
        source_locator=ALPHA_REPOSITORY_PATH.as_uri(),
        trusted=False,
        current_snapshot_id=snapshot.snapshot_id,
    )
    result = RepositoryImportResult(repository=repository, snapshot=snapshot)
    RepositoryStore(database_path).save_import(result)
    return result


def _config_factory(
    *,
    sessions_root: Path,
    database_path: Path,
    snapshot_id: str,
    working_copy_root: Path,
    with_snapshot_tools: bool,
    providers: list[FakeProvider],
):
    def create(agent_input) -> CodingSessionConfig:
        assert not hasattr(agent_input, "ground_truth")
        target = agent_input.user_input.partition(":")[2].strip()
        provider = _provider_for_case(
            agent_input.run_id,
            target,
            with_snapshot_tools=with_snapshot_tools,
        )
        providers.append(provider)
        config = CodingSessionConfig(
            provider=provider,
            model="day20-fake-provider-v1",
            storage=JsonlSessionStorage(sessions_root / f"{agent_input.run_id}.jsonl"),
            cwd=working_copy_root,
            system="回答代码问题时优先使用可见的 Snapshot 工具；不要猜测来源。",
            tool_access_policy=ToolProfileRuntime(
                repository_trusted=False,
                workspace_root=working_copy_root,
            ),
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
            trust_default="deny",
        )
        if with_snapshot_tools:
            config = add_codebase_tools(
                config,
                database_path=database_path,
                snapshot_id=snapshot_id,
            )
        return config

    return create


async def _run_variant(
    lab: EvaluationLab,
    *,
    snapshot_id: str,
    variant: str,
):
    request = EvaluationRequest(
        suite_id=ALPHA_CASES["suite_id"],
        snapshot_id=snapshot_id,
        cases=tuple(
            EvaluationCase(
                case_id=case["case_id"],
                user_input=case["user_input"],
                ground_truth=case["ground_truth"],
            )
            for case in ALPHA_CASES["cases"]
        ),
        repetitions=ALPHA_CASES["repetitions"],
        configuration={"variant": variant, "model": "FakeProvider"},
    )
    handle = await lab.evaluate(request)
    events = [event async for event in handle.events()]
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value
    return await handle.snapshot()


def _provider_for_case(
    run_id: str,
    target: str,
    *,
    with_snapshot_tools: bool,
) -> FakeProvider:
    model = "day20-fake-provider-v1"
    final_message = AssistantMessage(
        content=[TextContent(text=f"Public task processed for {target}.")],
        api="test",
        provider="fake",
        model=model,
        stop_reason="stop",
    )
    final_events = [
        AssistantStartEvent(partial=final_message.model_copy(deep=True)),
        AssistantDoneEvent(reason="stop", message=final_message),
    ]
    if not with_snapshot_tools:
        return FakeProvider([final_events])

    tool_message = AssistantMessage(
        content=[
            ToolCall(
                id=f"explain-{run_id}",
                name="explain_code",
                arguments={"target": target},
            )
        ],
        api="test",
        provider="fake",
        model=model,
        stop_reason="toolUse",
    )
    tool_events = [
        AssistantStartEvent(partial=tool_message.model_copy(deep=True)),
        AssistantDoneEvent(reason="toolUse", message=tool_message),
    ]
    return FakeProvider([tool_events, final_events])
