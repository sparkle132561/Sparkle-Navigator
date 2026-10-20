"""Sparkle Presentation 服务串联仓库导入、索引与证据浏览。"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from tau_agent.messages import AssistantMessage, TextContent, ToolCall
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSessionConfig

from sparkle_navigator.domain.evaluation import ExperimentStatus
from sparkle_navigator.presentation.tui.services import SparklePresentationServices


def test_local_import_builds_snapshot_indexes_for_workspace_ask(tmp_path: Path) -> None:
    asyncio.run(_local_import_builds_snapshot_indexes_for_workspace_ask(tmp_path))


async def _local_import_builds_snapshot_indexes_for_workspace_ask(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "sample-repository"
    source_path = repository_root / "src" / "answer.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def answer():\n    return 'sparkle answer'\n",
        encoding="utf-8",
    )
    (repository_root / "README.md").write_text(
        "A tiny repository for the Sparkle workspace view.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "add", "README.md", "src/answer.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "-c",
            "user.name=Sparkle Tests",
            "-c",
            "user.email=sparkle-tests@example.invalid",
            "commit",
            "-qm",
            "Initial test fixture",
        ],
        check=True,
    )

    services = SparklePresentationServices(
        database_path=tmp_path / "sparkle.sqlite3",
        snapshots_root=tmp_path / "snapshots",
        working_copies_root=tmp_path / "working-copies",
        reports_root=tmp_path / "reports",
    )
    imported = await services.import_local_repository(repository_root)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)
    repositories = await services.list_repositories()
    overview = await services.overview(imported.snapshot.snapshot_id)
    answer = await services.ask(
        imported.snapshot.snapshot_id,
        "Where is the answer function implemented?",
    )

    assert repositories == (imported.repository,)
    assert overview.snapshot_id == imported.snapshot.snapshot_id
    assert overview.file_count == 2
    assert answer.evidence.items
    assert answer.evidence.items[0].path == "src/answer.py"
    expected_sha256 = next(
        item.sha256
        for item in imported.snapshot.files
        if item.path == "src/answer.py"
    )
    assert answer.evidence.items[0].sha256 == expected_sha256


def test_workspace_can_run_local_evidence_evaluation_without_model_provider(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _workspace_can_run_local_evidence_evaluation_without_model_provider(tmp_path)
    )


async def _workspace_can_run_local_evidence_evaluation_without_model_provider(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "evaluation-repository"
    source_path = repository_root / "src" / "answer.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def answer():\n    return 'sparkle answer'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "add", "src/answer.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "-c",
            "user.name=Sparkle Tests",
            "-c",
            "user.email=sparkle-tests@example.invalid",
            "commit",
            "-qm",
            "Evaluation fixture",
        ],
        check=True,
    )

    services = SparklePresentationServices(
        database_path=tmp_path / "sparkle-evaluation.sqlite3",
        snapshots_root=tmp_path / "evaluation-snapshots",
        working_copies_root=tmp_path / "evaluation-working-copies",
        reports_root=tmp_path / "evaluation-reports",
    )
    imported = await services.import_local_repository(repository_root)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)

    handle = await services.start_evidence_evaluation(
        snapshot_id=imported.snapshot.snapshot_id,
        user_input="Where is the answer function implemented?",
        expected_paths=("src/answer.py",),
    )
    events = [event async for event in handle.events()]
    experiment = await handle.snapshot()

    assert experiment.experiment.status is ExperimentStatus.COMPLETED
    assert experiment.experiment.snapshot_id == imported.snapshot.snapshot_id
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value, [
        (event.sequence, event.event_type, dict(event.payload)) for event in events
    ]
    assert len(experiment.runs) == 1
    run = experiment.runs[0]
    assert run.output
    assert {result.metric_name: result.value for result in run.results} == {
        "evidence_precision": 1.0,
        "evidence_recall": 1.0,
    }
    detail = await services.get_run(experiment.experiment.experiment_id, run.run_id)
    assert detail.evidence[0].snapshot_id == imported.snapshot.snapshot_id

    with pytest.raises(ValueError, match="not present in Snapshot"):
        await services.start_evidence_evaluation(
            snapshot_id=imported.snapshot.snapshot_id,
            user_input="Find a missing file.",
            expected_paths=("src/missing.py",),
        )


def test_tui_live_evaluation_persists_fake_native_run_and_snapshot_evidence(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _tui_live_evaluation_persists_fake_native_run_and_snapshot_evidence(
            tmp_path,
        )
    )


async def _tui_live_evaluation_persists_fake_native_run_and_snapshot_evidence(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "live-evaluation-repository"
    source_path = repository_root / "src" / "answer.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def answer():\n    return 'sparkle answer'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "add", "src/answer.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "-c",
            "user.name=Sparkle Tests",
            "-c",
            "user.email=sparkle-tests@example.invalid",
            "commit",
            "-qm",
            "Live evaluation fixture",
        ],
        check=True,
    )

    tool_message = AssistantMessage(
        content=[
            ToolCall(
                id="explain-live-answer",
                name="explain_code",
                arguments={"target": "answer"},
            )
        ],
        api="test",
        provider="fake",
        model="tui-live-fake-v1",
        stop_reason="toolUse",
    )
    final_message = AssistantMessage(
        content=[TextContent(text="`src/answer.py:1` defines `answer`.")],
        api="test",
        provider="fake",
        model="tui-live-fake-v1",
        stop_reason="stop",
    )

    def provider_events(message: AssistantMessage):
        return [
            AssistantStartEvent(partial=message.model_copy(deep=True)),
            AssistantDoneEvent(reason=message.stop_reason, message=message),
        ]

    providers: list[FakeProvider] = []

    def live_session_config_factory(agent_input) -> CodingSessionConfig:
        provider = FakeProvider(
            [provider_events(tool_message), provider_events(final_message)]
        )
        providers.append(provider)
        return CodingSessionConfig(
            provider=provider,
            model="tui-live-fake-v1",
            storage=JsonlSessionStorage(tmp_path / f"{agent_input.run_id}.jsonl"),
            cwd=repository_root,
            system="Use only the verified Snapshot tools to answer.",
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

    services = SparklePresentationServices(
        database_path=tmp_path / "sparkle-live-evaluation.sqlite3",
        snapshots_root=tmp_path / "live-snapshots",
        working_copies_root=tmp_path / "live-working-copies",
        reports_root=tmp_path / "live-reports",
        live_session_config_factory=live_session_config_factory,
    )
    imported = await services.import_local_repository(repository_root)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)

    handle = await services.start_live_evidence_evaluation(
        snapshot_id=imported.snapshot.snapshot_id,
        user_input="Where is answer implemented?",
        expected_paths=("src/answer.py",),
    )
    events = [event async for event in handle.events()]
    experiment = await handle.snapshot()

    assert experiment.experiment.status is ExperimentStatus.COMPLETED
    assert events[-1].payload["status"] == ExperimentStatus.COMPLETED.value
    assert len(experiment.runs) == 1
    run = experiment.runs[0]
    assert run.tau_session_id == run.run_id
    assert {result.metric_name: result.value for result in run.results} == {
        "evidence_precision": 1.0,
        "evidence_recall": 1.0,
    }
    detail = await services.get_run(experiment.experiment.experiment_id, run.run_id)
    assert detail.evidence[0].path == "src/answer.py"
    assert "tool_execution_end" in [event.event_type for event in detail.trace_events]
    assert len(providers) == 1
    assert len(providers[0].calls) == 2
    assert "expected_paths" not in repr(providers[0].calls)


def test_live_evaluation_stops_at_twenty_provider_requests(
    tmp_path: Path,
) -> None:
    asyncio.run(_live_evaluation_stops_at_twenty_provider_requests(tmp_path))


async def _live_evaluation_stops_at_twenty_provider_requests(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "bounded-live-repository"
    source_path = repository_root / "src" / "answer.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def answer():\n    return 42\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository_root)], check=True)
    subprocess.run(
        ["git", "-C", str(repository_root), "add", "src/answer.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "-c",
            "user.name=Sparkle Tests",
            "-c",
            "user.email=sparkle-tests@example.invalid",
            "commit",
            "-qm",
            "Bounded live evaluation fixture",
        ],
        check=True,
    )

    streams = []
    for index in range(21):
        message = AssistantMessage(
            content=[
                ToolCall(
                    id=f"overview-{index}",
                    name="overview_repository",
                    arguments={},
                )
            ],
            api="test",
            provider="fake",
            model="bounded-live-fake-v1",
            stop_reason="toolUse",
        )
        streams.append(
            [
                AssistantStartEvent(partial=message.model_copy(deep=True)),
                AssistantDoneEvent(reason="toolUse", message=message),
            ]
        )

    providers: list[FakeProvider] = []

    def live_session_config_factory(agent_input) -> CodingSessionConfig:
        provider = FakeProvider(streams)
        providers.append(provider)
        return CodingSessionConfig(
            provider=provider,
            model="bounded-live-fake-v1",
            storage=JsonlSessionStorage(tmp_path / f"{agent_input.run_id}.jsonl"),
            cwd=repository_root,
            system="Use only the verified Snapshot tools.",
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

    services = SparklePresentationServices(
        database_path=tmp_path / "sparkle-bounded-live.sqlite3",
        snapshots_root=tmp_path / "bounded-live-snapshots",
        working_copies_root=tmp_path / "bounded-live-working-copies",
        reports_root=tmp_path / "bounded-live-reports",
        live_session_config_factory=live_session_config_factory,
    )
    imported = await services.import_local_repository(repository_root)
    await services.build_snapshot_indexes(imported.snapshot.snapshot_id)

    handle = await services.start_live_evidence_evaluation(
        snapshot_id=imported.snapshot.snapshot_id,
        user_input="Use the repository overview tool repeatedly.",
        expected_paths=("src/answer.py",),
    )
    events = [event async for event in handle.events()]
    experiment = await handle.snapshot()

    assert experiment.experiment.status is ExperimentStatus.FAILED
    assert events[-1].payload["status"] == ExperimentStatus.FAILED.value
    assert len(experiment.runs) == 1
    assert experiment.runs[0].failure_reason == "ProviderRequestLimitExceeded"
    assert len(providers) == 1
    assert len(providers[0].calls) == 20
