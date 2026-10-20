"""验证代码库工具通过 Tau CodingSession 的真实 Agent Loop 运行。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from tau_agent.events import MessageEndEvent, ToolExecutionEndEvent
from tau_agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from tau_agent.session import InMemorySessionStorage, JsonlSessionStorage
from tau_ai.fake import FakeProvider
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.events import AgentSettledEvent, SessionAgentEndEvent

from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.evaluation import EvaluationCase, EvaluationRequest
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore
from sparkle_navigator.evaluation import (
    EvaluationLab,
    EvidencePathEvaluator,
    NativeBackend,
    RunExplorer,
)
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime


def test_explain_tool_runs_through_tau_session_and_returns_evidence() -> None:
    asyncio.run(_run_explain_tool_through_tau_session())


def test_changed_snapshot_file_is_reported_as_tau_tool_error() -> None:
    asyncio.run(_changed_snapshot_file_is_reported_as_tau_tool_error())


def test_native_backend_persists_snapshot_evidence_from_tau_tool_events() -> None:
    asyncio.run(_native_backend_persists_snapshot_evidence_from_tau_tool_events())


async def _native_backend_persists_snapshot_evidence_from_tau_tool_events() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot, database_path = _register_snapshot(
            root,
            {"src/app.py": "def start():\n    return 'ready'\n"},
        )
        await SnapshotSymbolIndex(database_path=database_path).build(snapshot)
        await SnapshotCodeGraph(database_path=database_path).build(snapshot)
        await SnapshotTextIndex(database_path=database_path).build(snapshot)

        def config_for_run(agent_input) -> CodingSessionConfig:
            provider = _provider_with_tool_calls(
                (
                    ToolCall(
                        id="native-explain",
                        name="explain_code",
                        arguments={"target": "src/app.py"},
                    ),
                )
            )
            base_config = CodingSessionConfig(
                provider=provider,
                model="native-evidence-test-model",
                storage=JsonlSessionStorage(root / f"{agent_input.run_id}.jsonl"),
                cwd=snapshot.snapshot_root,
                system="Use the configured Snapshot evidence tool.",
                tool_access_policy=ToolProfileRuntime(),
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
            return add_codebase_tools(
                base_config,
                database_path=database_path,
                snapshot_id=snapshot.snapshot_id,
            )

        lab = EvaluationLab(
            database_path=database_path,
            runner=NativeBackend(session_config_factory=config_for_run),
            evaluator=EvidencePathEvaluator(),
        )
        handle = await lab.evaluate(
            EvaluationRequest(
                suite_id="native-evidence-smoke",
                snapshot_id=snapshot.snapshot_id,
                cases=(
                    EvaluationCase(
                        case_id="explain-app",
                        user_input="Explain src/app.py.",
                        ground_truth={"expected_paths": ["src/app.py"]},
                    ),
                ),
            )
        )
        [event async for event in handle.events()]
        run = (await handle.snapshot()).runs[0]
        detail = await RunExplorer(database_path=database_path).get_run(
            run.experiment_id,
            run.run_id,
        )

        assert detail.run.status.value == "COMPLETED"
        assert {item.metric_name: item.value for item in detail.run.results} == {
            "evidence_precision": 1.0,
            "evidence_recall": 1.0,
        }
        assert detail.evidence
        assert any(
            item.path == "src/app.py"
            and item.snapshot_id == snapshot.snapshot_id
            and item.sha256 == next(
                file.sha256 for file in snapshot.files if file.path == item.path
            )
            for item in detail.evidence
        )


async def _run_explain_tool_through_tau_session() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot, database_path = _register_snapshot(
            root,
            {
                "src/app.py": (
                    "from .service import work\n"
                    "\n"
                    "def start():\n"
                    "    return work()\n"
                ),
                "src/service.py": "def work():\n    return 'ready'\n",
                "tests/test_app.py": (
                    "from src.service import work\n"
                    "\n"
                    "def test_work():\n"
                    "    work()\n"
                ),
                "README.md": "# Demo repository\n",
            },
        )
        await SnapshotSymbolIndex(database_path=database_path).build(snapshot)
        await SnapshotCodeGraph(database_path=database_path).build(snapshot)
        await SnapshotTextIndex(database_path=database_path).build(snapshot)

        provider = _provider_with_tool_calls(
            (
                ToolCall(id="call-overview", name="overview_repository", arguments={}),
                ToolCall(
                    id="call-ask",
                    name="ask_codebase",
                    arguments={"question": "start"},
                ),
                ToolCall(
                    id="call-explain",
                    name="explain_code",
                    arguments={"target": "start"},
                ),
                ToolCall(
                    id="call-trace",
                    name="trace_call",
                    arguments={"start_path": "src/app.py", "start_symbol": "start"},
                ),
                ToolCall(
                    id="call-impact",
                    name="analyze_impact",
                    arguments={
                        "target_path": "src/service.py",
                        "target_symbol": "work",
                        "max_depth": 3,
                    },
                ),
            )
        )
        storage = InMemorySessionStorage()
        base_config = CodingSessionConfig(
            provider=provider,
            model="codebase-test-model",
            storage=storage,
            cwd=snapshot.snapshot_root,
            system="Explain code using verified evidence.",
            tool_access_policy=ToolProfileRuntime(),
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )

        session = await CodingSession.load(
            add_codebase_tools(
                base_config,
                database_path=database_path,
                snapshot_id=snapshot.snapshot_id,
            )
        )
        try:
            events = [event async for event in session.prompt("Explain start")]

            visible_tool_names = [tool.name for tool in provider.calls[0][3]]
            assert visible_tool_names == [
                "read",
                "overview_repository",
                "ask_codebase",
                "explain_code",
                "trace_call",
                "analyze_impact",
            ]
            tool_result = next(
                message
                for message in provider.calls[1][2]
                if isinstance(message, ToolResultMessage)
                and message.tool_name == "explain_code"
            )
            assert "src/app.py" in tool_result.text
            assert "def start" in tool_result.text
            tool_results = {
                message.tool_name: message
                for message in provider.calls[1][2]
                if isinstance(message, ToolResultMessage)
            }
            assert set(tool_results) == {
                "overview_repository",
                "ask_codebase",
                "explain_code",
                "trace_call",
                "analyze_impact",
            }
            assert "README.md" in tool_results["overview_repository"].text
            assert "src/app.py" in tool_results["ask_codebase"].text
            assert "CALLS" in tool_results["trace_call"].text
            assert "tests/test_app.py" in tool_results["analyze_impact"].text
            evidence_details = tool_results["explain_code"].details
            assert isinstance(evidence_details, dict)
            code_evidence = evidence_details["code_evidence"]
            assert isinstance(code_evidence, list)
            assert any(
                item["snapshot_id"] == snapshot.snapshot_id
                and item["path"] == "src/app.py"
                and item["sha256"]
                for item in code_evidence
            )
            assert any(
                isinstance(event, ToolExecutionEndEvent) and not event.is_error
                for event in events
            )
            assert sum(isinstance(event, ToolExecutionEndEvent) for event in events) == 5
            assert any(
                isinstance(event, MessageEndEvent)
                and isinstance(event.message, AssistantMessage)
                and event.message.text == "The verified source defines start."
                for event in events
            )
            assert any(isinstance(event, SessionAgentEndEvent) for event in events)
            assert any(isinstance(event, AgentSettledEvent) for event in events)
            assert len(await storage.read_all()) > 3
        finally:
            await session.aclose()


async def _changed_snapshot_file_is_reported_as_tau_tool_error() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot, database_path = _register_snapshot(
            root,
            {"src/app.py": "def start():\n    return 'original'\n"},
        )
        await SnapshotTextIndex(database_path=database_path).build(snapshot)
        await SnapshotSymbolIndex(database_path=database_path).build(snapshot)
        await SnapshotCodeGraph(database_path=database_path).build(snapshot)
        (snapshot.snapshot_root / "src" / "app.py").write_text(
            "def start():\n    return 'tampered_value'\n",
            encoding="utf-8",
        )
        provider = _provider_with_tool_calls(
            (
                ToolCall(
                    id="call-tampered-explain",
                    name="explain_code",
                    arguments={"target": "src/app.py"},
                ),
            )
        )
        session = await CodingSession.load(
            add_codebase_tools(
                CodingSessionConfig(
                    provider=provider,
                    model="codebase-test-model",
                    storage=InMemorySessionStorage(),
                    cwd=snapshot.snapshot_root,
                    system="Only use verified Snapshot evidence.",
                    tool_access_policy=ToolProfileRuntime(),
                    skills_enabled=False,
                    extensions_enabled=False,
                    project_extensions_enabled=False,
                ),
                database_path=database_path,
                snapshot_id=snapshot.snapshot_id,
            )
        )
        try:
            events = [event async for event in session.prompt("Explain src/app.py")]
            tool_result = next(
                message
                for message in provider.calls[1][2]
                if isinstance(message, ToolResultMessage)
            )

            assert tool_result.is_error
            assert "SHA-256" in tool_result.text
            assert "tampered_value" not in tool_result.text
            assert any(
                isinstance(event, ToolExecutionEndEvent) and event.is_error
                for event in events
            )
        finally:
            await session.aclose()


def _register_snapshot(
    root: Path,
    source_files: dict[str, str],
) -> tuple[RepositorySnapshot, Path]:
    snapshot_root = root / "snapshot"
    files: list[SnapshotFile] = []
    for relative_path, source in source_files.items():
        source_path = snapshot_root.joinpath(*relative_path.split("/"))
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        files.append(
            SnapshotFile(
                path=relative_path,
                sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
        )
    snapshot = RepositorySnapshot.create(
        source_commit="d" * 40,
        files=files,
        snapshot_root=snapshot_root,
    )
    database_path = root / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"codebase-tool-session").hexdigest(),
                name="codebase-tool-session",
                source_type="local",
                source_locator="test://codebase-tool-session",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot, database_path


def _provider_with_tool_calls(calls: tuple[ToolCall, ...]) -> FakeProvider:
    tool_message = AssistantMessage(
        content=list(calls),
        api="test",
        provider="fake",
        model="codebase-test-model",
        stop_reason="toolUse",
    )
    final_message = AssistantMessage(
        content=[TextContent(text="The verified source defines start.")],
        api="test",
        provider="fake",
        model="codebase-test-model",
        stop_reason="stop",
    )
    return FakeProvider(
        [
            [
                AssistantStartEvent(partial=tool_message.model_copy(deep=True)),
                AssistantDoneEvent(reason="toolUse", message=tool_message),
            ],
            [
                AssistantStartEvent(partial=final_message.model_copy(deep=True)),
                AssistantDoneEvent(reason="stop", message=final_message),
            ],
        ]
    )
