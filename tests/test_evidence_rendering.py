"""验证 Snapshot 工具证据在聊天 Transcript 中的渲染链。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.text import Text

from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.evidence import CodeEvidence
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore
from sparkle_navigator.runtime.codebase_tools import add_codebase_tools
from sparkle_navigator.runtime.evidence_rendering import (
    render_snapshot_evidence_result,
    snapshot_evidence_details,
)
from sparkle_navigator.runtime.tool_profile import ToolProfileRuntime
from tau_agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
)
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import InMemorySessionStorage
from tau_agent.tools import AgentToolResult
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.extensions.runtime import ExtensionRuntime
from tau_coding.tui.app import _session_tool_result_renderer
from tau_coding.tui.state import TuiState


def _evidence(**overrides: object) -> CodeEvidence:
    values: dict[str, object] = {
        "snapshot_id": "a" * 64,
        "path": "src/app.py",
        "line_start": 1,
        "line_end": 2,
        "snippet": "def identity():\n    return 'alpha'",
        "sha256": hashlib.sha256(b"alpha").hexdigest(),
        "source": "symbol",
        "confidence": "CONFIRMED",
    }
    values.update(overrides)
    return CodeEvidence(**values)  # type: ignore[arg-type]


def test_collapsed_render_shows_locations_and_snapshot() -> None:
    result = AgentToolResult(
        content=[],
        details=snapshot_evidence_details((_evidence(), _evidence(path="src/other.py"))),
    )
    markup = render_snapshot_evidence_result(result, expanded=False)
    assert markup is not None
    plain = Text.from_markup(markup).plain
    assert "aaaaaaaa" in plain
    assert "2 条证据" in plain
    assert "src/app.py:1-2" in plain
    assert "src/other.py:1-2" in plain
    # 折叠视图不展开片段。
    assert "return 'alpha'" not in plain


def test_expanded_render_shows_snippet_sha_and_confidence() -> None:
    result = AgentToolResult(
        content=[], details=snapshot_evidence_details((_evidence(),))
    )
    markup = render_snapshot_evidence_result(result, expanded=True)
    assert markup is not None
    plain = Text.from_markup(markup).plain
    assert "src/app.py:1-2" in plain
    assert _evidence().sha256[:8] in plain
    assert "CONFIRMED" in plain
    assert "return 'alpha'" in plain


def test_single_line_evidence_location_uses_one_line_number() -> None:
    result = AgentToolResult(
        content=[],
        details=snapshot_evidence_details((_evidence(line_start=3, line_end=None),)),
    )
    markup = render_snapshot_evidence_result(result, expanded=False)
    assert markup is not None
    assert "src/app.py:3" in Text.from_markup(markup).plain


def test_markup_in_snippet_is_escaped_not_interpreted() -> None:
    result = AgentToolResult(
        content=[],
        details=snapshot_evidence_details(
            (_evidence(snippet="x = items[0]  # [red]not markup[/red]"),),
        ),
    )
    markup = render_snapshot_evidence_result(result, expanded=True)
    assert markup is not None
    plain = Text.from_markup(markup).plain
    assert "items[0]" in plain
    assert "[red]" in plain


def test_more_than_five_locations_truncates_collapsed_summary() -> None:
    items = tuple(
        _evidence(path=f"src/file_{index}.py") for index in range(7)
    )
    result = AgentToolResult(content=[], details=snapshot_evidence_details(items))
    markup = render_snapshot_evidence_result(result, expanded=False)
    assert markup is not None
    plain = Text.from_markup(markup).plain
    assert "7 处" in plain
    assert "src/file_5.py" not in plain


def test_missing_or_malformed_details_fall_back_to_none() -> None:
    no_details = AgentToolResult(content=[])
    assert render_snapshot_evidence_result(no_details, expanded=False) is None

    wrong_marker = AgentToolResult(
        content=[], details={"sparkle_code_evidence_version": 2, "code_evidence": []}
    )
    assert render_snapshot_evidence_result(wrong_marker, expanded=False) is None

    malformed = AgentToolResult(
        content=[],
        details={
            "sparkle_code_evidence_version": 1,
            "code_evidence": [{"path": "broken"}],
        },
    )
    assert render_snapshot_evidence_result(malformed, expanded=False) is None

    non_list = AgentToolResult(
        content=[],
        details={"sparkle_code_evidence_version": 1, "code_evidence": "oops"},
    )
    assert render_snapshot_evidence_result(non_list, expanded=False) is None

    empty = AgentToolResult(
        content=[],
        details={"sparkle_code_evidence_version": 1, "code_evidence": []},
    )
    assert render_snapshot_evidence_result(empty, expanded=False) is None


def test_details_roundtrip_preserves_evidence_fields() -> None:
    items = (_evidence(),)
    details = snapshot_evidence_details(items)
    assert details is not None
    restored = CodeEvidence(**details["code_evidence"][0])
    assert restored == items[0]


class _ExtensionDecliningRuntime(ExtensionRuntime):
    """没有注册任何扩展工具的运行时；render_tool_result 恒为 None。"""


class _SessionStub:
    def __init__(self, tools: tuple) -> None:
        self.registered_tools = tools


class _FakeRegistration:
    def __init__(self, tool) -> None:
        self.tool = tool


def test_session_renderer_falls_back_to_session_tool_render_result() -> None:
    from tau_agent.tools import AgentTool

    session_tool = AgentTool(
        name="explain_code",
        label="explain_code",
        description="",
        parameters={"type": "object"},
        execute_fn=None,  # type: ignore[arg-type]
        render_result=render_snapshot_evidence_result,
    )
    renderer = _session_tool_result_renderer(
        _ExtensionDecliningRuntime(), _SessionStub((_FakeRegistration(session_tool),))
    )
    result = AgentToolResult(
        content=[], details=snapshot_evidence_details((_evidence(),))
    )
    markup = renderer("explain_code", result, expanded=False)
    assert markup is not None
    assert "src/app.py:1-2" in Text.from_markup(markup).plain
    # 未注册的工具返回 None，走通用回退。
    assert renderer("unknown_tool", result, expanded=False) is None
    # 渲染器抛异常时不向 transcript 传播。
    def _boom(result, *, expanded: bool) -> str | None:
        raise RuntimeError("renderer bug")

    broken_tool = AgentTool(
        name="explain_code",
        label="explain_code",
        description="",
        parameters={"type": "object"},
        execute_fn=None,  # type: ignore[arg-type]
        render_result=_boom,
    )
    broken_renderer = _session_tool_result_renderer(
        _ExtensionDecliningRuntime(), _SessionStub((_FakeRegistration(broken_tool),))
    )
    assert broken_renderer("explain_code", result, expanded=False) is None


def test_tui_state_uses_session_renderer_for_tool_results() -> None:
    from tau_agent.tools import AgentTool

    session_tool = AgentTool(
        name="explain_code",
        label="explain_code",
        description="",
        parameters={"type": "object"},
        execute_fn=None,  # type: ignore[arg-type]
        render_result=render_snapshot_evidence_result,
    )
    state = TuiState()
    state.tool_result_renderer = _session_tool_result_renderer(
        _ExtensionDecliningRuntime(), _SessionStub((_FakeRegistration(session_tool),))
    )
    state.add_tool_call(ToolCall(id="call-1", name="explain_code", arguments={}))
    state.record_tool_result(
        "call-1",
        "explain_code",
        AgentToolResult(
            content=[TextContent(text='{"snapshot_id": "..."}')],
            details=snapshot_evidence_details((_evidence(),)),
        ),
        is_error=False,
    )
    (item,) = (entry for entry in state.items if entry.role == "tool")
    markup = state.resolve_tool_result(item, expanded=True)
    assert markup is not None
    assert "src/app.py:1-2" in Text.from_markup(markup).plain


def test_real_tool_run_details_render_through_evidence_card() -> None:
    asyncio.run(_real_tool_run_details_render_through_evidence_card())


async def _real_tool_run_details_render_through_evidence_card() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot, database_path = _register_snapshot(
            root, {"src/app.py": "def start():\n    return 'ready'\n"}
        )
        await SnapshotSymbolIndex(database_path=database_path).build(snapshot)
        await SnapshotCodeGraph(database_path=database_path).build(snapshot)
        await SnapshotTextIndex(database_path=database_path).build(snapshot)
        tool_message = AssistantMessage(
            content=[
                ToolCall(
                    id="render-explain",
                    name="explain_code",
                    arguments={"target": "src/app.py"},
                )
            ],
            api="test",
            provider="fake",
            model="evidence-render-test",
            stop_reason="toolUse",
        )
        final_message = AssistantMessage(
            content=[TextContent(text="start 定义在 src/app.py:1-2。")],
            api="test",
            provider="fake",
            model="evidence-render-test",
            stop_reason="stop",
        )
        provider = FakeProvider(
            [
                [
                    AssistantStartEvent(partial=tool_message.model_copy(deep=True)),
                    AssistantDoneEvent(reason=tool_message.stop_reason, message=tool_message),
                ],
                [
                    AssistantStartEvent(partial=final_message.model_copy(deep=True)),
                    AssistantDoneEvent(reason=final_message.stop_reason, message=final_message),
                ],
            ]
        )
        base = CodingSessionConfig(
            provider=provider,
            model="evidence-render-test",
            cwd=root,
            storage=InMemorySessionStorage(),
            extensions_enabled=False,
            skills_enabled=False,
            trust_default="never",
        )
        transformed = add_codebase_tools(
            replace_tool_policy(base),
            database_path=database_path,
            snapshot_id=snapshot.snapshot_id,
        )
        session = await CodingSession.load(transformed)
        try:
            tool_result: AgentToolResult | None = None
            async for event in session.prompt("explain start"):
                if type(event).__name__ == "ToolExecutionEndEvent":
                    tool_result = event.result
            assert tool_result is not None
            assert "start 定义在" in " ".join(
                message.model_dump_json() for message in session.messages
            )
            markup = render_snapshot_evidence_result(tool_result, expanded=True)
            assert markup is not None
            plain = Text.from_markup(markup).plain
            assert snapshot.snapshot_id[:8] in plain
            assert "src/app.py" in plain
            assert "return 'ready'" in plain
        finally:
            await session.aclose()


def replace_tool_policy(base: CodingSessionConfig) -> CodingSessionConfig:
    from dataclasses import replace

    return replace(
        base,
        tool_access_policy=ToolProfileRuntime(
            approver=None,
            repository_trusted=False,
            workspace_root=base.cwd,
        ),
    )


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
        source_commit="e" * 40,
        files=files,
        snapshot_root=snapshot_root,
    )
    database_path = root / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"evidence-rendering").hexdigest(),
                name="evidence-rendering",
                source_type="local",
                source_locator="test://evidence-rendering",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot, database_path
