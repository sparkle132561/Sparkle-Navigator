"""Repository Trust 对 Tool Profile 可见性的契约测试。"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from tau_ai.fake import FakeProvider
from tau_agent.events import ToolExecutionEndEvent
from tau_agent.messages import AssistantMessage, TextContent, ToolCall
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import InMemorySessionStorage
from tau_agent.session.entries import CustomEntry
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.tools import CodingToolRegistration

from sparkle_navigator.domain.tool_profile import ToolProfile, ToolRisk
from sparkle_navigator.runtime.tool_profile import (
    ApprovalOutcome,
    ToolApprovalRequest,
    TOOL_PROFILE_NAMESPACE,
    ToolProfileRuntime,
)


def test_untrusted_repository_hides_side_effect_tools_in_coding_profile(
    tmp_path: Path,
) -> None:
    async def execute(
        _tool_call_id: str,
        _arguments: Mapping[str, JSONValue],
        _signal: object | None = None,
        _on_update: object | None = None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[])

    read_tool = AgentTool(
        name="read",
        label="read",
        description="Read a file",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )
    write_tool = AgentTool(
        name="write",
        label="write",
        description="Write a file",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )
    registrations = (
        CodingToolRegistration(
            tool=read_tool,
            identity="builtin:read",
            source="tau-coding-builtin",
            capabilities=frozenset({"read_only", "file_read"}),
        ),
        CodingToolRegistration(
            tool=write_tool,
            identity="builtin:write",
            source="tau-coding-builtin",
            capabilities=frozenset({"side_effect", "file_write"}),
        ),
    )
    coding_profile = CustomEntry(
        namespace=TOOL_PROFILE_NAMESPACE,
        data={"version": 1, "profile": "coding"},
    )
    runtime = ToolProfileRuntime(repository_trusted=False, workspace_root=tmp_path)

    visible = runtime.filter_tools(registrations, (coding_profile,))

    assert [tool.name for tool in visible] == ["read"]


def test_untrusted_repository_blocks_side_effect_before_approval(tmp_path: Path) -> None:
    asyncio.run(_untrusted_side_effect_is_blocked(tmp_path))


async def _untrusted_side_effect_is_blocked(tmp_path: Path) -> None:
    approval_requests: list[ToolApprovalRequest] = []

    async def approve(request: ToolApprovalRequest) -> ApprovalOutcome:
        approval_requests.append(request)
        return ApprovalOutcome.APPROVED

    tool = AgentTool(
        name="fake_write",
        label="fake write",
        description="A registered side-effect tool",
        parameters={"type": "object", "properties": {}},
        execute_fn=_no_op_executor,
    )
    registration = CodingToolRegistration(
        tool=tool,
        identity="test:fake-write",
        source="test-fixture",
        capabilities=frozenset({"side_effect"}),
    )
    coding_profile = CustomEntry(
        namespace=TOOL_PROFILE_NAMESPACE,
        data={"version": 1, "profile": ToolProfile.CODING.value},
    )
    runtime = ToolProfileRuntime(
        approver=approve,
        repository_trusted=False,
        workspace_root=tmp_path,
    )

    blocked, _reason = await runtime.before_tool_call(
        ToolCall(id="call-untrusted-write", name="fake_write", arguments={}),
        (registration,),
        (coding_profile,),
        cwd=tmp_path,
        shell_command_prefix=None,
    )

    assert blocked
    assert approval_requests == []


def test_untrusted_repository_cannot_select_coding_profile() -> None:
    asyncio.run(_coding_profile_is_rejected())


async def _coding_profile_is_rejected() -> None:
    runtime = ToolProfileRuntime(repository_trusted=False)
    registrations = (
        CodingToolRegistration(
            tool=_tool("read"),
            identity="builtin:read",
            source="tau-coding-builtin",
            capabilities=frozenset({"read_only", "file_read"}),
        ),
        CodingToolRegistration(
            tool=_tool("write"),
            identity="builtin:write",
            source="tau-coding-builtin",
            capabilities=frozenset({"side_effect", "file_write"}),
        ),
    )

    with TemporaryDirectory() as temporary_directory:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="repository-trust-test",
                storage=InMemorySessionStorage(),
                cwd=Path(temporary_directory),
                system="Repository Trust test",
                tool_registrations=registrations,
                tool_access_policy=runtime,
                skills_enabled=False,
                extensions_enabled=False,
                project_extensions_enabled=False,
            )
        )
        try:
            with pytest.raises(PermissionError):
                await runtime.set_profile(session, ToolProfile.CODING)
            assert [tool.name for tool in session.tools] == ["read"]
        finally:
            await session.aclose()


def test_file_tool_rejects_path_outside_working_copy(tmp_path: Path) -> None:
    runtime = ToolProfileRuntime(
        repository_trusted=True,
        workspace_root=tmp_path,
    )
    registration = CodingToolRegistration(
        tool=_tool("read"),
        identity="builtin:read",
        source="tau-coding-builtin",
        capabilities=frozenset({"read_only", "file_read"}),
    )

    blocked, reason = asyncio.run(
        runtime.before_tool_call(
            ToolCall(
                    id="call-outside-working-copy",
                name="read",
                arguments={"path": "../outside.txt"},
            ),
            (registration,),
            (),
            cwd=tmp_path,
            shell_command_prefix=None,
        )
    )

    assert blocked
    assert "Repository Working Copy" in (reason or "")


def test_file_tool_rejects_snapshot_path_when_bound_to_working_copy(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    workspace_root = tmp_path / "working-copy"
    snapshot_root.mkdir()
    workspace_root.mkdir()
    snapshot_file = snapshot_root / "README.md"
    snapshot_file.write_text("immutable baseline\n", encoding="utf-8")
    runtime = ToolProfileRuntime(
        repository_trusted=True,
        workspace_root=workspace_root,
    )
    registration = CodingToolRegistration(
        tool=_tool("read"),
        identity="builtin:read",
        source="tau-coding-builtin",
        capabilities=frozenset({"read_only", "file_read"}),
    )

    blocked, reason = asyncio.run(
        runtime.before_tool_call(
            ToolCall(
                id="call-immutable-snapshot",
                name="read",
                arguments={"path": str(snapshot_file)},
            ),
            (registration,),
            (),
            cwd=workspace_root,
            shell_command_prefix=None,
        )
    )

    assert blocked
    assert "Repository Working Copy" in (reason or "")


def test_tau_write_changes_working_copy_without_changing_snapshot(
    tmp_path: Path,
) -> None:
    asyncio.run(_tau_write_isolated_to_working_copy(tmp_path))


async def _tau_write_isolated_to_working_copy(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    workspace_root = tmp_path / "working-copy"
    snapshot_root.mkdir()
    snapshot_file = snapshot_root / "README.md"
    snapshot_file.write_text("snapshot version\n", encoding="utf-8")
    shutil.copytree(snapshot_root, workspace_root)
    workspace_file = workspace_root / "README.md"
    approval_requests: list[ToolApprovalRequest] = []

    async def approve(request: ToolApprovalRequest) -> ApprovalOutcome:
        approval_requests.append(request)
        return ApprovalOutcome.APPROVED

    tool_message = AssistantMessage(
        content=[
            ToolCall(
                id="call-write-working-copy",
                name="write",
                arguments={"path": "README.md", "content": "agent edit\n"},
            )
        ],
        api="test",
        provider="fake",
        model="repository-working-copy-test",
        stop_reason="toolUse",
    )
    final_message = AssistantMessage(
        content=[TextContent(text="updated working copy")],
        api="test",
        provider="fake",
        model="repository-working-copy-test",
        stop_reason="stop",
    )
    provider = FakeProvider(
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
    runtime = ToolProfileRuntime(
        approver=approve,
        repository_trusted=True,
        workspace_root=workspace_root,
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="repository-working-copy-test",
            storage=InMemorySessionStorage(),
            cwd=workspace_root,
            system="Write only inside the Repository Working Copy",
            tool_access_policy=runtime,
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )
    )
    try:
        await runtime.set_profile(session, ToolProfile.CODING)
        events = [event async for event in session.prompt("Update README")]

        assert workspace_file.read_text(encoding="utf-8") == "agent edit\n"
        assert snapshot_file.read_text(encoding="utf-8") == "snapshot version\n"
        assert len(approval_requests) == 1
        assert approval_requests[0].diff_preview is not None
        assert any(tool.name == "write" for tool in provider.calls[0][3])
        assert any(
            isinstance(event, ToolExecutionEndEvent) and not event.is_error
            for event in events
        )
    finally:
        await session.aclose()


def test_tau_builtin_read_registration_declares_file_read_capability(
    tmp_path: Path,
) -> None:
    asyncio.run(_read_tool_declares_file_capability(tmp_path))


async def _read_tool_declares_file_capability(workspace_root: Path) -> None:
    runtime = ToolProfileRuntime(
        repository_trusted=True,
        workspace_root=workspace_root,
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="repository-file-capability-test",
            storage=InMemorySessionStorage(),
            cwd=workspace_root,
            system="Repository file capability test",
            tool_access_policy=runtime,
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )
    )
    try:
        read_registration = next(
            item for item in session.registered_tools if item.tool.name == "read"
        )
        assert "file_read" in read_registration.capabilities
    finally:
        await session.aclose()


async def _no_op_executor(
    _tool_call_id: str,
    _arguments: Mapping[str, JSONValue],
    _signal: object | None = None,
    _on_update: object | None = None,
) -> AgentToolResult:
    return AgentToolResult(content=[])


def _tool(name: str) -> AgentTool:
    return AgentTool(
        name=name,
        label=name,
        description=f"Test {name} tool",
        parameters={"type": "object", "properties": {}},
        execute_fn=_no_op_executor,
    )
