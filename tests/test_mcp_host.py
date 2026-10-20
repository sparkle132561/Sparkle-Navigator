"""MCP Host 的公开行为测试。"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sparkle_navigator.domain.tool_profile import ToolProfile
from sparkle_navigator.mcp import (
    McpConnectionManager,
    McpServerConfig,
    McpToolAccess,
    McpToolCallError,
)
from sparkle_navigator.runtime.mcp_tools import add_mcp_tools
from sparkle_navigator.runtime.tool_profile import (
    ApprovalOutcome,
    TOOL_PROFILE_NAMESPACE,
    ToolProfileRuntime,
)
from tau_agent.events import ToolExecutionEndEvent
from tau_agent.messages import AssistantMessage, TextContent, ToolCall, ToolResultMessage
from tau_agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from tau_agent.session import CustomEntry, InMemorySessionStorage
from tau_ai.fake import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig, TraceRecorder

MCP_TEST_SERVER = Path(__file__).parent / "fixtures" / "mcp_tool_server.py"


def test_stdio_transport_discovers_and_calls_only_allowlisted_tools() -> None:
    asyncio.run(_stdio_transport_discovers_and_calls_only_allowlisted_tools())


def test_streamable_http_transport_discovers_and_calls_allowlisted_tools() -> None:
    asyncio.run(_streamable_http_transport_discovers_and_calls_allowlisted_tools())


def test_mcp_tool_capabilities_use_tau_tool_profile_filter_and_approval() -> None:
    asyncio.run(_mcp_tool_capabilities_use_tau_tool_profile_filter_and_approval())


def test_mcp_adapter_limits_and_redacts_output_and_honors_timeout_cancellation() -> None:
    asyncio.run(
        _mcp_adapter_limits_and_redacts_output_and_honors_timeout_cancellation()
    )


async def _stdio_transport_discovers_and_calls_only_allowlisted_tools() -> None:
    config = McpServerConfig(
        server_id="stdio-test",
        transport="stdio",
        command=sys.executable,
        args=(str(MCP_TEST_SERVER), "--transport", "stdio"),
        allowed_tools={"echo": McpToolAccess.READ_ONLY},
    )

    async with McpConnectionManager((config,)) as manager:
        registrations = manager.tool_registrations
        assert len(registrations) == 1
        registration = registrations[0]
        assert registration.source == "sparkle.mcp:stdio-test"
        assert registration.capabilities == frozenset({"read_only"})
        assert registration.tool.name.startswith("mcp__stdio-test__")

        result = await registration.tool.execute("stdio-call", {"text": "hello"})

    assert result.text == "echo:hello"


async def _streamable_http_transport_discovers_and_calls_allowlisted_tools() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(MCP_TEST_SERVER),
        "--transport",
        "streamable-http",
        "--port",
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await _wait_for_port(process, port)
        config = McpServerConfig(
            server_id="http-echo",
            transport="streamable_http",
            url=f"http://127.0.0.1:{port}/mcp",
            allowed_tools={
                "echo": McpToolAccess.READ_ONLY,
                "slow_echo": McpToolAccess.READ_ONLY,
            },
        )

        async with McpConnectionManager((config,)) as manager:
            registrations = {
                registration.identity.rsplit(":", maxsplit=2)[1]: registration
                for registration in manager.tool_registrations
            }
            assert registrations["echo"].capabilities == frozenset({"read_only"})
            result = await registrations["echo"].tool.execute(
                "http-call",
                {"text": "streamable"},
            )
            cancellation = _CancellationFlag()
            call = asyncio.create_task(
                registrations["slow_echo"].tool.execute(
                    "http-cancel",
                    {"delay_seconds": 2.0},
                    cancellation,
                )
            )
            await asyncio.sleep(0.05)
            cancellation.cancelled = True
            with pytest.raises(asyncio.CancelledError):
                await call

        assert result.text == "echo:streamable"

        timeout_config = McpServerConfig(
            server_id="http-timeout",
            transport="streamable_http",
            url=f"http://127.0.0.1:{port}/mcp",
            allowed_tools={"slow_echo": McpToolAccess.READ_ONLY},
            timeout_seconds=0.25,
        )
        async with McpConnectionManager((timeout_config,)) as timeout_manager:
            (slow_registration,) = timeout_manager.tool_registrations
            with pytest.raises(McpToolCallError, match="超时"):
                await slow_registration.tool.execute(
                    "http-timeout",
                    {"delay_seconds": 2.0},
                )
    finally:
        await _stop_test_server(process)


async def _mcp_tool_capabilities_use_tau_tool_profile_filter_and_approval() -> None:
    config = McpServerConfig(
        server_id="profile-test",
        transport="stdio",
        command=sys.executable,
        args=(str(MCP_TEST_SERVER), "--transport", "stdio"),
        allowed_tools={
            "read_value": McpToolAccess.READ_ONLY,
            "write_value": McpToolAccess.SIDE_EFFECT,
        },
    )
    with TemporaryDirectory() as temp_dir:
        async with McpConnectionManager((config,)) as manager:
            registrations = manager.tool_registrations
            read_name = next(
                registration.tool.name
                for registration in registrations
                if registration.identity.endswith(":read_value:v1")
            )
            write_name = next(
                registration.tool.name
                for registration in registrations
                if registration.identity.endswith(":write_value:v1")
            )

            explain_provider = _provider_with_tool_calls(
                "mcp-profile-explain-model",
                (
                    ToolCall(
                        id="explain-write",
                        name=write_name,
                        arguments={"value": "must-not-run"},
                    ),
                    ToolCall(id="explain-read", name=read_name, arguments={}),
                ),
            )
            explain_session = await CodingSession.load(
                add_mcp_tools(
                    _base_mcp_session_config(
                        explain_provider,
                        Path(temp_dir),
                        tool_access_policy=ToolProfileRuntime(),
                    ),
                    registrations,
                )
            )
            try:
                explain_events = [
                    event async for event in explain_session.prompt("Read MCP state")
                ]
                visible_names = {tool.name for tool in explain_provider.calls[0][3]}
                assert read_name in visible_names
                assert write_name not in visible_names
                explain_results = {
                    message.tool_name: message
                    for message in explain_provider.calls[1][2]
                    if isinstance(message, ToolResultMessage)
                }
                assert explain_results[write_name].is_error
                assert "empty" in explain_results[read_name].text
                assert any(
                    isinstance(event, ToolExecutionEndEvent)
                    and event.tool_name == write_name
                    and event.is_error
                    for event in explain_events
                )
            finally:
                await explain_session.aclose()

            approvals = []

            async def approve(request) -> ApprovalOutcome:
                approvals.append(request)
                return ApprovalOutcome.APPROVED

            coding_provider = _provider_with_tool_calls(
                "mcp-profile-coding-model",
                (
                    ToolCall(
                        id="coding-write",
                        name=write_name,
                        arguments={"value": "approved-state"},
                    ),
                    ToolCall(id="coding-read", name=read_name, arguments={}),
                ),
            )
            coding_storage = InMemorySessionStorage(
                (
                    CustomEntry(
                        namespace=TOOL_PROFILE_NAMESPACE,
                        data={"version": 1, "profile": ToolProfile.CODING.value},
                    ),
                )
            )
            coding_session = await CodingSession.load(
                add_mcp_tools(
                    _base_mcp_session_config(
                        coding_provider,
                        Path(temp_dir),
                        storage=coding_storage,
                        tool_access_policy=ToolProfileRuntime(approver=approve),
                    ),
                    registrations,
                )
            )
            try:
                recorder = TraceRecorder(session_id="mcp-coding-session")
                coding_events = [
                    event
                    async for event in recorder.capture(
                        coding_session.prompt("Update MCP state")
                    )
                ]
                trace_events = recorder.events
                visible_names = {tool.name for tool in coding_provider.calls[0][3]}
                assert {read_name, write_name} <= visible_names
                assert len(approvals) == 1
                assert approvals[0].tool_name == write_name
                coding_results = {
                    message.tool_name: message
                    for message in coding_provider.calls[1][2]
                    if isinstance(message, ToolResultMessage)
                }
                assert not coding_results[write_name].is_error
                assert "approved-state" in coding_results[read_name].text
                assert any(
                    isinstance(event, ToolExecutionEndEvent)
                    and event.tool_name == write_name
                    and not event.is_error
                    for event in coding_events
                )
                assert any(
                    trace.event_type == "tool_execution_start"
                    and trace.event.tool_name == write_name
                    for trace in trace_events
                )
                assert any(
                    trace.event_type == "tool_execution_end"
                    and trace.event.tool_name == write_name
                    and not trace.event.is_error
                    for trace in trace_events
                )
                assert trace_events[-1].event_type == "agent_settled"
                assert [trace.sequence for trace in trace_events] == list(
                    range(1, len(trace_events) + 1)
                )
                assert all(
                    trace.session_id == "mcp-coding-session"
                    for trace in trace_events
                )
                assert all(
                    trace.event is event
                    for trace, event in zip(trace_events, coding_events, strict=True)
                )
            finally:
                await coding_session.aclose()


async def _mcp_adapter_limits_and_redacts_output_and_honors_timeout_cancellation() -> None:
    secret = "mcp-secret-value-13"
    config = McpServerConfig(
        server_id="output-limits",
        transport="stdio",
        command=sys.executable,
        args=(str(MCP_TEST_SERVER), "--transport", "stdio"),
        env={"MCP_TEST_SECRET": secret},
        allowed_tools={
            "echo": McpToolAccess.READ_ONLY,
            "reveal_secret": McpToolAccess.READ_ONLY,
            "slow_echo": McpToolAccess.READ_ONLY,
        },
        timeout_seconds=30.0,
        max_output_chars=48,
    )
    assert secret not in repr(config)

    async with McpConnectionManager((config,)) as manager:
        registrations = {
            registration.identity.rsplit(":", maxsplit=2)[1]: registration
            for registration in manager.tool_registrations
        }
        long_output = await registrations["echo"].tool.execute(
            "echo-long",
            {"text": "x" * 200},
        )
        assert len(long_output.text) <= config.max_output_chars
        assert "MCP output truncated" in long_output.text

        secret_output = await registrations["reveal_secret"].tool.execute(
            "echo-secret",
            {},
        )
        assert secret not in secret_output.text
        assert "[REDACTED]" in secret_output.text

        cancellation = _CancellationFlag()
        call = asyncio.create_task(
            registrations["slow_echo"].tool.execute(
                "slow-cancel",
                {"delay_seconds": 2.0},
                cancellation,
            )
        )
        await asyncio.sleep(0.05)
        cancellation.cancelled = True
        with pytest.raises(asyncio.CancelledError):
            await call

    timeout_config = McpServerConfig(
        server_id="timeout-test",
        transport="stdio",
        command=sys.executable,
        args=(str(MCP_TEST_SERVER), "--transport", "stdio"),
        allowed_tools={"slow_echo": McpToolAccess.READ_ONLY},
        timeout_seconds=0.25,
    )
    async with McpConnectionManager((timeout_config,)) as timeout_manager:
        (slow_registration,) = timeout_manager.tool_registrations
        with pytest.raises(McpToolCallError, match="超时"):
            await slow_registration.tool.execute(
                "slow-timeout",
                {"delay_seconds": 2.0},
            )


class _CancellationFlag:
    def __init__(self) -> None:
        self.cancelled = False

    def is_cancelled(self) -> bool:
        return self.cancelled


async def _wait_for_port(
    process: asyncio.subprocess.Process,
    port: int,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while loop.time() < deadline:
        if process.returncode is not None:
            raise RuntimeError("本地 Streamable HTTP MCP 测试 Server 提前退出。")
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.05)
            continue
        writer.close()
        await writer.wait_closed()
        return
    raise TimeoutError("本地 Streamable HTTP MCP 测试 Server 未及时启动。")


async def _stop_test_server(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


def _base_mcp_session_config(
    provider: FakeProvider,
    cwd: Path,
    *,
    storage: InMemorySessionStorage | None = None,
    tool_access_policy: ToolProfileRuntime,
) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=provider,
        model="mcp-profile-test-model",
        storage=storage or InMemorySessionStorage(),
        cwd=cwd,
        system="Use configured MCP tools only.",
        tool_access_policy=tool_access_policy,
        skills_enabled=False,
        extensions_enabled=False,
        project_extensions_enabled=False,
    )


def _provider_with_tool_calls(model: str, calls: tuple[ToolCall, ...]) -> FakeProvider:
    tool_message = AssistantMessage(
        content=list(calls),
        api="test",
        provider="fake",
        model=model,
        stop_reason="toolUse",
    )
    final_message = AssistantMessage(
        content=[TextContent(text="MCP call flow complete.")],
        api="test",
        provider="fake",
        model=model,
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
