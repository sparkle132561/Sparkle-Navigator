"""管理 MCP Client 的连接生命周期、tools/list 分页和 allowlist。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import Tool

from tau_coding.tools import CodingToolRegistration

from sparkle_navigator.mcp.config import McpServerConfig
from sparkle_navigator.mcp.tool_adapter import McpToolAdapter


class McpConnectionError(RuntimeError):
    """MCP Server 连接、能力发现或 allowlist 校验失败。"""


class McpConnectionManager:
    """为每个显式配置的 Server 持有独立 MCP Client。"""

    def __init__(self, servers: Sequence[McpServerConfig]) -> None:
        self._servers = tuple(servers)
        server_ids = tuple(server.server_id for server in self._servers)
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("MCP server_id 不能重复。")
        self._stack: AsyncExitStack | None = None
        self._tool_registrations: tuple[CodingToolRegistration, ...] = ()

    async def __aenter__(self) -> McpConnectionManager:
        if self._stack is not None:
            raise RuntimeError("McpConnectionManager 不能重复进入。")

        stack = AsyncExitStack()
        await stack.__aenter__()
        registrations: list[CodingToolRegistration] = []
        try:
            for config in self._servers:
                try:
                    async with asyncio.timeout(config.connect_timeout_seconds):
                        client = await stack.enter_async_context(
                            Client(
                                _client_target(config),
                                read_timeout_seconds=config.connect_timeout_seconds,
                            )
                        )
                        definitions = await _list_all_tools(client)
                except TimeoutError as exc:
                    raise McpConnectionError(
                        f"MCP Server {config.server_id!r} 连接或发现工具超时。"
                    ) from exc
                except Exception as exc:
                    raise McpConnectionError(
                        f"MCP Server {config.server_id!r} 连接或发现工具失败："
                        f"{type(exc).__name__}"
                    ) from exc

                discovered_names = {tool.name for tool in definitions}
                missing_tools = set(config.allowed_tools) - discovered_names
                if missing_tools:
                    raise McpConnectionError(
                        f"MCP Server {config.server_id!r} 未发现 allowlist 工具："
                        f"{sorted(missing_tools)}"
                    )

                adapter = McpToolAdapter(config, client)
                registrations.extend(
                    adapter.adapt(definition)
                    for definition in definitions
                    if definition.name in config.allowed_tools
                )

            model_names = tuple(
                registration.tool.name for registration in registrations
            )
            if len(model_names) != len(set(model_names)):
                raise McpConnectionError("不同 MCP Tool 映射到了重复的模型工具名。")
            self._stack = stack
            self._tool_registrations = tuple(registrations)
            return self
        except BaseException:
            await stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc, traceback) -> bool | None:
        stack = self._stack
        self._stack = None
        self._tool_registrations = ()
        if stack is None:
            return None
        return await stack.__aexit__(exc_type, exc, traceback)

    @property
    def tool_registrations(self) -> tuple[CodingToolRegistration, ...]:
        """返回当前连接期间已发现且允许暴露的 Tau Tool 注册。"""

        if self._stack is None:
            raise RuntimeError("读取 MCP 工具前必须先进入连接管理器。")
        return self._tool_registrations


def _client_target(config: McpServerConfig) -> StdioServerParameters | str:
    if config.transport == "streamable_http":
        if config.url is None:
            raise McpConnectionError("Streamable HTTP Server 缺少 URL。")
        return config.url
    if config.command is None:
        raise McpConnectionError("stdio Server 缺少 command。")
    return StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env=dict(config.env),
        cwd=config.cwd,
    )


async def _list_all_tools(client: Client) -> tuple[Tool, ...]:
    tools: list[Tool] = []
    seen_names: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None
    while True:
        page = await client.list_tools(cursor=cursor)
        for tool in page.tools:
            if tool.name in seen_names:
                raise McpConnectionError(f"MCP tools/list 返回重复工具名：{tool.name}")
            seen_names.add(tool.name)
            tools.append(tool)
        cursor = page.next_cursor
        if cursor is None:
            return tuple(tools)
        if cursor in seen_cursors:
            raise McpConnectionError("MCP tools/list 返回循环分页 cursor。")
        seen_cursors.add(cursor)


__all__ = ["McpConnectionError", "McpConnectionManager"]
