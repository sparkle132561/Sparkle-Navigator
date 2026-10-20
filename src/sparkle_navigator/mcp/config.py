"""MCP Server 与 Tool Allowlist 的显式运行配置。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit


class McpToolAccess(StrEnum):
    """宿主为单个 MCP Tool 明确声明的能力等级。"""

    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"


McpTransport = Literal["stdio", "streamable_http"]


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """一个由用户显式配置的 MCP Server 及其 Tool Allowlist。"""

    server_id: str
    transport: McpTransport
    allowed_tools: Mapping[str, McpToolAccess]
    command: str | None = field(default=None, repr=False)
    args: tuple[str, ...] = field(default=(), repr=False)
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    cwd: Path | None = field(default=None, repr=False)
    url: str | None = field(default=None, repr=False)
    connect_timeout_seconds: float = 15.0
    timeout_seconds: float = 30.0
    max_output_chars: int = 20_000

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.server_id) is None:
            raise ValueError("MCP server_id 只能包含字母、数字、点、下划线和连字符。")
        if self.transport not in {"stdio", "streamable_http"}:
            raise ValueError("MCP transport 只支持 stdio 或 streamable_http。")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("MCP timeout_seconds 必须是有限正数。")
        if (
            not math.isfinite(self.connect_timeout_seconds)
            or self.connect_timeout_seconds <= 0
        ):
            raise ValueError("MCP connect_timeout_seconds 必须是有限正数。")
        if (
            not isinstance(self.max_output_chars, int)
            or isinstance(self.max_output_chars, bool)
            or self.max_output_chars < 1
        ):
            raise ValueError("MCP max_output_chars 必须是正整数。")

        if self.transport == "stdio":
            if not isinstance(self.command, str) or not self.command.strip():
                raise ValueError("stdio Server 必须配置 command。")
            if self.url is not None:
                raise ValueError("stdio Server 不能同时配置 HTTP URL。")
        else:
            if self.command is not None or self.args or self.env or self.cwd is not None:
                raise ValueError("Streamable HTTP Server 不能配置 stdio 参数。")
            if not isinstance(self.url, str):
                raise ValueError("Streamable HTTP Server 必须配置 URL。")
            parsed_url = urlsplit(self.url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
                raise ValueError("Streamable HTTP URL 必须是有效的 http/https 地址。")
            if parsed_url.username is not None or parsed_url.password is not None:
                raise ValueError("MCP HTTP URL 不能在 URL 中内嵌账号或密码。")

        if not isinstance(self.allowed_tools, Mapping):
            raise ValueError("MCP allowed_tools 必须是 Tool 名称到能力等级的映射。")
        normalized_access: dict[str, McpToolAccess] = {}
        for tool_name, access in self.allowed_tools.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("MCP allowlist 中的 Tool 名称不能为空。")
            try:
                normalized_access[tool_name] = McpToolAccess(access)
            except ValueError as exc:
                raise ValueError(f"MCP Tool 能力等级无效：{tool_name}") from exc

        if any(not isinstance(argument, str) for argument in self.args):
            raise ValueError("stdio args 必须全部是字符串。")
        normalized_env: dict[str, str] = {}
        for name, value in self.env.items():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
                or not isinstance(value, str)
            ):
                raise ValueError("stdio 环境变量必须使用有效名称和字符串值。")
            normalized_env[name] = value

        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "allowed_tools", MappingProxyType(normalized_access))
        object.__setattr__(self, "env", MappingProxyType(normalized_env))


__all__ = ["McpServerConfig", "McpToolAccess", "McpTransport"]
