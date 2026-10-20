"""把 allowlisted MCP Tool 注册追加到 Tau CodingSession。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from tau_coding.session import CodingSessionConfig
from tau_coding.tools import CodingToolRegistration


def add_mcp_tools(
    config: CodingSessionConfig,
    registrations: Sequence[CodingToolRegistration],
) -> CodingSessionConfig:
    """仅在启用 Tool Profile 时追加 MCP 工具，保留 Tau 默认工具。"""

    if config.tool_access_policy is None:
        raise ValueError("MCP 工具必须通过 Tool Profile 执行前授权。")
    if config.tools is not None or config.tool_registrations is not None:
        raise ValueError("MCP 工具只能追加到 Tau 默认 CodingSession 工具集。")

    additions = tuple(registrations)
    existing_names = {
        registration.tool.name
        for registration in config.additional_tool_registrations
    }
    added_names = tuple(registration.tool.name for registration in additions)
    if len(added_names) != len(set(added_names)) or existing_names & set(added_names):
        raise ValueError("MCP 工具名称与已有宿主工具重复。")

    return replace(
        config,
        additional_tool_registrations=(
            *config.additional_tool_registrations,
            *additions,
        ),
    )


__all__ = ["add_mcp_tools"]
