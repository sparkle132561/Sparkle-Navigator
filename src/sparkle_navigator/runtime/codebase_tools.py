"""把 Sparkle Snapshot 浏览能力注册为 Tau CodingSession 只读工具。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path

from sparkle_navigator.context.snapshot_explorer import SnapshotExplorer
from sparkle_navigator.runtime.evidence_rendering import (
    render_snapshot_evidence_result,
    snapshot_evidence_details,
)
from tau_agent.messages import TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.session import CodingSessionConfig
from tau_coding.tools import CodingToolRegistration

ToolOperation = Callable[[Mapping[str, JSONValue]], Awaitable[object]]
_TOOL_SOURCE = "sparkle.context-engine"
_READ_ONLY_CAPABILITIES = frozenset({"read_only"})


def add_codebase_tools(
    config: CodingSessionConfig,
    *,
    database_path: Path,
    snapshot_id: str,
) -> CodingSessionConfig:
    """返回保留 Tau 默认工具并追加 Snapshot 工具的 Session 配置。"""

    if config.tool_access_policy is None:
        raise ValueError("代码库工具必须与 Tool Profile 执行前授权一起配置。")
    if config.tools is not None or config.tool_registrations is not None:
        raise ValueError("代码库工具只能追加到 Tau 默认 CodingSession 工具集。")

    additions = build_codebase_tool_registrations(
        database_path=database_path,
        snapshot_id=snapshot_id,
    )
    existing_names = {item.tool.name for item in config.additional_tool_registrations}
    duplicate_names = existing_names & {item.tool.name for item in additions}
    if duplicate_names:
        raise ValueError(f"代码库工具名称重复：{sorted(duplicate_names)}")
    return replace(
        config,
        additional_tool_registrations=(
            *config.additional_tool_registrations,
            *additions,
        ),
    )


def build_codebase_tool_registrations(
    *,
    database_path: Path,
    snapshot_id: str,
) -> tuple[CodingToolRegistration, ...]:
    """建立固定 Snapshot 对应的 Overview/Ask/Explain/Trace/Impact 工具。"""

    explorer = SnapshotExplorer(database_path=database_path)

    async def overview(_arguments: Mapping[str, JSONValue]) -> object:
        return await explorer.overview(snapshot_id)

    async def ask(arguments: Mapping[str, JSONValue]) -> object:
        return await explorer.ask(
            snapshot_id,
            _required_string(arguments, "question"),
        )

    async def explain(arguments: Mapping[str, JSONValue]) -> object:
        return await explorer.explain(
            snapshot_id,
            _required_string(arguments, "target"),
        )

    async def trace(arguments: Mapping[str, JSONValue]) -> object:
        return await explorer.trace_call(
            snapshot_id,
            _required_string(arguments, "start_path"),
            _required_string(arguments, "start_symbol"),
            max_depth=_optional_positive_integer(arguments, "max_depth", default=3),
        )

    async def impact(arguments: Mapping[str, JSONValue]) -> object:
        target_symbol = arguments.get("target_symbol")
        if target_symbol is not None and not isinstance(target_symbol, str):
            raise ValueError("target_symbol 必须是字符串。")
        return await explorer.impact(
            snapshot_id,
            _required_string(arguments, "target_path"),
            target_symbol=target_symbol,
            max_depth=_optional_positive_integer(arguments, "max_depth", default=3),
        )

    tool_specs = (
        (
            "overview_repository",
            "列出当前固定 Snapshot 的目录、文件、测试与配置文件概览，并返回 README/项目配置证据。",
            _object_schema(),
            overview,
        ),
        (
            "ask_codebase",
            "在当前固定 Snapshot 中检索问题相关代码；回答必须以工具返回的路径、行号和哈希证据为依据。",
            _object_schema(
                {"question": {"type": "string", "description": "要查找证据的问题。"}},
                required=("question",),
            ),
            ask,
        ),
        (
            "explain_code",
            "按 Snapshot 文件路径或精确 Python 符号名取得源码、声明和直接静态关系证据。",
            _object_schema(
                {"target": {"type": "string", "description": "Snapshot 内文件路径或符号名。"}},
                required=("target",),
            ),
            explain,
        ),
        (
            "trace_call",
            "沿当前 Snapshot 的静态调用关系追踪函数，并返回调用点行号和关系置信度。",
            _object_schema(
                {
                    "start_path": {"type": "string"},
                    "start_symbol": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 1, "default": 3},
                },
                required=("start_path", "start_symbol"),
            ),
            trace,
        ),
        (
            "analyze_impact",
            "查询目标符号的静态调用者、目标文件的依赖者和文本命中的测试候选；不推断动态调用。",
            _object_schema(
                {
                    "target_path": {"type": "string"},
                    "target_symbol": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 1, "default": 3},
                },
                required=("target_path",),
            ),
            impact,
        ),
    )
    return tuple(
        CodingToolRegistration(
            tool=_agent_tool(name, description, schema, operation),
            identity=f"{_TOOL_SOURCE}:{name}:v1",
            source=_TOOL_SOURCE,
            capabilities=_READ_ONLY_CAPABILITIES,
        )
        for name, description, schema, operation in tool_specs
    )


def _agent_tool(
    name: str,
    description: str,
    parameters: Mapping[str, JSONValue],
    operation: ToolOperation,
) -> AgentTool:
    async def execute(
        _tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del on_update
        if signal is not None and signal.is_cancelled():
            raise asyncio.CancelledError
        result = await operation(arguments)
        evidence = getattr(result, "evidence", None)
        items = getattr(evidence, "items", None)
        return AgentToolResult(
            content=[
                TextContent(
                    text=json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
                )
            ],
            details=snapshot_evidence_details(items)
            if isinstance(items, tuple)
            else None,
        )

    return AgentTool(
        name=name,
        label=name,
        description=description,
        parameters=parameters,
        execute_fn=execute,
        execution_mode="sequential",
        prompt_snippet="只引用此工具返回的 Snapshot 证据；证据不足时明确说明，不要猜测。",
        render_result=render_snapshot_evidence_result,
    )


def _required_string(arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串。")
    return value


def _optional_positive_integer(
    arguments: Mapping[str, JSONValue],
    name: str,
    *,
    default: int,
) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} 必须是正整数。")
    return value


def _object_schema(
    properties: Mapping[str, JSONValue] | None = None,
    *,
    required: tuple[str, ...] = (),
) -> Mapping[str, JSONValue]:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": False,
    }


__all__ = ["add_codebase_tools", "build_codebase_tool_registrations"]
