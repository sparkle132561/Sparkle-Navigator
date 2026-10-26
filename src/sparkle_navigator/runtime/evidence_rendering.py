"""把 Snapshot 工具结果中的代码证据渲染为聊天 Transcript 证据块。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import cast, get_args

from rich.markup import escape

from sparkle_navigator.domain.code_index import (
    RelationshipConfidence,
    RelationshipType,
)
from sparkle_navigator.domain.evidence import CodeEvidence, EvidenceSource
from tau_agent.tools import AgentToolResult
from tau_agent.types import JSONValue

EVIDENCE_DETAILS_MARKER = "sparkle_code_evidence_version"
EVIDENCE_DETAILS_VERSION = 1
_MAX_SUMMARY_LOCATIONS = 5
_MAX_SNIPPET_LINES = 20

# 词汇随域类型派生，避免与 Literal 定义脱节。
_EVIDENCE_SOURCES = frozenset(get_args(EvidenceSource))
_EVIDENCE_CONFIDENCES = frozenset(get_args(RelationshipConfidence))
_EVIDENCE_RELATIONSHIPS = frozenset(get_args(RelationshipType))


def snapshot_evidence_details(items: tuple[CodeEvidence, ...]) -> dict[str, JSONValue] | None:
    """构造工具结果 details 中的证据 payload；无证据时返回 None。"""

    evidence = tuple(item for item in items if isinstance(item, CodeEvidence))
    if not evidence:
        return None
    return {
        EVIDENCE_DETAILS_MARKER: EVIDENCE_DETAILS_VERSION,
        "code_evidence": [asdict(item) for item in evidence],
    }


def render_snapshot_evidence_result(
    result: AgentToolResult,
    *,
    expanded: bool,
) -> str | None:
    """渲染 Snapshot 工具结果为证据块；无法解析时返回 None 走通用回退。

    证据块只陈述工具已核对的事实（路径、行范围、SHA-256、关系置信度），
    不代表证据必然支持模型的回答结论。
    """

    items = _evidence_from_details(result.details)
    if not items:
        return None
    snapshot_id = items[0].snapshot_id
    if expanded:
        return _render_expanded(snapshot_id, items)
    return _render_collapsed(snapshot_id, items)


def _evidence_from_details(details: JSONValue) -> tuple[CodeEvidence, ...]:
    if (
        not isinstance(details, Mapping)
        or details.get(EVIDENCE_DETAILS_MARKER) != EVIDENCE_DETAILS_VERSION
    ):
        return ()
    raw = details.get("code_evidence")
    if not isinstance(raw, list):
        return ()
    items: list[CodeEvidence] = []
    for entry in raw:
        evidence = _evidence_from_entry(entry)
        if evidence is None:
            return ()
        items.append(evidence)
    return tuple(items)


def _evidence_from_entry(entry: JSONValue) -> CodeEvidence | None:
    if not isinstance(entry, Mapping):
        return None
    snapshot_id = entry.get("snapshot_id")
    path = entry.get("path")
    snippet = entry.get("snippet")
    sha256 = entry.get("sha256")
    source = entry.get("source")
    confidence = entry.get("confidence", "CONFIRMED")
    if not (
        isinstance(snapshot_id, str)
        and isinstance(path, str)
        and isinstance(snippet, str)
        and isinstance(sha256, str)
        and isinstance(source, str)
        and isinstance(confidence, str)
    ):
        return None
    if source not in _EVIDENCE_SOURCES or confidence not in _EVIDENCE_CONFIDENCES:
        return None
    line_start = _optional_line_number(entry.get("line_start"))
    line_end = _optional_line_number(entry.get("line_end"))
    if (line_start is None and entry.get("line_start") is not None) or (
        line_end is None and entry.get("line_end") is not None
    ):
        return None
    symbol_name = entry.get("symbol_name")
    if symbol_name is not None and not isinstance(symbol_name, str):
        return None
    relationship_type = entry.get("relationship_type")
    if relationship_type is not None and (
        not isinstance(relationship_type, str)
        or relationship_type not in _EVIDENCE_RELATIONSHIPS
    ):
        return None
    try:
        return CodeEvidence(
            snapshot_id=snapshot_id,
            path=path,
            line_start=line_start,
            line_end=line_end,
            snippet=snippet,
            sha256=sha256,
            source=cast("EvidenceSource", source),
            confidence=cast("RelationshipConfidence", confidence),
            symbol_name=symbol_name,
            relationship_type=cast("RelationshipType | None", relationship_type),
        )
    except (TypeError, ValueError):
        return None


def _optional_line_number(value: JSONValue) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _render_collapsed(snapshot_id: str, items: tuple[CodeEvidence, ...]) -> str:
    locations = tuple(
        _clickable_location(snapshot_id, item) for item in items
    )
    shown = "、".join(locations[:_MAX_SUMMARY_LOCATIONS])
    suffix = (
        f" 等 {len(locations)} 处" if len(locations) > _MAX_SUMMARY_LOCATIONS else ""
    )
    return (
        f"[dim]Snapshot {escape(snapshot_id[:8])} · {len(items)} 条证据 · "
        f"{shown}{suffix}[/dim]"
    )


def _render_expanded(snapshot_id: str, items: tuple[CodeEvidence, ...]) -> str:
    lines = [f"[dim]Snapshot {escape(snapshot_id[:8])} · {len(items)} 条证据[/dim]"]
    for index, item in enumerate(items, start=1):
        header = (
            f"[dim]{index}. {_clickable_location(snapshot_id, item)}"
            f" · sha256 {escape(item.sha256[:8])}"
            f" · {escape(item.confidence)}[/dim]"
        )
        lines.append(header)
        lines.extend(f"    {escape(line)}" for line in _snippet_lines(item.snippet))
    return "\n".join(lines)


def _clickable_location(snapshot_id: str, item: CodeEvidence) -> str:
    """把证据位置包成宿主查看动作；key 由 evidence_viewer 负责解析。

    显示文本经 Rich 转义。key 以 ``|`` 分隔（Snapshot ID 为十六进制，行号为
    整数）；Snapshot 路径来自 git 清单，可能含引号、方括号或 ``|``——这类
    路径会破坏动作串或 key 解析，此时降级为不可点击的纯文本位置。
    行号 0 表示整文件视图（与 evidence_viewer.EvidenceKey 约定一致）。
    """

    start = item.line_start if item.line_start is not None else 0
    end = item.line_end if item.line_end is not None else start
    if any(char in item.path for char in "|[]'"):
        return escape(item.location)
    key = f"{snapshot_id}|{item.path}|{start}|{end}"
    return f"[@click=app.view_product_evidence('{key}')]{escape(item.location)}[/]"


def _snippet_lines(snippet: str) -> tuple[str, ...]:
    lines = snippet.splitlines() or (snippet,)
    if len(lines) <= _MAX_SNIPPET_LINES:
        return tuple(lines)
    return (*lines[:_MAX_SNIPPET_LINES], "…（片段过长，已截断显示）")


__all__ = [
    "EVIDENCE_DETAILS_MARKER",
    "EVIDENCE_DETAILS_VERSION",
    "render_snapshot_evidence_result",
    "snapshot_evidence_details",
]
