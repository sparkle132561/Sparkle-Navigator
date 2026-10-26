"""解析回答中的 path:line 引用并对照固定 Snapshot 验证。"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    read_verified_snapshot_files,
)
from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile
from sparkle_navigator.persistence.repository_store import RepositoryStore

# 路径必须带扩展名，避免把 "http://host:8080"、时间戳等误认成引用。
_CITATION_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_\-./\\]+\.[A-Za-z0-9]+):(?P<start>\d{1,6})(?:-(?P<end>\d{1,6}))?"
)

CitationStatus = Literal["verified", "missing_path", "out_of_range", "unverifiable"]
_STATUS_LABELS: dict[CitationStatus, str] = {
    "verified": "有效",
    "missing_path": "未在当前 Snapshot 中找到",
    "out_of_range": "行号超出文件范围",
    "unverifiable": "无法核对",
}

# 呈现给用户的固定边界：核对只确认引用位置存在于固定 Snapshot。
_CITATION_BOUNDARY_NOTE = "（仅核对位置存在于固定 Snapshot，不代表证据支持回答结论）"


@dataclass(frozen=True, slots=True)
class CitationRef:
    """回答文本中提取出的一个待核对引用。"""

    raw: str
    path: str
    line_start: int | None
    line_end: int | None


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """一条引用在指定 Snapshot 中的核对结果。"""

    ref: CitationRef
    status: CitationStatus
    detail: str | None = None


def extract_citations(text: str) -> tuple[CitationRef, ...]:
    """按出现顺序提取去重后的 path:line(-end) 引用候选。"""

    refs: list[CitationRef] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for match in _CITATION_PATTERN.finditer(text):
        raw_path = match.group("path").replace("\\", "/")
        if not any(char.isalpha() for char in raw_path):
            # 纯数字-点分段（版本号、时间等）不是文件路径。
            continue
        line_start = int(match.group("start"))
        raw_end = match.group("end")
        line_end = int(raw_end) if raw_end is not None else None
        key = (raw_path, line_start, line_end)
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            CitationRef(
                raw=match.group(0),
                path=raw_path,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return tuple(refs)


def verify_citations(
    database_path: Path,
    *,
    repository_id: str,
    snapshot_id: str,
    citations: tuple[CitationRef, ...],
) -> tuple[CitationCheck, ...]:
    """对照 Snapshot 文件清单与实际内容核对引用；同步调用，IO 放线程执行。"""

    if not citations:
        return ()
    store = RepositoryStore(database_path)
    try:
        snapshot = store.get_repository_snapshot(repository_id, snapshot_id)
    except KeyError:
        # 绑定记录存在但数据库中已无对应 Snapshot（如换库）时按无法核对处理，
        # 不让回合收尾因失效绑定中断。
        return tuple(
            CitationCheck(
                ref=ref,
                status="unverifiable",
                detail="Snapshot 不在本地数据库中，无法核对。",
            )
            for ref in citations
        )
    files_by_path = {item.path: item for item in snapshot.files}
    checks: list[CitationCheck] = []
    for ref in citations:
        checks.append(
            _check_ref(ref, snapshot, files_by_path.get(ref.path))
        )
    return tuple(checks)


def _check_ref(
    ref: CitationRef,
    snapshot: RepositorySnapshot,
    snapshot_file: SnapshotFile | None,
) -> CitationCheck:
    if snapshot_file is None:
        return CitationCheck(ref=ref, status="missing_path")
    if ref.line_start is None:
        return CitationCheck(ref=ref, status="verified")
    try:
        (_verified, _size, content) = read_verified_snapshot_files(
            snapshot, (ref.path,)
        )[0]
    except SnapshotFileScanError as exc:
        return CitationCheck(ref=ref, status="unverifiable", detail=str(exc))
    if content is None:
        # 二进制或非 UTF-8 文件没有可核对的文本行号。
        return CitationCheck(
            ref=ref,
            status="unverifiable",
            detail=f"{ref.path} 不是文本文件，无法核对行号。",
        )
    line_count = len(content.splitlines())
    line_end = ref.line_end if ref.line_end is not None else ref.line_start
    if ref.line_start < 1 or line_end < ref.line_start or line_end > line_count:
        return CitationCheck(
            ref=ref,
            status="out_of_range",
            detail=f"{ref.path} 共 {line_count} 行，引用为 "
            f"{ref.line_start}-{line_end}",
        )
    return CitationCheck(ref=ref, status="verified")


async def verify_citations_async(
    database_path: Path,
    *,
    repository_id: str,
    snapshot_id: str,
    citations: tuple[CitationRef, ...],
) -> tuple[CitationCheck, ...]:
    """线程池中执行 Snapshot 读取与验证。"""

    return await asyncio.to_thread(
        verify_citations,
        database_path,
        repository_id=repository_id,
        snapshot_id=snapshot_id,
        citations=citations,
    )


def format_citation_checks(checks: tuple[CitationCheck, ...]) -> str | None:
    """渲染引用核对行；没有引用时返回 None。"""

    if not checks:
        return None
    parts = []
    for check in checks:
        location = _location(check.ref)
        label = _STATUS_LABELS[check.status]
        if check.status == "verified":
            parts.append(f"{location} {label}")
        else:
            detail = f"（{check.detail}）" if check.detail else ""
            parts.append(f"{location} {label}{detail}")
    return "引用核对：" + "；".join(parts) + _CITATION_BOUNDARY_NOTE


def _location(ref: CitationRef) -> str:
    if ref.line_start is None:
        return ref.path
    if ref.line_end is None or ref.line_end == ref.line_start:
        return f"{ref.path}:{ref.line_start}"
    return f"{ref.path}:{ref.line_start}-{ref.line_end}"


__all__ = [
    "CitationCheck",
    "CitationRef",
    "CitationStatus",
    "extract_citations",
    "format_citation_checks",
    "verify_citations",
    "verify_citations_async",
]
