"""按证据 key 打开固定 Snapshot 的只读文件视图。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    read_verified_snapshot_files,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore

EVIDENCE_KEY_SEPARATOR = "|"
_MAX_VIEW_LINES = 80


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    """证据卡点击动作携带的定位 key。

    行号 0/0 表示整文件视图（显示前 _MAX_VIEW_LINES 行）；该约定由
    evidence_rendering._clickable_location 写入、此处读取。
    """

    snapshot_id: str
    path: str
    line_start: int
    line_end: int


def parse_evidence_key(key: str) -> EvidenceKey | None:
    """解析证据卡生成的查看 key；格式不符返回 None。"""

    parts = key.split(EVIDENCE_KEY_SEPARATOR)
    if len(parts) != 4:
        return None
    snapshot_id, path, raw_start, raw_end = parts
    if not snapshot_id or not path:
        return None
    try:
        start = int(raw_start)
        end = int(raw_end)
    except ValueError:
        return None
    if start < 0 or end < start:
        return None
    return EvidenceKey(
        snapshot_id=snapshot_id, path=path, line_start=start, line_end=end
    )


def format_evidence_key(key: EvidenceKey) -> str:
    """构造证据卡使用的查看 key。"""

    return (
        f"{key.snapshot_id}{EVIDENCE_KEY_SEPARATOR}{key.path}"
        f"{EVIDENCE_KEY_SEPARATOR}{key.line_start}"
        f"{EVIDENCE_KEY_SEPARATOR}{key.line_end}"
    )


def view_evidence(
    database_path: Path,
    key: str,
) -> tuple[str, str] | None:
    """读取固定 Snapshot 中被点击的行范围；返回 (标题, 正文)。同步、IO 走线程。"""

    parsed = parse_evidence_key(key)
    if parsed is None:
        return None
    store = RepositoryStore(database_path)
    try:
        snapshot = store.get_snapshot(parsed.snapshot_id)
    except KeyError:
        return (
            _title(parsed),
            "无法定位：该 Snapshot 已不存在于本地数据库。",
        )
    location = _location_label(parsed)
    try:
        (verified, _size, content) = read_verified_snapshot_files(
            snapshot, (parsed.path,)
        )[0]
        del verified
        all_lines = (content or "").splitlines()
        if parsed.line_start == 0:
            body_lines = _numbered(all_lines[:_MAX_VIEW_LINES], start_line=1)
        else:
            if parsed.line_start < 1 or parsed.line_end > len(all_lines):
                raise SnapshotFileScanError(
                    f"行号 {parsed.line_start}-{parsed.line_end} 超出文件共 "
                    f"{len(all_lines)} 行。"
                )
            body_lines = _numbered(
                all_lines[parsed.line_start - 1 : parsed.line_end],
                start_line=parsed.line_start,
            )
    except SnapshotFileScanError as exc:
        return (_title(parsed), f"{location}\n\n无法核对：{exc}")
    if not body_lines:
        return (
            _title(parsed),
            f"{location}\n\n文件为空或为二进制内容，无法显示文本。",
        )
    header = f"{location} · Snapshot {parsed.snapshot_id[:8]}（SHA-256 已核对）"
    return _title(parsed), "\n".join((header, "", *body_lines))


def _numbered(lines: list[str], *, start_line: int) -> list[str]:
    return [f"{start_line + offset:>5} | {text}" for offset, text in enumerate(lines)]


def _location_label(parsed: EvidenceKey) -> str:
    if parsed.line_start == 0:
        return parsed.path
    if parsed.line_end == parsed.line_start:
        return f"{parsed.path}:{parsed.line_start}"
    return f"{parsed.path}:{parsed.line_start}-{parsed.line_end}"


def _title(parsed: EvidenceKey) -> str:
    pure = PurePosixPath(parsed.path)
    return f"{pure.name} · Snapshot {parsed.snapshot_id[:8]}"


async def view_evidence_async(
    database_path: Path,
    key: str,
) -> tuple[str, str] | None:
    """线程池中执行 Snapshot 读取。"""

    return await asyncio.to_thread(view_evidence, database_path, key)


__all__ = [
    "EvidenceKey",
    "format_evidence_key",
    "parse_evidence_key",
    "view_evidence",
    "view_evidence_async",
]
