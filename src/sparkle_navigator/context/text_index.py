"""在不可变 Repository Snapshot 上建立和查询 FTS5 文本索引。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from sparkle_navigator.domain.repository import RepositorySnapshot
from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    scan_verified_snapshot_files,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


class SnapshotTextIndexError(RuntimeError):
    """Snapshot 文件与清单不一致或文本索引操作失败。"""


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """一次 Snapshot 索引构建的可观察计数。"""

    snapshot_id: str
    file_count: int
    indexed_file_count: int
    skipped_file_count: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一个 Snapshot 内的关键词命中。"""

    path: str
    snippet: str
    rank: float


class SnapshotTextIndex:
    """校验 Snapshot 文件、持久化文本索引并按 Snapshot 隔离查询。"""

    def __init__(self, *, database_path: Path) -> None:
        self._store = RepositoryStore(database_path)

    async def build(self, snapshot: RepositorySnapshot) -> IndexBuildResult:
        """校验 Snapshot 内容后建立或原子替换该快照的文本索引。"""

        try:
            scanned_files = await asyncio.to_thread(scan_verified_snapshot_files, snapshot)
        except SnapshotFileScanError as exc:
            raise SnapshotTextIndexError(str(exc)) from exc
        self._store.replace_snapshot_file_index(snapshot.snapshot_id, scanned_files)
        indexed_file_count = sum(content is not None for _file, _size, content in scanned_files)
        return IndexBuildResult(
            snapshot_id=snapshot.snapshot_id,
            file_count=len(scanned_files),
            indexed_file_count=indexed_file_count,
            skipped_file_count=len(scanned_files) - indexed_file_count,
        )

    def search_text(
        self,
        snapshot_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[SearchHit, ...]:
        """以空格分隔的关键词做 AND 查询，结果只来自指定 Snapshot。"""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit 必须是正整数。")
        match_query = _literal_match_query(query)
        rows = self._store.search_snapshot_text(
            snapshot_id,
            match_query,
            limit=limit,
        )
        return tuple(
            SearchHit(path=path, snippet=snippet, rank=rank)
            for path, snippet, rank in rows
        )


def _literal_match_query(query: str) -> str:
    terms = query.split()
    if not terms:
        raise ValueError("搜索关键词不能为空。")
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)


__all__ = [
    "IndexBuildResult",
    "SearchHit",
    "SnapshotTextIndex",
    "SnapshotTextIndexError",
]
