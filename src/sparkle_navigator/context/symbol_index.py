"""在固定 Repository Snapshot 上建立和查询 Python 符号索引。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from sparkle_navigator.context._snapshot_files import (
    SnapshotFileScanError,
    scan_verified_snapshot_files,
)
from sparkle_navigator.context.python_symbols import (
    ParseDiagnostic,
    parse_python_file,
)
from sparkle_navigator.domain.code_index import CodeSymbol, ImportRecord, SymbolKind
from sparkle_navigator.domain.repository import RepositorySnapshot, SnapshotFile
from sparkle_navigator.persistence.repository_store import RepositoryStore


class SnapshotSymbolIndexError(RuntimeError):
    """Snapshot 校验失败或符号索引操作失败。"""


@dataclass(frozen=True, slots=True)
class SymbolIndexBuildResult:
    """一次 Snapshot 符号索引构建的结果。"""

    snapshot_id: str
    python_file_count: int
    symbol_count: int
    import_count: int
    diagnostics: tuple[ParseDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class SymbolSearchHit:
    """一个 Snapshot 中按名称找到的代码符号。"""

    path: str
    name: str
    qualified_name: str
    kind: SymbolKind
    line_start: int
    line_end: int


class SnapshotSymbolIndex:
    """校验 Snapshot 内容、持久化 Python 声明并按 Snapshot 隔离查询。"""

    def __init__(self, *, database_path: Path) -> None:
        self._store = RepositoryStore(database_path)

    async def build(self, snapshot: RepositorySnapshot) -> SymbolIndexBuildResult:
        """校验 Snapshot 文件后解析 Python 声明并原子替换索引。"""

        try:
            scanned_files = await asyncio.to_thread(scan_verified_snapshot_files, snapshot)
        except SnapshotFileScanError as exc:
            raise SnapshotSymbolIndexError(str(exc)) from exc

        python_file_count, symbols, imports, diagnostics = await asyncio.to_thread(
            _parse_python_files,
            scanned_files,
        )
        self._store.replace_snapshot_symbol_index(snapshot.snapshot_id, symbols, imports)
        return SymbolIndexBuildResult(
            snapshot_id=snapshot.snapshot_id,
            python_file_count=python_file_count,
            symbol_count=len(symbols),
            import_count=len(imports),
            diagnostics=diagnostics,
        )

    def search_symbols(self, snapshot_id: str, name: str) -> tuple[SymbolSearchHit, ...]:
        """按声明名或限定名查找指定 Snapshot 中的符号。"""

        if not name.strip():
            raise ValueError("符号名称不能为空。")
        rows = self._store.search_snapshot_symbols(snapshot_id, name)
        return tuple(
            SymbolSearchHit(
                path=path,
                name=symbol.name,
                qualified_name=symbol.qualified_name,
                kind=symbol.kind,
                line_start=symbol.line_start,
                line_end=symbol.line_end,
            )
            for path, symbol in rows
        )

    def imports_for_file(self, snapshot_id: str, path: str) -> tuple[ImportRecord, ...]:
        """读取指定 Snapshot 文件中的 import 语法记录。"""

        return self._store.get_snapshot_imports(snapshot_id, path)


def _parse_python_files(
    scanned_files: tuple[tuple[SnapshotFile, int, str | None], ...],
) -> tuple[
    int,
    tuple[tuple[str, CodeSymbol], ...],
    tuple[tuple[str, ImportRecord], ...],
    tuple[ParseDiagnostic, ...],
]:
    symbols: list[tuple[str, CodeSymbol]] = []
    imports: list[tuple[str, ImportRecord]] = []
    diagnostics: list[ParseDiagnostic] = []
    python_file_count = 0
    for snapshot_file, _size_bytes, content in scanned_files:
        path = snapshot_file.path
        if not path.lower().endswith(".py") or content is None:
            continue
        python_file_count += 1
        result = parse_python_file(path, content)
        if result.diagnostic is not None:
            diagnostics.append(result.diagnostic)
            continue
        symbols.extend((path, symbol) for symbol in result.symbols)
        imports.extend((path, record) for record in result.imports)
    return (
        python_file_count,
        tuple(symbols),
        tuple(imports),
        tuple(diagnostics),
    )


__all__ = [
    "SnapshotSymbolIndex",
    "SnapshotSymbolIndexError",
    "SymbolIndexBuildResult",
    "SymbolSearchHit",
]
