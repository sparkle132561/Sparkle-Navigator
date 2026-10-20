"""通过 SnapshotSymbolIndex 公共接口验证持久化的符号与导入记录。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


def test_snapshot_symbol_index_persists_symbols_and_imports(tmp_path: Path) -> None:
    asyncio.run(_snapshot_symbol_index_persists_symbols_and_imports(tmp_path))


def test_snapshot_symbol_index_reports_bad_python_file_and_indexes_other_files(
    tmp_path: Path,
) -> None:
    asyncio.run(_snapshot_symbol_index_reports_bad_python_file_and_indexes_other_files(tmp_path))


def test_snapshot_symbol_index_isolated_by_snapshot_id(tmp_path: Path) -> None:
    asyncio.run(_snapshot_symbol_index_isolated_by_snapshot_id(tmp_path))


async def _snapshot_symbol_index_persists_symbols_and_imports(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    source_file = snapshot_root / "src" / "cache.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "from pathlib import Path as PathAlias\n"
        "\n"
        "class SnapshotReader:\n"
        "    def read(self, path):\n"
        "        return PathAlias(path)\n"
        "\n"
        "async def refresh_reader(reader):\n"
        "    return reader\n",
        encoding="utf-8",
    )
    snapshot = RepositorySnapshot.create(
        source_commit="a" * 40,
        files=(
            SnapshotFile(
                path="src/cache.py",
                sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
            ),
        ),
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="b" * 64,
                name="symbol-index-test",
                source_type="local",
                source_locator="test://symbol-index-test",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    index = SnapshotSymbolIndex(database_path=database_path)

    build_result = await index.build(snapshot)
    reopened_index = SnapshotSymbolIndex(database_path=database_path)
    class_hits = reopened_index.search_symbols(
        snapshot.snapshot_id,
        "SnapshotReader",
    )
    method_hits = reopened_index.search_symbols(snapshot.snapshot_id, "read")
    qualified_method_hits = reopened_index.search_symbols(
        snapshot.snapshot_id,
        "SnapshotReader.read",
    )
    imports = reopened_index.imports_for_file(snapshot.snapshot_id, "src/cache.py")

    assert build_result.symbol_count == 3
    assert build_result.import_count == 1
    assert [(hit.path, hit.kind, hit.qualified_name) for hit in class_hits] == [
        ("src/cache.py", "class", "SnapshotReader")
    ]
    assert [(hit.path, hit.kind, hit.qualified_name) for hit in method_hits] == [
        ("src/cache.py", "method", "SnapshotReader.read")
    ]
    assert [(hit.path, hit.qualified_name) for hit in qualified_method_hits] == [
        ("src/cache.py", "SnapshotReader.read")
    ]
    assert [(item.module, item.imported_name, item.alias) for item in imports] == [
        ("pathlib", "Path", "PathAlias")
    ]


async def _snapshot_symbol_index_reports_bad_python_file_and_indexes_other_files(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _save_test_snapshot(
        database_path,
        tmp_path / "snapshot",
        {
            "src/valid.py": "class KeptSymbol:\n    pass\n",
            "src/broken.py": "class BrokenSymbol(\n",
        },
        source_commit="c" * 40,
    )
    index = SnapshotSymbolIndex(database_path=database_path)

    result = await index.build(snapshot)

    assert result.python_file_count == 2
    assert result.symbol_count == 1
    assert [(item.path, item.line) for item in result.diagnostics] == [
        ("src/broken.py", 1)
    ]
    assert [hit.path for hit in index.search_symbols(snapshot.snapshot_id, "KeptSymbol")] == [
        "src/valid.py"
    ]


async def _snapshot_symbol_index_isolated_by_snapshot_id(tmp_path: Path) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    first_snapshot = _save_test_snapshot(
        database_path,
        tmp_path / "first-snapshot",
        {"src/shared.py": "from alpha import FirstAlias\nclass FirstOnly:\n    pass\n"},
        source_commit="d" * 40,
    )
    second_snapshot = _save_test_snapshot(
        database_path,
        tmp_path / "second-snapshot",
        {"src/shared.py": "from beta import SecondAlias\nclass SecondOnly:\n    pass\n"},
        source_commit="e" * 40,
    )
    index = SnapshotSymbolIndex(database_path=database_path)
    await index.build(first_snapshot)
    await index.build(second_snapshot)

    first_symbols = index.search_symbols(first_snapshot.snapshot_id, "FirstOnly")
    second_symbols = index.search_symbols(second_snapshot.snapshot_id, "SecondOnly")

    assert [hit.qualified_name for hit in first_symbols] == ["FirstOnly"]
    assert [hit.qualified_name for hit in second_symbols] == ["SecondOnly"]
    assert index.search_symbols(first_snapshot.snapshot_id, "SecondOnly") == ()
    assert index.search_symbols(second_snapshot.snapshot_id, "FirstOnly") == ()
    assert [
        (item.module, item.imported_name, item.alias)
        for item in index.imports_for_file(first_snapshot.snapshot_id, "src/shared.py")
    ] == [
        ("alpha", "FirstAlias", None)
    ]
    assert [
        (item.module, item.imported_name, item.alias)
        for item in index.imports_for_file(second_snapshot.snapshot_id, "src/shared.py")
    ] == [
        ("beta", "SecondAlias", None)
    ]


def _save_test_snapshot(
    database_path: Path,
    snapshot_root: Path,
    source_files: dict[str, str],
    *,
    source_commit: str,
) -> RepositorySnapshot:
    files: list[SnapshotFile] = []
    for relative_path, source in source_files.items():
        source_path = snapshot_root.joinpath(*relative_path.split("/"))
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        files.append(
            SnapshotFile(
                path=relative_path,
                sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
        )
    snapshot = RepositorySnapshot.create(
        source_commit=source_commit,
        files=tuple(files),
        snapshot_root=snapshot_root,
    )
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="f" * 64,
                name="symbol-index-test",
                source_type="local",
                source_locator="test://symbol-index-test",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot
