"""通过 SnapshotTextIndex 公共接口验证固定 Snapshot 的关键词检索。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from sparkle_navigator.context.text_index import SnapshotTextIndex, SnapshotTextIndexError
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


def test_snapshot_text_index_finds_keyword_in_the_expected_file(
    tmp_path: Path,
) -> None:
    asyncio.run(_snapshot_text_index_finds_keyword(tmp_path))


async def _snapshot_text_index_finds_keyword(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    source_file = snapshot_root / "src" / "workspace.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "def open_repository_workspace():\n    return 'snapshot'\n",
        encoding="utf-8",
    )
    content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    snapshot = RepositorySnapshot.create(
        source_commit="a" * 40,
        files=(SnapshotFile(path="src/workspace.py", sha256=content_hash),),
        index_version=0,
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "sparkle.sqlite3"
    store = RepositoryStore(database_path)
    store.save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="b" * 64,
                name="smoke-repository",
                source_type="local",
                source_locator="test://smoke-repository",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    index = SnapshotTextIndex(database_path=database_path)

    build_result = await index.build(snapshot)
    reopened_index = SnapshotTextIndex(database_path=database_path)
    hits = reopened_index.search_text(snapshot.snapshot_id, "repository", limit=5)

    assert build_result.snapshot_id == snapshot.snapshot_id
    assert build_result.indexed_file_count == 1
    assert [hit.path for hit in hits] == ["src/workspace.py"]
    assert "repository" in hits[0].snippet.casefold()
    with pytest.raises(ValueError, match="正整数"):
        reopened_index.search_text(snapshot.snapshot_id, "repository", limit=0)

def test_snapshot_text_index_rejects_id_that_does_not_match_snapshot_manifest(
    tmp_path: Path,
) -> None:
    asyncio.run(_snapshot_text_index_rejects_mismatched_id(tmp_path))


async def _snapshot_text_index_rejects_mismatched_id(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    source_file = snapshot_root / "README.md"
    snapshot_root.mkdir()
    source_file.write_text("repository baseline\n", encoding="utf-8")
    snapshot_file = SnapshotFile(
        path="README.md",
        sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    )
    snapshot = RepositorySnapshot.create(
        source_commit="c" * 40,
        files=(snapshot_file,),
        index_version=0,
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="d" * 64,
                name="smoke-repository",
                source_type="local",
                source_locator="test://smoke-repository",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    forged_snapshot = RepositorySnapshot(
        snapshot_id="e" * 64,
        source_commit=snapshot.source_commit,
        files=snapshot.files,
        index_version=snapshot.index_version,
        snapshot_root=snapshot.snapshot_root,
    )
    index = SnapshotTextIndex(database_path=database_path)

    with pytest.raises(SnapshotTextIndexError, match="Snapshot ID"):
        await index.build(forged_snapshot)


def test_snapshot_text_index_rejects_changed_file_without_replacing_existing_index(
    tmp_path: Path,
) -> None:
    asyncio.run(_snapshot_text_index_rejects_changed_file(tmp_path))


async def _snapshot_text_index_rejects_changed_file(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    source_file = snapshot_root / "README.md"
    snapshot_root.mkdir()
    source_file.write_text("trusted repository baseline\n", encoding="utf-8")
    snapshot_file = SnapshotFile(
        path="README.md",
        sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    )
    snapshot = RepositorySnapshot.create(
        source_commit="f" * 40,
        files=(snapshot_file,),
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="1" * 64,
                name="smoke-repository",
                source_type="local",
                source_locator="test://smoke-repository",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    index = SnapshotTextIndex(database_path=database_path)
    await index.build(snapshot)
    source_file.write_text("tampered replacement\n", encoding="utf-8")

    with pytest.raises(SnapshotTextIndexError, match="SHA-256"):
        await index.build(snapshot)

    assert len(index.search_text(snapshot.snapshot_id, "trusted")) == 1
    assert index.search_text(snapshot.snapshot_id, "tampered") == ()


@pytest.mark.parametrize("content", [b"opaque\xff payload", b"opaque\x00 payload"])
def test_snapshot_text_index_skips_binary_content_but_keeps_file_in_snapshot(
    tmp_path: Path,
    content: bytes,
) -> None:
    asyncio.run(_snapshot_text_index_skips_binary_content(tmp_path, content))


async def _snapshot_text_index_skips_binary_content(
    tmp_path: Path,
    content: bytes,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    source_file = snapshot_root / "asset.bin"
    snapshot_root.mkdir()
    source_file.write_bytes(content)
    snapshot_file = SnapshotFile(
        path="asset.bin",
        sha256=hashlib.sha256(content).hexdigest(),
    )
    snapshot = RepositorySnapshot.create(
        source_commit="9" * 40,
        files=(snapshot_file,),
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "sparkle.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="8" * 64,
                name="smoke-repository",
                source_type="local",
                source_locator="test://smoke-repository",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    index = SnapshotTextIndex(database_path=database_path)

    result = await index.build(snapshot)

    assert result.file_count == 1
    assert result.indexed_file_count == 0
    assert result.skipped_file_count == 1
    assert index.search_text(snapshot.snapshot_id, "opaque") == ()


def test_snapshot_text_index_search_isolated_by_snapshot_id(tmp_path: Path) -> None:
    asyncio.run(_snapshot_text_index_search_isolated_by_snapshot_id(tmp_path))


async def _snapshot_text_index_search_isolated_by_snapshot_id(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-snapshot"
    first_file = first_root / "first.py"
    first_root.mkdir()
    first_file.write_text("shared retrieval phrase\n", encoding="utf-8")
    first_snapshot = RepositorySnapshot.create(
        source_commit="1" * 40,
        files=(
            SnapshotFile(
                path="first.py",
                sha256=hashlib.sha256(first_file.read_bytes()).hexdigest(),
            ),
        ),
        snapshot_root=first_root,
    )

    second_root = tmp_path / "second-snapshot"
    second_file = second_root / "second.py"
    second_root.mkdir()
    second_file.write_text("shared retrieval phrase\n", encoding="utf-8")
    second_snapshot = RepositorySnapshot.create(
        source_commit="2" * 40,
        files=(
            SnapshotFile(
                path="second.py",
                sha256=hashlib.sha256(second_file.read_bytes()).hexdigest(),
            ),
        ),
        snapshot_root=second_root,
    )

    database_path = tmp_path / "sparkle.sqlite3"
    store = RepositoryStore(database_path)
    for repository_id, snapshot in (
        ("3" * 64, first_snapshot),
        ("4" * 64, second_snapshot),
    ):
        store.save_import(
            RepositoryImportResult(
                repository=Repository(
                    repository_id=repository_id,
                    name="smoke-repository",
                    source_type="local",
                    source_locator=f"test://{repository_id}",
                    current_snapshot_id=snapshot.snapshot_id,
                ),
                snapshot=snapshot,
            )
        )
    index = SnapshotTextIndex(database_path=database_path)
    await index.build(first_snapshot)
    await index.build(second_snapshot)

    first_hits = index.search_text(first_snapshot.snapshot_id, "shared retrieval")
    second_hits = index.search_text(second_snapshot.snapshot_id, "shared retrieval")

    assert [hit.path for hit in first_hits] == ["first.py"]
    assert [hit.path for hit in second_hits] == ["second.py"]
