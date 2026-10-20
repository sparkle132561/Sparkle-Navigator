"""通过 SemanticIndex 公共接口验证向量文档的构建、保存和检索。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Sequence

import pytest

from sparkle_navigator.context.semantic_index import SemanticIndex, SemanticIndexError
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


class _FakeEmbedder:
    model_id = "fake-cache-encoder-v1"

    def embed_documents(self, documents: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(_vector_for(document) for document in documents)

    def embed_query(self, query: str) -> tuple[float, ...]:
        return _vector_for(query)


def test_semantic_index_persists_file_and_symbol_vectors(tmp_path: Path) -> None:
    asyncio.run(_semantic_index_persists_file_and_symbol_vectors(tmp_path))


def test_semantic_index_isolated_by_snapshot_id(tmp_path: Path) -> None:
    asyncio.run(_semantic_index_isolated_by_snapshot_id(tmp_path))


def test_semantic_index_keeps_old_vectors_when_snapshot_hash_fails(tmp_path: Path) -> None:
    asyncio.run(_semantic_index_keeps_old_vectors_when_snapshot_hash_fails(tmp_path))


def test_semantic_index_keeps_old_vectors_when_encoder_batch_fails(tmp_path: Path) -> None:
    asyncio.run(_semantic_index_keeps_old_vectors_when_encoder_batch_fails(tmp_path))


def test_semantic_index_rejects_empty_query_and_nonpositive_limit(tmp_path: Path) -> None:
    index = SemanticIndex(database_path=tmp_path / "sparkle.sqlite3", embedder=_FakeEmbedder())

    with pytest.raises(ValueError, match="不能为空"):
        index.search("snapshot-id", " ")
    with pytest.raises(ValueError, match="正整数"):
        index.search("snapshot-id", "cache", limit=0)


async def _semantic_index_persists_file_and_symbol_vectors(tmp_path: Path) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot_root = tmp_path / "snapshot"
    source_files = {
        "src/cache.py": (
            "class CacheReader:\n"
            "    def read_cache(self):\n"
            "        return 'cache contents'\n"
        ),
        "src/settings.py": "DEFAULT_REGION = 'north'\n",
    }
    snapshot = _register_snapshot(database_path, snapshot_root, source_files)
    index = SemanticIndex(database_path=database_path, embedder=_FakeEmbedder())

    build_result = await index.build(snapshot)
    reopened_index = SemanticIndex(database_path=database_path, embedder=_FakeEmbedder())
    hits = reopened_index.search(snapshot.snapshot_id, "cache lookup", limit=3)

    assert build_result.file_document_count == 2
    assert build_result.symbol_document_count == 2
    assert build_result.document_count == 4
    assert hits
    assert all(hit.path == "src/cache.py" for hit in hits)
    assert any(hit.symbol_name == "CacheReader.read_cache" for hit in hits)


async def _semantic_index_isolated_by_snapshot_id(tmp_path: Path) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    first_snapshot = _register_snapshot(
        database_path,
        tmp_path / "first-snapshot",
        {"src/cache.py": "CACHE_LOOKUP = 'cache'\n"},
    )
    second_snapshot = _register_snapshot(
        database_path,
        tmp_path / "second-snapshot",
        {"src/settings.py": "DEFAULT_REGION = 'north'\n"},
    )
    index = SemanticIndex(database_path=database_path, embedder=_FakeEmbedder())
    await index.build(first_snapshot)
    await index.build(second_snapshot)

    first_hits = index.search(first_snapshot.snapshot_id, "cache", limit=1)
    second_hits = index.search(second_snapshot.snapshot_id, "cache", limit=1)

    assert first_hits[0].path == "src/cache.py"
    assert first_hits[0].score == pytest.approx(1.0)
    assert second_hits[0].path == "src/settings.py"
    assert second_hits[0].score == pytest.approx(0.0)


async def _semantic_index_keeps_old_vectors_when_snapshot_hash_fails(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot_root = tmp_path / "snapshot"
    snapshot = _register_snapshot(
        database_path,
        snapshot_root,
        {"src/cache.py": "CACHE_LOOKUP = 'cache'\n"},
    )
    index = SemanticIndex(database_path=database_path, embedder=_FakeEmbedder())
    await index.build(snapshot)
    (snapshot_root / "src" / "cache.py").write_text(
        "CACHE_LOOKUP = 'changed'\n",
        encoding="utf-8",
    )

    with pytest.raises(SemanticIndexError, match="SHA-256"):
        await index.build(snapshot)

    hits = index.search(snapshot.snapshot_id, "cache", limit=1)
    assert hits[0].path == "src/cache.py"
    assert hits[0].score == pytest.approx(1.0)


async def _semantic_index_keeps_old_vectors_when_encoder_batch_fails(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _register_snapshot(
        database_path,
        tmp_path / "snapshot",
        {"src/cache.py": "CACHE_LOOKUP = 'cache'\n"},
    )
    working_index = SemanticIndex(database_path=database_path, embedder=_FakeEmbedder())
    await working_index.build(snapshot)
    broken_index = SemanticIndex(
        database_path=database_path,
        embedder=_MismatchedBatchEmbedder(),
    )

    with pytest.raises(SemanticIndexError, match="向量数量"):
        await broken_index.build(snapshot)

    hits = working_index.search(snapshot.snapshot_id, "cache", limit=1)
    assert hits[0].path == "src/cache.py"
    assert hits[0].score == pytest.approx(1.0)


class _MismatchedBatchEmbedder:
    model_id = "fake-cache-encoder-v1"

    def embed_documents(self, documents: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return ()

    def embed_query(self, query: str) -> tuple[float, ...]:
        return (1.0, 0.0)


def _vector_for(text: str) -> tuple[float, float]:
    return (1.0, 0.0) if "cache" in text.lower() else (0.0, 1.0)


def _register_snapshot(
    database_path: Path,
    snapshot_root: Path,
    source_files: dict[str, str],
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
        source_commit="a" * 40,
        files=tuple(files),
        snapshot_root=snapshot_root,
    )
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id="d" * 64,
                name="semantic-index-test",
                source_type="local",
                source_locator="test://semantic-index",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot
