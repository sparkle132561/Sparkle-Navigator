"""SnapshotExplorer 的公开行为测试。"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

from sparkle_navigator.context.code_graph import SnapshotCodeGraph
from sparkle_navigator.context.semantic_index import SemanticIndex
from sparkle_navigator.context.snapshot_explorer import (
    SnapshotExplorer,
    SnapshotExplorerError,
)
from sparkle_navigator.context.symbol_index import SnapshotSymbolIndex
from sparkle_navigator.context.text_index import SnapshotTextIndex
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore

SMOKE_REPOSITORY = Path(__file__).parent / "fixtures" / "smoke_set" / "repository"


def test_ask_returns_hash_verified_source_line_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_smoke_snapshot(tmp_path)
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).ask(
            snapshot.snapshot_id,
            "fts5 searchable",
        )
    )

    assert result.snapshot_id == snapshot.snapshot_id
    assert result.evidence.items
    evidence = result.evidence.items[0]
    assert evidence.path == "src/text_index.py"
    assert evidence.line_start <= 2 <= evidence.line_end
    assert "FTS5 index stores searchable text" in evidence.snippet
    assert evidence.sha256 == next(
        item.sha256 for item in snapshot.files if item.path == evidence.path
    )


def test_overview_summarizes_snapshot_structure_with_readme_evidence(
    tmp_path: Path,
) -> None:
    snapshot, database_path = _registered_smoke_snapshot(tmp_path)

    overview = asyncio.run(
        SnapshotExplorer(database_path=database_path).overview(snapshot.snapshot_id)
    )

    assert overview.snapshot_id == snapshot.snapshot_id
    assert overview.file_count == len(snapshot.files)
    assert overview.python_file_count == 3
    assert overview.top_level_paths == ("README.md", "src", "tests")
    assert overview.test_paths == ("tests/test_snapshot.py",)
    readme = next(item for item in overview.evidence.items if item.path == "README.md")
    assert readme.line_start == 1
    assert "固定的检索 Smoke Set 仓库" in readme.snippet
    assert readme.sha256 == next(
        item.sha256 for item in snapshot.files if item.path == "README.md"
    )


def test_overview_fails_closed_when_readme_hash_changes(tmp_path: Path) -> None:
    snapshot, database_path = _registered_smoke_snapshot(tmp_path)
    (snapshot.snapshot_root / "README.md").write_text(
        "Changed after snapshot import\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotExplorerError, match="SHA-256"):
        asyncio.run(
            SnapshotExplorer(database_path=database_path).overview(snapshot.snapshot_id)
        )


def test_explain_file_returns_symbol_and_dependency_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/app.py": (
                "from .service import run\n"
                "\n"
                "def start():\n"
                "    run()\n"
            ),
            "src/service.py": "def run():\n    return 'ok'\n",
        },
    )
    asyncio.run(SnapshotSymbolIndex(database_path=database_path).build(snapshot))
    asyncio.run(SnapshotCodeGraph(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).explain(
            snapshot.snapshot_id,
            "src/app.py",
        )
    )

    assert result.target_kind == "file"
    assert any(
        item.symbol_name == "start" and item.line_start == 3
        for item in result.evidence.items
    )
    dependency = next(
        item
        for item in result.evidence.items
        if item.relationship_type == "DEPENDS_ON"
    )
    assert dependency.path == "src/app.py"
    assert dependency.line_start == 1
    assert dependency.confidence == "CONFIRMED"


def test_explain_symbol_returns_definition_and_call_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/app.py": (
                "def start():\n"
                "    run()\n"
                "\n"
                "def run():\n"
                "    return None\n"
            ),
        },
    )
    asyncio.run(SnapshotSymbolIndex(database_path=database_path).build(snapshot))
    asyncio.run(SnapshotCodeGraph(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).explain(
            snapshot.snapshot_id,
            "start",
        )
    )

    assert result.target_kind == "symbol"
    assert any(
        item.source == "symbol" and item.symbol_name == "start" and item.line_start == 1
        for item in result.evidence.items
    )
    assert any(
        item.relationship_type == "CALLS" and item.line_start == 2
        for item in result.evidence.items
    )


def test_explain_unknown_target_returns_no_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {"src/app.py": "def start():\n    return None\n"},
    )

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).explain(
            snapshot.snapshot_id,
            "missing_symbol",
        )
    )

    assert result.target_kind == "not_found"
    assert result.evidence.items == ()


def test_trace_call_returns_verified_callsite_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/entry.py": (
                "from .service import run\n"
                "\n"
                "def start():\n"
                "    run()\n"
            ),
            "src/service.py": "def run():\n    return None\n",
        },
    )
    asyncio.run(SnapshotCodeGraph(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).trace_call(
            snapshot.snapshot_id,
            "src/entry.py",
            "start",
        )
    )

    assert [item.target_symbol for item in result.relationships] == ["run"]
    assert [
        (item.path, item.line_start, item.confidence)
        for item in result.evidence.items
    ] == [("src/entry.py", 4, "CONFIRMED")]


def test_impact_returns_callers_dependencies_and_test_candidates(
    tmp_path: Path,
) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/entry.py": (
                "from .service import run\n"
                "\n"
                "def start():\n"
                "    run()\n"
            ),
            "src/service.py": (
                "from .core import work\n"
                "\n"
                "def run():\n"
                "    work()\n"
            ),
            "src/core.py": "def work():\n    return None\n",
            "tests/test_core.py": (
                "from src.core import work\n"
                "\n"
                "def test_work():\n"
                "    work()\n"
            ),
        },
    )
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))
    asyncio.run(SnapshotCodeGraph(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).impact(
            snapshot.snapshot_id,
            "src/core.py",
            target_symbol="work",
            max_depth=3,
        )
    )

    assert {item.source_symbol for item in result.callers} == {
        "run",
        "start",
        "test_work",
    }
    assert {item.source_path for item in result.dependents} == {
        "src/service.py",
        "src/entry.py",
        "tests/test_core.py",
    }
    assert result.test_candidates == ("tests/test_core.py",)
    assert any(
        item.path == "tests/test_core.py" and item.source == "text"
        for item in result.evidence.items
    )


def test_ask_returns_no_claim_when_snapshot_has_no_matching_evidence(
    tmp_path: Path,
) -> None:
    snapshot, database_path = _registered_smoke_snapshot(tmp_path)
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).ask(
            snapshot.snapshot_id,
            "no_such_token_xyz",
        )
    )

    assert result.evidence.items == ()
    assert result.answer == "当前 Snapshot 中没有找到可引用的代码证据。"


@pytest.mark.parametrize("question", [" ", "!!!"])
def test_ask_rejects_questions_without_searchable_terms(
    tmp_path: Path,
    question: str,
) -> None:
    snapshot, database_path = _registered_smoke_snapshot(tmp_path)

    with pytest.raises(ValueError, match="可检索词"):
        asyncio.run(
            SnapshotExplorer(database_path=database_path).ask(
                snapshot.snapshot_id,
                question,
            )
        )


def test_ask_fails_closed_when_snapshot_file_hash_changes(tmp_path: Path) -> None:
    snapshot, database_path = _registered_smoke_snapshot(tmp_path)
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))
    changed_file = snapshot.snapshot_root / "src" / "text_index.py"
    changed_file.write_text("class ChangedAfterIndex: pass\n", encoding="utf-8")

    with pytest.raises(SnapshotExplorerError, match="SHA-256"):
        asyncio.run(
            SnapshotExplorer(database_path=database_path).ask(
                snapshot.snapshot_id,
                "fts5 searchable",
            )
        )


def test_ask_validates_only_files_that_can_be_cited(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/text_index.py": "FTS5 searchable content\n",
            "src/unrelated.py": "unrelated original content\n",
        },
    )
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))
    (snapshot.snapshot_root / "src" / "unrelated.py").write_text(
        "unrelated changed content\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).ask(
            snapshot.snapshot_id,
            "fts5 searchable",
        )
    )

    assert {item.path for item in result.evidence.items} == {"src/text_index.py"}


def test_ask_returns_symbol_and_callsite_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/app.py": (
                "def entry():\n"
                "    target()\n"
                "\n"
                "def target():\n"
                "    return None\n"
            ),
        },
    )
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))
    asyncio.run(SnapshotSymbolIndex(database_path=database_path).build(snapshot))
    asyncio.run(SnapshotCodeGraph(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).ask(snapshot.snapshot_id, "entry")
    )

    assert any(
        item.source == "symbol" and item.symbol_name == "entry" and item.line_start == 1
        for item in result.evidence.items
    )
    callsite = next(
        item
        for item in result.evidence.items
        if item.source == "relationship" and item.relationship_type == "CALLS"
    )
    assert callsite.path == "src/app.py"
    assert callsite.line_start == 2


def test_ask_deduplicates_same_source_span_across_retrievers(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {"src/app.py": "def entry(): return None\n"},
    )
    asyncio.run(SnapshotTextIndex(database_path=database_path).build(snapshot))
    asyncio.run(SnapshotSymbolIndex(database_path=database_path).build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(database_path=database_path).ask(snapshot.snapshot_id, "entry")
    )

    same_location = [
        item
        for item in result.evidence.items
        if (item.path, item.line_start, item.line_end) == ("src/app.py", 1, 1)
    ]
    assert len(same_location) == 1
    assert same_location[0].source == "symbol"


def test_ask_uses_optional_semantic_symbol_evidence(tmp_path: Path) -> None:
    snapshot, database_path = _registered_source_snapshot(
        tmp_path,
        {
            "src/cache.py": (
                "class CacheReader:\n"
                "    def read_cache(self):\n"
                "        return 'stored values'\n"
            ),
        },
    )
    semantic_index = SemanticIndex(
        database_path=database_path,
        embedder=_AllMatchEmbedder(),
    )
    asyncio.run(semantic_index.build(snapshot))

    result = asyncio.run(
        SnapshotExplorer(
            database_path=database_path,
            semantic_index=semantic_index,
        ).ask(snapshot.snapshot_id, "retrieve stored values")
    )

    semantic_evidence = next(
        item
        for item in result.evidence.items
        if item.source == "semantic" and item.symbol_name == "CacheReader.read_cache"
    )
    assert semantic_evidence.path == "src/cache.py"
    assert semantic_evidence.line_start == 2
    assert "def read_cache" in semantic_evidence.snippet


def _registered_smoke_snapshot(tmp_path: Path) -> tuple[RepositorySnapshot, Path]:
    snapshot_root = tmp_path / "snapshot"
    shutil.copytree(
        SMOKE_REPOSITORY,
        snapshot_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    files = tuple(
        SnapshotFile(
            path=path.relative_to(snapshot_root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(snapshot_root.rglob("*"))
        if path.is_file()
    )
    snapshot = RepositorySnapshot.create(
        source_commit="7" * 40,
        files=files,
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "snapshot-explorer.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"snapshot-explorer").hexdigest(),
                name="snapshot-explorer-smoke",
                source_type="local",
                source_locator="test://snapshot-explorer-smoke",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot, database_path


def _registered_source_snapshot(
    tmp_path: Path,
    source_files: dict[str, str],
) -> tuple[RepositorySnapshot, Path]:
    snapshot_root = tmp_path / "source-snapshot"
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
        source_commit="8" * 40,
        files=files,
        snapshot_root=snapshot_root,
    )
    database_path = tmp_path / "source-snapshot.sqlite3"
    RepositoryStore(database_path).save_import(
        RepositoryImportResult(
            repository=Repository(
                repository_id=hashlib.sha256(b"source-snapshot").hexdigest(),
                name="source-snapshot",
                source_type="local",
                source_locator="test://source-snapshot",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot, database_path


class _AllMatchEmbedder:
    model_id = "snapshot-explorer-all-match-v1"

    def embed_documents(self, documents: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (1.0, 0.0)
            if "Symbol: CacheReader.read_cache" in document
            else (0.0, 1.0)
            for document in documents
        )

    def embed_query(self, query: str) -> tuple[float, float]:
        return (1.0, 0.0)
