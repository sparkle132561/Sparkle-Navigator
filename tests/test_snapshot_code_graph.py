"""通过 SnapshotCodeGraph 公共接口验证调用链和本地依赖链。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from sparkle_navigator.context.code_graph import SnapshotCodeGraph, SnapshotCodeGraphError
from sparkle_navigator.domain.repository import (
    Repository,
    RepositoryImportResult,
    RepositorySnapshot,
    SnapshotFile,
)
from sparkle_navigator.persistence.repository_store import RepositoryStore


def test_snapshot_code_graph_traces_calls_and_local_dependencies(tmp_path: Path) -> None:
    asyncio.run(_snapshot_code_graph_traces_calls_and_local_dependencies(tmp_path))


def test_snapshot_code_graph_keeps_valid_edges_when_one_file_has_syntax_error(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _snapshot_code_graph_keeps_valid_edges_when_one_file_has_syntax_error(tmp_path)
    )


def test_snapshot_code_graph_stops_traversal_at_a_call_cycle(tmp_path: Path) -> None:
    asyncio.run(_snapshot_code_graph_stops_traversal_at_a_call_cycle(tmp_path))


def test_snapshot_code_graph_isolated_by_snapshot_id(tmp_path: Path) -> None:
    asyncio.run(_snapshot_code_graph_isolated_by_snapshot_id(tmp_path))


def test_snapshot_code_graph_traces_callers_and_dependents(tmp_path: Path) -> None:
    asyncio.run(_snapshot_code_graph_traces_callers_and_dependents(tmp_path))


def test_snapshot_code_graph_reports_unresolved_dynamic_calls(tmp_path: Path) -> None:
    asyncio.run(_snapshot_code_graph_reports_unresolved_dynamic_calls(tmp_path))


def test_snapshot_code_graph_rejects_changed_file_without_replacing_relations(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _snapshot_code_graph_rejects_changed_file_without_replacing_relations(tmp_path)
    )


async def _snapshot_code_graph_traces_calls_and_local_dependencies(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _register_snapshot(
        database_path,
        tmp_path / "snapshot",
        {
            "src/pkg/entry.py": (
                "from .service import run as execute\n"
                "\n"
                "def start():\n"
                "    execute()\n"
            ),
            "src/pkg/service.py": (
                "def run():\n"
                "    helper()\n"
                "\n"
                "def helper():\n"
                "    return None\n"
            ),
        },
        source_commit="a" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)

    build_result = await graph.build(snapshot)
    reopened_graph = SnapshotCodeGraph(database_path=database_path)
    calls = reopened_graph.trace_calls(
        snapshot.snapshot_id,
        "src/pkg/entry.py",
        "start",
    )
    dependencies = reopened_graph.trace_dependencies(
        snapshot.snapshot_id,
        "src/pkg/entry.py",
    )

    assert build_result.relationship_count == 7
    assert build_result.definition_count == 3
    assert build_result.import_count == 1
    assert build_result.call_count == 2
    assert build_result.dependency_count == 1
    assert [
        (
            edge.source_path,
            edge.source_symbol,
            edge.target_path,
            edge.target_symbol,
            edge.relation_type,
            edge.confidence,
            edge.evidence_line,
        )
        for edge in calls
    ] == [
        ("src/pkg/entry.py", "start", "src/pkg/service.py", "run", "CALLS", "CONFIRMED", 4),
        ("src/pkg/service.py", "run", "src/pkg/service.py", "helper", "CALLS", "LIKELY", 2),
    ]
    assert [
        (
            edge.source_path,
            edge.target_path,
            edge.relation_type,
            edge.confidence,
            edge.evidence_line,
        )
        for edge in dependencies
    ] == [("src/pkg/entry.py", "src/pkg/service.py", "DEPENDS_ON", "CONFIRMED", 1)]


async def _snapshot_code_graph_keeps_valid_edges_when_one_file_has_syntax_error(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _register_snapshot(
        database_path,
        tmp_path / "snapshot",
        {
            "pkg/good.py": "def start():\n    worker()\n\ndef worker():\n    return None\n",
            "pkg/broken.py": "def broken(:\n    pass\n",
        },
        source_commit="c" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)

    result = await graph.build(snapshot)
    calls = graph.trace_calls(snapshot.snapshot_id, "pkg/good.py", "start")

    assert result.python_file_count == 2
    assert result.call_count == 1
    assert [(item.path, item.line) for item in result.diagnostics] == [
        ("pkg/broken.py", 1)
    ]
    assert [(edge.target_symbol, edge.confidence) for edge in calls] == [
        ("worker", "LIKELY")
    ]


async def _snapshot_code_graph_stops_traversal_at_a_call_cycle(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _register_snapshot(
        database_path,
        tmp_path / "snapshot",
        {
            "pkg/cycle.py": (
                "def first():\n"
                "    second()\n"
                "\n"
                "def second():\n"
                "    first()\n"
            ),
        },
        source_commit="d" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)
    await graph.build(snapshot)

    calls = graph.trace_calls(
        snapshot.snapshot_id,
        "pkg/cycle.py",
        "first",
        max_depth=10,
    )

    assert [(edge.source_symbol, edge.target_symbol) for edge in calls] == [
        ("first", "second"),
        ("second", "first"),
    ]


async def _snapshot_code_graph_isolated_by_snapshot_id(tmp_path: Path) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    first_snapshot = _register_snapshot(
        database_path,
        tmp_path / "first-snapshot",
        {
            "pkg/entry.py": (
                "def start():\n"
                "    first_worker()\n"
                "\n"
                "def first_worker():\n"
                "    return None\n"
            )
        },
        source_commit="e" * 40,
    )
    second_snapshot = _register_snapshot(
        database_path,
        tmp_path / "second-snapshot",
        {
            "pkg/entry.py": (
                "def start():\n"
                "    second_worker()\n"
                "\n"
                "def second_worker():\n"
                "    return None\n"
            )
        },
        source_commit="f" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)
    await graph.build(first_snapshot)
    await graph.build(second_snapshot)

    first_calls = graph.trace_calls(first_snapshot.snapshot_id, "pkg/entry.py", "start")
    second_calls = graph.trace_calls(second_snapshot.snapshot_id, "pkg/entry.py", "start")

    assert [edge.target_symbol for edge in first_calls] == ["first_worker"]
    assert [edge.target_symbol for edge in second_calls] == ["second_worker"]


async def _snapshot_code_graph_traces_callers_and_dependents(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _register_snapshot(
        database_path,
        tmp_path / "snapshot",
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
            "tests/test_entry.py": (
                "from src.entry import start\n"
                "\n"
                "def test_start():\n"
                "    start()\n"
            ),
        },
        source_commit="c" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)
    await graph.build(snapshot)

    callers = graph.trace_callers(
        snapshot.snapshot_id,
        "src/core.py",
        "work",
        max_depth=3,
    )
    dependents = graph.trace_dependents(
        snapshot.snapshot_id,
        "src/core.py",
        max_depth=3,
    )

    assert {
        (item.source_path, item.source_symbol, item.target_symbol)
        for item in callers
    } == {
        ("src/service.py", "run", "work"),
        ("src/entry.py", "start", "run"),
        ("tests/test_entry.py", "test_start", "start"),
    }
    assert {item.source_path for item in dependents} == {
        "src/service.py",
        "src/entry.py",
        "tests/test_entry.py",
    }


async def _snapshot_code_graph_reports_unresolved_dynamic_calls(tmp_path: Path) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot = _register_snapshot(
        database_path,
        tmp_path / "snapshot",
        {"pkg/factory.py": "def start(factory):\n    factory()()\n"},
        source_commit="a" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)

    result = await graph.build(snapshot)

    assert result.call_count == 0
    assert result.unresolved_call_count == 2
    assert graph.trace_calls(snapshot.snapshot_id, "pkg/factory.py", "start") == ()


async def _snapshot_code_graph_rejects_changed_file_without_replacing_relations(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sparkle.sqlite3"
    snapshot_root = tmp_path / "snapshot"
    source_file = snapshot_root / "pkg" / "worker.py"
    snapshot = _register_snapshot(
        database_path,
        snapshot_root,
        {
            "pkg/worker.py": (
                "def start():\n"
                "    worker()\n"
                "\n"
                "def worker():\n"
                "    return None\n"
            )
        },
        source_commit="b" * 40,
    )
    graph = SnapshotCodeGraph(database_path=database_path)
    await graph.build(snapshot)
    source_file.write_text(
        "def start():\n    changed()\n\ndef changed():\n    return None\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotCodeGraphError, match="SHA-256"):
        await graph.build(snapshot)

    calls = graph.trace_calls(snapshot.snapshot_id, "pkg/worker.py", "start")
    assert [edge.target_symbol for edge in calls] == ["worker"]


def _register_snapshot(
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
                repository_id="b" * 64,
                name="code-graph-test",
                source_type="local",
                source_locator="test://code-graph",
                current_snapshot_id=snapshot.snapshot_id,
            ),
            snapshot=snapshot,
        )
    )
    return snapshot
